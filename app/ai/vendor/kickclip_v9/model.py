from __future__ import annotations

import math
from typing import Any


def _torch_modules() -> tuple[Any, Any]:
    import torch
    import torch.nn as nn

    return torch, nn


_, nn = _torch_modules()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        torch, _ = _torch_modules()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=False)

    def forward(self, features: Any) -> Any:
        return features + self.pe[:, : features.size(1)]


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Any) -> Any:
        return features + self.net(features)


class SoccerSpotterV9(nn.Module):
    """Exact inference architecture paired with the supplied v9 checkpoint."""

    def __init__(
        self,
        *,
        dim: int = 512,
        n_cls: int = 5,
        d_model: int = 256,
        tcn_channels: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        use_eventness_head: bool = True,
    ) -> None:
        super().__init__()
        self.n_cls = n_cls
        self.use_eventness_head = use_eventness_head
        self.input_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.short_proj = nn.Conv1d(d_model, tcn_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            ResidualTCNBlock(tcn_channels, 1, dropout),
            ResidualTCNBlock(tcn_channels, 2, dropout),
            ResidualTCNBlock(tcn_channels, 4, dropout),
            ResidualTCNBlock(tcn_channels, 8, dropout),
            ResidualTCNBlock(tcn_channels, 16, dropout),
        )
        self.tcn_out = nn.Conv1d(tcn_channels, d_model, kernel_size=1)
        self.pos = SinusoidalPositionalEncoding(d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder,
            num_layers=num_transformer_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.heatmap_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_cls),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_cls * 2),
        )
        if self.use_eventness_head:
            self.eventness_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, features: Any) -> tuple[Any, Any, Any, Any | None]:
        torch, _ = _torch_modules()
        hidden = self.input_proj(features)
        local = self.short_proj(hidden.transpose(1, 2))
        local = self.tcn(local)
        local = self.tcn_out(local).transpose(1, 2)
        hidden = self.pos(hidden + local)
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)
        logits = self.heatmap_head(hidden)
        offset_output = self.offset_head(hidden)
        offset_mean = 8.0 * torch.tanh(offset_output[..., : self.n_cls])
        offset_logvar = offset_output[..., self.n_cls :]
        eventness = (
            self.eventness_head(hidden).squeeze(-1)
            if self.use_eventness_head
            else None
        )
        return logits, offset_mean, offset_logvar, eventness
