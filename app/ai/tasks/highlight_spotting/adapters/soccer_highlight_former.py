from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import yaml

from app.ai.tasks.highlight_spotting.adapters.base import (
    ChampionAdapterPreflightReport,
    ChampionArtifactPaths,
    ChampionCheckpointSummary,
    ChampionModelSpec,
    HighlightRawPrediction,
)
if TYPE_CHECKING:
    from app.ai.tasks.highlight_spotting.feature_loader import HighlightFeatureBundle
from app.core.paths import get_project_root


DEFAULT_CHAMPION_MODEL_DIR = "storage/models/highlight_spotting/champion"
STATE_DICT_CANDIDATE_KEYS = ["model_state_dict", "state_dict", "model", "net"]


class SoccerHighlightFormerAdapter:
    """Real adapter for the champion SoccerHighlightFormer checkpoint.

    Round 25 scope:
    - Reimplement the torch module architecture required by best.pt.
    - Load model_state_dict with strict=True.
    - Run sliding-window inference over SoccerNet PCA512 .npy features.
    - Convert heatmap probabilities to raw event predictions.

    Timestamp validation and render-level tuning remain round 26 work.
    """

    def __init__(self, paths: ChampionArtifactPaths) -> None:
        self.paths = paths
        self.spec = load_champion_model_spec(paths)
        self._torch = None
        self._device = None
        self._model = None

    @classmethod
    def from_model_dir(cls, model_dir: str | Path | None = None) -> "SoccerHighlightFormerAdapter":
        return cls(resolve_champion_artifact_paths(model_dir or DEFAULT_CHAMPION_MODEL_DIR))

    def preflight(self) -> ChampionAdapterPreflightReport:
        reasons: list[str] = []
        paths_report = self.paths.to_metadata()
        missing = _find_missing_required_files(self.paths)
        if missing:
            reasons.extend([f"missing required artifact: {item}" for item in missing])

        spec = self.spec
        checkpoint = inspect_champion_checkpoint(self.paths.checkpoint_path)

        if not checkpoint.loadable:
            reasons.append(checkpoint.reason or "checkpoint is not loadable")

        _validate_spec_consistency(spec, reasons)
        _validate_checkpoint_for_adapter(checkpoint, reasons)

        # Round 25 now includes a backend implementation of this architecture.
        # If torch and required artifacts are available, real adapter is ready.
        if checkpoint.loadable and checkpoint.state_dict_key == "model_state_dict":
            _validate_model_can_load(paths=self.paths, spec=spec, reasons=reasons)

        return ChampionAdapterPreflightReport(
            ready_for_real_adapter=not reasons,
            reasons=reasons,
            paths=paths_report,
            spec=spec.to_metadata(),
            checkpoint=checkpoint.to_metadata(),
            implementation_todos=build_implementation_todos(spec, checkpoint, round25_done=True),
        )

    def predict(
        self,
        *,
        feature_bundle: HighlightFeatureBundle,
        match_duration_sec: float | None = None,
        device: str = "auto",
        batch_size: int | None = None,
        max_candidates: int | None = None,
        debug_min_threshold: float | None = None,
    ) -> list[HighlightRawPrediction]:
        """Run real inference over a validated HighlightFeatureBundle."""

        if not feature_bundle.available:
            raise RuntimeError(
                "feature_bundle is not available for real inference: "
                f"{feature_bundle.to_metadata()}"
            )

        model, torch_module, resolved_device = self._load_model(device=device)
        inference_batch_size = int(batch_size or _read_inference_batch_size(self.paths) or 16)
        max_candidates = int(max_candidates or _read_max_candidates(self.paths) or 80)

        predictions: list[HighlightRawPrediction] = []
        segments = self._build_feature_segments(feature_bundle)

        with torch_module.no_grad():
            for segment in segments:
                feature_array = np.load(segment.path, mmap_mode="r", allow_pickle=False)
                windows, window_starts, original_rows = build_sliding_windows(
                    feature_array=feature_array,
                    window_size=int(self.spec.window_size or 128),
                    stride_size=int(self.spec.stride_size or 32),
                )

                for batch_windows, batch_starts in _iter_batches(
                    windows,
                    window_starts,
                    batch_size=inference_batch_size,
                ):
                    tensor = torch_module.from_numpy(batch_windows).float().to(resolved_device)
                    output = model(tensor)
                    heatmap_logits = _extract_output_tensor(output, "heatmap")
                    offset_logits = _extract_output_tensor(output, "offset")
                    probabilities = torch_module.sigmoid(heatmap_logits).detach().cpu().numpy()
                    offsets = None
                    if offset_logits is not None:
                        offsets = torch_module.tanh(offset_logits).detach().cpu().numpy()

                    predictions.extend(
                        self._window_outputs_to_predictions(
                            probabilities=probabilities,
                            offsets=offsets,
                            window_starts=batch_starts,
                            original_rows=original_rows,
                            segment=segment,
                            match_duration_sec=match_duration_sec,
                            debug_min_threshold=debug_min_threshold,
                        )
                    )

        ranked = self._rank_and_deduplicate_predictions(predictions)
        return ranked[:max_candidates]

    def _load_model(self, *, device: str):
        if self._model is not None and self._torch is not None and self._device is not None:
            return self._model, self._torch, self._device

        torch_module, nn_module = _require_torch()
        resolved_device = _resolve_device(torch_module, device)
        model = _build_torch_model_from_spec(self.spec, nn_module).to(resolved_device)

        try:
            checkpoint = torch_module.load(
                self.paths.checkpoint_path,
                map_location=resolved_device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch_module.load(self.paths.checkpoint_path, map_location=resolved_device)

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"checkpoint must be a dict, got {type(checkpoint).__name__}"
            )

        state_dict_key = _find_state_dict_key(checkpoint)
        if not state_dict_key:
            raise RuntimeError(
                "checkpoint does not contain a recognized state_dict key; "
                f"available keys={list(checkpoint.keys())}"
            )

        state_dict = checkpoint[state_dict_key]
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "state_dict strict load failed: "
                f"missing={missing}, unexpected={unexpected}"
            )

        model.eval()
        self._model = model
        self._torch = torch_module
        self._device = resolved_device
        return model, torch_module, resolved_device

    def _build_feature_segments(self, feature_bundle: HighlightFeatureBundle) -> list["FeatureSegment"]:
        fps = float(self.spec.feature_fps or 2.0)
        segments: list[FeatureSegment] = []

        if feature_bundle.layout == "halves" and len(feature_bundle.resolved_paths) >= 2:
            offset_sec = 0.0
            for index, path in enumerate(feature_bundle.resolved_paths):
                shape = _feature_info_shape(feature_bundle.feature_infos[index]) if index < len(feature_bundle.feature_infos) else ()
                rows = int(shape[0]) if shape else int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
                half = index + 1
                segments.append(FeatureSegment(path=path, offset_sec=offset_sec, half=half, fps=fps))
                offset_sec += rows / fps
            return segments

        # Combined feature is the default output from SoccerNetFeatureExtractionTask.
        return [FeatureSegment(path=feature_bundle.resolved_paths[0], offset_sec=0.0, half=None, fps=fps)]

    def _window_outputs_to_predictions(
        self,
        *,
        probabilities: np.ndarray,
        offsets: np.ndarray | None,
        window_starts: np.ndarray,
        original_rows: int,
        segment: "FeatureSegment",
        match_duration_sec: float | None,
        debug_min_threshold: float | None = None,
    ) -> list[HighlightRawPrediction]:
        labels = self.spec.labels or [f"class_{idx}" for idx in range(int(self.spec.num_classes or probabilities.shape[-1]))]
        thresholds = self.spec.thresholds or {label: 0.2 for label in labels}
        class_priority = self.spec.class_priority or {}
        output: list[HighlightRawPrediction] = []

        batch_size, window_size, num_classes = probabilities.shape
        for batch_idx in range(batch_size):
            start = int(window_starts[batch_idx])
            for local_idx in range(window_size):
                feature_idx = start + local_idx
                if feature_idx >= original_rows:
                    continue
                for class_idx in range(num_classes):
                    label = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"
                    confidence = float(probabilities[batch_idx, local_idx, class_idx])
                    configured_threshold = float(thresholds.get(label, 0.2))
                    if debug_min_threshold is None:
                        threshold = configured_threshold
                        threshold_source = "configured"
                    else:
                        # Debug-only relaxation for timestamp/render inspection.
                        # Production jobs should omit debug_min_threshold so the
                        # trained class thresholds remain authoritative.
                        threshold = min(configured_threshold, float(debug_min_threshold))
                        threshold_source = "debug_min_threshold"
                    if confidence < threshold:
                        continue

                    offset_value = 0.0
                    if offsets is not None:
                        # Offset target scale was trained in feature-index units. Keep it
                        # conservative and bounded in the first service integration.
                        offset_value = float(np.clip(offsets[batch_idx, local_idx, class_idx], -1.0, 1.0))

                    timestamp_sec = segment.offset_sec + ((feature_idx + offset_value) / segment.fps)
                    if match_duration_sec is not None:
                        timestamp_sec = min(max(timestamp_sec, 0.0), max(float(match_duration_sec) - 0.001, 0.0))

                    half = segment.half
                    if half is None and match_duration_sec:
                        half = 1 if timestamp_sec < float(match_duration_sec) / 2.0 else 2

                    priority = float(class_priority.get(label, 0.5))
                    highlight_score = round((confidence * 8.0) + (priority * 2.0), 4)

                    output.append(
                        HighlightRawPrediction(
                            label=label,
                            timestamp_sec=timestamp_sec,
                            confidence=confidence,
                            half=half,
                            highlight_score=highlight_score,
                            metadata={
                                "predictor_mode": "real",
                                "model_adapter": "soccer_highlight_former",
                                "model_checkpoint": self.paths.checkpoint_path.as_posix(),
                                "feature_path": segment.path.as_posix(),
                                "feature_index": int(feature_idx),
                                "window_start": int(start),
                                "window_local_index": int(local_idx),
                                "class_index": int(class_idx),
                                "threshold": threshold,
                                "configured_threshold": configured_threshold,
                                "threshold_source": threshold_source,
                                "debug_min_threshold": debug_min_threshold,
                                "offset_feature_index": offset_value,
                                "feature_fps": segment.fps,
                                "segment_offset_sec": segment.offset_sec,
                            },
                        )
                    )
        return output

    def _rank_and_deduplicate_predictions(
        self,
        predictions: list[HighlightRawPrediction],
    ) -> list[HighlightRawPrediction]:
        nms_windows = self.spec.nms_windows or {}
        predictions = sorted(
            predictions,
            key=lambda item: (
                item.highlight_score if item.highlight_score is not None else item.confidence * 10,
                item.confidence,
            ),
            reverse=True,
        )

        selected: list[HighlightRawPrediction] = []
        for candidate in predictions:
            nms_window = float(nms_windows.get(candidate.label, 10.0))
            duplicate = False
            for existing in selected:
                if existing.label != candidate.label:
                    continue
                if abs(existing.timestamp_sec - candidate.timestamp_sec) <= nms_window:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)

        selected.sort(key=lambda item: item.timestamp_sec)
        return selected


@dataclass(frozen=True)
class FeatureSegment:
    path: Path
    offset_sec: float
    half: int | None
    fps: float


def resolve_champion_artifact_paths(model_dir: str | Path) -> ChampionArtifactPaths:
    project_root = get_project_root()
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute():
        model_dir_path = project_root / model_dir_path

    return ChampionArtifactPaths(
        model_dir=model_dir_path,
        checkpoint_path=model_dir_path / "best.pt",
        model_config_path=model_dir_path / "model.yaml",
        inference_config_path=model_dir_path / "inference.yaml",
        data_config_path=model_dir_path / "data.yaml",
        manifest_path=model_dir_path / "manifest.json",
        label_map_path=_optional_existing_path(model_dir_path / "label_map.json"),
        threshold_sweep_path=_optional_existing_path(model_dir_path / "threshold_sweep.json"),
        valid_eval_path=_optional_existing_path(model_dir_path / "valid_eval.json"),
        train_history_path=_optional_existing_path(model_dir_path / "train_history.json"),
    )


def load_champion_model_spec(paths: ChampionArtifactPaths) -> ChampionModelSpec:
    model_config = _load_yaml(paths.model_config_path)
    inference_config = _load_yaml(paths.inference_config_path)
    data_config = _load_yaml(paths.data_config_path)
    manifest = _load_json(paths.manifest_path)
    label_map = _load_json(paths.label_map_path) if paths.label_map_path else {}

    model_section = model_config.get("model") or {}
    former_section = model_config.get("soccer_highlight_former") or {}
    data_feature = data_config.get("feature") or {}
    data_labels = data_config.get("labels") or {}
    data_window = data_config.get("window") or {}
    inference = inference_config.get("inference") or {}

    labels = _extract_labels(data_labels=data_labels, manifest=manifest, label_map=label_map)

    return ChampionModelSpec(
        model_name=model_section.get("name") or manifest.get("model_name"),
        model_type=model_section.get("type") or manifest.get("model_type"),
        input_dim=_optional_int(former_section.get("input_dim") or model_section.get("input_dim")),
        num_classes=_optional_int(former_section.get("num_classes") or model_section.get("num_classes") or data_labels.get("num_classes") or manifest.get("num_classes")),
        labels=labels,
        feature_dim=_optional_int(inference.get("feature_dim") or data_feature.get("dim") or manifest.get("feature_dim")),
        feature_fps=_optional_float(inference.get("feature_fps") or data_feature.get("fps") or manifest.get("feature_fps")),
        window_size=_optional_int(inference.get("window_size") or data_window.get("window_size") or manifest.get("window_size")),
        window_sec=_optional_float(inference.get("window_sec") or data_window.get("window_sec")),
        stride_size=_optional_int(inference.get("stride_size") or data_window.get("stride_size") or manifest.get("stride_size")),
        stride_sec=_optional_float(inference.get("stride_sec") or data_window.get("stride_sec")),
        thresholds=_float_dict(inference_config.get("thresholds") or {}),
        nms_windows=_float_dict(inference_config.get("nms") or {}),
        class_priority=_float_dict((inference_config.get("ranking") or {}).get("class_priority") or {}),
        raw_model_config=model_config,
        raw_inference_config=inference_config,
        raw_data_config=data_config,
        raw_manifest=manifest,
        raw_label_map=label_map,
    )


def inspect_champion_checkpoint(checkpoint_path: str | Path) -> ChampionCheckpointSummary:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return ChampionCheckpointSummary(
            loadable=False,
            reason=f"checkpoint file does not exist: {checkpoint_path.as_posix()}",
        )

    try:
        import torch  # type: ignore
    except Exception as exc:
        return ChampionCheckpointSummary(
            loadable=False,
            reason=f"torch is not importable: {exc}",
        )

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        return ChampionCheckpointSummary(
            loadable=False,
            reason=f"torch.load failed: {exc}",
        )

    checkpoint_type = type(checkpoint).__name__
    if not isinstance(checkpoint, dict):
        return ChampionCheckpointSummary(
            loadable=True,
            checkpoint_type=checkpoint_type,
            reason="checkpoint is loadable, but it is not a dict checkpoint",
        )

    checkpoint_keys = list(checkpoint.keys())
    state_dict_key = _find_state_dict_key(checkpoint)
    state_dict = checkpoint.get(state_dict_key) if state_dict_key else None
    tensor_shapes: dict[str, list[int]] = {}
    sample_keys: list[str] = []
    num_tensors: int | None = None

    if isinstance(state_dict, dict):
        sample_keys = list(state_dict.keys())[:80]
        num_tensors = len(state_dict)
        for key in sample_keys:
            value = state_dict.get(key)
            shape = getattr(value, "shape", None)
            if shape is not None:
                tensor_shapes[key] = [int(v) for v in shape]

    return ChampionCheckpointSummary(
        loadable=True,
        checkpoint_type=checkpoint_type,
        checkpoint_keys=checkpoint_keys[:80],
        state_dict_key=state_dict_key,
        state_dict_num_tensors=num_tensors,
        state_dict_sample_keys=sample_keys,
        tensor_shapes=tensor_shapes,
        epoch=_optional_int(checkpoint.get("epoch")),
        best=_json_safe(checkpoint.get("best")) if isinstance(checkpoint.get("best"), dict) else None,
        metrics=_summarize_metrics(checkpoint.get("metrics")),
    )


def build_implementation_todos(
    spec: ChampionModelSpec,
    checkpoint: ChampionCheckpointSummary,
    *,
    round25_done: bool = False,
) -> list[str]:
    if round25_done:
        todos = [
            "Validate event timestamp alignment against source videos in round 26.",
            "Tune thresholds/NMS/max_candidates with service videos after real inference smoke test.",
            "Consider reading feature_metadata.json split_sec for more exact half2 offset handling.",
        ]
    else:
        todos = [
            "Copy or reimplement the SoccerHighlightFormer torch module used during training.",
            "Implement multi-scale temporal stem modules so checkpoint keys like temporal_stem.layers.* can load.",
            "Build sliding-window inference: feature array [T, 512] -> windows [B, 128, 512].",
            "Load best.pt model_state_dict with strict=True only after module names match the checkpoint.",
            "Convert model heatmap logits/probabilities to event candidates per class.",
            "Use offset head output when available to refine timestamp_sec.",
            "Use feature_fps and feature_metadata split_sec for combined/half timestamp conversion.",
            "Apply thresholds, class-aware NMS, max_candidates, and clip offsets from inference.yaml/highlight_spotting.yaml.",
        ]

    if checkpoint.state_dict_key != "model_state_dict":
        todos.append(
            "Confirm state_dict key name before loading; expected model_state_dict for this champion."
        )
    if spec.window_size is None or spec.stride_size is None:
        todos.append("Confirm window_size and stride_size from inference/data config before real inference.")
    if spec.labels:
        todos.append(f"Use label order exactly as trained: {spec.labels}")

    return todos


def build_sliding_windows(
    *,
    feature_array: np.ndarray,
    window_size: int,
    stride_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(feature_array.shape) != 2:
        raise ValueError(f"feature_array must be 2-D, got shape={feature_array.shape}")
    if window_size <= 0 or stride_size <= 0:
        raise ValueError("window_size and stride_size must be positive")

    total_rows = int(feature_array.shape[0])
    feature_dim = int(feature_array.shape[1])
    if total_rows <= 0:
        raise ValueError("feature_array is empty")

    if total_rows <= window_size:
        window = np.zeros((window_size, feature_dim), dtype=np.float32)
        loaded = np.asarray(feature_array[:total_rows], dtype=np.float32)
        window[:total_rows] = loaded
        if total_rows > 0 and total_rows < window_size:
            window[total_rows:] = loaded[-1]
        return window[None, :, :], np.asarray([0], dtype=np.int64), total_rows

    starts = list(range(0, total_rows - window_size + 1, stride_size))
    last_start = total_rows - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    windows = np.stack(
        [np.asarray(feature_array[start : start + window_size], dtype=np.float32) for start in starts],
        axis=0,
    )
    return windows, np.asarray(starts, dtype=np.int64), total_rows


def _build_torch_model_from_spec(spec: ChampionModelSpec, nn_module):
    model_config = spec.raw_model_config or {}
    former = model_config.get("soccer_highlight_former") or {}
    transformer = former.get("transformer") or {}
    temporal_stem = former.get("temporal_stem") or {}
    projection = former.get("projection") or {}

    input_dim = int(spec.input_dim or 512)
    d_model = int(former.get("d_model") or 256)
    num_classes = int(spec.num_classes or len(spec.labels) or 6)
    dropout = float(projection.get("dropout", former.get("dropout", 0.1)))
    stem_dropout = float(temporal_stem.get("dropout", 0.1))
    num_stem_layers = int(temporal_stem.get("num_layers", 2))
    kernel_sizes = list(temporal_stem.get("multi_scale_kernel_sizes") or [3, 5, 9])

    num_transformer_layers = int(transformer.get("num_layers", 4))
    num_heads = int(transformer.get("num_heads", 8))
    dim_feedforward = int(transformer.get("dim_feedforward", 1024))
    transformer_dropout = float(transformer.get("dropout", 0.1))
    activation = str(transformer.get("activation", "gelu"))
    max_len = int((former.get("positional_encoding") or {}).get("max_len", 4096))

    return SoccerHighlightFormerTorch(
        input_dim=input_dim,
        d_model=d_model,
        num_classes=num_classes,
        projection_dropout=dropout,
        stem_num_layers=num_stem_layers,
        stem_kernel_sizes=kernel_sizes,
        stem_dropout=stem_dropout,
        transformer_num_layers=num_transformer_layers,
        transformer_num_heads=num_heads,
        transformer_dim_feedforward=dim_feedforward,
        transformer_dropout=transformer_dropout,
        transformer_activation=activation,
        max_len=max_len,
        nn_module=nn_module,
    )


class SoccerHighlightFormerTorch:  # dynamically becomes nn.Module at runtime
    pass


def _install_torch_model_classes() -> None:
    global SoccerHighlightFormerTorch
    try:
        import torch as torch_module  # type: ignore
        from torch import nn as nn_module  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"torch is required for real SoccerHighlightFormer inference: {exc}") from exc

    if hasattr(SoccerHighlightFormerTorch, "_kickclip_installed"):
        return

    class ConvBNAct(nn_module.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, groups: int = 1, dropout: float = 0.1):
            super().__init__()
            padding = kernel_size // 2
            self.net = nn_module.Sequential(
                nn_module.Conv1d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False),
                nn_module.BatchNorm1d(out_channels),
                nn_module.GELU(),
                nn_module.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class MultiScaleTemporalLayer(nn_module.Module):
        def __init__(self, d_model: int, kernel_sizes: list[int], dropout: float):
            super().__init__()
            self.branches = nn_module.ModuleList(
                [
                    nn_module.Sequential(
                        ConvBNAct(d_model, d_model, int(kernel), groups=d_model, dropout=dropout),
                        ConvBNAct(d_model, d_model, 1, groups=1, dropout=dropout),
                    )
                    for kernel in kernel_sizes
                ]
            )
            self.fuse = nn_module.Sequential(
                nn_module.Conv1d(d_model * len(kernel_sizes), d_model, 1, bias=False),
                nn_module.BatchNorm1d(d_model),
                nn_module.GELU(),
                nn_module.Dropout(dropout),
            )
            self.norm = nn_module.LayerNorm(d_model)

        def forward(self, x):
            residual = x
            x_conv = x.transpose(1, 2)
            branches = [branch(x_conv) for branch in self.branches]
            fused = self.fuse(torch_module.cat(branches, dim=1)).transpose(1, 2)
            return self.norm(residual + fused)

    class MultiScaleTemporalStem(nn_module.Module):
        def __init__(self, d_model: int, num_layers: int, kernel_sizes: list[int], dropout: float):
            super().__init__()
            self.layers = nn_module.ModuleList(
                [MultiScaleTemporalLayer(d_model, kernel_sizes, dropout) for _ in range(num_layers)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class SinusoidalPositionalEncoding(nn_module.Module):
        def __init__(self, d_model: int, max_len: int = 4096):
            super().__init__()
            position = torch_module.arange(max_len).unsqueeze(1).float()
            div_term = torch_module.exp(
                torch_module.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe = torch_module.zeros(max_len, d_model)
            pe[:, 0::2] = torch_module.sin(position * div_term)
            pe[:, 1::2] = torch_module.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            length = x.size(1)
            return x + self.pe[:, :length, :].to(dtype=x.dtype, device=x.device)

    class _SoccerHighlightFormerTorch(nn_module.Module):
        _kickclip_installed = True

        def __init__(
            self,
            *,
            input_dim: int,
            d_model: int,
            num_classes: int,
            projection_dropout: float,
            stem_num_layers: int,
            stem_kernel_sizes: list[int],
            stem_dropout: float,
            transformer_num_layers: int,
            transformer_num_heads: int,
            transformer_dim_feedforward: int,
            transformer_dropout: float,
            transformer_activation: str,
            max_len: int,
            nn_module,
        ):
            super().__init__()
            self.feature_projection = nn_module.Sequential(
                nn_module.Linear(input_dim, d_model),
                nn_module.LayerNorm(d_model),
                nn_module.Dropout(projection_dropout),
            )
            self.temporal_stem = MultiScaleTemporalStem(
                d_model=d_model,
                num_layers=stem_num_layers,
                kernel_sizes=stem_kernel_sizes,
                dropout=stem_dropout,
            )
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
            encoder_layer = nn_module.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_num_heads,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                activation=transformer_activation,
                batch_first=True,
                norm_first=False,
            )
            self.transformer_encoder = nn_module.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_num_layers,
            )
            self.encoder_norm = nn_module.LayerNorm(d_model)
            self.heatmap_head = nn_module.Sequential(
                nn_module.LayerNorm(d_model),
                nn_module.Linear(d_model, num_classes),
            )
            self.offset_head = nn_module.Sequential(
                nn_module.LayerNorm(d_model),
                nn_module.Linear(d_model, num_classes),
            )

        def forward(self, x):
            x = self.feature_projection(x)
            x = self.temporal_stem(x)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            x = self.encoder_norm(x)
            return {
                "heatmap": self.heatmap_head(x),
                "offset": self.offset_head(x),
            }

    SoccerHighlightFormerTorch = _SoccerHighlightFormerTorch


def _validate_model_can_load(
    *,
    paths: ChampionArtifactPaths,
    spec: ChampionModelSpec,
    reasons: list[str],
) -> None:
    try:
        adapter = SoccerHighlightFormerAdapter(paths)
        adapter._load_model(device="cpu")
    except Exception as exc:
        reasons.append(f"SoccerHighlightFormer backend module cannot load checkpoint strictly: {exc}")


def _find_missing_required_files(paths: ChampionArtifactPaths) -> list[str]:
    required = {
        "best.pt": paths.checkpoint_path,
        "model.yaml": paths.model_config_path,
        "inference.yaml": paths.inference_config_path,
        "data.yaml": paths.data_config_path,
        "manifest.json": paths.manifest_path,
    }
    return [name for name, path in required.items() if not path.exists()]


def _validate_spec_consistency(spec: ChampionModelSpec, reasons: list[str]) -> None:
    if spec.feature_dim != 512:
        reasons.append(f"feature_dim must be 512 for SoccerNet PCA512, got {spec.feature_dim}")
    if spec.input_dim != 512:
        reasons.append(f"model input_dim must be 512, got {spec.input_dim}")
    if spec.num_classes is not None and spec.labels and spec.num_classes != len(spec.labels):
        reasons.append(
            f"num_classes={spec.num_classes} does not match labels length={len(spec.labels)}"
        )
    if spec.window_size is None:
        reasons.append("window_size is not configured")
    if spec.stride_size is None:
        reasons.append("stride_size is not configured")


def _validate_checkpoint_for_adapter(
    checkpoint: ChampionCheckpointSummary,
    reasons: list[str],
) -> None:
    if not checkpoint.loadable:
        return
    if not checkpoint.state_dict_key:
        reasons.append("checkpoint does not contain a recognized state_dict key")
    elif checkpoint.state_dict_num_tensors is None or checkpoint.state_dict_num_tensors <= 0:
        reasons.append("checkpoint state_dict is empty")
    elif checkpoint.state_dict_key != "model_state_dict":
        reasons.append(
            f"expected checkpoint state_dict key 'model_state_dict', got {checkpoint.state_dict_key!r}"
        )


def _find_state_dict_key(checkpoint: dict[str, Any]) -> str | None:
    for key in STATE_DICT_CANDIDATE_KEYS:
        if isinstance(checkpoint.get(key), dict):
            return key
    return None


def _extract_labels(
    *,
    data_labels: dict[str, Any],
    manifest: dict[str, Any],
    label_map: dict[str, Any],
) -> list[str]:
    labels = data_labels.get("classes") or manifest.get("labels")
    if isinstance(labels, list):
        return [str(label) for label in labels]

    id_to_label = label_map.get("id_to_label") or data_labels.get("id_to_label") or {}
    if isinstance(id_to_label, dict) and id_to_label:
        return [str(id_to_label[key]) for key in sorted(id_to_label, key=lambda item: int(item))]

    return []


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.as_posix()} must contain a YAML object.")
    return loaded


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.as_posix()} must contain a JSON object.")
    return loaded


def _optional_existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _float_dict(value: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)


def _summarize_metrics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (int, float, str, bool)) or item is None:
            summary[key] = item
        elif isinstance(item, list) and item:
            summary[key] = item[-1]
        else:
            summary[key] = str(type(item).__name__)
    return summary



def _feature_info_shape(info: Any) -> tuple[int, ...]:
    shape = getattr(info, "shape", None)
    if shape is None and isinstance(info, dict):
        shape = info.get("shape")
    if shape is None:
        return ()
    return tuple(int(v) for v in shape)


def _read_inference_batch_size(paths: ChampionArtifactPaths) -> int | None:
    cfg = _load_yaml(paths.inference_config_path)
    return _optional_int((cfg.get("inference") or {}).get("batch_size"))


def _read_max_candidates(paths: ChampionArtifactPaths) -> int | None:
    manifest = _load_json(paths.manifest_path)
    recommended = manifest.get("recommended_postprocess") or {}
    return _optional_int(recommended.get("max_candidates_per_match"))


def _iter_batches(
    windows: np.ndarray,
    starts: np.ndarray,
    *,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for index in range(0, len(windows), batch_size):
        yield windows[index : index + batch_size], starts[index : index + batch_size]


def _extract_output_tensor(output: Any, key: str):
    if isinstance(output, dict):
        return output.get(key)
    if isinstance(output, (tuple, list)):
        if key == "heatmap":
            return output[0]
        if key == "offset" and len(output) > 1:
            return output[1]
    if key == "heatmap":
        return output
    return None


def _require_torch():
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"torch is required for real SoccerHighlightFormer inference: {exc}") from exc

    _install_torch_model_classes()
    return torch, nn


def _resolve_device(torch_module, device: str):
    requested = (device or "auto").lower()
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    return torch_module.device(requested)
