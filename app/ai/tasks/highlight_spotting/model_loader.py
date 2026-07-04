from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HighlightModelReadiness:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    checkpoint_path: str | None = None
    model_config_path: str | None = None
    label_map_path: str | None = None
    checkpoint_keys: list[str] = field(default_factory=list)
    checkpoint_summary: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reasons": self.reasons,
            "checkpoint_path": self.checkpoint_path,
            "model_config_path": self.model_config_path,
            "label_map_path": self.label_map_path,
            "checkpoint_keys": self.checkpoint_keys,
            "checkpoint_summary": self.checkpoint_summary,
        }


class HighlightModelLoader:
    """Lightweight readiness checker for the champion model files.

    This class intentionally does not import project-specific torch model classes at
    module import time. The backend can run without torch or the experiment code.
    """

    def inspect(
        self,
        *,
        checkpoint_path: Path | None,
        model_config_path: Path | None,
        label_map_path: Path | None,
        model_enabled: bool,
        adapter_name: str,
    ) -> HighlightModelReadiness:
        reasons: list[str] = []
        checkpoint_keys: list[str] = []
        checkpoint_summary: dict[str, Any] = {}

        if not model_enabled:
            reasons.append("model registry entry is disabled")

        if checkpoint_path is None:
            reasons.append("checkpoint_path is not configured")
        elif not checkpoint_path.exists():
            reasons.append(f"checkpoint file does not exist: {checkpoint_path.as_posix()}")

        if model_config_path is None:
            reasons.append("config_path is not configured")
        elif not model_config_path.exists():
            reasons.append(f"model config file does not exist: {model_config_path.as_posix()}")

        if label_map_path is None:
            reasons.append("label_map_path is not configured")
        elif not label_map_path.exists():
            reasons.append(f"label map file does not exist: {label_map_path.as_posix()}")

        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_keys, checkpoint_summary = self._read_checkpoint_summary(checkpoint_path)

        # 16회차에서는 실제 SoccerHighlightFormer adapter가 아직 백엔드에 이식되지 않았다.
        # 따라서 파일이 모두 있어도 real inference는 아직 ready가 아니다.
        if adapter_name.startswith("placeholder"):
            reasons.append(
                "real model adapter is not implemented yet; "
                "copying best.pt alone is not enough"
            )

        return HighlightModelReadiness(
            ready=not reasons,
            reasons=reasons,
            checkpoint_path=checkpoint_path.as_posix() if checkpoint_path else None,
            model_config_path=model_config_path.as_posix() if model_config_path else None,
            label_map_path=label_map_path.as_posix() if label_map_path else None,
            checkpoint_keys=checkpoint_keys,
            checkpoint_summary=checkpoint_summary,
        )

    def _read_checkpoint_summary(self, checkpoint_path: Path) -> tuple[list[str], dict[str, Any]]:
        try:
            import torch  # type: ignore
        except Exception as exc:
            return [], {
                "loadable": False,
                "reason": f"torch is not importable: {exc}",
            }

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            return [], {
                "loadable": False,
                "reason": f"torch.load failed: {exc}",
            }

        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            summary = {
                "loadable": True,
                "type": "dict",
                "keys": keys[:50],
            }
            for likely_key in ["state_dict", "model_state_dict", "model", "net"]:
                value = checkpoint.get(likely_key)
                if isinstance(value, dict):
                    summary[f"{likely_key}_num_tensors"] = len(value)
                    summary[f"{likely_key}_sample_keys"] = list(value.keys())[:20]
            return keys, summary

        return [], {
            "loadable": True,
            "type": type(checkpoint).__name__,
        }
