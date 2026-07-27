from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ChampionArtifactPaths:
    """Resolved artifact paths required by the highlight champion adapter."""

    model_dir: Path
    checkpoint_path: Path
    model_config_path: Path
    train_config_path: Path
    config_snapshot_path: Path
    inference_policy_path: Path
    manifest_path: Path
    valid_eval_path: Path | None = None
    train_history_path: Path | None = None

    def to_metadata(self, *, include_paths: bool = False) -> dict[str, Any]:
        if not include_paths:
            return {
                "model_dir_exists": self.model_dir.is_dir(),
                "checkpoint_exists": self.checkpoint_path.is_file(),
                "model_config_exists": self.model_config_path.is_file(),
                "train_config_exists": self.train_config_path.is_file(),
                "config_snapshot_exists": self.config_snapshot_path.is_file(),
                "inference_policy_exists": self.inference_policy_path.is_file(),
                "manifest_exists": self.manifest_path.is_file(),
            }
        return {
            "model_dir": self.model_dir.as_posix(),
            "checkpoint_path": self.checkpoint_path.as_posix(),
            "model_config_path": self.model_config_path.as_posix(),
            "train_config_path": self.train_config_path.as_posix(),
            "config_snapshot_path": self.config_snapshot_path.as_posix(),
            "inference_policy_path": self.inference_policy_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "valid_eval_path": self.valid_eval_path.as_posix() if self.valid_eval_path else None,
            "train_history_path": self.train_history_path.as_posix() if self.train_history_path else None,
        }


@dataclass(frozen=True)
class ChampionModelSpec:
    """Minimal model/inference contract extracted from champion artifacts."""

    model_name: str | None
    model_type: str | None
    input_dim: int | None
    num_classes: int | None
    labels: list[str]
    feature_dim: int | None
    feature_fps: float | None
    window_size: int | None
    window_sec: float | None
    stride_size: int | None
    stride_sec: float | None
    threshold: float | None = None
    local_peak_window_sec: float | None = None
    nms_window_sec: float | None = None
    apply_offset: bool | None = None
    class_aware_nms: bool | None = None
    max_candidates_per_class_per_half: int | None = None
    max_candidates_per_match: int | None = None
    offset_merge: str | None = None
    champion_identifier: str | None = None
    checkpoint_sha256: str | None = None
    raw_model_config: dict[str, Any] = field(default_factory=dict)
    raw_inference_policy: dict[str, Any] = field(default_factory=dict)
    raw_config_snapshot: dict[str, Any] = field(default_factory=dict)
    raw_manifest: dict[str, Any] = field(default_factory=dict)
    raw_label_map: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "labels": self.labels,
            "feature_dim": self.feature_dim,
            "feature_fps": self.feature_fps,
            "window_size": self.window_size,
            "window_sec": self.window_sec,
            "stride_size": self.stride_size,
            "stride_sec": self.stride_sec,
            "threshold": self.threshold,
            "local_peak_window_sec": self.local_peak_window_sec,
            "nms_window_sec": self.nms_window_sec,
            "apply_offset": self.apply_offset,
            "class_aware_nms": self.class_aware_nms,
            "max_candidates_per_class_per_half": self.max_candidates_per_class_per_half,
            "max_candidates_per_match": self.max_candidates_per_match,
            "offset_merge": self.offset_merge,
            "champion_identifier": self.champion_identifier,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class ChampionCheckpointSummary:
    loadable: bool
    reason: str | None = None
    checkpoint_type: str | None = None
    checkpoint_keys: list[str] = field(default_factory=list)
    state_dict_key: str | None = None
    state_dict_num_tensors: int | None = None
    state_dict_sample_keys: list[str] = field(default_factory=list)
    tensor_shapes: dict[str, list[int]] = field(default_factory=dict)
    epoch: int | None = None
    best: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "loadable": self.loadable,
            "reason": self.reason,
            "checkpoint_type": self.checkpoint_type,
            "checkpoint_keys": self.checkpoint_keys,
            "state_dict_key": self.state_dict_key,
            "state_dict_num_tensors": self.state_dict_num_tensors,
            "state_dict_sample_keys": self.state_dict_sample_keys,
            "tensor_shapes": self.tensor_shapes,
            "epoch": self.epoch,
            "best": self.best,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class ChampionAdapterPreflightReport:
    """Path-redacted readiness report for Champion torch inference."""

    ready_for_real_adapter: bool
    reasons: list[str]
    paths: dict[str, Any]
    spec: dict[str, Any]
    checkpoint: dict[str, Any]
    implementation_todos: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ready_for_real_adapter": self.ready_for_real_adapter,
            "reasons": self.reasons,
            "paths": self.paths,
            "spec": self.spec,
            "checkpoint": self.checkpoint,
            "implementation_todos": self.implementation_todos,
        }


@dataclass(frozen=True)
class HighlightRawPrediction:
    label: str
    timestamp_sec: float
    confidence: float
    half: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    highlight_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_postprocessor_input(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "timestamp_sec": self.timestamp_sec,
            "confidence": self.confidence,
            "half": self.half,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "highlight_score": self.highlight_score,
            "metadata": self.metadata,
        }


class HighlightModelAdapter(Protocol):
    """Interface implemented by Action Spotting model adapters."""

    def preflight(self) -> ChampionAdapterPreflightReport:
        ...

    def predict(self, *args: Any, **kwargs: Any) -> list[HighlightRawPrediction]:
        ...
