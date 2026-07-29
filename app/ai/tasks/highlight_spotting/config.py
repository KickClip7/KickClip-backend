from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.ai.registry.model_card import ModelCard
from app.ai.tasks.highlight_spotting.postprocessor import DEFAULT_EVENT_WINDOWS
from app.core.paths import get_project_root
from app.domains.action_spotting.errors import inference_policy_incomplete


DEFAULT_EXPECTED_FEATURE_ASSET_TYPES = [
    "SOCCERNET_FEATURE_HALF1",
    "SOCCERNET_FEATURE_HALF2",
]


@dataclass(frozen=True)
class HighlightSpottingRuntimeConfig:
    predictor_mode: str = "real"
    allow_fallback_when_missing: bool = False
    strict_real_model: bool = True
    raw_video_fallback_enabled: bool = False
    expected_feature_asset_types: list[str] = field(
        default_factory=lambda: DEFAULT_EXPECTED_FEATURE_ASSET_TYPES.copy()
    )
    model_adapter: str = "soccer_spotter_v9"
    device: str = "auto"
    real_adapter_batch_size: int | None = None
    real_adapter_max_candidates: int | None = None
    debug_min_threshold: float | None = None
    checkpoint_path: Path | None = None
    model_config_path: Path | None = None
    label_map_path: Path | None = None
    postprocess_config_path: Path | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def force_fallback(self) -> bool:
        return False

    @property
    def request_real_model(self) -> bool:
        return self.predictor_mode == "real"


def build_highlight_runtime_config(
    *,
    model_card: ModelCard,
    job_options: dict[str, Any] | None,
) -> HighlightSpottingRuntimeConfig:
    job_options = job_options or {}
    raw_config = _load_highlight_yaml(model_card)
    inference = raw_config.get("inference") or {}
    requested_mode = str(
        job_options.get("highlight_predictor_mode")
        or job_options.get("predictor_mode")
        or inference.get("mode")
        or "real"
    ).lower()
    predictor_mode = (
        "real"
        if requested_mode == "real" and job_options.get("dummy_mode") is not True
        else "unsupported"
    )
    return HighlightSpottingRuntimeConfig(
        predictor_mode=predictor_mode,
        expected_feature_asset_types=list(
            inference.get("expected_feature_asset_types")
            or DEFAULT_EXPECTED_FEATURE_ASSET_TYPES
        ),
        model_adapter=str(
            inference.get("model_adapter")
            or "soccer_spotter_v9"
        ),
        device=str(job_options.get("device") or inference.get("device") or "auto"),
        real_adapter_batch_size=_optional_int(
            inference.get("batch_size")
        ),
        # These policies are intentionally not user-overridable.
        real_adapter_max_candidates=_optional_int(
            inference.get("max_candidates")
        ),
        debug_min_threshold=None,
        checkpoint_path=_resolve_optional_path(model_card.checkpoint_path),
        model_config_path=_resolve_optional_path(model_card.config_path),
        postprocess_config_path=_resolve_optional_path(
            model_card.extra.get("postprocess_config_path")
            or "configs/highlight_spotting.yaml"
        ),
        raw_config=raw_config,
    )


def load_highlight_postprocess_config(model_card: ModelCard) -> dict[str, Any]:
    postprocess = (_load_highlight_yaml(model_card).get("postprocess") or {})
    config = {
        # Candidate threshold/NMS are the exact Champion evaluation policy.
        "default_threshold": float(postprocess.get("default_threshold", 0.0)),
        "max_candidates": int(postprocess.get("max_candidates", 113)),
        "nms_window_sec": float(postprocess.get("nms_window_sec", 0.0)),
        # Scene-window behavior is a backend presentation policy and remains
        # separate from model event decoding.
        "merge_overlapping_scenes": bool(
            postprocess.get("merge_overlapping_scenes", False)
        ),
        "event_windows": postprocess.get("event_windows"),
        "min_duration_sec": float(postprocess.get("min_duration_sec", 4.0)),
        "max_duration_sec": float(postprocess.get("max_duration_sec", 12.0)),
        "pre_event_sec": float(postprocess.get("pre_event_sec", 4.0)),
        "post_event_sec": float(postprocess.get("post_event_sec", 4.0)),
        "class_thresholds": postprocess.get("class_thresholds") or {},
        "class_priority": postprocess.get("class_priority") or {
            "goal": 100,
            "penalty": 90,
            "shot": 80,
            "free_kick": 70,
            "corner": 60,
            "foul": 50,
            "card": 40,
        },
    }
    expected_decode_policy = {
        "default_threshold": 0.0,
        "max_candidates": 113,
        "nms_window_sec": 0.0,
        "class_thresholds": {
            "goal": 0.25,
            "shot": 0.55,
            "penalty": 0.20,
            "card": 0.75,
            "corner": 0.30,
        },
    }
    actual_decode_policy = {
        key: config[key] for key in expected_decode_policy
    }
    if actual_decode_policy != expected_decode_policy:
        raise inference_policy_incomplete(
            "Backend Champion candidate policy is inconsistent.",
            expected=expected_decode_policy,
            actual=actual_decode_policy,
        )
    actual_event_windows = {
        label: {
            "before_sec": float(window["before_sec"]),
            "after_sec": float(window["after_sec"]),
        }
        for label, window in (config["event_windows"] or {}).items()
    }
    if (
        config["merge_overlapping_scenes"] is not False
        or actual_event_windows != DEFAULT_EVENT_WINDOWS
    ):
        raise inference_policy_incomplete(
            "Backend event clip-window policy is inconsistent.",
            expected={
                "merge_overlapping_scenes": False,
                "event_windows": DEFAULT_EVENT_WINDOWS,
            },
            actual={
                "merge_overlapping_scenes": config["merge_overlapping_scenes"],
                "event_windows": actual_event_windows,
            },
        )
    config["event_windows"] = actual_event_windows
    return config


def _load_highlight_yaml(model_card: ModelCard) -> dict[str, Any]:
    path = (
        model_card.extra.get("postprocess_config_path")
        or "configs/highlight_spotting.yaml"
    )
    config_path = _resolve_optional_path(path)
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("highlight_spotting.yaml must contain a YAML object")
    return loaded


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else get_project_root() / candidate


def _optional_int(value: Any) -> int | None:
    return int(value) if value not in (None, "") else None
