from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.ai.registry.model_card import ModelCard
from app.core.paths import get_project_root


DEFAULT_EXPECTED_FEATURE_ASSET_TYPES = [
    "SOCCERNET_FEATURE_HALF1",
    "SOCCERNET_FEATURE_HALF2",
    "SOCCERNET_FEATURE",
]


@dataclass(frozen=True)
class HighlightSpottingRuntimeConfig:
    """Runtime config for HighlightSpottingTask.

    This object intentionally separates the service contract from the concrete
    model implementation. The task can keep producing EVENT_CANDIDATES through
    fallback mode while the champion checkpoint / feature pipeline is not ready.
    """

    predictor_mode: str = "auto"  # auto | fallback | real
    allow_fallback_when_missing: bool = True
    strict_real_model: bool = False
    raw_video_fallback_enabled: bool = True
    expected_feature_asset_types: list[str] = field(
        default_factory=lambda: DEFAULT_EXPECTED_FEATURE_ASSET_TYPES.copy()
    )
    model_adapter: str = "placeholder_soccer_highlight_former"
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
        return self.predictor_mode == "fallback"

    @property
    def request_real_model(self) -> bool:
        return self.predictor_mode == "real" or self.strict_real_model


def build_highlight_runtime_config(
    *,
    model_card: ModelCard,
    job_options: dict[str, Any] | None,
) -> HighlightSpottingRuntimeConfig:
    job_options = job_options or {}
    raw_config = _load_highlight_yaml(model_card)
    inference = raw_config.get("inference") or {}

    # job option이 가장 우선이다.
    # dummy_mode=true는 기존 E2E 흐름을 절대 깨지 않도록 fallback을 강제한다.
    if job_options.get("dummy_mode") is True:
        predictor_mode = "fallback"
    else:
        predictor_mode = str(
            job_options.get("highlight_predictor_mode")
            or job_options.get("predictor_mode")
            or inference.get("mode")
            or "auto"
        ).lower()

    if predictor_mode not in {"auto", "fallback", "real"}:
        raise ValueError(
            "highlight predictor mode must be one of: auto, fallback, real"
        )

    strict_real_model = bool(
        job_options.get("strict_real_model", inference.get("strict_real_model", False))
    )

    allow_fallback_when_missing = bool(
        job_options.get(
            "allow_fallback_when_missing",
            inference.get("allow_fallback_when_missing", True),
        )
    )

    if strict_real_model:
        allow_fallback_when_missing = False
        predictor_mode = "real"

    expected_feature_asset_types = list(
        job_options.get("expected_feature_asset_types")
        or inference.get("expected_feature_asset_types")
        or DEFAULT_EXPECTED_FEATURE_ASSET_TYPES
    )

    return HighlightSpottingRuntimeConfig(
        predictor_mode=predictor_mode,
        allow_fallback_when_missing=allow_fallback_when_missing,
        strict_real_model=strict_real_model,
        raw_video_fallback_enabled=bool(
            inference.get("raw_video_fallback_enabled", True)
        ),
        expected_feature_asset_types=expected_feature_asset_types,
        model_adapter=str(inference.get("model_adapter") or "placeholder_soccer_highlight_former"),
        device=str(job_options.get("device") or inference.get("device") or "auto"),
        real_adapter_batch_size=_optional_int(
            job_options.get("real_adapter_batch_size")
            or inference.get("real_adapter_batch_size")
            or inference.get("batch_size")
        ),
        real_adapter_max_candidates=_optional_int(
            job_options.get("real_adapter_max_candidates")
            or inference.get("real_adapter_max_candidates")
            or inference.get("max_candidates")
        ),
        debug_min_threshold=_optional_float(
            job_options.get("debug_min_threshold")
            or inference.get("debug_min_threshold")
        ),
        checkpoint_path=_resolve_optional_path(model_card.checkpoint_path),
        model_config_path=_resolve_optional_path(model_card.config_path),
        label_map_path=_resolve_optional_path(model_card.extra.get("label_map_path")),
        postprocess_config_path=_resolve_optional_path(
            model_card.extra.get("postprocess_config_path")
            or "configs/highlight_spotting.yaml"
        ),
        raw_config=raw_config,
    )


def load_highlight_postprocess_config(model_card: ModelCard) -> dict[str, Any]:
    raw_config = _load_highlight_yaml(model_card)
    postprocess = raw_config.get("postprocess") or {}

    default_config = {
        "default_threshold": 0.35,
        "max_candidates": 20,
        "nms_window_sec": 8.0,
        "min_duration_sec": 4.0,
        "max_duration_sec": 12.0,
        "pre_event_sec": 4.0,
        "post_event_sec": 4.0,
        "class_thresholds": {},
    }

    return {
        "default_threshold": postprocess.get(
            "default_threshold",
            default_config["default_threshold"],
        ),
        "max_candidates": postprocess.get(
            "max_candidates",
            default_config["max_candidates"],
        ),
        "nms_window_sec": postprocess.get(
            "nms_window_sec",
            default_config["nms_window_sec"],
        ),
        "min_duration_sec": postprocess.get(
            "min_duration_sec",
            default_config["min_duration_sec"],
        ),
        "max_duration_sec": postprocess.get(
            "max_duration_sec",
            default_config["max_duration_sec"],
        ),
        "pre_event_sec": postprocess.get(
            "pre_event_sec",
            default_config["pre_event_sec"],
        ),
        "post_event_sec": postprocess.get(
            "post_event_sec",
            default_config["post_event_sec"],
        ),
        "class_thresholds": postprocess.get(
            "class_thresholds",
            default_config["class_thresholds"],
        ),
    }


def _load_highlight_yaml(model_card: ModelCard) -> dict[str, Any]:
    path = model_card.extra.get("postprocess_config_path") or "configs/highlight_spotting.yaml"
    config_path = _resolve_optional_path(path)

    if config_path is None or not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("configs/highlight_spotting.yaml must contain a YAML object.")

    return loaded


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    return get_project_root() / candidate


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
