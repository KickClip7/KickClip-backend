from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.config import get_settings
from app.core.paths import get_project_root


VALID_FEATURE_EXTRACTION_MODES = {"auto", "force", "skip"}
VALID_SPLIT_STRATEGIES = {"midpoint", "manual", "none"}
DEFAULT_CONFIG_PATH = "configs/soccernet_feature_extraction.yaml"


@dataclass(frozen=True)
class SoccerNetFeatureExtractionConfig:
    """Runtime config for SoccerNet-style feature extraction.

    이 config는 YAML 기본값과 AnalysisJob options override를 합친 결과다.
    모든 경로 필드는 project root 기준 relative path도 absolute Path로 변환해서 들고 있다.
    """

    enabled: bool = True
    mode: str = "auto"  # auto | force | skip

    extractor_name: str = "sn_spotting_resnet_tf2_pca512"
    output_dim: int = 512
    chunk_sec: float = 300.0
    batch_size: int = 16
    python_executable: str = field(default_factory=lambda: sys.executable)
    runtime_probe_timeout_sec: float = 120.0

    sn_spotting_root: Path = field(default_factory=lambda: get_project_root() / "external/sn-spotting")
    video_feature_extractor: Path = field(
        default_factory=lambda: get_project_root()
        / "external/sn-spotting/Features/VideoFeatureExtractor.py"
    )
    pca_path: Path = field(
        default_factory=lambda: get_project_root()
        / "external/sn-spotting/Features/pca_512_TF2.pkl"
    )
    pca_scaler_path: Path = field(
        default_factory=lambda: get_project_root()
        / "external/sn-spotting/Features/average_512_TF2.pkl"
    )

    split_strategy: str = "midpoint"  # midpoint | manual | none
    allow_job_option_halftime_split_sec: bool = True
    halftime_split_sec: float | None = None

    reuse_existing_feature: bool = True
    overwrite: bool = False

    allow_analysis_to_continue_on_failure: bool = True

    config_path: Path | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def should_skip(self) -> bool:
        return (not self.enabled) or self.mode == "skip"

    @property
    def should_force(self) -> bool:
        return self.mode == "force" or self.overwrite

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "extractor_name": self.extractor_name,
            "output_dim": self.output_dim,
            "chunk_sec": self.chunk_sec,
            "batch_size": self.batch_size,
            "python_executable": self.python_executable,
            "runtime_probe_timeout_sec": self.runtime_probe_timeout_sec,
            "sn_spotting_root": self.sn_spotting_root.as_posix(),
            "video_feature_extractor": self.video_feature_extractor.as_posix(),
            "pca_path": self.pca_path.as_posix(),
            "pca_scaler_path": self.pca_scaler_path.as_posix(),
            "split_strategy": self.split_strategy,
            "allow_job_option_halftime_split_sec": self.allow_job_option_halftime_split_sec,
            "halftime_split_sec": self.halftime_split_sec,
            "reuse_existing_feature": self.reuse_existing_feature,
            "overwrite": self.overwrite,
            "allow_analysis_to_continue_on_failure": self.allow_analysis_to_continue_on_failure,
            "config_path": self.config_path.as_posix() if self.config_path else None,
        }


def build_soccernet_feature_extraction_config(
    job_options: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
) -> SoccerNetFeatureExtractionConfig:
    """Build feature extraction config from YAML and job options.

    Priority:
    1. AnalysisJob options
    2. configs/soccernet_feature_extraction.yaml
    3. dataclass defaults
    """

    job_options = job_options or {}
    settings = get_settings()
    resolved_config_path = _resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    raw_config = _load_yaml_config(resolved_config_path)
    section = raw_config.get("feature_extraction") or {}

    split = section.get("split") or {}
    cache = section.get("cache") or {}
    failure_policy = section.get("failure_policy") or {}

    enabled = bool(section.get("enabled", True))
    if "run_feature_extraction" in job_options:
        enabled = bool(job_options["run_feature_extraction"])

    mode = str(
        job_options.get("feature_extraction_mode")
        or section.get("mode")
        or "auto"
    ).lower()

    output_dim = int(
        job_options.get("feature_output_dim")
        or job_options.get("soccernet_feature_output_dim")
        or section.get("output_dim")
        or 512
    )

    chunk_sec = float(
        job_options.get("feature_chunk_sec")
        or job_options.get("soccernet_feature_chunk_sec")
        or section.get("chunk_sec")
        or 300
    )

    batch_size = int(
        job_options.get("feature_batch_size")
        or job_options.get("soccernet_feature_batch_size")
        or section.get("batch_size")
        or 16
    )

    python_executable = str(
        job_options.get("feature_python_executable")
        or settings.ACTION_SPOTTING_PYTHON_EXECUTABLE
        or section.get("python_executable")
        or sys.executable
    )

    runtime_probe_timeout_sec = float(
        job_options.get("feature_runtime_probe_timeout_sec")
        or section.get("runtime_probe_timeout_sec")
        or 120
    )

    allow_job_option_halftime_split_sec = bool(
        split.get("allow_job_option_halftime_split_sec", True)
    )

    halftime_split_sec = _optional_float(job_options.get("halftime_split_sec"))

    split_strategy = str(
        job_options.get("feature_split_strategy")
        or split.get("strategy")
        or "midpoint"
    ).lower()

    # 사용자가 halftime_split_sec를 주면 기본 midpoint 설정이어도 manual split으로 해석한다.
    if halftime_split_sec is not None and allow_job_option_halftime_split_sec:
        split_strategy = "manual"

    overwrite = bool(
        job_options.get(
            "feature_extraction_overwrite",
            cache.get("overwrite", False),
        )
    )

    # force 모드에서는 VideoFeatureExtractor.py에도 overwrite를 넘기는 것이 자연스럽다.
    if mode == "force":
        overwrite = True

    config = SoccerNetFeatureExtractionConfig(
        enabled=enabled,
        mode=mode,
        extractor_name=str(
            job_options.get("feature_extractor_name")
            or section.get("extractor_name")
            or "sn_spotting_resnet_tf2_pca512"
        ),
        output_dim=output_dim,
        chunk_sec=chunk_sec,
        batch_size=batch_size,
        python_executable=python_executable,
        runtime_probe_timeout_sec=runtime_probe_timeout_sec,
        sn_spotting_root=_resolve_project_path(
            job_options.get("sn_spotting_root")
            or section.get("sn_spotting_root")
            or "external/sn-spotting"
        ),
        video_feature_extractor=_resolve_project_path(
            job_options.get("video_feature_extractor")
            or section.get("video_feature_extractor")
            or "external/sn-spotting/Features/VideoFeatureExtractor.py"
        ),
        pca_path=_resolve_project_path(
            job_options.get("pca_path")
            or section.get("pca_path")
            or "external/sn-spotting/Features/pca_512_TF2.pkl"
        ),
        pca_scaler_path=_resolve_project_path(
            job_options.get("pca_scaler_path")
            or section.get("pca_scaler_path")
            or "external/sn-spotting/Features/average_512_TF2.pkl"
        ),
        split_strategy=split_strategy,
        allow_job_option_halftime_split_sec=allow_job_option_halftime_split_sec,
        halftime_split_sec=halftime_split_sec,
        reuse_existing_feature=bool(
            job_options.get(
                "reuse_existing_feature",
                cache.get("reuse_existing_feature", True),
            )
        ),
        overwrite=overwrite,
        allow_analysis_to_continue_on_failure=bool(
            job_options.get(
                "feature_extraction_allow_continue_on_failure",
                failure_policy.get("allow_analysis_to_continue_on_failure", True),
            )
        ),
        config_path=resolved_config_path,
        raw_config=raw_config,
    )
    _validate_config(config)
    return config


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"{config_path.as_posix()} must contain a YAML object at the top level."
        )

    return loaded


def _resolve_project_path(path: str | Path | None) -> Path:
    if path is None:
        return get_project_root()

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    return get_project_root() / candidate


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _validate_config(config: SoccerNetFeatureExtractionConfig) -> None:
    if config.mode not in VALID_FEATURE_EXTRACTION_MODES:
        raise ValueError(
            "feature_extraction_mode must be one of: "
            f"{sorted(VALID_FEATURE_EXTRACTION_MODES)}"
        )

    if config.split_strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(
            "feature split strategy must be one of: "
            f"{sorted(VALID_SPLIT_STRATEGIES)}"
        )

    if config.output_dim <= 0:
        raise ValueError("feature output_dim must be greater than 0.")

    if config.batch_size <= 0:
        raise ValueError("feature batch_size must be greater than 0.")

    if not config.python_executable.strip():
        raise ValueError("feature python_executable must not be empty.")

    if config.runtime_probe_timeout_sec <= 0:
        raise ValueError("feature runtime_probe_timeout_sec must be greater than 0.")

    if config.chunk_sec <= 0:
        raise ValueError("feature chunk_sec must be greater than 0.")

    if config.halftime_split_sec is not None and config.halftime_split_sec <= 0:
        raise ValueError("halftime_split_sec must be greater than 0 when provided.")
