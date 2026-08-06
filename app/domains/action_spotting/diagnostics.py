from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ai.tasks.highlight_spotting.adapters.soccer_spotter_v9 import (
    CHAMPION_CLASS_ORDER,
    CHAMPION_IDENTIFIER,
    DEFAULT_CHAMPION_MODEL_DIR,
    EXPECTED_CHECKPOINT_SHA256,
    SoccerSpotterV9Adapter,
    resolve_v9_artifact_paths,
    sha256_file,
)
from app.ai.tasks.soccernet_feature_extraction.config import (
    build_soccernet_feature_extraction_config,
)
from app.core.config import get_settings
from app.core.paths import get_project_root


def collect_action_spotting_diagnostics(db: Session | None = None) -> dict[str, Any]:
    """Return path-redacted runtime readiness diagnostics."""

    settings = get_settings()
    project_root = get_project_root()
    configured_ai_project_root = settings.ACTION_SPOTTING_AI_PROJECT_ROOT.strip()
    ai_project_root = Path(configured_ai_project_root) if configured_ai_project_root else project_root
    configured_python = settings.ACTION_SPOTTING_PYTHON_EXECUTABLE.strip() or sys.executable
    python_executable = _resolve_executable(configured_python)
    paths = resolve_v9_artifact_paths(DEFAULT_CHAMPION_MODEL_DIR)
    checkpoint_sha256 = (
        sha256_file(paths.checkpoint_path)
        if paths.checkpoint_path.is_file()
        else None
    )
    result: dict[str, Any] = {
        "champion_identifier": CHAMPION_IDENTIFIER,
        "ai_project_root_configured": bool(configured_ai_project_root),
        "ai_project_root_exists": ai_project_root.is_dir(),
        "python_executable_exists": python_executable is not None,
        "checkpoint_exists": paths.checkpoint_path.is_file(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_matches": checkpoint_sha256
        == EXPECTED_CHECKPOINT_SHA256,
        "model_config_exists": paths.model_config_path.is_file(),
        # Kept for API compatibility; v9 has one checkpoint-paired JSON config.
        "train_config_exists": paths.model_config_path.is_file(),
        "class_order": CHAMPION_CLASS_ORDER,
        "model_module_import_success": False,
        "temporal_stem_import_success": False,
        "strict_checkpoint_load_success": False,
        "feature_extractor_available": False,
        "feature_runtime_import_success": False,
        "sample_feature_contract_valid": False,
        "cuda_available": False,
        "mps_available": False,
        "selected_device": None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
        "postgresql_connectivity": False,
        "errors": [],
    }

    try:
        from app.ai.vendor.kickclip_v9 import SoccerSpotterV9

        result["model_module_import_success"] = SoccerSpotterV9 is not None
    except Exception as exc:
        result["errors"].append(_safe_error("model_import", exc))

    try:
        from app.ai.vendor.kickclip_v9 import ResidualTCNBlock

        result["temporal_stem_import_success"] = ResidualTCNBlock is not None
    except Exception as exc:
        result["errors"].append(_safe_error("temporal_stem_import", exc))

    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        result["mps_available"] = bool(
            mps_backend is not None
            and mps_backend.is_available()
            and mps_backend.is_built()
        )
        adapter = SoccerSpotterV9Adapter(paths)
        adapter._validate_contract()
        model, _, device = adapter._load_model(device="auto")
        result["strict_checkpoint_load_success"] = True
        result["selected_device"] = str(device)
        with torch.no_grad():
            logits, offset_mean, offset_logvar, eventness = model(
                torch.zeros((1, 128, 512), dtype=torch.float32).to(device)
            )
        result["sample_feature_contract_valid"] = all(
            (
                tuple(logits.shape) == (1, 128, 5),
                tuple(offset_mean.shape) == (1, 128, 5),
                tuple(offset_logvar.shape) == (1, 128, 5),
                tuple(eventness.shape) == (1, 128),
            )
        )
    except Exception as exc:
        result["errors"].append(_safe_error("model_contract", exc))

    try:
        feature_config = build_soccernet_feature_extraction_config({})
        result["feature_extractor_available"] = all(
            (
                feature_config.sn_spotting_root.is_dir(),
                feature_config.video_feature_extractor.is_file(),
                feature_config.pca_path.is_file(),
                feature_config.pca_scaler_path.is_file(),
                feature_config.output_dim == 512,
            )
        )
        feature_python = _resolve_executable(feature_config.python_executable)
        result["python_executable_exists"] = feature_python is not None
        if feature_python is not None:
            completed = subprocess.run(
                [
                    feature_python,
                    "-c",
                    (
                        "import tensorflow; import SoccerNet; import cv2; import sklearn; "
                        "import skvideo.io; import imutils; import numpy; "
                        "print('feature-runtime-ok')"
                    ),
                ],
                cwd=str(feature_config.sn_spotting_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=feature_config.runtime_probe_timeout_sec,
            )
            result["feature_runtime_import_success"] = completed.returncode == 0
            if completed.returncode != 0:
                result["errors"].append(
                    {
                        "component": "feature_runtime",
                        "type": "ImportError",
                        "message": "feature extraction Python dependencies are unavailable",
                    }
                )
    except Exception as exc:
        result["errors"].append(_safe_error("feature_extractor", exc))

    if db is not None:
        try:
            dialect = db.get_bind().dialect.name
            db.execute(text("SELECT 1"))
            result["postgresql_connectivity"] = dialect == "postgresql"
            result["database_dialect"] = dialect
        except Exception as exc:
            result["errors"].append(_safe_error("database", exc))

    result["ready"] = all(
        (
            result["ai_project_root_exists"],
            result["python_executable_exists"],
            result["checkpoint_sha256_matches"],
            result["model_config_exists"],
            result["train_config_exists"],
            result["model_module_import_success"],
            result["temporal_stem_import_success"],
            result["strict_checkpoint_load_success"],
            result["feature_extractor_available"],
            result["feature_runtime_import_success"],
            result["sample_feature_contract_valid"],
            result["ffmpeg_available"],
            result["ffprobe_available"],
            result["postgresql_connectivity"] if db is not None else True,
        )
    )
    return result


def _safe_error(component: str, exc: Exception) -> dict[str, str]:
    return {
        "component": component,
        "type": type(exc).__name__,
        "message": "runtime check failed; inspect the authenticated job diagnostics artifact",
    }


def _resolve_executable(value: str) -> str | None:
    candidate = Path(value)
    if candidate.is_file():
        return str(candidate)
    return shutil.which(value)
