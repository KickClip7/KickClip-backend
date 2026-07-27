from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ai.tasks.highlight_spotting.adapters.soccer_highlight_former import (
    CHAMPION_CLASS_ORDER,
    EXPECTED_CHECKPOINT_SHA256,
    SoccerHighlightFormerAdapter,
    resolve_champion_artifact_paths,
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
    ai_project_root = Path(
        settings.ACTION_SPOTTING_AI_PROJECT_ROOT
        or project_root.parent / "KickClip"
    )
    python_executable = Path(
        settings.ACTION_SPOTTING_PYTHON_EXECUTABLE or sys.executable
    )
    paths = resolve_champion_artifact_paths(
        "storage/models/action_spotting/sampling_v1_loss_v2_ms_stem_v1"
    )
    checkpoint_sha256 = (
        sha256_file(paths.checkpoint_path)
        if paths.checkpoint_path.is_file()
        else None
    )
    result: dict[str, Any] = {
        "champion_identifier": "sampling_v1_loss_v2_ms_stem_v1",
        "ai_project_root_exists": ai_project_root.is_dir(),
        "python_executable_exists": python_executable.is_file(),
        "checkpoint_exists": paths.checkpoint_path.is_file(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_matches": checkpoint_sha256
        == EXPECTED_CHECKPOINT_SHA256,
        "model_config_exists": paths.model_config_path.is_file(),
        "train_config_exists": paths.train_config_path.is_file(),
        "class_order": CHAMPION_CLASS_ORDER,
        "model_module_import_success": False,
        "temporal_stem_import_success": False,
        "strict_checkpoint_load_success": False,
        "feature_extractor_available": False,
        "sample_feature_contract_valid": False,
        "cuda_available": False,
        "mps_available": False,
        "selected_device": None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "postgresql_connectivity": False,
        "errors": [],
    }

    try:
        from app.ai.vendor.kickclip_champion.models.soccer_highlight_former import (
            SoccerHighlightFormer,
        )

        result["model_module_import_success"] = SoccerHighlightFormer is not None
    except Exception as exc:
        result["errors"].append(_safe_error("model_import", exc))

    try:
        from app.ai.vendor.kickclip_champion.models.multiscale_temporal_stem import (
            MultiScaleTemporalStem,
        )

        result["temporal_stem_import_success"] = MultiScaleTemporalStem is not None
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
        adapter = SoccerHighlightFormerAdapter(paths)
        adapter._validate_contract()
        model, _, device = adapter._load_model(device="auto")
        result["strict_checkpoint_load_success"] = True
        result["selected_device"] = str(device)
        with torch.no_grad():
            output = model(torch.zeros((1, 128, 512), dtype=torch.float32).to(device))
        result["sample_feature_contract_valid"] = (
            tuple(output["heatmap_logits"].shape) == (1, 128, 6)
            and tuple(output["offset"].shape) == (1, 128, 6)
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
            result["sample_feature_contract_valid"],
            result["ffmpeg_available"],
            result["postgresql_connectivity"],
        )
    )
    return result


def _safe_error(component: str, exc: Exception) -> dict[str, str]:
    return {
        "component": component,
        "type": type(exc).__name__,
        "message": "runtime check failed; inspect the authenticated job diagnostics artifact",
    }
