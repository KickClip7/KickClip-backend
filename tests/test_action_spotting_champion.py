from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from app.ai.registry.model_card import ModelCard
from app.ai.tasks.highlight_spotting.adapters.soccer_highlight_former import (
    CHAMPION_CLASS_ORDER,
    CHAMPION_IDENTIFIER,
    EXPECTED_CHECKPOINT_SHA256,
    SoccerHighlightFormerAdapter,
    _Candidate,
    _HalfTimeline,
    get_window_start_indices,
    sha256_file,
)
from app.ai.tasks.highlight_spotting.config import build_highlight_runtime_config
from app.ai.tasks.highlight_spotting.feature_loader import (
    HighlightFeatureArrayInfo,
    HighlightFeatureBundle,
)
from app.ai.tasks.highlight_spotting.label_map import normalize_label
from app.domains.action_spotting.errors import (
    ACTION_SPOTTING_FEATURE_MISMATCH,
    ActionSpottingError,
)
from app.domains.action_spotting.diagnostics import (
    collect_action_spotting_diagnostics,
)
from app.domains.action_spotting.status import action_spotting_workflow_status


def _adapter() -> SoccerHighlightFormerAdapter:
    return SoccerHighlightFormerAdapter.from_model_dir()


def _model_card() -> ModelCard:
    return ModelCard(
        id=CHAMPION_IDENTIFIER,
        model_name="SoccerHighlightFormer",
        model_version=CHAMPION_IDENTIFIER,
        checkpoint_path=(
            "storage/models/action_spotting/"
            "sampling_v1_loss_v2_ms_stem_v1/transformer_best.pt"
        ),
        config_path=(
            "configs/models/action_spotting/"
            "sampling_v1_loss_v2_ms_stem_v1/model.yaml"
        ),
        extra={"postprocess_config_path": "configs/highlight_spotting.yaml"},
    )


def test_champion_artifacts_and_contract_load_strictly() -> None:
    adapter = _adapter()
    report = adapter.preflight()

    assert report.ready_for_real_adapter, report.reasons
    assert adapter.spec.labels == CHAMPION_CLASS_ORDER
    assert adapter.spec.feature_dim == 512
    assert adapter.spec.feature_fps == 2.0
    assert adapter.spec.window_size == 128
    assert adapter.spec.stride_size == 32
    assert adapter.spec.threshold == 0.20
    assert adapter.spec.local_peak_window_sec == 3.0
    assert adapter.spec.nms_window_sec == 10.0
    assert adapter.spec.offset_merge == "weighted"
    assert (
        sha256_file(adapter.paths.checkpoint_path)
        == EXPECTED_CHECKPOINT_SHA256
    )


def test_champion_output_shape_is_heatmap_and_offset() -> None:
    adapter = _adapter()
    model, torch, device = adapter._load_model(device="cpu")

    with torch.no_grad():
        output = model(torch.zeros((1, 128, 512), dtype=torch.float32).to(device))

    assert tuple(output["heatmap_logits"].shape) == (1, 128, 6)
    assert tuple(output["offset"].shape) == (1, 128, 6)
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_098_828


def test_champion_runs_end_to_end_on_half_feature_contract(tmp_path) -> None:
    paths = [tmp_path / "half1.npy", tmp_path / "half2.npy"]
    for path in paths:
        np.save(path, np.zeros((16, 512), dtype=np.float32))
    assets = [
        SimpleNamespace(
            asset_id=f"asset_{half}",
            match_id="match_1",
            asset_type=f"SOCCERNET_FEATURE_HALF{half}",
            file_path=path.as_posix(),
            duration_sec=8.0,
            fps=2.0,
        )
        for half, path in enumerate(paths, start=1)
    ]
    infos = [
        HighlightFeatureArrayInfo(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            path=path,
            shape=(16, 512),
            dtype="float32",
        )
        for asset, path in zip(assets, paths)
    ]
    bundle = HighlightFeatureBundle(
        assets=assets,
        resolved_paths=paths,
        feature_infos=infos,
        layout="halves",
    )

    predictions = _adapter().predict(feature_bundle=bundle, device="cpu")

    assert len(predictions) <= 80
    assert all(item.label in CHAMPION_CLASS_ORDER for item in predictions)
    assert all(
        item.metadata["player_involvement"]["status"] == "unknown"
        for item in predictions
    )


def test_window_sampling_keeps_the_exact_tail_window() -> None:
    assert get_window_start_indices(80, 128, 32) == [0]
    assert get_window_start_indices(128, 128, 32) == [0]
    assert get_window_start_indices(200, 128, 32) == [0, 32, 64, 72]


def test_offset_is_decoded_in_seconds_and_nms_is_class_aware() -> None:
    adapter = _adapter()
    scores = np.zeros((32, 6), dtype=np.float32)
    offsets = np.zeros_like(scores)
    scores[10, 0] = 0.9
    offsets[10, 0] = 1.25

    candidates = adapter._extract_half_candidates(
        _HalfTimeline(half=1, timesteps=32, scores=scores, offsets=offsets)
    )

    assert len(candidates) == 1
    assert candidates[0].raw_timestamp_sec == 5.0
    assert candidates[0].timestamp_sec == pytest.approx(6.25)

    kept = adapter._apply_class_aware_nms(
        [
            candidates[0],
            _Candidate(
                half=1,
                label="goal",
                label_id=0,
                timestamp_sec=16.25,
                raw_timestamp_sec=15.0,
                offset_sec=1.25,
                timestep_idx=30,
                score=0.8,
            ),
            _Candidate(
                half=1,
                label="shot",
                label_id=1,
                timestamp_sec=6.25,
                raw_timestamp_sec=5.0,
                offset_sec=1.25,
                timestep_idx=10,
                score=0.7,
            ),
        ]
    )
    assert [(item.label, item.score) for item in kept] == [
        ("goal", pytest.approx(0.9)),
        ("shot", pytest.approx(0.7)),
    ]


def test_unsupported_legacy_labels_are_not_remapped() -> None:
    assert normalize_label("penalty") == "penalty"
    assert normalize_label("substitution") == "substitution"


def test_dummy_or_fallback_mode_is_not_accepted() -> None:
    config = build_highlight_runtime_config(
        model_card=_model_card(),
        job_options={"dummy_mode": True},
    )
    assert config.predictor_mode == "unsupported"
    assert config.allow_fallback_when_missing is False


def test_stable_error_status_is_preserved() -> None:
    error = ActionSpottingError(
        ACTION_SPOTTING_FEATURE_MISMATCH,
        "features invalid",
        retryable=True,
    )
    options = {
        "action_spotting_state": error.code,
        "action_spotting_error": error.to_public_dict(),
    }
    assert (
        action_spotting_workflow_status(status="FAILED", options=options)
        == ACTION_SPOTTING_FEATURE_MISMATCH
    )


def test_runtime_diagnostics_cover_required_contract_without_paths() -> None:
    diagnostics = collect_action_spotting_diagnostics()
    assert {
        "ai_project_root_exists",
        "python_executable_exists",
        "checkpoint_exists",
        "checkpoint_sha256",
        "model_config_exists",
        "train_config_exists",
        "model_module_import_success",
        "temporal_stem_import_success",
        "strict_checkpoint_load_success",
        "feature_extractor_available",
        "sample_feature_contract_valid",
        "cuda_available",
        "mps_available",
        "selected_device",
        "ffmpeg_available",
        "postgresql_connectivity",
    }.issubset(diagnostics)
    assert diagnostics["strict_checkpoint_load_success"] is True
    assert diagnostics["sample_feature_contract_valid"] is True
    assert not any("path" in key.lower() for key in diagnostics)
