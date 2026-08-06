from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    PROJECT_ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("kickclip_phase4b_r9_adapter_test", ADAPTER_PATH)
VERIFIER_PATH = PROJECT_ROOT / "scripts" / "verify_phase4b_provenance_role_review_contract.py"
VERIFIER = _load("kickclip_phase4b_r91_verifier_test", VERIFIER_PATH)


def _unit(*values: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32)
    return row / np.linalg.norm(row)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_bank(policy: str, dimension: int = 3) -> dict[str, object]:
    return {
        "policy": policy,
        "gallery": np.empty((0, dimension), dtype=np.float32),
        "scale_classes": [],
        "clusters": [],
        "embedding_count": 0,
        "cluster_count": 0,
    }


def _candidate_row(
    candidate_id: str,
    start_frame: int,
    end_frame: int,
    detection_count: int,
    clean_count: int,
    clean_ratio: float,
    retrieval_score: float,
    prototype_similarity: float,
    prototype: np.ndarray,
    *,
    target_context_similarity: float = 0.50,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "start_frame": start_frame,
        "end_frame_inclusive": end_frame,
        "tracklet_detection_count": detection_count,
        "detection_count": detection_count,
        "retrieval_score": retrieval_score,
        "prototype_target_similarity": prototype_similarity,
        "identity_observability": {
            "passed": True,
            "clean_frame_count": clean_count,
            "clean_frame_ratio": clean_ratio,
        },
        "identity_purity": {
            "policy": ADAPTER.PHASE4B_IDENTITY_PURITY_POLICY,
            "passed": True,
            "parent_tracklet_id": candidate_id,
            "internal_pairwise_cosine_median": 0.90,
            "target_memory_used_for_segmentation": False,
            "automatic_target_confirmation": False,
        },
        "parent_tracklet_id": candidate_id,
        "negative_review_gate": {
            "passed": True,
            "user_confirmed_role_gate": {"passed": True},
            "identity_negative_gate": {"passed": True},
        },
        "candidate_scale_class_counts": {"wide": 12},
        "target_context_policy": ADAPTER.PHASE4B_TARGET_CONTEXT_POLICY,
        "target_context_similarity": target_context_similarity,
        "target_context_is_soft_evidence_only": True,
        "_prototype_vector": prototype,
    }


def _assignments(
    candidate_id: str,
    start_frame: int,
    end_frame: int,
    *,
    x_start: float,
    x_step: float,
) -> list[dict[str, float | int | str]]:
    rows = []
    for frame in range(start_frame, end_frame + 1):
        x1 = x_start + (frame - start_frame) * x_step
        rows.append(
            {
                "candidate_id": candidate_id,
                "analysis_local_frame_index": frame,
                "x1": x1,
                "y1": 400.0,
                "x2": x1 + 50.0,
                "y2": 500.0,
            }
        )
    return rows


def test_detector_role_evidence_is_soft_and_cannot_reject_candidate() -> None:
    target = np.stack([_unit(1.0, 0.0, 0.0)])
    candidate = np.stack(
        [
            _unit(0.98, 0.15, 0.0),
            _unit(0.97, 0.20, 0.0),
            _unit(0.96, 0.25, 0.0),
        ]
    )
    detector_role_gallery = np.stack(
        [
            _unit(0.99, 0.10, 0.0),
            _unit(0.98, 0.15, 0.0),
        ]
    )
    metrics = ADAPTER._phase4b_compute_banked_negative_metrics(
        candidate_embeddings=candidate,
        candidate_scale_classes=["wide", "wide", "wide"],
        candidate_prototype=_unit(0.97, 0.20, 0.0),
        target_gallery=target,
        identity_bank=_empty_bank(
            ADAPTER.PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY
        ),
        user_confirmed_role_bank=_empty_bank(
            ADAPTER.PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY
        ),
        detector_role_negative_gallery=detector_role_gallery,
        np=np,
    )
    assert metrics["detector_role_crop_margin_median"] < 0.05
    assert metrics["user_role_negative_scale_compatible_evidence_available"] is False

    gate = ADAPTER._phase4b_candidate_negative_review_gate(
        metrics,
        role_negative_memory_available=False,
        detector_role_memory_available=True,
        identity_negative_memory_available=False,
    )
    assert gate["passed"] is True
    assert gate["detector_role_soft_gate"]["passed"] is True
    assert gate["detector_role_soft_gate"]["soft_warning"] is True
    assert gate["rejection_reasons"] == []


def test_user_confirmed_role_negative_is_scale_compatible_hard_gate() -> None:
    target = np.stack([_unit(1.0, 0.0, 0.0)])
    candidate = np.stack(
        [
            _unit(0.92, 0.39, 0.0),
            _unit(0.90, 0.43, 0.0),
            _unit(0.88, 0.47, 0.0),
        ]
    )
    user_role_gallery = np.stack(
        [
            _unit(0.91, 0.41, 0.0),
            _unit(0.89, 0.45, 0.0),
            _unit(0.87, 0.49, 0.0),
        ]
    )
    user_bank = {
        "gallery": user_role_gallery,
        "scale_classes": ["wide", "wide", "wide"],
        "clusters": [
            {
                "candidate_id": "confirmed_non_player",
                "scale_class": "wide",
                "prototype": _unit(0.89, 0.45, 0.0),
            }
        ],
    }
    metrics = ADAPTER._phase4b_compute_banked_negative_metrics(
        candidate_embeddings=candidate,
        candidate_scale_classes=["wide", "wide", "wide"],
        candidate_prototype=_unit(0.90, 0.43, 0.0),
        target_gallery=target,
        identity_bank=_empty_bank(
            ADAPTER.PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY
        ),
        user_confirmed_role_bank=user_bank,
        detector_role_negative_gallery=np.empty((0, 3), dtype=np.float32),
        np=np,
    )
    assert metrics["user_role_negative_scale_compatible_evidence_available"] is True
    gate = ADAPTER._phase4b_candidate_negative_review_gate(
        metrics,
        role_negative_memory_available=True,
        detector_role_memory_available=False,
        identity_negative_memory_available=False,
    )
    assert gate["user_confirmed_role_gate"]["passed"] is False
    assert gate["passed"] is False
    assert all(
        reason.startswith("user_role:") for reason in gate["rejection_reasons"]
    )


def test_cross_scale_user_role_memory_does_not_hard_reject() -> None:
    row = {
        "user_role_negative_scale_compatible_crop_count": 0,
        "user_role_negative_compatible_cluster_count": 0,
    }
    gate = ADAPTER._phase4b_user_confirmed_role_review_gate(
        row,
        negative_memory_available=True,
    )
    assert gate["passed"] is True
    assert gate["scale_compatible_evidence_available"] is False


def test_loader_uses_only_user_confirmed_role_references(tmp_path: Path) -> None:
    user_embeddings = np.stack([_unit(1.0, 0.0, 0.0), _unit(0.9, 0.1, 0.0)])
    detector_embeddings = np.stack([_unit(0.0, 1.0, 0.0)])
    user_path = tmp_path / "user.npy"
    detector_path = tmp_path / "detector.npy"
    np.save(user_path, user_embeddings, allow_pickle=False)
    np.save(detector_path, detector_embeddings, allow_pickle=False)

    candidate_manifest = tmp_path / "candidate.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                "reference_gallery": [
                    {"scale_class": "wide"},
                    {"scale_class": "wide"},
                ]
            }
        ),
        encoding="utf-8",
    )
    persistent_manifest = tmp_path / "persistent.json"
    persistent_manifest.write_text(
        json.dumps(
            {
                "references": [
                    {
                        "source_kind": "USER_CONFIRMED_NON_PLAYER_ROLE_CANDIDATE",
                        "candidate_id": "user_role",
                        "embeddings_path": str(user_path),
                        "embeddings_sha256": _sha(user_path),
                        "manifest_path": str(candidate_manifest),
                        "manifest_sha256": _sha(candidate_manifest),
                    },
                    {
                        "source_kind": "DETECTOR_LABELED_STAFF_REFEREE",
                        "candidate_id": "detector_role",
                        "embeddings_path": str(detector_path),
                        "embeddings_sha256": _sha(detector_path),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    bank = ADAPTER._phase4b_load_user_confirmed_role_scoring_bank(
        memory={
            "manifest_path": str(persistent_manifest),
            "manifest_sha256": _sha(persistent_manifest),
        },
        target_dimension=3,
        np=np,
    )
    assert bank["embedding_count"] == 2
    assert bank["cluster_count"] == 1
    assert bank["scale_classes"] == ["wide", "wide"]


def test_stable_trajectory_selection_surfaces_fragmented_long_candidate() -> None:
    policy = {
        "minimum_retrieval_score": 0.65,
        "minimum_prototype_similarity": 0.60,
        "minimum_plausible_score": 0.45,
        "minimum_plausible_prototype": 0.45,
        "review_candidate_count": 3,
    }
    rows = [
        _candidate_row(
            "short_high",
            900,
            904,
            5,
            5,
            1.0,
            0.67,
            0.73,
            _unit(1.0, 0.0, 0.0),
        ),
        _candidate_row(
            "continuation_a",
            918,
            952,
            29,
            27,
            0.93,
            0.527,
            0.574,
            _unit(0.0, 1.0, 0.0),
        ),
        _candidate_row(
            "continuation_b",
            952,
            976,
            25,
            24,
            0.96,
            0.501,
            0.533,
            _unit(0.02, 0.999, 0.0),
        ),
        _candidate_row(
            "continuation_c",
            971,
            1062,
            77,
            36,
            0.47,
            0.524,
            0.562,
            _unit(0.01, 0.9999, 0.0),
        ),
        _candidate_row(
            "other_long",
            1070,
            1100,
            30,
            20,
            0.70,
            0.60,
            0.66,
            _unit(0.0, 0.0, 1.0),
        ),
    ]
    assignments = {
        "short_high": _assignments(
            "short_high", 900, 904, x_start=100.0, x_step=0.0
        ),
        "continuation_a": _assignments(
            "continuation_a", 918, 952, x_start=500.0, x_step=2.0
        ),
        "continuation_b": _assignments(
            "continuation_b", 952, 976, x_start=568.0, x_step=2.0
        ),
        "continuation_c": _assignments(
            "continuation_c", 971, 1062, x_start=606.0, x_step=2.0
        ),
        "other_long": _assignments(
            "other_long", 1070, 1100, x_start=1500.0, x_step=0.0
        ),
    }
    selected, rescue, _, groups, _ = ADAPTER._phase4b_select_assisted_review_candidates(
        rows,
        policy=policy,
        assignments_by_candidate=assignments,
        np=np,
    )
    selected_ids = {str(row["candidate_id"]) for row in selected}
    assert "short_high" in selected_ids
    assert selected_ids & {"continuation_a", "continuation_b", "continuation_c"}
    continuation_group = next(
        group
        for group in groups
        if set(group["member_candidate_ids"])
        == {"continuation_a", "continuation_b", "continuation_c"}
    )
    assert continuation_group["member_count"] == 3
    assert continuation_group["source_frame_span"] == 145
    assert any(
        row.get("rescue_review_required") is True
        and set(row.get("continuation_group_member_ids") or [])
        == {"continuation_a", "continuation_b", "continuation_c"}
        for row in rescue
    )


def test_r8_terminal_attempts_reopen_without_touching_user_decisions() -> None:
    state = {
        "status": "COMPLETE_WITH_SAFE_BLOCK",
        "decision": ADAPTER.PHASE4B_ALL_EXHAUSTED_DECISION,
        "pending_action": None,
        "runtime": {
            "phase4b_policy": ADAPTER.PHASE4B_PREVIOUS_POLICY_R8,
            "automatic_target_confirmation": False,
        },
        "shots": [
            {
                "shot_id": "shot_0005",
                "shot_index": 5,
                "status": ADAPTER.PHASE4B_NONE_OF_THESE_SHOT_STATUS,
            },
            {
                "shot_id": "shot_0008",
                "shot_index": 8,
                "status": ADAPTER.PHASE4B_UNREVIEWABLE_GROUP_OCCLUSION_STATUS,
            },
            {
                "shot_id": "shot_0010",
                "shot_index": 10,
                "status": ADAPTER.PHASE4B_SEARCH_EXHAUSTED_STATUS,
            },
        ],
        "shot_search_results": {
            "shot_0005": {
                "phase4b_policy": "OLDER_USER_REVIEW_POLICY",
                "review_decision": "NONE_OF_THESE",
            },
            "shot_0008": {
                "phase4b_policy": ADAPTER.PHASE4B_PREVIOUS_POLICY_R8,
                "identity_observability_policy": (
                    ADAPTER.PHASE4B_IDENTITY_OBSERVABILITY_POLICY
                ),
            },
            "shot_0010": {
                "phase4b_policy": ADAPTER.PHASE4B_PREVIOUS_POLICY_R8,
                "identity_observability_policy": (
                    ADAPTER.PHASE4B_IDENTITY_OBSERVABILITY_POLICY
                ),
            },
        },
        "ambiguities": [],
    }
    reopened = ADAPTER._phase4b_reopen_stale_terminal_results_for_rescoring(
        state
    )
    assert reopened == ["shot_0008", "shot_0010"]
    assert state["shots"][0]["status"] == ADAPTER.PHASE4B_NONE_OF_THESE_SHOT_STATUS
    assert state["shots"][1]["status"] == "SEARCHING_MEMORY_READY"
    assert state["shots"][2]["status"] == "SEARCHING_MEMORY_READY"
    assert "shot_0005" in state["shot_search_results"]
    assert "shot_0008" not in state["shot_search_results"]
    assert state["runtime"]["phase4b_policy"] == ADAPTER.PHASE4B_POLICY


def test_r10_source_contains_no_sample_specific_candidate_ids() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "shot_0010_track_0047" not in source
    assert "shot_0010_track_0059" not in source
    assert "shot_0010_track_0052" not in source
    assert "shot_0010_track_0022" not in source
    assert "frame 180" not in source.lower()
    assert "jersey_10" not in source.lower()
    assert "goal_replay" not in source.lower()


def test_attempt_report_embeds_review_candidate_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    target_embeddings = tmp_path / "target_embeddings.npy"
    target_prototype = tmp_path / "target_prototype.npy"
    checkpoint.write_bytes(b"checkpoint")
    np.save(target_embeddings, np.stack([_unit(1.0, 0.0, 0.0)]), allow_pickle=False)
    np.save(target_prototype, _unit(1.0, 0.0, 0.0), allow_pickle=False)
    candidate = {"candidate_id": "shot_test_track_0001"}
    report = ADAPTER._phase4b_build_attempt_report(
        shot={
            "shot_id": "shot_test",
            "shot_index": 1,
            "start_frame": 10,
            "end_frame_inclusive": 20,
        },
        memory={"memory_revision_id": "ecmem_test"},
        memory_path=tmp_path / "memory.json",
        memory_sha="a" * 64,
        all_refs=[],
        scoring_refs=[],
        generation=2,
        detector_contract={},
        runtime={
            "negative_memory_available": False,
            "negative_memory_policy": "PROVENANCE_ROLE_USER_HARD_DETECTOR_SOFT_IDENTITY_ROBUST_R4",
            "raw_detection_count": 0,
            "analysis_raw_detection_count": 0,
            "detection_count": 0,
            "model_contract": {},
            "checkpoint": checkpoint,
            "target_embeddings_path": target_embeddings,
            "target_embeddings_sha256": _sha(target_embeddings),
            "target_prototype_path": target_prototype,
            "target_prototype_sha256": _sha(target_prototype),
        },
        result={
            "candidate_count": 1,
            "review_candidates": [candidate],
            "candidate_role_filter": {},
            "gate": {},
        },
        runtime_seconds=0.1,
    )
    assert report["review_candidate_count"] == 1
    assert report["review_candidates"] == [candidate]
    assert report["selected_review_candidate_ids"] == [
        "shot_test_track_0001"
    ]


def test_verifier_backfills_early_r9_attempt_from_ranked_artifact() -> None:
    errors: list[str] = []
    rows, source = VERIFIER._resolve_review_rows(
        {
            "review_candidate_count": 1,
            "safe_gate": {
                "selected_review_candidate_ids": ["shot_0009_track_0002"]
            },
        },
        {
            "reviewable_candidates": [
                {"candidate_id": "shot_0009_track_0002"}
            ]
        },
        errors,
    )
    assert source == "RANKED_CANDIDATES_COMPAT_BACKFILL"
    assert [row["candidate_id"] for row in rows] == [
        "shot_0009_track_0002"
    ]
    assert errors == []


def test_verifier_rejects_backfill_candidate_id_mismatch() -> None:
    errors: list[str] = []
    VERIFIER._resolve_review_rows(
        {
            "review_candidate_count": 1,
            "safe_gate": {"selected_review_candidate_ids": ["expected"]},
        },
        {"reviewable_candidates": [{"candidate_id": "different"}]},
        errors,
    )
    assert "DECLARED_REVIEW_SET_MISMATCH" in errors
