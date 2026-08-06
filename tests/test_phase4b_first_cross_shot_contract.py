from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)

if not MODULE_PATH.is_file():
    raise FileNotFoundError(
        f"Phase 4-B adapter module was not found at the expected backend path: {MODULE_PATH}"
    )
SPEC = importlib.util.spec_from_file_location("kickclip_phase4b_adapter_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)



@pytest.fixture
def tmp_path(request: pytest.FixtureRequest):
    """Use a project-local writable directory instead of Windows TEMP."""
    base = PROJECT_ROOT / "storage" / "pytest_phase4b_tmp"
    base.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in request.node.name
    )
    path = base / f"{safe_name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)

    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase4b_scoring_references_use_only_explicitly_eligible_rows(tmp_path: Path) -> None:
    eligible = []
    pending = []
    for index in range(3):
        path = tmp_path / f"eligible_{index}.jpg"
        path.write_bytes(f"eligible-{index}".encode())
        eligible.append(
            {
                "path": str(path),
                "sha256": _sha(path),
                "scoring_eligible": True,
            }
        )
    path = tmp_path / "pending.jpg"
    path.write_bytes(b"pending")
    pending.append(
        {
            "path": str(path),
            "sha256": _sha(path),
            "scoring_eligible": False,
        }
    )
    scoring, all_rows = MODULE._phase4b_scoring_references(
        {"references": [*eligible, *pending]}
    )
    assert len(scoring) == 3
    assert len(all_rows) == 4
    assert all(row["scoring_eligible"] is True for row in scoring)


def test_phase4b_gate_never_auto_links_without_negative_memory() -> None:
    ranked = [
        {
            "candidate_id": "shot_0004_track_0001",
            "retrieval_score": 0.99,
            "prototype_target_similarity": 0.99,
        },
        {
            "candidate_id": "shot_0004_track_0002",
            "retrieval_score": 0.70,
            "prototype_target_similarity": 0.80,
        },
    ]
    policy = {
        "minimum_retrieval_score": 0.65,
        "minimum_prototype_similarity": 0.60,
        "minimum_top1_top2_gap": 0.08,
        "minimum_plausible_score": 0.45,
        "minimum_plausible_prototype": 0.45,
    }
    gate = MODULE._phase4b_gate_candidate_rows(
        ranked,
        policy=policy,
        negative_memory_available=False,
    )
    assert gate["operational_state"] == "AMBIGUOUS"
    assert gate["automatically_proposed_candidate"] is None
    assert gate["all_auto_gates_passed"] is False
    assert gate["automatic_target_confirmation"] is False


def _state() -> dict:
    return {
        "status": "COMPLETE_WITH_SAFE_BLOCK",
        "decision": "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING",
        "pending_action": None,
        "shots": [
            {
                "shot_id": "shot_0004",
                "shot_index": 4,
                "start_frame": 348,
                "end_frame_inclusive": 398,
                "status": "SEARCHING_MEMORY_READY",
            }
        ],
        "ambiguities": [],
        "runtime": {},
    }


def _memory() -> dict:
    return {
        "memory_revision_id": "ecmem_initial_test",
        "reference_count": 11,
        "scoring_reference_count": 6,
        "references": [{}] * 11,
    }


def test_phase4b_result_pauses_with_nonempty_dynamic_ambiguity(tmp_path: Path) -> None:
    state = _state()
    candidate = {
        "candidate_id": "shot_0004_track_0001",
        "manifest_path": str(tmp_path / "candidate_manifest.json"),
        "full_frame_context_path": str(tmp_path / "frame.jpg"),
        "shot_clip_path": str(tmp_path / "shot.mp4"),
        "reference_gallery_path": str(tmp_path / "gallery.jpg"),
        "score_evidence": {
            "phase4b_policy": MODULE.PHASE4B_POLICY,
            "memory_revision_id": "ecmem_initial_test",
            "memory_revision_sha256": "a" * 64,
            "candidate_scoring_generation": 2,
            "backend_memory_used_by_phase4b_scoring": True,
            "automatic_target_confirmation": False,
        },
    }
    result = {
        "candidate_count": 1,
        "ranked_candidates": [candidate],
        "review_candidates": [candidate],
        "gate": {
            "operational_state": "AMBIGUOUS",
            "automatic_target_confirmation": False,
        },
        "assignments_path": str(tmp_path / "assignments.csv"),
        "contact_sheet_path": str(tmp_path / "sheet.jpg"),
    }
    updated = MODULE._phase4b_apply_result_to_state(
        state=state,
        shot=state["shots"][0],
        result=result,
        memory=_memory(),
        memory_path=tmp_path / "memory.json",
        memory_sha="a" * 64,
        generation=2,
        report_path=tmp_path / "report.json",
    )
    assert updated["status"] == "NEEDS_CONFIRMATION"
    assert updated["pending_action"]["type"] == "CROSS_SHOT_CONFIRMATION"
    assert updated["pending_action"]["recommended_candidate"] is None
    assert len(updated["ambiguities"]) == 1
    assert len(updated["ambiguities"][0]["review_candidates"]) == 1
    assert updated["runtime"]["backend_memory_used_by_phase4b_scoring"] is True
    assert updated["runtime"]["automatic_target_confirmation"] is False


def test_phase4b_empty_result_safe_blocks_without_empty_ambiguity(tmp_path: Path) -> None:
    state = _state()
    updated = MODULE._phase4b_apply_result_to_state(
        state=state,
        shot=state["shots"][0],
        result={
            "candidate_count": 0,
            "ranked_candidates": [],
            "review_candidates": [],
            "gate": {"operational_state": "SAFE_REJECTED_SEARCHING"},
        },
        memory=_memory(),
        memory_path=tmp_path / "memory.json",
        memory_sha="a" * 64,
        generation=2,
        report_path=tmp_path / "report.json",
    )
    assert updated["status"] == "COMPLETE_WITH_SAFE_BLOCK"
    assert updated["decision"] == MODULE.PHASE4B_ALL_EXHAUSTED_DECISION
    assert updated["pending_action"] is None
    assert updated["ambiguities"] == []


def test_memory_review_approval_continues_into_phase4b(tmp_path: Path) -> None:
    memory = tmp_path / "memory.json"
    memory.write_text("{}", encoding="utf-8")
    state = {
        "status": "NEEDS_CONFIRMATION",
        "decision": "PAUSE_FOR_PHASE4A_INITIAL_TARGET_MEMORY_REVIEW",
        "pending_action": {"type": "MEMORY_REVIEW"},
        "shots": [
            {"shot_id": "shot_0004", "status": "SEARCHING_MEMORY_REVIEW_PENDING"}
        ],
        "runtime": {
            "phase4a_initial_memory_build_complete": True,
            "phase4a_initial_memory_review_status": "PENDING",
            "memory_revision_path": str(memory),
            "memory_revision_sha256": _sha(memory),
        },
    }
    (tmp_path / "pipeline_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    args = argparse.Namespace(
        resume=True,
        approve_review="MEMORY",
        reject_review=None,
        target_memory_revision=memory,
        target_memory_sha256=_sha(memory),
    )
    result = MODULE._handle_phase4a_memory_review_resume(
        args=args,
        output_dir=tmp_path,
    )
    assert result == MODULE.PHASE4B_APPROVED_CONTINUE
    updated = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert updated["decision"] == "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING"
    assert updated["runtime"]["phase4b_cross_shot_scoring_authorized"] is True
    assert updated["shots"][0]["status"] == "SEARCHING_MEMORY_READY"


def test_phase4b_runner_writes_report_and_pending_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = []
    for index in range(3):
        path = tmp_path / f"memory_{index}.jpg"
        path.write_bytes(f"memory-{index}".encode())
        refs.append(
            {
                "path": str(path),
                "sha256": _sha(path),
                "scoring_eligible": True,
            }
        )
    memory_path = tmp_path / "memory.json"
    memory_doc = {
        "schema_version": MODULE.PHASE4A_MEMORY_SCHEMA,
        "memory_revision_id": "ecmem_initial_test",
        "references": refs,
        "reference_count": 3,
        "scoring_reference_count": 3,
        "pending_review_reference_count": 0,
        "automatic_target_confirmation": False,
    }
    memory_path.write_text(json.dumps(memory_doc), encoding="utf-8")
    state = {
        **_state(),
        "runtime": {
            "phase4a_initial_memory_review_status": "PASS",
            "phase4b_cross_shot_scoring_authorized": True,
            "candidate_scoring_generation": 1,
        },
    }
    (tmp_path / "pipeline_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    (tmp_path / "pipeline_summary.json").write_text("{}", encoding="utf-8")
    clip = tmp_path / "clip.mp4"
    detections = tmp_path / "detections.csv"
    clip.write_bytes(b"clip")
    detections.write_text("frame_index\n", encoding="utf-8")

    monkeypatch.setattr(
        MODULE,
        "_phase4b_prepare_detector_source",
        lambda **kwargs: (
            clip,
            {"frame_count": 250, "width": 1920, "height": 1080},
            detections,
            {"detections_path": str(detections), "detections_sha256": _sha(detections)},
        ),
    )
    checkpoint = tmp_path / "sports.pth"
    checkpoint.write_bytes(b"checkpoint")
    target_embeddings = tmp_path / "target.npy"
    target_prototype = tmp_path / "prototype.npy"
    target_embeddings.write_bytes(b"embeddings")
    target_prototype.write_bytes(b"prototype")
    monkeypatch.setattr(
        MODULE,
        "_phase4b_load_runtime",
        lambda **kwargs: {
            "model_contract": {"name": "fake"},
            "checkpoint": checkpoint,
            "target_embeddings_path": target_embeddings,
            "target_embeddings_sha256": _sha(target_embeddings),
            "target_prototype_path": target_prototype,
            "target_prototype_sha256": _sha(target_prototype),
            "negative_memory_available": False,
            "negative_role_memory": {
                "policy": MODULE.PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
                "negative_memory_available": False,
                "selected_count": 0,
            },
            "raw_detection_count": 0,
            "analysis_raw_detection_count": 0,
            "detection_count": 0,
        },
    )
    candidate = {
        "candidate_id": "shot_0004_track_0001",
        "score_evidence": {
            "phase4b_policy": MODULE.PHASE4B_POLICY,
            "memory_revision_id": "ecmem_initial_test",
            "memory_revision_sha256": _sha(memory_path),
            "candidate_scoring_generation": 2,
            "backend_memory_used_by_phase4b_scoring": True,
            "automatic_target_confirmation": False,
        },
    }
    monkeypatch.setattr(
        MODULE,
        "_phase4b_process_candidates",
        lambda **kwargs: {
            "candidate_count": 1,
            "ranked_candidates": [candidate],
            "review_candidates": [candidate],
            "gate": {"operational_state": "AMBIGUOUS"},
            "assignments_path": str(tmp_path / "assignments.csv"),
            "contact_sheet_path": str(tmp_path / "sheet.jpg"),
        },
    )
    args = argparse.Namespace(
        candidate_scoring_generation=1,
        device="cpu",
        overwrite=False,
    )
    report = MODULE.run_phase4b_first_cross_shot(
        root=tmp_path,
        output_dir=tmp_path,
        args=args,
        memory_path=memory_path,
        memory_sha=_sha(memory_path),
    )
    assert report["status"] == "PASS"
    assert report["candidate_scoring_generation"] == 2
    assert report["backend_memory_used_by_phase4b_scoring"] is True
    updated = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert updated["status"] == "NEEDS_CONFIRMATION"
    assert updated["pending_action"]["type"] == "CROSS_SHOT_CONFIRMATION"


def _detection(
    class_id: int,
    class_name: str,
    detection_id: str,
    *,
    frame: int = 0,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 200.0),
    confidence: float = 0.9,
):
    return SimpleNamespace(
        class_id=class_id,
        class_name=class_name,
        detection_id=detection_id,
        frame=frame,
        bbox=list(bbox),
        confidence=confidence,
    )


def test_phase4b_candidate_role_filter_splits_candidates_and_role_negatives() -> None:
    by_frame = {
        0: [
            _detection(0, "player", "p0"),
            _detection(1, "goalkeeper", "g0"),
            _detection(2, "referee", "r0"),
            _detection(3, "staff", "s0"),
            _detection(4, "ball", "b0"),
        ],
        5: [_detection(3, "staff", "tail", frame=5)],
    }
    filtered, negatives, metrics = MODULE._phase4b_filter_candidate_detections(
        by_frame,
        end_frame_inclusive=0,
    )
    assert [row.detection_id for row in filtered[0]] == ["p0", "g0"]
    assert [row.detection_id for row in negatives[0]] == ["r0", "s0"]
    assert 5 not in negatives
    assert metrics["policy_version"] == MODULE.PHASE4B_CANDIDATE_ROLE_FILTER_POLICY
    assert metrics["negative_role_memory_policy"] == MODULE.PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY
    assert metrics["raw_detection_count"] == 5
    assert metrics["eligible_candidate_detection_count"] == 2
    assert metrics["negative_role_detection_count"] == 2
    assert metrics["excluded_non_target_role_count"] == 3
    assert metrics["excluded_role_counts"] == {
        "ball": 1,
        "referee": 1,
        "staff": 1,
    }
    assert metrics["analysis_tail_excluded_from_role_counts"] is True


def test_phase4b_candidate_role_filter_rejects_class_mapping_mismatch() -> None:
    with pytest.raises(RuntimeError, match="class mapping mismatch"):
        MODULE._phase4b_filter_candidate_detections(
            {0: [_detection(3, "player", "bad")]}
        )


def test_phase4b_temporal_role_confusion_rejects_label_flipping_staff() -> None:
    candidate = [
        _detection(0, "player", "p0", frame=0, bbox=(10, 10, 110, 210)),
        _detection(0, "player", "p1", frame=1, bbox=(12, 10, 112, 210)),
        _detection(0, "player", "p2", frame=2, bbox=(14, 10, 114, 210)),
        _detection(0, "player", "p3", frame=3, bbox=(16, 10, 116, 210)),
    ]
    negatives = {
        1: [_detection(3, "staff", "s1", frame=1, bbox=(11, 10, 111, 210))],
        3: [_detection(3, "staff", "s3", frame=3, bbox=(15, 10, 115, 210))],
    }
    evidence = MODULE._phase4b_role_confusion_evidence(candidate, negatives)
    assert evidence["passed"] is False
    assert evidence["confused_observation_count"] >= 2
    assert evidence["rejection_reason"] == "TEMPORAL_STAFF_REFEREE_ROLE_CONFLICT"


def test_phase4b_negative_role_margin_gate_rejects_staff_like_candidate() -> None:
    gate = MODULE._phase4b_candidate_negative_review_gate(
        {
            "crop_margin_median": 0.01,
            "positive_margin_support_ratio": 0.50,
            "prototype_negative_margin": -0.02,
        },
        negative_memory_available=True,
    )
    assert gate["passed"] is False
    assert set(gate["rejection_reasons"]) == {
        "crop_margin_median",
        "positive_margin_support_ratio",
        "prototype_negative_margin",
    }


def test_phase4b_negative_role_margin_gate_allows_target_dominant_candidate() -> None:
    gate = MODULE._phase4b_candidate_negative_review_gate(
        {
            "crop_margin_median": 0.12,
            "positive_margin_support_ratio": 0.90,
            "prototype_negative_margin": 0.08,
        },
        negative_memory_available=True,
    )
    assert gate["passed"] is True


def test_phase4b_duplicate_track_evidence_detects_overlapping_fragments() -> None:
    left = [
        {
            "analysis_local_frame_index": frame,
            "x1": 10,
            "y1": 10,
            "x2": 110,
            "y2": 210,
        }
        for frame in range(6)
    ]
    right = [
        {
            "analysis_local_frame_index": frame,
            "x1": 12,
            "y1": 12,
            "x2": 112,
            "y2": 212,
        }
        for frame in range(1, 6)
    ]
    evidence = MODULE._phase4b_track_duplicate_evidence(left, right)
    assert evidence["duplicate"] is True
    assert evidence["common_frame_count"] == 5


def test_phase4b_old_policy_pending_review_is_invalidated_for_rescoring() -> None:
    state = {
        "status": "NEEDS_CONFIRMATION",
        "decision": "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION",
        "pending_action": {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": "ambiguity_old",
            "shot_id": "shot_0004",
        },
        "shots": [
            {"shot_id": "shot_0004", "status": "AMBIGUOUS_REVIEW_REQUIRED"}
        ],
        "ambiguities": [
            {"ambiguity_id": "ambiguity_old", "shot_id": "shot_0004"}
        ],
        "runtime": {
            "phase4b_policy": "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R1",
            "automatic_target_confirmation": False,
        },
    }
    assert MODULE._phase4b_invalidate_stale_candidate_review(state) is True
    assert state["pending_action"] is None
    assert state["ambiguities"] == []
    assert state["shots"][0]["status"] == "SEARCHING_MEMORY_READY"
    assert state["decision"] == "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING"
    assert state["runtime"]["phase4b_policy"] == MODULE.PHASE4B_POLICY



def _multi_shot_state() -> dict:
    return {
        "status": "COMPLETE_WITH_SAFE_BLOCK",
        "decision": "SAFE_BLOCK_PHASE4B_NO_REVIEWABLE_CANDIDATES",
        "pending_action": None,
        "shots": [
            {
                "shot_id": "shot_0004",
                "shot_index": 4,
                "start_frame": 348,
                "end_frame_inclusive": 398,
                "status": "SAFE_REJECTED_SEARCHING",
            },
            {
                "shot_id": "shot_0005",
                "shot_index": 5,
                "start_frame": 399,
                "end_frame_inclusive": 467,
                "status": "SEARCHING_MEMORY_READY",
            },
        ],
        "shot_search_results": {
            "shot_0004": {
                "status": "SAFE_REJECTED_SEARCHING",
                "candidate_count": 0,
            }
        },
        "ambiguities": [],
        "runtime": {
            "phase4a_initial_memory_review_status": "PASS",
            "phase4b_cross_shot_scoring_authorized": True,
            "candidate_scoring_generation": 4,
            "automatic_target_confirmation": False,
        },
    }


def test_phase4b_migrates_prior_single_shot_safe_block_to_continuation() -> None:
    state = _multi_shot_state()
    assert MODULE._phase4b_migrate_prior_no_candidate_results(state) is True
    assert state["shots"][0]["status"] == MODULE.PHASE4B_SEARCH_EXHAUSTED_STATUS
    assert state["status"] == "READY_FOR_PHASE4B_CONTINUATION"
    assert state["decision"] == MODULE.PHASE4B_CONTINUE_DECISION
    assert [row["shot_id"] for row in MODULE._phase4b_ready_shots(state)] == ["shot_0005"]


def test_phase4b_empty_shot_continues_when_another_reviewed_shot_remains(tmp_path: Path) -> None:
    state = _multi_shot_state()
    MODULE._phase4b_migrate_prior_no_candidate_results(state)
    shot = state["shots"][1]
    state["shots"].append(
        {
            "shot_id": "shot_0006",
            "shot_index": 6,
            "start_frame": 468,
            "end_frame_inclusive": 555,
            "status": "SEARCHING_MEMORY_READY",
        }
    )
    updated = MODULE._phase4b_apply_result_to_state(
        state=state,
        shot=shot,
        result={
            "candidate_count": 0,
            "ranked_candidates": [],
            "review_candidates": [],
            "gate": {"operational_state": "SAFE_REJECTED_SEARCHING"},
        },
        memory=_memory(),
        memory_path=tmp_path / "memory.json",
        memory_sha="a" * 64,
        generation=5,
        report_path=tmp_path / "attempt.json",
        remaining_ready_shot_ids=["shot_0006"],
    )
    assert updated["status"] == "RUNNING"
    assert updated["decision"] == MODULE.PHASE4B_CONTINUE_DECISION
    assert updated["pending_action"] is None
    assert shot["status"] == MODULE.PHASE4B_SEARCH_EXHAUSTED_STATUS
    assert updated["runtime"]["phase4b_next_shot_id"] == "shot_0006"


def test_phase4b_empty_final_shot_terminates_only_after_all_remaining_exhausted(tmp_path: Path) -> None:
    state = _state()
    shot = state["shots"][0]
    updated = MODULE._phase4b_apply_result_to_state(
        state=state,
        shot=shot,
        result={
            "candidate_count": 0,
            "ranked_candidates": [],
            "review_candidates": [],
            "gate": {"operational_state": "SAFE_REJECTED_SEARCHING"},
        },
        memory=_memory(),
        memory_path=tmp_path / "memory.json",
        memory_sha="a" * 64,
        generation=5,
        report_path=tmp_path / "attempt.json",
        remaining_ready_shot_ids=[],
    )
    assert updated["status"] == "COMPLETE_WITH_SAFE_BLOCK"
    assert updated["decision"] == MODULE.PHASE4B_ALL_EXHAUSTED_DECISION
    assert shot["status"] == MODULE.PHASE4B_SEARCH_EXHAUSTED_STATUS


def test_phase4b_runner_skips_exhausted_shot_and_pauses_on_next_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = []
    for index in range(3):
        path = tmp_path / f"memory_multi_{index}.jpg"
        path.write_bytes(f"memory-{index}".encode())
        refs.append({"path": str(path), "sha256": _sha(path), "scoring_eligible": True})
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": MODULE.PHASE4A_MEMORY_SCHEMA,
                "memory_revision_id": "ecmem_initial_test",
                "references": refs,
                "reference_count": 3,
                "scoring_reference_count": 3,
                "automatic_target_confirmation": False,
            }
        ),
        encoding="utf-8",
    )
    state = _multi_shot_state()
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "pipeline_summary.json").write_text("{}", encoding="utf-8")

    clip = tmp_path / "clip.mp4"
    detections = tmp_path / "detections.csv"
    clip.write_bytes(b"clip")
    detections.write_text("frame_index\n", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_phase4b_prepare_detector_source",
        lambda **kwargs: (
            clip,
            {"frame_count": 250, "width": 1920, "height": 1080},
            detections,
            {
                "analysis_clip_path": str(clip),
                "analysis_clip_sha256": _sha(clip),
                "analysis_source_start_frame": int(kwargs["shot"]["start_frame"]),
                "analysis_source_end_frame_inclusive": int(kwargs["shot"]["start_frame"]) + 249,
                "analysis_clip_frame_count": 250,
                "detections_path": str(detections),
                "detections_sha256": _sha(detections),
            },
        ),
    )
    checkpoint = tmp_path / "sports.pth"
    checkpoint.write_bytes(b"checkpoint")
    target_embeddings = tmp_path / "target.npy"
    target_prototype = tmp_path / "prototype.npy"
    target_embeddings.write_bytes(b"embeddings")
    target_prototype.write_bytes(b"prototype")
    monkeypatch.setattr(
        MODULE,
        "_phase4b_load_runtime",
        lambda **kwargs: {
            "model_contract": {"name": "fake"},
            "checkpoint": checkpoint,
            "target_embeddings_path": target_embeddings,
            "target_embeddings_sha256": _sha(target_embeddings),
            "target_prototype_path": target_prototype,
            "target_prototype_sha256": _sha(target_prototype),
            "negative_memory_available": False,
            "negative_role_memory": {
                "policy": MODULE.PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
                "negative_memory_available": False,
                "selected_count": 0,
            },
            "raw_detection_count": 0,
            "analysis_raw_detection_count": 0,
            "detection_count": 0,
        },
    )
    candidate = {
        "candidate_id": "shot_0005_track_0001",
        "score_evidence": {
            "phase4b_policy": MODULE.PHASE4B_POLICY,
            "memory_revision_id": "ecmem_initial_test",
            "memory_revision_sha256": _sha(memory_path),
            "candidate_scoring_generation": 5,
            "backend_memory_used_by_phase4b_scoring": True,
            "automatic_target_confirmation": False,
        },
    }
    monkeypatch.setattr(
        MODULE,
        "_phase4b_process_candidates",
        lambda **kwargs: {
            "candidate_count": 1,
            "ranked_candidates": [candidate],
            "review_candidates": [candidate],
            "gate": {"operational_state": "AMBIGUOUS"},
            "assignments_path": str(tmp_path / "assignments.csv"),
            "contact_sheet_path": str(tmp_path / "sheet.jpg"),
        },
    )
    args = argparse.Namespace(candidate_scoring_generation=4, device="cpu", overwrite=False)
    report = MODULE.run_phase4b_first_cross_shot(
        root=tmp_path,
        output_dir=tmp_path,
        args=args,
        memory_path=memory_path,
        memory_sha=_sha(memory_path),
    )
    assert report["multi_shot_continuation"] is True
    assert report["attempted_shot_ids"] == ["shot_0005"]
    assert report["first_reviewable_candidate_shot_id"] == "shot_0005"
    updated = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert updated["shots"][0]["status"] == MODULE.PHASE4B_SEARCH_EXHAUSTED_STATUS
    assert updated["status"] == "NEEDS_CONFIRMATION"
    assert updated["pending_action"]["shot_id"] == "shot_0005"
