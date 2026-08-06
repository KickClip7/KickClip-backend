from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
VERIFIER_PATH = PROJECT_ROOT / "scripts" / "verify_phase4b_identity_purity_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("kickclip_phase4b_r10_adapter_test", ADAPTER_PATH)
VERIFIER = _load("kickclip_phase4b_r10_verifier_test", VERIFIER_PATH)


def _detection(frame: int, *, x: float = 100.0, h: float = 100.0):
    return SimpleNamespace(
        frame=frame,
        detection_id=f"d{frame:04d}",
        bbox=[x, 100.0, x + 50.0, 100.0 + h],
        confidence=0.9,
        class_id=0,
        class_name="player",
    )


def _unit(*values: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32)
    return row / np.linalg.norm(row)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_long_coherent_motion_tracklet_is_not_force_split_by_time() -> None:
    detections = [_detection(frame, x=100.0 + frame) for frame in range(75)]
    embeddings = {
        row.detection_id: _unit(1.0, 0.0, 0.0) for row in detections
    }
    plan = ADAPTER._phase4b_plan_identity_purity_segments(
        detections,
        detections,
        embeddings,
        fps=25.0,
        np=np,
    )
    assert plan["split_parent"] is False
    assert len(plan["segments"]) == 1
    assert plan["segments"][0]["passed"] is True
    assert plan["maximum_review_window_frames"] == 0
    assert plan["forced_time_window_split"] is False
    assert "MAX_REVIEW_WINDOW" not in plan["split_reasons"]


def test_sustained_appearance_change_creates_identity_boundary() -> None:
    detections = [_detection(frame, x=100.0 + frame) for frame in range(20)]
    embeddings = {}
    for row in detections[:10]:
        embeddings[row.detection_id] = _unit(1.0, 0.0, 0.0)
    for row in detections[10:]:
        embeddings[row.detection_id] = _unit(0.0, 1.0, 0.0)
    plan = ADAPTER._phase4b_plan_identity_purity_segments(
        detections,
        detections,
        embeddings,
        fps=25.0,
        np=np,
    )
    assert plan["split_parent"] is True
    assert "APPEARANCE_DISCONTINUITY" in plan["split_reasons"]
    boundary = next(
        row
        for row in plan["change_points"]
        if "APPEARANCE_DISCONTINUITY" in row["reasons"]
    )
    assert boundary["split_before_frame"] == 10


def test_short_coherent_tracklet_remains_single_segment() -> None:
    detections = [_detection(frame) for frame in range(12)]
    embeddings = {
        row.detection_id: _unit(1.0, 0.05, 0.0) for row in detections
    }
    plan = ADAPTER._phase4b_plan_identity_purity_segments(
        detections,
        detections,
        embeddings,
        fps=25.0,
        np=np,
    )
    assert plan["split_parent"] is False
    assert len(plan["segments"]) == 1
    assert plan["segments"][0]["passed"] is True


def test_same_parent_segments_cannot_be_remerged_by_continuation() -> None:
    left = {
        "candidate_id": "parent_seg_001",
        "parent_tracklet_id": "parent",
        "start_frame": 100,
        "end_frame_inclusive": 120,
    }
    right = {
        "candidate_id": "parent_seg_002",
        "parent_tracklet_id": "parent",
        "start_frame": 121,
        "end_frame_inclusive": 140,
    }
    evidence = ADAPTER._phase4b_candidate_continuation_evidence(
        left,
        right,
        assignments_by_candidate={},
        np=np,
    )
    assert evidence["linked"] is False
    assert evidence["reason"] == "IDENTITY_PURITY_SEGMENT_BOUNDARY"
    assert evidence["same_parent_tracklet"] is True


def test_r11_source_has_no_sample_specific_identity_rules() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8").lower()
    assert "shot_0010_track_0022" not in source
    assert "frame 180" not in source
    assert "jersey_10" not in source
    assert "white_uniform" not in source
    assert "goal_replay" not in source


def test_identity_purity_verifier_accepts_grounded_synthetic_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    ranked_path = job / "ranked_candidates.json"
    contact_path = job / "ranked_candidates.jpg"
    rejection_path = job / "identity_purity_rejected.jpg"
    ranked_path.write_text(json.dumps({"reviewable_candidates": []}), encoding="utf-8")
    contact_path.write_bytes(b"contact")
    rejection_path.write_bytes(b"rejected")

    parent = {
        "parent_tracklet_id": "parent_001",
        "identity_purity_policy": VERIFIER.PURITY_POLICY,
        "split_parent": True,
        "segment_count": 2,
        "target_memory_used_for_segmentation": False,
        "segments": [
            {
                "candidate_id": "parent_001_seg_001",
                "policy": VERIFIER.PURITY_POLICY,
                "passed": True,
                "segment_start_frame": 10,
                "segment_end_frame_inclusive": 20,
                "target_memory_used_for_segmentation": False,
                "automatic_target_confirmation": False,
            },
            {
                "candidate_id": "parent_001_seg_002",
                "policy": VERIFIER.PURITY_POLICY,
                "passed": False,
                "segment_start_frame": 21,
                "segment_end_frame_inclusive": 30,
                "target_memory_used_for_segmentation": False,
                "automatic_target_confirmation": False,
            },
        ],
    }
    rejected = {
        "candidate_id": "parent_001_seg_002",
        "identity_purity": parent["segments"][1],
    }
    attempt = {
        "policy": VERIFIER.PHASE4B_POLICY,
        "shot_id": "shot_test",
        "identity_purity_policy": VERIFIER.PURITY_POLICY,
        "assisted_review_selection_policy": VERIFIER.SELECTION_POLICY,
        "automatic_target_confirmation": False,
        "candidate_link_created": False,
        "parent_tracklet_count_before_purity_segmentation": 1,
        "split_parent_tracklet_count": 1,
        "identity_segment_candidate_count": 1,
        "rejected_identity_purity_segment_count": 1,
        "parent_tracklet_summaries": [parent],
        "rejected_identity_purity_segments": [rejected],
        "review_candidate_count": 0,
        "review_candidates": [],
        "continuation_groups": [],
        "ranked_candidates_path": str(ranked_path),
        "ranked_candidates_sha256": _sha(ranked_path),
        "contact_sheet_path": str(contact_path),
        "contact_sheet_sha256": _sha(contact_path),
        "identity_purity_rejection_sheet_path": str(rejection_path),
        "identity_purity_rejection_sheet_sha256": _sha(rejection_path),
    }
    (job / "pipeline_state.json").write_text(
        json.dumps({"pending_action": None}), encoding="utf-8"
    )
    (job / "phase4b_first_cross_shot_report.json").write_text(
        json.dumps({"policy": VERIFIER.PHASE4B_POLICY, "attempts": [attempt]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--job-root",
            str(job),
            "--shot-id",
            "shot_test",
            "--require-split-parent",
        ],
    )
    assert VERIFIER.main() == 0
