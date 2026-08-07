from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "post_confirmation_finalizer.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "phase4c_post_confirmation_finalizer_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _job(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "job"
    candidate_id = "shot_0010_track_0047"
    ambiguity_id = "ambiguity_test"
    decision_id = "decision_test"
    memory_id = "memory_test"

    frames = [
        {
            "frame_index": index,
            "time_seconds": index / 25.0,
            "shot_id": "shot_0010",
            "state": "SEARCHING",
            "bbox_xyxy": None,
            "tracking_confidence": 0.0,
            "identity_confidence": 0.0,
            "identity_source": "NONE",
            "selected_detection_id": None,
            "decision_reason": "SEARCHING",
            "review_required": False,
            "ambiguity_id": None,
        }
        for index in range(10)
    ]
    timeline = {
        "video": {
            "path": str(tmp_path / "missing.mp4"),
            "width": 100,
            "height": 80,
            "fps": 25.0,
            "frame_count": 10,
        },
        "status": "COMPLETE_WITH_SAFE_BLOCK",
        "shots": [
            {
                "shot_id": "shot_0010",
                "start_frame": 0,
                "end_frame_inclusive": 9,
                "status": "AMBIGUOUS_REVIEW_REQUIRED",
            }
        ],
        "frames": frames,
        "ambiguities": [],
        "confirmations": [],
        "provenance": {
            "automatic_target_confirmation": False,
        },
    }
    _write(root / "target_timeline.json", timeline)
    _write(
        root / "pipeline_state.json",
        {
            "test_name": "test",
            "status": "NEEDS_CONFIRMATION",
            "decision": "PAUSE",
            "shots": timeline["shots"],
            "ambiguities": [
                {
                    "ambiguity_id": ambiguity_id,
                    "shot_id": "shot_0010",
                    "status": "PENDING",
                }
            ],
            "confirmations": [],
            "pending_action": {
                "type": "CROSS_SHOT_CONFIRMATION",
                "ambiguity_id": ambiguity_id,
                "shot_id": "shot_0010",
                "candidate_ids": [candidate_id],
                "automatic_target_confirmation": False,
            },
            "runtime": {
                "candidate_scoring_generation": 20,
                "automatic_target_confirmation": False,
            },
        },
    )
    _write(root / "pipeline_summary.json", {"status": "NEEDS_CONFIRMATION"})
    _write(
        root / "review_decisions" / f"{decision_id}.json",
        {
            "decision_id": decision_id,
            "ambiguity_id": ambiguity_id,
            "state": "SAME_PLAYER",
            "candidate_id": candidate_id,
            "reviewer": "user",
            "reviewed_at": "2026-08-06T00:00:00+00:00",
            "note": "verified",
            "automatic_target_confirmation": False,
        },
    )
    manifest = {
        "candidate_id": candidate_id,
        "shot_id": "shot_0010",
        "tracklet_id": candidate_id,
        "parent_tracklet_id": candidate_id,
        "start_frame": 2,
        "end_frame_inclusive": 6,
        "quality": {
            "identity_pure": True,
            "identity_purity_gate_passed": True,
            "identity_observability_gate_passed": True,
        },
        "identity_purity": {"passed": True},
        "identity_observability": {
            "per_observation": [
                {
                    "frame": 20,
                    "detection_id": "d2",
                    "confidence": 0.8,
                    "bbox_xyxy": [10, 10, 20, 40],
                    "clean_for_reid": True,
                    "rejection_reasons": [],
                },
                {
                    "frame": 22,
                    "detection_id": "d4",
                    "confidence": 0.9,
                    "bbox_xyxy": [14, 10, 24, 40],
                    "clean_for_reid": True,
                    "rejection_reasons": [],
                },
                {
                    "frame": 24,
                    "detection_id": "d6",
                    "confidence": 0.85,
                    "bbox_xyxy": [18, 10, 28, 40],
                    "clean_for_reid": False,
                    "rejection_reasons": ["CROWDED_PERSON_OVERLAP"],
                },
            ]
        },
        "automatic_target_confirmation": False,
    }
    _write(
        root
        / "phase4b_cross_shot"
        / "shot_0010"
        / "candidates"
        / candidate_id
        / "candidate_manifest.json",
        manifest,
    )
    memory_path = root / "target_memory_revisions" / f"{memory_id}.json"
    _write(
        memory_path,
        {
            "memory_revision_id": memory_id,
            "source_candidate_id": candidate_id,
            "source_shot_id": "shot_0010",
            "confirmation_decision_id": decision_id,
            "candidate_scoring_generation": 21,
            "reference_count": 3,
            "automatic_target_confirmation": False,
        },
    )
    return root, memory_path, hashlib.sha256(memory_path.read_bytes()).hexdigest()


def test_finalizer_commits_real_observations_without_interpolation(tmp_path):
    module = _module()
    root, memory, sha = _job(tmp_path)
    report = module.finalize_confirmed_candidate(
        output_dir=root,
        candidate_id="shot_0010_track_0047",
        ambiguity_id="ambiguity_test",
        memory_path=memory,
        memory_sha256=sha,
        no_preview=True,
    )
    assert report["status"] == "PASS"
    assert report["real_detector_observation_count"] == 3
    assert report["observation_frame_ids"] == [2, 4, 6]
    assert report["interpolation_used"] is False
    assert report["synthetic_tracking_used"] is False

    timeline = json.loads((root / "target_timeline.json").read_text())
    committed = [
        row for row in timeline["frames"]
        if row.get("candidate_id") == "shot_0010_track_0047"
    ]
    assert [row["frame_index"] for row in committed] == [2, 4, 6]
    assert all(row["bbox_xyxy"] for row in committed)
    assert all(
        row["identity_source"] == "USER_CONFIRMED_SAME_PLAYER"
        for row in committed
    )
    assert timeline["frames"][3]["bbox_xyxy"] is None


def test_finalizer_is_idempotent(tmp_path):
    module = _module()
    root, memory, sha = _job(tmp_path)
    first = module.finalize_confirmed_candidate(
        output_dir=root,
        candidate_id="shot_0010_track_0047",
        ambiguity_id="ambiguity_test",
        memory_path=memory,
        memory_sha256=sha,
        no_preview=True,
    )
    second = module.finalize_confirmed_candidate(
        output_dir=root,
        candidate_id="shot_0010_track_0047",
        ambiguity_id="ambiguity_test",
        memory_path=memory,
        memory_sha256=sha,
        no_preview=True,
    )
    assert first["candidate_id"] == second["candidate_id"]
    timeline = json.loads((root / "target_timeline.json").read_text())
    assert len(timeline["confirmations"]) == 1


def test_finalizer_rejects_wrong_memory_hash(tmp_path):
    module = _module()
    root, memory, _ = _job(tmp_path)
    try:
        module.finalize_confirmed_candidate(
            output_dir=root,
            candidate_id="shot_0010_track_0047",
            ambiguity_id="ambiguity_test",
            memory_path=memory,
            memory_sha256="0" * 64,
            no_preview=True,
        )
    except RuntimeError as exc:
        assert "memory SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("Expected memory hash validation failure.")


def test_finalizer_rejects_automatic_confirmation(tmp_path):
    module = _module()
    root, memory, _ = _job(tmp_path)
    document = json.loads(memory.read_text())
    document["automatic_target_confirmation"] = True
    _write(memory, document)
    sha = hashlib.sha256(memory.read_bytes()).hexdigest()
    try:
        module.finalize_confirmed_candidate(
            output_dir=root,
            candidate_id="shot_0010_track_0047",
            ambiguity_id="ambiguity_test",
            memory_path=memory,
            memory_sha256=sha,
            no_preview=True,
        )
    except RuntimeError as exc:
        assert "automatic_target_confirmation=false" in str(exc)
    else:
        raise AssertionError("Expected automatic confirmation safety failure.")
