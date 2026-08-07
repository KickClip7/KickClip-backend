from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)
spec = importlib.util.spec_from_file_location("kickclip_phase4a_adapter", ADAPTER_PATH)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _workspace() -> Path:
    path = ROOT / "tests" / ".phase4a_contract_tmp"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _fixture_contract(work: Path, *, active: bool = True):
    source = work / "source.bin"
    source.write_bytes(b"source-video-contract")

    references = []
    for index in range(3):
        image_path = work / f"native_{index}.jpg"
        image = np.full((40, 24, 3), 50 + index * 40, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        references.append(
            {
                "frame_id": 10 + index,
                "path": str(image_path),
                "sha256": _sha(image_path),
                "scale": "close-up",
            }
        )

    target_reference = work / "target_reference_set.json"
    _write_json(target_reference, {"references": references})
    target_selection = work / "target_selection.json"
    _write_json(
        target_selection,
        {
            "selection_id": "selection_001",
            "selected_candidate_id": "shot_0003_track_0003",
            "shot_id": "shot_0003",
            "tracklet_id": "track_0003",
        },
    )

    frames = [
        {
            "frame_index": 330,
            "state": "ACTIVE" if active else "OCCLUDED",
            "bbox_xyxy": [10.0, 10.0, 40.0, 70.0],
            "bbox_source": (
                "RFDETR_ASSOCIATED_DETECTION"
                if active
                else "MOTION_PREDICTION_UNCONFIRMED"
            ),
            "selected_detection_id": "f000001_d000" if active else None,
            "tracking_confidence": 0.9,
            "identity_confidence": 0.92,
        },
        {
            "frame_index": 345,
            "state": "ACTIVE" if active else "LOST",
            "bbox_xyxy": [12.0, 9.0, 42.0, 71.0] if active else None,
            "bbox_source": "RFDETR_ASSOCIATED_DETECTION" if active else "NONE",
            "selected_detection_id": "f000002_d000" if active else None,
            "tracking_confidence": 0.91 if active else 0.0,
            "identity_confidence": 0.93 if active else 0.0,
        },
    ]
    timeline = work / "phase3c_selected_shot_timeline.json"
    _write_json(
        timeline,
        {
            "selected_shot_id": "shot_0003",
            "frames": frames,
        },
    )
    report = work / "phase3c_bidirectional_timeline_merge_report.json"
    _write_json(
        report,
        {
            "status": "PASS",
            "decision": "AUTHORIZE_PHASE3D_SERVICE_INTEGRATION_VALIDATION",
            "timeline_merge_complete": True,
            "source_frame_coverage_complete": True,
            "duplicate_source_frame_count": 0,
            "synthetic_tracking_used": False,
            "automatic_target_confirmation": False,
            "merged_timeline_sha256": _sha(timeline),
        },
    )
    launch = {
        "target_reference_set": {
            "path": str(target_reference),
            "sha256": _sha(target_reference),
        },
        "target_selection": {
            "path": str(target_selection),
            "sha256": _sha(target_selection),
        },
    }
    view = {
        "selected_shot_id": "shot_0003",
        "source_video": {"path": str(source), "sha256": _sha(source)},
        "source_video_sha256": _sha(source),
    }
    return launch, view


def _fake_crop(_source: Path, _frame: int, _bbox, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 32, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(output), image)


def test_phase4a_memory_uses_only_source_backed_active_evidence() -> None:
    work = _workspace()
    launch, view = _fixture_contract(work)
    with patch.object(adapter, "_write_bbox_crop", side_effect=_fake_crop):
        report = adapter.build_phase4a_initial_memory_from_phase3c(
            output_dir=work,
            launch=launch,
            view=view,
        )
    assert report["status"] == "PASS"
    assert report["decision"] == "AUTHORIZE_PHASE4A_INITIAL_MEMORY_REVIEW"
    assert report["immutable_native_reference_count"] == 3
    assert report["selected_phase3c_active_reference_count"] == 2
    assert report["automatic_target_confirmation"] is False

    memory = json.loads(Path(report["memory_revision_path"]).read_text(encoding="utf-8"))
    native = memory["native_references"]
    active = memory["active_references"]
    assert all(item["scoring_eligible"] is True for item in native)
    assert all(item["scoring_eligible"] is False for item in active)
    assert all(item["kind"] == "REAL_PHASE3C_ACTIVE_OBSERVATION" for item in active)
    assert memory["cross_shot_scoring_performed"] is False


def test_phase4a_rejects_motion_only_or_lost_evidence() -> None:
    work = _workspace()
    launch, view = _fixture_contract(work, active=False)
    with pytest.raises(RuntimeError, match="ACTIVE/REACQUIRED"):
        adapter.build_phase4a_initial_memory_from_phase3c(
            output_dir=work,
            launch=launch,
            view=view,
        )


def test_phase4a_memory_review_approval_does_not_claim_scoring() -> None:
    work = _workspace()
    memory = work / "initial_memory.json"
    _write_json(memory, {"memory_revision_id": "ecmem_initial_001"})
    state = {
        "status": "NEEDS_CONFIRMATION",
        "decision": "PAUSE_FOR_PHASE4A_INITIAL_TARGET_MEMORY_REVIEW",
        "pending_action": {"type": "MEMORY_REVIEW"},
        "shots": [{"shot_id": "shot_0004", "status": "SEARCHING_MEMORY_REVIEW_PENDING"}],
        "runtime": {
            "phase4a_initial_memory_build_complete": True,
            "phase4a_initial_memory_review_status": "PENDING",
            "memory_revision_path": str(memory),
            "memory_revision_sha256": _sha(memory),
        },
    }
    _write_json(work / "pipeline_state.json", state)
    args = SimpleNamespace(
        resume=True,
        approve_review="MEMORY",
        reject_review=None,
        target_memory_revision=memory,
        target_memory_sha256=_sha(memory),
    )
    result = adapter._handle_phase4a_memory_review_resume(args=args, output_dir=work)
    assert result == 0
    updated = json.loads((work / "pipeline_state.json").read_text(encoding="utf-8"))
    assert updated["decision"] == "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING"
    assert updated["runtime"]["phase4b_cross_shot_scoring_authorized"] is True
    assert updated["runtime"]["cross_shot_scoring_performed"] is False
    assert updated["runtime"]["backend_memory_used_by_provided_e2e_scoring"] is False


def test_diverse_active_selection_is_deterministic() -> None:
    rows = [{"frame_index": index} for index in range(10, 20)]
    selected = adapter._phase4a_select_diverse_active_rows(rows, maximum_count=5)
    assert [row["frame_index"] for row in selected] == [10, 12, 14, 17, 19]
