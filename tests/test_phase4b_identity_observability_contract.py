from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    PROJECT_ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)
RUNTIME_SYNC_PATH = (
    PROJECT_ROOT / "app" / "domains" / "candidate_handoff_r1" / "runtime_sync.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("kickclip_observability_adapter_test", ADAPTER_PATH)


def _det(
    detection_id: str,
    frame: int,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
):
    return SimpleNamespace(
        detection_id=detection_id,
        frame=frame,
        bbox=bbox,
        confidence=confidence,
        class_id=0,
        class_name="player",
    )


def test_isolated_upright_tracklet_is_identity_observable() -> None:
    candidate = [
        _det(f"target_{frame}", frame, (400 + frame * 2, 180, 650 + frame * 2, 900))
        for frame in range(8)
    ]
    by_frame = {row.frame: [row] for row in candidate}

    result = ADAPTER._phase4b_identity_observability(
        candidate,
        by_frame,
        frame_width=1920,
        frame_height=1080,
    )

    assert result["policy"] == ADAPTER.PHASE4B_IDENTITY_OBSERVABILITY_POLICY
    assert result["passed"] is True
    assert result["classification"] == "REVIEWABLE_SINGLE_PERSON_IDENTITY"
    assert result["clean_frame_count"] == len(candidate)
    assert result["crowded_frame_count"] == 0
    assert set(result["clean_detection_ids"]) == {
        row.detection_id for row in candidate
    }


def test_group_pile_tracklet_is_rejected_before_reid() -> None:
    candidate = [
        _det(f"candidate_{frame}", frame, (500, 520, 1100, 860))
        for frame in range(10)
    ]
    by_frame = {}
    for row in candidate:
        frame = row.frame
        by_frame[frame] = [
            row,
            _det(f"other_a_{frame}", frame, (430, 480, 850, 920)),
            _det(f"other_b_{frame}", frame, (760, 500, 1250, 930)),
            _det(f"other_c_{frame}", frame, (620, 390, 1000, 830)),
        ]

    result = ADAPTER._phase4b_identity_observability(
        candidate,
        by_frame,
        frame_width=1920,
        frame_height=1080,
    )

    assert result["passed"] is False
    assert result["classification"] == "UNREVIEWABLE_GROUP_OCCLUSION"
    assert result["clean_frame_count"] == 0
    assert result["crowded_frame_ratio"] == 1.0
    assert result["multi_person_contamination_ratio"] == 1.0
    assert "EXCESSIVE_GROUP_OCCLUSION" in result["rejection_reasons"]
    assert "MULTI_PERSON_CONTAMINATION_DOMINANT" in result["rejection_reasons"]


def test_near_duplicate_detector_box_is_not_treated_as_another_person() -> None:
    candidate = [
        _det(f"candidate_{frame}", frame, (500, 160, 760, 920))
        for frame in range(6)
    ]
    by_frame = {}
    for row in candidate:
        frame = row.frame
        by_frame[frame] = [
            row,
            _det(f"duplicate_{frame}", frame, (505, 165, 765, 925), 0.5),
        ]

    result = ADAPTER._phase4b_identity_observability(
        candidate,
        by_frame,
        frame_width=1920,
        frame_height=1080,
    )

    assert result["passed"] is True
    assert result["crowded_frame_count"] == 0
    assert all(
        row["duplicate_like_detection_count"] == 1
        for row in result["per_observation"]
    )


def test_horizontal_partial_body_tracklet_is_unreviewable() -> None:
    candidate = [
        _det(f"horizontal_{frame}", frame, (350, 600, 1250, 900))
        for frame in range(8)
    ]
    by_frame = {row.frame: [row] for row in candidate}

    result = ADAPTER._phase4b_identity_observability(
        candidate,
        by_frame,
        frame_width=1920,
        frame_height=1080,
    )

    assert result["passed"] is False
    assert result["classification"] == "UNREVIEWABLE_GROUP_OCCLUSION"
    assert result["horizontal_or_partial_frame_ratio"] == 1.0
    assert "NON_UPRIGHT_OR_PARTIAL_BODY_DOMINANT" in result["rejection_reasons"]


def test_mixed_tracklet_uses_only_clean_detection_ids() -> None:
    candidate = []
    by_frame = {}
    for frame in range(10):
        row = _det(f"candidate_{frame}", frame, (500, 180, 760, 920))
        candidate.append(row)
        if frame < 6:
            by_frame[frame] = [row]
        else:
            by_frame[frame] = [
                row,
                _det(f"other_{frame}", frame, (560, 220, 880, 940)),
            ]

    result = ADAPTER._phase4b_identity_observability(
        candidate,
        by_frame,
        frame_width=1920,
        frame_height=1080,
    )

    assert result["passed"] is True
    assert result["clean_frame_count"] == 6
    assert result["crowded_frame_count"] == 4
    assert set(result["clean_detection_ids"]) == {
        f"candidate_{frame}" for frame in range(6)
    }


def test_group_occlusion_result_marks_shot_exhausted_and_continues(tmp_path: Path) -> None:
    state = {
        "runtime": {},
        "shots": [
            {"shot_id": "shot_0008", "status": "SEARCHING_MEMORY_READY"},
            {"shot_id": "shot_0009", "status": "SEARCHING_MEMORY_READY"},
        ],
        "ambiguities": [],
    }
    result = {
        "candidate_count": 0,
        "review_candidates": [],
        "ranked_candidates": [],
        "gate": {"operational_state": "SAFE_REJECTED_UNREVIEWABLE_GROUP_OCCLUSION"},
        "candidate_role_filter": {},
        "negative_role_memory": {},
        "identity_negative_memory": {},
        "persistent_role_negative_memory": {},
        "negative_memory_policy": "",
        "raw_tracklet_count": 5,
        "unique_candidate_count_before_negative_gate": 0,
        "rejected_role_confusion_tracklets": [],
        "rejected_identity_observability_candidates": [
            {
                "candidate_id": "shot_0008_track_0001",
                "identity_observability": {
                    "policy": ADAPTER.PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                    "passed": False,
                    "classification": "UNREVIEWABLE_GROUP_OCCLUSION",
                },
            }
        ],
        "rejected_negative_margin_candidates": [],
        "deduplicated_candidates": [],
        "identity_observability_policy": ADAPTER.PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
        "identity_observability_raw_candidate_count": 5,
        "identity_observability_passed_candidate_count": 0,
        "exhaustion_reason": "UNREVIEWABLE_GROUP_OCCLUSION",
    }
    memory_path = tmp_path / "memory.json"
    memory_path.write_text("{}", encoding="utf-8")

    updated = ADAPTER._phase4b_apply_result_to_state(
        state=state,
        shot={
            "shot_id": "shot_0008",
            "shot_index": 8,
            "start_frame": 635,
            "end_frame_inclusive": 684,
        },
        result=result,
        memory={
            "memory_revision_id": "ecmem_test",
            "reference_count": 6,
            "scoring_reference_count": 6,
        },
        memory_path=memory_path,
        memory_sha="a" * 64,
        generation=8,
        report_path=tmp_path / "attempt.json",
        remaining_ready_shot_ids=["shot_0009"],
    )

    assert updated["pending_action"] is None
    assert updated["status"] == "RUNNING"
    assert updated["decision"] == ADAPTER.PHASE4B_CONTINUE_DECISION
    assert updated["shots"][0]["status"] == (
        ADAPTER.PHASE4B_UNREVIEWABLE_GROUP_OCCLUSION_STATUS
    )
    assert updated["runtime"]["phase4b_next_shot_id"] == "shot_0009"
    assert updated["runtime"]["automatic_target_confirmation"] is False


def test_runtime_sync_treats_group_occlusion_exhaustion_as_terminal_shot() -> None:
    source = RUNTIME_SYNC_PATH.read_text(encoding="utf-8")
    assert "SEARCH_EXHAUSTED_UNREVIEWABLE_GROUP_OCCLUSION" in source


def test_process_candidates_does_not_embed_group_occlusion_tracklet(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    candidate_detections = [
        _det(f"candidate_{frame}", frame, (500, 520, 1100, 860))
        for frame in range(8)
    ]
    by_frame = {}
    detection_by_id = {}
    for row in candidate_detections:
        frame = row.frame
        others = [
            _det(f"other_a_{frame}", frame, (430, 480, 850, 920)),
            _det(f"other_b_{frame}", frame, (760, 500, 1250, 930)),
        ]
        by_frame[frame] = [row, *others]
        for item in by_frame[frame]:
            detection_by_id[item.detection_id] = item

    observations = [
        SimpleNamespace(
            detection_id=row.detection_id,
            frame_index=row.frame,
            confidence=row.confidence,
            bbox_xyxy=row.bbox,
        )
        for row in candidate_detections
    ]
    track = SimpleNamespace(
        observations=observations,
        start_frame=0,
        end_frame=7,
        internal_id=1,
    )

    class FakeB0:
        @staticmethod
        def build_tracklets(**kwargs):
            return [track], {}

        @staticmethod
        def make_tracklet_strip(clip, candidate_id, track, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas = np.zeros((120, 300, 3), dtype=np.uint8)
            assert cv2.imwrite(str(output), canvas)

    class FailIfEmbedded:
        @staticmethod
        def collect_crops(*args, **kwargs):
            raise AssertionError("observability-rejected tracklet reached crop collection")

        @staticmethod
        def embed_crops(*args, **kwargs):
            raise AssertionError("observability-rejected tracklet reached ReID")

    runtime = {
        "b0": FakeB0(),
        "b1": SimpleNamespace(),
        "stage2b": FailIfEmbedded(),
        "b3": SimpleNamespace(),
        "by_frame": by_frame,
        "negative_role_by_frame": {},
        "detection_by_id": detection_by_id,
        "b0_policy": {
            "max_age": 3,
            "minimum_predicted_iou": 0.1,
            "maximum_center_distance": 2.0,
            "minimum_area_ratio": 0.2,
            "maximum_area_ratio": 5.0,
            "minimum_match_score": 0.1,
            "minimum_tracklet_frames": 3,
        },
        "b2_policy": {
            "minimum_retrieval_score": 0.65,
            "minimum_prototype_similarity": 0.6,
            "minimum_top1_top2_gap": 0.08,
            "minimum_plausible_score": 0.5,
            "minimum_plausible_prototype": 0.5,
            "review_candidate_count": 3,
        },
        "candidate_role_filter": {},
        "negative_memory_available": True,
        "negative_role_memory": {},
        "identity_negative_memory": {},
        "persistent_role_negative_memory": {},
        "negative_memory_policy": "test",
        "clip": tmp_path / "analysis_clip.mp4",
    }
    state = {
        "video": {
            "path": str(tmp_path / "source.mp4"),
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
        }
    }
    result = ADAPTER._phase4b_process_candidates(
        output_dir=tmp_path,
        state=state,
        shot={
            "shot_id": "shot_0008",
            "start_frame": 635,
            "end_frame_inclusive": 684,
        },
        runtime=runtime,
        memory={"memory_revision_id": "ecmem_test"},
        memory_path=tmp_path / "memory.json",
        memory_sha="a" * 64,
        generation=8,
    )

    assert result["candidate_count"] == 0
    assert result["review_candidates"] == []
    assert result["exhaustion_reason"] == "UNREVIEWABLE_GROUP_OCCLUSION"
    assert len(result["rejected_identity_observability_candidates"]) == 1
    candidate_dir = (
        tmp_path
        / "phase4b_cross_shot"
        / "shot_0008"
        / "candidates"
        / "shot_0008_track_0001"
    )
    assert not (candidate_dir / "candidate_embeddings.npy").exists()
    assert Path(result["contact_sheet_path"]).is_file()
