from __future__ import annotations

from app.domains.candidate_handoff_r1.runtime.r1_v1_v2_adapter_cli import (
    _normalize_source_shots,
    _read_shots,
    _shot_for_global_frame,
    _source_mapped_directional_frame,
)


def _boundaries() -> dict:
    return {
        "shots": [
            {"shot_id": "shot_A", "start_frame": 0, "end_frame_inclusive": 1},
            {"shot_id": "shot_B", "start_frame": 2, "end_frame_inclusive": 3},
            {"shot_id": "shot_C", "start_frame": 4, "end_frame_inclusive": 5},
        ]
    }


def _view() -> dict:
    reviewed = _read_shots(_boundaries(), frame_count=6)
    return {
        "source_offset_frame": 2,
        "selected_shot_id": "shot_B",
        "reviewed_shots": reviewed,
        "shot_map": [
            {
                "raw_shot_id": "shot_0000",
                "raw_shot_index": 0,
                "source_shot_id": "shot_B",
                "source_start_frame": 2,
                "source_end_frame_inclusive": 3,
                "processed_source_start_frame": 2,
                "local_start_frame": 0,
                "local_end_frame_inclusive": 1,
            },
            {
                "raw_shot_id": "shot_0001",
                "raw_shot_index": 1,
                "source_shot_id": "shot_C",
                "source_start_frame": 4,
                "source_end_frame_inclusive": 5,
                "processed_source_start_frame": 4,
                "local_start_frame": 2,
                "local_end_frame_inclusive": 3,
            },
        ],
    }


def test_source_mapped_directional_frame_emits_public_required_fields():
    result = _source_mapped_directional_frame(
        raw={
            "frame_index": 0,
            "state": "ACTIVE",
            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
            "bbox_source": "RFDETR_ASSOCIATED_DETECTION",
            "selected_detection_id": "f000000_d000",
            "tracking_confidence": 0.9,
            "identity_confidence": 0.8,
        },
        direction="FORWARD",
        source_frame=2,
        local_frame=0,
        fps=25.0,
        shot_id="shot_B",
    )

    assert result["identity_source"] == "RFDETR_ASSOCIATED_DETECTION"
    assert result["review_required"] is False
    assert result["frame_index"] == 2
    assert result["shot_id"] == "shot_B"
    assert result["phase3c_source"] == (
        "REAL_FROZEN_STAGE2_DIRECTIONAL_TIMELINE"
    )


def test_reviewed_shots_are_unique_contiguous_and_full_coverage():
    shots = _read_shots(_boundaries(), frame_count=6)

    assert [shot["shot_id"] for shot in shots] == ["shot_A", "shot_B", "shot_C"]
    assert [shot["shot_index"] for shot in shots] == [0, 1, 2]
    assert [shot["frame_count"] for shot in shots] == [2, 2, 2]
    assert shots[0]["cut_in_frame"] is None
    assert shots[-1]["cut_out_frame"] is None


def test_global_frame_shot_assignment_uses_all_reviewed_shots():
    view = _view()

    assert _shot_for_global_frame(view, 0) == "shot_A"
    assert _shot_for_global_frame(view, 2) == "shot_B"
    assert _shot_for_global_frame(view, 5) == "shot_C"


def test_normalized_output_shots_keep_global_ranges_and_runtime_status():
    result = _normalize_source_shots(
        raw_shots=[
            {
                "shot_id": "shot_0000",
                "start_frame": 0,
                "end_frame_inclusive": 1,
                "status": "INITIAL_TARGET_TRACKING_ONLY",
            },
            {
                "shot_id": "shot_0001",
                "start_frame": 2,
                "end_frame_inclusive": 3,
                "status": "SEARCHING_NO_MEMORY",
            },
        ],
        view=_view(),
    )

    assert [shot["shot_id"] for shot in result] == ["shot_A", "shot_B", "shot_C"]
    assert [(shot["start_frame"], shot["end_frame_inclusive"]) for shot in result] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert result[0]["status"] == "OUTSIDE_SELECTION_ANCHOR_VIEW"
    assert result[1]["runtime_shot_id"] == "shot_0000"
    assert result[1]["status"] == "INITIAL_TARGET_TRACKING_ONLY"
    assert result[2]["runtime_shot_id"] == "shot_0001"
    assert result[2]["status"] == "SEARCHING_NO_MEMORY"
