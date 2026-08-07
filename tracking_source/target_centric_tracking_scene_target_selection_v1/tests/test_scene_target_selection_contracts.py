from __future__ import annotations

import json
from pathlib import Path

import pytest

from common import atomic_json, sha256_file
from contracts import (
    ShotBoundaryContractError,
    tracking_cache_key,
    validate_reviewed_shot_boundaries,
)
from earlier_anchor import (
    confirm_earlier_anchor,
    propose_earlier_candidates,
    selected_shot_fallback,
)
from gallery import render_gallery_html
from selection import create_target_reference_set, create_target_selection


def _candidate(candidate_id: str, shot: int, frame: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "scene_id": "scene_goal",
        "shot_id": f"shot_{shot:04d}",
        "shot_index": shot,
        "local_tracklet_id": candidate_id.rsplit("_", 1)[-1],
        "first_frame": frame,
        "last_frame": frame + 20,
        "observation_count": 21,
        "representative_observation": {
            "frame_index": frame + 10,
            "time_sec": (frame + 10) / 25,
            "bbox_xyxy": [10, 20, 80, 180],
            "detector_confidence": 0.9,
            "thumbnail_artifact": f"candidates/{candidate_id}/representative.jpg",
        },
        "tracking_initialization_observation": {
            "frame_index": frame,
            "time_sec": frame / 25,
            "bbox_xyxy": [10, 20, 80, 180],
            "validation_state": "VALID",
            "stable_observation_count": 21,
        },
        "quality": {
            "trackability_score": 0.8,
            "identity_switch_risk": "LOW",
        },
        "artifacts": {
            "contact_sheet": f"candidates/{candidate_id}/contact_sheet.jpg",
            "tracklet_review_video": f"candidates/{candidate_id}/tracklet.mp4",
            "crops": [],
        },
        "observations": [],
        "provenance": {"cross_shot_identity_linked": False},
    }


def _write_candidates(root: Path, candidates: list[dict]) -> None:
    atomic_json(
        root / "scene_candidates.json",
        {
            "schema_version": "kickclip.scene_player_candidates.v1",
            "scene_id": "scene_goal",
            "candidates": candidates,
        },
    )


def test_pending_cut_blocks_scene_wide_discovery(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"scene")
    boundary = tmp_path / "shot_boundaries.json"
    atomic_json(
        boundary,
        {
            "video": {"sha256": sha256_file(video), "frame_count": 100},
            "shots": [
                {
                    "shot_index": 0,
                    "start_frame": 0,
                    "end_frame_inclusive": 99,
                    "review_state": "REVIEWED_PASS",
                }
            ],
            "diagnostics": {
                "review_required": True,
                "pending_cut_frames": [40],
            },
            "review_contract": {"pending_cut_frames": [40]},
        },
    )
    with pytest.raises(
        ShotBoundaryContractError, match="WAITING_SHOT_BOUNDARY_REVIEW"
    ):
        validate_reviewed_shot_boundaries(
            path=boundary,
            video_sha256=sha256_file(video),
            frame_count=100,
        )


def test_first_frame_only_regression_and_later_selection(tmp_path: Path) -> None:
    candidates = [
        _candidate("scene_candidate_scene_goal_shot_0000_track_12", 0, 0),
        _candidate("scene_candidate_scene_goal_shot_0003_goal_actor", 3, 300),
        _candidate("scene_candidate_scene_goal_shot_0008_closeup", 8, 632),
    ]
    _write_candidates(tmp_path, candidates)
    selection = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id=candidates[2]["candidate_id"],
        reviewer="human",
        production_manifest_sha256="r2",
    )
    assert {row["shot_index"] for row in candidates} == {0, 3, 8}
    assert selection["selected_shot_id"] == "shot_0008"
    assert selection["selection_source"] == (
        "USER_SELECTED_SCENE_WIDE_CANDIDATE"
    )
    assert selection["automatic_target_selection"] is False


def test_selection_revisions_are_immutable(tmp_path: Path) -> None:
    candidates = [_candidate("candidate_a", 1, 40), _candidate("candidate_b", 2, 99)]
    _write_candidates(tmp_path, candidates)
    first = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id="candidate_a",
        reviewer="human",
        production_manifest_sha256="r2",
    )
    second = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id="candidate_b",
        reviewer="human",
        production_manifest_sha256="r2",
    )
    assert first["target_selection_revision"] == 1
    assert second["target_selection_revision"] == 2
    assert (tmp_path / "target_selection_r0001.json").is_file()
    assert (tmp_path / "target_selection_r0002.json").is_file()


def test_earlier_positive_anchor_requires_confirmation(tmp_path: Path) -> None:
    early = _candidate("early_goal_actor", 2, 120)
    selected = _candidate("later_closeup", 7, 575)
    _write_candidates(tmp_path, [early, selected])
    selection = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id=selected["candidate_id"],
        reviewer="human",
        production_manifest_sha256="r2",
    )

    def score(_selected, candidates):
        return [{"candidate": candidates[0], "retrieval_score": 0.99}]

    proposals = propose_earlier_candidates(
        output_root=tmp_path,
        selection=selection,
        score_candidates=score,
        maximum_candidates=3,
    )
    assert proposals["state"] == "WAITING_EARLIER_ANCHOR_CONFIRMATION"
    assert proposals["automatic_confirmation_allowed"] is False
    decision = confirm_earlier_anchor(
        output_root=tmp_path,
        selection=selection,
        candidate_id=early["candidate_id"],
        reviewer="human",
        reject_all=False,
    )
    assert decision["state"] == "USER_CONFIRMED_EARLIER_ANCHOR"
    assert decision["anchor_frame_index"] == 120


def test_earlier_ambiguous_candidates_never_auto_confirm(tmp_path: Path) -> None:
    earlier = [_candidate("similar_a", 1, 40), _candidate("similar_b", 2, 99)]
    selected = _candidate("later", 4, 300)
    _write_candidates(tmp_path, [*earlier, selected])
    selection = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id="later",
        reviewer="human",
        production_manifest_sha256="r2",
    )
    proposals = propose_earlier_candidates(
        output_root=tmp_path,
        selection=selection,
        score_candidates=lambda _selected, rows: [
            {"candidate": rows[0], "retrieval_score": 0.8},
            {"candidate": rows[1], "retrieval_score": 0.79},
        ],
        maximum_candidates=3,
    )
    assert len(proposals["proposals"]) == 2
    assert proposals["state"] == "WAITING_EARLIER_ANCHOR_CONFIRMATION"
    assert not (tmp_path / "earlier_anchor_decision.json").exists()


def test_no_earlier_candidate_starts_selected_shot_with_no_preanchor_bbox(
    tmp_path: Path,
) -> None:
    selected = _candidate("later", 4, 300)
    _write_candidates(tmp_path, [selected])
    selection = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id="later",
        reviewer="human",
        production_manifest_sha256="r2",
    )
    proposals = propose_earlier_candidates(
        output_root=tmp_path,
        selection=selection,
        score_candidates=lambda _selected, rows: [],
        maximum_candidates=3,
    )
    assert proposals["state"] == "NO_SAFE_EARLIER_CANDIDATE"
    decision = selected_shot_fallback(
        output_root=tmp_path,
        selection=selection,
        reviewer="human",
    )
    assert decision["state"] == "START_FROM_SELECTED_SHOT"
    assert decision["anchor_frame_index"] == 300


def test_foreign_or_unlisted_earlier_candidate_is_rejected(tmp_path: Path) -> None:
    selected = _candidate("later", 4, 300)
    _write_candidates(tmp_path, [selected])
    selection = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id="later",
        reviewer="human",
        production_manifest_sha256="r2",
    )
    atomic_json(
        tmp_path / "earlier_candidate_proposals.json",
        {"selection_id": selection["selection_id"], "proposals": []},
    )
    with pytest.raises(ValueError, match="current proposal list"):
        confirm_earlier_anchor(
            output_root=tmp_path,
            selection=selection,
            candidate_id="foreign_job_candidate",
            reviewer="human",
            reject_all=False,
        )


def test_candidate_fragmentation_is_not_auto_merged() -> None:
    rows = [
        _candidate("same_shot_track_1", 2, 100),
        _candidate("same_shot_track_2", 2, 140),
    ]
    assert len(rows) == 2
    assert rows[0]["candidate_id"] != rows[1]["candidate_id"]
    assert all(
        row["provenance"]["cross_shot_identity_linked"] is False
        for row in rows
    )


def test_tracking_cache_isolated_by_selection_revision_and_candidate(
    tmp_path: Path,
) -> None:
    references = tmp_path / "target_reference_set.json"
    references.write_text(json.dumps({"references": []}), encoding="utf-8")
    base = dict(
        candidate_cache_key_value="gallery",
        earlier_anchor_decision={"state": "START_FROM_SELECTED_SHOT"},
        target_reference_set_path=references,
        r2_production_manifest_sha256="r2",
        cross_shot_mode="assisted",
    )
    player_12 = tracking_cache_key(
        **base,
        target_selection_revision=1,
        selected_candidate_id="player_12",
    )
    player_10 = tracking_cache_key(
        **base,
        target_selection_revision=2,
        selected_candidate_id="player_10",
    )
    assert player_12 != player_10


def test_later_shot_selection_builds_multi_crop_reference_set(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "scene.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (160, 120),
    )
    for index in range(30):
        frame = np.full((120, 160, 3), index * 3, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    candidate = _candidate("later_closeup", 3, 5)
    candidate["last_frame"] = 25
    candidate["observations"] = [
        {
            "frame_index": index,
            "time_sec": index / 25,
            "bbox_xyxy": [20, 10, 100, 110],
            "confidence": 0.9,
        }
        for index in range(5, 26)
    ]
    _write_candidates(tmp_path, [candidate])
    selection = create_target_selection(
        output_root=tmp_path,
        selected_candidate_id="later_closeup",
        reviewer="human",
        production_manifest_sha256="r2",
    )
    references = create_target_reference_set(
        output_root=tmp_path,
        video=video,
        selection=selection,
        maximum_references=6,
    )
    assert len(references["references"]) == 6
    assert all(
        not Path(row["crop_artifact"]).is_absolute()
        for row in references["references"]
    )


def test_portable_gallery_groups_all_shots_and_emits_selection_event(
    tmp_path: Path,
) -> None:
    rows = [
        _candidate("shot0_player12", 0, 0),
        _candidate("shot4_goal_actor", 4, 300),
        _candidate("shot8_closeup", 8, 632),
    ]
    page = render_gallery_html(
        output_root=tmp_path,
        candidates={"scene_id": "scene_goal", "candidates": rows},
        gallery={
            "shots": [
                {
                    "shot_id": f"shot_{shot:04d}",
                    "start_time_sec": shot,
                    "end_time_sec": shot + 1,
                    "candidate_ids": [
                        row["candidate_id"]
                        for row in rows
                        if row["shot_index"] == shot
                    ],
                }
                for shot in (0, 4, 8)
            ]
        },
    )
    html = page.read_text(encoding="utf-8")
    assert all(f"shot_{shot:04d}" in html for shot in (0, 4, 8))
    assert "kickclip:scene-target-selected" in html
    assert "첫 화면에 없는 선수도 선택할 수 있습니다" in html
