from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.domains.highlight.event_candidate_ranking_v1_1.backend_adapter import (
    cache_fingerprint,
)
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    ImmutableCandidateInput,
    canonical_event_label,
    parse_candidate_sequences,
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1.diversity import (
    diverse_shortlist,
)
from app.domains.highlight.event_candidate_ranking_v1_1.evaluation import (
    ACTOR_NOT_IN_CANDIDATE_SET,
    NOT_RUN,
    ApprovedAnnotation,
    aggregate_events,
    evaluate_event,
    finalize_reviews,
)
from app.domains.highlight.event_candidate_ranking_v1_1.feature_extractor import (
    BALL_SIGNAL_LOW_COVERAGE,
    BALL_SIGNAL_UNAVAILABLE,
    Detection,
    RawFeatureExtractor,
)
from app.domains.highlight.event_candidate_ranking_v1_1.policy import (
    load_policy,
)
from app.domains.highlight.event_candidate_ranking_v1_1.service import (
    EventCandidateRankingV11Engine,
)
from app.domains.highlight.event_candidate_ranking_v1_1.verifier import (
    EventCandidateRankingV11Verifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1"
)


def _write_video(path: Path, *, frame_count: int = 10) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (100, 80),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (10 + index, 10),
            (40 + index, 70),
            (255, 255, 255),
            -1,
        )
        writer.write(frame)
    writer.release()


def _candidate_document(count: int = 1, observations: int = 10) -> dict:
    return {
        "candidates": [
            {
                "candidate_id": f"candidate_{candidate_index}",
                "shot_id": f"shot_{candidate_index}",
                "shot_index": candidate_index,
                "local_tracklet_id": f"track_{candidate_index}",
                "quality": {"trackability_score": 0.7},
                "observations": [
                    {
                        "global_frame": frame,
                        "scene_local_frame": frame,
                        "scene_local_time_sec": frame / 5,
                        "bbox_xyxy": [
                            10 + frame + candidate_index,
                            10,
                            40 + frame + candidate_index,
                            70,
                        ],
                        "detector_confidence": 0.9,
                        "detection_id": f"d_{candidate_index}_{frame}",
                    }
                    for frame in range(observations)
                ],
            }
            for candidate_index in range(count)
        ]
    }


def _immutable(tmp_path: Path, *, detection_document: dict | None = None):
    video = tmp_path / "scene.mp4"
    _write_video(video)
    candidates = tmp_path / "scene_candidates.json"
    candidates.write_text(json.dumps(_candidate_document()), encoding="utf-8")
    detections = tmp_path / "detections.json"
    detections.write_text(
        json.dumps(detection_document or {"detections": []}),
        encoding="utf-8",
    )
    boundaries = tmp_path / "boundaries.json"
    boundaries.write_text('{"shots":[]}', encoding="utf-8")
    return ImmutableCandidateInput.load(
        discovery_root=tmp_path,
        scene_candidates_relative_path=candidates.name,
        scene_candidates_sha256=sha256_file(candidates),
        detections_relative_path=detections.name,
        detections_sha256=sha256_file(detections),
        source_video_relative_path=video.name,
        source_video_sha256=sha256_file(video),
        shot_boundaries_relative_path=boundaries.name,
        shot_boundaries_sha256=sha256_file(boundaries),
        candidate_manifest_sha256="a" * 64,
        video_width=100,
        video_height=80,
        video_fps=5.0,
        video_frame_count=10,
    )


def test_supported_and_unsupported_event_mapping() -> None:
    assert canonical_event_label("Goal") == "goal"
    assert canonical_event_label("shot-on-target") == "shot"
    assert canonical_event_label("Free Kick") == "free_kick"
    assert canonical_event_label("corner_kick") == "corner"
    for label in ("Foul", "Card", "Penalty"):
        assert canonical_event_label(label) is None


def test_actual_observation_parsing_and_time_conversion() -> None:
    parsed = parse_candidate_sequences(
        _candidate_document(observations=3),
        fps=5.0,
        frame_count=10,
    )
    assert len(parsed) == 1
    assert len(parsed[0].observations) == 3
    assert parsed[0].observations[2].global_frame == 2
    assert parsed[0].observations[2].scene_local_time_sec == pytest.approx(0.4)


def test_bbox_visual_and_motion_features_are_actual(tmp_path: Path) -> None:
    immutable = _immutable(tmp_path)
    extractor = RawFeatureExtractor(
        width=100,
        height=80,
        event_scene_local_sec=0.8,
        event_window_before_sec=1.0,
        event_window_after_sec=1.0,
        closeup_area_ratio=0.08,
        ball_labels={"ball"},
        ball_low_coverage_threshold=0.2,
    )
    rows, summary = extractor.extract_all(
        candidates=immutable.candidates,
        detections=(),
        video_path=immutable.source_video_path,
    )
    visual = rows[0]["raw_features"]["visual"]
    motion = rows[0]["raw_features"]["motion"]
    assert visual["bbox_area_ratio"] == pytest.approx(0.225)
    assert 0 < visual["center_proximity"] <= 1
    assert visual["crop_sharpness"] is not None
    assert motion["center_velocity"] is not None
    assert motion["center_velocity"] > 0
    assert summary["group_coverage"]["visual"] == 1.0
    assert summary["group_coverage"]["motion"] == 1.0


def test_ball_unavailable_and_low_coverage_are_not_fake_zero(
    tmp_path: Path,
) -> None:
    immutable = _immutable(tmp_path)
    extractor = RawFeatureExtractor(
        width=100,
        height=80,
        event_scene_local_sec=0.8,
        event_window_before_sec=1,
        event_window_after_sec=1,
        closeup_area_ratio=0.08,
        ball_labels={"ball"},
        ball_low_coverage_threshold=0.5,
    )
    rows, _ = extractor.extract_all(
        candidates=immutable.candidates,
        detections=(),
        video_path=immutable.source_video_path,
    )
    ball = rows[0]["raw_features"]["ball"]
    assert ball["state"] == BALL_SIGNAL_UNAVAILABLE
    assert ball["coverage"] is None
    assert ball["foot_region_distance"] is None
    one_ball = (
        Detection(
            frame=0,
            label="ball",
            bbox_xyxy=(24, 65, 30, 72),
            confidence=0.9,
        ),
    )
    rows, _ = extractor.extract_all(
        candidates=immutable.candidates,
        detections=one_ball,
        video_path=immutable.source_video_path,
    )
    ball = rows[0]["raw_features"]["ball"]
    assert ball["state"] == BALL_SIGNAL_LOW_COVERAGE
    assert ball["coverage"] == pytest.approx(0.1)
    assert ball["foot_region_distance"] is None


def _diversity_row(
    candidate_id: str,
    *,
    tracklet: str,
    shot: str,
    offset: float,
    score: float,
    closeup_only: bool = False,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "local_tracklet_id": tracklet,
        "shot_id": shot,
        "recommendation_score": score,
        "raw_features": {
            "broadcast": {
                "first_post_event_closeup_delay_sec": 1.0
                if closeup_only
                else None
            },
            "temporal": {
                "visible_duration_before_event_sec": 0.0
                if closeup_only
                else 1.0
            },
        },
        "_trajectory": [
            {
                "frame": frame,
                "time_sec": frame / 5,
                "bbox_xyxy": [10 + offset + frame, 10, 40 + offset + frame, 70],
            }
            for frame in range(5)
        ],
    }


def test_shortlist_fragmentation_and_closeup_diversity() -> None:
    policy = load_policy(PACKAGE_ROOT / "event_candidate_ranking_policy.json")
    rows = [
        _diversity_row("a", tracklet="t1", shot="s1", offset=0, score=0.9),
        _diversity_row("b", tracklet="t1", shot="s1", offset=0, score=0.89),
        _diversity_row("c", tracklet="t2", shot="s1", offset=1, score=0.88),
        _diversity_row("d", tracklet="t3", shot="s2", offset=20, score=0.87, closeup_only=True),
        _diversity_row("e", tracklet="t4", shot="s3", offset=30, score=0.86, closeup_only=True),
        _diversity_row("f", tracklet="t5", shot="s4", offset=40, score=0.85, closeup_only=True),
    ]
    shortlist, decisions = diverse_shortlist(
        rows,
        size=5,
        width=100,
        height=80,
        policy=policy,
    )
    assert "b" not in {row["candidate_id"] for row in shortlist}
    assert "c" not in {row["candidate_id"] for row in shortlist}
    assert "f" not in {row["candidate_id"] for row in shortlist}
    reasons = {row["candidate_id"]: row["reason"] for row in decisions}
    assert reasons["b"] == "DUPLICATE_LOCAL_TRACKLET"
    assert reasons["c"] == "OVERLAPPING_FRAGMENT_SAME_SHOT"
    assert reasons["f"] == "POST_EVENT_CLOSEUP_DIVERSITY_CAP"


def test_unsupported_event_returns_full_gallery(tmp_path: Path) -> None:
    result = EventCandidateRankingV11Engine(PACKAGE_ROOT).run(
        immutable_input=_immutable(tmp_path),
        event_id="event_1",
        event_label="Penalty",
        event_time_sec=0.5,
        event_confidence=0.9,
        scene_id="scene_1",
        scene_start_sec=0,
        scene_end_sec=2,
        shortlist_size=5,
    )
    assert result["ranking_status"] == "UNSUPPORTED_EVENT_CLASS"
    assert result["shortlist"] == []
    assert result["full_gallery_fallback"] == ["candidate_0"]
    assert result["automatic_target_confirmation"] is False


def test_actor_missing_and_incomplete_annotation_gate() -> None:
    incomplete = ApprovedAnnotation(
        annotation_status="IN_REVIEW",
        approval_mode="UNAPPROVED",
        primary_actor_visible=True,
        primary_actor_in_candidate_set=False,
        primary_actor_candidate_ids=(),
        directly_related_candidate_ids=(),
        actor_missing_reason=ACTOR_NOT_IN_CANDIDATE_SET,
        reviewer="reviewer_a",
        reviewed_at="2026-07-30T00:00:00Z",
    )
    assert evaluate_event(
        ranked_candidate_ids=["a"],
        shortlist_candidate_ids=["a"],
        annotation=incomplete,
    )["metric_status"] == NOT_RUN
    complete_missing = ApprovedAnnotation(
        **{
            **incomplete.__dict__,
            "annotation_status": "COMPLETE",
            "approval_mode": "FINAL_APPROVED",
        }
    )
    result = evaluate_event(
        ranked_candidate_ids=["a"],
        shortlist_candidate_ids=["a"],
        annotation=complete_missing,
    )
    assert result["candidate_generation_actor_coverage"] == 0.0
    assert result["primary_actor_recall_at_3"] == NOT_RUN


def test_final_or_consensus_required_and_aggregate_is_separate() -> None:
    invalid = ApprovedAnnotation(
        annotation_status="COMPLETE",
        approval_mode="UNAPPROVED",
        primary_actor_visible=True,
        primary_actor_in_candidate_set=True,
        primary_actor_candidate_ids=("a",),
        directly_related_candidate_ids=(),
        actor_missing_reason=None,
        reviewer="reviewer_a",
        reviewed_at="2026-07-30T00:00:00Z",
    )
    with pytest.raises(ValueError, match="final approval or consensus"):
        evaluate_event(
            ranked_candidate_ids=["a"],
            shortlist_candidate_ids=["a"],
            annotation=invalid,
        )
    approved = ApprovedAnnotation(
        **{**invalid.__dict__, "approval_mode": "EXPLICIT_CONSENSUS"}
    )
    event = evaluate_event(
        ranked_candidate_ids=["a", "b"],
        shortlist_candidate_ids=["a"],
        annotation=approved,
    )
    aggregate = aggregate_events([event, {"metric_status": NOT_RUN}])
    assert event["scope"] == "SINGLE_EVENT"
    assert aggregate["scope"] == "DATASET_AGGREGATE"
    assert aggregate["evaluated_event_count"] == 1


def test_multiple_reviewer_finalization_never_unions_labels() -> None:
    first = {
        "primary_actor_visible": True,
        "primary_actor_in_candidate_set": True,
        "primary_actor_candidate_ids": ["a"],
        "directly_related_candidate_ids": ["b"],
        "actor_missing_reason": None,
        "reviewer": "reviewer_a",
        "reviewed_at": "2026-07-30T00:00:00Z",
    }
    second = {
        **first,
        "reviewer": "reviewer_b",
        "reviewed_at": "2026-07-30T00:01:00Z",
    }
    consensus = finalize_reviews(
        [first, second],
        approval_mode="EXPLICIT_CONSENSUS",
    )
    assert consensus.primary_actor_candidate_ids == ("a",)
    disagreeing = {**second, "primary_actor_candidate_ids": ["c"]}
    with pytest.raises(ValueError, match="union is forbidden"):
        finalize_reviews(
            [first, disagreeing],
            approval_mode="EXPLICIT_CONSENSUS",
        )
    final = finalize_reviews(
        [first, disagreeing],
        approval_mode="FINAL_APPROVED",
        approved_reviewer="reviewer_b",
    )
    assert final.primary_actor_candidate_ids == ("c",)


def test_source_and_policy_change_invalidate_cache() -> None:
    base = {
        "ranking_source_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "feature_schema_sha256": "c" * 64,
        "candidate_manifest_sha256": "d" * 64,
        "shot_boundaries_sha256": "e" * 64,
        "source_video_sha256": "f" * 64,
        "event_id": "event",
        "event_time_sec": 1.0,
        "event_label": "goal",
    }
    first = cache_fingerprint(base)
    assert first != cache_fingerprint({**base, "policy_sha256": "0" * 64})
    assert first != cache_fingerprint(
        {**base, "ranking_source_manifest_sha256": "1" * 64}
    )


def test_v1_freeze_baseline_is_unchanged() -> None:
    baseline = json.loads(
        (PACKAGE_ROOT / "frozen_v1_baseline.json").read_text(encoding="utf-8")
    )
    for relative_path, expected in baseline["sha256"].items():
        assert sha256_file(PROJECT_ROOT / relative_path) == expected


def test_shadow_verifier_state_separation_and_auto_confirmation() -> None:
    status = EventCandidateRankingV11Verifier(PACKAGE_ROOT).check()
    assert status.event_ranking_shadow_runtime_verified is True
    assert status.full_event_recommendation_e2e_verified is False
    manifest = json.loads((PACKAGE_ROOT / "manifest.json").read_text())
    assert manifest["automatic_target_confirmation"] is False
    assert manifest["full_event_recommendation_e2e_verified"] is False
