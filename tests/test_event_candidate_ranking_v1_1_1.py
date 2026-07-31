from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    CandidateObservation,
    CandidateSequence,
    canonical_event_label,
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1.feature_extractor import (
    Detection,
)
from app.domains.highlight.event_candidate_ranking_v1_1.policy import (
    load_policy,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.contract import (
    audit_shot_contract,
    resolve_event_context,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.backend_adapter import (
    cache_fingerprint_v111,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.diversity import (
    diverse_shortlist_v111,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.feature_extractor import (
    RawFeatureExtractorV111,
    select_ball_trajectory,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.frame_reader import (
    BoundedVideoFrameReader,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.verifier import (
    EventCandidateRankingV111Verifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V11_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1"
)
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1_1"
)


class _FakeSession:
    def __init__(self, rows: dict[tuple[type, str], object]) -> None:
        self.rows = rows

    def get(self, model, identity):
        return self.rows.get((model, identity))


def _resolved_rows():
    from app.domains.highlight.model import HighlightRevision
    from app.domains.timeline.model import TimelineEvent

    revision = SimpleNamespace(
        revision_id="revision",
        project_id="project",
        selected_scene_ids=["scene"],
        action_spotting_job_id="job_action",
    )
    event = SimpleNamespace(
        timeline_event_id="event",
        match_id="match",
        source_job_id="job_action",
        source_artifact_id="artifact_action",
        label="Goal",
        timestamp_sec=12.4,
        confidence=0.81,
    )
    scene = SimpleNamespace(
        timeline_event_id="scene",
        match_id="match",
        source_job_id="job_action",
        start_sec=2.4,
        end_sec=28.4,
        metadata_={"source_event_ids": ["event"]},
    )
    session = _FakeSession(
        {
            (HighlightRevision, "revision"): revision,
            (TimelineEvent, "event"): event,
            (TimelineEvent, "scene"): scene,
        }
    )
    project = SimpleNamespace(project_id="project", match_id="match")
    return session, project, revision, event, scene


def _sequence(
    *,
    candidate_id: str = "candidate",
    shot_id: str = "shot_1",
    tracklet_id: str = "local_1",
    count: int = 8,
) -> CandidateSequence:
    return CandidateSequence(
        candidate_id=candidate_id,
        shot_id=shot_id,
        shot_index=0,
        local_tracklet_id=tracklet_id,
        trackability_score=0.8,
        observations=tuple(
            CandidateObservation(
                global_frame=frame,
                scene_local_frame=frame,
                scene_local_time_sec=frame / 5,
                bbox_xyxy=(10 + frame, 10, 40 + frame, 70),
                detector_confidence=0.9,
                detection_id=f"d_{candidate_id}_{frame}",
                shot_id=shot_id,
            )
            for frame in range(count)
        ),
    )


def _write_video(path: Path, count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (100, 80),
    )
    assert writer.isOpened()
    for index in range(count):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + index, 10), (40 + index, 70), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_v1_1_baseline_and_weight_policy_are_unchanged() -> None:
    baseline = json.loads(
        (PACKAGE_ROOT / "frozen_v1_1_baseline.json").read_text()
    )
    for relative, expected in baseline["sha256"].items():
        assert sha256_file(PROJECT_ROOT / relative) == expected
    safety = json.loads((PACKAGE_ROOT / "safety_policy.json").read_text())
    assert safety["ranking_weights_changed"] is False
    assert sha256_file(V11_ROOT / "event_candidate_ranking_policy.json") == (
        "7359adf60fba7128080bcd59895074f883a5c18f456ad16f1510d92b8af21e44"
    )


def test_event_and_scene_are_resolved_from_server_relationships() -> None:
    session, project, _, event, _ = _resolved_rows()
    resolved = resolve_event_context(
        session,
        project=project,
        revision_id="revision",
        event_id="event",
        scene_id="scene",
    )
    assert resolved.event_label == event.label
    assert resolved.event_time_sec == event.timestamp_sec
    assert resolved.scene_start_sec == 2.4
    assert resolved.scene_end_sec == 28.4
    assert resolved.canonical_event_label == "goal"


def test_event_revision_project_mismatch_is_rejected() -> None:
    session, project, revision, event, _ = _resolved_rows()
    event.source_job_id = "other_job"
    with pytest.raises(ValueError, match="Action Spotting job"):
        resolve_event_context(
            session,
            project=project,
            revision_id=revision.revision_id,
            event_id="event",
            scene_id="scene",
        )


def test_safety_policy_and_source_change_invalidate_cache() -> None:
    material = {
        "ranking_source_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "safety_policy_sha256": "c" * 64,
        "feature_schema_sha256": "d" * 64,
        "candidate_manifest_sha256": "e" * 64,
        "shot_boundaries_sha256": "f" * 64,
        "event_id": "event",
        "event_time_sec": 12.4,
        "event_label": "goal",
        "source_video_sha256": "0" * 64,
    }
    baseline = cache_fingerprint_v111(material)
    assert baseline != cache_fingerprint_v111(
        {**material, "safety_policy_sha256": "1" * 64}
    )
    assert baseline != cache_fingerprint_v111(
        {**material, "ranking_source_manifest_sha256": "2" * 64}
    )


def test_bounded_frame_reader_never_exceeds_lru_limit(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    _write_video(video, count=12)
    reader = BoundedVideoFrameReader(video, max_cached_frames=3)
    try:
        for index in range(12):
            assert reader.get(index) is not None
        stats = reader.stats()
    finally:
        reader.close()
    assert stats["peak_cached_frames"] == 3
    assert stats["peak_cached_bytes"] <= 3 * 100 * 80 * 3


def test_shot_audit_requires_full_unique_reviewed_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shots.json"
    path.write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "shot_1",
                        "start_frame": 0,
                        "end_frame": 7,
                        "status": "REVIEWED",
                    }
                ]
            }
        )
    )
    assert audit_shot_contract(
        path, candidates=(_sequence(),), frame_count=8
    )["status"] == "PASS"
    document = json.loads(path.read_text())
    document["shots"][0]["status"] = "PENDING_REVIEW"
    path.write_text(json.dumps(document))
    failed = audit_shot_contract(
        path, candidates=(_sequence(),), frame_count=8
    )
    assert failed["pending_review_count"] == 1
    assert failed["status"] == "FAIL"


def _diversity_row(candidate_id: str, shot: str, tracklet: str, score: float):
    return {
        "candidate_id": candidate_id,
        "shot_id": shot,
        "local_tracklet_id": tracklet,
        "recommendation_score": score,
        "reliability_state": "PARTIAL_FEATURES",
        "raw_features": {
            "broadcast": {"first_post_event_closeup_delay_sec": None},
            "temporal": {"visible_duration_before_event_sec": 1.0},
        },
        "_trajectory": [
            {
                "frame": frame,
                "time_sec": frame / 5,
                "bbox_xyxy": [10, 10, 40, 70],
            }
            for frame in range(4)
        ],
    }


def test_duplicate_key_is_scoped_by_shot_and_local_tracklet() -> None:
    policy = load_policy(V11_ROOT / "event_candidate_ranking_policy.json")
    rows = [
        _diversity_row("a", "shot_1", "local_1", 0.9),
        _diversity_row("b", "shot_2", "local_1", 0.8),
    ]
    shortlist, decisions = diverse_shortlist_v111(
        rows, size=3, width=100, height=80, policy=policy
    )
    assert [row["candidate_id"] for row in shortlist] == ["a", "b"]
    assert all(row["decision"] == "INCLUDED" for row in decisions)


def test_ball_selection_one_per_frame_prefers_valid_continuity() -> None:
    safety = json.loads((PACKAGE_ROOT / "safety_policy.json").read_text())
    detections = (
        Detection(0, "ball", (10, 10, 14, 14), 0.8),
        Detection(0, "ball", (80, 70, 84, 74), 0.95),
        Detection(1, "ball", (12, 11, 16, 15), 0.75),
        Detection(1, "ball", (82, 70, 86, 74), 0.76),
    )
    selected = select_ball_trajectory(
        detections,
        width=100,
        height=80,
        policy=safety["ball_selection"],
    )
    assert len(selected) == 2
    assert set(selected) == {0, 1}


def test_broadcast_cross_shot_unavailable_and_low_ball_null(
    tmp_path: Path,
) -> None:
    video = tmp_path / "scene.mp4"
    _write_video(video)
    safety = json.loads((PACKAGE_ROOT / "safety_policy.json").read_text())
    policy = load_policy(V11_ROOT / "event_candidate_ranking_policy.json")
    extractor = RawFeatureExtractorV111(
        canonical_event_label="goal",
        safety_policy=safety,
        width=100,
        height=80,
        event_scene_local_sec=0.8,
        event_window_before_sec=policy["event_window"]["before_sec"],
        event_window_after_sec=policy["event_window"]["after_sec"],
        closeup_area_ratio=policy["visual"]["closeup_area_ratio"],
        ball_labels={"ball"},
        ball_low_coverage_threshold=0.2,
    )
    rows, summary = extractor.extract_all(
        candidates=(_sequence(),),
        detections=(
            Detection(0, "ball", (24, 64, 29, 70), 0.9),
        ),
        video_path=video,
    )
    row = rows[0]
    broadcast = row["raw_features"]["broadcast"]
    assert broadcast["local_post_event_focus"] is not None
    assert broadcast["cross_shot_repeated_focus"] is None
    assert "repeated_post_event_focus" not in broadcast
    ball = row["raw_features"]["ball"]
    assert ball["state"] == "LOW_COVERAGE"
    assert ball["event_near_median_foot_distance"] is None
    assert row["feature_completeness_score"] < 1
    assert row["reliability_state"] == "PARTIAL_FEATURES"
    assert summary["frame_decoding"]["peak_cached_frames"] <= 4


def test_v1_1_1_safety_verifier_keeps_full_recommendation_blocked() -> None:
    result = EventCandidateRankingV111Verifier(PACKAGE_ROOT).check()
    assert result.event_ranking_safety_runtime_verified is True
    assert result.full_event_recommendation_e2e_verified is False
