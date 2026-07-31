from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    CandidateObservation,
    CandidateSequence,
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1.feature_extractor import (
    Detection,
)
from app.domains.highlight.event_candidate_ranking_v1_1.schema import (
    EventAnnotationFinalizeRequest,
    EventAnnotationReviewRequest,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.annotation_service import (
    EventAnnotationCompatibilityService,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.contract import (
    APPROVED_SHOT_REVIEW_STATES,
    ShotInterval,
    audit_shot_contract_v112,
    extract_manifest_candidate_sha,
    resolve_event_context_v112,
    validate_scene_video_duration,
    verify_candidate_artifact_immutability,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.feature_extractor import (
    RawFeatureExtractorV112,
    select_ball_trajectory_by_shot,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.schema import (
    EventCandidateRankingV112Request,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.verifier import (
    EventCandidateRankingV112Verifier,
)
from app.domains.match.model import Match
from app.domains.project.model import Project


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V11_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1"
)
V111_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1_1"
)
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1_2"
)


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows

    def get(self, model, identity):
        return self.rows.get((model, identity))


class _TempStorage:
    def __init__(self, root: Path):
        self.project_root = root.resolve()
        self.storage_root = self.project_root

    def resolve_path(self, value):
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )


def _sequence(
    *,
    candidate_id: str = "candidate",
    shot_id: str = "shot_1",
    count: int = 8,
) -> CandidateSequence:
    return CandidateSequence(
        candidate_id=candidate_id,
        shot_id=shot_id,
        shot_index=0,
        local_tracklet_id=f"track_{candidate_id}",
        trackability_score=0.8,
        observations=tuple(
            CandidateObservation(
                global_frame=frame,
                scene_local_frame=frame,
                scene_local_time_sec=frame / 5,
                bbox_xyxy=(10 + frame, 10, 40 + frame, 70),
                detector_confidence=0.9,
                detection_id=f"d_{frame}",
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
        cv2.rectangle(
            frame, (10 + index, 10), (40 + index, 70), (255, 255, 255), -1
        )
        writer.write(frame)
    writer.release()


@pytest.mark.parametrize(
    "state",
    ["", "REJECTED", "UNREVIEWED", "REVIEWED_REJECT", "FAILED"],
)
def test_strict_shot_review_rejects_every_non_allowlisted_state(
    tmp_path: Path,
    state: str,
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
                        "status": state,
                    }
                ]
            }
        )
    )
    audit, _ = audit_shot_contract_v112(
        path, candidates=(_sequence(),), frame_count=8
    )
    assert audit["status"] == "FAIL"
    assert audit["unapproved_review_count"] == 1
    assert audit["unapproved_shot_ids"] == ["shot_1"]


@pytest.mark.parametrize("state", sorted(APPROVED_SHOT_REVIEW_STATES))
def test_strict_shot_review_accepts_only_explicit_allowlist(
    tmp_path: Path,
    state: str,
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
                        "status": state,
                    }
                ]
            }
        )
    )
    audit, intervals = audit_shot_contract_v112(
        path, candidates=(_sequence(),), frame_count=8
    )
    assert audit["status"] == "PASS"
    assert intervals[0].shot_id == "shot_1"


def test_candidate_discovery_and_manifest_sha_must_all_match() -> None:
    sha = "a" * 64
    assert extract_manifest_candidate_sha(
        {"files": {"scene_candidates.json": {"sha256": sha}}}
    ) == sha
    verify_candidate_artifact_immutability(
        current_sha256=sha,
        stored_discovery_sha256=sha,
        manifest_declared_sha256=sha,
    )
    with pytest.raises(ValueError, match="differ"):
        verify_candidate_artifact_immutability(
            current_sha256=sha,
            stored_discovery_sha256="b" * 64,
            manifest_declared_sha256=sha,
        )
    with pytest.raises(ValueError, match="does not declare"):
        extract_manifest_candidate_sha({"files": {}})


def test_linked_event_still_must_be_inside_valid_scene() -> None:
    from app.domains.highlight.model import HighlightRevision
    from app.domains.timeline.model import TimelineEvent

    revision = SimpleNamespace(
        revision_id="revision",
        project_id="project",
        selected_scene_ids=["scene"],
        action_spotting_job_id="job",
    )
    event = SimpleNamespace(
        timeline_event_id="event",
        match_id="match",
        source_job_id="job",
        source_artifact_id="artifact",
        label="goal",
        timestamp_sec=30.0,
        confidence=0.8,
    )
    scene = SimpleNamespace(
        timeline_event_id="scene",
        match_id="match",
        source_job_id="job",
        start_sec=2.0,
        end_sec=20.0,
        metadata_={"source_event_ids": ["event"]},
    )
    db = _FakeSession(
        {
            (HighlightRevision, "revision"): revision,
            (TimelineEvent, "event"): event,
            (TimelineEvent, "scene"): scene,
        }
    )
    with pytest.raises(ValueError, match="outside"):
        resolve_event_context_v112(
            db,
            project=SimpleNamespace(project_id="project", match_id="match"),
            revision_id="revision",
            event_id="event",
            scene_id="scene",
        )


def test_scene_video_duration_requires_tolerance() -> None:
    resolved = SimpleNamespace(scene_start_sec=2.0, scene_end_sec=12.0)
    contract = validate_scene_video_duration(
        resolved,
        video_fps=25.0,
        video_frame_count=249,
        tolerance_sec=0.25,
    )
    assert contract["difference_sec"] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="tolerance"):
        validate_scene_video_duration(
            resolved,
            video_fps=25.0,
            video_frame_count=200,
            tolerance_sec=0.25,
        )


def test_request_rejects_client_event_and_scene_fields() -> None:
    with pytest.raises(ValidationError):
        EventCandidateRankingV112Request.model_validate(
            {
                "event_id": "event",
                "scene_id": "scene",
                "event_label": "goal",
                "event_time_sec": 12.4,
                "scene_start_sec": 2.4,
                "scene_end_sec": 28.4,
            }
        )


def test_ball_keys_reset_per_shot() -> None:
    intervals = (
        ShotInterval("shot_1", 0, 3),
        ShotInterval("shot_2", 4, 7),
    )
    policy = json.loads((V111_ROOT / "safety_policy.json").read_text())[
        "ball_selection"
    ]
    detections = (
        Detection(2, "ball", (10, 10, 14, 14), 0.9),
        Detection(3, "ball", (11, 10, 15, 14), 0.9),
        Detection(4, "ball", (80, 60, 84, 64), 0.9),
        Detection(5, "ball", (81, 60, 85, 64), 0.9),
    )
    selected = select_ball_trajectory_by_shot(
        detections,
        intervals=intervals,
        width=100,
        height=80,
        policy=policy,
    )
    assert set(selected) == {
        ("shot_1", 2),
        ("shot_1", 3),
        ("shot_2", 4),
        ("shot_2", 5),
    }


def test_ball_coverage_continuity_and_cross_shot_broadcast_contract(
    tmp_path: Path,
) -> None:
    video = tmp_path / "scene.mp4"
    _write_video(video)
    safety = json.loads((V111_ROOT / "safety_policy.json").read_text())
    ranking_policy = json.loads(
        (V11_ROOT / "event_candidate_ranking_policy.json").read_text()
    )
    extractor = RawFeatureExtractorV112(
        shot_intervals=(ShotInterval("shot_1", 0, 7),),
        video_fps=5.0,
        canonical_event_label="goal",
        safety_policy=safety,
        width=100,
        height=80,
        event_scene_local_sec=0.8,
        event_window_before_sec=6,
        event_window_after_sec=8,
        closeup_area_ratio=0.08,
        ball_labels=set(ranking_policy["ball"]["labels"]),
        ball_low_coverage_threshold=0.2,
    )
    detections = tuple(
        Detection(frame, "ball", (25 + frame, 62, 30 + frame, 68), 0.9)
        for frame in (1, 3, 5)
    )
    rows, _ = extractor.extract_all(
        candidates=(_sequence(),),
        detections=detections,
        video_path=video,
    )
    ball = rows[0]["raw_features"]["ball"]
    assert ball["candidate_frame_coverage"] == pytest.approx(3 / 8)
    assert ball["continuity_selected_pair_count"] == 2
    assert ball["trajectory_continuity"] == 1.0
    broadcast = rows[0]["raw_features"]["broadcast"]
    assert broadcast["post_event_shots_with_appearance"] is None
    assert broadcast["post_event_shots_with_appearance_deprecated"][
        "deprecated"
    ] is True


def _ranking_document(package: str, schema: str, policy_sha: str) -> dict:
    return {
        "package": package,
        "schema_version": schema,
        "freeze": {
            "source_manifest_sha256": "b" * 64,
            "feature_schema_sha256": "c" * 64,
            "ranking_policy_sha256": policy_sha,
        },
        "all_candidates": [
            {"candidate_id": "candidate_a"},
            {"candidate_id": "candidate_b"},
        ],
        "shortlist": [{"candidate_id": "candidate_a"}],
    }


def test_v1_1_1_annotation_review_finalize_evaluation_e2e(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(
            user_id="user",
            email="user@example.com",
            password_hash="x",
            display_name="User",
            role="USER",
            is_active=True,
            developer_mode_enabled=False,
            event_weights={},
        )
        match = Match(match_id="match", owner_id="user", metadata_={})
        project = Project(
            project_id="project",
            match_id="match",
            owner_id="user",
            title="Project",
            status="DRAFT",
        )
        db.add_all([user, match, project])
        db.flush()
        ranking_path = tmp_path / "ranking.json"
        ranking_path.write_text(
            json.dumps(
                _ranking_document(
                    "target_centric_tracking_event_candidate_ranking_v1_1_1",
                    "kickclip.event_candidate_ranking.v1_1_1",
                    "a" * 64,
                )
            )
        )
        ranking = Artifact(
            artifact_id="ranking_artifact",
            match_id="match",
            project_id="project",
            analysis_job_id=None,
            artifact_type="EVENT_CANDIDATE_RANKING_V1_1_1_SHADOW",
            file_path=ranking_path.name,
            mime_type="application/json",
            metadata_={
                "sha256": sha256_file(ranking_path),
                "ranking_id": "ranking",
                "owner_id": "user",
            },
        )
        db.add(ranking)
        db.commit()
        service = EventAnnotationCompatibilityService(db)
        service.storage = _TempStorage(tmp_path)
        review = service.submit_review(
            ranking_artifact_id=ranking.artifact_id,
            user=user,
            payload=EventAnnotationReviewRequest(
                primary_actor_visible=True,
                primary_actor_in_candidate_set=True,
                primary_actor_candidate_ids=["candidate_a"],
                directly_related_candidate_ids=["candidate_b"],
                candidate_roles={"candidate_a": "PRIMARY_EVENT_ACTOR"},
            ),
        )
        final = service.finalize(
            ranking_artifact_id=ranking.artifact_id,
            user=user,
            payload=EventAnnotationFinalizeRequest(
                review_artifact_ids=[review.artifact_id],
                approval_mode="FINAL_APPROVED",
                approved_reviewer="user",
            ),
        )
        evaluation = service.evaluate(
            ranking_artifact_id=ranking.artifact_id,
            annotation_artifact_id=final.artifact_id,
            user=user,
        )
        assert evaluation["primary_actor_recall_at_1"] == 1.0
        assert evaluation["ranking_identity"]["ranking_schema_version"].endswith(
            "v1_1_1"
        )
        assert evaluation["cross_version_metrics_mixed"] is False


def test_v1_1_2_contract_verifier_validates_schema_smoke() -> None:
    result = EventCandidateRankingV112Verifier(PACKAGE_ROOT).check()
    assert result.event_ranking_contract_runtime_verified, (
        result.code,
        result.message,
    )
    assert result.full_event_recommendation_e2e_verified is False
