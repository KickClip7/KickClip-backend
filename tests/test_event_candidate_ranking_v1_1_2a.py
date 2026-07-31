from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
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
    ShotInterval,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.backend_adapter import (
    StaleDiscoveryInputError,
    cache_fingerprint_v112a,
    resolve_frozen_discovery_root,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.contract import (
    audit_shot_contract_v112a,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.feature_extractor import (
    RawFeatureExtractorV112a,
    select_ball_trajectory_segments_by_shot,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.verifier import (
    EventCandidateRankingV112aVerifier,
)
from app.domains.match.model import Match
from app.domains.project.model import Project


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1_2a"
)
FIXTURE = (
    PROJECT_ROOT
    / "tests/fixtures/r2_r3_reviewed_shot_artifact.json"
)


def _observation(frame: int, shot_id: str) -> CandidateObservation:
    return CandidateObservation(
        global_frame=frame,
        scene_local_frame=frame,
        scene_local_time_sec=frame / 5,
        bbox_xyxy=(20.0, 10.0, 50.0, 70.0),
        detector_confidence=0.9,
        detection_id=f"d_{frame}",
        shot_id=shot_id,
    )


def _candidate(
    candidate_id: str,
    shot_id: str,
    frames: list[int],
) -> CandidateSequence:
    return CandidateSequence(
        candidate_id=candidate_id,
        shot_id=shot_id,
        shot_index=0,
        local_tracklet_id="track",
        trackability_score=0.8,
        observations=tuple(_observation(frame, shot_id) for frame in frames),
    )


def test_actual_r2_r3_shot_fixture_aliases_are_parsed() -> None:
    audit, intervals = audit_shot_contract_v112a(
        FIXTURE,
        candidates=(
            _candidate("a", "shot_0000", list(range(5))),
            _candidate("b", "shot_0001", list(range(5, 10))),
        ),
        frame_count=10,
    )
    assert audit["status"] == "PASS"
    assert audit["unapproved_review_count"] == 0
    assert intervals[-1].end_frame == 9


@pytest.mark.parametrize(
    "state",
    ["", "REVIEWED", "REJECTED", "UNREVIEWED", "FAILED"],
)
def test_shot_aliases_do_not_widen_approved_states(
    tmp_path: Path,
    state: str,
) -> None:
    path = tmp_path / "shots.json"
    path.write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "shot",
                        "start_frame": 0,
                        "end_frame_inclusive": 1,
                        "review_state": state,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit, _ = audit_shot_contract_v112a(
        path,
        candidates=(_candidate("a", "shot", [0, 1]),),
        frame_count=2,
    )
    assert audit["status"] == "FAIL"
    assert audit["unapproved_shot_ids"] == ["shot"]


def _safety_policy() -> dict:
    return json.loads(
        (
            PROJECT_ROOT
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_1/"
            "safety_policy.json"
        ).read_text(encoding="utf-8")
    )


def test_ball_reset_gap_pairs_are_excluded(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (100, 80),
    )
    for _ in range(17):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()
    frames = [0, 1, 15, 16]
    detections = tuple(
        Detection(
            frame=frame,
            label="ball",
            bbox_xyxy=(40.0 + frame, 60.0, 44.0 + frame, 64.0),
            confidence=0.95,
        )
        for frame in frames
    )
    policy = _safety_policy()
    selection = select_ball_trajectory_segments_by_shot(
        detections,
        intervals=(ShotInterval("shot", 0, 16),),
        width=100,
        height=80,
        policy=policy["ball_selection"],
    )
    assert selection.reset_count == 1
    assert selection.segment_ids[("shot", 1)] == 0
    assert selection.segment_ids[("shot", 15)] == 1
    extractor = RawFeatureExtractorV112a(
        shot_intervals=(ShotInterval("shot", 0, 16),),
        video_fps=5.0,
        canonical_event_label="goal",
        safety_policy=policy,
        width=100,
        height=80,
        event_scene_local_sec=3.0,
        event_window_before_sec=3.0,
        event_window_after_sec=3.0,
        closeup_area_ratio=0.1,
        ball_labels={"ball"},
        ball_low_coverage_threshold=0.2,
    )
    rows, _ = extractor.extract_all(
        candidates=(_candidate("candidate", "shot", frames),),
        detections=detections,
        video_path=video,
    )
    ball = rows[0]["raw_features"]["ball"]
    assert ball["selected_ball_observation_count"] == 4
    assert ball["valid_continuity_pair_count"] == 2
    assert ball["reset_gap_pair_count"] == 1
    assert ball["trajectory_continuity"] == 1.0


class _SnapshotStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve_path(self, value) -> Path:
        path = Path(value)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.root / path).resolve()
        )
        if not resolved.is_relative_to(self.root):
            raise ValueError("escape")
        return resolved


def test_frozen_discovery_root_is_used_and_stale_change_fails(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    manifest = old / "scene_candidate_manifest.json"
    manifest.write_text('{"version": 1}', encoding="utf-8")
    storage = _SnapshotStorage(tmp_path)
    revision = SimpleNamespace(
        options={
            "scene_target_selection": {
                "discovery_id": "discovery_old",
                "scene_id": "scene",
                "artifact_root": "old",
            }
        }
    )
    freeze = {
        "discovery_id": "discovery_old",
        "scene_id": "scene",
        "discovery_artifact_root": "old",
        "scene_candidate_manifest_sha256": sha256_file(manifest),
    }
    assert resolve_frozen_discovery_root(
        storage,
        revision=revision,
        freeze_material=freeze,
    ) == old.resolve()
    revision.options["scene_target_selection"] = {
        "discovery_id": "discovery_new",
        "scene_id": "scene",
        "artifact_root": "new",
    }
    with pytest.raises(
        StaleDiscoveryInputError,
        match="STALE_DISCOVERY_INPUT",
    ):
        resolve_frozen_discovery_root(
            storage,
            revision=revision,
            freeze_material=freeze,
        )
    assert not (new / "rankings").exists()


def test_adapter_run_stops_before_writing_new_discovery(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    manifest = old / "scene_candidate_manifest.json"
    manifest.write_text('{"version": 1}', encoding="utf-8")
    revision = SimpleNamespace(
        options={
            "scene_target_selection": {
                "discovery_id": "discovery_new",
                "scene_id": "scene",
                "artifact_root": "new",
            }
        }
    )

    class _DB:
        def get(self, model, identity):
            return revision

    from app.domains.highlight.event_candidate_ranking_v1_1_2a.backend_adapter import (
        EventCandidateRankingV112aBackendAdapter,
    )

    adapter = EventCandidateRankingV112aBackendAdapter.__new__(
        EventCandidateRankingV112aBackendAdapter
    )
    adapter.db = _DB()
    adapter.storage = _SnapshotStorage(tmp_path)
    with pytest.raises(
        StaleDiscoveryInputError,
        match="STALE_DISCOVERY_INPUT",
    ):
        adapter.run(
            project=SimpleNamespace(),
            user=SimpleNamespace(),
            revision_id="revision",
            shortlist_size=5,
            resolved_event={},
            freeze_material={
                "shortlist_size": 5,
                "discovery_id": "discovery_old",
                "scene_id": "scene",
                "discovery_artifact_root": "old",
                "scene_candidate_manifest_sha256": sha256_file(
                    manifest
                ),
            },
        )
    assert not (new / "rankings").exists()


def _cache_material() -> dict:
    return {
        "ranking_source_manifest_sha256": "1" * 64,
        "ranking_policy_sha256": "2" * 64,
        "safety_policy_sha256": "3" * 64,
        "contract_policy_sha256": "4" * 64,
        "input_schema_sha256": "5" * 64,
        "candidate_feature_schema_sha256": "6" * 64,
        "ranking_output_schema_sha256": "7" * 64,
        "current_scene_candidates_sha256": "8" * 64,
        "stored_discovery_scene_candidates_sha256": "8" * 64,
        "manifest_declared_scene_candidates_sha256": "8" * 64,
        "scene_candidate_manifest_sha256": "9" * 64,
        "detections_sha256": "a" * 64,
        "shot_boundaries_sha256": "b" * 64,
        "source_video_sha256": "c" * 64,
        "discovery_id": "discovery",
        "scene_id": "scene",
        "scene_start_sec": 1.0,
        "scene_end_sec": 5.0,
        "shortlist_size": 5,
        "event_id": "event",
        "event_time_sec": 3.0,
        "event_label": "Goal",
    }


@pytest.mark.parametrize(
    "field",
    [
        "detections_sha256",
        "discovery_id",
        "scene_id",
        "scene_start_sec",
        "scene_end_sec",
        "shortlist_size",
    ],
)
def test_cache_fingerprint_contains_hotfix_fields(field: str) -> None:
    material = _cache_material()
    baseline = cache_fingerprint_v112a(material)
    material[field] = (
        material[field] + "_changed"
        if isinstance(material[field], str)
        else material[field] + 1
    )
    assert cache_fingerprint_v112a(material) != baseline


class _TempStorage:
    def __init__(self, root: Path) -> None:
        self.project_root = root.resolve()
        self.storage_root = self.project_root

    def resolve_path(self, value) -> Path:
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )


def _ranking_document(version: str, policy: str) -> dict:
    return {
        "schema_version": f"kickclip.event_candidate_ranking.{version}",
        "package": (
            "target_centric_tracking_event_candidate_ranking_"
            + version
        ),
        "freeze": {
            "source_manifest_sha256": "a" * 64,
            "candidate_feature_schema_sha256": "b" * 64,
            "ranking_policy_sha256": policy,
        },
        "all_candidates": [
            {"candidate_id": "candidate_a"},
            {"candidate_id": "candidate_b"},
        ],
        "shortlist": [{"candidate_id": "candidate_a"}],
    }


def _add_ranking(
    db: Session,
    root: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    version: str,
    policy: str,
) -> Artifact:
    path = root / f"{artifact_id}.json"
    path.write_text(
        json.dumps(_ranking_document(version, policy)),
        encoding="utf-8",
    )
    artifact = Artifact(
        artifact_id=artifact_id,
        match_id="match",
        project_id="project",
        analysis_job_id=None,
        artifact_type=artifact_type,
        file_path=path.name,
        mime_type="application/json",
        metadata_={
            "sha256": sha256_file(path),
            "ranking_id": artifact_id,
            "owner_id": "user",
        },
    )
    db.add(artifact)
    db.flush()
    return artifact


def test_v1_1_2_annotation_e2e_and_cross_version_block(
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
        db.add_all(
            [
                user,
                Match(match_id="match", owner_id="user", metadata_={}),
                Project(
                    project_id="project",
                    match_id="match",
                    owner_id="user",
                    title="Project",
                    status="DRAFT",
                ),
            ]
        )
        db.flush()
        v112 = _add_ranking(
            db,
            tmp_path,
            artifact_id="ranking_v112",
            artifact_type=(
                "EVENT_CANDIDATE_RANKING_V1_1_2_SHADOW"
            ),
            version="v1_1_2",
            policy="c" * 64,
        )
        v111 = _add_ranking(
            db,
            tmp_path,
            artifact_id="ranking_v111",
            artifact_type=(
                "EVENT_CANDIDATE_RANKING_V1_1_1_SHADOW"
            ),
            version="v1_1_1",
            policy="c" * 64,
        )
        db.commit()
        service = EventAnnotationCompatibilityService(db)
        service.storage = _TempStorage(tmp_path)
        payload = EventAnnotationReviewRequest(
            primary_actor_visible=True,
            primary_actor_in_candidate_set=True,
            primary_actor_candidate_ids=["candidate_a"],
            directly_related_candidate_ids=["candidate_b"],
            candidate_roles={
                "candidate_a": "PRIMARY_EVENT_ACTOR"
            },
        )
        review_v112 = service.submit_review(
            ranking_artifact_id=v112.artifact_id,
            user=user,
            payload=payload,
        )
        review_v111 = service.submit_review(
            ranking_artifact_id=v111.artifact_id,
            user=user,
            payload=payload,
        )
        with pytest.raises(
            ValueError,
            match="ranking version does not match",
        ):
            service.finalize(
                ranking_artifact_id=v112.artifact_id,
                user=user,
                payload=EventAnnotationFinalizeRequest(
                    review_artifact_ids=[review_v111.artifact_id],
                    approval_mode="FINAL_APPROVED",
                    approved_reviewer="user",
                ),
            )
        final = service.finalize(
            ranking_artifact_id=v112.artifact_id,
            user=user,
            payload=EventAnnotationFinalizeRequest(
                review_artifact_ids=[review_v112.artifact_id],
                approval_mode="FINAL_APPROVED",
                approved_reviewer="user",
            ),
        )
        evaluation = service.evaluate(
            ranking_artifact_id=v112.artifact_id,
            annotation_artifact_id=final.artifact_id,
            user=user,
        )
        identity = evaluation["ranking_identity"]
        assert identity["ranking_package"].endswith("v1_1_2")
        assert identity["ranking_schema_version"].endswith("v1_1_2")
        assert identity["ranking_policy_sha256"] == "c" * 64
        assert evaluation["cross_version_metrics_mixed"] is False


def _valid_candidate_schema_row() -> dict:
    return {
        "candidate_id": "candidate",
        "shot_id": "shot",
        "rank": 1,
        "raw_features": {
            "temporal": {},
            "visual": {},
            "motion": {},
            "broadcast": {
                "local_post_event_focus": 0.5,
                "cross_shot_repeated_focus": None,
                "post_event_shots_with_appearance": None,
            },
            "ball": {
                "state": "AVAILABLE",
                "candidate_frame_coverage": 0.5,
                "selected_ball_observation_count": 2,
                "valid_continuity_pair_count": 1,
                "reset_gap_pair_count": 0,
                "continuity_passing_pair_count": 1,
                "trajectory_continuity": 1.0,
            },
            "field_context": {},
        },
        "feature_availability": {},
        "feature_evidence": {
            "ball_frames": [1, 2],
            "ball_shot_frames": [
                {
                    "shot_id": "shot",
                    "global_frame": 1,
                    "trajectory_segment_id": 0,
                }
            ],
        },
        "feature_completeness_score": 0.8,
        "critical_features_missing": [],
        "reliability_state": "READY",
        "event_relevance_score": 0.5,
        "recommendation_score": 0.5,
    }


def test_schema_tightens_ranges_frames_and_list_items() -> None:
    feature_schema = json.loads(
        (PACKAGE_ROOT / "candidate_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(feature_schema)
    validator.validate(_valid_candidate_schema_row())
    bad_coverage = _valid_candidate_schema_row()
    bad_coverage["raw_features"]["ball"][
        "candidate_frame_coverage"
    ] = 1.1
    with pytest.raises(ValidationError):
        validator.validate(bad_coverage)
    bad_frame = _valid_candidate_schema_row()
    bad_frame["feature_evidence"]["ball_frames"] = [1.5]
    with pytest.raises(ValidationError):
        validator.validate(bad_frame)
    output_schema = json.loads(
        (PACKAGE_ROOT / "ranking_output_schema.json").read_text(
            encoding="utf-8"
        )
    )
    output_validator = Draft202012Validator(output_schema)
    with pytest.raises(ValidationError):
        output_validator.validate(
            {
                "schema_version": (
                    "kickclip.event_candidate_ranking.v1_1_2a"
                ),
                "package": (
                    "target_centric_tracking_"
                    "event_candidate_ranking_v1_1_2a"
                ),
                "status": "PROVISIONAL_SHADOW_ONLY",
                "ranking_status": "PROVISIONAL_SHADOW_ONLY",
                "automatic_target_confirmation": False,
                "production_recommendation_ui": "BLOCKED",
                "event_context": {
                    "event_id": "e",
                    "event_label": "Goal",
                    "event_time_sec": 1,
                    "scene_id": "s",
                    "scene_start_sec": 0,
                    "scene_end_sec": 2,
                    "source": "SERVER_RESOLVED_TIMELINE_EVENT",
                    "client_event_fields_accepted": False,
                },
                "discovery_snapshot": {
                    "discovery_id": "d",
                    "discovery_artifact_root": "root",
                    "scene_id": "s",
                    "scene_candidate_manifest_sha256": "a" * 64,
                },
                "immutable_candidate_contract": {
                    "current_scene_candidates_sha256": "a" * 64,
                    "stored_discovery_scene_candidates_sha256": "a" * 64,
                    "manifest_declared_scene_candidates_sha256": "a" * 64,
                },
                "scene_video_duration_contract": {},
                "freeze": {},
                "full_gallery_fallback": [],
                "feature_completeness_state": "READY",
                "shortlist": [{}],
                "all_candidates": [],
                "shortlist_decisions": [
                    {
                        "candidate_id": "c",
                        "decision": "MAYBE",
                        "reason": "x",
                    }
                ],
            }
        )


def test_v1_1_2a_verifier_runs_tightened_schema_smoke() -> None:
    result = EventCandidateRankingV112aVerifier(PACKAGE_ROOT).check()
    assert result.event_ranking_compatibility_runtime_verified, (
        result.code,
        result.message,
    )
    assert result.full_event_recommendation_e2e_verified is False
