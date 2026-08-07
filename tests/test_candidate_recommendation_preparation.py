from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1.artifacts import sha256_file
from app.api.v1 import event_candidate_handoff_r1 as handoff_api
from app.domains.candidate_handoff_r1.errors import (
    CandidatePreparationError,
    CandidateRecommendationNotPrepared,
    CandidateSelectionProvenanceMismatch,
)
from app.domains.candidate_handoff_r1.preparation import (
    EventCandidateRecommendationPreparationService,
    PREPARATION_TASK_TYPE,
)
from app.domains.candidate_handoff_r1.review_bundle import (
    MEDIA_MATERIALIZATION_POLICY,
)
from app.domains.candidate_handoff_r1.schema import (
    CandidateRecommendationPrepareRequest,
)
from app.domains.candidate_handoff_r1.service import (
    BUNDLE_ARTIFACT_TYPE,
    CandidateHandoffR1Service,
)
from app.domains.highlight.event_candidate_ranking_v1_2 import (
    backend_adapter as v12_adapter_module,
)
from app.domains.highlight.event_candidate_ranking_v1_2.backend_adapter import (
    ARTIFACT_TYPE,
    EventCandidateRankingV12BackendAdapter,
)
from app.domains.highlight.model import HighlightRevision, SceneAITask
from app.domains.highlight.schema import HighlightSceneSelectionRequest
from app.domains.highlight import scene_ai_task as scene_task_module
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project


class TempStorage:
    def __init__(self, root: Path) -> None:
        self.project_root = root.resolve()
        self.storage_root = self.project_root

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )
        if not resolved.is_relative_to(self.storage_root):
            raise ValueError("path escape")
        return resolved


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
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
        session.add_all([user, match, project])
        session.commit()
        yield session


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)


def add_artifact(
    db: Session,
    root: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    metadata: dict,
    document: dict | None = None,
) -> Artifact:
    path = root / f"{artifact_id}.json"
    digest = write_json(path, document or {"artifact_id": artifact_id})
    artifact = Artifact(
        artifact_id=artifact_id,
        match_id="match",
        project_id="project",
        analysis_job_id=None,
        artifact_type=artifact_type,
        file_path=path.relative_to(root).as_posix(),
        mime_type="application/json",
        metadata_={**metadata, "sha256": digest},
    )
    db.add(artifact)
    db.flush()
    return artifact


@pytest.mark.parametrize(
    (
        "artifact_type",
        "boundary_origin",
        "human_reviewed",
        "status",
    ),
    [
        (
            "AUTO_SHOT_BOUNDARIES",
            "AUTO_DETECTED",
            False,
            "STRUCTURALLY_VALID",
        ),
        (
            "REVIEWED_SHOT_BOUNDARIES",
            "HUMAN_REVIEWED",
            True,
            "REVIEWED_PASS",
        ),
    ],
)
def test_tracking_boundary_provenance_accepts_auto_and_reviewed_without_aliasing(
    db: Session,
    tmp_path: Path,
    artifact_type: str,
    boundary_origin: str,
    human_reviewed: bool,
    status: str,
) -> None:
    video_sha = "a" * 64
    shot = {
        "shot_index": 0,
        "shot_id": "shot_0000",
        "start_frame": 0,
        "end_frame_inclusive": 9,
        "review_status": "REVIEWED_PASS" if human_reviewed else None,
    }
    document = {
        "artifact_type": artifact_type,
        "boundary_origin": boundary_origin,
        "human_reviewed": human_reviewed,
        "automatic_target_confirmation": False,
        "automatic_confirmation": False,
        "video": {"sha256": video_sha, "frame_count": 10},
        "shots": [shot],
        "structural_validation": {
            "status": "PASS",
            "complete_event_window_coverage": True,
            "gap_count": 0,
            "overlap_count": 0,
        },
    }
    artifact = add_artifact(
        db,
        tmp_path,
        artifact_id="boundaries",
        artifact_type=artifact_type,
        metadata={
            "revision_id": "revision",
            "event_id": "event",
            "scene_id": "scene",
            "status": status,
            "boundary_origin": boundary_origin,
            "human_reviewed": human_reviewed,
            "automatic_target_confirmation": False,
            "automatic_confirmation": False,
        },
        document=document,
    )
    db.commit()
    service = CandidateHandoffR1Service(db)
    service.storage = TempStorage(tmp_path)
    provenance = service._resolve_boundary_provenance(
        project=db.get(Project, "project"),
        revision_id="revision",
        event_id="event",
        scene_id="scene",
        source_video_sha256=video_sha,
        manifest={
            # Exercise legacy immutable selections as well as new manifests.
            "reviewed_shot_boundaries_sha256": artifact.metadata_["sha256"],
        },
    )
    assert provenance.artifact_type == artifact_type
    assert provenance.boundary_origin == boundary_origin
    assert provenance.human_reviewed is human_reviewed
    assert provenance.as_dict()["automatic_target_confirmation"] is False


def test_tracking_boundary_provenance_rejects_invalid_auto_structure(
    db: Session,
    tmp_path: Path,
) -> None:
    video_sha = "a" * 64
    artifact = add_artifact(
        db,
        tmp_path,
        artifact_id="invalid-auto-boundaries",
        artifact_type="AUTO_SHOT_BOUNDARIES",
        metadata={
            "revision_id": "revision",
            "event_id": "event",
            "scene_id": "scene",
            "status": "STRUCTURALLY_VALID",
            "boundary_origin": "AUTO_DETECTED",
            "human_reviewed": False,
            "automatic_target_confirmation": False,
        },
        document={
            "artifact_type": "AUTO_SHOT_BOUNDARIES",
            "boundary_origin": "AUTO_DETECTED",
            "human_reviewed": False,
            "automatic_target_confirmation": False,
            "automatic_confirmation": False,
            "video": {"sha256": video_sha, "frame_count": 10},
            "shots": [
                {
                    "shot_index": 0,
                    "shot_id": "shot_0000",
                    "start_frame": 1,
                    "end_frame_inclusive": 9,
                }
            ],
            "structural_validation": {
                "status": "PASS",
                "complete_event_window_coverage": True,
                "gap_count": 0,
                "overlap_count": 0,
            },
        },
    )
    db.commit()
    service = CandidateHandoffR1Service(db)
    service.storage = TempStorage(tmp_path)
    with pytest.raises(
        CandidateSelectionProvenanceMismatch,
        match="not contiguous full-frame coverage",
    ):
        service._resolve_boundary_provenance(
            project=db.get(Project, "project"),
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            source_video_sha256=video_sha,
            manifest={"shot_boundaries_sha256": artifact.metadata_["sha256"]},
        )


def test_v12_cache_is_scoped_to_exact_event_scene_and_inputs(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = (
        tmp_path / "configs/models/event_candidate_ranking/"
        "target_centric_tracking_event_candidate_ranking_v1_2"
    )
    for name in (
        "manifest.json",
        "shortlist_policy.json",
        "input_schema.json",
        "output_schema.json",
    ):
        write_json(package / name, {})

    class FakePatch:
        def __init__(self, _root: Path) -> None:
            pass

        def run(self, **kwargs):
            return {
                "schema_version": "kickclip.event_candidate_ranking.v1_2",
                "ranking_status": "PROVISIONAL_SHADOW_ONLY",
                "automatic_target_confirmation": False,
                "shortlist": [],
            }

    monkeypatch.setattr(
        v12_adapter_module, "EventCandidateRankingV12ShortlistPatch", FakePatch
    )
    storage = TempStorage(tmp_path)
    project = db.get(Project, "project")
    user = db.get(User, "user")
    assert project is not None and user is not None
    sources = []
    shots = []
    for index in (1, 2):
        sources.append(
            add_artifact(
                db,
                tmp_path,
                artifact_id=f"source_{index}",
                artifact_type="EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW",
                metadata={
                    "revision_id": "revision",
                    "event_id": f"event_{index}",
                    "scene_id": f"scene_{index}",
                    "ranking_id": f"ranking_{index}",
                    "owner_id": "user",
                },
            )
        )
        shots.append(
            add_artifact(
                db,
                tmp_path,
                artifact_id=f"shots_{index}",
                artifact_type="REVIEWED_SHOT_BOUNDARIES",
                metadata={"review_state": "REVIEWED_PASS"},
            )
        )
    db.commit()
    adapter = EventCandidateRankingV12BackendAdapter(db)
    adapter.storage = storage
    adapter.package_root = package
    first = adapter.run(
        project=project,
        user=user,
        source_ranking_artifact_id=sources[0].artifact_id,
        reviewed_shots_artifact_id=shots[0].artifact_id,
    )
    second = adapter.run(
        project=project,
        user=user,
        source_ranking_artifact_id=sources[1].artifact_id,
        reviewed_shots_artifact_id=shots[1].artifact_id,
    )
    repeated = adapter.run(
        project=project,
        user=user,
        source_ranking_artifact_id=sources[0].artifact_id,
        reviewed_shots_artifact_id=shots[0].artifact_id,
    )
    assert first.artifact_id != second.artifact_id
    assert repeated.artifact_id == first.artifact_id
    assert (
        db.scalar(
            select(func.count())
            .select_from(Artifact)
            .where(Artifact.artifact_type == ARTIFACT_TYPE)
        )
        == 2
    )


def preparation_fixture(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_candidate: str | None = None,
    use_real_builder: bool = False,
):
    storage = TempStorage(tmp_path)
    source = add_artifact(
        db,
        tmp_path,
        artifact_id="source",
        artifact_type="EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW",
        metadata={
            "revision_id": "revision",
            "event_id": "event",
            "scene_id": "scene",
            "status": "PROVISIONAL_SHADOW_ONLY",
            "ranking_id": "ranking-source",
        },
    )
    # Newer distractors must never be selected for another event/scene.
    add_artifact(
        db,
        tmp_path,
        artifact_id="source-other-event",
        artifact_type="EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW",
        metadata={
            "revision_id": "revision",
            "event_id": "other-event",
            "scene_id": "other-scene",
            "status": "PROVISIONAL_SHADOW_ONLY",
            "ranking_id": "wrong-ranking",
        },
    )
    reviewed = add_artifact(
        db,
        tmp_path,
        artifact_id="reviewed",
        artifact_type="REVIEWED_SHOT_BOUNDARIES",
        metadata={
            "revision_id": "revision",
            "scene_id": "scene",
            "review_state": "REVIEWED_PASS",
        },
        document={"shots": []},
    )
    candidates_path = tmp_path / "scene_candidates.json"
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "shot_id": f"shot-{index}",
            "local_tracklet_id": f"track-{index}",
            "observations": [
                {
                    "global_frame": 10 + index,
                    "scene_local_frame": 2 + index,
                    "bbox_xyxy": [1, 2, 20, 40],
                    "detector_confidence": 0.9,
                }
            ],
        }
        for index, candidate_id in enumerate(("candidate-a", "candidate-b"))
    ]
    write_json(candidates_path, {"candidates": candidate_rows})
    video_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (100, 80)
    )
    for frame_index in range(20):
        frame = np.full((80, 100, 3), 20, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (1, 2),
            (20, 40),
            (80 + frame_index, 180, 240),
            -1,
        )
        writer.write(frame)
    writer.release()
    immutable = SimpleNamespace(
        scene_candidates_path=candidates_path,
        source_video_path=video_path,
        source_video_sha256=sha256_file(video_path),
        candidate_manifest_sha256="c" * 64,
    )

    class FakeV12Adapter:
        def __init__(self, session: Session) -> None:
            self.db = session

        def run(self, **kwargs) -> Artifact:
            existing = self.db.scalar(
                select(Artifact).where(Artifact.artifact_type == ARTIFACT_TYPE)
            )
            if existing is not None:
                return existing
            path = tmp_path / "v12.json"
            digest = write_json(
                path,
                {
                    "schema_version": "kickclip.event_candidate_ranking.v1_2",
                    "automatic_target_confirmation": False,
                    "shortlist": [
                        {
                            "candidate_id": row["candidate_id"],
                            "shot_id": row["shot_id"],
                            "rank": index,
                            "original_global_rank": index,
                        }
                        for index, row in enumerate(candidate_rows, start=1)
                    ],
                },
            )
            artifact = Artifact(
                artifact_id="ranking-v12",
                match_id="match",
                project_id="project",
                analysis_job_id=None,
                artifact_type=ARTIFACT_TYPE,
                file_path=path.relative_to(tmp_path).as_posix(),
                mime_type="application/json",
                metadata_={
                    "ranking_id": "patch-v12",
                    "revision_id": "revision",
                    "event_id": "event",
                    "scene_id": "scene",
                    "source_ranking_artifact_id": source.artifact_id,
                    "reviewed_shots_artifact_id": reviewed.artifact_id,
                    "cache_key": "k" * 64,
                    "sha256": digest,
                    "automatic_target_confirmation": False,
                },
            )
            self.db.add(artifact)
            self.db.flush()
            return artifact

    offsets: list[int] = []
    failed_once: set[str] = set()

    def fake_build(**kwargs):
        candidate = kwargs["candidate"]
        if (
            candidate["candidate_id"] == fail_candidate
            and candidate["candidate_id"] not in failed_once
        ):
            failed_once.add(candidate["candidate_id"])
            raise ValueError("forced bundle failure")
        offsets.append(kwargs["frame_offset"])
        root = kwargs["output_root"]
        root.mkdir(parents=True)
        manifest = {
            "schema_version": "kickclip.candidate_review_bundle.r1",
            "candidate_id": candidate["candidate_id"],
            "candidate_media_id": candidate["candidate_id"],
            "ranking_id": kwargs["ranking_id"],
            "shortlist_patch_id": kwargs["shortlist_patch_id"],
            "shot_id": candidate["shot_id"],
            "tracklet_id": candidate["tracklet_id"],
            "source_video_sha256": kwargs["source_video_sha256"],
            "candidate_manifest_sha256": kwargs["candidate_manifest_sha256"],
            "shot_boundaries_artifact_id": kwargs["shot_boundaries_artifact_id"],
            "shot_boundaries_sha256": kwargs["shot_boundaries_sha256"],
            "shot_boundary_artifact_type": kwargs["shot_boundary_artifact_type"],
            "boundary_origin": kwargs["boundary_origin"],
            "human_reviewed": kwargs["human_reviewed"],
            "media_materialization_policy": MEDIA_MATERIALIZATION_POLICY,
            "candidate_grouping_policy": candidate["candidate_grouping_policy"],
            "candidate_grouping_sha256": candidate["candidate_grouping_sha256"],
            "candidate_group_id": candidate["candidate_group_id"],
            "group_member_fingerprint": candidate["group_member_fingerprint"],
            "group_member_candidate_ids": candidate["group_member_candidate_ids"],
            "grouping_is_identity_confirmation": False,
            "quality": {
                "reviewability": "USABLE",
                "selected_best_frame": 10,
                "purity_diagnostics": {},
            },
            "files": {},
            "automatic_target_confirmation": False,
        }
        digest = write_json(root / "manifest.json", manifest)
        return manifest, digest

    monkeypatch.setattr(
        "app.domains.candidate_handoff_r1.preparation.EventCandidateRankingV12BackendAdapter",
        FakeV12Adapter,
    )
    if not use_real_builder:
        monkeypatch.setattr(
            "app.domains.candidate_handoff_r1.preparation.build_candidate_review_bundle",
            fake_build,
        )
    service = EventCandidateRecommendationPreparationService(db)
    service.storage = storage
    service._immutable_input = MethodType(
        lambda self, artifact: (
            immutable,
            {"discovery_id": "discovery", "source_artifact_id": artifact.artifact_id},
        ),
        service,
    )
    db.commit()
    return service, offsets


def test_prepare_builds_every_bundle_registers_artifacts_and_get_is_read_only(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, offsets = preparation_fixture(db, tmp_path, monkeypatch)
    project = db.get(Project, "project")
    user = db.get(User, "user")
    assert project is not None and user is not None
    result = service.prepare(
        project=project,
        user=user,
        revision_id="revision",
        event_id="event",
        scene_id="scene",
        shortlist_size=5,
    )
    assert result["ready"] is True
    assert result["candidate_count"] == 2
    assert result["review_bundle_count"] == 2
    assert result["automatic_target_confirmation"] is False
    assert offsets == [8, 8]
    bundles = db.scalars(
        select(Artifact).where(Artifact.artifact_type == BUNDLE_ARTIFACT_TYPE)
    ).all()
    assert len(bundles) == 2
    assert all((row.metadata_ or {})["status"] == "READY" for row in bundles)
    artifact_count = db.scalar(select(func.count()).select_from(Artifact))
    repeated = service.prepare(
        project=project,
        user=user,
        revision_id="revision",
        event_id="event",
        scene_id="scene",
        shortlist_size=5,
    )
    assert {
        key: value
        for key, value in repeated.items()
        if key not in {"work_metrics", "work_metrics_path", "work_metrics_sha256"}
    } == {
        key: value
        for key, value in result.items()
        if key not in {"work_metrics", "work_metrics_path", "work_metrics_sha256"}
    }
    assert offsets == [8, 8]
    assert db.scalar(select(func.count()).select_from(Artifact)) == artifact_count
    before = db.scalar(select(func.count()).select_from(Artifact))
    handoff = CandidateHandoffR1Service(db)
    handoff.storage = TempStorage(tmp_path)
    recommendations = handoff.list_recommendations(
        project=project,
        revision_id="revision",
        event_id="event",
        scene_id="scene",
    )
    after = db.scalar(select(func.count()).select_from(Artifact))
    assert len(recommendations.candidates) == 2
    assert before == after


def test_prepare_is_not_ready_when_one_bundle_fails(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = preparation_fixture(
        db, tmp_path, monkeypatch, fail_candidate="candidate-b"
    )
    project = db.get(Project, "project")
    user = db.get(User, "user")
    assert project is not None and user is not None
    with pytest.raises(ValueError, match="forced bundle failure"):
        service.prepare(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            shortlist_size=5,
        )
    handoff = CandidateHandoffR1Service(db)
    handoff.storage = TempStorage(tmp_path)
    with pytest.raises(CandidateRecommendationNotPrepared) as caught:
        handoff.list_recommendations(
            project=project,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
        )
    assert caught.value.reason.startswith(
        ("REVIEW_BUNDLE_MISSING", "REVIEW_BUNDLE_PROVENANCE_MISMATCH")
    )
    retried = service.prepare(
        project=project,
        user=user,
        revision_id="revision",
        event_id="event",
        scene_id="scene",
        shortlist_size=5,
    )
    assert retried["ready"] is True
    assert retried["review_bundle_count"] == 2


def test_prepare_uses_real_review_bundle_builder_and_writes_media(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = preparation_fixture(db, tmp_path, monkeypatch, use_real_builder=True)
    project = db.get(Project, "project")
    user = db.get(User, "user")
    assert project is not None and user is not None
    result = service.prepare(
        project=project,
        user=user,
        revision_id="revision",
        event_id="event",
        scene_id="scene",
        shortlist_size=5,
    )
    assert result["ready"] is True
    bundles = db.scalars(
        select(Artifact).where(Artifact.artifact_type == BUNDLE_ARTIFACT_TYPE)
    ).all()
    assert len(bundles) == 2
    for artifact in bundles:
        manifest_path = tmp_path / artifact.file_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["automatic_target_confirmation"] is False
        assert manifest["frame_mapping"]["candidate_source_to_tracking_offset"] == 8
        for record in manifest["files"].values():
            media_path = manifest_path.parent / record["path"]
            assert media_path.is_file()
            assert sha256_file(media_path) == record["sha256"]


def test_reviewed_shot_sha_mismatch_is_blocked(db: Session, tmp_path: Path) -> None:
    service = EventCandidateRecommendationPreparationService(db)
    service.storage = TempStorage(tmp_path)
    artifact = add_artifact(
        db,
        tmp_path,
        artifact_id="reviewed",
        artifact_type="REVIEWED_SHOT_BOUNDARIES",
        metadata={
            "revision_id": "revision",
            "scene_id": "scene",
            "review_state": "REVIEWED_PASS",
        },
    )
    db.commit()
    (tmp_path / artifact.file_path).write_text("changed", encoding="utf-8")
    project = db.get(Project, "project")
    assert project is not None
    with pytest.raises(CandidatePreparationError) as caught:
        service._latest_reviewed_shots(
            project=project, revision_id="revision", scene_id="scene"
        )
    assert caught.value.code == "REVIEWED_SHOT_BOUNDARIES_NOT_READY"


def test_source_video_sha_mismatch_and_frame_zero_guess_are_blocked(
    db: Session, tmp_path: Path
) -> None:
    service = EventCandidateRecommendationPreparationService(db)
    service.storage = TempStorage(tmp_path)
    source = SimpleNamespace(metadata_={"freeze_material": {}})
    material = {
        "storage_root": str(tmp_path),
        "scene_candidates_relative_path": "candidates.json",
        "scene_candidates_sha256": "a" * 64,
        "detections_relative_path": "detections.json",
        "detections_sha256": "b" * 64,
        "source_video_relative_path": "source.mp4",
        "source_video_sha256": "0" * 64,
        "shot_boundaries_relative_path": "shots.json",
        "shot_boundaries_sha256": "d" * 64,
        "candidate_manifest_sha256": "e" * 64,
        "video_width": 10,
        "video_height": 10,
        "video_fps": 5.0,
        "video_frame_count": 1,
        "discovery_id": "discovery",
        "discovery_artifact_root": "discovery",
    }
    source.metadata_["freeze_material"] = material
    (tmp_path / "source.mp4").write_bytes(b"changed-video")
    with pytest.raises(CandidatePreparationError) as caught:
        service._immutable_input(source)
    assert caught.value.code == "SOURCE_VIDEO_NOT_READY"
    with pytest.raises(CandidatePreparationError) as frame_error:
        service._bundle_candidate(
            {
                "candidate_id": "dynamic",
                "shot_id": "shot",
                "local_tracklet_id": "track",
                "observations": [{"global_frame": 10, "bbox_xyxy": [1, 1, 2, 2]}],
            },
            discovery_id="discovery",
        )
    assert frame_error.value.code == "FRAME_MAPPING_NOT_READY"


def test_scene_local_observations_require_explicit_frame_mapping_contract(
    db: Session, tmp_path: Path
) -> None:
    service = EventCandidateRecommendationPreparationService(db)
    candidate = {
        "candidate_id": "dynamic",
        "shot_id": "shot",
        "local_tracklet_id": "track",
        "observations": [
            {"frame_index": 7, "bbox_xyxy": [1, 1, 20, 30], "confidence": 0.9}
        ],
    }
    with pytest.raises(CandidatePreparationError) as caught:
        service._bundle_candidate(candidate, discovery_id="discovery")
    assert caught.value.code == "FRAME_MAPPING_NOT_READY"

    mapped, video_offset = service._bundle_candidate(
        candidate,
        discovery_id="discovery",
        frame_offset_contract=65063,
        video_frame_offset_contract=0,
    )
    assert mapped["observations"][0]["frame_index"] == 7
    assert video_offset == 0


def test_prepare_api_enqueues_scene_ai_task_and_get_returns_prepare_contract(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeExecutor:
        submitted: list[str] = []

        def submit(self, task_id: str) -> bool:
            self.submitted.append(task_id)
            return True

    executor = FakeExecutor()
    monkeypatch.setattr(handoff_api, "get_scene_ai_task_executor", lambda: executor)

    class FakeBoundaryService:
        def __init__(self, _db: Session) -> None:
            pass

        @staticmethod
        def prepare_candidate_discovery_inputs(**_kwargs):
            return {
                "shot_boundaries_artifact_id": "auto-boundaries",
                "shot_boundaries_sha256": "a" * 64,
                "detections_artifact_id": "sampled-detections",
                "boundary_origin": "AUTO_DETECTED",
            }

    monkeypatch.setattr(handoff_api, "ShotBoundaryReviewService", FakeBoundaryService)
    user = db.get(User, "user")
    assert user is not None
    response = handoff_api.prepare_event_candidate_recommendations(
        project_id="project",
        revision_id="revision",
        event_id="event",
        scene_id="scene",
        payload=CandidateRecommendationPrepareRequest(shortlist_size=5),
        db=db,
        current_user=user,
    )
    assert response.status == "QUEUED"
    assert response.task_type == PREPARATION_TASK_TYPE
    assert executor.submitted == [response.task_id]
    task = db.get(SceneAITask, response.task_id)
    assert task is not None
    assert task.payload == {
        "revision_id": "revision",
        "event_id": "event",
        "scene_id": "scene",
        "shortlist_size": 5,
        "shot_boundaries_artifact_id": "auto-boundaries",
        "shot_boundaries_sha256": "a" * 64,
        "detections_artifact_id": "sampled-detections",
        "boundary_origin": "AUTO_DETECTED",
        "automatic_target_confirmation": False,
    }
    prepare_route = next(
        route
        for route in handoff_api.router.routes
        if route.path.endswith("candidate-recommendations/prepare")
    )
    assert prepare_route.status_code == 202

    expected_task_result = {
        "ready": True,
        "ranking_version": "v1.2",
        "ranking_artifact_id": "ranking-v12",
        "ranking_id": "ranking-source",
        "shortlist_patch_id": "patch-v12",
        "candidate_count": 2,
        "review_bundle_count": 2,
        "recommendations_url": "/recommendations",
        "automatic_target_confirmation": False,
    }

    class FakePreparationService:
        def __init__(self, _db: Session) -> None:
            pass

        def prepare(self, **_kwargs):
            return expected_task_result

    monkeypatch.setattr(
        "app.domains.candidate_handoff_r1.preparation."
        "EventCandidateRecommendationPreparationService",
        FakePreparationService,
    )
    monkeypatch.setattr(
        scene_task_module,
        "SessionLocal",
        lambda: Session(db.get_bind()),
    )
    scene_task_module.SceneAITaskExecutor()._execute(response.task_id)
    db.expire_all()
    completed = db.get(SceneAITask, response.task_id)
    assert completed is not None
    assert completed.status == "COMPLETED"
    assert completed.attempt_count == 1
    assert completed.result == expected_task_result

    with pytest.raises(HTTPException) as caught:
        handoff_api.list_event_candidate_recommendations(
            project_id="project",
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            db=db,
            current_user=user,
        )
    assert caught.value.status_code == 409
    assert caught.value.detail == {
        "code": "CANDIDATE_RECOMMENDATION_NOT_PREPARED",
        "message": "V1.2 ranking or candidate review bundles are not ready.",
        "prepare_url": (
            "/api/v1/projects/project/highlight/revisions/revision/events/"
            "event/candidate-recommendations/prepare?scene_id=scene"
        ),
        "reason": "V1_2_RANKING_MISSING",
        "preparation_task": {
            "task_id": response.task_id,
            "status": "COMPLETED",
            "status_url": f"/api/v1/scene-ai-tasks/{response.task_id}",
            "error_message": None,
        },
    }

    completed.status = "RUNNING"
    db.commit()
    with pytest.raises(HTTPException) as running_error:
        handoff_api.list_event_candidate_recommendations(
            project_id="project",
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            db=db,
            current_user=user,
        )
    assert running_error.value.detail["reason"] == "PREPARATION_RUNNING"
    assert running_error.value.detail["message"] == (
        "Candidate recommendation preparation is running."
    )
    assert running_error.value.detail["preparation_task"]["status"] == "RUNNING"


def test_scene_selection_exposes_automatic_discovery_readiness_without_review_gate(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = HighlightRevision(
        revision_id="child-revision",
        project_id="project",
        revision_number=1,
        user_request="test",
        structured_request={},
        selected_scene_ids=["event-scene"],
        scene_selection=[],
        focus_mode="NONE",
        status="SCENES_SELECTED",
        options={},
    )
    db.add(revision)
    db.commit()

    class FakeWorkflow:
        def __init__(self, _db: Session) -> None:
            pass

        def select_scenes(self, **_kwargs):
            return revision

        @staticmethod
        def revision_read(value):
            return value

    class FakeExecutor:
        submitted: list[str] = []

        def submit(self, task_id: str) -> bool:
            self.submitted.append(task_id)
            return True

    executor = FakeExecutor()
    monkeypatch.setattr(handoff_api, "HighlightWorkflowService", FakeWorkflow)
    monkeypatch.setattr(handoff_api, "get_scene_ai_task_executor", lambda: executor)
    user = db.get(User, "user")
    assert user is not None
    returned = handoff_api.select_highlight_scenes_and_prepare(
        project_id="project",
        payload=HighlightSceneSelectionRequest(scene_ids=["event-scene"]),
        db=db,
        current_user=user,
    )
    assert returned is revision
    task = db.scalar(
        select(SceneAITask).where(SceneAITask.task_type == PREPARATION_TASK_TYPE)
    )
    assert task is None
    assert executor.submitted == []
    readiness = revision.options["candidate_discovery"]["shot_boundary_readiness"][0]
    assert readiness["status"] == "READY_FOR_AUTOMATIC_CANDIDATE_DISCOVERY"
    assert readiness["boundary_origin"] == "AUTO_DETECTED"
    assert readiness["human_reviewed"] is False
    assert (
        revision.options["candidate_discovery"]["automatic_target_confirmation"]
        is False
    )


def test_selection_uses_revision_canonical_video_instead_of_stale_match_video(
    db: Session,
) -> None:
    raw = MediaAsset(
        asset_id="raw-video",
        match_id="match",
        asset_type="RAW_VIDEO",
        file_path="raw.mp4",
        original_filename="raw.mp4",
        mime_type="video/mp4",
        sha256="a" * 64,
    )
    canonical = MediaAsset(
        asset_id="scene-video",
        match_id="match",
        asset_type="HIGHLIGHT_SCENE_CLIP",
        file_path="scene.mp4",
        original_filename="scene.mp4",
        mime_type="video/mp4",
        sha256="b" * 64,
    )
    revision = HighlightRevision(
        revision_id="canonical-revision",
        project_id="project",
        revision_number=1,
        user_request="test",
        structured_request={},
        selected_scene_ids=["scene"],
        scene_selection=[],
        focus_mode="NONE",
        status="PLAYER_SELECTION_REQUIRED",
        options={
            "candidate_pipeline_inputs": {
                "scene_id": "scene",
                "scene_video_asset_id": canonical.asset_id,
            }
        },
    )
    db.add_all([raw, canonical, revision])
    db.commit()
    user = db.get(User, "user")
    assert user is not None
    resolved = handoff_api._canonical_candidate_video(
        db,
        project_id="project",
        revision_id=revision.revision_id,
        scene_id="scene",
        requested_asset_id=raw.asset_id,
        user=user,
    )
    assert resolved.asset_id == canonical.asset_id
