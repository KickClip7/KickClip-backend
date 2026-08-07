from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api.v1 import event_candidate_handoff_r1 as handoff_api
from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.analysis.model import AnalysisJob
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_2.contract import (
    load_reviewed_shots,
)
from app.domains.highlight.model import HighlightRevision
from app.domains.highlight.player_detector import PlayerDetection
from app.domains.highlight.scene_ai_task import SceneAITaskService
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.shot_boundary.model import ShotBoundaryReviewDecision
from app.domains.shot_boundary.runtime_contract import (
    validate_scene_target_selection_contract,
)
from app.domains.shot_boundary.schema import (
    ShotBoundaryConfirmRequest,
    ShotBoundaryDraftUpdate,
)
from app.domains.shot_boundary.service import (
    DETECTION_BBOX_GEOMETRY_CONTRACT_VERSION,
    FAST_CANDIDATE_DETECTION_POLICY_VERSION,
    ShotBoundaryReviewService,
    ShotBoundaryWorkflowError,
    _detections_csv_has_valid_geometry,
    _normalize_detection_bbox,
)
from app.domains.timeline.model import TimelineEvent


def test_detection_bbox_normalization_uses_inclusive_pixel_bounds() -> None:
    assert _normalize_detection_bbox(
        [-3.25, -4.5, 1919.741699, 1080.25],
        width=1920,
        height=1080,
    ) == (0.0, 0.0, 1919.0, 1079.0)
    assert (
        _normalize_detection_bbox(
            [1920.1, 20, 1930, 40],
            width=1920,
            height=1080,
        )
        is None
    )
    assert (
        _normalize_detection_bbox(
            [10, 10, float("nan"), 20],
            width=1920,
            height=1080,
        )
        is None
    )


def test_cached_detection_csv_requires_same_geometry_contract(tmp_path: Path) -> None:
    cached = tmp_path / "detections.csv"
    cached.write_text(
        "x1,y1,x2,y2\n0,0,1919,1079\n",
        encoding="utf-8",
    )
    assert _detections_csv_has_valid_geometry(cached, width=1920, height=1080)

    cached.write_text(
        "x1,y1,x2,y2\n0,0,1919.741699,1079\n",
        encoding="utf-8",
    )
    assert not _detections_csv_has_valid_geometry(
        cached,
        width=1920,
        height=1080,
    )

    cached.write_text(
        "x1,y1,x2,y2\n1919,0,1919,1079\n",
        encoding="utf-8",
    )
    assert not _detections_csv_has_valid_geometry(
        cached,
        width=1920,
        height=1080,
    )


def test_complete_frame_coverage_contract() -> None:
    valid = [
        {"shot_id": "shot_0000", "start_frame": 0, "end_frame_inclusive": 9},
        {"shot_id": "shot_0001", "start_frame": 10, "end_frame_inclusive": 24},
    ]
    assert ShotBoundaryReviewService.validate_shots(valid, frame_count=25) == []


@pytest.mark.parametrize(
    ("shots", "frame_count", "expected_indexes"),
    [
        (
            [{"shot_id": "only", "start_frame": 0, "end_frame_inclusive": 9}],
            10,
            [0],
        ),
        (
            [
                {"shot_id": "third", "start_frame": 20, "end_frame_inclusive": 29},
                {"shot_id": "first", "start_frame": 0, "end_frame_inclusive": 9},
                {"shot_id": "second", "start_frame": 10, "end_frame_inclusive": 19},
            ],
            30,
            [0, 1, 2],
        ),
    ],
)
def test_canonical_shot_indexes_follow_frame_order(
    shots: list[dict], frame_count: int, expected_indexes: list[int]
) -> None:
    canonical = ShotBoundaryReviewService.canonicalize_shots(
        shots, frame_count=frame_count
    )
    assert [row["shot_index"] for row in canonical] == expected_indexes
    assert [row["start_frame"] for row in canonical] == sorted(
        row["start_frame"] for row in shots
    )
    assert canonical[0]["cut_in_frame"] is None
    assert canonical[-1]["cut_out_frame"] is None
    assert [row["frame_count"] for row in canonical] == [
        row["end_frame_inclusive"] - row["start_frame"] + 1 for row in canonical
    ]
    assert (
        ShotBoundaryReviewService.validate_shots(
            canonical,
            frame_count=frame_count,
            require_shot_indexes=True,
        )
        == []
    )


def test_installed_frozen_runtime_accepts_canonical_reviewed_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "reviewed.json"
    artifact.write_text(
        json.dumps(
            {
                "video": {"sha256": "a" * 64, "frame_count": 10},
                "automatic_confirmation": False,
                "shots": [
                    {
                        "shot_index": 0,
                        "shot_id": "not-an-index-source",
                        "start_frame": 0,
                        "end_frame_inclusive": 9,
                        "review_state": "REVIEWED_PASS",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    validate_scene_target_selection_contract(
        package_root=(
            Path(__file__).parents[1]
            / ".tracking-runtime/target_centric_tracking_scene_target_selection_v1"
        ),
        artifact_path=artifact,
        video_sha256="a" * 64,
        frame_count=10,
    )


def test_installed_frozen_runtime_accepts_structurally_valid_automatic_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "automatic.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": "AUTO_SHOT_BOUNDARIES",
                "boundary_origin": "AUTO_DETECTED",
                "human_reviewed": False,
                "automatic_target_confirmation": False,
                "automatic_confirmation": False,
                "video": {"sha256": "b" * 64, "frame_count": 10},
                "structural_validation": {"status": "PASS"},
                "shots": [
                    {
                        "shot_index": 0,
                        "shot_id": "auto-shot",
                        "start_frame": 0,
                        "end_frame_inclusive": 9,
                        "boundary_state": "AUTO_DETECTED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    validate_scene_target_selection_contract(
        package_root=(
            Path(__file__).parents[1]
            / ".tracking-runtime/target_centric_tracking_scene_target_selection_v1"
        ),
        artifact_path=artifact,
        video_sha256="b" * 64,
        frame_count=10,
    )


def test_v12_ranking_consumes_auto_boundaries_without_review_state() -> None:
    document = {
        "artifact_type": "AUTO_SHOT_BOUNDARIES",
        "boundary_origin": "AUTO_DETECTED",
        "human_reviewed": False,
        "automatic_target_confirmation": False,
        "structural_validation": {"status": "PASS"},
        "video": {"frame_count": 10},
        "shots": [
            {
                "shot_index": 0,
                "shot_id": "auto-shot",
                "start_frame": 0,
                "end_frame_inclusive": 9,
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
            }
        ],
    }
    assert [row.shot_id for row in load_reviewed_shots(document)] == ["auto-shot"]
    disguised = json.loads(json.dumps(document))
    disguised["shots"][0]["review_state"] = "REVIEWED_PASS"
    with pytest.raises(ValueError, match="impersonate"):
        load_reviewed_shots(disguised)


@pytest.mark.parametrize(
    ("indexes", "message"),
    [(["0", 1], "int"), ([0, 0], "unique"), ([1, 0], "contiguous")],
)
def test_confirm_validator_rejects_invalid_shot_index_contract(
    indexes: list[object], message: str
) -> None:
    shots = [
        {
            "shot_id": "first",
            "shot_index": indexes[0],
            "start_frame": 0,
            "end_frame_inclusive": 4,
        },
        {
            "shot_id": "second",
            "shot_index": indexes[1],
            "start_frame": 5,
            "end_frame_inclusive": 9,
        },
    ]
    errors = ShotBoundaryReviewService.validate_shots(
        shots, frame_count=10, require_shot_indexes=True
    )
    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("shots", "message"),
    [
        ([{"shot_id": "a", "start_frame": 1, "end_frame_inclusive": 9}], "first shot"),
        (
            [
                {"shot_id": "a", "start_frame": 0, "end_frame_inclusive": 8},
                {"shot_id": "b", "start_frame": 10, "end_frame_inclusive": 19},
            ],
            "gap",
        ),
        (
            [
                {"shot_id": "a", "start_frame": 0, "end_frame_inclusive": 10},
                {"shot_id": "b", "start_frame": 10, "end_frame_inclusive": 19},
            ],
            "overlap",
        ),
        (
            [
                {"shot_id": "same", "start_frame": 0, "end_frame_inclusive": 9},
                {"shot_id": "same", "start_frame": 10, "end_frame_inclusive": 19},
            ],
            "unique",
        ),
    ],
)
def test_invalid_confirmation_coverage_is_rejected(
    shots: list[dict], message: str
) -> None:
    errors = ShotBoundaryReviewService.validate_shots(shots, frame_count=20)
    assert any(message in error for error in errors)


def test_draft_contract_cannot_request_automatic_confirmation() -> None:
    payload = ShotBoundaryDraftUpdate.model_validate(
        {
            "draft_revision": 3,
            "shots": [
                {"shot_id": "shot_0000", "start_frame": 0, "end_frame_inclusive": 9}
            ],
            "note": "human edit",
            "automatic_confirmation": True,
        }
    )
    assert "automatic_confirmation" not in payload.model_dump()


def test_candidate_preparation_does_not_require_human_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeReviewService:
        def __init__(self, _db) -> None:
            pass

        @staticmethod
        def prepare_candidate_discovery_inputs(**_kwargs):
            return {
                "shot_boundaries_artifact_id": "auto-boundaries",
                "shot_boundaries_sha256": "a" * 64,
                "detections_artifact_id": "sampled-detections",
                "boundary_origin": "AUTO_DETECTED",
            }

    monkeypatch.setattr(handoff_api, "ShotBoundaryReviewService", FakeReviewService)
    contract = handoff_api._require_candidate_discovery_inputs(
        object(),
        project=SimpleNamespace(project_id="project"),
        user=SimpleNamespace(user_id="user"),
        revision_id="revision",
        event_id="event",
        scene_id="scene",
    )
    assert contract["boundary_origin"] == "AUTO_DETECTED"
    assert contract["shot_boundaries_artifact_id"] == "auto-boundaries"


def test_candidate_preparation_opens_review_only_when_automatic_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeReviewService:
        def __init__(self, _db) -> None:
            pass

        @staticmethod
        def prepare_candidate_discovery_inputs(**_kwargs):
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_REVIEW_REQUIRED",
                "invalid",
                detail={"reason": "AUTOMATIC_BOUNDARY_STRUCTURAL_GATE_FAILED"},
            )

    monkeypatch.setattr(handoff_api, "ShotBoundaryReviewService", FakeReviewService)
    with pytest.raises(HTTPException) as caught:
        handoff_api._require_candidate_discovery_inputs(
            object(),
            project=SimpleNamespace(project_id="project"),
            user=SimpleNamespace(user_id="user"),
            revision_id="revision",
            event_id="event",
            scene_id="scene",
        )
    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "SHOT_BOUNDARY_REVIEW_REQUIRED"
    assert caught.value.detail["detail"]["reason"] == (
        "AUTOMATIC_BOUNDARY_STRUCTURAL_GATE_FAILED"
    )
    assert caught.value.detail["automatic_target_confirmation"] is False


class _TempStorage:
    def __init__(self, root: Path) -> None:
        self.project_root = root.resolve()
        self.storage_root = root.resolve()

    def resolve_path(self, value) -> Path:
        path = Path(value)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )
        if not resolved.is_relative_to(self.storage_root):
            raise ValueError("path escape")
        return resolved


class _FakeDetector:
    runtime_metadata: ClassVar[dict] = {
        "backend": "test",
        "checkpoint_sha256": "d" * 64,
        "model_class": "FakeRFDETR",
    }

    @staticmethod
    def detect_batch(frames):
        return [
            [
                PlayerDetection(
                    bbox_xyxy=[2, 2, 95.741699, 63.5],
                    confidence=0.9,
                    class_id=0,
                    class_name="player",
                ),
                PlayerDetection(
                    bbox_xyxy=[96.25, 2, 110, 20],
                    confidence=0.8,
                    class_id=0,
                    class_name="player",
                ),
            ]
            for _ in frames
        ]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_storage_prepare_confirm_and_detection_materialization(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64)
    )
    assert writer.isOpened()
    for index in range(20):
        value = 15 if index < 10 else 235
        writer.write(np.full((64, 96, 3), value, dtype=np.uint8))
    writer.release()

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = User(
            user_id="user",
            email="fresh@example.com",
            password_hash="x",
            display_name="Fresh",
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
            title="Fresh",
            status="DRAFT",
        )
        source = MediaAsset(
            asset_id="source",
            match_id="match",
            asset_type="RAW_VIDEO",
            file_path="source.mp4",
            original_filename="source.mp4",
            mime_type="video/mp4",
            duration_sec=2,
            fps=10,
            width=96,
            height=64,
            size_bytes=source_path.stat().st_size,
            sha256=_file_sha(source_path),
        )
        job = AnalysisJob(
            analysis_job_id="job",
            match_id="match",
            media_asset_id="source",
            video_sha256=source.sha256,
            model_version="test",
            policy_version="test",
            job_type="HIGHLIGHT_SPOTTING",
            status="COMPLETED",
            progress=100,
            options={},
        )
        event = TimelineEvent(
            timeline_event_id="event",
            match_id="match",
            source_job_id="job",
            event_type="goal",
            label="goal",
            timestamp_sec=1,
            start_sec=0.8,
            end_sec=1.2,
            duration_sec=0.4,
            player_ids=[],
            metadata_={},
        )
        scene = TimelineEvent(
            timeline_event_id="scene",
            match_id="match",
            source_job_id="job",
            event_type="scene",
            label="goal",
            timestamp_sec=1,
            start_sec=0,
            end_sec=2,
            duration_sec=2,
            player_ids=[],
            metadata_={},
        )
        revision = HighlightRevision(
            revision_id="revision",
            project_id="project",
            revision_number=1,
            action_spotting_job_id="job",
            user_request="test",
            structured_request={},
            selected_scene_ids=["scene"],
            scene_selection=[],
            focus_mode="NONE",
            status="SCENES_SELECTED",
            options={},
        )
        db.add_all([user, match, project, source, job, event, scene, revision])
        db.commit()

        service = ShotBoundaryReviewService(
            db, detector_factory=lambda _settings: _FakeDetector()
        )
        service.storage = _TempStorage(tmp_path)
        service.settings = service.settings.model_copy(
            update={
                "SCENE_TARGET_SELECTION_PROJECT_ROOT": str(
                    Path(__file__).parents[1] / ".tracking-runtime"
                )
            }
        )
        service.media = service.media.__class__(db)
        discovery_inputs = service.prepare_candidate_discovery_inputs(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
        )
        assert discovery_inputs["boundary_origin"] == "AUTO_DETECTED"
        automatic_artifact = db.get(
            Artifact, discovery_inputs["shot_boundaries_artifact_id"]
        )
        automatic_document = json.loads(
            service.storage.resolve_path(automatic_artifact.file_path).read_text(
                encoding="utf-8"
            )
        )
        assert automatic_artifact.artifact_type == "AUTO_SHOT_BOUNDARIES"
        assert automatic_document["human_reviewed"] is False
        assert automatic_document["automatic_target_confirmation"] is False
        assert all(
            "review_state" not in row and "review_status" not in row
            for row in automatic_document["shots"]
        )
        assert db.scalar(select(ShotBoundaryReviewDecision)) is None
        automatic_detections = db.get(
            Artifact, discovery_inputs["detections_artifact_id"]
        )
        assert (
            automatic_detections.metadata_["sampling"]["policy_version"]
            == FAST_CANDIDATE_DETECTION_POLICY_VERSION
        )
        assert (
            automatic_detections.metadata_["bbox_geometry_contract_version"]
            == DETECTION_BBOX_GEOMETRY_CONTRACT_VERSION
        )
        with service.storage.resolve_path(automatic_detections.file_path).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            detection_rows = list(csv.DictReader(handle))
        assert detection_rows
        assert all(
            0 <= float(row["x1"]) < float(row["x2"]) <= 95
            and 0 <= float(row["y1"]) < float(row["y2"]) <= 63
            for row in detection_rows
        )
        assert all(float(row["x2"]) == 95 for row in detection_rows)
        assert all(float(row["y2"]) == 63 for row in detection_rows)
        assert automatic_detections.metadata_["detection_count"] == len(detection_rows)
        assert (
            len(detection_rows) == automatic_detections.metadata_["sampled_frame_count"]
        )
        assert all(int(row["detection_index"]) == 0 for row in detection_rows)
        automatic_session = service._required_session(
            "project", "revision", "event", "scene", "user"
        )
        automatic_live = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).parents[1]
                    / ".tracking-runtime/target_centric_tracking_scene_target_selection_v1/run_scene_target_selection.py"
                ),
                "discover",
                "--project-root",
                str(Path(__file__).parents[1] / ".tracking-runtime"),
                "--scene-id",
                "scene",
                "--discovery-id",
                "automatic-live-contract",
                "--video",
                str(
                    service.storage.resolve_path(
                        automatic_session.scene_video_asset.file_path
                    )
                ),
                "--detections-csv",
                str(service.storage.resolve_path(automatic_detections.file_path)),
                "--shot-boundaries",
                str(service.storage.resolve_path(automatic_artifact.file_path)),
                "--output-root",
                str(tmp_path / "automatic-live-scene-target-selection"),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert automatic_live.returncode == 0, automatic_live.stderr
        prepared = service.prepare(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
        )
        assert prepared.status == "WAITING_REVIEW"
        assert prepared.automatic_confirmation is False
        assert prepared.shots[0].start_frame == 0
        assert prepared.shots[-1].end_frame_inclusive == prepared.frame_count - 1
        confirmed = service.confirm(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            payload=ShotBoundaryConfirmRequest(
                draft_revision=prepared.draft_revision,
                reviewer_note="Reviewed in integration test.",
                idempotency_key="fresh-machine-confirm-001",
                scene_video_sha256=prepared.scene_video_sha256,
            ),
        )
        assert confirmed.status == "CONFIRMED"
        assert confirmed.detections_status == "READY"
        artifacts = db.scalars(select(Artifact)).all()
        assert {artifact.artifact_type for artifact in artifacts} >= {
            "SCENE_VIDEO",
            "SHOT_BOUNDARY_DRAFT",
            "REVIEWED_SHOT_BOUNDARIES",
            "SCENE_PLAYER_DETECTIONS",
        }
        reviewed = db.get(Artifact, confirmed.artifact_id)
        assert reviewed.metadata_["automatic_confirmation"] is False
        assert not Path(reviewed.file_path).is_absolute()
        original_path = service.storage.resolve_path(reviewed.file_path)
        original_bytes = original_path.read_bytes()
        reviewed_document = json.loads(original_bytes)
        reviewed_rows = reviewed_document["shots"]
        assert [row["shot_index"] for row in reviewed_rows] == list(
            range(len(reviewed_rows))
        )
        assert reviewed_rows[-1]["cut_out_frame"] is None
        assert all(
            row["frame_count"] == row["end_frame_inclusive"] - row["start_frame"] + 1
            for row in reviewed_rows
        )
        contract = service.validate_candidate_contract(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
        )
        assert contract["reviewed_shots_artifact_id"] == confirmed.artifact_id
        assert contract["reviewed_shot_boundaries_sha256"] == confirmed.artifact_sha256

        replacement = service.prepare(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            new_review_revision=True,
        )
        assert replacement.review_session_id != prepared.review_session_id
        reversed_shots = list(reversed(replacement.shots))
        saved = service.update_draft(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            payload=ShotBoundaryDraftUpdate(
                draft_revision=replacement.draft_revision,
                scene_video_sha256=replacement.scene_video_sha256,
                shots=reversed_shots,
                note="Intentionally sent out of order.",
            ),
        )
        replacement_confirmed = service.confirm(
            project=project,
            user=user,
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            payload=ShotBoundaryConfirmRequest(
                draft_revision=saved.draft_revision,
                reviewer_note="Confirmed replacement immutable revision.",
                idempotency_key="fresh-machine-confirm-002",
                scene_video_sha256=saved.scene_video_sha256,
            ),
        )
        assert replacement_confirmed.artifact_id != confirmed.artifact_id
        assert replacement_confirmed.artifact_sha256 != confirmed.artifact_sha256
        assert original_path.read_bytes() == original_bytes

        replacement_artifact = db.get(Artifact, replacement_confirmed.artifact_id)
        replacement_detections = db.get(
            Artifact, replacement_confirmed.detections_artifact_id
        )
        replacement_session = service._required_session(
            "project", "revision", "event", "scene", "user"
        )
        runtime_root = Path(__file__).parents[1] / ".tracking-runtime"
        live_output = tmp_path / "live-scene-target-selection"
        live = subprocess.run(
            [
                sys.executable,
                str(
                    runtime_root
                    / "target_centric_tracking_scene_target_selection_v1"
                    / "run_scene_target_selection.py"
                ),
                "discover",
                "--project-root",
                str(runtime_root),
                "--scene-id",
                "scene",
                "--discovery-id",
                "r15-live-contract",
                "--video",
                str(
                    service.storage.resolve_path(
                        replacement_session.scene_video_asset.file_path
                    )
                ),
                "--detections-csv",
                str(service.storage.resolve_path(replacement_detections.file_path)),
                "--shot-boundaries",
                str(service.storage.resolve_path(replacement_artifact.file_path)),
                "--output-root",
                str(live_output),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert live.returncode == 0, live.stderr
        assert (live_output / "scene_candidate_manifest.json").is_file()

        task_service = SceneAITaskService(db)
        old_task, _ = task_service.enqueue(
            user=user,
            project=project,
            task_type="EVENT_CANDIDATE_RECOMMENDATION_PREPARE_R1C",
            payload={
                "revision_id": "revision",
                "event_id": "event",
                "scene_id": "scene",
                "shortlist_size": 5,
                "reviewed_shots_artifact_id": confirmed.artifact_id,
                "reviewed_shot_boundaries_sha256": confirmed.artifact_sha256,
            },
        )
        old_task.status = "FAILED"
        db.commit()
        new_task, reused = task_service.enqueue(
            user=user,
            project=project,
            task_type="EVENT_CANDIDATE_RECOMMENDATION_PREPARE_R1C",
            payload={
                "revision_id": "revision",
                "event_id": "event",
                "scene_id": "scene",
                "shortlist_size": 5,
                "reviewed_shots_artifact_id": replacement_confirmed.artifact_id,
                "reviewed_shot_boundaries_sha256": (
                    replacement_confirmed.artifact_sha256
                ),
            },
        )
        assert reused is False
        assert new_task.task_id != old_task.task_id
        assert old_task.status == "FAILED"
        assert new_task.status == "QUEUED"
        db.refresh(revision)
        inputs = revision.options["candidate_pipeline_inputs"]
        assert inputs["reviewed_shots_artifact_id"] == replacement_confirmed.artifact_id
        assert (
            inputs["detections_artifact_id"]
            == replacement_confirmed.detections_artifact_id
        )
