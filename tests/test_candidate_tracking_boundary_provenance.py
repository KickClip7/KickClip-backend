from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.v1.event_candidate_handoff_r1 import _raise
from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1.errors import (
    CandidateTrackingConfigurationInvalid,
)
from app.domains.candidate_handoff_r1.model import EventCandidateSelectionR1
from app.domains.candidate_handoff_r1.service import CandidateHandoffR1Service
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.model import TrackingJob


class _TempStorage:
    def __init__(self, root: Path) -> None:
        self.project_root = root.resolve()
        self.storage_root = root.resolve()

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return _sha(path)


def test_tracking_output_root_must_be_absolute_and_inside_storage(
    tmp_path: Path,
) -> None:
    service = object.__new__(CandidateHandoffR1Service)
    service.storage = _TempStorage(tmp_path / "storage")

    with pytest.raises(CandidateTrackingConfigurationInvalid) as relative_error:
        service._tracking_output_root("tracking-runs")
    assert relative_error.value.reason == "TRACKING_OUTPUT_ROOT_NOT_ABSOLUTE"

    outside = tmp_path / "outside"
    with pytest.raises(CandidateTrackingConfigurationInvalid) as outside_error:
        service._tracking_output_root(str(outside.resolve()))
    assert outside_error.value.reason == "TRACKING_OUTPUT_ROOT_OUTSIDE_STORAGE"

    configured = (tmp_path / "storage" / "tracking-runs").resolve()
    assert service._tracking_output_root(str(configured)) == configured


def test_tracking_configuration_error_is_not_hidden_by_generic_handoff_error() -> None:
    error = CandidateTrackingConfigurationInvalid(
        "TRACKING_OUTPUT_ROOT_OUTSIDE_STORAGE",
        "TRACKING_OUTPUT_ROOT must be inside STORAGE_ROOT.",
    )
    with pytest.raises(HTTPException) as raised:
        _raise(error)

    assert raised.value.status_code == 503
    assert raised.value.detail == {
        "code": "CANDIDATE_TRACKING_CONFIGURATION_INVALID",
        "message": "TRACKING_OUTPUT_ROOT must be inside STORAGE_ROOT.",
        "reason": "TRACKING_OUTPUT_ROOT_OUTSIDE_STORAGE",
    }


def test_legacy_selection_with_structurally_valid_auto_boundaries_starts_tracking(
    tmp_path: Path,
    monkeypatch,
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
        video_path = tmp_path / "scene.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24)
        )
        assert writer.isOpened()
        for index in range(10):
            writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
        writer.release()
        video = MediaAsset(
            asset_id="scene-video",
            match_id="match",
            asset_type="HIGHLIGHT_SCENE_CLIP",
            file_path="scene.mp4",
            original_filename="scene.mp4",
            mime_type="video/mp4",
            duration_sec=2,
            fps=5,
            width=32,
            height=24,
            size_bytes=video_path.stat().st_size,
            sha256=_sha(video_path),
        )
        boundary_path = tmp_path / "auto_shot_boundaries.json"
        boundary_sha = _write_json(
            boundary_path,
            {
                "artifact_type": "AUTO_SHOT_BOUNDARIES",
                "boundary_origin": "AUTO_DETECTED",
                "human_reviewed": False,
                "automatic_target_confirmation": False,
                "automatic_confirmation": False,
                "project_id": "project",
                "revision_id": "revision",
                "event_id": "event",
                "scene_id": "scene",
                "video": {"sha256": video.sha256, "frame_count": 10},
                "shots": [
                    {
                        "shot_index": 0,
                        "shot_id": "shot_0000",
                        "start_frame": 0,
                        "end_frame_inclusive": 9,
                        "boundary_state": "AUTO_DETECTED",
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
        boundary = Artifact(
            artifact_id="auto-boundaries",
            match_id="match",
            project_id="project",
            analysis_job_id=None,
            artifact_type="AUTO_SHOT_BOUNDARIES",
            file_path="auto_shot_boundaries.json",
            mime_type="application/json",
            metadata_={
                "revision_id": "revision",
                "event_id": "event",
                "scene_id": "scene",
                "status": "STRUCTURALLY_VALID",
                "boundary_origin": "AUTO_DETECTED",
                "human_reviewed": False,
                "automatic_target_confirmation": False,
                "sha256": boundary_sha,
            },
        )

        bundle_root = tmp_path / "bundle"
        references = []
        for index in range(3):
            crop = bundle_root / f"reference_{index}.jpg"
            crop.parent.mkdir(parents=True, exist_ok=True)
            crop.write_bytes(f"crop-{index}".encode())
            references.append(
                {
                    "candidate_id": "shot_0000_track_0000",
                    "frame": 4 + index,
                    "bbox_xyxy": [1, 2, 20, 22],
                    "path": crop.name,
                    "crop_sha256": _sha(crop),
                    "scale_class": "medium",
                }
            )
        manifest_path = bundle_root / "manifest.json"
        manifest_sha = _write_json(
            manifest_path,
            {
                "candidate_id": "shot_0000_track_0000",
                "candidate_media_id": "shot_0000_track_0000",
                "shot_id": "shot_0000",
                "tracklet_id": "track_0000",
                "source_video_sha256": video.sha256,
                # This is the exact legacy shape that previously produced 409.
                "reviewed_shot_boundaries_sha256": boundary_sha,
                "best_observation": {
                    "candidate_id": "shot_0000_track_0000",
                    "frame": 5,
                    "bbox_xyxy": [1, 2, 20, 22],
                },
                "quality": {"identity_pure": True, "reviewability": "USABLE"},
                "reference_gallery": references,
                "observations": [],
            },
        )
        selection_record = tmp_path / "selection_record.json"
        selection_sha = _write_json(selection_record, {"immutable": True})
        selection = EventCandidateSelectionR1(
            selection_id="selection",
            project_id="project",
            owner_id="user",
            revision_id="revision",
            event_id="event",
            scene_id="scene",
            ranking_id="ranking",
            shortlist_patch_id="patch",
            discovery_id="discovery",
            candidate_id="shot_0000_track_0000",
            shot_id="shot_0000",
            tracklet_id="track_0000",
            selected_at=datetime.now(timezone.utc),
            candidate_manifest_sha256="c" * 64,
            candidate_media_bundle_sha256=manifest_sha,
            source_video_sha256=video.sha256,
            reviewed_shot_boundaries_sha256=boundary_sha,
            selection_artifact_path="selection_record.json",
            selection_artifact_sha256=selection_sha,
            media_bundle_manifest_path="bundle/manifest.json",
            metadata_={},
        )
        db.add_all([user, match, project, video, boundary, selection])
        db.commit()

        captured: dict = {}

        class _FakeAdapter:
            def __init__(self, **_kwargs) -> None:
                pass

            def build(self, **kwargs):
                captured.update(kwargs["shot_boundaries_provenance"])
                return SimpleNamespace()

            @staticmethod
            def attach_to_job(job, _result) -> None:
                job.runtime_metadata = {
                    **(job.runtime_metadata or {}),
                    "scene_target_selection": {"verified": True},
                }

        class _FakeOrchestrator:
            def __init__(self, _db) -> None:
                pass

            @staticmethod
            def initialize(**_kwargs) -> None:
                pass

        submitted: list[str] = []
        monkeypatch.setattr(
            "app.domains.candidate_handoff_r1.service.get_settings",
            lambda: SimpleNamespace(
                TRACKING_OUTPUT_ROOT=str(tmp_path / "tracking"),
                TRACKING_DEVICE="cpu",
                TRACKING_REACQUISITION_MODE="assisted",
            ),
        )
        monkeypatch.setattr(
            "app.domains.candidate_handoff_r1.service.R1R3InputAdapter",
            _FakeAdapter,
        )
        monkeypatch.setattr(
            "app.domains.candidate_handoff_r1.service.R1PipelineOrchestrator",
            _FakeOrchestrator,
        )
        monkeypatch.setattr(
            "app.domains.highlight.service.HighlightWorkflowService.include_tracking_job_candidate",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "app.domains.candidate_handoff_r1.service.get_r1_tracking_executor",
            lambda: SimpleNamespace(submit=lambda job_id: submitted.append(job_id)),
        )

        service = CandidateHandoffR1Service(db)
        service.storage = _TempStorage(tmp_path)
        response = service.create_tracking_handoff(
            selection=selection,
            source_video=video,
            project=project,
            user=user,
        )

        assert response.status == "QUEUED"
        assert captured == {
            "artifact_id": "auto-boundaries",
            "artifact_type": "AUTO_SHOT_BOUNDARIES",
            "sha256": boundary_sha,
            "boundary_origin": "AUTO_DETECTED",
            "human_reviewed": False,
            "automatic_target_confirmation": False,
        }
        assert submitted == [response.tracking_job_id]
        job = db.get(TrackingJob, response.tracking_job_id)
        assert job is not None
        handoff = job.runtime_metadata["event_candidate_handoff_r1"]
        assert handoff["shot_boundaries"] == captured
        payload = json.loads(
            (Path(job.output_directory) / "tracking_payload.json").read_text()
        )
        assert payload["shot_boundaries"] == captured
        assert "reviewed_shot_boundaries" not in payload


def test_backend_adapter_manifest_matches_current_sources() -> None:
    project_root = Path(__file__).parents[1]
    manifest_path = (
        project_root
        / "app/domains/candidate_handoff_r1/runtime/r1_v1_v2_adapter_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["automatic_shot_boundary_input_supported"] is True
    assert (
        manifest["shot_boundary_provenance_policy"]
        == "EXPLICIT_AUTO_OR_HUMAN_REVIEWED_R1"
    )
    for row in manifest["source_files"]:
        source = project_root / row["path"]
        assert source.is_file()
        assert _sha(source) == row["sha256"]
