import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import models  # noqa: F401
from app.db.base import Base
from app.domains.auth.access import (
    require_analysis_job_access,
    require_match_access,
    require_media_access,
    require_project_access,
)
from app.domains.auth.model import User
from app.domains.analysis.model import AnalysisJob
from app.domains.artifact.model import Artifact
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.clip_plan.schema import (
    ClipPlanItemCreate,
    ManualClipPlanCreateRequest,
)
from app.domains.clip_plan.service import ClipPlanService
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.project.schema import ProjectCreate
from app.domains.project.service import ProjectService
from app.domains.render.model import RenderJob
from app.domains.studio.service import StudioService
from app.domains.timeline.model import TimelineEvent
from app.storage.local_storage import StoredFile


class FakeStorage:
    def __init__(self, root: Path):
        self.project_root = root
        self.storage_root = root / "storage"
        self.storage_root.mkdir(parents=True)
        self.deleted: list[str] = []

    def save_upload_file(self, *, upload_file: UploadFile, subdir: str) -> StoredFile:
        directory = self.storage_root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (upload_file.filename or "upload.bin")
        path.write_bytes(upload_file.file.read())
        return StoredFile(
            absolute_path=path,
            relative_path=path.relative_to(self.project_root).as_posix(),
            filename=path.name,
            size_bytes=path.stat().st_size,
        )

    def delete_file_if_exists(self, value: str) -> bool:
        self.deleted.append(value)
        path = self.project_root / value
        if path.exists():
            path.unlink()
            return True
        return False


class ProjectMatchRelationshipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )
        self.db: Session = self.SessionLocal()
        self.owner = self._create_user("usr_owner", "owner@example.com")
        self.other = self._create_user("usr_other", "other@example.com")
        self.db.commit()
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.tempdir.cleanup()

    def test_upload_creates_match_and_raw_asset_but_no_project(self) -> None:
        response = self._upload(playable=True)

        self.assertTrue(response.match_id.startswith("match_"))
        self.assertNotIn("project_id", response.model_dump())
        self.assertEqual(self._count(Match), 1)
        self.assertEqual(self._count(Project), 0)
        raw_asset = self.db.scalar(
            select(MediaAsset).where(MediaAsset.asset_type == "RAW_VIDEO")
        )
        self.assertIsNotNone(raw_asset)
        self.assertEqual(raw_asset.match_id, response.match_id)

    def test_incompatible_upload_attaches_preview_to_same_match(self) -> None:
        response = self._upload(playable=False)

        assets = list(
            self.db.scalars(
                select(MediaAsset).order_by(MediaAsset.asset_type)
            ).all()
        )
        self.assertEqual(len(assets), 2)
        self.assertEqual({asset.match_id for asset in assets}, {response.match_id})
        self.assertEqual(
            {asset.asset_type for asset in assets},
            {"RAW_VIDEO", "WEB_PREVIEW_VIDEO"},
        )
        self.assertEqual(response.preview_status, "PREVIEW_READY")

    def test_upload_failure_rolls_back_rows_and_deletes_file(self) -> None:
        storage = FakeStorage(Path(self.tempdir.name))
        service = StudioService(self.db)
        service.storage = storage
        upload = UploadFile(filename="broken.mp4", file=BytesIO(b"broken"))

        with patch(
            "app.domains.studio.service.extract_video_metadata",
            side_effect=RuntimeError("ffprobe failed"),
        ):
            with self.assertRaises(RuntimeError):
                service.upload_match_video(
                    file=upload,
                    owner_id=self.owner.user_id,
                )

        self.assertEqual(self._count(Match), 0)
        self.assertEqual(self._count(MediaAsset), 0)
        self.assertEqual(len(storage.deleted), 1)
        self.assertFalse((Path(self.tempdir.name) / storage.deleted[0]).exists())

    def test_same_match_can_have_multiple_isolated_projects(self) -> None:
        match = self._create_match()
        service = ProjectService(self.db)
        first = service.create_project(
            ProjectCreate(title="First"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )
        second = service.create_project(
            ProjectCreate(title="Second"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )

        self.assertNotEqual(first.project_id, second.project_id)
        self.assertEqual(first.match_id, match.match_id)
        self.assertEqual(second.match_id, match.match_id)
        self.assertEqual(len(service.list_match_projects(match.match_id)), 2)

    def test_clip_plans_are_scoped_to_project(self) -> None:
        match = self._create_match()
        project_service = ProjectService(self.db)
        first = project_service.create_project(
            ProjectCreate(title="First"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )
        second = project_service.create_project(
            ProjectCreate(title="Second"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )
        event = TimelineEvent(
            match_id=match.match_id,
            event_type="action_spotting",
            label="goal",
            timestamp_sec=10,
            start_sec=8,
            end_sec=12,
            duration_sec=4,
        )
        self.db.add(event)
        self.db.commit()

        clip_service = ClipPlanService(self.db)
        first_plan = clip_service.create_manual_clip_plan(
            ManualClipPlanCreateRequest(
                project_id=first.project_id,
                items=[
                    ClipPlanItemCreate(
                        timeline_event_id=event.timeline_event_id,
                        start_sec=8,
                        end_sec=12,
                        order_index=0,
                    )
                ],
            ),
            created_by=self.owner.user_id,
        )
        second_plan = clip_service.create_manual_clip_plan(
            ManualClipPlanCreateRequest(
                project_id=second.project_id,
                items=[
                    ClipPlanItemCreate(
                        timeline_event_id=event.timeline_event_id,
                        start_sec=9,
                        end_sec=11,
                        order_index=0,
                    )
                ],
            ),
            created_by=self.owner.user_id,
        )

        repository = ClipPlanRepository(self.db)
        self.assertEqual(
            [plan.clip_plan_id for plan in repository.list_by_project(first.project_id)],
            [first_plan.clip_plan_id],
        )
        self.assertEqual(
            [plan.clip_plan_id for plan in repository.list_by_project(second.project_id)],
            [second_plan.clip_plan_id],
        )
        first_state = StudioService(self.db).get_project_edit_state(
            first.project_id
        )
        second_state = StudioService(self.db).get_project_edit_state(
            second.project_id
        )
        self.assertIsNotNone(first_state)
        self.assertIsNotNone(second_state)
        self.assertEqual(
            [plan.clip_plan_id for plan in first_state.clip_plans],
            [first_plan.clip_plan_id],
        )
        self.assertEqual(
            [plan.clip_plan_id for plan in second_state.clip_plans],
            [second_plan.clip_plan_id],
        )

    def test_project_creation_rejects_non_owner(self) -> None:
        match = self._create_match()
        with self.assertRaises(ValueError):
            ProjectService(self.db).create_project(
                ProjectCreate(title="Forbidden"),
                match_id=match.match_id,
                owner_id=self.other.user_id,
            )

    def test_project_edit_state_includes_completed_render_download_url(self) -> None:
        match = self._create_match()
        project = ProjectService(self.db).create_project(
            ProjectCreate(title="Rendered"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )
        clip_plan = ClipPlanService(self.db).create_manual_clip_plan(
            ManualClipPlanCreateRequest(project_id=project.project_id),
            created_by=self.owner.user_id,
        )
        artifact = Artifact(
            match_id=match.match_id,
            project_id=project.project_id,
            artifact_type="RENDERED_VIDEO",
            file_path="storage/rendered.mp4",
        )
        self.db.add(artifact)
        self.db.flush()
        render_job = RenderJob(
            clip_plan_id=clip_plan.clip_plan_id,
            status="completed",
            progress=100,
            output_artifact_id=artifact.artifact_id,
        )
        self.db.add(render_job)
        self.db.commit()

        state = StudioService(self.db).get_project_edit_state(project.project_id)

        self.assertIsNotNone(state)
        self.assertEqual(len(state.render_jobs), 1)
        self.assertEqual(
            state.render_jobs[0].download_url,
            f"/api/v1/renders/{render_job.render_job_id}/download",
        )

    def test_deleting_project_deletes_project_scoped_artifacts(self) -> None:
        match = self._create_match()
        project = ProjectService(self.db).create_project(
            ProjectCreate(title="Disposable"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )
        project_artifact = Artifact(
            match_id=match.match_id,
            project_id=project.project_id,
            artifact_type="RENDERED_VIDEO",
            file_path="storage/project-render.mp4",
        )
        shared_artifact = Artifact(
            match_id=match.match_id,
            project_id=None,
            artifact_type="ANALYSIS_RESULT",
            file_path="storage/analysis.json",
        )
        self.db.add_all([project_artifact, shared_artifact])
        self.db.commit()
        project_artifact_id = project_artifact.artifact_id
        shared_artifact_id = shared_artifact.artifact_id

        self.db.delete(project)
        self.db.commit()

        self.assertIsNone(self.db.get(Artifact, project_artifact_id))
        self.assertIsNotNone(self.db.get(Artifact, shared_artifact_id))

    def test_access_follows_match_owner_for_all_match_resources(self) -> None:
        match = self._create_match()
        project = ProjectService(self.db).create_project(
            ProjectCreate(title="Owned"),
            match_id=match.match_id,
            owner_id=self.owner.user_id,
        )
        asset = MediaAsset(
            match_id=match.match_id,
            asset_type="RAW_VIDEO",
            file_path="storage/test.mp4",
        )
        job = AnalysisJob(
            match_id=match.match_id,
            job_type="ACTION_SPOTTING",
            status="QUEUED",
            progress=0,
        )
        self.db.add_all([asset, job])
        self.db.commit()

        self.assertEqual(
            require_match_access(self.db, match.match_id, self.owner).match_id,
            match.match_id,
        )
        self.assertEqual(
            require_project_access(self.db, project.project_id, self.owner).project_id,
            project.project_id,
        )
        self.assertEqual(
            require_media_access(self.db, asset.asset_id, self.owner).asset_id,
            asset.asset_id,
        )
        self.assertEqual(
            require_analysis_job_access(
                self.db,
                job.analysis_job_id,
                self.owner,
            ).analysis_job_id,
            job.analysis_job_id,
        )
        for assertion in (
            lambda: require_match_access(self.db, match.match_id, self.other),
            lambda: require_project_access(self.db, project.project_id, self.other),
            lambda: require_media_access(self.db, asset.asset_id, self.other),
            lambda: require_analysis_job_access(
                self.db,
                job.analysis_job_id,
                self.other,
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                assertion()
            self.assertEqual(exc.exception.status_code, 404)

    def _upload(self, *, playable: bool):
        storage = FakeStorage(Path(self.tempdir.name))
        service = StudioService(self.db)
        service.storage = storage
        upload = UploadFile(filename="match.mp4", file=BytesIO(b"video"))
        raw_metadata = {
            "duration_sec": 90.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "size_bytes": 5,
            "codec_name": "h264",
        }
        playability = SimpleNamespace(
            is_browser_playable=playable,
            reason="test",
        )

        def create_preview(**kwargs):
            output = Path(kwargs["output_dir"]) / "preview.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"preview")
            return StoredFile(
                absolute_path=output,
                relative_path=output.relative_to(storage.project_root).as_posix(),
                filename=output.name,
                size_bytes=output.stat().st_size,
            )

        with (
            patch(
                "app.domains.studio.service.extract_video_metadata",
                return_value=raw_metadata,
            ),
            patch(
                "app.domains.studio.service.check_browser_playability",
                return_value=playability,
            ),
            patch(
                "app.domains.studio.service.create_browser_preview_clip",
                side_effect=create_preview,
            ),
        ):
            return service.upload_match_video(
                file=upload,
                owner_id=self.owner.user_id,
            )

    def _create_match(self) -> Match:
        match = Match(owner_id=self.owner.user_id)
        self.db.add(match)
        self.db.commit()
        return match

    def _create_user(self, user_id: str, email: str) -> User:
        user = User(
            user_id=user_id,
            email=email,
            password_hash="!",
            display_name=user_id,
        )
        self.db.add(user)
        return user

    def _count(self, model: type) -> int:
        return int(self.db.scalar(select(func.count()).select_from(model)) or 0)


if __name__ == "__main__":
    unittest.main()
