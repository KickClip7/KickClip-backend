import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.v1.analysis_jobs import list_match_analysis_jobs
from app.api.v1.matches import get_match, list_matches
from app.api.v1.router import api_router
from app.db import models  # noqa: F401
from app.db.base import Base
from app.domains.analysis.model import AnalysisJob
from app.domains.analysis.schema import AnalysisJobCreateRequest
from app.domains.analysis.service import AnalysisJobService
from app.domains.artifact.model import Artifact
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset


class MatchRecoveryApiTest(unittest.TestCase):
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
        self.match = Match(
            match_id="match_owner",
            owner_id=self.owner.user_id,
            home_team="Home",
            away_team="Away",
        )
        self.other_match = Match(
            match_id="match_other",
            owner_id=self.other.user_id,
        )
        self.db.add_all([self.match, self.other_match])
        self.db.flush()
        self.raw_asset = MediaAsset(
            asset_id="asset_raw",
            match_id=self.match.match_id,
            asset_type="RAW_VIDEO",
            file_path="storage/raw.mp4",
        )
        self.preview_asset = MediaAsset(
            asset_id="asset_preview",
            match_id=self.match.match_id,
            asset_type="WEB_PREVIEW_VIDEO",
            file_path="storage/preview.mp4",
        )
        self.db.add_all([self.raw_asset, self.preview_asset])
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_match_routes_include_collection_and_analysis_job_listing(self) -> None:
        routes = {
            (route.path, frozenset(route.methods or set()))
            for route in api_router.routes
        }

        self.assertIn(("/matches", frozenset({"GET"})), routes)
        self.assertIn(
            ("/matches/{match_id}/analysis-jobs", frozenset({"GET"})),
            routes,
        )

    def test_artifact_download_allows_handler_to_validate_signed_token(self) -> None:
        route = next(
            route
            for route in api_router.routes
            if route.path == "/artifacts/{artifact_id}/download"
        )
        dependency_calls = {
            dependency.call for dependency in route.dependant.dependencies
        }
        self.assertNotIn(get_current_user, dependency_calls)

    def test_match_list_is_owner_scoped_and_restores_preview_urls(self) -> None:
        with patch(
            "app.api.v1.matches.build_signed_media_url",
            side_effect=self._signed_url,
        ):
            response = list_matches(db=self.db, current_user=self.owner)

        self.assertEqual([item.match_id for item in response], [self.match.match_id])
        item = response[0]
        self.assertEqual(item.raw_video_asset_id, self.raw_asset.asset_id)
        self.assertEqual(item.video_asset_id, self.preview_asset.asset_id)
        self.assertEqual(item.preview_video_asset_id, self.preview_asset.asset_id)
        self.assertEqual(
            item.video_url,
            f"/signed/{self.preview_asset.asset_id}?user={self.owner.user_id}",
        )
        self.assertEqual(item.preview_url, item.video_url)

    def test_match_detail_restores_media_urls(self) -> None:
        with patch(
            "app.api.v1.matches.build_signed_media_url",
            side_effect=self._signed_url,
        ):
            response = get_match(
                match_id=self.match.match_id,
                db=self.db,
                current_user=self.owner,
            )

        self.assertEqual(response.match_id, self.match.match_id)
        self.assertEqual(response.video_asset_id, self.preview_asset.asset_id)
        self.assertIsNotNone(response.video_url)

    def test_match_list_includes_real_thumbnail_artifact_url(self) -> None:
        thumbnail = Artifact(
            artifact_id="art_match_thumbnail",
            match_id=self.match.match_id,
            artifact_type="MATCH_THUMBNAIL",
            file_path="storage/matches/match_owner/thumbnails/frame.jpg",
            mime_type="image/jpeg",
        )
        self.db.add(thumbnail)
        self.db.commit()

        with (
            patch(
                "app.api.v1.matches.build_signed_media_url",
                side_effect=self._signed_url,
            ),
            patch(
                "app.api.v1.matches.build_signed_artifact_url",
                return_value=("/signed/thumbnail.jpg", 900),
            ),
        ):
            item = list_matches(db=self.db, current_user=self.owner)[0]

        self.assertEqual(item.thumbnail_artifact_id, thumbnail.artifact_id)
        self.assertEqual(item.thumbnail_url, "/signed/thumbnail.jpg")

    def test_match_analysis_jobs_are_discoverable_newest_first(self) -> None:
        created_at = datetime.now(timezone.utc)
        completed = AnalysisJob(
            analysis_job_id="job_completed",
            match_id=self.match.match_id,
            job_type="FULL_MATCH_ANALYSIS",
            status="COMPLETED",
            progress=100,
            options={},
            created_at=created_at,
            completed_at=created_at,
        )
        running = AnalysisJob(
            analysis_job_id="job_running",
            match_id=self.match.match_id,
            job_type="FULL_MATCH_ANALYSIS",
            status="RUNNING",
            progress=45,
            options={},
            created_at=created_at + timedelta(seconds=1),
            started_at=created_at + timedelta(seconds=1),
        )
        self.db.add_all([completed, running])
        self.db.commit()

        response = list_match_analysis_jobs(
            match_id=self.match.match_id,
            db=self.db,
            current_user=self.owner,
        )

        self.assertEqual(
            [job.analysis_job_id for job in response],
            [running.analysis_job_id, completed.analysis_job_id],
        )
        self.assertEqual([job.status for job in response], ["running", "completed"])

    def test_match_list_and_detail_include_latest_durable_analysis_snapshot(self) -> None:
        created_at = datetime.now(timezone.utc)
        self.db.add_all(
            [
                AnalysisJob(
                    analysis_job_id="job_old",
                    match_id=self.match.match_id,
                    job_type="FULL_MATCH_ANALYSIS",
                    status="COMPLETED",
                    progress=100,
                    current_step="timeline",
                    options={},
                    created_at=created_at,
                ),
                AnalysisJob(
                    analysis_job_id="job_latest",
                    match_id=self.match.match_id,
                    job_type="FULL_MATCH_ANALYSIS",
                    status="RUNNING",
                    progress=47,
                    current_step="highlight_spotting",
                    error_message=None,
                    options={},
                    created_at=created_at + timedelta(seconds=1),
                ),
            ]
        )
        self.db.commit()

        with patch(
            "app.api.v1.matches.build_signed_media_url",
            side_effect=self._signed_url,
        ):
            item = list_matches(db=self.db, current_user=self.owner)[0]
            detail = get_match(
                match_id=self.match.match_id,
                db=self.db,
                current_user=self.owner,
            )

        for response in (item, detail):
            self.assertEqual(response.analysis_job_id, "job_latest")
            self.assertEqual(response.analysis_status, "RUNNING")
            self.assertEqual(response.analysis_progress, 47)
            self.assertEqual(
                response.analysis_current_step,
                "highlight_spotting",
            )

    def test_analysis_create_reuses_identical_active_job(self) -> None:
        service = AnalysisJobService(self.db)
        payload = AnalysisJobCreateRequest(
            job_type="FULL_MATCH_ANALYSIS",
            options={"run_player_tracking": True},
        )
        first = service.create_analysis_job_for_match(
            self.match.match_id,
            payload,
        )
        second = service.create_analysis_job_for_match(
            self.match.match_id,
            payload,
        )

        self.assertEqual(first.analysis_job_id, second.analysis_job_id)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)

    def test_match_analysis_jobs_hide_other_users_matches(self) -> None:
        with self.assertRaises(HTTPException) as exc:
            list_match_analysis_jobs(
                match_id=self.match.match_id,
                db=self.db,
                current_user=self.other,
            )

        self.assertEqual(exc.exception.status_code, 404)

    def _create_user(self, user_id: str, email: str) -> User:
        user = User(
            user_id=user_id,
            email=email,
            password_hash="!",
            display_name=user_id,
        )
        self.db.add(user)
        self.db.flush()
        return user

    @staticmethod
    def _signed_url(asset_id: str, user_id: str) -> tuple[str, int]:
        return f"/signed/{asset_id}?user={user_id}", 900


if __name__ == "__main__":
    unittest.main()
