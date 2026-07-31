from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking import (
    EventCandidateRankingService,
)
from app.domains.highlight.model import EventCandidateRankingJob
from app.domains.highlight.repository import HighlightRepository
from app.domains.match.model import Match
from app.domains.project.model import Project


def test_ranking_models_preserve_owner_scope_and_do_not_confirm_target() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine, expire_on_commit=False) as db:
        user = User(
            user_id="usr_rank",
            email="rank@example.com",
            password_hash="!",
            display_name="Rank",
        )
        match = Match(
            match_id="match_rank",
            owner_id=user.user_id,
            duration_sec=100.0,
        )
        project = Project(
            project_id="proj_rank",
            match_id=match.match_id,
            owner_id=user.user_id,
            title="Rank",
        )
        db.add_all([user, match, project])
        db.commit()
        job = EventCandidateRankingJob(
            ranking_job_id="erjob_rank",
            ranking_id="erank_rank",
            owner_id=user.user_id,
            match_id=match.match_id,
            project_id=project.project_id,
            revision_id="hrev_external",
            scene_id="evt_external",
            event_id="evt_external",
            event_label="goal",
            event_time_sec=10.0,
            status="PARTIAL_FEATURES",
            policy_state="PROVISIONAL_SHADOW_ONLY",
            cache_key="a" * 64,
            artifact_root="storage/ranking",
            input_contract={},
            feature_availability={"ball": "UNAVAILABLE"},
            ranking_artifact={
                "candidates": [],
                "shortlist": [],
                "automatic_target_confirmation": False,
            },
            shortlist_artifact={},
            manifest_sha256="b" * 64,
            metadata_={
                "automatic_target_confirmation": False,
                "ranking_launches_tracking": False,
            },
            created_at=now,
            updated_at=now,
        )
        # The unit only exercises read/scoping behavior; omit invalid external
        # FKs by disabling enforcement in in-memory SQLite.
        db.add(job)
        db.commit()
        service = EventCandidateRankingService(db)
        owned = service.require_owned(
            ranking_job_id=job.ranking_job_id, user=user
        )
        response = service.read(owned)
        assert response.automatic_target_confirmation is False
        assert response.selection_contract["candidate_must_be_user_selected"]
        assert response.selection_contract["ranking_launches_tracking"] is False
        assert (
            HighlightRepository(db)
            .get_event_ranking_by_cache_key("a" * 64)
            .ranking_job_id
            == job.ranking_job_id
        )
    engine.dispose()

