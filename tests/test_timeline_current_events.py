from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import models  # noqa: F401
from app.db.base import Base
from app.domains.analysis.model import AnalysisJob
from app.domains.auth.model import User
from app.domains.match.model import Match
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository


def _event(
    *,
    event_id: str,
    match_id: str,
    source_job_id: str | None,
    timestamp_sec: float,
    label: str = "goal",
) -> TimelineEvent:
    return TimelineEvent(
        timeline_event_id=event_id,
        match_id=match_id,
        source_job_id=source_job_id,
        event_type="action_spotting" if source_job_id else "manual",
        label=label,
        timestamp_sec=timestamp_sec,
        start_sec=timestamp_sec - 1,
        end_sec=timestamp_sec + 1,
        duration_sec=2,
    )


def test_current_timeline_uses_only_latest_completed_action_spotting_job() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        db.add(
            User(
                user_id="usr_timeline",
                email="timeline@example.com",
                password_hash="!",
                display_name="Timeline Test",
            )
        )
        db.add(Match(match_id="match_timeline", owner_id="usr_timeline"))
        db.add_all(
            [
                AnalysisJob(
                    analysis_job_id="job_old",
                    match_id="match_timeline",
                    job_type="HIGHLIGHT_SPOTTING",
                    status="COMPLETED",
                    completed_at=now,
                ),
                AnalysisJob(
                    analysis_job_id="job_latest",
                    match_id="match_timeline",
                    job_type="HIGHLIGHT_SPOTTING",
                    status="COMPLETED",
                    completed_at=now + timedelta(minutes=1),
                ),
                AnalysisJob(
                    analysis_job_id="job_running",
                    match_id="match_timeline",
                    job_type="HIGHLIGHT_SPOTTING",
                    status="RUNNING",
                    started_at=now + timedelta(minutes=2),
                ),
            ]
        )
        db.flush()
        db.add_all(
            [
                _event(
                    event_id="evt_old",
                    match_id="match_timeline",
                    source_job_id="job_old",
                    timestamp_sec=10,
                ),
                _event(
                    event_id="evt_latest",
                    match_id="match_timeline",
                    source_job_id="job_latest",
                    timestamp_sec=20,
                ),
                _event(
                    event_id="evt_running",
                    match_id="match_timeline",
                    source_job_id="job_running",
                    timestamp_sec=30,
                ),
                _event(
                    event_id="evt_manual",
                    match_id="match_timeline",
                    source_job_id=None,
                    timestamp_sec=40,
                    label="card",
                ),
            ]
        )
        db.commit()

        repository = TimelineEventRepository(db)
        assert [
            row.timeline_event_id
            for row in repository.list_current_by_match("match_timeline")
        ] == ["evt_latest", "evt_manual"]
        assert [
            row.timeline_event_id
            for row in repository.list_current_by_match(
                "match_timeline",
                label="goal",
            )
        ] == ["evt_latest"]

    engine.dispose()
