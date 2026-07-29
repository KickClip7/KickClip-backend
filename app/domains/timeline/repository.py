from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.domains.analysis.model import AnalysisJob
from app.domains.timeline.model import TimelineEvent


class TimelineEventRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> TimelineEvent:
        event = TimelineEvent(**kwargs)
        self.db.add(event)
        self.db.flush()
        return event

    def bulk_create(self, rows: list[dict]) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []

        for row in rows:
            event = TimelineEvent(**row)
            self.db.add(event)
            events.append(event)

        self.db.flush()
        return events

    def get_by_id(self, timeline_event_id: str) -> TimelineEvent | None:
        stmt = select(TimelineEvent).where(
            TimelineEvent.timeline_event_id == timeline_event_id
        )
        return self.db.scalar(stmt)

    def list_by_match(
        self,
        match_id: str,
        label: str | None = None,
        half: int | None = None,
    ) -> list[TimelineEvent]:
        stmt = select(TimelineEvent).where(TimelineEvent.match_id == match_id)

        if label:
            stmt = stmt.where(TimelineEvent.label == label)

        if half is not None:
            stmt = stmt.where(TimelineEvent.half == half)

        stmt = stmt.order_by(
            TimelineEvent.half.asc().nulls_last(),
            TimelineEvent.timestamp_sec.asc(),
        )

        return list(self.db.scalars(stmt).all())

    def list_current_by_match(
        self,
        match_id: str,
        label: str | None = None,
        half: int | None = None,
    ) -> list[TimelineEvent]:
        """Return non-Action-Spotting events plus the latest completed result.

        Action Spotting cache keys include the inference policy version, so a
        policy change legitimately creates a new AnalysisJob. Historical job
        rows remain available for audit, but their TimelineEvents must not be
        mixed into the match's current editing timeline.
        """
        action_job_ids = select(AnalysisJob.analysis_job_id).where(
            AnalysisJob.match_id == match_id,
            AnalysisJob.job_type == "HIGHLIGHT_SPOTTING",
        )
        latest_job_id = self.db.scalar(
            select(AnalysisJob.analysis_job_id)
            .where(
                AnalysisJob.match_id == match_id,
                AnalysisJob.job_type == "HIGHLIGHT_SPOTTING",
                AnalysisJob.status == "COMPLETED",
            )
            .order_by(
                AnalysisJob.completed_at.desc().nulls_last(),
                AnalysisJob.created_at.desc(),
            )
            .limit(1)
        )

        current_sources = [
            TimelineEvent.source_job_id.is_(None),
            TimelineEvent.source_job_id.not_in(action_job_ids),
        ]
        if latest_job_id is not None:
            current_sources.append(TimelineEvent.source_job_id == latest_job_id)

        stmt = select(TimelineEvent).where(
            TimelineEvent.match_id == match_id,
            or_(*current_sources),
        )

        if label:
            stmt = stmt.where(TimelineEvent.label == label)

        if half is not None:
            stmt = stmt.where(TimelineEvent.half == half)

        stmt = stmt.order_by(
            TimelineEvent.half.asc().nulls_last(),
            TimelineEvent.timestamp_sec.asc(),
        )
        return list(self.db.scalars(stmt).all())

    def list_by_source_job(self, source_job_id: str) -> list[TimelineEvent]:
        stmt = (
            select(TimelineEvent)
            .where(TimelineEvent.source_job_id == source_job_id)
            .order_by(TimelineEvent.timestamp_sec.asc())
        )
        return list(self.db.scalars(stmt).all())

    def delete_by_source_job(self, source_job_id: str) -> int:
        stmt = delete(TimelineEvent).where(TimelineEvent.source_job_id == source_job_id)
        result = self.db.execute(stmt)
        return int(result.rowcount or 0)
