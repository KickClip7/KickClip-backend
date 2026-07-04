from sqlalchemy import delete, select
from sqlalchemy.orm import Session

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