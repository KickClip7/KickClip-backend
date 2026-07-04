from sqlalchemy.orm import Session

from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.timeline.schema import TimelineEventCreate


class TimelineEventService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = TimelineEventRepository(db)

    def create_timeline_event(self, data: TimelineEventCreate) -> TimelineEvent:
        payload = data.model_dump()
        payload["metadata_"] = payload.pop("metadata", {})

        event = self.repository.create(**payload)

        self.db.commit()
        self.db.refresh(event)
        return event

    def create_timeline_events(
        self,
        rows: list[TimelineEventCreate],
        commit: bool = True,
    ) -> list[TimelineEvent]:
        payloads: list[dict] = []

        for row in rows:
            payload = row.model_dump()
            payload["metadata_"] = payload.pop("metadata", {})
            payloads.append(payload)

        events = self.repository.bulk_create(payloads)

        if commit:
            self.db.commit()
            for event in events:
                self.db.refresh(event)

        return events

    def list_match_events(
        self,
        match_id: str,
        label: str | None = None,
        half: int | None = None,
    ) -> list[TimelineEvent]:
        return self.repository.list_by_match(
            match_id=match_id,
            label=label,
            half=half,
        )

    def replace_events_for_job(
        self,
        source_job_id: str,
        rows: list[TimelineEventCreate],
        commit: bool = True,
    ) -> list[TimelineEvent]:
        self.repository.delete_by_source_job(source_job_id)

        payloads: list[dict] = []
        for row in rows:
            payload = row.model_dump()
            payload["metadata_"] = payload.pop("metadata", {})
            payloads.append(payload)

        events = self.repository.bulk_create(payloads)

        if commit:
            self.db.commit()
            for event in events:
                self.db.refresh(event)

        return events