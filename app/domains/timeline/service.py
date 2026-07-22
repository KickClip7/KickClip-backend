import json

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.paths import get_storage_root
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.timeline.schema import TimelineEventCreate
from app.storage.workspace import get_match_timeline_events_path


def get_timeline_events(match_id: str) -> dict:
    """match_id의 타임라인 이벤트를 조회한다.

    USE_MOCK_DATA=true면 storage/matches/{match_id}/timeline_events.json을 읽어 반환한다.
    실DB 조회는 Session이 필요해 이 시그니처로는 구현할 수 없어 TODO로 남겨둔다
    (실 연동 시 Session을 받는 시그니처로 확장 필요).
    """
    settings = get_settings()

    if not settings.USE_MOCK_DATA:
        raise NotImplementedError(
            "실DB 기반 get_timeline_events는 아직 미구현입니다 (Session 인자가 필요해 시그니처 확장 필요)."
        )

    path = get_storage_root() / get_match_timeline_events_path(match_id)
    if not path.exists():
        raise FileNotFoundError(f"목업 timeline_events.json이 없습니다: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


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