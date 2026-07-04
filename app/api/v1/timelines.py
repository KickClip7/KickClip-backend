from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.timeline.fusion import (
    build_frontend_event,
    normalize_timeline_category_filter,
)
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.timeline.schema import TimelineEventsResponse


router = APIRouter()


@router.get(
    "/matches/{match_id}/timeline-events",
    response_model=TimelineEventsResponse,
    summary="경기 타임라인 이벤트 목록 조회",
)
def list_match_timeline_events(
    match_id: str,
    category: str | None = Query(
        default=None,
        description="Frontend category. 예: goal, shot, foul, card, freekick, corner",
    ),
    half: int | None = Query(default=None, ge=1, le=2),
    db: Session = Depends(get_db),
) -> TimelineEventsResponse:
    backend_label = normalize_timeline_category_filter(category)

    repository = TimelineEventRepository(db)
    events = repository.list_by_match(
        match_id=match_id,
        label=backend_label,
        half=half,
    )
    frontend_events = [build_frontend_event(event) for event in events]

    return TimelineEventsResponse(
        match_id=match_id,
        events=frontend_events,
        count=len(frontend_events),
    )
