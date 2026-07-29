from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.domains.auth.access import require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.auth.event_weights import resolve_event_weights
from app.domains.timeline.fusion import (
    build_frontend_event,
    normalize_timeline_category_filter,
)
from app.domains.timeline.dev_context import select_dev_timeline_events
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
        description=(
            "Frontend category. 예: goal, shot, penalty, card, "
            "substitution, corner"
        ),
    ),
    half: int | None = Query(default=None, ge=1, le=2),
    sort: Literal["time_asc", "importance_desc"] = Query(default="time_asc"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TimelineEventsResponse:
    match = require_match_access(db, match_id, current_user)
    backend_label = normalize_timeline_category_filter(category)

    repository = TimelineEventRepository(db)
    events = select_dev_timeline_events(
        repository.list_current_by_match(
            match_id=match_id,
            label=backend_label,
            half=half,
        ),
        match_id=match_id,
        settings=get_settings(),
    )
    event_weights = resolve_event_weights(current_user.event_weights)
    frontend_events = [
        build_frontend_event(
            event,
            match_duration_sec=match.duration_sec,
            event_weights=event_weights,
        )
        for event in events
    ]
    if sort == "importance_desc":
        frontend_events.sort(
            key=lambda event: event.importance_score,
            reverse=True,
        )

    return TimelineEventsResponse(
        match_id=match_id,
        events=frontend_events,
        count=len(frontend_events),
    )
