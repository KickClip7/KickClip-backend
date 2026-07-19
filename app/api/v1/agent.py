from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.agent.schema import (
    AgentClipPlanRequest,
    AgentClipPlanResponse,
    PlayerProfileResolveRequest,
    PlayerProfileResolveResponse,
)
from app.domains.agent.service import AgentService


router = APIRouter()


@router.post(
    "/clip-plan",
    response_model=AgentClipPlanResponse,
    summary="Agent 프롬프트 기반 ClipPlan 생성",
)
def create_agent_clip_plan(
    data: AgentClipPlanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgentClipPlanResponse:
    require_match_access(db, data.match_id, current_user)
    service = AgentService(db)

    try:
        return service.create_clip_plan(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/player-profile/resolve",
    response_model=PlayerProfileResolveResponse,
    summary="선수 프로필 연결",
)
def resolve_player_profile(
    data: PlayerProfileResolveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PlayerProfileResolveResponse:
    require_match_access(db, data.match_id, current_user)
    service = AgentService(db)

    try:
        return service.resolve_player_profile(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
