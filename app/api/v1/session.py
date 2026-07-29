from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.domains.auth.access import require_match_access, require_project_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.session.schema import (
    SessionChatRequest,
    SessionChatResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionStateResponse,
)
from app.domains.session.service import SessionNotFoundError, SessionService
from app.domains.timeline.dev_context import resolve_agent_match_id


router = APIRouter()


@router.post(
    "/start",
    response_model=SessionStartResponse,
    summary="DB timeline_events 기반 편집 세션 시작",
)
def start_session(
    data: SessionStartRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionStartResponse:
    match_id = resolve_agent_match_id(data.match_id, get_settings())
    require_match_access(db, match_id, current_user)
    if data.project_id is not None:
        project = require_project_access(db, data.project_id, current_user)
        if project.match_id != match_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_id does not belong to match_id",
            )
    try:
        result = SessionService(db).start(
            data.match_id,
            current_user.user_id,
            project_id=data.project_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionStartResponse(**result)


@router.post(
    "/chat",
    response_model=SessionChatResponse,
    summary="LangGraph 편집 워크플로우 그래프에 자연어 요청 전달",
)
def chat_session(
    data: SessionChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionChatResponse:
    try:
        result = SessionService(db).chat(
            data.session_id,
            data.message,
            user_id=current_user.user_id,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionChatResponse(**result)


@router.get(
    "/{session_id}/state",
    response_model=SessionStateResponse,
    summary="세션의 현재 클립 구성/전체 후보 조회",
)
def get_session_state(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionStateResponse:
    try:
        result = SessionService(db).get_state(
            session_id,
            user_id=current_user.user_id,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionStateResponse(**result)
