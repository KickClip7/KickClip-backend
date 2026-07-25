from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_project_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.auth.event_weights import resolve_event_weights
from app.domains.media.signed_url import build_signed_media_url
from app.domains.project.schema import (
    ProjectCreate,
    ProjectRead,
    ProjectRecentListResponse,
)
from app.domains.project.service import ProjectService
from app.domains.studio.schema import ProjectEditStateResponse
from app.domains.studio.service import StudioService


router = APIRouter()


@router.get(
    "/recent",
    response_model=ProjectRecentListResponse,
    summary="최근 프로젝트 목록 조회",
)
def get_recent_projects(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectRecentListResponse:
    service = ProjectService(db)
    owner_id = None if current_user.developer_mode_enabled else current_user.user_id
    projects = service.list_recent_project_cards(limit=limit, owner_id=owner_id)
    return ProjectRecentListResponse(projects=projects)


@router.post(
    "",
    status_code=status.HTTP_410_GONE,
    summary="폐기된 프로젝트 생성 API",
    deprecated=True,
)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Use POST /api/v1/matches/{match_id}/projects instead.",
    )


@router.get(
    "/{project_id}",
    response_model=ProjectRead,
    summary="프로젝트 단건 조회",
)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectRead:
    return require_project_access(db, project_id, current_user)


@router.post(
    "/{project_id}/matches",
    status_code=status.HTTP_410_GONE,
    summary="폐기된 프로젝트→경기 생성 API",
    deprecated=True,
)
def create_match_for_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Matches are created by video upload and cannot be attached to projects.",
    )


@router.get(
    "/{project_id}/edit-state",
    response_model=ProjectEditStateResponse,
    summary="프로젝트별 편집 화면 초기 상태 조회",
)
def get_project_edit_state(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectEditStateResponse:
    require_project_access(db, project_id, current_user)
    edit_state = StudioService(db).get_project_edit_state(
        project_id,
        event_weights=resolve_event_weights(current_user.event_weights),
    )
    if edit_state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    if edit_state.video.asset_id is not None:
        edit_state.video.url = build_signed_media_url(
            edit_state.video.asset_id,
            current_user.user_id,
        )[0]
    return edit_state
