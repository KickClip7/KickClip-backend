from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.match.schema import MatchCreateRequest, MatchRead
from app.domains.match.service import MatchService
from app.domains.project.schema import (
    ProjectCreate,
    ProjectRead,
    ProjectRecentListResponse,
)
from app.domains.project.service import ProjectService


router = APIRouter()


@router.get(
    "/recent",
    response_model=ProjectRecentListResponse,
    summary="최근 프로젝트 목록 조회",
)
def get_recent_projects(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> ProjectRecentListResponse:
    service = ProjectService(db)
    projects = service.list_recent_project_cards(limit=limit)
    return ProjectRecentListResponse(projects=projects)


@router.post(
    "",
    response_model=ProjectRead,
    status_code=status.HTTP_201_CREATED,
    summary="프로젝트 생성",
)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
) -> ProjectRead:
    service = ProjectService(db)
    project = service.create_project(payload)
    return project


@router.get(
    "/{project_id}",
    response_model=ProjectRead,
    summary="프로젝트 단건 조회",
)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
) -> ProjectRead:
    service = ProjectService(db)
    project = service.get_project(project_id)

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return project


@router.post(
    "/{project_id}/matches",
    response_model=MatchRead,
    status_code=status.HTTP_201_CREATED,
    summary="프로젝트에 경기 생성",
)
def create_match_for_project(
    project_id: str,
    payload: MatchCreateRequest,
    db: Session = Depends(get_db),
) -> MatchRead:
    service = MatchService(db)

    try:
        match = service.create_match_for_project(
            project_id=project_id,
            data=payload,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return match