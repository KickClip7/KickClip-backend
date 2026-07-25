from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.analysis.model import AnalysisJob
from app.domains.analysis.repository import AnalysisJobRepository
from app.domains.auth.model import User
from app.domains.clip_plan.model import ClipPlan
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.match.mock import is_mock_match
from app.domains.match.model import Match
from app.domains.match.repository import MatchRepository
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.project.model import Project
from app.domains.project.repository import ProjectRepository
from app.domains.render.model import RenderJob
from app.domains.render.repository import RenderJobRepository


def can_access_project(user: User, project: Project) -> bool:
    return bool(user.developer_mode_enabled or project.owner_id == user.user_id)


def require_project_access(db: Session, project_id: str, user: User) -> Project:
    project = ProjectRepository(db).get_by_id(project_id)
    if project is None or not can_access_project(user, project):
        _not_found("Project")
    return project


def require_match_access(db: Session, match_id: str, user: User) -> Match:
    match = MatchRepository(db).get_by_id(match_id)
    if match is None:
        _not_found("Match")

    settings = get_settings()
    shared_mock_access = (
        settings.ENV in {"local", "dev", "test"}
        and settings.USE_MOCK_DATA
        and settings.MOCK_SHARED_ACCESS_ENABLED
        and is_mock_match(match)
    )
    if shared_mock_access:
        # 인증은 그대로 유지하되, 명시적으로 시드된 목업 Match만
        # 로컬/개발 환경의 모든 로그인 사용자에게 공유한다.
        return match

    require_project_access(db, match.project_id, user)
    return match


def require_media_access(db: Session, asset_id: str, user: User) -> MediaAsset:
    asset = MediaAssetRepository(db).get_by_id(asset_id)
    if asset is None:
        _not_found("Media asset")
    require_match_access(db, asset.match_id, user)
    return asset


def require_analysis_job_access(db: Session, job_id: str, user: User) -> AnalysisJob:
    job = AnalysisJobRepository(db).get_by_id(job_id)
    if job is None:
        _not_found("Analysis job")
    require_match_access(db, job.match_id, user)
    return job


def require_clip_plan_access(db: Session, clip_plan_id: str, user: User) -> ClipPlan:
    plan = ClipPlanRepository(db).get_by_id(clip_plan_id)
    if plan is None:
        _not_found("Clip plan")
    require_match_access(db, plan.match_id, user)
    return plan


def require_render_job_access(db: Session, render_job_id: str, user: User) -> RenderJob:
    render = RenderJobRepository(db).get_by_id(render_job_id)
    if render is None:
        _not_found("Render job")
    require_clip_plan_access(db, render.clip_plan_id, user)
    return render


def _not_found(resource: str) -> None:
    # 404 avoids revealing whether another user owns the identifier.
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{resource} not found",
    )
