from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, get_db
from app.domains.auth.access import require_clip_plan_access, require_render_job_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.render.schema import (
    RenderCreateRequest,
    RenderCreateResponse,
    RenderJobRead,
)
from app.domains.render.service import RenderJobService


router = APIRouter()


@router.post(
    "/renders",
    response_model=RenderCreateResponse,
    summary="렌더링 작업 시작",
)
def create_render_job(
    data: RenderCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RenderCreateResponse:
    require_clip_plan_access(db, data.clip_plan_id, current_user)
    service = RenderJobService(db)

    try:
        render_job = service.create_render_job(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(_run_render_job_background, render_job.render_job_id)

    return RenderCreateResponse(
        render_job_id=render_job.render_job_id,
        status=render_job.status,
        progress=render_job.progress,
        reused=bool(getattr(render_job, "reused", False)),
    )


@router.get(
    "/renders/{render_job_id}",
    response_model=RenderJobRead,
    summary="렌더링 작업 조회",
)
def get_render_job(
    render_job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RenderJobRead:
    require_render_job_access(db, render_job_id, current_user)
    service = RenderJobService(db)
    render_job = service.get_render_job(render_job_id)

    if render_job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="RenderJob not found",
        )

    response = RenderJobRead.model_validate(render_job)
    response.retryable = render_job.status == "failed"
    response.download_url = (
        f"/api/v1/renders/{render_job.render_job_id}/download"
        if render_job.output_artifact_id and render_job.status == "completed"
        else None
    )
    return response


@router.get(
    "/renders/{render_job_id}/download",
    summary="렌더링 결과 다운로드",
)
def download_rendered_video(
    render_job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    require_render_job_access(db, render_job_id, current_user)
    service = RenderJobService(db)
    result = service.get_output_artifact_path(render_job_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rendered output not found",
        )

    render_job, file_path = result
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rendered file not found on storage",
        )

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"{render_job.render_job_id}.mp4",
    )


def _run_render_job_background(render_job_id: str) -> None:
    db = SessionLocal()
    try:
        service = RenderJobService(db)
        service.run_render_job(render_job_id)
    finally:
        db.close()
