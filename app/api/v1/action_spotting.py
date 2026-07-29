from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.ai.registry.model_registry import ModelRegistry
from app.ai.runtime.job_runner import run_analysis_job_background
from app.ai.tasks.highlight_spotting.adapters.soccer_spotter_v9 import (
    SoccerSpotterV9Adapter,
)
from app.core.paths import get_project_root
from app.db.session import get_db
from app.domains.auth.access import require_analysis_job_access, require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.action_spotting.schema import (
    ActionSpottingDiagnosticsResponse,
    ActionSpottingEventsResponse,
    ActionSpottingJobRequest,
    ActionSpottingJobResponse,
    ActionSpottingModelResponse,
)
from app.domains.action_spotting.diagnostics import (
    collect_action_spotting_diagnostics,
)
from app.domains.action_spotting.status import (
    action_spotting_public_error,
    action_spotting_workflow_status,
)
from app.domains.analysis.service import AnalysisJobService
from app.domains.highlight.action_cache import ActionSpottingCacheService
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.timeline.schema import TimelineEventRead


router = APIRouter()


@router.get(
    "/model",
    response_model=ActionSpottingModelResponse,
    summary="액션 스팟팅 모델 준비 상태",
)
def get_action_spotting_model() -> ActionSpottingModelResponse:
    registry = ModelRegistry()
    model_card = registry.get_model_card("highlight_spotting", "champion")
    checkpoint_path = _resolve_path(model_card.checkpoint_path)
    adapter = SoccerSpotterV9Adapter.from_model_dir(
        checkpoint_path.parent if checkpoint_path else None
    )
    preflight = adapter.preflight()

    return ActionSpottingModelResponse(
        ready=model_card.enabled and preflight.ready_for_real_adapter,
        model_id=model_card.id,
        model_name=model_card.model_name,
        model_version=model_card.model_version,
        classes=adapter.spec.labels,
        checkpoint_sha256=_sha256(checkpoint_path),
        device_requested="auto",
        preflight=preflight.to_metadata(),
    )


@router.get(
    "/diagnostics",
    response_model=ActionSpottingDiagnosticsResponse,
    summary="Champion Action Spotting runtime diagnostics",
)
def get_action_spotting_diagnostics(
    db: Session = Depends(get_db),
) -> ActionSpottingDiagnosticsResponse:
    return ActionSpottingDiagnosticsResponse(
        diagnostics=collect_action_spotting_diagnostics(db)
    )


@router.post(
    "/matches/{match_id}/jobs",
    response_model=ActionSpottingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="액션 스팟팅 작업 시작",
)
def create_action_spotting_job(
    match_id: str,
    payload: ActionSpottingJobRequest,
    background_tasks: BackgroundTasks,
    auto_start: bool = Query(default=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ActionSpottingJobResponse:
    require_match_access(db, match_id, current_user)
    try:
        job, reused = ActionSpottingCacheService(db).get_or_create(
            match_id=match_id,
            request_options=payload.to_job_options(),
        )
        created = AnalysisJobService(db).to_create_response(job)
    except ValueError as exc:
        code = status.HTTP_404_NOT_FOUND if str(exc) == "Match not found" else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    if auto_start and job.status == "QUEUED":
        background_tasks.add_task(
            run_analysis_job_background,
            created.analysis_job_id,
        )

    return ActionSpottingJobResponse(
        **created.model_dump(),
        status_url=f"/api/v1/analysis-jobs/{created.analysis_job_id}",
        events_url=f"/api/v1/action-spotting/jobs/{created.analysis_job_id}/events",
        workflow_status=action_spotting_workflow_status(
            status=job.status,
            options=job.options,
        ),
        workflow_state_history=list(
            (job.options or {}).get("action_spotting_state_history") or []
        ),
        error=action_spotting_public_error(job.options),
        cache_reused=reused,
    )


@router.get(
    "/jobs/{job_id}/events",
    response_model=ActionSpottingEventsResponse,
    summary="액션 스팟팅 결과 조회",
)
def get_action_spotting_events(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ActionSpottingEventsResponse:
    require_analysis_job_access(db, job_id, current_user)
    analysis = AnalysisJobService(db).get_analysis_job_status(job_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found",
        )
    if analysis.job_type != "HIGHLIGHT_SPOTTING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The requested job is not an action spotting job",
        )

    rows = TimelineEventRepository(db).list_by_source_job(job_id)
    events = [TimelineEventRead.model_validate(row) for row in rows]
    return ActionSpottingEventsResponse(
        analysis_job_id=job_id,
        status=analysis.status,
        count=len(events),
        events=events,
        workflow_status=action_spotting_workflow_status(
            status=analysis.status,
            options=analysis.options,
        ),
        workflow_state_history=list(
            analysis.options.get("action_spotting_state_history") or []
        ),
        error=action_spotting_public_error(analysis.options),
    )


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else get_project_root() / path


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    from app.ai.tasks.highlight_spotting.adapters.soccer_spotter_v9 import (
        sha256_file,
    )

    return sha256_file(path)
