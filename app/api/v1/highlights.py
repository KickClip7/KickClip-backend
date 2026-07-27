from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ai.runtime.job_runner import run_analysis_job_background
from app.db.session import SessionLocal, get_db
from app.domains.auth.access import require_project_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.auth.repository import UserRepository
from app.domains.highlight.schema import (
    HighlightAnalyzeRequest,
    HighlightAnalyzeResponse,
    HighlightClipPlanRequest,
    HighlightClipPlanResponse,
    HighlightRenderRequest,
    HighlightRenderResponse,
    HighlightRevisionRead,
    HighlightSceneSelectionRequest,
    HighlightScenesResponse,
    HighlightStatusResponse,
    PlayerCandidatesResponse,
    PlayerConfirmationRequest,
    PlayerFocusSelectRequest,
    PlayerFocusStartRequest,
)
from app.domains.highlight.service import HighlightWorkflowService
from app.domains.render.service import RenderJobService


router = APIRouter()


@router.post(
    "/projects/{project_id}/highlight/analyze",
    response_model=HighlightAnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start or reuse Action Spotting and create a highlight revision",
)
def analyze_highlight(
    project_id: str,
    payload: HighlightAnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightAnalyzeResponse:
    project = require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        revision, reused, action_job = service.analyze(
            project=project,
            user_request=payload.request,
            structured_request=payload.structured_request,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)

    if action_job.status == "QUEUED":
        background_tasks.add_task(
            _run_action_spotting_and_reconcile,
            action_job.analysis_job_id,
            revision.revision_id,
        )
    elif revision.status == "PLAYER_DISCOVERY_RUNNING":
        background_tasks.add_task(
            _run_candidate_discovery,
            revision.revision_id,
        )
    return HighlightAnalyzeResponse(
        revision=service.revision_read(revision),
        action_spotting_reused=reused,
        action_spotting_status=action_job.status,
        status_url=f"/api/v1/projects/{project_id}/highlight/status",
        scenes_url=f"/api/v1/projects/{project_id}/highlight/scenes",
    )


@router.get(
    "/projects/{project_id}/highlight/scenes",
    response_model=HighlightScenesResponse,
)
def get_highlight_scenes(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightScenesResponse:
    require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        revision, scenes = service.scenes(project_id)
    except ValueError as exc:
        _raise_workflow_error(exc)
    return HighlightScenesResponse(
        revision=service.revision_read(revision),
        scenes=scenes,
    )


@router.post(
    "/projects/{project_id}/highlight/scenes/select",
    response_model=HighlightRevisionRead,
)
def select_highlight_scenes(
    project_id: str,
    payload: HighlightSceneSelectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightRevisionRead:
    project = require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        revision = service.select_scenes(
            project=project,
            scene_ids=payload.scene_ids,
            selection_source=payload.selection_source,
            render_strategies=payload.render_strategies,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)
    return service.revision_read(revision)


@router.post(
    "/projects/{project_id}/highlight/player-focus",
    response_model=HighlightRevisionRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_player_focus(
    project_id: str,
    payload: PlayerFocusStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightRevisionRead:
    project = require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        revision = service.start_player_focus(
            project=project,
            user_request=payload.request,
            source_revision_id=payload.revision_id,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)
    background_tasks.add_task(
        _run_candidate_discovery,
        revision.revision_id,
    )
    return service.revision_read(revision)


@router.get(
    "/projects/{project_id}/highlight/player-candidates",
    response_model=PlayerCandidatesResponse,
)
def get_player_candidates(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PlayerCandidatesResponse:
    require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        revision, candidates = service.candidates(project_id)
    except ValueError as exc:
        _raise_workflow_error(exc)
    return PlayerCandidatesResponse(
        revision=service.revision_read(revision),
        candidates=candidates,
    )


@router.post(
    "/projects/{project_id}/highlight/player-focus/select",
    response_model=HighlightStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def select_focus_subject(
    project_id: str,
    payload: PlayerFocusSelectRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightStatusResponse:
    project = require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        _, _, binding = service.select_focus_subject(
            project=project,
            revision_id=payload.revision_id,
            display_name=payload.display_name,
            anchor_scene_id=payload.anchor_scene_id,
            candidate_id=payload.candidate_id,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)
    background_tasks.add_task(
        _extract_and_queue_tracking,
        binding.binding_id,
        current_user.user_id,
    )
    return service.status(project_id)


@router.post(
    "/projects/{project_id}/highlight/scenes/{scene_id}/player-confirmation",
    response_model=HighlightStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def confirm_scene_player(
    project_id: str,
    scene_id: str,
    payload: PlayerConfirmationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightStatusResponse:
    project = require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        _, binding = service.confirm_scene_player(
            project=project,
            scene_id=scene_id,
            revision_id=payload.revision_id,
            decision=payload.decision,
            candidate_id=payload.candidate_id,
            render_strategy=payload.render_strategy,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)
    if payload.decision == "candidate" and binding.tracking_job_id is None:
        background_tasks.add_task(
            _extract_and_queue_tracking,
            binding.binding_id,
            current_user.user_id,
        )
    return service.status(project_id)


@router.get(
    "/projects/{project_id}/highlight/status",
    response_model=HighlightStatusResponse,
)
def get_highlight_status(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightStatusResponse:
    require_project_access(db, project_id, current_user)
    try:
        return HighlightWorkflowService(db).status(project_id)
    except ValueError as exc:
        _raise_workflow_error(exc)


@router.post(
    "/projects/{project_id}/highlight/clip-plan",
    response_model=HighlightClipPlanResponse,
)
def create_highlight_clip_plan(
    project_id: str,
    payload: HighlightClipPlanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightClipPlanResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        return HighlightWorkflowService(db).create_clip_plan(
            project=project,
            revision_id=payload.revision_id,
            allow_absent_full_frame=payload.allow_absent_full_frame,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)


@router.post(
    "/projects/{project_id}/highlight/render",
    response_model=HighlightRenderResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def render_highlight(
    project_id: str,
    payload: HighlightRenderRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightRenderResponse:
    project = require_project_access(db, project_id, current_user)
    service = HighlightWorkflowService(db)
    try:
        revision, render = service.create_render(
            project=project,
            revision_id=payload.revision_id,
            options=payload.options,
        )
    except ValueError as exc:
        _raise_workflow_error(exc)
    background_tasks.add_task(
        _run_render_and_reconcile,
        render.render_job_id,
        revision.revision_id,
    )
    return HighlightRenderResponse(
        revision=service.revision_read(revision),
        render_job_id=render.render_job_id,
        status=render.status,
    )


def _run_action_spotting_and_reconcile(
    analysis_job_id: str,
    revision_id: str,
) -> None:
    run_analysis_job_background(analysis_job_id)
    db = SessionLocal()
    try:
        service = HighlightWorkflowService(db)
        revision = service.repository.get_revision(revision_id)
        if revision is not None:
            revision = service.reconcile_revision(revision)
            if revision.focus_mode == "PLAYER" and revision.selected_scene_ids:
                revision.status = "PLAYER_DISCOVERY_RUNNING"
                revision.pending_action = "SELECT_PLAYER"
                db.commit()
                service.run_candidate_discovery(revision.revision_id)
    finally:
        db.close()


def _run_candidate_discovery(revision_id: str) -> None:
    db = SessionLocal()
    try:
        HighlightWorkflowService(db).run_candidate_discovery(revision_id)
    finally:
        db.close()


def _extract_and_queue_tracking(binding_id: str, user_id: str) -> None:
    db = SessionLocal()
    try:
        user = UserRepository(db).get_by_id(user_id)
        if user is None:
            return
        try:
            HighlightWorkflowService(db).start_tracking_for_binding(
                binding_id=binding_id,
                user=user,
            )
        except Exception:
            # The service persists the binding failure for status polling.
            return
    finally:
        db.close()


def _run_render_and_reconcile(render_job_id: str, revision_id: str) -> None:
    db = SessionLocal()
    try:
        RenderJobService(db).run_render_job(render_job_id)
        revision = HighlightWorkflowService(db).repository.get_revision(revision_id)
        if revision is not None:
            HighlightWorkflowService(db).reconcile_revision(revision)
    finally:
        db.close()


def _raise_workflow_error(exc: ValueError) -> None:
    detail = str(exc)
    status_code = (
        status.HTTP_404_NOT_FOUND
        if detail.endswith("not found.") or "has not been started" in detail
        else status.HTTP_409_CONFLICT
    )
    raise HTTPException(
        status_code=status_code,
        detail={"code": detail if detail == "NO_TARGET_SCENES" else "HIGHLIGHT_WORKFLOW_ERROR", "message": detail},
    ) from exc
