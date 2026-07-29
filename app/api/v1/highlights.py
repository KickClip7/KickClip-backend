from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)
from sqlalchemy.orm import Session

from app.ai.runtime.job_runner import run_analysis_job_background
from app.db.session import SessionLocal, get_db
from app.domains.auth.access import (
    require_artifact_access,
    require_media_access,
    require_project_access,
)
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.auth.repository import UserRepository
from app.domains.highlight.draft_service import (
    HighlightDraftService,
    HighlightDraftVersionConflict,
)
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.scene_target_selection import (
    SceneTargetSelectionService,
)
from app.domains.highlight.schema import (
    EarlierAnchorConfirmationRequest,
    HighlightAnalyzeRequest,
    HighlightAnalyzeResponse,
    HighlightClipPlanRequest,
    HighlightClipPlanResponse,
    HighlightDraftRead,
    HighlightDraftSaveRequest,
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
    SceneTargetSelectionCreateRequest,
    SceneTargetSelectionRead,
    SceneTargetTrackingCreateRequest,
    SceneWideCandidateRead,
    SceneWideCandidateDiscoveryRequest,
    SceneWideCandidateGalleryResponse,
)
from app.domains.highlight.service import HighlightWorkflowService
from app.domains.render.service import RenderJobService
from app.domains.tracking.schema import TrackingJobCreateResponse

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
        revision, candidates = service.candidates(
            project_id,
            user_id=current_user.user_id,
        )
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


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/player-candidates/discover",
    response_model=SceneWideCandidateGalleryResponse,
    summary="Discover scene-wide, shot-local player candidates",
)
def discover_scene_wide_candidates(
    project_id: str,
    revision_id: str,
    payload: SceneWideCandidateDiscoveryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneWideCandidateGalleryResponse:
    project = require_project_access(db, project_id, current_user)
    video = require_media_access(
        db, payload.scene_video_asset_id, current_user
    )
    boundaries = require_artifact_access(
        db, payload.shot_boundaries_artifact_id, current_user
    )
    detections = require_artifact_access(
        db, payload.detections_artifact_id, current_user
    )
    service = SceneTargetSelectionService(db)
    try:
        service.discover(
            project=project,
            revision_id=revision_id,
            scene_id=payload.scene_id,
            scene_video=video,
            shot_boundaries_artifact=boundaries,
            detections_artifact=detections,
        )
        revision, rows = service.candidate_rows(
            project=project,
            revision_id=revision_id,
            scene_id=payload.scene_id,
        )
    except (ValueError, RuntimeError) as exc:
        _raise_scene_selection_error(exc)
    return _scene_candidate_gallery(revision, payload.scene_id, rows)


@router.get(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/player-candidates",
    response_model=SceneWideCandidateGalleryResponse,
    summary="List scene-wide candidates in shot/time order",
)
def list_scene_wide_candidates(
    project_id: str,
    revision_id: str,
    scene_id: str = Query(min_length=1, max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    show_hidden: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneWideCandidateGalleryResponse:
    project = require_project_access(db, project_id, current_user)
    service = SceneTargetSelectionService(db)
    try:
        revision, rows = service.candidate_rows(
            project=project,
            revision_id=revision_id,
            scene_id=scene_id,
        )
    except ValueError as exc:
        _raise_scene_selection_error(exc)
    if not show_hidden:
        rows = [
            row for row in rows
            if row["gallery_visibility"] == "VISIBLE"
        ]
    start = (page - 1) * page_size
    return _scene_candidate_gallery(
        revision,
        scene_id,
        rows[start : start + page_size],
        total_count=len(rows),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/player-candidates/{candidate_id}",
    response_model=SceneWideCandidateRead,
)
def get_scene_wide_candidate(
    project_id: str,
    revision_id: str,
    candidate_id: str,
    scene_id: str = Query(min_length=1, max_length=64),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneWideCandidateRead:
    project = require_project_access(db, project_id, current_user)
    service = SceneTargetSelectionService(db)
    try:
        _, rows = service.candidate_rows(
            project=project,
            revision_id=revision_id,
            scene_id=scene_id,
        )
        row = next(
            item for item in rows
            if item["candidate_id"] == candidate_id
        )
    except (ValueError, StopIteration) as exc:
        _raise_scene_selection_error(
            ValueError("Scene-wide candidate not found.")
        )
    return SceneWideCandidateRead(**row)


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/target-selections",
    response_model=SceneTargetSelectionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an immutable scene-wide target selection revision",
)
def create_scene_target_selection(
    project_id: str,
    revision_id: str,
    payload: SceneTargetSelectionCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneTargetSelectionRead:
    project = require_project_access(db, project_id, current_user)
    video = require_media_access(
        db, payload.scene_video_asset_id, current_user
    )
    service = SceneTargetSelectionService(db)
    try:
        selection = service.create_selection(
            project=project,
            revision_id=revision_id,
            scene_id=payload.scene_id,
            candidate_id=payload.candidate_id,
            scene_video=video,
            user=current_user,
        )
    except (ValueError, RuntimeError) as exc:
        _raise_scene_selection_error(exc)
    return service.read(selection)


@router.get(
    "/target-selections/{selection_id}",
    response_model=SceneTargetSelectionRead,
)
def get_scene_target_selection(
    selection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneTargetSelectionRead:
    service, selection = _owned_target_selection(
        db, selection_id, current_user
    )
    return service.read(selection)


@router.post(
    "/target-selections/{selection_id}/earlier-candidates/discover",
    response_model=SceneTargetSelectionRead,
    summary="Blind-rank earlier local candidates without auto-confirming",
)
def discover_earlier_anchor_candidates(
    selection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneTargetSelectionRead:
    service, selection = _owned_target_selection(
        db, selection_id, current_user, for_update=True
    )
    try:
        selection = service.discover_earlier(
            selection=selection,
            user=current_user,
        )
    except (ValueError, RuntimeError) as exc:
        _raise_scene_selection_error(exc)
    return service.read(selection)


@router.post(
    "/target-selections/{selection_id}/earlier-anchor/confirm",
    response_model=SceneTargetSelectionRead,
)
def decide_earlier_anchor(
    selection_id: str,
    payload: EarlierAnchorConfirmationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneTargetSelectionRead:
    service, selection = _owned_target_selection(
        db, selection_id, current_user, for_update=True
    )
    try:
        selection = service.decide_earlier(
            selection=selection,
            user=current_user,
            decision=payload.decision,
            candidate_id=payload.candidate_id,
        )
    except (ValueError, RuntimeError) as exc:
        _raise_scene_selection_error(exc)
    return service.read(selection)


@router.post(
    "/target-selections/{selection_id}/tracking-jobs",
    response_model=TrackingJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_scene_target_tracking_job(
    selection_id: str,
    payload: SceneTargetTrackingCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingJobCreateResponse:
    service, selection = _owned_target_selection(
        db, selection_id, current_user, for_update=True
    )
    project = require_project_access(
        db, selection.project_id, current_user
    )
    video = require_media_access(
        db, payload.scene_video_asset_id, current_user
    )
    boundaries = require_artifact_access(
        db, payload.shot_boundaries_artifact_id, current_user
    )
    try:
        return service.create_tracking_job(
            selection=selection,
            project=project,
            user=current_user,
            scene_video=video,
            shot_boundaries_artifact=boundaries,
        )
    except (ValueError, RuntimeError) as exc:
        _raise_scene_selection_error(exc)


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


@router.get(
    "/projects/{project_id}/highlight/draft",
    response_model=HighlightDraftRead,
    summary="하이라이트 편집 임시저장 조회",
)
def get_highlight_draft(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightDraftRead:
    require_project_access(db, project_id, current_user)
    service = HighlightDraftService(db)
    draft = service.get(project_id)
    if draft is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "HIGHLIGHT_DRAFT_NOT_FOUND",
                "message": "No saved highlight draft exists for this project.",
            },
        )
    return service.read(draft)


@router.put(
    "/projects/{project_id}/highlight/draft",
    response_model=HighlightDraftRead,
    summary="하이라이트 편집 임시저장 생성 또는 갱신",
)
def save_highlight_draft(
    project_id: str,
    payload: HighlightDraftSaveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HighlightDraftRead:
    project = require_project_access(db, project_id, current_user)
    service = HighlightDraftService(db)
    try:
        draft = service.save(
            project=project,
            user_id=current_user.user_id,
            data=payload,
        )
    except HighlightDraftVersionConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "HIGHLIGHT_DRAFT_VERSION_CONFLICT",
                "message": str(exc),
                "expected_version": exc.expected,
                "actual_version": exc.actual,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "HIGHLIGHT_DRAFT_INVALID",
                "message": str(exc),
            },
        ) from exc
    return service.read(draft)


@router.delete(
    "/projects/{project_id}/highlight/draft",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="하이라이트 편집 임시저장 삭제",
)
def delete_highlight_draft(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    require_project_access(db, project_id, current_user)
    HighlightDraftService(db).delete(project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
        reused=bool(getattr(render, "reused", False)),
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


def _scene_candidate_gallery(
    revision,
    scene_id: str,
    rows: list[dict],
    *,
    total_count: int | None = None,
    page: int = 1,
    page_size: int | None = None,
) -> SceneWideCandidateGalleryResponse:
    contract = (revision.options or {}).get("scene_target_selection") or {}
    total = len(rows) if total_count is None else total_count
    return SceneWideCandidateGalleryResponse(
        revision_id=revision.revision_id,
        scene_id=scene_id,
        status=str(contract.get("status") or revision.status),
        shot_count=int(contract.get("shot_count") or 0),
        candidate_count=total,
        candidates=[SceneWideCandidateRead(**row) for row in rows],
        pagination={
            "page": page,
            "page_size": page_size or max(1, len(rows)),
            "total": total,
            "has_more": (
                page_size is not None and page * page_size < total
            ),
        },
    )


def _owned_target_selection(
    db: Session,
    selection_id: str,
    user: User,
    *,
    for_update: bool = False,
):
    repository = HighlightRepository(db)
    selection = (
        repository.get_target_selection_for_update(selection_id)
        if for_update
        else repository.get_target_selection(selection_id)
    )
    if selection is None or (
        selection.owner_id != user.user_id
        and not user.developer_mode_enabled
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TARGET_SELECTION_NOT_FOUND",
                "message": "Target selection not found.",
            },
        )
    return SceneTargetSelectionService(db), selection


def _raise_scene_selection_error(
    exc: ValueError | RuntimeError,
) -> None:
    message = str(exc)
    if "not found" in message.lower():
        code = status.HTTP_404_NOT_FOUND
    elif "WAITING_SHOT_BOUNDARY_REVIEW" in message:
        code = status.HTTP_409_CONFLICT
    else:
        code = status.HTTP_422_UNPROCESSABLE_ENTITY
    raise HTTPException(
        status_code=code,
        detail={
            "code": (
                "WAITING_SHOT_BOUNDARY_REVIEW"
                if "WAITING_SHOT_BOUNDARY_REVIEW" in message
                else "SCENE_TARGET_SELECTION_ERROR"
            ),
            "message": message,
        },
    ) from exc


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
