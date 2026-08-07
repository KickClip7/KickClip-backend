from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import (
    require_media_access,
    require_project_access,
    require_tracking_job_access,
)
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1.errors import (
    CandidateHandoffR1Error,
    CandidateRecommendationNotPrepared,
)
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateSelectionR1,
)
from app.domains.candidate_handoff_r1.preparation import (
    PREPARATION_TASK_TYPE,
    install_candidate_preparation_integration,
)
from app.domains.candidate_handoff_r1.initial_target_gallery import (
    INITIAL_TARGET_GALLERY_POLICY_VERSION,
)
from app.domains.candidate_handoff_r1.schema import (
    CandidateRecommendationPrepareRequest,
    CandidateReviewDecisionRequest,
    CandidateReviewResponse,
    EventCandidateRecommendationResponse,
    EventCandidateSelectionCreateRequest,
    EventCandidateSelectionRead,
    EventCandidateTrackingCreateRequest,
    EventCandidateTrackingCreateResponse,
)
from app.domains.candidate_handoff_r1.service import CandidateHandoffR1Service
from app.domains.highlight.model import HighlightRevision, SceneAITask
from app.domains.highlight.scene_ai_task import (
    SceneAITaskService,
    get_scene_ai_task_executor,
)
from app.domains.highlight.schema import (
    HighlightRevisionRead,
    HighlightSceneSelectionRequest,
    SceneAITaskRead,
)
from app.domains.highlight.service import HighlightWorkflowService
from app.domains.shot_boundary.service import ShotBoundaryReviewService
from app.storage.local_storage import LocalStorage

router = APIRouter()
install_candidate_preparation_integration()


def _prepare_url(
    project_id: str, revision_id: str, event_id: str, scene_id: str
) -> str:
    return (
        f"/api/v1/projects/{project_id}/highlight/revisions/{revision_id}"
        f"/events/{event_id}/candidate-recommendations/prepare"
        f"?scene_id={scene_id}"
    )


def _shot_boundary_urls(
    project_id: str, revision_id: str, event_id: str, scene_id: str
) -> tuple[str, str]:
    root = (
        f"/api/v1/projects/{project_id}/highlight/revisions/{revision_id}"
        f"/events/{event_id}/shot-boundaries"
    )
    return f"{root}/prepare?scene_id={scene_id}", f"{root}?scene_id={scene_id}"


def _require_candidate_discovery_inputs(
    db: Session,
    *,
    project,
    user: User,
    revision_id: str,
    event_id: str,
    scene_id: str,
) -> dict[str, str]:
    try:
        return ShotBoundaryReviewService(db).prepare_candidate_discovery_inputs(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
    except Exception as exc:
        if getattr(exc, "code", None) != "SHOT_BOUNDARY_REVIEW_REQUIRED":
            code = getattr(exc, "code", "CANDIDATE_DISCOVERY_INPUTS_NOT_READY")
            raise HTTPException(
                status_code=getattr(exc, "http_status", status.HTTP_409_CONFLICT),
                detail={
                    "code": code,
                    "message": str(exc),
                    "detail": dict(getattr(exc, "detail", {}) or {}),
                    "automatic_target_confirmation": False,
                },
            ) from exc
        prepare_url, status_url = _shot_boundary_urls(
            project.project_id, revision_id, event_id, scene_id
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "SHOT_BOUNDARY_REVIEW_REQUIRED",
                "message": (
                    "Automatic shot-boundary structural validation failed. "
                    "Manual correction is required only for this exceptional scene."
                ),
                "reason": "AUTOMATIC_BOUNDARY_STRUCTURAL_GATE_FAILED",
                "prepare_review_url": prepare_url,
                "review_status_url": status_url,
                "detail": dict(getattr(exc, "detail", {}) or {}),
                "automatic_target_confirmation": False,
            },
        ) from exc


def _matching_preparation_task(
    db: Session,
    *,
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
) -> SceneAITask | None:
    rows = db.scalars(
        select(SceneAITask).where(
            SceneAITask.project_id == project_id,
            SceneAITask.task_type == PREPARATION_TASK_TYPE,
        )
    ).all()
    matches = [
        task
        for task in rows
        if (task.payload or {}).get("revision_id") == revision_id
        and (task.payload or {}).get("event_id") == event_id
        and (task.payload or {}).get("scene_id") == scene_id
    ]
    matches.sort(key=lambda task: task.created_at, reverse=True)
    return matches[0] if matches else None


def _canonical_candidate_video(
    db: Session,
    *,
    project_id: str,
    revision_id: str,
    scene_id: str,
    requested_asset_id: str | None,
    user: User,
):
    revision = db.get(HighlightRevision, revision_id)
    if revision is None or revision.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "HIGHLIGHT_REVISION_NOT_FOUND",
                "message": "Highlight revision not found.",
            },
        )
    inputs = (revision.options or {}).get("candidate_pipeline_inputs") or {}
    canonical_id = str(inputs.get("scene_video_asset_id") or "")
    if inputs.get("scene_id") != scene_id or not canonical_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "CANDIDATE_SOURCE_VIDEO_NOT_READY",
                "message": (
                    "The immutable candidate scene video is not ready for "
                    "this revision and scene."
                ),
            },
        )
    if requested_asset_id:
        # Preserve access validation for legacy clients, but the immutable
        # revision contract remains authoritative over a stale Match video ID.
        require_media_access(db, requested_asset_id, user)
    return require_media_access(db, canonical_id, user)


def _raise(exc: Exception) -> None:
    code = getattr(exc, "code", "EVENT_CANDIDATE_HANDOFF_R1_ERROR")
    message = str(exc)
    if "INVALID_TARGET_MEMORY_REFERENCE" in message:
        code = "INVALID_TARGET_MEMORY_REFERENCE"
    detail = {"code": code, "message": message}
    reason = getattr(exc, "reason", None)
    if reason:
        detail["reason"] = reason
    raise HTTPException(
        status_code=(
            status.HTTP_503_SERVICE_UNAVAILABLE
            if code == "CANDIDATE_TRACKING_CONFIGURATION_INVALID"
            else (
                status.HTTP_422_UNPROCESSABLE_ENTITY
                if code
                in {
                    "INVALID_TARGET_MEMORY_REFERENCE",
                    "CANDIDATE_TRACKING_INPUT_INVALID",
                    "CANDIDATE_TRACKING_RUNTIME_CONTRACT_INVALID",
                }
                else status.HTTP_409_CONFLICT
            )
        ),
        detail=detail,
    ) from exc


@router.post(
    "/projects/{project_id}/highlight/scenes/select",
    response_model=HighlightRevisionRead,
    summary="Select scenes for automatic candidate discovery",
)
def select_highlight_scenes_and_prepare(
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
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "HIGHLIGHT_WORKFLOW_ERROR",
                "message": str(exc),
            },
        ) from exc
    readiness = []
    for scene_id in revision.selected_scene_ids or []:
        readiness.append(
            {
                "scene_id": scene_id,
                "status": "READY_FOR_AUTOMATIC_CANDIDATE_DISCOVERY",
                "boundary_origin": "AUTO_DETECTED",
                "human_reviewed": False,
                "automatic_target_confirmation": False,
            }
        )
    revision.options = {
        **(revision.options or {}),
        "candidate_discovery": {
            **((revision.options or {}).get("candidate_discovery") or {}),
            "shot_boundary_readiness": readiness,
            "recommendation_preparation": [],
            "automatic_target_confirmation": False,
        },
    }
    db.commit()
    db.refresh(revision)
    return service.revision_read(revision)


@router.get(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/events/{event_id}/candidate-recommendations",
    response_model=EventCandidateRecommendationResponse,
    summary="Serve the immutable V1.2 shortlist with review bundles",
)
def list_event_candidate_recommendations(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EventCandidateRecommendationResponse:
    project = require_project_access(db, project_id, current_user)
    _require_candidate_discovery_inputs(
        db,
        project=project,
        user=current_user,
        revision_id=revision_id,
        event_id=event_id,
        scene_id=scene_id,
    )
    try:
        return CandidateHandoffR1Service(db).list_recommendations(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
    except CandidateRecommendationNotPrepared as exc:
        task = _matching_preparation_task(
            db,
            project_id=project_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        running = task is not None and task.status in {"QUEUED", "RUNNING"}
        detail = {
            "code": exc.code,
            "message": (
                "Candidate recommendation preparation is running."
                if running
                else str(exc)
            ),
            "prepare_url": _prepare_url(project_id, revision_id, event_id, scene_id),
            "reason": "PREPARATION_RUNNING" if running else exc.reason,
        }
        if task is not None:
            detail["preparation_task"] = {
                "task_id": task.task_id,
                "status": task.status,
                "status_url": f"/api/v1/scene-ai-tasks/{task.task_id}",
                "error_message": task.error_message,
            }
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        ) from exc
    except (ValueError, CandidateHandoffR1Error) as exc:
        _raise(exc)


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/events/{event_id}/candidate-recommendations/prepare",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Prepare the immutable V1.2 shortlist and R1 review bundles",
)
def prepare_event_candidate_recommendations(
    project_id: str,
    revision_id: str,
    event_id: str,
    payload: CandidateRecommendationPrepareRequest,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    project = require_project_access(db, project_id, current_user)
    contract = _require_candidate_discovery_inputs(
        db,
        project=project,
        user=current_user,
        revision_id=revision_id,
        event_id=event_id,
        scene_id=scene_id,
    )
    task_service = SceneAITaskService(db)
    task, _ = task_service.enqueue(
        user=current_user,
        project=project,
        task_type=PREPARATION_TASK_TYPE,
        payload={
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "shortlist_size": payload.shortlist_size,
            # Included in the idempotency payload so an already-completed R1C
            # preparation cannot hide the wider temporal-diversity gallery.
            "initial_target_gallery_policy": (
                INITIAL_TARGET_GALLERY_POLICY_VERSION
            ),
            "shot_boundaries_artifact_id": contract[
                "shot_boundaries_artifact_id"
            ],
            "shot_boundaries_sha256": contract["shot_boundaries_sha256"],
            "detections_artifact_id": contract["detections_artifact_id"],
            "boundary_origin": contract["boundary_origin"],
            "automatic_target_confirmation": False,
        },
    )
    if task.status == "QUEUED":
        get_scene_ai_task_executor().submit(task.task_id)
    return task_service.read(task)


@router.get(
    "/event-candidate-recommendations/{shortlist_patch_id}"
    "/candidates/{candidate_id}/media/{media_name}",
    summary="Download one allowlisted immutable candidate review asset",
)
def candidate_review_media(
    shortlist_patch_id: str,
    candidate_id: str,
    media_name: str,
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        path, mime = CandidateHandoffR1Service(db).resolve_media(
            project=project,
            shortlist_patch_id=shortlist_patch_id,
            candidate_id=candidate_id,
            media_name=media_name,
        )
    except (ValueError, CandidateHandoffR1Error) as exc:
        _raise(exc)
    return FileResponse(path, media_type=mime, filename=path.name)


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/events/{event_id}/candidate-selections",
    response_model=EventCandidateSelectionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_event_candidate_selection(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    payload: EventCandidateSelectionCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EventCandidateSelectionRead:
    project = require_project_access(db, project_id, current_user)
    video = _canonical_candidate_video(
        db,
        project_id=project_id,
        revision_id=revision_id,
        scene_id=scene_id,
        requested_asset_id=payload.source_video_asset_id,
        user=current_user,
    )
    service = CandidateHandoffR1Service(db)
    try:
        selection = service.create_selection(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            ranking_id=payload.ranking_id,
            shortlist_patch_id=payload.shortlist_patch_id,
            candidate_id=payload.candidate_id,
            source_video=video,
            user=current_user,
        )
    except (ValueError, CandidateHandoffR1Error) as exc:
        _raise(exc)
    return service.selection_read(selection)


@router.post(
    "/event-candidate-selections/{selection_id}/tracking-jobs",
    response_model=EventCandidateTrackingCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_selected_candidate_tracking(
    selection_id: str,
    payload: EventCandidateTrackingCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EventCandidateTrackingCreateResponse:
    selection = db.get(EventCandidateSelectionR1, selection_id)
    if selection is None:
        raise HTTPException(status_code=404, detail="Selection not found.")
    project = require_project_access(db, selection.project_id, current_user)
    video = _canonical_candidate_video(
        db,
        project_id=selection.project_id,
        revision_id=selection.revision_id,
        scene_id=selection.scene_id,
        requested_asset_id=payload.source_video_asset_id,
        user=current_user,
    )
    try:
        return CandidateHandoffR1Service(db).create_tracking_handoff(
            selection=selection,
            source_video=video,
            project=project,
            user=current_user,
        )
    except (ValueError, CandidateHandoffR1Error) as exc:
        _raise(exc)


@router.post(
    "/tracking/jobs/{job_id}/ambiguities/{ambiguity_id}/candidate-review",
    status_code=status.HTTP_201_CREATED,
    response_model=CandidateReviewResponse,
)
def record_candidate_review_state(
    job_id: str,
    ambiguity_id: str,
    payload: CandidateReviewDecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    job = require_tracking_job_access(db, job_id, current_user)
    try:
        row = CandidateHandoffR1Service(db).record_review_decision(
            job=job,
            ambiguity_id=ambiguity_id,
            request=payload,
            user=current_user,
        )
    except (ValueError, CandidateHandoffR1Error) as exc:
        _raise(exc)
    return {
        "decision_id": row.decision_id,
        "state": row.decision_state,
        "candidate_id": row.candidate_id,
        "artifact_sha256": row.decision_artifact_sha256,
        "continuation": (row.metadata_ or {}).get("continuation", "TRACKING_RESUMED"),
        "remaining_candidate_count": int(
            (row.metadata_ or {}).get("remaining_candidate_count") or 0
        ),
        "next_candidate_id": (row.metadata_ or {}).get("next_candidate_id"),
        "next_ambiguity_id": (row.metadata_ or {}).get("next_ambiguity_id"),
        "automatic_target_confirmation": False,
    }


@router.get(
    "/tracking/jobs/{job_id}/ambiguities/{ambiguity_id}/evidence/{evidence_name}",
    summary="Download authenticated immutable R1 ambiguity evidence",
)
def get_ambiguity_evidence(
    job_id: str,
    ambiguity_id: str,
    evidence_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    require_tracking_job_access(db, job_id, current_user)
    ambiguity = db.scalar(
        select(EventCandidateAmbiguityR1).where(
            EventCandidateAmbiguityR1.tracking_job_id == job_id,
            EventCandidateAmbiguityR1.ambiguity_id == ambiguity_id,
        )
    )
    if ambiguity is None:
        raise HTTPException(status_code=404, detail="Ambiguity not found.")
    if evidence_name == "full-frame-context":
        relative = ambiguity.full_frame_context_path
        mime = "image/jpeg"
    elif evidence_name == "full-shot-clip":
        relative = ambiguity.shot_clip_path
        mime = "video/mp4"
    else:
        raise HTTPException(status_code=404, detail="Evidence not found.")
    path = LocalStorage().resolve_path(relative)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence artifact is missing.")
    return FileResponse(path, media_type=mime, filename=path.name)


@router.get(
    "/tracking/jobs/{job_id}/ambiguities/{ambiguity_id}"
    "/candidates/{candidate_id}/media/{media_name}",
    summary="Download authenticated immutable R1 candidate review media",
)
def get_ambiguity_candidate_media(
    job_id: str,
    ambiguity_id: str,
    candidate_id: str,
    media_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    require_tracking_job_access(db, job_id, current_user)
    ambiguity = db.scalar(
        select(EventCandidateAmbiguityR1).where(
            EventCandidateAmbiguityR1.tracking_job_id == job_id,
            EventCandidateAmbiguityR1.ambiguity_id == ambiguity_id,
        )
    )
    if ambiguity is None:
        raise HTTPException(status_code=404, detail="Ambiguity not found.")
    candidate = next(
        (
            dict(row)
            for row in ambiguity.candidates or []
            if str(row.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    media_fields = {
        "reference-gallery": ("reference_gallery_path", "image/jpeg"),
        "full-frame-context": ("full_frame_context_path", "image/jpeg"),
    }
    field = media_fields.get(media_name)
    if field is None:
        raise HTTPException(status_code=404, detail="Candidate media not found.")
    path = LocalStorage().resolve_path(str(candidate.get(field[0]) or ""))
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail="Candidate media artifact is missing."
        )
    return FileResponse(path, media_type=field[1], filename=path.name)
