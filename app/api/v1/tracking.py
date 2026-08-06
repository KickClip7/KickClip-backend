from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import (
    require_media_access,
    require_project_access,
    require_tracking_job_access,
)
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.errors import (
    TrackingContractError,
    TrackingError,
    TrackingValidationError,
)
from app.domains.tracking.executor import get_tracking_executor
from app.domains.tracking.execution import R1_EXECUTION_KIND
from app.domains.tracking.r1_executor import get_r1_tracking_executor
from app.domains.tracking.schema import (
    TrackingAmbiguityConfirmationRequest,
    TrackingArtifactsResponse,
    TrackingDiagnosticsResponse,
    TrackingJobCreateRequest,
    TrackingJobCreateResponse,
    TrackingJobResponse,
    TrackingReviewRequest,
    TrackingTimelineResponse,
)
from app.domains.tracking.service import TrackingJobService
from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.timeline import TrackingTimelineService
from app.domains.tracking.verifier import (
    get_scene_target_tracking_verifier,
    get_tracking_verifier,
)


router = APIRouter()


def _submit_tracking_job(job) -> bool:
    if job.execution_kind == R1_EXECUTION_KIND:
        return get_r1_tracking_executor().submit(job.tracking_job_id)
    return get_tracking_executor().submit(job.tracking_job_id)


@router.get(
    "/diagnostics",
    response_model=TrackingDiagnosticsResponse,
    summary="Target tracking runtime diagnostics",
    description=(
        "Runs or returns the cached frozen-runtime verifier result. "
        "A failed verifier disables tracking jobs without stopping the backend."
    ),
)
def tracking_diagnostics(
    refresh: bool = Query(default=False),
) -> TrackingDiagnosticsResponse:
    result = get_tracking_verifier().check(force=refresh)
    return TrackingDiagnosticsResponse(
        enabled=result.enabled,
        available=result.available,
        checked_at=result.checked_at,
        code=result.code,
        message=result.message,
        verifier_return_code=result.verifier_return_code,
        components=dict(result.components or {}),
    )


@router.get(
    "/diagnostics/scene-target-r3",
    response_model=TrackingDiagnosticsResponse,
    summary="Selection-assisted R3 runtime diagnostics",
)
def scene_target_r3_diagnostics(
    refresh: bool = Query(default=False),
) -> TrackingDiagnosticsResponse:
    result = get_scene_target_tracking_verifier().check(force=refresh)
    return TrackingDiagnosticsResponse(
        enabled=result.enabled,
        available=result.available,
        checked_at=result.checked_at,
        code=result.code,
        message=result.message,
        verifier_return_code=result.verifier_return_code,
        components=dict(result.components or {}),
    )


@router.post(
    "/jobs",
    response_model=TrackingJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start target-centric tracking",
    description=(
        "Queues GPU tracking and immediately returns a job ID. The client selects "
        "a server-owned MediaAsset and a source-resolution xyxy bbox; filesystem "
        "paths are never accepted."
    ),
)
def create_tracking_job(
    payload: TrackingJobCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingJobCreateResponse:
    asset = require_media_access(db, payload.media_asset_id, current_user)
    project = (
        require_project_access(db, payload.project_id, current_user)
        if payload.project_id
        else None
    )
    try:
        response = TrackingJobService(db).create_job(
            user=current_user,
            asset=asset,
            payload=payload,
            project=project,
        )
    except TrackingError as exc:
        _raise_tracking_http_error(exc)
    except ValueError as exc:
        _raise_tracking_http_error(
            TrackingValidationError("MediaAsset storage path is invalid.")
        )
    get_tracking_executor().submit(response.job_id)
    return response


@router.get(
    "/jobs/{job_id}",
    response_model=TrackingJobResponse,
    summary="Get target tracking job",
    description=(
        "NEEDS_CONFIRMATION is represented by one of the WAITING_* states; "
        "COMPLETE_WITH_SAFE_BLOCK is a successful safety outcome."
    ),
)
def get_tracking_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingJobResponse:
    job = require_tracking_job_access(db, job_id, current_user)
    try:
        return TrackingJobService(db).to_response(job)
    except TrackingError as exc:
        _raise_tracking_http_error(exc)
        raise AssertionError("unreachable") from exc


@router.post(
    "/jobs/{job_id}/reviews",
    response_model=TrackingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume a visual review",
    description=(
        "Approves or rejects only the exact MEMORY/SEGMENT/STAGE2* review currently "
        "requested by pipeline_state.json. Duplicate identical actions are idempotent."
    ),
)
def review_tracking_job(
    job_id: str,
    payload: TrackingReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingJobResponse:
    require_tracking_job_access(db, job_id, current_user)
    service = TrackingJobService(db)
    try:
        job = service.queue_review(
            tracking_job_id=job_id,
            user=current_user,
            payload=payload,
        )
    except TrackingError as exc:
        _raise_tracking_http_error(exc)
    if job.status == TrackingBackendStatus.QUEUED.value:
        _submit_tracking_job(job)
    return service.to_response(job)


@router.post(
    "/jobs/{job_id}/ambiguities/{ambiguity_id}/confirm",
    response_model=TrackingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Confirm a cross-shot candidate or target absence",
    description=(
        "candidate_id must be one of the current pipeline review candidates. "
        "No top-ranked candidate is auto-approved in assisted mode."
    ),
)
def confirm_tracking_ambiguity(
    job_id: str,
    ambiguity_id: str,
    payload: TrackingAmbiguityConfirmationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingJobResponse:
    require_tracking_job_access(db, job_id, current_user)
    service = TrackingJobService(db)
    try:
        job = service.queue_ambiguity_confirmation(
            tracking_job_id=job_id,
            ambiguity_id=ambiguity_id,
            user=current_user,
            payload=payload,
        )
    except TrackingError as exc:
        _raise_tracking_http_error(exc)
    if job.status == TrackingBackendStatus.QUEUED.value:
        _submit_tracking_job(job)
    return service.to_response(job)


@router.get(
    "/jobs/{job_id}/timeline",
    response_model=TrackingTimelineResponse,
    summary="Get frame-level target timeline",
    description=(
        "Returns the original JSON artifact with internal paths redacted. "
        "start_frame/end_frame are inclusive; uncertain frames retain null bbox."
    ),
)
def get_tracking_timeline(
    job_id: str,
    start_frame: int | None = Query(default=None, ge=0),
    end_frame: int | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingTimelineResponse:
    if (
        start_frame is not None
        and end_frame is not None
        and end_frame < start_frame
    ):
        _raise_tracking_http_error(
            TrackingValidationError("end_frame must be >= start_frame.")
        )
    job = require_tracking_job_access(db, job_id, current_user)
    try:
        payload = TrackingTimelineService().read(
            job,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return TrackingTimelineResponse.model_validate(payload)
    except TrackingError as exc:
        _raise_tracking_http_error(exc)
    except ValidationError as exc:
        _raise_tracking_http_error(
            TrackingContractError(
                "Tracking timeline does not match the public response schema."
            )
        )
        raise AssertionError("unreachable") from exc


@router.get(
    "/jobs/{job_id}/artifacts",
    response_model=TrackingArtifactsResponse,
    summary="List authenticated tracking artifacts",
)
def list_tracking_artifacts(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingArtifactsResponse:
    job = require_tracking_job_access(db, job_id, current_user)
    return TrackingJobService(db).artifacts_response(job)


@router.get(
    "/jobs/{job_id}/artifacts/{artifact_key}",
    summary="Download an allowlisted tracking artifact",
    description=(
        "The artifact key must have been generated from this job's pipeline state; "
        "arbitrary filenames and paths are rejected."
    ),
)
def download_tracking_artifact(
    job_id: str,
    artifact_key: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    job = require_tracking_job_access(db, job_id, current_user)
    try:
        path, mime_type, filename = TrackingArtifactService().resolve(
            job,
            artifact_key,
        )
    except TrackingError as exc:
        _raise_tracking_http_error(exc)
    return FileResponse(path=path, media_type=mime_type, filename=filename)


def _raise_tracking_http_error(exc: Exception) -> None:
    if isinstance(exc, TrackingError):
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "code": "TRACKING_VALIDATION_ERROR",
            "message": str(exc),
        },
    ) from exc
