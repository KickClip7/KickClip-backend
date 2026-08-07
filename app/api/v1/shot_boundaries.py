from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_project_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.shot_boundary.schema import (
    ShotBoundaryConfirmRequest,
    ShotBoundaryConfirmResponse,
    ShotBoundaryDraftUpdate,
    ShotBoundaryReviewResponse,
)
from app.domains.shot_boundary.service import (
    ShotBoundaryReviewService,
    ShotBoundaryWorkflowError,
)

router = APIRouter()


def _raise(exc: ShotBoundaryWorkflowError) -> None:
    raise HTTPException(
        status_code=exc.http_status,
        detail={"code": exc.code, "message": str(exc), **exc.detail},
    ) from exc


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/prepare",
    response_model=ShotBoundaryReviewResponse,
    summary="Automatically prepare authoritative shot boundaries and candidate detections",
)
def prepare_shot_boundaries(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    response: Response,
    new_review_revision: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShotBoundaryReviewResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        result = ShotBoundaryReviewService(db).prepare_for_candidate_discovery(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            new_review_revision=new_review_revision,
        )
        response.status_code = status.HTTP_200_OK
        return result
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.get(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries",
    response_model=ShotBoundaryReviewResponse,
    summary="Get the authoritative automatic-or-reviewed shot-boundary state",
)
def get_shot_boundaries(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShotBoundaryReviewResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        return ShotBoundaryReviewService(db).status(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.put(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/draft",
    response_model=ShotBoundaryReviewResponse,
    summary="Optional fallback: save a human-corrected shot-boundary draft",
)
def update_shot_boundary_draft(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    payload: ShotBoundaryDraftUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShotBoundaryReviewResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        return ShotBoundaryReviewService(db).update_draft(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            payload=payload,
        )
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/reset",
    response_model=ShotBoundaryReviewResponse,
    summary="Optional fallback: reset a corrected draft to automatic cut detection",
)
def reset_shot_boundary_draft(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShotBoundaryReviewResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        return ShotBoundaryReviewService(db).reset_draft(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/confirm",
    response_model=ShotBoundaryConfirmResponse,
    summary="Optional fallback: replace automatic boundaries with a human-reviewed artifact",
)
def confirm_shot_boundaries(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    payload: ShotBoundaryConfirmRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShotBoundaryConfirmResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        return ShotBoundaryReviewService(db).confirm(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            payload=payload,
        )
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/detections/retry",
    response_model=ShotBoundaryReviewResponse,
    summary="Retry detection materialization without changing the reviewed artifact",
)
def retry_shot_boundary_detections(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ShotBoundaryReviewResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        return ShotBoundaryReviewService(db).retry_detections(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.get(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/media/contact-sheet",
    summary="Download the authenticated review contact sheet",
)
def shot_boundary_contact_sheet(
    project_id: str,
    revision_id: str,
    event_id: str,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        path = ShotBoundaryReviewService(db).media_path(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            media_kind="contact-sheet",
        )
        return FileResponse(path, media_type="image/jpeg")
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)


@router.get(
    "/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/media/cut/{cut_frame}/{side}",
    summary="Download an authenticated frame immediately before or after an automatic cut",
)
def shot_boundary_cut_thumbnail(
    project_id: str,
    revision_id: str,
    event_id: str,
    cut_frame: int,
    side: str,
    scene_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    project = require_project_access(db, project_id, current_user)
    try:
        path = ShotBoundaryReviewService(db).media_path(
            project=project,
            user=current_user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            media_kind="cut",
            cut_frame=cut_frame,
            side=side,
        )
        return FileResponse(path, media_type="image/jpeg")
    except ShotBoundaryWorkflowError as exc:
        _raise(exc)
