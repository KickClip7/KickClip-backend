from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import (
    require_artifact_access,
    require_media_access,
    require_project_access,
)
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.highlight.scene_target_reviewability import (
    SceneTargetReviewabilityService,
)
from app.domains.tracking.schema import TrackingJobCreateResponse


router = APIRouter()


class ReviewMediaRequest(BaseModel):
    scene_video_asset_id: str = Field(min_length=1, max_length=64)
    shot_boundaries_artifact_id: str = Field(min_length=1, max_length=64)


class SelectedIdentityDecisionRequest(BaseModel):
    decision: Literal["confirm", "reject", "uncertain"]
    identity_basis: str = Field(min_length=3, max_length=2000)


class ManualAnchorValidationRequest(ReviewMediaRequest):
    global_frame: int = Field(ge=0)
    click_xy: list[float] | None = Field(default=None, min_length=2, max_length=2)
    drawn_bbox_xyxy: list[float] | None = Field(
        default=None, min_length=4, max_length=4
    )
    identity_basis: str = Field(min_length=3, max_length=2000)

    @model_validator(mode="after")
    def exactly_one_geometry(self):
        if (self.click_xy is None) == (self.drawn_bbox_xyxy is None):
            raise ValueError("Provide exactly one click or drawn bbox.")
        return self


def _context(
    db: Session,
    user: User,
    selection_id: str,
    payload: ReviewMediaRequest,
):
    service = SceneTargetReviewabilityService(db)
    selection = service.owned(selection_id, user, for_update=True)
    project = require_project_access(db, selection.project_id, user)
    video = require_media_access(db, payload.scene_video_asset_id, user)
    boundaries = require_artifact_access(
        db, payload.shot_boundaries_artifact_id, user
    )
    return service, selection, project, video, boundaries


def _error(exc: Exception) -> None:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=str(exc),
    ) from exc


@router.post("/target-selections/{selection_id}/selected-target-review/prepare")
def prepare_selected_target_review(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    try:
        return service.prepare_selected_review(
            selection=selection,
            user=current_user,
            project=project,
            video=video,
            boundaries=boundaries,
        )
    except (ValueError, RuntimeError) as exc:
        _error(exc)


@router.post("/target-selections/{selection_id}/selected-target-review/decision")
def decide_selected_target_identity(
    selection_id: str,
    payload: SelectedIdentityDecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service = SceneTargetReviewabilityService(db)
    selection = service.owned(selection_id, current_user, for_update=True)
    try:
        return service.decide_selected_identity(
            selection=selection,
            user=current_user,
            decision=payload.decision,
            identity_basis=payload.identity_basis,
        )
    except (ValueError, RuntimeError) as exc:
        _error(exc)


@router.post("/target-selections/{selection_id}/earlier-candidates/review-media")
def prepare_earlier_candidate_media(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    try:
        return service.prepare_earlier_review(
            selection=selection,
            project=project,
            video=video,
            boundaries=boundaries,
        )
    except (ValueError, RuntimeError) as exc:
        _error(exc)


@router.post("/target-selections/{selection_id}/review-ui")
def render_target_identity_review_ui(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    try:
        return service.render_review_ui(
            selection=selection,
            project=project,
            video=video,
            boundaries=boundaries,
        )
    except (ValueError, RuntimeError) as exc:
        _error(exc)


@router.post("/target-selections/{selection_id}/manual-earlier-anchor/validate")
def validate_manual_earlier_anchor(
    selection_id: str,
    payload: ManualAnchorValidationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    try:
        return service.validate_manual_anchor(
            selection=selection,
            user=current_user,
            project=project,
            video=video,
            boundaries=boundaries,
            global_frame=payload.global_frame,
            click_xy=payload.click_xy,
            drawn_bbox=payload.drawn_bbox_xyxy,
            identity_basis=payload.identity_basis,
        )
    except (ValueError, RuntimeError) as exc:
        _error(exc)


@router.post("/target-selections/{selection_id}/manual-earlier-anchor/confirm")
def confirm_manual_earlier_anchor(
    selection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    service = SceneTargetReviewabilityService(db)
    selection = service.owned(selection_id, current_user, for_update=True)
    try:
        created = service.confirm_manual_anchor(
            selection=selection,
            user=current_user,
        )
        return {
            "selection_id": created.selection_id,
            "selection_revision": created.selection_revision,
            "status": created.status,
            "previous_selection_id": selection.selection_id,
        }
    except (ValueError, RuntimeError) as exc:
        _error(exc)


@router.post(
    "/target-selections/{selection_id}/manual-anchor-tracking-jobs",
    response_model=TrackingJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_manual_anchor_tracking_job(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrackingJobCreateResponse:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    try:
        return service.create_manual_tracking_job(
            selection=selection,
            user=current_user,
            project=project,
            video=video,
            boundaries=boundaries,
        )
    except (ValueError, RuntimeError) as exc:
        _error(exc)

