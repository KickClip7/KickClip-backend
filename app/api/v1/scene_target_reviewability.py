from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, status
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
from app.domains.highlight.scene_ai_task import (
    MANUAL_ANCHOR_CONFIRM,
    MANUAL_ANCHOR_VALIDATE,
    MANUAL_TRACKING_PREPARATION,
    REVIEW_EARLIER_PREPARE,
    REVIEW_SELECTED_DECISION,
    REVIEW_SELECTED_PREPARE,
    REVIEW_UI_RENDER,
    SceneAITaskService,
    get_scene_ai_task_executor,
)
from app.domains.highlight.schema import SceneAITaskRead


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


def _enqueue(
    *,
    db: Session,
    user: User,
    project,
    task_type: str,
    payload: dict[str, Any],
) -> SceneAITaskRead:
    service = SceneAITaskService(db)
    task, _ = service.enqueue(
        user=user,
        project=project,
        task_type=task_type,
        payload=payload,
    )
    if task.status == "QUEUED":
        get_scene_ai_task_executor().submit(task.task_id)
    return service.read(task)


@router.post(
    "/target-selections/{selection_id}/selected-target-review/prepare",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def prepare_selected_target_review(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=REVIEW_SELECTED_PREPARE,
        payload={
            "selection_id": selection.selection_id,
            "scene_video_asset_id": video.asset_id,
            "shot_boundaries_artifact_id": boundaries.artifact_id,
        },
    )


@router.post(
    "/target-selections/{selection_id}/selected-target-review/decision",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def decide_selected_target_identity(
    selection_id: str,
    payload: SelectedIdentityDecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service = SceneTargetReviewabilityService(db)
    selection = service.owned(selection_id, current_user, for_update=True)
    project = require_project_access(
        db, selection.project_id, current_user
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=REVIEW_SELECTED_DECISION,
        payload={
            "selection_id": selection.selection_id,
            "decision": payload.decision,
            "identity_basis": payload.identity_basis,
        },
    )


@router.post(
    "/target-selections/{selection_id}/earlier-candidates/review-media",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def prepare_earlier_candidate_media(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=REVIEW_EARLIER_PREPARE,
        payload={
            "selection_id": selection.selection_id,
            "scene_video_asset_id": video.asset_id,
            "shot_boundaries_artifact_id": boundaries.artifact_id,
        },
    )


@router.post(
    "/target-selections/{selection_id}/review-ui",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def render_target_identity_review_ui(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=REVIEW_UI_RENDER,
        payload={
            "selection_id": selection.selection_id,
            "scene_video_asset_id": video.asset_id,
            "shot_boundaries_artifact_id": boundaries.artifact_id,
        },
    )


@router.post(
    "/target-selections/{selection_id}/manual-earlier-anchor/validate",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def validate_manual_earlier_anchor(
    selection_id: str,
    payload: ManualAnchorValidationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=MANUAL_ANCHOR_VALIDATE,
        payload={
            "selection_id": selection.selection_id,
            "scene_video_asset_id": video.asset_id,
            "shot_boundaries_artifact_id": boundaries.artifact_id,
            "global_frame": payload.global_frame,
            "click_xy": payload.click_xy,
            "drawn_bbox_xyxy": payload.drawn_bbox_xyxy,
            "identity_basis": payload.identity_basis,
        },
    )


@router.post(
    "/target-selections/{selection_id}/manual-earlier-anchor/confirm",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def confirm_manual_earlier_anchor(
    selection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service = SceneTargetReviewabilityService(db)
    selection = service.owned(selection_id, current_user, for_update=True)
    project = require_project_access(
        db, selection.project_id, current_user
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=MANUAL_ANCHOR_CONFIRM,
        payload={"selection_id": selection.selection_id},
    )


@router.post(
    "/target-selections/{selection_id}/manual-anchor-tracking-jobs",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_manual_anchor_tracking_job(
    selection_id: str,
    payload: ReviewMediaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    service, selection, project, video, boundaries = _context(
        db, current_user, selection_id, payload
    )
    return _enqueue(
        db=db,
        user=current_user,
        project=project,
        task_type=MANUAL_TRACKING_PREPARATION,
        payload={
            "selection_id": selection.selection_id,
            "scene_video_asset_id": video.asset_id,
            "shot_boundaries_artifact_id": boundaries.artifact_id,
        },
    )

