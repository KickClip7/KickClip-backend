from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_clip_plan_access, require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.clip_plan.schema import (
    ClipPlanRead,
    ClipPlanUpdateRequest,
    ExportOptionsUpdateRequest,
    ManualClipPlanCreateRequest,
)
from app.domains.clip_plan.service import ClipPlanService


router = APIRouter()


@router.post(
    "",
    response_model=ClipPlanRead,
    status_code=status.HTTP_201_CREATED,
    summary="수동 ClipPlan 생성",
)
def create_clip_plan(
    data: ManualClipPlanCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClipPlanRead:
    require_match_access(db, data.match_id, current_user)
    service = ClipPlanService(db)

    try:
        return service.create_manual_clip_plan(
            data,
            created_by=current_user.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.patch(
    "/{clip_plan_id}/export-options",
    response_model=ClipPlanRead,
    summary="ClipPlan 미리보기/내보내기 옵션 저장",
)
def update_clip_plan_export_options(
    clip_plan_id: str,
    data: ExportOptionsUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClipPlanRead:
    require_clip_plan_access(db, clip_plan_id, current_user)
    service = ClipPlanService(db)
    clip_plan = service.update_export_options(clip_plan_id, data)

    if clip_plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ClipPlan not found",
        )

    return clip_plan


@router.get(
    "/{clip_plan_id}",
    response_model=ClipPlanRead,
    summary="ClipPlan 단건 조회",
)
def get_clip_plan(
    clip_plan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClipPlanRead:
    require_clip_plan_access(db, clip_plan_id, current_user)
    service = ClipPlanService(db)
    clip_plan = service.get_clip_plan(clip_plan_id)

    if clip_plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ClipPlan not found",
        )

    return clip_plan


@router.patch(
    "/{clip_plan_id}",
    response_model=ClipPlanRead,
    summary="ClipPlan 수정",
)
def update_clip_plan(
    clip_plan_id: str,
    data: ClipPlanUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClipPlanRead:
    require_clip_plan_access(db, clip_plan_id, current_user)
    service = ClipPlanService(db)

    try:
        clip_plan = service.update_clip_plan(clip_plan_id, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if clip_plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ClipPlan not found",
        )

    return clip_plan
