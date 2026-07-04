import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.studio.schema import (
    StudioEditStateResponse,
    UploadMatchVideoResponse,
)
from app.domains.studio.service import StudioService


router = APIRouter()


@router.post(
    "/upload-match-video",
    response_model=UploadMatchVideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="경기 영상 업로드",
)
def upload_match_video(
    file: UploadFile = File(...),
    project_title: str | None = Form(default=None),
    project_description: str | None = Form(default=None),
    owner_id: str | None = Form(default=None),
    home_team: str | None = Form(default=None),
    away_team: str | None = Form(default=None),
    home_score: int | None = Form(default=None),
    away_score: int | None = Form(default=None),
    match_date: datetime | None = Form(default=None),
    competition: str | None = Form(default=None),
    season: str | None = Form(default=None),
    duration_sec: float | None = Form(default=None),
    metadata: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> UploadMatchVideoResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    metadata_dict = _parse_metadata(metadata)

    service = StudioService(db)

    try:
        return service.upload_match_video(
            file=file,
            project_title=project_title,
            project_description=project_description,
            owner_id=owner_id,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            match_date=match_date,
            competition=competition,
            season=season,
            duration_sec=duration_sec,
            match_metadata=metadata_dict,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload match video: {exc}",
        ) from exc


def _parse_metadata(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be a valid JSON string.",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be a JSON object.",
        )

    return parsed


@router.get(
    "/matches/{match_id}/edit-state",
    response_model=StudioEditStateResponse,
    summary="편집 화면 초기 상태 조회",
)
def get_match_edit_state(
    match_id: str,
    db: Session = Depends(get_db),
) -> StudioEditStateResponse:
    service = StudioService(db)
    edit_state = service.get_edit_state(match_id)

    if edit_state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found",
        )

    return edit_state

