import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.artifact.model import Artifact
from app.domains.artifact.signed_url import build_signed_artifact_url
from app.domains.auth.access import require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.match.model import Match
from app.domains.match.schema import MatchRead
from app.domains.match.service import MatchService
from app.domains.media.model import MediaAsset
from app.domains.media.signed_url import build_signed_media_url
from app.domains.project.schema import ProjectCreate, ProjectRead
from app.domains.project.service import ProjectService
from app.domains.studio.schema import UploadMatchVideoResponse
from app.domains.studio.service import StudioService


router = APIRouter()


@router.get(
    "",
    response_model=list[MatchRead],
    summary="업로드한 경기 목록 조회",
)
def list_matches(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[MatchRead]:
    owner_id = None if current_user.developer_mode_enabled else current_user.user_id
    service = MatchService(db)
    matches = service.list_matches(owner_id=owner_id)
    latest_jobs = service.latest_analysis_by_match_ids(
        [match.match_id for match in matches]
    )
    return [
        _build_match_read(
            match,
            current_user.user_id,
            latest_analysis_job=latest_jobs.get(match.match_id),
        )
        for match in matches
    ]


@router.post(
    "",
    response_model=UploadMatchVideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="영상 업로드 및 Match 생성",
)
def upload_match_video(
    file: UploadFile = File(...),
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
    current_user: User = Depends(get_current_user),
) -> UploadMatchVideoResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )
    metadata_dict = _parse_metadata(metadata)
    try:
        response = StudioService(db).upload_match_video(
            file=file,
            owner_id=current_user.user_id,
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
    response.video_url = build_signed_media_url(
        response.video_asset_id,
        current_user.user_id,
    )[0]
    if response.preview_video_asset_id is not None:
        response.preview_url = build_signed_media_url(
            response.preview_video_asset_id,
            current_user.user_id,
        )[0]
    return response


@router.post(
    "/{match_id}/projects",
    response_model=ProjectRead,
    status_code=status.HTTP_201_CREATED,
    summary="Match에 편집 Project 생성",
)
def create_match_project(
    match_id: str,
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ProjectRead:
    require_match_access(db, match_id, current_user)
    try:
        return ProjectService(db).create_project(
            payload,
            match_id=match_id,
            owner_id=current_user.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{match_id}/projects",
    response_model=list[ProjectRead],
    summary="Match의 편집 Project 목록",
)
def list_match_projects(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ProjectRead]:
    require_match_access(db, match_id, current_user)
    return ProjectService(db).list_match_projects(
        match_id,
        user_id=current_user.user_id,
    )


@router.get(
    "/{match_id}",
    response_model=MatchRead,
    summary="경기 단건 조회",
)
def get_match(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MatchRead:
    match = require_match_access(db, match_id, current_user)
    latest_job = MatchService(db).latest_analysis_by_match_ids([match.match_id]).get(
        match.match_id
    )
    return _build_match_read(
        match,
        current_user.user_id,
        latest_analysis_job=latest_job,
    )


def _build_match_read(
    match: Match,
    user_id: str,
    *,
    latest_analysis_job=None,
) -> MatchRead:
    assets_by_type = {asset.asset_type: asset for asset in match.media_assets}
    raw_asset = assets_by_type.get("RAW_VIDEO")
    preview_asset = assets_by_type.get("WEB_PREVIEW_VIDEO")
    video_asset = preview_asset or raw_asset
    thumbnail = _select_match_thumbnail(match)

    signed_urls: dict[str, str] = {}
    for asset in (video_asset, preview_asset):
        if asset is not None and asset.asset_id not in signed_urls:
            signed_urls[asset.asset_id] = build_signed_media_url(
                asset.asset_id,
                user_id,
            )[0]

    response = MatchRead.model_validate(match)
    return response.model_copy(
        update={
            "raw_video_asset_id": _asset_id(raw_asset),
            "video_asset_id": _asset_id(video_asset),
            "video_url": _asset_url(video_asset, signed_urls),
            "preview_video_asset_id": _asset_id(preview_asset),
            "preview_url": _asset_url(preview_asset, signed_urls),
            "thumbnail_artifact_id": (
                thumbnail.artifact_id if thumbnail is not None else None
            ),
            "thumbnail_url": (
                build_signed_artifact_url(
                    thumbnail.artifact_id,
                    user_id,
                )[0]
                if thumbnail is not None
                else None
            ),
            "analysis_status": (
                latest_analysis_job.status
                if latest_analysis_job is not None
                else "NOT_STARTED"
            ),
            "analysis_progress": (
                latest_analysis_job.progress
                if latest_analysis_job is not None
                else 0
            ),
            "analysis_job_id": (
                latest_analysis_job.analysis_job_id
                if latest_analysis_job is not None
                else None
            ),
            "analysis_current_step": (
                latest_analysis_job.current_step
                if latest_analysis_job is not None
                else None
            ),
            "analysis_error_message": (
                latest_analysis_job.error_message
                if latest_analysis_job is not None
                else None
            ),
            "analysis_retryable": bool(
                latest_analysis_job is not None
                and latest_analysis_job.status == "FAILED"
            ),
        }
    )


def _asset_id(asset: MediaAsset | None) -> str | None:
    return asset.asset_id if asset is not None else None


def _select_match_thumbnail(match: Match) -> Artifact | None:
    thumbnails = [
        artifact
        for artifact in match.artifacts
        if artifact.artifact_type == "MATCH_THUMBNAIL"
    ]
    return max(thumbnails, key=lambda artifact: artifact.created_at, default=None)


def _asset_url(
    asset: MediaAsset | None,
    signed_urls: dict[str, str],
) -> str | None:
    return signed_urls.get(asset.asset_id) if asset is not None else None


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
