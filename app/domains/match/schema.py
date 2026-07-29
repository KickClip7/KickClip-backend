from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MatchCreateRequest(BaseModel):
    home_team: str | None = None
    away_team: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    match_date: datetime | None = None
    competition: str | None = None
    season: str | None = None
    duration_sec: float | None = None
    metadata: dict = Field(default_factory=dict)


class MatchCreate(MatchCreateRequest):
    owner_id: str


class MatchUpdate(BaseModel):
    home_team: str | None = None
    away_team: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    match_date: datetime | None = None
    competition: str | None = None
    season: str | None = None
    duration_sec: float | None = None
    metadata: dict | None = None


class MatchRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    match_id: str
    owner_id: str
    home_team: str | None
    away_team: str | None
    home_score: int | None
    away_score: int | None
    match_date: datetime | None
    competition: str | None
    season: str | None
    duration_sec: float | None
    metadata: dict = Field(
        default_factory=dict,
        validation_alias="metadata_",
        serialization_alias="metadata",
    )
    raw_video_asset_id: str | None = None
    video_asset_id: str | None = None
    video_url: str | None = None
    preview_video_asset_id: str | None = None
    preview_url: str | None = None
    thumbnail_artifact_id: str | None = None
    thumbnail_url: str | None = None
    analysis_status: str = "NOT_STARTED"
    analysis_progress: int = Field(default=0, ge=0, le=100)
    analysis_job_id: str | None = None
    analysis_current_step: str | None = None
    analysis_error_message: str | None = None
    analysis_retryable: bool = False
    created_at: datetime
    updated_at: datetime
