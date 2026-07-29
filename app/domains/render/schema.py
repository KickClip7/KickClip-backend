from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RenderRatio = Literal["9:16", "1:1", "16:9"]


class RenderOptions(BaseModel):
    ratio: RenderRatio | None = None
    quality: str | None = None
    captions_enabled: bool | None = None
    music: str | None = None
    music_asset_id: str | None = None
    title: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    thumbnail_timestamp_sec: float | None = None
    thumbnail_source_timestamp_sec: float | None = None

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, value: str | None) -> str | None:
        if value is None:
            return value

        normalized = value.strip()
        if normalized.lower() == "4k":
            normalized = "4K"

        allowed_values = {"720p", "1080p", "4K"}
        if normalized not in allowed_values:
            raise ValueError("quality must be one of: 720p, 1080p, 4K")

        return normalized

    @field_validator("music", "music_asset_id", "title")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None


class RenderCreateRequest(BaseModel):
    clip_plan_id: str
    options: RenderOptions = Field(default_factory=RenderOptions)


class RenderCreateResponse(BaseModel):
    render_job_id: str
    status: str
    progress: int
    reused: bool = False


class RenderJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    render_job_id: str
    clip_plan_id: str
    status: str
    progress: int
    ratio: str
    resolution: str | None
    quality: str
    captions_enabled: bool
    music_asset_id: str | None
    output_artifact_id: str | None
    subtitle_artifact_id: str | None
    options: dict[str, Any]
    runtime_sec: float | None
    error_message: str | None
    retryable: bool = False
    download_url: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
