from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ClipPlanMode = Literal["MANUAL", "MATCH_FOCUS", "PLAYER_FOCUS", "AGENT_GENERATED"]


class ClipPlanItemCreate(BaseModel):
    timeline_event_id: str
    start_sec: float
    end_sec: float
    order_index: int
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_sec")
    @classmethod
    def validate_end_sec(cls, value: float, info):
        start_sec = info.data.get("start_sec")
        if start_sec is not None and value <= start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return value


class ClipPlanCreate(BaseModel):
    match_id: str
    mode: ClipPlanMode = "AGENT_GENERATED"
    summary: str | None = None
    target_duration_sec: float | None = None
    actual_duration_sec: float | None = None
    created_by: str = "agent"
    options: dict[str, Any] = Field(default_factory=dict)
    items: list[ClipPlanItemCreate] = Field(default_factory=list)


class ManualClipPlanCreateRequest(BaseModel):
    match_id: str
    mode: ClipPlanMode = "MANUAL"
    summary: str | None = None
    target_duration_sec: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    items: list[ClipPlanItemCreate] = Field(default_factory=list)


class ClipPlanUpdateRequest(BaseModel):
    mode: ClipPlanMode | None = None
    summary: str | None = None
    target_duration_sec: float | None = None
    options: dict[str, Any] | None = None

    # items가 None이면 기존 items 유지, []이면 items 전체 삭제, 값이 있으면 전체 교체한다.
    items: list[ClipPlanItemCreate] | None = None


class ExportOptionsUpdateRequest(BaseModel):
    title: str | None = None
    ratio: Literal["9:16", "1:1", "16:9"] | None = None
    captions_enabled: bool | None = None
    music: str | None = None
    quality: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    thumbnail_timestamp_sec: float | None = None
    thumbnail_source_timestamp_sec: float | None = None

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, value: str | None) -> str | None:
        if value is None:
            return value

        normalized = value.strip()
        allowed_values = {"720p", "1080p", "4K"}
        if normalized.lower() == "4k":
            normalized = "4K"

        if normalized not in allowed_values:
            raise ValueError("quality must be one of: 720p, 1080p, 4K")

        return normalized

    @field_validator("title", "music")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value

        normalized = value.strip()
        return normalized or None


class ClipPlanItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    clip_plan_item_id: str
    clip_plan_id: str
    timeline_event_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    order_index: int
    reason: str | None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_",
        serialization_alias="metadata",
    )
    created_at: datetime
    updated_at: datetime


class ClipPlanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    clip_plan_id: str
    match_id: str
    mode: str
    summary: str | None
    target_duration_sec: float | None
    actual_duration_sec: float | None
    created_by: str
    options: dict[str, Any]
    items: list[ClipPlanItemRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
