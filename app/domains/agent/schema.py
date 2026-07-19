from typing import Any

from pydantic import BaseModel, Field, model_validator


class ExportMetadataRecommendRequest(BaseModel):
    clip_plan_id: str | None = None
    match_id: str | None = None
    language: str = "ko"

    @model_validator(mode="after")
    def validate_source(self):
        if not self.clip_plan_id and not self.match_id:
            raise ValueError("clip_plan_id 또는 match_id가 필요합니다.")
        return self


class ThumbnailRecommendation(BaseModel):
    timestamp_sec: float
    source_timestamp_sec: float
    timestamp_label: str
    reason: str
    image_data_url: str


class ExportMetadataRecommendResponse(BaseModel):
    title: str
    hashtags: list[str]
    thumbnail: ThumbnailRecommendation
    model_id: str
    sampled_frame_count: int


class AgentClipPlanRequest(BaseModel):
    match_id: str
    mode: str = "AGENT_GENERATED"
    prompt: str
    target_duration_sec: int | None = 30
    selected_player_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class AgentClipPlanItemResponse(BaseModel):
    timeline_event_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    reason: str


class AgentClipPlanResponse(BaseModel):
    clip_plan_id: str
    summary: str
    total_duration_sec: float
    items: list[AgentClipPlanItemResponse] = Field(default_factory=list)


class PlayerProfileResolveRequest(BaseModel):
    match_id: str
    player_id: str
    user_message: str


class PlayerProfileResolveResponse(BaseModel):
    player_id: str
    display_name: str
    real_team: str | None = None
    position: str | None = None
    nationality: str | None = None
    traits: list[str] = Field(default_factory=list)
    summary: str
    identity_status: str
