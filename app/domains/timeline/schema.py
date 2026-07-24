from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TimelineEventCreate(BaseModel):
    match_id: str
    source_artifact_id: str | None = None
    source_job_id: str | None = None

    event_type: str
    label: str

    half: int | None = None
    timestamp_sec: float
    start_sec: float
    end_sec: float
    duration_sec: float

    confidence: float | None = None
    highlight_score: float | None = None

    title: str | None = None
    description: str | None = None
    team_name: str | None = None

    player_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineEventRead(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    timeline_event_id: str
    match_id: str
    source_artifact_id: str | None
    source_job_id: str | None

    event_type: str
    label: str

    half: int | None
    timestamp_sec: float
    start_sec: float
    end_sec: float
    duration_sec: float

    confidence: float | None
    highlight_score: float | None

    title: str | None
    description: str | None
    team_name: str | None

    player_ids: list[str]
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_",
        serialization_alias="metadata",
    )

    created_at: datetime
    updated_at: datetime

class FrontendTimelineEvent(BaseModel):
    """Frontend-compatible event shape used by KickClip Studio edit screen."""

    id: str
    category: str
    tag: str
    title: str
    time: str
    minute: int
    duration: int
    start: str
    end: str
    duration_sec: float
    min_start_sec: float
    natural_end_sec: float
    half: int | None
    importance_score: float
    score: float
    highlightScore: int
    description: str
    playerIds: list[str] = Field(default_factory=list)
    team: str = ""


class TimelineEventsResponse(BaseModel):
    match_id: str
    events: list[FrontendTimelineEvent] = Field(default_factory=list)
    count: int
