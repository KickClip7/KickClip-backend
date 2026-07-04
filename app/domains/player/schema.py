from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlayerCreate(BaseModel):
    player_id: str | None = None
    match_id: str
    display_name: str
    number: int | None = None
    team_name: str | None = None
    role: str | None = None
    identity_status: str = "UNKNOWN"
    profile_source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlayerRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    player_id: str
    match_id: str
    display_name: str
    number: int | None
    team_name: str | None
    role: str | None
    identity_status: str
    profile_source: str | None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_",
        serialization_alias="metadata",
    )
    created_at: datetime
    updated_at: datetime


class PlayerTrackCreate(BaseModel):
    match_id: str
    player_id: str
    source_job_id: str | None = None
    start_sec: float
    end_sec: float
    duration_sec: float
    track_artifact_id: str | None = None
    summary: str | None = None
    linked_event_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlayerTrackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    player_track_id: str
    match_id: str
    player_id: str
    source_job_id: str | None
    start_sec: float
    end_sec: float
    duration_sec: float
    track_artifact_id: str | None
    summary: str | None
    linked_event_ids: list[str]
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_",
        serialization_alias="metadata",
    )
    created_at: datetime
    updated_at: datetime

class FrontendPlayer(BaseModel):
    """Frontend-compatible player shape used by KickClip Studio edit screen."""

    id: str
    name: str
    number: int | None = None
    team: str = ""
    role: str = ""
    identityStatus: str = "UNKNOWN"


class PlayersResponse(BaseModel):
    match_id: str
    players: list[FrontendPlayer] = Field(default_factory=list)
    count: int

