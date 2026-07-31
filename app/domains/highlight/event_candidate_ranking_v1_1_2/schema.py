from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EventCandidateRankingV112Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=64)
    scene_id: str = Field(min_length=1, max_length=64)
    shortlist_size: int = Field(default=5, ge=3, le=5)
