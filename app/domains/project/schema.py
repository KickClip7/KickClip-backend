from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    status: str | None = None
    thumbnail_artifact_id: str | None = None


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    project_id: str
    owner_id: str | None
    title: str
    description: str | None
    status: str
    thumbnail_artifact_id: str | None
    last_opened_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ProjectRecentItem(BaseModel):
    project_id: str
    title: str
    status: str
    thumbnail_url: str | None = None
    duration_sec: float | None = None
    created_at: datetime


class ProjectRecentListResponse(BaseModel):
    projects: list[ProjectRecentItem]
