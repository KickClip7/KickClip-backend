from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ArtifactCreate(BaseModel):
    match_id: str
    analysis_job_id: str | None = None
    artifact_type: str
    file_path: str
    mime_type: str | None = None
    metadata: dict = Field(default_factory=dict)


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    artifact_id: str
    match_id: str
    analysis_job_id: str | None
    artifact_type: str
    file_path: str
    mime_type: str | None
    metadata_: dict = Field(alias="metadata")
    created_at: datetime
    updated_at: datetime