from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalysisJobOptions(BaseModel):
    run_highlight_spotting: bool = True
    run_player_tracking: bool = True
    build_timeline: bool = True


class AnalysisJobCreateRequest(BaseModel):
    job_type: str = "FULL_MATCH_ANALYSIS"
    options: dict[str, Any] = Field(default_factory=dict)


class AnalysisJobCreate(BaseModel):
    match_id: str
    job_type: str = "FULL_MATCH_ANALYSIS"
    options: dict[str, Any] = Field(default_factory=dict)


class AnalysisJobStepCreate(BaseModel):
    analysis_job_id: str
    step_key: str
    label: str
    status: str = "QUEUED"
    progress: int = Field(default=0, ge=0, le=100)


class AnalysisJobStepRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step_id: str
    analysis_job_id: str
    step_key: str
    label: str
    status: str
    progress: int
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def serialize_status_lowercase(cls, value: str) -> str:
        return value.lower() if isinstance(value, str) else value


class AnalysisJobStepCompactRead(BaseModel):
    key: str
    label: str
    status: str
    progress: int


class AnalysisJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    analysis_job_id: str
    match_id: str
    job_type: str
    status: str
    progress: int
    current_step: str | None
    options: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    steps: list[AnalysisJobStepRead] = []
    created_at: datetime
    updated_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def serialize_status_lowercase(cls, value: str) -> str:
        return value.lower() if isinstance(value, str) else value


class AnalysisJobCreateResponse(BaseModel):
    analysis_job_id: str
    status: str
    progress: int
    current_step: str | None
    steps: list[AnalysisJobStepCompactRead]


class AnalysisJobStatusResponse(BaseModel):
    analysis_job_id: str
    match_id: str
    job_type: str
    status: str
    progress: int
    current_step: str | None
    options: dict[str, Any]
    steps: list[AnalysisJobStepCompactRead]
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None