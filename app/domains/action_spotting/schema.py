from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domains.analysis.schema import AnalysisJobStepCompactRead
from app.domains.timeline.schema import TimelineEventRead


class ActionSpottingJobRequest(BaseModel):
    run_feature_extraction: bool = True
    feature_extraction_mode: Literal["auto", "force", "skip"] = "auto"
    feature_batch_size: int | None = Field(default=None, ge=1, le=512)
    device: str = "auto"
    real_adapter_batch_size: int | None = Field(default=None, ge=1, le=1024)
    real_adapter_max_candidates: Literal[113] | None = None
    halftime_split_sec: float | None = Field(default=None, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)

    def to_job_options(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "run_feature_extraction": self.run_feature_extraction,
            "feature_extraction_mode": self.feature_extraction_mode,
            "highlight_predictor_mode": "real",
            "strict_real_model": True,
            "allow_fallback_when_missing": False,
            "feature_extraction_allow_continue_on_failure": False,
            "device": self.device,
            "action_spotting_state": "ACTION_SPOTTING_QUEUED",
            "action_spotting_state_history": ["ACTION_SPOTTING_QUEUED"],
        }
        if self.real_adapter_batch_size is not None:
            values["real_adapter_batch_size"] = self.real_adapter_batch_size
        if self.feature_batch_size is not None:
            values["feature_batch_size"] = self.feature_batch_size
        if self.real_adapter_max_candidates is not None:
            values["real_adapter_max_candidates"] = self.real_adapter_max_candidates
        if self.halftime_split_sec is not None:
            values["halftime_split_sec"] = self.halftime_split_sec
        values.update(self.options)
        # This endpoint promises real champion inference and never demo fallback.
        values.update(
            {
                "highlight_predictor_mode": "real",
                "strict_real_model": True,
                "allow_fallback_when_missing": False,
                "feature_extraction_allow_continue_on_failure": False,
            }
        )
        return values


class ActionSpottingJobResponse(BaseModel):
    analysis_job_id: str
    status: str
    progress: int
    current_step: str | None
    steps: list[AnalysisJobStepCompactRead]
    status_url: str
    events_url: str
    workflow_status: str
    workflow_state_history: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    cache_reused: bool = False


class ActionSpottingEventsResponse(BaseModel):
    analysis_job_id: str
    status: str
    count: int
    events: list[TimelineEventRead] = Field(default_factory=list)
    workflow_status: str
    workflow_state_history: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None


class ActionSpottingModelResponse(BaseModel):
    ready: bool
    model_id: str
    model_name: str
    model_version: str | None
    classes: list[str]
    checkpoint_sha256: str | None
    device_requested: str
    preflight: dict[str, Any]


class ActionSpottingDiagnosticsResponse(BaseModel):
    diagnostics: dict[str, Any]
