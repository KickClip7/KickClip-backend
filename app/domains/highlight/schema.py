from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class HighlightScopeType(str, Enum):
    FULL_MATCH = "FULL_MATCH"
    FIRST_HALF = "FIRST_HALF"
    SECOND_HALF = "SECOND_HALF"
    CUSTOM_RANGE = "CUSTOM_RANGE"
    SELECTED_SCENES = "SELECTED_SCENES"


class HighlightFocusMode(str, Enum):
    NONE = "NONE"
    PLAYER = "PLAYER"


class HighlightScope(BaseModel):
    type: HighlightScopeType = HighlightScopeType.FULL_MATCH
    start_time_sec: float | None = Field(default=None, ge=0)
    end_time_sec: float | None = Field(default=None, ge=0)
    relative_start_sec: float | None = Field(default=None, ge=0)
    relative_end_sec: float | None = Field(default=None, ge=0)
    boundary_source: str | None = None
    requires_period_boundary: bool = False

    @model_validator(mode="after")
    def validate_range(self):
        if (
            self.start_time_sec is not None
            and self.end_time_sec is not None
            and self.end_time_sec <= self.start_time_sec
        ):
            raise ValueError("scope end_time_sec must be greater than start_time_sec")
        return self


class HighlightRequest(BaseModel):
    scope: HighlightScope = Field(default_factory=HighlightScope)
    event_labels: list[str] = Field(default_factory=list)
    desired_duration_sec: float = Field(default=60, gt=0, le=600)
    focus_mode: HighlightFocusMode = HighlightFocusMode.NONE
    focus_subject_id: str | None = None
    subtitle_style: str | None = None
    aspect_ratio: Literal["9:16", "1:1", "16:9"] = "9:16"
    parser_provenance: dict[str, Any] = Field(default_factory=dict)


class HighlightAnalyzeRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)
    structured_request: HighlightRequest | None = None


class HighlightRevisionRead(BaseModel):
    revision_id: str
    project_id: str
    revision_number: int
    parent_revision_id: str | None
    action_spotting_job_id: str | None
    user_request: str
    request: HighlightRequest
    selected_scene_ids: list[str]
    focus_mode: HighlightFocusMode
    focus_subject_id: str | None
    clip_plan_id: str | None
    render_job_id: str | None
    status: str
    pending_action: str | None
    error_message: str | None
    candidate_discovery: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class HighlightAnalyzeResponse(BaseModel):
    revision: HighlightRevisionRead
    action_spotting_reused: bool
    action_spotting_status: str
    status_url: str
    scenes_url: str


class SourcePredictionRead(BaseModel):
    time_sec: float
    label: str
    confidence: float


class HighlightSceneRead(BaseModel):
    scene_id: str
    match_id: str
    media_asset_id: str | None
    start_time_sec: float
    end_time_sec: float
    representative_time_sec: float
    primary_label: str
    confidence: float | None
    source_predictions: list[SourcePredictionRead] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    selection_state: str
    render_strategy: str


class HighlightScenesResponse(BaseModel):
    revision: HighlightRevisionRead
    scenes: list[HighlightSceneRead]


class HighlightSceneSelectionRequest(BaseModel):
    scene_ids: list[str] = Field(min_length=1)
    selection_source: Literal["USER", "AGENT"] = "USER"
    # Explicit per-scene overrides support "third scene full-frame".
    render_strategies: dict[str, Literal["TARGET_CENTERED", "FULL_FRAME", "EXCLUDE"]] = (
        Field(default_factory=dict)
    )

    @field_validator("scene_ids")
    @classmethod
    def unique_scene_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class PlayerFocusStartRequest(BaseModel):
    revision_id: str | None = None
    request: str = Field(min_length=1, max_length=2000)


class PlayerCandidateRead(BaseModel):
    candidate_id: str
    scene_id: str
    display_label: str
    anchor_time_sec: float
    anchor_source_time_sec: float
    anchor_frame_index: int
    bbox_xyxy: list[float]
    thumbnail_artifact_id: str | None
    thumbnail_url: str | None
    track_length_frames: int
    trackability_score: float
    status: str
    detector_provenance: dict[str, Any] = Field(default_factory=dict)


class PlayerCandidatesResponse(BaseModel):
    revision: HighlightRevisionRead
    candidates: list[PlayerCandidateRead]


class PlayerFocusSelectRequest(BaseModel):
    revision_id: str | None = None
    display_name: str = Field(min_length=1, max_length=100)
    anchor_scene_id: str
    candidate_id: str


class PlayerFocusSubjectRead(BaseModel):
    focus_subject_id: str
    display_name: str
    identity_source: str
    anchor_scene_id: str
    anchor_candidate_id: str


class PlayerConfirmationRequest(BaseModel):
    revision_id: str | None = None
    decision: Literal["candidate", "absent"]
    candidate_id: str | None = None
    render_strategy: Literal["TARGET_CENTERED", "FULL_FRAME", "EXCLUDE"] | None = None

    @model_validator(mode="after")
    def validate_decision(self):
        if self.decision == "candidate" and not self.candidate_id:
            raise ValueError("candidate_id is required for candidate decision")
        if self.decision == "candidate" and self.render_strategy in {
            "FULL_FRAME",
            "EXCLUDE",
        }:
            raise ValueError("candidate decision requires TARGET_CENTERED strategy")
        if self.decision == "absent" and self.render_strategy == "TARGET_CENTERED":
            raise ValueError("ABSENT scene cannot use TARGET_CENTERED")
        return self


class SceneTrackingRead(BaseModel):
    scene_id: str
    binding_id: str
    candidate_id: str | None
    tracking_job_id: str | None
    status: str
    tracking_status: str | None
    target_presence_status: str
    render_strategy: str
    timeline_summary: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class HighlightStatusResponse(BaseModel):
    revision: HighlightRevisionRead
    action_spotting_status: str | None
    pending_action: str | None
    focus_subject: PlayerFocusSubjectRead | None
    scene_tracking: list[SceneTrackingRead] = Field(default_factory=list)


class HighlightClipPlanRequest(BaseModel):
    revision_id: str | None = None
    allow_absent_full_frame: bool = False


class HighlightClipPlanResponse(BaseModel):
    revision: HighlightRevisionRead
    clip_plan_id: str
    total_duration_sec: float
    item_count: int


class HighlightRenderRequest(BaseModel):
    revision_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class HighlightRenderResponse(BaseModel):
    revision: HighlightRevisionRead
    render_job_id: str
    status: str
