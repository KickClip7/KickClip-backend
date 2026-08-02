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
    player_id: str | None = None
    track_id: str | None = None
    scene_id: str
    display_label: str
    number: str | None = None
    jersey_number: str | None = None
    team: str | None = None
    confidence: float | None = None
    anchor_time_sec: float
    anchor_source_time_sec: float
    anchor_frame_index: int
    bbox_xyxy: list[float]
    bounding_box: list[float] | None = None
    thumbnail_artifact_id: str | None
    thumbnail_url: str | None
    representative_image_url: str | None = None
    track_length_frames: int
    trackability_score: float
    status: str
    detector_provenance: dict[str, Any] = Field(default_factory=dict)
    tracking_metadata: dict[str, Any] = Field(default_factory=dict)


class PlayerCandidatesResponse(BaseModel):
    revision: HighlightRevisionRead
    candidates: list[PlayerCandidateRead]


class PlayerFocusSelectRequest(BaseModel):
    revision_id: str | None = None
    display_name: str | None = Field(default=None, min_length=1, max_length=100)
    anchor_scene_id: str | None = None
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
    pending_action: str | None = None
    progress: int = Field(default=0, ge=0, le=100)
    current_stage: str | None = None
    retryable: bool = False
    status_url: str | None = None
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
    reused: bool = False


class HighlightDraftTimelineItem(BaseModel):
    scene_id: str = Field(min_length=1, max_length=64)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    order_index: int = Field(ge=0)
    render_strategy: Literal["TARGET_CENTERED", "FULL_FRAME", "EXCLUDE"] = (
        "FULL_FRAME"
    )
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self


class HighlightDraftState(BaseModel):
    selected_scene_ids: list[str] = Field(default_factory=list, max_length=100)
    timeline_items: list[HighlightDraftTimelineItem] = Field(
        default_factory=list,
        max_length=200,
    )
    export_options: dict[str, Any] = Field(default_factory=dict)
    editor_state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("selected_scene_ids")
    @classmethod
    def unique_draft_scene_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_timeline_items(self):
        order_indices = [item.order_index for item in self.timeline_items]
        if len(order_indices) != len(set(order_indices)):
            raise ValueError("timeline item order_index values must be unique")
        selected = set(self.selected_scene_ids)
        unknown = {
            item.scene_id
            for item in self.timeline_items
            if item.scene_id not in selected
        }
        if unknown:
            raise ValueError(
                "timeline items must reference selected_scene_ids"
            )
        return self


class HighlightDraftSaveRequest(HighlightDraftState):
    revision_id: str | None = None
    clip_plan_id: str | None = None
    expected_version: int | None = Field(default=None, ge=0)


class HighlightDraftRead(HighlightDraftState):
    draft_id: str
    project_id: str
    revision_id: str | None
    clip_plan_id: str | None
    saved_by_user_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime


class SceneWideCandidateDiscoveryRequest(BaseModel):
    scene_id: str = Field(min_length=1, max_length=64)
    scene_video_asset_id: str = Field(min_length=1, max_length=64)
    shot_boundaries_artifact_id: str = Field(min_length=1, max_length=64)
    detections_artifact_id: str = Field(min_length=1, max_length=64)


class SceneWideCandidateRead(BaseModel):
    candidate_id: str
    scene_id: str
    shot_id: str
    shot_index: int
    first_frame: int
    last_frame: int
    observation_count: int
    representative_observation: dict[str, Any]
    tracking_initialization_observation: dict[str, Any]
    quality: dict[str, Any]
    artifacts: dict[str, Any]
    artifact_ids: dict[str, str] = Field(default_factory=dict)
    gallery_visibility: str


class SceneWideCandidateGalleryResponse(BaseModel):
    revision_id: str
    scene_id: str
    status: str
    shot_count: int
    candidate_count: int
    candidates: list[SceneWideCandidateRead] = Field(default_factory=list)
    pagination: dict[str, Any] = Field(default_factory=dict)


class SceneTargetSelectionCreateRequest(BaseModel):
    scene_id: str = Field(min_length=1, max_length=64)
    candidate_id: str = Field(min_length=1, max_length=255)
    scene_video_asset_id: str = Field(min_length=1, max_length=64)


class EarlierAnchorConfirmationRequest(BaseModel):
    candidate_id: str | None = Field(default=None, max_length=255)
    decision: Literal["candidate", "reject_all", "start_selected_shot"]

    @model_validator(mode="after")
    def validate_candidate_decision(self):
        if self.decision == "candidate" and not self.candidate_id:
            raise ValueError("candidate_id is required for candidate decision")
        if self.decision != "candidate" and self.candidate_id is not None:
            raise ValueError("candidate_id is allowed only for candidate decision")
        return self


class SceneTargetSelectionRead(BaseModel):
    selection_id: str
    selection_revision: int
    project_id: str
    revision_id: str
    scene_id: str
    selected_candidate_id: str
    status: str
    target_selection: dict[str, Any]
    target_reference_set: dict[str, Any]
    earlier_candidate_proposals: dict[str, Any]
    earlier_anchor_decision: dict[str, Any]
    tracking_job_id: str | None
    tracking_cache_key: str | None


class SceneTargetTrackingCreateRequest(BaseModel):
    scene_video_asset_id: str = Field(min_length=1, max_length=64)
    shot_boundaries_artifact_id: str = Field(min_length=1, max_length=64)


class EventCandidateRankingRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    event_label: str = Field(min_length=1, max_length=64)
    event_time_sec: float = Field(ge=0)
    event_confidence: float | None = Field(default=None, ge=0, le=1)
    scene_id: str = Field(min_length=1, max_length=64)
    scene_start_sec: float = Field(ge=0)
    scene_end_sec: float = Field(gt=0)
    scene_candidate_manifest_sha256: str = Field(
        min_length=64, max_length=64
    )
    shot_boundaries_sha256: str = Field(min_length=64, max_length=64)
    shortlist_size: int = Field(default=5, ge=3, le=5)

    @model_validator(mode="after")
    def validate_time_contract(self):
        if self.scene_end_sec <= self.scene_start_sec:
            raise ValueError("scene_end_sec must be greater than scene_start_sec")
        if not self.scene_start_sec <= self.event_time_sec <= self.scene_end_sec:
            raise ValueError(
                "event_time_sec must use source-video seconds within the scene"
            )
        return self


class EventCandidateScoreRead(BaseModel):
    candidate_id: str
    rank: int
    raw_features: dict[str, Any]
    event_relevance_score: float
    trackability_score: float
    recommendation_score: float
    reason_codes: list[str]
    risk_codes: list[str]
    artifact_ids: dict[str, str] = Field(default_factory=dict)


class EventCandidateRankingRead(BaseModel):
    ranking_id: str
    status: str
    time_coordinate_system: Literal["SOURCE_VIDEO_SECONDS"]
    event_id: str
    event_label: str
    event_time_sec: float
    event_confidence: float | None
    scene_id: str
    scene_start_sec: float
    scene_end_sec: float
    shortlist_size: int
    shortlist: list[EventCandidateScoreRead]
    all_candidates: list[EventCandidateScoreRead]
    automatic_target_confirmation: Literal[False] = False
    shadow_only: bool = True


EventCandidateRole = Literal[
    "PRIMARY_EVENT_ACTOR",
    "DIRECTLY_RELATED_PLAYER",
    "BROADCAST_CLOSEUP_NON_ACTOR",
    "UNRELATED_PLAYER",
    "NON_PLAYER",
    "UNCERTAIN",
]


class EventCandidateLabelRequest(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=255)
    role: EventCandidateRole
    note: str | None = Field(default=None, max_length=1000)


class EventCandidateEvaluationRead(BaseModel):
    ranking_id: str
    status: Literal["MEASURED", "NOT_RUN"]
    candidate_generation_actor_coverage: float | None
    primary_actor_recall_at_1: float | None
    primary_actor_recall_at_3: float | None
    primary_actor_recall_at_5: float | None
    mrr: float | None
    broadcast_non_actor_top_1_rate: float | None
    labeled_candidate_count: int


class SceneAITaskRead(BaseModel):
    task_id: str
    task_type: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]
    attempt_count: int
    max_attempts: int
    result: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    status_url: str
    retry_url: str | None = None
