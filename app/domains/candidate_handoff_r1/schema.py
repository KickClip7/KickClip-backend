from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Reviewability(str, Enum):
    CLEAR = "CLEAR"
    USABLE = "USABLE"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    UNREVIEWABLE = "UNREVIEWABLE"


class Visibility(str, Enum):
    VISIBLE = "VISIBLE"
    PARTIAL = "PARTIAL"
    NOT_VISIBLE = "NOT_VISIBLE"
    UNKNOWN = "UNKNOWN"


class CandidateReviewState(str, Enum):
    SAME_PLAYER = "SAME_PLAYER"
    DIFFERENT_PLAYER = "DIFFERENT_PLAYER"
    UNREVIEWABLE_LOW_RESOLUTION = "UNREVIEWABLE_LOW_RESOLUTION"
    TARGET_ABSENT = "TARGET_ABSENT"


class CandidateMediaRead(BaseModel):
    full_frame_context_url: str
    best_crop_native_url: str
    best_crop_display_url: str
    first_middle_last_url: str
    tracklet_video_url: str
    reference_gallery_url: str


class EventCandidateRecommendationRead(BaseModel):
    ranking_version: Literal["v1.2"]
    ranking_id: str
    shortlist_patch_id: str
    candidate_id: str
    shortlist_rank: int = Field(ge=1)
    global_rank: int = Field(ge=1)
    shot_id: str
    tracklet_id: str
    reviewability: Reviewability
    best_frame: int = Field(ge=0)
    media: CandidateMediaRead
    reason_codes: list[str] = Field(default_factory=list)
    risk_codes: list[str] = Field(default_factory=list)
    possible_fragment_duplicate: bool = False
    automatic_target_confirmation: Literal[False] = False


class EventCandidateRecommendationResponse(BaseModel):
    ranking_version: Literal["v1.2"]
    ranking_id: str
    shortlist_patch_id: str
    revision_id: str
    event_id: str
    scene_id: str
    source_video_asset_id: str
    candidates: list[EventCandidateRecommendationRead]
    automatic_target_confirmation: Literal[False] = False
    production_recommendation_ui: Literal["BLOCKED"] = "BLOCKED"


class CandidateRecommendationPrepareRequest(BaseModel):
    shortlist_size: Literal[5] = 5


class EventCandidateSelectionCreateRequest(BaseModel):
    ranking_id: str = Field(min_length=1, max_length=64)
    shortlist_patch_id: str = Field(min_length=1, max_length=64)
    candidate_id: str = Field(min_length=1, max_length=255)
    source_video_asset_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )


class EventCandidateSelectionRead(BaseModel):
    selection_id: str
    project_id: str
    revision_id: str
    event_id: str
    scene_id: str
    ranking_id: str
    shortlist_patch_id: str
    discovery_id: str
    candidate_id: str
    shot_id: str
    tracklet_id: str
    user_id: str
    selected_at: datetime
    candidate_manifest_sha256: str
    candidate_media_bundle_sha256: str
    source_video_sha256: str
    reviewed_shot_boundaries_sha256: str
    selection_artifact_sha256: str
    tracking_job_id: str | None = None


class EventCandidateTrackingCreateRequest(BaseModel):
    source_video_asset_id: str | None = Field(
        default=None, min_length=1, max_length=64
    )


class EventCandidateTrackingCreateResponse(BaseModel):
    tracking_job_id: str
    selection_id: str
    selected_candidate_id: str
    selected_shot_id: str
    selected_reference_frame: int
    selected_bbox: list[float]
    anchor_mode: Literal["SELECTED_CANDIDATE_BEST_REFERENCE"]
    provenance_artifact_sha256: str
    status: str


class CandidateReviewDecisionRequest(BaseModel):
    state: CandidateReviewState
    candidate_id: str | None = Field(default=None, max_length=255)
    full_frame_context_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    shot_clip_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    note: str = Field(default="", max_length=1000)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=255)

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.state == CandidateReviewState.TARGET_ABSENT:
            if self.candidate_id is not None:
                raise ValueError(
                    "TARGET_ABSENT is a full-shot decision and has no candidate_id."
                )
            if not self.full_frame_context_sha256 or not self.shot_clip_sha256:
                raise ValueError(
                    "TARGET_ABSENT requires full-frame context and shot clip evidence."
                )
        elif not self.candidate_id:
            raise ValueError(f"{self.state.value} requires candidate_id.")
        return self


class CandidateQuality(BaseModel):
    bbox_width_px: float
    bbox_height_px: float
    bbox_area_ratio: float
    sharpness: float
    blur_score: float
    border_clipping: bool
    occlusion_estimate: float
    available_observation_count: int
    identity_consistency_score: float
    jersey_number_visibility: Visibility
    face_visibility: Visibility
    reviewability: Reviewability
    selected_best_frame: int
    selected_best_frame_reason: str
    reference_frame_ids: list[int] = Field(default_factory=list)
    identity_pure: bool
    purity_status: str
    purity_diagnostics: dict[str, Any] = Field(default_factory=dict)
