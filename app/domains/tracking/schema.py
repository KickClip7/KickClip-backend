from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.validation import validate_bbox_xyxy

BBox = Annotated[list[float], Field(min_length=4, max_length=4)]


class TrackingJobCreateRequest(BaseModel):
    """Start tracking from a server-owned video and a source-pixel bbox."""

    media_asset_id: str = Field(
        min_length=1,
        description="Accessible KickClip video MediaAsset identifier.",
    )
    initial_bbox_xyxy: BBox = Field(
        description="[x1, y1, x2, y2] in original-video pixels.",
    )
    bbox_format: Literal["xyxy_pixels"] = Field(
        default="xyxy_pixels",
        description="V1 supports source-resolution xyxy pixels only.",
    )
    project_id: str | None = Field(
        default=None,
        description="Optional project associated with the MediaAsset's Match.",
    )
    match_id: str | None = Field(
        default=None,
        description="Optional consistency check; the MediaAsset remains authoritative.",
    )
    reacquisition_mode: Literal["assisted"] = Field(
        default="assisted",
        description=(
            "Assisted mode never confirms a cross-shot candidate without a user action."
        ),
    )

    @field_validator("initial_bbox_xyxy")
    @classmethod
    def validate_bbox_shape(cls, value: list[float]) -> list[float]:
        return validate_bbox_xyxy(value)


class ReviewStage(str, Enum):
    MEMORY = "MEMORY"
    SEGMENT = "SEGMENT"
    STAGE2B = "STAGE2B"
    STAGE2D = "STAGE2D"
    STAGE2D1 = "STAGE2D1"
    STAGE2D2 = "STAGE2D2"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class TrackingReviewRequest(BaseModel):
    stage: ReviewStage = Field(
        description="The exact review stage currently requested by pipeline_state.json.",
    )
    decision: ReviewDecision
    note: str = Field(default="", max_length=1000)


class AmbiguityDecision(str, Enum):
    CANDIDATE = "candidate"
    ABSENT = "absent"


class TrackingAmbiguityConfirmationRequest(BaseModel):
    decision: AmbiguityDecision
    candidate_id: str | None = Field(default=None, max_length=255)
    note: str = Field(default="", max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_candidate_decision(cls, value: Any) -> Any:
        """Accept the candidate-only payload emitted by the current frontend."""

        if not isinstance(value, dict) or value.get("decision"):
            return value
        normalized = dict(value)
        candidate_id = str(normalized.get("candidate_id") or "").strip()
        if candidate_id.lower() == AmbiguityDecision.ABSENT.value:
            normalized["decision"] = AmbiguityDecision.ABSENT.value
            normalized["candidate_id"] = None
        elif candidate_id:
            normalized["decision"] = AmbiguityDecision.CANDIDATE.value
        return normalized

    @field_validator("candidate_id")
    @classmethod
    def normalize_candidate_id(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class TrackingJobCreateResponse(BaseModel):
    job_id: str
    status: TrackingBackendStatus
    outcome: str
    progress: int = Field(default=0, ge=0, le=100)
    retryable: bool = False
    reused: bool = False
    status_url: str


class TrackingArtifactRead(BaseModel):
    key: str
    kind: str
    exists: bool
    mime_type: str | None = None
    size_bytes: int | None = None
    updated_at: datetime | None = None
    sha256: str | None = None
    url: str | None = None


class TrackingArtifactsResponse(BaseModel):
    job_id: str
    artifacts: list[TrackingArtifactRead] = Field(default_factory=list)


class TrackingCandidateRead(BaseModel):
    candidate_id: str
    rank: int | None = None
    score: float | None = None
    frame_image_urls: list[str] = Field(default_factory=list)
    shot_id: str | None = None
    tracklet_id: str | None = None
    reviewability: str | None = None
    best_frame: int | None = None
    review_bundle: dict[str, Any] = Field(default_factory=dict)


class TrackingPendingActionResponse(BaseModel):
    type: str
    review_stage: str | None = None
    ambiguity_id: str | None = None
    shot_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[TrackingCandidateRead] = Field(default_factory=list)
    recommended_candidate: str | None = None
    artifact_keys: list[str] = Field(default_factory=list)
    required_action: str


class TrackingErrorRead(BaseModel):
    type: str | None = None
    message: str | None = None


class TrackingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    owner_user_id: str
    match_id: str
    project_id: str | None
    media_asset_id: str
    status: TrackingBackendStatus
    execution_kind: str
    pipeline_stage: str | None
    processing_status: str
    outcome: str
    progress: int = Field(ge=0, le=100)
    retryable: bool
    status_url: str
    pipeline_status: str | None
    pipeline_decision: str | None
    current_stage: str | None
    pending_ambiguity_id: str | None
    pending_candidates: list[TrackingCandidateRead] = Field(default_factory=list)
    latest_decision: dict[str, Any] | None = None
    current_memory_revision: dict[str, Any] | None = None
    next_ambiguity: dict[str, Any] | None = None
    current_shot: str | None = None
    next_shot: str | None = None
    completed: bool
    completed_at: datetime | None = None
    failure_code: str | None = None
    artifact_readiness: dict[str, bool] = Field(default_factory=dict)
    preview_readiness: dict[str, bool] = Field(default_factory=dict)
    initial_bbox_xyxy: list[float]
    bbox_format: str
    device: str
    reacquisition_mode: str
    pending_action: TrackingPendingActionResponse | None
    artifacts_url: str
    timeline_url: str | None
    schema_version: str | None
    pipeline_version: str | None
    video_sha256: str | None
    frozen_manifest_present: bool
    process_return_code: int | None
    error: TrackingErrorRead | None
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None


class TrackingTimelineResponse(BaseModel):
    """Public form of kickclip.target_centric_e2e.v1.

    The runtime's local video path is replaced by a MediaAsset reference.
    Frame ranges are inclusive.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str
    pipeline_version: str
    test_name: str
    target_id: str
    status: str
    video: dict[str, Any]
    shots: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    ambiguities: list[dict[str, Any]]
    confirmations: list[dict[str, Any]]
    provenance: dict[str, Any]
    range: dict[str, int | None]


class TrackingDiagnosticsResponse(BaseModel):
    enabled: bool
    available: bool
    checked_at: datetime
    code: str
    message: str
    verifier_return_code: int | None = None
    components: dict[str, bool] = Field(default_factory=dict)
