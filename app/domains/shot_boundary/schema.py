from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ShotBoundaryDraftItem(BaseModel):
    shot_id: str = Field(min_length=1, max_length=64)
    start_frame: int = Field(ge=0)
    end_frame_inclusive: int = Field(ge=0)
    start_seconds: float | None = None
    end_seconds_inclusive: float | None = None
    review_status: Literal["PENDING", "REVIEWED_PASS"] = "PENDING"


class ShotCutCandidate(BaseModel):
    cut_frame: int = Field(ge=1)
    score: float = Field(ge=0, le=1)
    hard_cut_candidate: bool = True
    before_thumbnail_url: str | None = None
    after_thumbnail_url: str | None = None


class ShotBoundaryReviewResponse(BaseModel):
    status: Literal[
        "NOT_PREPARED", "PREPARING", "WAITING_REVIEW", "CONFIRMED", "FAILED"
    ]
    project_id: str
    revision_id: str
    event_id: str
    scene_id: str
    review_session_id: str | None = None
    scene_video_url: str | None = None
    scene_video_sha256: str | None = None
    source_video_asset_id: str | None = None
    source_video_sha256: str | None = None
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    draft_revision: int = 0
    shots: list[ShotBoundaryDraftItem] = Field(default_factory=list)
    automatic_cuts: list[ShotCutCandidate] = Field(default_factory=list)
    contact_sheet_url: str | None = None
    confirmed_artifact_id: str | None = None
    detections_status: str | None = None
    automatic_confirmation: Literal[False] = False
    error: dict | None = None


class ShotBoundaryDraftUpdate(BaseModel):
    draft_revision: int = Field(ge=1)
    shots: list[ShotBoundaryDraftItem] = Field(min_length=1)
    note: str = Field(default="", max_length=2000)
    scene_video_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class ShotBoundaryConfirmRequest(BaseModel):
    draft_revision: int = Field(ge=1)
    reviewer_note: str = Field(min_length=1, max_length=2000)
    idempotency_key: str = Field(min_length=8, max_length=255)
    scene_video_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class ShotBoundaryConfirmResponse(BaseModel):
    status: Literal["CONFIRMED"] = "CONFIRMED"
    artifact_id: str
    artifact_sha256: str
    reviewed_shot_count: int
    scene_video_sha256: str
    automatic_confirmation: Literal[False] = False
    detections_status: str
    detections_artifact_id: str | None = None
    candidate_preparation_retry_started: bool = False
    preparation_task_id: str | None = None
