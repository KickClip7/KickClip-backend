from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EventCandidateRankingV11Request(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    event_label: str = Field(min_length=1, max_length=64)
    event_time_sec: float = Field(ge=0)
    event_confidence: float | None = Field(default=None, ge=0, le=1)
    scene_id: str = Field(min_length=1, max_length=64)
    scene_start_sec: float = Field(ge=0)
    scene_end_sec: float = Field(gt=0)
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


class EventCandidateRankingV11TaskResult(BaseModel):
    ranking_id: str
    artifact_id: str
    ranking_status: str
    automatic_target_confirmation: Literal[False] = False
    shortlist_candidate_ids: list[str]
    full_gallery_fallback_candidate_ids: list[str]
    output_sha256: str


class EventAnnotationReviewRequest(BaseModel):
    primary_actor_visible: bool
    primary_actor_in_candidate_set: bool
    primary_actor_candidate_ids: list[str] = Field(default_factory=list)
    directly_related_candidate_ids: list[str] = Field(default_factory=list)
    actor_missing_reason: str | None = Field(default=None, max_length=128)
    candidate_roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_actor_state(self):
        if (
            self.primary_actor_visible
            and not self.primary_actor_in_candidate_set
            and self.actor_missing_reason != "ACTOR_NOT_IN_CANDIDATE_SET"
        ):
            raise ValueError(
                "Visible missing actor requires ACTOR_NOT_IN_CANDIDATE_SET"
            )
        if (
            self.primary_actor_in_candidate_set
            and not self.primary_actor_candidate_ids
        ):
            raise ValueError("primary_actor_candidate_ids are required")
        return self


class EventAnnotationFinalizeRequest(BaseModel):
    review_artifact_ids: list[str] = Field(min_length=1)
    approval_mode: Literal["FINAL_APPROVED", "EXPLICIT_CONSENSUS"]
    approved_reviewer: str | None = Field(default=None, max_length=64)


class EventAnnotationArtifactRead(BaseModel):
    artifact_id: str
    ranking_id: str
    annotation_status: str
    approval_mode: str
    sha256: str


class EventCandidateEvaluationV11Read(BaseModel):
    ranking_id: str
    evaluation: dict
