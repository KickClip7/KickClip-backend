from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class EventCandidateSelectionR1(Base, TimestampMixin):
    """Immutable user selection and its exact ranking/media provenance."""

    __tablename__ = "event_candidate_selections_r1"

    selection_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("ecselr1"),
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    revision_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    scene_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ranking_id: Mapped[str] = mapped_column(String(64), nullable=False)
    shortlist_patch_id: Mapped[str] = mapped_column(String(64), nullable=False)
    discovery_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(255), nullable=False)
    shot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tracklet_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    candidate_manifest_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    candidate_media_bundle_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    source_video_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    reviewed_shot_boundaries_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    selection_artifact_path: Mapped[str] = mapped_column(
        String(2048), nullable=False
    )
    selection_artifact_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    media_bundle_manifest_path: Mapped[str] = mapped_column(
        String(2048), nullable=False
    )
    tracking_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("tracking_jobs.tracking_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    project = relationship("Project")
    owner = relationship("User")
    tracking_job = relationship("TrackingJob")


class EventCandidateReviewDecisionR1(Base, TimestampMixin):
    """Immutable four-way review decision for one tracking ambiguity."""

    __tablename__ = "event_candidate_review_decisions_r1"
    __table_args__ = (
        UniqueConstraint(
            "tracking_job_id",
            "ambiguity_id",
            "decision_artifact_sha256",
            name="uq_event_candidate_review_decision_r1",
        ),
    )

    decision_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("ecdecr1"),
    )
    tracking_job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tracking_jobs.tracking_job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reviewer_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
    )
    ambiguity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decision_state: Mapped[str] = mapped_column(String(64), nullable=False)
    confirmation_artifact_path: Mapped[str] = mapped_column(
        String(2048), nullable=False
    )
    decision_artifact_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True, index=True
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    tracking_job = relationship("TrackingJob")
    reviewer = relationship("User")


class EventCandidateHandoffPointerR1(Base, TimestampMixin):
    """Transactional pointer to the latest fully-written immutable state."""

    __tablename__ = "event_candidate_handoff_pointers_r1"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "revision_id",
            "event_id",
            "scene_id",
            name="uq_event_candidate_handoff_pointer_r1_scope",
        ),
    )

    pointer_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("ecptrr1"),
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scene_id: Mapped[str] = mapped_column(String(64), nullable=False)
    current_selection_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("event_candidate_selections_r1.selection_id"),
        nullable=True,
    )
    current_tracking_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("tracking_jobs.tracking_job_id"),
        nullable=True,
    )
    current_ambiguity_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    state_artifact_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    state_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    task_state: Mapped[str] = mapped_column(String(64), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_memory_revision_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    latest_decision_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    snapshot_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )


class EventCandidatePipelineR1(Base, TimestampMixin):
    """Authoritative transactional state for one R1 tracking pipeline."""

    __tablename__ = "event_candidate_pipelines_r1"

    pipeline_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("ecpipe")
    )
    tracking_job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tracking_jobs.tracking_job_id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True
    )
    selection_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("event_candidate_selections_r1.selection_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    generation: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    execution_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(64), nullable=False)
    current_shot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pending_ambiguity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_completed_ambiguity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latest_decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_memory_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_shot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state_artifact_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    state_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    pointer_snapshot_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    pointer_snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class EventCandidateAmbiguityR1(Base, TimestampMixin):
    __tablename__ = "event_candidate_ambiguities_r1"
    __table_args__ = (
        UniqueConstraint("tracking_job_id", "ambiguity_id", name="uq_event_candidate_ambiguity_r1"),
    )

    ambiguity_row_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("ecamb")
    )
    tracking_job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tracking_jobs.tracking_job_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    ambiguity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    shot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    candidates: Mapped[list] = mapped_column(JSON, nullable=False)
    full_frame_context_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    full_frame_context_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    shot_clip_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    shot_clip_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)


class EventCandidateMemoryRevisionR1(Base, TimestampMixin):
    __tablename__ = "event_candidate_memory_revisions_r1"

    memory_revision_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("ecmem")
    )
    tracking_job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tracking_jobs.tracking_job_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    source_candidate_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_shot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_tracklet_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(64), nullable=False)
    confirmation_decision_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    previous_memory_revision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    previous_memory_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    new_memory_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    reference_frame_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    references: Mapped[list] = mapped_column(JSON, nullable=False)
    scale_banks: Mapped[dict] = mapped_column(JSON, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class EventCandidateOutboxR1(Base, TimestampMixin):
    __tablename__ = "event_candidate_outbox_r1"

    outbox_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("ecout")
    )
    tracking_job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tracking_jobs.tracking_job_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
