from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.execution import LEGACY_EXECUTION_KIND
from app.utils.id_generator import generate_prefixed_id


class TrackingJob(Base, TimestampMixin):
    """Durable adapter state for one external target-centric pipeline run."""

    __tablename__ = "tracking_jobs"

    tracking_job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("trk"),
    )
    owner_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    media_asset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("media_assets.asset_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    test_name: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
        index=True,
    )
    initial_bbox: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    bbox_format: Mapped[str] = mapped_column(
        String(32),
        default="xyxy_pixels",
        nullable=False,
    )
    device: Mapped[str] = mapped_column(String(16), nullable=False)
    reacquisition_mode: Mapped[str] = mapped_column(
        String(32),
        default="assisted",
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(64),
        default=TrackingBackendStatus.QUEUED.value,
        nullable=False,
        index=True,
    )
    execution_kind: Mapped[str] = mapped_column(
        String(64), default=LEGACY_EXECUTION_KIND, nullable=False, index=True
    )
    pipeline_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_status: Mapped[str] = mapped_column(
        String(64), default="IDLE", nullable=False
    )
    current_shot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_completed_ambiguity_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    latest_decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_memory_revision_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    next_shot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pipeline_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pipeline_decision: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pending_action_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    pending_ambiguity_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    output_directory: Mapped[str] = mapped_column(String(2048), nullable=False)
    pipeline_state_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    timeline_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    summary_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    tracking_preview_path: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
    )
    target_centered_preview_path: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
    )

    queued_action: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    artifact_index: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    runtime_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    last_action_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    schema_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    video_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frozen_manifest_present: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    process_return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_attempt: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    runtime_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    runtime_finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    owner = relationship("User")
    match = relationship("Match")
    project = relationship("Project")
    media_asset = relationship("MediaAsset")
