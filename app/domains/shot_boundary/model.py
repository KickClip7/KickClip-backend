from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class ShotBoundaryReviewSession(Base, TimestampMixin):
    __tablename__ = "shot_boundary_review_sessions"
    __table_args__ = (
        UniqueConstraint(
            "owner_user_id",
            "project_id",
            "revision_id",
            "event_id",
            "scene_id",
            "session_revision",
            name="uq_shot_boundary_review_session_scope_revision",
        ),
    )

    session_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("sbreview")
    )
    owner_user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("highlight_revisions.revision_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_video_asset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("media_assets.asset_id", ondelete="RESTRICT"),
        nullable=False,
    )
    scene_video_asset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("media_assets.asset_id", ondelete="RESTRICT"),
        nullable=False,
    )
    scene_video_artifact_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    session_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    draft_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    automatic_draft_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    draft_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    confirmed_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="RESTRICT"),
        nullable=True,
    )
    detections_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
    )
    error_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    confirmed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    owner = relationship("User")
    project = relationship("Project")
    revision = relationship("HighlightRevision")
    event = relationship("TimelineEvent", foreign_keys=[event_id])
    scene = relationship("TimelineEvent", foreign_keys=[scene_id])
    source_video_asset = relationship(
        "MediaAsset", foreign_keys=[source_video_asset_id]
    )
    scene_video_asset = relationship("MediaAsset", foreign_keys=[scene_video_asset_id])
    scene_video_artifact = relationship(
        "Artifact", foreign_keys=[scene_video_artifact_id]
    )
    confirmed_artifact = relationship("Artifact", foreign_keys=[confirmed_artifact_id])
    detections_artifact = relationship(
        "Artifact", foreign_keys=[detections_artifact_id]
    )


class ShotBoundaryReviewDecision(Base, TimestampMixin):
    __tablename__ = "shot_boundary_review_decisions"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "idempotency_key",
            name="uq_shot_boundary_decision_idempotency",
        ),
    )

    decision_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("sbdecision")
    )
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("shot_boundary_review_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reviewer_user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    draft_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    automatic_confirmation: Mapped[bool] = mapped_column(default=False, nullable=False)
    artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
    )

    session = relationship("ShotBoundaryReviewSession")
    reviewer = relationship("User")
    artifact = relationship("Artifact")
