from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class RenderJob(Base, TimestampMixin):
    __tablename__ = "render_jobs"

    render_job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("render"),
    )

    clip_plan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("clip_plans.clip_plan_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="queued",
        nullable=False,
        index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    ratio: Mapped[str] = mapped_column(String(16), default="9:16", nullable=False)
    resolution: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quality: Mapped[str] = mapped_column(String(16), default="1080p", nullable=False)

    captions_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    music_asset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    output_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    subtitle_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    options: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    runtime_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    clip_plan = relationship("ClipPlan", back_populates="render_jobs")
    output_artifact = relationship("Artifact", foreign_keys=[output_artifact_id])
    subtitle_artifact = relationship("Artifact", foreign_keys=[subtitle_artifact_id])
