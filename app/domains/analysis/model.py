from datetime import datetime

from sqlalchemy import ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class AnalysisJob(Base, TimestampMixin):
    __tablename__ = "analysis_jobs"

    analysis_job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("job"),
    )
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    job_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_step: Mapped[str | None] = mapped_column(String(64), nullable=True)

    options: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    match = relationship("Match", back_populates="analysis_jobs")
    steps = relationship(
        "AnalysisJobStep",
        back_populates="analysis_job",
        cascade="all, delete-orphan",
        order_by="AnalysisJobStep.created_at",
    )
    artifacts = relationship(
        "Artifact",
        back_populates="analysis_job",
    )


class AnalysisJobStep(Base, TimestampMixin):
    __tablename__ = "analysis_job_steps"

    step_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("step"),
    )
    analysis_job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("analysis_jobs.analysis_job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    step_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis_job = relationship("AnalysisJob", back_populates="steps")