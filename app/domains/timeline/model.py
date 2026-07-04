from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class TimelineEvent(Base, TimestampMixin):
    __tablename__ = "timeline_events"

    timeline_event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("evt"),
    )

    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    source_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("analysis_jobs.analysis_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    half: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    highlight_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    team_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    player_ids: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    match = relationship("Match", back_populates="timeline_events")