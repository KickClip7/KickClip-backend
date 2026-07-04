from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class Match(Base, TimestampMixin):
    __tablename__ = "matches"

    match_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("match"),
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    home_team: Mapped[str | None] = mapped_column(String(100), nullable=True)
    away_team: Mapped[str | None] = mapped_column(String(100), nullable=True)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    match_date: Mapped[datetime | None] = mapped_column(nullable=True)
    competition: Mapped[str | None] = mapped_column(String(100), nullable=True)
    season: Mapped[str | None] = mapped_column(String(50), nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    project = relationship("Project", back_populates="matches")

    media_assets = relationship(
        "MediaAsset",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    analysis_jobs = relationship(
        "AnalysisJob",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    artifacts = relationship(
        "Artifact",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    timeline_events = relationship(
        "TimelineEvent",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    players = relationship(
        "Player",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    player_tracks = relationship(
        "PlayerTrack",
        back_populates="match",
        cascade="all, delete-orphan",
    )