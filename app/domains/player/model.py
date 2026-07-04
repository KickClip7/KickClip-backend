from sqlalchemy import Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class Player(Base, TimestampMixin):
    __tablename__ = "players"

    player_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("player"),
    )

    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    number: Mapped[int | None] = mapped_column(nullable=True, index=True)
    team_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)

    identity_status: Mapped[str] = mapped_column(
        String(32),
        default="UNKNOWN",
        nullable=False,
        index=True,
    )
    profile_source: Mapped[str | None] = mapped_column(String(100), nullable=True)

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    match = relationship("Match", back_populates="players")

    tracks = relationship(
        "PlayerTrack",
        back_populates="player",
        cascade="all, delete-orphan",
    )


class PlayerTrack(Base, TimestampMixin):
    __tablename__ = "player_tracks"

    player_track_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("ptrack"),
    )

    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    player_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("players.player_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("analysis_jobs.analysis_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    start_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)

    track_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    linked_event_ids: Mapped[list[str]] = mapped_column(
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

    match = relationship("Match", back_populates="player_tracks")
    player = relationship("Player", back_populates="tracks")