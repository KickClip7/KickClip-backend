from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class Artifact(Base, TimestampMixin):
    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("art"),
    )
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    analysis_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("analysis_jobs.analysis_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    match = relationship("Match", back_populates="artifacts")
    analysis_job = relationship("AnalysisJob", back_populates="artifacts")