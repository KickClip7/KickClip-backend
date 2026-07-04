from datetime import datetime

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("proj"),
    )
    owner_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT", nullable=False)
    thumbnail_artifact_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_opened_at: Mapped[datetime | None] = mapped_column(nullable=True)

    matches = relationship(
        "Match",
        back_populates="project",
        cascade="all, delete-orphan",
    )