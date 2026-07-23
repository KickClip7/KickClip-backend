from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.domains.auth.event_weights import default_event_weights
from app.utils.id_generator import generate_prefixed_id


class User(Base, TimestampMixin):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("usr"),
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="USER", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    developer_mode_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    event_weights: Mapped[dict[str, float]] = mapped_column(
        JSON,
        default=default_event_weights,
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    refresh_tokens = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class RefreshToken(Base, TimestampMixin):
    __tablename__ = "refresh_tokens"

    refresh_token_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("rtk"),
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    replaced_by_token_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user = relationship("User", back_populates="refresh_tokens")
