from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class ClipPlan(Base, TimestampMixin):
    __tablename__ = "clip_plans"

    clip_plan_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("clip"),
    )

    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    mode: Mapped[str] = mapped_column(
        String(64),
        default="AGENT_GENERATED",
        nullable=False,
        index=True,
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    target_duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_by: Mapped[str] = mapped_column(
        String(64),
        default="agent",
        nullable=False,
        index=True,
    )

    options: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    match = relationship("Match")
    items = relationship(
        "ClipPlanItem",
        back_populates="clip_plan",
        cascade="all, delete-orphan",
        order_by="ClipPlanItem.order_index",
    )


class ClipPlanItem(Base, TimestampMixin):
    __tablename__ = "clip_plan_items"

    clip_plan_item_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("citem"),
    )

    clip_plan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("clip_plans.clip_plan_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    timeline_event_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)

    order_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    clip_plan = relationship("ClipPlan", back_populates="items")
    timeline_event = relationship("TimelineEvent")