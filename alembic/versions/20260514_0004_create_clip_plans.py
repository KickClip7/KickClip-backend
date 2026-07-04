"""create clip plans

Revision ID: 20260514_0004
Revises: 20260514_0003
Create Date: 2026-05-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260514_0004"
down_revision: Union[str, None] = "20260514_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "clip_plans",
        sa.Column("clip_plan_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("target_duration_sec", sa.Float(), nullable=True),
        sa.Column("actual_duration_sec", sa.Float(), nullable=True),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("clip_plan_id"),
    )
    op.create_index("ix_clip_plans_match_id", "clip_plans", ["match_id"])
    op.create_index("ix_clip_plans_mode", "clip_plans", ["mode"])
    op.create_index("ix_clip_plans_created_by", "clip_plans", ["created_by"])

    op.create_table(
        "clip_plan_items",
        sa.Column("clip_plan_item_id", sa.String(length=64), nullable=False),
        sa.Column("clip_plan_id", sa.String(length=64), nullable=False),
        sa.Column("timeline_event_id", sa.String(length=64), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["clip_plan_id"],
            ["clip_plans.clip_plan_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["timeline_event_id"],
            ["timeline_events.timeline_event_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("clip_plan_item_id"),
    )
    op.create_index(
        "ix_clip_plan_items_clip_plan_id",
        "clip_plan_items",
        ["clip_plan_id"],
    )
    op.create_index(
        "ix_clip_plan_items_timeline_event_id",
        "clip_plan_items",
        ["timeline_event_id"],
    )
    op.create_index(
        "ix_clip_plan_items_order_index",
        "clip_plan_items",
        ["order_index"],
    )


def downgrade() -> None:
    op.drop_index("ix_clip_plan_items_order_index", table_name="clip_plan_items")
    op.drop_index("ix_clip_plan_items_timeline_event_id", table_name="clip_plan_items")
    op.drop_index("ix_clip_plan_items_clip_plan_id", table_name="clip_plan_items")
    op.drop_table("clip_plan_items")

    op.drop_index("ix_clip_plans_created_by", table_name="clip_plans")
    op.drop_index("ix_clip_plans_mode", table_name="clip_plans")
    op.drop_index("ix_clip_plans_match_id", table_name="clip_plans")
    op.drop_table("clip_plans")