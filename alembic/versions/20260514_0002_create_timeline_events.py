"""create timeline events

Revision ID: 20260514_0002
Revises: 20260514_0001
Create Date: 2026-05-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260514_0002"
down_revision: Union[str, None] = "20260514_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "timeline_events",
        sa.Column("timeline_event_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("source_artifact_id", sa.String(length=64), nullable=True),
        sa.Column("source_job_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("half", sa.Integer(), nullable=True),
        sa.Column("timestamp_sec", sa.Float(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("highlight_score", sa.Float(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("team_name", sa.String(length=100), nullable=True),
        sa.Column("player_ids", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_artifact_id"],
            ["artifacts.artifact_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_job_id"],
            ["analysis_jobs.analysis_job_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("timeline_event_id"),
    )

    op.create_index("ix_timeline_events_match_id", "timeline_events", ["match_id"])
    op.create_index(
        "ix_timeline_events_source_artifact_id",
        "timeline_events",
        ["source_artifact_id"],
    )
    op.create_index(
        "ix_timeline_events_source_job_id",
        "timeline_events",
        ["source_job_id"],
    )
    op.create_index(
        "ix_timeline_events_event_type",
        "timeline_events",
        ["event_type"],
    )
    op.create_index("ix_timeline_events_label", "timeline_events", ["label"])
    op.create_index("ix_timeline_events_half", "timeline_events", ["half"])
    op.create_index(
        "ix_timeline_events_timestamp_sec",
        "timeline_events",
        ["timestamp_sec"],
    )
    op.create_index("ix_timeline_events_start_sec", "timeline_events", ["start_sec"])
    op.create_index("ix_timeline_events_end_sec", "timeline_events", ["end_sec"])


def downgrade() -> None:
    op.drop_index("ix_timeline_events_end_sec", table_name="timeline_events")
    op.drop_index("ix_timeline_events_start_sec", table_name="timeline_events")
    op.drop_index("ix_timeline_events_timestamp_sec", table_name="timeline_events")
    op.drop_index("ix_timeline_events_half", table_name="timeline_events")
    op.drop_index("ix_timeline_events_label", table_name="timeline_events")
    op.drop_index("ix_timeline_events_event_type", table_name="timeline_events")
    op.drop_index("ix_timeline_events_source_job_id", table_name="timeline_events")
    op.drop_index(
        "ix_timeline_events_source_artifact_id",
        table_name="timeline_events",
    )
    op.drop_index("ix_timeline_events_match_id", table_name="timeline_events")
    op.drop_table("timeline_events")