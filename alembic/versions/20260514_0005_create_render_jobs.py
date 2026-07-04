"""create render jobs

Revision ID: 20260514_0005
Revises: 20260514_0004
Create Date: 2026-05-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260514_0005"
down_revision: Union[str, None] = "20260514_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "render_jobs",
        sa.Column("render_job_id", sa.String(length=64), nullable=False),
        sa.Column("clip_plan_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("ratio", sa.String(length=16), nullable=False),
        sa.Column("resolution", sa.String(length=32), nullable=True),
        sa.Column("quality", sa.String(length=16), nullable=False),
        sa.Column("captions_enabled", sa.Boolean(), nullable=False),
        sa.Column("music_asset_id", sa.String(length=64), nullable=True),
        sa.Column("output_artifact_id", sa.String(length=64), nullable=True),
        sa.Column("subtitle_artifact_id", sa.String(length=64), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("runtime_sec", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["clip_plan_id"],
            ["clip_plans.clip_plan_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["output_artifact_id"],
            ["artifacts.artifact_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["subtitle_artifact_id"],
            ["artifacts.artifact_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("render_job_id"),
    )
    op.create_index("ix_render_jobs_clip_plan_id", "render_jobs", ["clip_plan_id"])
    op.create_index("ix_render_jobs_status", "render_jobs", ["status"])
    op.create_index("ix_render_jobs_output_artifact_id", "render_jobs", ["output_artifact_id"])
    op.create_index("ix_render_jobs_subtitle_artifact_id", "render_jobs", ["subtitle_artifact_id"])


def downgrade() -> None:
    op.drop_index("ix_render_jobs_subtitle_artifact_id", table_name="render_jobs")
    op.drop_index("ix_render_jobs_output_artifact_id", table_name="render_jobs")
    op.drop_index("ix_render_jobs_status", table_name="render_jobs")
    op.drop_index("ix_render_jobs_clip_plan_id", table_name="render_jobs")
    op.drop_table("render_jobs")
