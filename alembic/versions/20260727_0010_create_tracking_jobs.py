"""create durable target-centric tracking jobs

Revision ID: 20260727_0010
Revises: 20260725_0009
Create Date: 2026-07-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260727_0010"
down_revision: Union[str, None] = "20260725_0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tracking_jobs",
        sa.Column("tracking_job_id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("match_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=True),
        sa.Column("media_asset_id", sa.String(64), nullable=False),
        sa.Column("test_name", sa.String(128), nullable=False),
        sa.Column("initial_bbox", sa.JSON(), nullable=False),
        sa.Column("bbox_format", sa.String(32), nullable=False, server_default="xyxy_pixels"),
        sa.Column("device", sa.String(16), nullable=False),
        sa.Column("reacquisition_mode", sa.String(32), nullable=False, server_default="assisted"),
        sa.Column("status", sa.String(64), nullable=False, server_default="QUEUED"),
        sa.Column("pipeline_status", sa.String(64), nullable=True),
        sa.Column("pipeline_decision", sa.String(255), nullable=True),
        sa.Column("current_stage", sa.String(64), nullable=True),
        sa.Column("pending_action_type", sa.String(64), nullable=True),
        sa.Column("pending_ambiguity_id", sa.String(128), nullable=True),
        sa.Column("output_directory", sa.String(2048), nullable=False),
        sa.Column("pipeline_state_path", sa.String(2048), nullable=False),
        sa.Column("timeline_path", sa.String(2048), nullable=True),
        sa.Column("summary_path", sa.String(2048), nullable=True),
        sa.Column("tracking_preview_path", sa.String(2048), nullable=True),
        sa.Column("target_centered_preview_path", sa.String(2048), nullable=True),
        sa.Column(
            "queued_action",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column(
            "artifact_index",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column(
            "runtime_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column("last_action_key", sa.String(512), nullable=True),
        sa.Column("schema_version", sa.String(128), nullable=True),
        sa.Column("pipeline_version", sa.String(64), nullable=True),
        sa.Column("video_sha256", sa.String(64), nullable=True),
        sa.Column(
            "frozen_manifest_present",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("process_return_code", sa.Integer(), nullable=True),
        sa.Column("process_pid", sa.Integer(), nullable=True),
        sa.Column("run_attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_type", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("runtime_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("runtime_finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["match_id"], ["matches.match_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["media_asset_id"],
            ["media_assets.asset_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("tracking_job_id"),
        sa.UniqueConstraint("test_name"),
    )
    op.create_index("ix_tracking_jobs_owner_id", "tracking_jobs", ["owner_id"])
    op.create_index("ix_tracking_jobs_match_id", "tracking_jobs", ["match_id"])
    op.create_index("ix_tracking_jobs_project_id", "tracking_jobs", ["project_id"])
    op.create_index(
        "ix_tracking_jobs_media_asset_id",
        "tracking_jobs",
        ["media_asset_id"],
    )
    op.create_index(
        "ix_tracking_jobs_test_name",
        "tracking_jobs",
        ["test_name"],
        unique=True,
    )
    op.create_index("ix_tracking_jobs_status", "tracking_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_tracking_jobs_status", table_name="tracking_jobs")
    op.drop_index("ix_tracking_jobs_test_name", table_name="tracking_jobs")
    op.drop_index("ix_tracking_jobs_media_asset_id", table_name="tracking_jobs")
    op.drop_index("ix_tracking_jobs_project_id", table_name="tracking_jobs")
    op.drop_index("ix_tracking_jobs_match_id", table_name="tracking_jobs")
    op.drop_index("ix_tracking_jobs_owner_id", table_name="tracking_jobs")
    op.drop_table("tracking_jobs")
