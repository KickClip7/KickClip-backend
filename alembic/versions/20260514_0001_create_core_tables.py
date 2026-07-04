"""create core tables

Revision ID: 20260514_0001
Revises:
Create Date: 2026-05-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260514_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("owner_id", sa.String(length=64), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("thumbnail_artifact_id", sa.String(length=64), nullable=True),
        sa.Column("last_opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_index("ix_projects_owner_id", "projects", ["owner_id"])

    op.create_table(
        "matches",
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("home_team", sa.String(length=100), nullable=True),
        sa.Column("away_team", sa.String(length=100), nullable=True),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("match_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("competition", sa.String(length=100), nullable=True),
        sa.Column("season", sa.String(length=50), nullable=True),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.project_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("match_id"),
    )
    op.create_index("ix_matches_project_id", "matches", ["project_id"])

    op.create_table(
        "media_assets",
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("asset_type", sa.String(length=64), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("asset_id"),
    )
    op.create_index("ix_media_assets_match_id", "media_assets", ["match_id"])
    op.create_index("ix_media_assets_asset_type", "media_assets", ["asset_type"])

    op.create_table(
        "analysis_jobs",
        sa.Column("analysis_job_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("current_step", sa.String(length=64), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("analysis_job_id"),
    )
    op.create_index("ix_analysis_jobs_match_id", "analysis_jobs", ["match_id"])
    op.create_index("ix_analysis_jobs_job_type", "analysis_jobs", ["job_type"])

    op.create_table(
        "analysis_job_steps",
        sa.Column("step_id", sa.String(length=64), nullable=False),
        sa.Column("analysis_job_id", sa.String(length=64), nullable=False),
        sa.Column("step_key", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_job_id"],
            ["analysis_jobs.analysis_job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("step_id"),
    )
    op.create_index(
        "ix_analysis_job_steps_analysis_job_id",
        "analysis_job_steps",
        ["analysis_job_id"],
    )
    op.create_index(
        "ix_analysis_job_steps_step_key",
        "analysis_job_steps",
        ["step_key"],
    )

    op.create_table(
        "artifacts",
        sa.Column("artifact_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("analysis_job_id", sa.String(length=64), nullable=True),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["analysis_job_id"],
            ["analysis_jobs.analysis_job_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("artifact_id"),
    )
    op.create_index("ix_artifacts_match_id", "artifacts", ["match_id"])
    op.create_index("ix_artifacts_analysis_job_id", "artifacts", ["analysis_job_id"])
    op.create_index("ix_artifacts_artifact_type", "artifacts", ["artifact_type"])


def downgrade() -> None:
    op.drop_index("ix_artifacts_artifact_type", table_name="artifacts")
    op.drop_index("ix_artifacts_analysis_job_id", table_name="artifacts")
    op.drop_index("ix_artifacts_match_id", table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index("ix_analysis_job_steps_step_key", table_name="analysis_job_steps")
    op.drop_index(
        "ix_analysis_job_steps_analysis_job_id",
        table_name="analysis_job_steps",
    )
    op.drop_table("analysis_job_steps")

    op.drop_index("ix_analysis_jobs_job_type", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_match_id", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")

    op.drop_index("ix_media_assets_asset_type", table_name="media_assets")
    op.drop_index("ix_media_assets_match_id", table_name="media_assets")
    op.drop_table("media_assets")

    op.drop_index("ix_matches_project_id", table_name="matches")
    op.drop_table("matches")

    op.drop_index("ix_projects_owner_id", table_name="projects")
    op.drop_table("projects")