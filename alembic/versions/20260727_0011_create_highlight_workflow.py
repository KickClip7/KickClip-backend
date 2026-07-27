"""create unified highlight workflow revisions and scene tracking bindings

Revision ID: 20260727_0011
Revises: 20260727_0010
Create Date: 2026-07-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260727_0011"
down_revision: Union[str, None] = "20260727_0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "media_assets",
        sa.Column("sha256", sa.String(64), nullable=True),
    )
    op.create_index("ix_media_assets_sha256", "media_assets", ["sha256"])

    op.add_column(
        "analysis_jobs",
        sa.Column("media_asset_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("cache_key", sa.String(64), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("video_sha256", sa.String(64), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("model_version", sa.String(64), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("policy_version", sa.String(64), nullable=True),
    )
    op.create_foreign_key(
        "fk_analysis_jobs_media_asset_id",
        "analysis_jobs",
        "media_assets",
        ["media_asset_id"],
        ["asset_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_analysis_jobs_media_asset_id",
        "analysis_jobs",
        ["media_asset_id"],
    )
    op.create_index(
        "ix_analysis_jobs_cache_key",
        "analysis_jobs",
        ["cache_key"],
        unique=True,
    )

    op.create_table(
        "player_focus_subjects",
        sa.Column("focus_subject_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("identity_source", sa.String(32), nullable=False),
        sa.Column("anchor_scene_id", sa.String(64), nullable=False),
        sa.Column("anchor_candidate_id", sa.String(64), nullable=False),
        sa.Column("appearance_memory_ref", sa.String(1024), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.project_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["anchor_scene_id"],
            ["timeline_events.timeline_event_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("focus_subject_id"),
    )
    op.create_index(
        "ix_player_focus_subjects_project_id",
        "player_focus_subjects",
        ["project_id"],
    )

    op.create_table(
        "highlight_revisions",
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("parent_revision_id", sa.String(64), nullable=True),
        sa.Column("action_spotting_job_id", sa.String(64), nullable=True),
        sa.Column("user_request", sa.Text(), nullable=False),
        sa.Column("structured_request", sa.JSON(), nullable=False),
        sa.Column("selected_scene_ids", sa.JSON(), nullable=False),
        sa.Column("scene_selection", sa.JSON(), nullable=False),
        sa.Column("focus_mode", sa.String(16), nullable=False),
        sa.Column("focus_subject_id", sa.String(64), nullable=True),
        sa.Column("clip_plan_id", sa.String(64), nullable=True),
        sa.Column("render_job_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("pending_action", sa.String(64), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.project_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_revision_id"],
            ["highlight_revisions.revision_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["action_spotting_job_id"],
            ["analysis_jobs.analysis_job_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["focus_subject_id"],
            ["player_focus_subjects.focus_subject_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["clip_plan_id"],
            ["clip_plans.clip_plan_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["render_job_id"],
            ["render_jobs.render_job_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("revision_id"),
        sa.UniqueConstraint(
            "project_id",
            "revision_number",
            name="uq_highlight_revisions_project_number",
        ),
    )
    for column in (
        "project_id",
        "parent_revision_id",
        "action_spotting_job_id",
        "focus_mode",
        "focus_subject_id",
        "status",
    ):
        op.create_index(
            f"ix_highlight_revisions_{column}",
            "highlight_revisions",
            [column],
        )

    op.create_table(
        "scene_player_candidates",
        sa.Column("candidate_row_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("scene_id", sa.String(64), nullable=False),
        sa.Column("anchor_time_sec", sa.Float(), nullable=False),
        sa.Column("anchor_source_time_sec", sa.Float(), nullable=False),
        sa.Column("anchor_frame_index", sa.Integer(), nullable=False),
        sa.Column("bbox_xyxy", sa.JSON(), nullable=False),
        sa.Column("thumbnail_artifact_id", sa.String(64), nullable=True),
        sa.Column("track_length_frames", sa.Integer(), nullable=False),
        sa.Column("trackability_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["highlight_revisions.revision_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["scene_id"],
            ["timeline_events.timeline_event_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["thumbnail_artifact_id"],
            ["artifacts.artifact_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("candidate_row_id"),
        sa.UniqueConstraint(
            "revision_id",
            "candidate_id",
            name="uq_scene_player_candidates_revision_candidate",
        ),
    )
    for column in ("revision_id", "scene_id", "status"):
        op.create_index(
            f"ix_scene_player_candidates_{column}",
            "scene_player_candidates",
            [column],
        )

    op.create_table(
        "scene_tracking_bindings",
        sa.Column("binding_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("scene_id", sa.String(64), nullable=False),
        sa.Column("focus_subject_id", sa.String(64), nullable=False),
        sa.Column("selected_candidate_id", sa.String(64), nullable=True),
        sa.Column("tracking_job_id", sa.String(64), nullable=True),
        sa.Column("scene_clip_asset_id", sa.String(64), nullable=True),
        sa.Column("source_start_time_sec", sa.Float(), nullable=True),
        sa.Column("source_end_time_sec", sa.Float(), nullable=True),
        sa.Column("source_start_frame", sa.Integer(), nullable=True),
        sa.Column("source_end_frame", sa.Integer(), nullable=True),
        sa.Column("source_fps", sa.Float(), nullable=True),
        sa.Column("clip_fps", sa.Float(), nullable=True),
        sa.Column("clip_frame_count", sa.Integer(), nullable=True),
        sa.Column("confirmation_source", sa.String(32), nullable=False),
        sa.Column("target_presence_status", sa.String(32), nullable=False),
        sa.Column("render_strategy", sa.String(32), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("timeline_summary", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["revision_id"],
            ["highlight_revisions.revision_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["scene_id"],
            ["timeline_events.timeline_event_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["focus_subject_id"],
            ["player_focus_subjects.focus_subject_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["scene_clip_asset_id"],
            ["media_assets.asset_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("binding_id"),
        sa.UniqueConstraint(
            "revision_id",
            "scene_id",
            name="uq_scene_tracking_bindings_revision_scene",
        ),
    )
    for column in (
        "revision_id",
        "scene_id",
        "focus_subject_id",
        "tracking_job_id",
        "target_presence_status",
        "status",
    ):
        op.create_index(
            f"ix_scene_tracking_bindings_{column}",
            "scene_tracking_bindings",
            [column],
        )


def downgrade() -> None:
    op.drop_table("scene_tracking_bindings")
    op.drop_table("scene_player_candidates")
    op.drop_table("highlight_revisions")
    op.drop_table("player_focus_subjects")

    op.drop_index("ix_analysis_jobs_cache_key", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_media_asset_id", table_name="analysis_jobs")
    op.drop_constraint(
        "fk_analysis_jobs_media_asset_id",
        "analysis_jobs",
        type_="foreignkey",
    )
    for column in (
        "policy_version",
        "model_version",
        "video_sha256",
        "cache_key",
        "media_asset_id",
    ):
        op.drop_column("analysis_jobs", column)

    op.drop_index("ix_media_assets_sha256", table_name="media_assets")
    op.drop_column("media_assets", "sha256")
