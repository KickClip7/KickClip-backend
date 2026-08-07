"""create candidate handoff r1 tables

Revision ID: c2bde2254d48
Revises: 20260730_0016
Create Date: 2026-08-07 00:39:03.486491
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c2bde2254d48"
down_revision: Union[str, None] = "20260730_0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # tracking_jobs: R1 orchestration state.
    # server_default is used so existing rows can be migrated safely.
    op.add_column(
        "tracking_jobs",
        sa.Column(
            "execution_kind",
            sa.String(length=64),
            nullable=False,
            server_default="LEGACY_TARGET_TRACKING",
        ),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("pipeline_stage", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column(
            "processing_status",
            sa.String(length=64),
            nullable=False,
            server_default="IDLE",
        ),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("current_shot_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("last_completed_ambiguity_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("latest_decision_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("current_memory_revision_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("next_shot_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "tracking_jobs",
        sa.Column("failure_code", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_tracking_jobs_execution_kind",
        "tracking_jobs",
        ["execution_kind"],
        unique=False,
    )

    op.create_table(
        "event_candidate_ambiguities_r1",
        sa.Column("ambiguity_row_id", sa.String(length=64), nullable=False),
        sa.Column("tracking_job_id", sa.String(length=64), nullable=False),
        sa.Column("ambiguity_id", sa.String(length=128), nullable=False),
        sa.Column("shot_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("candidate_ids", sa.JSON(), nullable=False),
        sa.Column("candidates", sa.JSON(), nullable=False),
        sa.Column("full_frame_context_path", sa.String(length=2048), nullable=False),
        sa.Column("full_frame_context_sha256", sa.String(length=64), nullable=False),
        sa.Column("shot_clip_path", sa.String(length=2048), nullable=False),
        sa.Column("shot_clip_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("ambiguity_row_id"),
        sa.UniqueConstraint(
            "tracking_job_id",
            "ambiguity_id",
            name="uq_event_candidate_ambiguity_r1",
        ),
    )
    op.create_index(
        "ix_event_candidate_ambiguities_r1_tracking_job_id",
        "event_candidate_ambiguities_r1",
        ["tracking_job_id"],
        unique=False,
    )

    op.create_table(
        "event_candidate_memory_revisions_r1",
        sa.Column("memory_revision_id", sa.String(length=64), nullable=False),
        sa.Column("tracking_job_id", sa.String(length=64), nullable=False),
        sa.Column("source_candidate_id", sa.String(length=255), nullable=False),
        sa.Column("source_shot_id", sa.String(length=64), nullable=False),
        sa.Column("source_tracklet_id", sa.String(length=128), nullable=False),
        sa.Column("reviewer_id", sa.String(length=64), nullable=False),
        sa.Column("confirmation_decision_id", sa.String(length=64), nullable=False),
        sa.Column("previous_memory_revision_id", sa.String(length=64), nullable=True),
        sa.Column("previous_memory_sha256", sa.String(length=64), nullable=True),
        sa.Column("new_memory_sha256", sa.String(length=64), nullable=False),
        sa.Column("reference_frame_ids", sa.JSON(), nullable=False),
        sa.Column("references", sa.JSON(), nullable=False),
        sa.Column("scale_banks", sa.JSON(), nullable=False),
        sa.Column("artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_revision_id"),
        sa.UniqueConstraint("confirmation_decision_id"),
    )
    op.create_index(
        "ix_event_candidate_memory_revisions_r1_tracking_job_id",
        "event_candidate_memory_revisions_r1",
        ["tracking_job_id"],
        unique=False,
    )

    op.create_table(
        "event_candidate_outbox_r1",
        sa.Column("outbox_id", sa.String(length=64), nullable=False),
        sa.Column("tracking_job_id", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("artifact_path", sa.String(length=2048), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("outbox_id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index(
        "ix_event_candidate_outbox_r1_tracking_job_id",
        "event_candidate_outbox_r1",
        ["tracking_job_id"],
        unique=False,
    )

    op.create_table(
        "event_candidate_review_decisions_r1",
        sa.Column("decision_id", sa.String(length=64), nullable=False),
        sa.Column("tracking_job_id", sa.String(length=64), nullable=False),
        sa.Column("reviewer_id", sa.String(length=64), nullable=False),
        sa.Column("ambiguity_id", sa.String(length=128), nullable=False),
        sa.Column("candidate_id", sa.String(length=255), nullable=True),
        sa.Column("decision_state", sa.String(length=64), nullable=False),
        sa.Column("confirmation_artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("decision_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["reviewer_id"],
            ["users.user_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("decision_id"),
        sa.UniqueConstraint(
            "tracking_job_id",
            "ambiguity_id",
            "decision_artifact_sha256",
            name="uq_event_candidate_review_decision_r1",
        ),
    )
    op.create_index(
        "ix_event_candidate_review_decisions_r1_idempotency_key",
        "event_candidate_review_decisions_r1",
        ["idempotency_key"],
        unique=True,
    )
    op.create_index(
        "ix_event_candidate_review_decisions_r1_tracking_job_id",
        "event_candidate_review_decisions_r1",
        ["tracking_job_id"],
        unique=False,
    )

    op.create_table(
        "event_candidate_selections_r1",
        sa.Column("selection_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("owner_id", sa.String(length=64), nullable=False),
        sa.Column("revision_id", sa.String(length=64), nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("scene_id", sa.String(length=64), nullable=False),
        sa.Column("ranking_id", sa.String(length=64), nullable=False),
        sa.Column("shortlist_patch_id", sa.String(length=64), nullable=False),
        sa.Column("discovery_id", sa.String(length=128), nullable=False),
        sa.Column("candidate_id", sa.String(length=255), nullable=False),
        sa.Column("shot_id", sa.String(length=64), nullable=False),
        sa.Column("tracklet_id", sa.String(length=128), nullable=False),
        sa.Column("selected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("candidate_media_bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_video_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewed_shot_boundaries_sha256", sa.String(length=64), nullable=False),
        sa.Column("selection_artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("selection_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("media_bundle_manifest_path", sa.String(length=2048), nullable=False),
        sa.Column("tracking_job_id", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.user_id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("selection_id"),
        sa.UniqueConstraint("selection_artifact_sha256"),
    )
    for column in (
        "event_id",
        "owner_id",
        "project_id",
        "revision_id",
        "scene_id",
        "tracking_job_id",
    ):
        op.create_index(
            f"ix_event_candidate_selections_r1_{column}",
            "event_candidate_selections_r1",
            [column],
            unique=False,
        )

    op.create_table(
        "event_candidate_handoff_pointers_r1",
        sa.Column("pointer_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("revision_id", sa.String(length=64), nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("scene_id", sa.String(length=64), nullable=False),
        sa.Column("current_selection_id", sa.String(length=64), nullable=True),
        sa.Column("current_tracking_job_id", sa.String(length=64), nullable=True),
        sa.Column("current_ambiguity_id", sa.String(length=128), nullable=True),
        sa.Column("state_artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("state_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("task_state", sa.String(length=64), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("current_stage", sa.String(length=64), nullable=True),
        sa.Column("current_memory_revision_id", sa.String(length=64), nullable=True),
        sa.Column("latest_decision_id", sa.String(length=64), nullable=True),
        sa.Column("snapshot_sha256", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["current_selection_id"],
            ["event_candidate_selections_r1.selection_id"],
        ),
        sa.ForeignKeyConstraint(
            ["current_tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("pointer_id"),
        sa.UniqueConstraint(
            "project_id",
            "revision_id",
            "event_id",
            "scene_id",
            name="uq_event_candidate_handoff_pointer_r1_scope",
        ),
    )
    op.create_index(
        "ix_event_candidate_handoff_pointers_r1_project_id",
        "event_candidate_handoff_pointers_r1",
        ["project_id"],
        unique=False,
    )

    op.create_table(
        "event_candidate_pipelines_r1",
        sa.Column("pipeline_id", sa.String(length=64), nullable=False),
        sa.Column("tracking_job_id", sa.String(length=64), nullable=False),
        sa.Column("selection_id", sa.String(length=64), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("execution_kind", sa.String(length=64), nullable=False),
        sa.Column("pipeline_stage", sa.String(length=64), nullable=False),
        sa.Column("processing_status", sa.String(length=64), nullable=False),
        sa.Column("current_shot_id", sa.String(length=64), nullable=True),
        sa.Column("pending_ambiguity_id", sa.String(length=128), nullable=True),
        sa.Column("last_completed_ambiguity_id", sa.String(length=128), nullable=True),
        sa.Column("latest_decision_id", sa.String(length=64), nullable=True),
        sa.Column("current_memory_revision_id", sa.String(length=64), nullable=True),
        sa.Column("next_shot_id", sa.String(length=64), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(length=128), nullable=True),
        sa.Column("state_artifact_path", sa.String(length=2048), nullable=False),
        sa.Column("state_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("pointer_snapshot_path", sa.String(length=2048), nullable=False),
        sa.Column("pointer_snapshot_sha256", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["selection_id"],
            ["event_candidate_selections_r1.selection_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tracking_job_id"],
            ["tracking_jobs.tracking_job_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("pipeline_id"),
    )
    op.create_index(
        "ix_event_candidate_pipelines_r1_selection_id",
        "event_candidate_pipelines_r1",
        ["selection_id"],
        unique=False,
    )
    op.create_index(
        "ix_event_candidate_pipelines_r1_tracking_job_id",
        "event_candidate_pipelines_r1",
        ["tracking_job_id"],
        unique=True,
    )

    # Keep model defaults in Python; the temporary DB defaults were only needed
    # to backfill existing tracking_jobs rows during this migration.
    op.alter_column("tracking_jobs", "execution_kind", server_default=None)
    op.alter_column("tracking_jobs", "processing_status", server_default=None)


def downgrade() -> None:
    op.drop_index(
        "ix_event_candidate_pipelines_r1_tracking_job_id",
        table_name="event_candidate_pipelines_r1",
    )
    op.drop_index(
        "ix_event_candidate_pipelines_r1_selection_id",
        table_name="event_candidate_pipelines_r1",
    )
    op.drop_table("event_candidate_pipelines_r1")

    op.drop_index(
        "ix_event_candidate_handoff_pointers_r1_project_id",
        table_name="event_candidate_handoff_pointers_r1",
    )
    op.drop_table("event_candidate_handoff_pointers_r1")

    for column in reversed(
        (
            "event_id",
            "owner_id",
            "project_id",
            "revision_id",
            "scene_id",
            "tracking_job_id",
        )
    ):
        op.drop_index(
            f"ix_event_candidate_selections_r1_{column}",
            table_name="event_candidate_selections_r1",
        )
    op.drop_table("event_candidate_selections_r1")

    op.drop_index(
        "ix_event_candidate_review_decisions_r1_tracking_job_id",
        table_name="event_candidate_review_decisions_r1",
    )
    op.drop_index(
        "ix_event_candidate_review_decisions_r1_idempotency_key",
        table_name="event_candidate_review_decisions_r1",
    )
    op.drop_table("event_candidate_review_decisions_r1")

    op.drop_index(
        "ix_event_candidate_outbox_r1_tracking_job_id",
        table_name="event_candidate_outbox_r1",
    )
    op.drop_table("event_candidate_outbox_r1")

    op.drop_index(
        "ix_event_candidate_memory_revisions_r1_tracking_job_id",
        table_name="event_candidate_memory_revisions_r1",
    )
    op.drop_table("event_candidate_memory_revisions_r1")

    op.drop_index(
        "ix_event_candidate_ambiguities_r1_tracking_job_id",
        table_name="event_candidate_ambiguities_r1",
    )
    op.drop_table("event_candidate_ambiguities_r1")

    op.drop_index("ix_tracking_jobs_execution_kind", table_name="tracking_jobs")
    op.drop_column("tracking_jobs", "failure_code")
    op.drop_column("tracking_jobs", "completed_at")
    op.drop_column("tracking_jobs", "next_shot_id")
    op.drop_column("tracking_jobs", "current_memory_revision_id")
    op.drop_column("tracking_jobs", "latest_decision_id")
    op.drop_column("tracking_jobs", "last_completed_ambiguity_id")
    op.drop_column("tracking_jobs", "current_shot_id")
    op.drop_column("tracking_jobs", "processing_status")
    op.drop_column("tracking_jobs", "pipeline_stage")
    op.drop_column("tracking_jobs", "execution_kind")
