"""create event-aware player candidate ranking tables

Revision ID: 20260730_0014
Revises: 20260730_0013
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260730_0014"
down_revision: Union[str, None] = "20260730_0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "event_candidate_ranking_jobs",
        sa.Column("ranking_job_id", sa.String(64), primary_key=True),
        sa.Column("ranking_id", sa.String(64), nullable=True, unique=True),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("match_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("scene_id", sa.String(64), nullable=False),
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("event_label", sa.String(64), nullable=False),
        sa.Column("event_time_sec", sa.Float(), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("policy_state", sa.String(64), nullable=False),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("artifact_root", sa.String(2048), nullable=False),
        sa.Column("input_contract", sa.JSON(), nullable=False),
        sa.Column("feature_availability", sa.JSON(), nullable=False),
        sa.Column("ranking_artifact", sa.JSON(), nullable=False),
        sa.Column("shortlist_artifact", sa.JSON(), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["match_id"], ["matches.match_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["revision_id"], ["highlight_revisions.revision_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scene_id"], ["timeline_events.timeline_event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["timeline_events.timeline_event_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("cache_key", name="uq_event_candidate_ranking_jobs_cache_key"),
    )
    for column in (
        "owner_id",
        "match_id",
        "project_id",
        "revision_id",
        "scene_id",
        "event_id",
        "status",
        "cache_key",
    ):
        op.create_index(
            f"ix_event_candidate_ranking_jobs_{column}",
            "event_candidate_ranking_jobs",
            [column],
        )
    op.create_table(
        "event_candidate_features",
        sa.Column("feature_id", sa.String(64), primary_key=True),
        sa.Column("ranking_job_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(255), nullable=False),
        sa.Column("raw_features", sa.JSON(), nullable=False),
        sa.Column("normalized_features", sa.JSON(), nullable=False),
        sa.Column("score_components", sa.JSON(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("risk_codes", sa.JSON(), nullable=False),
        sa.Column("recommendation_rank", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["ranking_job_id"],
            ["event_candidate_ranking_jobs.ranking_job_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "ranking_job_id",
            "candidate_id",
            name="uq_event_candidate_features_job_candidate",
        ),
    )
    op.create_index(
        "ix_event_candidate_features_ranking_job_id",
        "event_candidate_features",
        ["ranking_job_id"],
    )
    op.create_index(
        "ix_event_candidate_features_candidate_id",
        "event_candidate_features",
        ["candidate_id"],
    )
    op.create_table(
        "event_candidate_recommendations",
        sa.Column("recommendation_id", sa.String(64), primary_key=True),
        sa.Column("ranking_job_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(255), nullable=False),
        sa.Column("recommendation_rank", sa.Integer(), nullable=False),
        sa.Column("hypothesis", sa.String(64), nullable=False),
        sa.Column("hypothesis_state", sa.String(32), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("risk_codes", sa.JSON(), nullable=False),
        sa.Column("score_components", sa.JSON(), nullable=False),
        sa.Column("artifacts", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["ranking_job_id"],
            ["event_candidate_ranking_jobs.ranking_job_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "ranking_job_id",
            "recommendation_rank",
            name="uq_event_candidate_recommendations_job_rank",
        ),
        sa.UniqueConstraint(
            "ranking_job_id",
            "candidate_id",
            name="uq_event_candidate_recommendations_job_candidate",
        ),
    )
    op.create_index(
        "ix_event_candidate_recommendations_ranking_job_id",
        "event_candidate_recommendations",
        ["ranking_job_id"],
    )
    op.create_index(
        "ix_event_candidate_recommendations_candidate_id",
        "event_candidate_recommendations",
        ["candidate_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_event_candidate_recommendations_candidate_id",
        table_name="event_candidate_recommendations",
    )
    op.drop_index(
        "ix_event_candidate_recommendations_ranking_job_id",
        table_name="event_candidate_recommendations",
    )
    op.drop_table("event_candidate_recommendations")
    op.drop_index(
        "ix_event_candidate_features_candidate_id",
        table_name="event_candidate_features",
    )
    op.drop_index(
        "ix_event_candidate_features_ranking_job_id",
        table_name="event_candidate_features",
    )
    op.drop_table("event_candidate_features")
    for column in reversed(
        (
            "owner_id",
            "match_id",
            "project_id",
            "revision_id",
            "scene_id",
            "event_id",
            "status",
            "cache_key",
        )
    ):
        op.drop_index(
            f"ix_event_candidate_ranking_jobs_{column}",
            table_name="event_candidate_ranking_jobs",
        )
    op.drop_table("event_candidate_ranking_jobs")
