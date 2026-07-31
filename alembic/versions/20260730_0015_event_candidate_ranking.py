"""add event candidate ranking and proposal scope

Revision ID: 20260730_0015
Revises: 20260730_0014
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260730_0015"
down_revision: Union[str, None] = "20260730_0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("scene_target_selections") as batch_op:
        batch_op.create_foreign_key(
            "fk_target_selection_revision_candidate",
            "scene_player_candidates",
            ["revision_id", "selected_candidate_id"],
            ["revision_id", "candidate_id"],
            ondelete="RESTRICT",
        )
    with op.batch_alter_table("earlier_anchor_proposals") as batch_op:
        batch_op.add_column(
            sa.Column("source_revision_id", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("source_discovery_id", sa.String(255), nullable=True)
        )
    op.execute(
        """
        UPDATE earlier_anchor_proposals
        SET source_revision_id = (
                SELECT revision_id
                FROM scene_target_selections
                WHERE scene_target_selections.selection_id =
                    earlier_anchor_proposals.selection_id
            ),
            source_discovery_id = 'legacy'
        """
    )
    with op.batch_alter_table("earlier_anchor_proposals") as batch_op:
        batch_op.alter_column("source_revision_id", nullable=False)
        batch_op.alter_column("source_discovery_id", nullable=False)

    op.create_table(
        "event_candidate_rankings",
        sa.Column("ranking_id", sa.String(64), primary_key=True),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("match_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("scene_id", sa.String(64), nullable=False),
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("event_label", sa.String(64), nullable=False),
        sa.Column("event_time_sec", sa.Float(), nullable=False),
        sa.Column("event_confidence", sa.Float(), nullable=True),
        sa.Column("scene_start_sec", sa.Float(), nullable=False),
        sa.Column("scene_end_sec", sa.Float(), nullable=False),
        sa.Column("scene_candidate_manifest_sha256", sa.String(64), nullable=False),
        sa.Column("shot_boundaries_sha256", sa.String(64), nullable=False),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("feature_schema_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("shortlist_size", sa.Integer(), nullable=False),
        sa.Column("output_path", sa.String(2048), nullable=False),
        sa.Column("output_sha256", sa.String(64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["match_id"], ["matches.match_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["revision_id"], ["highlight_revisions.revision_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scene_id"], ["timeline_events.timeline_event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["timeline_events.timeline_event_id"], ondelete="RESTRICT"),
    )
    for column in (
        "owner_id", "match_id", "project_id", "revision_id", "scene_id",
        "event_id", "status",
    ):
        op.create_index(
            f"ix_event_candidate_rankings_{column}",
            "event_candidate_rankings",
            [column],
        )
    op.create_table(
        "event_candidate_scores",
        sa.Column("score_id", sa.String(64), primary_key=True),
        sa.Column("ranking_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(255), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("raw_features", sa.JSON(), nullable=False),
        sa.Column("event_relevance_score", sa.Float(), nullable=False),
        sa.Column("trackability_score", sa.Float(), nullable=False),
        sa.Column("recommendation_score", sa.Float(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("risk_codes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["ranking_id"],
            ["event_candidate_rankings.ranking_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "ranking_id",
            "candidate_id",
            name="uq_event_candidate_score_candidate",
        ),
    )
    op.create_index(
        "ix_event_candidate_scores_ranking_id",
        "event_candidate_scores",
        ["ranking_id"],
    )
    op.create_table(
        "event_candidate_labels",
        sa.Column("label_id", sa.String(64), primary_key=True),
        sa.Column("ranking_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(255), nullable=False),
        sa.Column("reviewer_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["ranking_id"],
            ["event_candidate_rankings.ranking_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"],
            ["users.user_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "ranking_id",
            "candidate_id",
            "reviewer_id",
            name="uq_event_candidate_label_reviewer",
        ),
    )
    for column in ("ranking_id", "reviewer_id", "role"):
        op.create_index(
            f"ix_event_candidate_labels_{column}",
            "event_candidate_labels",
            [column],
        )


def downgrade() -> None:
    for column in ("role", "reviewer_id", "ranking_id"):
        op.drop_index(
            f"ix_event_candidate_labels_{column}",
            table_name="event_candidate_labels",
        )
    op.drop_table("event_candidate_labels")
    op.drop_index(
        "ix_event_candidate_scores_ranking_id",
        table_name="event_candidate_scores",
    )
    op.drop_table("event_candidate_scores")
    for column in (
        "status", "event_id", "scene_id", "revision_id",
        "project_id", "match_id", "owner_id",
    ):
        op.drop_index(
            f"ix_event_candidate_rankings_{column}",
            table_name="event_candidate_rankings",
        )
    op.drop_table("event_candidate_rankings")
    with op.batch_alter_table("earlier_anchor_proposals") as batch_op:
        batch_op.drop_column("source_discovery_id")
        batch_op.drop_column("source_revision_id")
    with op.batch_alter_table("scene_target_selections") as batch_op:
        batch_op.drop_constraint(
            "fk_target_selection_revision_candidate",
            type_="foreignkey",
        )
