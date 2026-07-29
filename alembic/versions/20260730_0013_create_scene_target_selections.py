"""create scene-wide target selection tables

Revision ID: 20260730_0013
Revises: 20260728_0012
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260730_0013"
down_revision: Union[str, None] = "20260728_0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("scene_player_candidates") as batch_op:
        batch_op.alter_column(
            "candidate_id",
            existing_type=sa.String(64),
            type_=sa.String(255),
            existing_nullable=False,
        )
    op.create_table(
        "scene_target_selections",
        sa.Column("selection_id", sa.String(64), primary_key=True),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("match_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("scene_id", sa.String(64), nullable=False),
        sa.Column("selection_revision", sa.Integer(), nullable=False),
        sa.Column("selected_candidate_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("artifact_root", sa.String(2048), nullable=False),
        sa.Column("selection_artifact", sa.JSON(), nullable=False),
        sa.Column("reference_set_artifact", sa.JSON(), nullable=False),
        sa.Column("earlier_proposals_artifact", sa.JSON(), nullable=False),
        sa.Column("earlier_decision_artifact", sa.JSON(), nullable=False),
        sa.Column("candidate_cache_key", sa.String(64), nullable=False),
        sa.Column("tracking_cache_key", sa.String(64), nullable=True),
        sa.Column("tracking_job_id", sa.String(64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["match_id"], ["matches.match_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["revision_id"], ["highlight_revisions.revision_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scene_id"], ["timeline_events.timeline_event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tracking_job_id"], ["tracking_jobs.tracking_job_id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "revision_id",
            "scene_id",
            "selection_revision",
            name="uq_scene_target_selection_revision",
        ),
    )
    for column in (
        "owner_id", "match_id", "project_id", "revision_id", "scene_id",
        "selected_candidate_id", "status", "candidate_cache_key",
        "tracking_cache_key", "tracking_job_id",
    ):
        op.create_index(
            f"ix_scene_target_selections_{column}",
            "scene_target_selections",
            [column],
        )
    op.create_table(
        "scene_target_selection_references",
        sa.Column("reference_row_id", sa.String(64), primary_key=True),
        sa.Column("selection_id", sa.String(64), nullable=False),
        sa.Column("reference_id", sa.String(255), nullable=False),
        sa.Column("source_candidate_id", sa.String(255), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("bbox_xyxy", sa.JSON(), nullable=False),
        sa.Column("crop_artifact", sa.String(2048), nullable=False),
        sa.Column("quality", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["selection_id"],
            ["scene_target_selections.selection_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "selection_id",
            "reference_id",
            name="uq_scene_target_selection_reference",
        ),
    )
    op.create_index(
        "ix_scene_target_selection_references_selection_id",
        "scene_target_selection_references",
        ["selection_id"],
    )
    op.create_table(
        "earlier_anchor_proposals",
        sa.Column("proposal_id", sa.String(64), primary_key=True),
        sa.Column("selection_id", sa.String(64), nullable=False),
        sa.Column("candidate_id", sa.String(255), nullable=False),
        sa.Column("retrieval_rank", sa.Integer(), nullable=False),
        sa.Column("retrieval_score", sa.Float(), nullable=True),
        sa.Column("prototype_similarity", sa.Float(), nullable=True),
        sa.Column("decision_state", sa.String(64), nullable=False),
        sa.Column("artifacts", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["selection_id"],
            ["scene_target_selections.selection_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "selection_id",
            "candidate_id",
            name="uq_earlier_anchor_proposal_candidate",
        ),
    )
    op.create_index(
        "ix_earlier_anchor_proposals_selection_id",
        "earlier_anchor_proposals",
        ["selection_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_earlier_anchor_proposals_selection_id",
        table_name="earlier_anchor_proposals",
    )
    op.drop_table("earlier_anchor_proposals")
    op.drop_index(
        "ix_scene_target_selection_references_selection_id",
        table_name="scene_target_selection_references",
    )
    op.drop_table("scene_target_selection_references")
    for column in (
        "tracking_job_id", "tracking_cache_key", "candidate_cache_key", "status",
        "selected_candidate_id", "scene_id", "revision_id", "project_id",
        "match_id", "owner_id",
    ):
        op.drop_index(
            f"ix_scene_target_selections_{column}",
            table_name="scene_target_selections",
        )
    op.drop_table("scene_target_selections")
    with op.batch_alter_table("scene_player_candidates") as batch_op:
        batch_op.alter_column(
            "candidate_id",
            existing_type=sa.String(255),
            type_=sa.String(64),
            existing_nullable=False,
        )
