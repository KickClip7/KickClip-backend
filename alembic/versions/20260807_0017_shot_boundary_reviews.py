"""add fresh-machine shot-boundary review sessions

Revision ID: 20260807_0017
Revises: c2bde2254d48
Create Date: 2026-08-07
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260807_0017"
down_revision: Union[str, None] = "c2bde2254d48"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shot_boundary_review_sessions",
        sa.Column("session_id", sa.String(64), primary_key=True),
        sa.Column("owner_user_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=False),
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("scene_id", sa.String(64), nullable=False),
        sa.Column("source_video_asset_id", sa.String(64), nullable=False),
        sa.Column("scene_video_asset_id", sa.String(64), nullable=False),
        sa.Column("scene_video_artifact_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("session_revision", sa.Integer(), nullable=False),
        sa.Column("draft_revision", sa.Integer(), nullable=False),
        sa.Column("automatic_draft_json", sa.JSON(), nullable=False),
        sa.Column("draft_json", sa.JSON(), nullable=False),
        sa.Column("confirmed_artifact_id", sa.String(64), nullable=True),
        sa.Column("detections_artifact_id", sa.String(64), nullable=True),
        sa.Column("error_json", sa.JSON(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["revision_id"], ["highlight_revisions.revision_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["timeline_events.timeline_event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scene_id"], ["timeline_events.timeline_event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_video_asset_id"], ["media_assets.asset_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["scene_video_asset_id"], ["media_assets.asset_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["scene_video_artifact_id"], ["artifacts.artifact_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["confirmed_artifact_id"], ["artifacts.artifact_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["detections_artifact_id"], ["artifacts.artifact_id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "owner_user_id", "project_id", "revision_id", "event_id", "scene_id", "session_revision",
            name="uq_shot_boundary_review_session_scope_revision",
        ),
    )
    for column in ("owner_user_id", "project_id", "revision_id", "event_id", "scene_id", "status"):
        op.create_index(f"ix_shot_boundary_review_sessions_{column}", "shot_boundary_review_sessions", [column])

    op.create_table(
        "shot_boundary_review_decisions",
        sa.Column("decision_id", sa.String(64), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("reviewer_user_id", sa.String(64), nullable=False),
        sa.Column("draft_revision", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("automatic_confirmation", sa.Boolean(), nullable=False),
        sa.Column("artifact_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["shot_boundary_review_sessions.session_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewer_user_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.artifact_id"], ondelete="SET NULL"),
        sa.UniqueConstraint("session_id", "idempotency_key", name="uq_shot_boundary_decision_idempotency"),
    )
    op.create_index("ix_shot_boundary_review_decisions_session_id", "shot_boundary_review_decisions", ["session_id"])
    op.create_index("ix_shot_boundary_review_decisions_reviewer_user_id", "shot_boundary_review_decisions", ["reviewer_user_id"])


def downgrade() -> None:
    op.drop_index("ix_shot_boundary_review_decisions_reviewer_user_id", table_name="shot_boundary_review_decisions")
    op.drop_index("ix_shot_boundary_review_decisions_session_id", table_name="shot_boundary_review_decisions")
    op.drop_table("shot_boundary_review_decisions")
    for column in reversed(("owner_user_id", "project_id", "revision_id", "event_id", "scene_id", "status")):
        op.drop_index(f"ix_shot_boundary_review_sessions_{column}", table_name="shot_boundary_review_sessions")
    op.drop_table("shot_boundary_review_sessions")
