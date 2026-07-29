"""create highlight editor drafts

Revision ID: 20260728_0012
Revises: 20260727_0011
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260728_0012"
down_revision: str | None = "20260727_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "highlight_drafts",
        sa.Column("draft_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("revision_id", sa.String(64), nullable=True),
        sa.Column("clip_plan_id", sa.String(64), nullable=True),
        sa.Column("saved_by_user_id", sa.String(64), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
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
            ["revision_id"],
            ["highlight_revisions.revision_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["clip_plan_id"],
            ["clip_plans.clip_plan_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["saved_by_user_id"],
            ["users.user_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("draft_id"),
        sa.UniqueConstraint(
            "project_id",
            name="uq_highlight_drafts_project_id",
        ),
    )
    for column in (
        "project_id",
        "revision_id",
        "clip_plan_id",
        "saved_by_user_id",
    ):
        op.create_index(
            f"ix_highlight_drafts_{column}",
            "highlight_drafts",
            [column],
        )


def downgrade() -> None:
    for column in (
        "saved_by_user_id",
        "clip_plan_id",
        "revision_id",
        "project_id",
    ):
        op.drop_index(
            f"ix_highlight_drafts_{column}",
            table_name="highlight_drafts",
        )
    op.drop_table("highlight_drafts")
