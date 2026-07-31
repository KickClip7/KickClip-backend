"""create durable scene AI task queue

Revision ID: 20260730_0016
Revises: 20260730_0015
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260730_0016"
down_revision: Union[str, None] = "20260730_0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scene_ai_tasks",
        sa.Column("task_id", sa.String(64), primary_key=True),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("match_id", sa.String(64), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("process_pid", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["match_id"], ["matches.match_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
    )
    for column in (
        "owner_id", "match_id", "project_id", "task_type", "status",
        "idempotency_key",
    ):
        op.create_index(
            f"ix_scene_ai_tasks_{column}",
            "scene_ai_tasks",
            [column],
            unique=column == "idempotency_key",
        )


def downgrade() -> None:
    for column in (
        "idempotency_key", "status", "task_type", "project_id",
        "match_id", "owner_id",
    ):
        op.drop_index(
            f"ix_scene_ai_tasks_{column}",
            table_name="scene_ai_tasks",
        )
    op.drop_table("scene_ai_tasks")
