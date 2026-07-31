"""isolate scene target selection artifacts

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
    with op.batch_alter_table("scene_target_selections") as batch_op:
        batch_op.add_column(
            sa.Column("selection_artifact_root", sa.String(2048), nullable=True)
        )
        batch_op.add_column(
            sa.Column("target_selection_path", sa.String(2048), nullable=True)
        )
        batch_op.add_column(
            sa.Column("target_selection_sha256", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("target_reference_set_path", sa.String(2048), nullable=True)
        )
        batch_op.add_column(
            sa.Column("target_reference_set_sha256", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("earlier_proposals_path", sa.String(2048), nullable=True)
        )
        batch_op.add_column(
            sa.Column("earlier_proposals_sha256", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("earlier_decision_path", sa.String(2048), nullable=True)
        )
        batch_op.add_column(
            sa.Column("earlier_decision_sha256", sa.String(64), nullable=True)
        )

    op.execute(
        """
        UPDATE scene_target_selections
        SET selection_artifact_root = artifact_root,
            target_selection_path = artifact_root || '/target_selection.json',
            target_reference_set_path =
                artifact_root || '/target_reference_set.json',
            target_selection_sha256 = repeat('0', 64),
            target_reference_set_sha256 = repeat('0', 64)
        """
    )

    with op.batch_alter_table("scene_target_selections") as batch_op:
        batch_op.alter_column("selection_artifact_root", nullable=False)
        batch_op.alter_column("target_selection_path", nullable=False)
        batch_op.alter_column("target_selection_sha256", nullable=False)
        batch_op.alter_column("target_reference_set_path", nullable=False)
        batch_op.alter_column("target_reference_set_sha256", nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("scene_target_selections") as batch_op:
        for column in (
            "earlier_decision_sha256",
            "earlier_decision_path",
            "earlier_proposals_sha256",
            "earlier_proposals_path",
            "target_reference_set_sha256",
            "target_reference_set_path",
            "target_selection_sha256",
            "target_selection_path",
            "selection_artifact_root",
        ):
            batch_op.drop_column(column)
