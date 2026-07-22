"""add user event weights

Revision ID: 20260722_0007
Revises: 20260719_0006
Create Date: 2026-07-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260722_0007"
down_revision: Union[str, None] = "20260719_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "event_weights",
            sa.JSON(),
            server_default=sa.text(
                "'{\"goal\": 0.30, \"penalty\": 0.24, \"shot\": 0.18, "
                "\"card\": 0.13, \"corner\": 0.09, "
                "\"substitution\": 0.06}'"
            ),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "event_weights")
