"""fix media_assets.size_bytes to BIGINT

media_assets.size_bytes가 INTEGER(최대 약 2.15GB)로 생성되어 있어서
2GB를 넘는 영상(예: 90분 풀경기 3.6GB) 업로드 시 오버플로우 에러가 발생했다.
app/domains/media/model.py의 SQLAlchemy 모델은 이미 BigInteger로 정의돼 있었으나
최초 마이그레이션(20260514_0001)이 Integer로 테이블을 만들어 실제 컬럼과 어긋나 있었다.

Revision ID: 20260722_0008
Revises: 20260722_0007
Create Date: 2026-07-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260722_0008"
down_revision: Union[str, None] = "20260722_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("media_assets") as batch_op:
        batch_op.alter_column(
            "size_bytes",
            type_=sa.BigInteger(),
            existing_type=sa.Integer(),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("media_assets") as batch_op:
        batch_op.alter_column(
            "size_bytes",
            type_=sa.Integer(),
            existing_type=sa.BigInteger(),
            existing_nullable=True,
        )
