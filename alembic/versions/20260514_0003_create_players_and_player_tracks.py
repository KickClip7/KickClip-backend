"""create players and player tracks

Revision ID: 20260514_0003
Revises: 20260514_0002
Create Date: 2026-05-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260514_0003"
down_revision: Union[str, None] = "20260514_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "players",
        sa.Column("player_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("number", sa.Integer(), nullable=True),
        sa.Column("team_name", sa.String(length=100), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("identity_status", sa.String(length=32), nullable=False),
        sa.Column("profile_source", sa.String(length=100), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("player_id"),
    )
    op.create_index("ix_players_match_id", "players", ["match_id"])
    op.create_index("ix_players_number", "players", ["number"])
    op.create_index("ix_players_identity_status", "players", ["identity_status"])

    op.create_table(
        "player_tracks",
        sa.Column("player_track_id", sa.String(length=64), nullable=False),
        sa.Column("match_id", sa.String(length=64), nullable=False),
        sa.Column("player_id", sa.String(length=64), nullable=False),
        sa.Column("source_job_id", sa.String(length=64), nullable=True),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=False),
        sa.Column("track_artifact_id", sa.String(length=64), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("linked_event_ids", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.match_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["players.player_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_job_id"],
            ["analysis_jobs.analysis_job_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["track_artifact_id"],
            ["artifacts.artifact_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("player_track_id"),
    )
    op.create_index("ix_player_tracks_match_id", "player_tracks", ["match_id"])
    op.create_index("ix_player_tracks_player_id", "player_tracks", ["player_id"])
    op.create_index("ix_player_tracks_source_job_id", "player_tracks", ["source_job_id"])
    op.create_index("ix_player_tracks_start_sec", "player_tracks", ["start_sec"])
    op.create_index("ix_player_tracks_end_sec", "player_tracks", ["end_sec"])
    op.create_index(
        "ix_player_tracks_track_artifact_id",
        "player_tracks",
        ["track_artifact_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_player_tracks_track_artifact_id", table_name="player_tracks")
    op.drop_index("ix_player_tracks_end_sec", table_name="player_tracks")
    op.drop_index("ix_player_tracks_start_sec", table_name="player_tracks")
    op.drop_index("ix_player_tracks_source_job_id", table_name="player_tracks")
    op.drop_index("ix_player_tracks_player_id", table_name="player_tracks")
    op.drop_index("ix_player_tracks_match_id", table_name="player_tracks")
    op.drop_table("player_tracks")

    op.drop_index("ix_players_identity_status", table_name="players")
    op.drop_index("ix_players_number", table_name="players")
    op.drop_index("ix_players_match_id", table_name="players")
    op.drop_table("players")