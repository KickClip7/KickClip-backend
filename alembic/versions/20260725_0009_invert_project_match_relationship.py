"""invert project-match ownership and scope clip plans to projects

Revision ID: 20260725_0009
Revises: 20260722_0008
Create Date: 2026-07-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260725_0009"
down_revision: Union[str, None] = "20260722_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Expansion phase: all ownership columns remain nullable until backfill ends.
    op.add_column("matches", sa.Column("owner_id", sa.String(64), nullable=True))
    op.add_column("projects", sa.Column("match_id", sa.String(64), nullable=True))
    op.add_column("clip_plans", sa.Column("project_id", sa.String(64), nullable=True))
    op.add_column("artifacts", sa.Column("project_id", sa.String(64), nullable=True))

    op.create_index("ix_matches_owner_id", "matches", ["owner_id"])
    op.create_index("ix_projects_match_id", "projects", ["match_id"])
    op.create_index("ix_clip_plans_project_id", "clip_plans", ["project_id"])
    op.create_index("ix_artifacts_project_id", "artifacts", ["project_id"])

    op.create_foreign_key(
        "fk_projects_match_id_matches",
        "projects",
        "matches",
        ["match_id"],
        ["match_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_clip_plans_project_id_projects",
        "clip_plans",
        "projects",
        ["project_id"],
        ["project_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_artifacts_project_id_projects",
        "artifacts",
        "projects",
        ["project_id"],
        ["project_id"],
        ondelete="CASCADE",
    )

    # Retained as an audit/backward-conversion map. It records which projects were
    # cloned when one legacy Project owned more than one Match.
    op.create_table(
        "project_match_legacy_map",
        sa.Column("match_id", sa.String(64), nullable=False),
        sa.Column("legacy_project_id", sa.String(64), nullable=False),
        sa.Column("mapped_project_id", sa.String(64), nullable=False),
        sa.Column("project_was_cloned", sa.Boolean(), nullable=False),
        sa.Column("placeholder_match", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("match_id"),
        sa.UniqueConstraint("mapped_project_id"),
    )

    # Some early installations allowed owner IDs without a users FK. Preserve the
    # IDs by creating disabled legacy accounts rather than dropping ownership.
    op.execute(
        """
        INSERT INTO users (
            user_id, email, password_hash, display_name, role, is_active,
            developer_mode_enabled, event_weights, last_login_at,
            created_at, updated_at
        )
        SELECT
            p.owner_id,
            'legacy-' || md5(p.owner_id) || '@kickclip.invalid',
            '!',
            'Migrated legacy owner',
            'SYSTEM',
            false,
            false,
            '{}'::json,
            NULL,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        FROM (
            SELECT DISTINCT owner_id
            FROM projects
            WHERE owner_id IS NOT NULL
        ) p
        LEFT JOIN users u ON u.user_id = p.owner_id
        WHERE u.user_id IS NULL
        """
    )
    op.execute(
        """
        INSERT INTO users (
            user_id, email, password_hash, display_name, role, is_active,
            developer_mode_enabled, event_weights, last_login_at,
            created_at, updated_at
        )
        SELECT
            'usr_legacy_unowned',
            'legacy-unowned@kickclip.invalid',
            '!',
            'Migrated unowned data',
            'SYSTEM',
            false,
            false,
            '{}'::json,
            NULL,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        WHERE EXISTS (SELECT 1 FROM projects WHERE owner_id IS NULL)
          AND NOT EXISTS (
              SELECT 1 FROM users WHERE user_id = 'usr_legacy_unowned'
          )
        """
    )
    op.execute(
        """
        UPDATE projects
        SET owner_id = 'usr_legacy_unowned'
        WHERE owner_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE matches m
        SET owner_id = p.owner_id
        FROM projects p
        WHERE p.project_id = m.project_id
        """
    )

    # First Match keeps the original Project. Every additional Match receives an
    # exact Project clone, so no per-project edit state is discarded.
    op.execute(
        """
        INSERT INTO project_match_legacy_map (
            match_id, legacy_project_id, mapped_project_id,
            project_was_cloned, placeholder_match
        )
        SELECT
            ranked.match_id,
            ranked.project_id,
            CASE
                WHEN ranked.position = 1 THEN ranked.project_id
                ELSE 'proj_mig_' || md5(ranked.match_id)
            END,
            ranked.position > 1,
            false
        FROM (
            SELECT
                m.match_id,
                m.project_id,
                row_number() OVER (
                    PARTITION BY m.project_id
                    ORDER BY m.created_at, m.match_id
                ) AS position
            FROM matches m
        ) ranked
        """
    )
    op.execute(
        """
        INSERT INTO projects (
            project_id, owner_id, title, description, status,
            thumbnail_artifact_id, last_opened_at, created_at, updated_at,
            match_id
        )
        SELECT
            map.mapped_project_id,
            source.owner_id,
            source.title,
            source.description,
            source.status,
            source.thumbnail_artifact_id,
            source.last_opened_at,
            source.created_at,
            source.updated_at,
            map.match_id
        FROM project_match_legacy_map map
        JOIN projects source ON source.project_id = map.legacy_project_id
        WHERE map.project_was_cloned
        """
    )
    op.execute(
        """
        UPDATE projects p
        SET match_id = map.match_id
        FROM project_match_legacy_map map
        WHERE map.mapped_project_id = p.project_id
          AND p.match_id IS NULL
        """
    )

    # A legacy Project was allowed to exist before a video/Match was attached.
    # Preserve it with an explicitly marked placeholder Match instead of deleting it.
    op.execute(
        """
        INSERT INTO matches (
            match_id, owner_id, home_team, away_team, home_score, away_score,
            match_date, competition, season, duration_sec, metadata,
            created_at, updated_at, project_id
        )
        SELECT
            'match_mig_' || md5(p.project_id),
            p.owner_id,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            '{"kickclip_migration": {"legacy_orphan_project": true}}'::json,
            p.created_at,
            p.updated_at,
            p.project_id
        FROM projects p
        WHERE p.match_id IS NULL
        """
    )
    op.execute(
        """
        INSERT INTO project_match_legacy_map (
            match_id, legacy_project_id, mapped_project_id,
            project_was_cloned, placeholder_match
        )
        SELECT
            'match_mig_' || md5(p.project_id),
            p.project_id,
            p.project_id,
            false,
            true
        FROM projects p
        WHERE p.match_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE projects p
        SET match_id = map.match_id
        FROM project_match_legacy_map map
        WHERE map.mapped_project_id = p.project_id
          AND map.placeholder_match
        """
    )

    op.execute(
        """
        UPDATE clip_plans cp
        SET project_id = map.mapped_project_id
        FROM project_match_legacy_map map
        WHERE map.match_id = cp.match_id
        """
    )

    # A thumbnail copied from a legacy multi-Match Project is retained only by the
    # Project whose Match actually owns that artifact.
    op.execute(
        """
        UPDATE projects p
        SET thumbnail_artifact_id = NULL
        FROM artifacts a
        WHERE p.thumbnail_artifact_id = a.artifact_id
          AND a.match_id <> p.match_id
        """
    )
    op.execute(
        """
        UPDATE artifacts a
        SET project_id = p.project_id
        FROM projects p
        WHERE p.thumbnail_artifact_id = a.artifact_id
          AND a.match_id = p.match_id
        """
    )

    # Existing render/subtitle artifacts become project-scoped. Analysis artifacts
    # intentionally keep project_id NULL and remain shared at Match level.
    op.execute(
        """
        UPDATE artifacts a
        SET project_id = cp.project_id
        FROM render_jobs rj
        JOIN clip_plans cp ON cp.clip_plan_id = rj.clip_plan_id
        WHERE a.artifact_id IN (rj.output_artifact_id, rj.subtitle_artifact_id)
        """
    )

    # Preserve duplicate files but move extra singleton assets to an explicit
    # legacy type before adding the partial uniqueness invariant.
    op.execute(
        """
        WITH duplicates AS (
            SELECT
                asset_id,
                asset_type,
                row_number() OVER (
                    PARTITION BY match_id, asset_type
                    ORDER BY created_at DESC, asset_id DESC
                ) AS position
            FROM media_assets
            WHERE asset_type IN ('RAW_VIDEO', 'WEB_PREVIEW_VIDEO')
        )
        UPDATE media_assets ma
        SET asset_type = duplicates.asset_type || '_LEGACY_DUPLICATE_'
            || duplicates.position::text
        FROM duplicates
        WHERE duplicates.asset_id = ma.asset_id
          AND duplicates.position > 1
        """
    )
    op.create_index(
        "uq_media_assets_match_singleton_video",
        "media_assets",
        ["match_id", "asset_type"],
        unique=True,
        postgresql_where=sa.text(
            "asset_type IN ('RAW_VIDEO', 'WEB_PREVIEW_VIDEO')"
        ),
    )

    # Contract phase.
    op.create_foreign_key(
        "fk_matches_owner_id_users",
        "matches",
        "users",
        ["owner_id"],
        ["user_id"],
        ondelete="RESTRICT",
    )
    op.alter_column("matches", "owner_id", nullable=False)
    op.alter_column("projects", "match_id", nullable=False)
    op.alter_column("clip_plans", "project_id", nullable=False)

    op.drop_constraint(
        "clip_plans_match_id_fkey",
        "clip_plans",
        type_="foreignkey",
    )
    op.drop_index("ix_clip_plans_match_id", table_name="clip_plans")
    op.drop_column("clip_plans", "match_id")

    op.drop_constraint("matches_project_id_fkey", "matches", type_="foreignkey")
    op.drop_index("ix_matches_project_id", table_name="matches")
    op.drop_column("matches", "project_id")


def downgrade() -> None:
    # Re-expand the legacy direction.
    op.add_column("matches", sa.Column("project_id", sa.String(64), nullable=True))
    op.add_column("clip_plans", sa.Column("match_id", sa.String(64), nullable=True))

    op.execute(
        """
        UPDATE clip_plans cp
        SET match_id = p.match_id
        FROM projects p
        WHERE p.project_id = cp.project_id
        """
    )
    op.execute(
        """
        UPDATE matches m
        SET project_id = chosen.project_id
        FROM (
            SELECT DISTINCT ON (p.match_id)
                p.match_id,
                COALESCE(map.legacy_project_id, p.project_id) AS project_id
            FROM projects p
            LEFT JOIN project_match_legacy_map map
                ON map.mapped_project_id = p.project_id
            ORDER BY p.match_id, map.project_was_cloned, p.created_at, p.project_id
        ) chosen
        WHERE chosen.match_id = m.match_id
        """
    )

    # Matches uploaded after this migration may never have received a Project.
    # Create a reversible legacy container so the old NOT NULL FK can be restored.
    op.execute(
        """
        INSERT INTO projects (
            project_id, owner_id, title, description, status,
            thumbnail_artifact_id, last_opened_at, created_at, updated_at,
            match_id
        )
        SELECT
            'proj_down_' || md5(m.match_id),
            m.owner_id,
            'Downgraded match project',
            'Created by relationship downgrade',
            'DRAFT',
            NULL,
            NULL,
            m.created_at,
            m.updated_at,
            m.match_id
        FROM matches m
        WHERE m.project_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE matches m
        SET project_id = 'proj_down_' || md5(m.match_id)
        WHERE m.project_id IS NULL
        """
    )

    op.alter_column("matches", "project_id", nullable=False)
    op.alter_column("clip_plans", "match_id", nullable=False)
    op.create_foreign_key(
        "matches_project_id_fkey",
        "matches",
        "projects",
        ["project_id"],
        ["project_id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_matches_project_id", "matches", ["project_id"])
    op.create_foreign_key(
        "clip_plans_match_id_fkey",
        "clip_plans",
        "matches",
        ["match_id"],
        ["match_id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_clip_plans_match_id", "clip_plans", ["match_id"])

    op.drop_index(
        "uq_media_assets_match_singleton_video",
        table_name="media_assets",
    )

    op.drop_constraint(
        "fk_artifacts_project_id_projects",
        "artifacts",
        type_="foreignkey",
    )
    op.drop_index("ix_artifacts_project_id", table_name="artifacts")
    op.drop_column("artifacts", "project_id")

    op.drop_constraint(
        "fk_clip_plans_project_id_projects",
        "clip_plans",
        type_="foreignkey",
    )
    op.drop_index("ix_clip_plans_project_id", table_name="clip_plans")
    op.drop_column("clip_plans", "project_id")

    op.drop_constraint(
        "fk_projects_match_id_matches",
        "projects",
        type_="foreignkey",
    )
    op.drop_index("ix_projects_match_id", table_name="projects")
    op.drop_column("projects", "match_id")

    op.drop_constraint(
        "fk_matches_owner_id_users",
        "matches",
        type_="foreignkey",
    )
    op.drop_index("ix_matches_owner_id", table_name="matches")
    op.drop_column("matches", "owner_id")

    op.drop_table("project_match_legacy_map")
