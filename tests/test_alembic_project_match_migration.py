import os
import unittest
from uuid import uuid4

import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from app.core.config import get_settings


@unittest.skipUnless(
    os.getenv("KICKCLIP_RUN_POSTGRES_MIGRATION_TEST") == "1",
    "set KICKCLIP_RUN_POSTGRES_MIGRATION_TEST=1 to run PostgreSQL migration test",
)
class ProjectMatchAlembicMigrationTest(unittest.TestCase):
    def test_upgrade_and_downgrade_with_legacy_data(self) -> None:
        engine = sa.create_engine(get_settings().DATABASE_URL)
        schema = f"kickclip_migration_test_{uuid4().hex}"

        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(
                    sa.text(f'CREATE SCHEMA "{schema}"')
                )
                connection.execute(
                    sa.text(f'SET LOCAL search_path TO "{schema}", public')
                )
                context = MigrationContext.configure(connection)
                config = Config("alembic.ini")
                scripts = ScriptDirectory.from_config(config)
                revisions = list(
                    scripts.walk_revisions(
                        base="base",
                        head="20260722_0008",
                    )
                )
                target = scripts.get_revision("20260725_0009")
                if target is None:
                    self.fail("20260725_0009 migration not found")

                with Operations.context(context):
                    for revision in reversed(revisions):
                        revision.module.upgrade()
                    self._insert_legacy_fixture(connection)
                    target.module.upgrade()

                    inspector = sa.inspect(connection)
                    self.assertIn(
                        "owner_id",
                        {column["name"] for column in inspector.get_columns("matches")},
                    )
                    self.assertNotIn(
                        "project_id",
                        {column["name"] for column in inspector.get_columns("matches")},
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT COUNT(*) FROM matches "
                                "WHERE owner_id IS NULL"
                            )
                        ),
                        0,
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT COUNT(*) FROM projects "
                                "WHERE match_id IS NULL"
                            )
                        ),
                        0,
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT COUNT(DISTINCT project_id) "
                                "FROM clip_plans"
                            )
                        ),
                        2,
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT COUNT(*) FROM media_assets "
                                "WHERE asset_type = 'RAW_VIDEO'"
                            )
                        ),
                        1,
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT asset_id FROM media_assets "
                                "WHERE asset_type = 'RAW_VIDEO'"
                            )
                        ),
                        "asset_raw_2",
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT asset_id FROM media_assets "
                                "WHERE asset_type LIKE "
                                "'RAW_VIDEO_LEGACY_DUPLICATE_%'"
                            )
                        ),
                        "asset_raw_1",
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT COUNT(*) FROM media_assets "
                                "WHERE asset_type LIKE "
                                "'RAW_VIDEO_LEGACY_DUPLICATE_%'"
                            )
                        ),
                        1,
                    )
                    connection.execute(
                        sa.text(
                            """
                            INSERT INTO projects (
                                project_id, match_id, owner_id, title, status,
                                created_at, updated_at
                            ) VALUES (
                                'proj_delete_fixture', 'match_first',
                                'usr_fixture', 'Delete fixture', 'DRAFT',
                                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                            )
                            """
                        )
                    )
                    connection.execute(
                        sa.text(
                            """
                            INSERT INTO artifacts (
                                artifact_id, match_id, project_id,
                                artifact_type, file_path, metadata,
                                created_at, updated_at
                            ) VALUES (
                                'artifact_delete_fixture', 'match_first',
                                'proj_delete_fixture', 'RENDERED_VIDEO',
                                'delete-fixture.mp4', '{}'::json,
                                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                            )
                            """
                        )
                    )
                    connection.execute(
                        sa.text(
                            "DELETE FROM projects "
                            "WHERE project_id = 'proj_delete_fixture'"
                        )
                    )
                    self.assertEqual(
                        connection.scalar(
                            sa.text(
                                "SELECT COUNT(*) FROM artifacts "
                                "WHERE artifact_id = 'artifact_delete_fixture'"
                            )
                        ),
                        0,
                    )

                    target.module.downgrade()
                    inspector = sa.inspect(connection)
                    self.assertIn(
                        "project_id",
                        {column["name"] for column in inspector.get_columns("matches")},
                    )
                    self.assertNotIn(
                        "owner_id",
                        {column["name"] for column in inspector.get_columns("matches")},
                    )
                    self.assertIn(
                        "match_id",
                        {
                            column["name"]
                            for column in inspector.get_columns("clip_plans")
                        },
                    )
            finally:
                transaction.rollback()
                engine.dispose()

    @staticmethod
    def _insert_legacy_fixture(connection: sa.Connection) -> None:
        connection.execute(
            sa.text(
                """
                INSERT INTO users (
                    user_id, email, password_hash, display_name, role,
                    is_active, developer_mode_enabled, event_weights,
                    created_at, updated_at
                ) VALUES (
                    'usr_fixture', 'fixture@example.com', '!', 'Fixture',
                    'USER', true, false, '{}'::json,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO projects (
                    project_id, owner_id, title, description, status,
                    created_at, updated_at
                ) VALUES
                    ('proj_multi', 'usr_fixture', 'Multi', NULL, 'DRAFT',
                     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                    ('proj_orphan', NULL, 'Orphan', NULL, 'DRAFT',
                     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO matches (
                    match_id, project_id, metadata, created_at, updated_at
                ) VALUES
                    ('match_first', 'proj_multi', '{}'::json,
                     CURRENT_TIMESTAMP - INTERVAL '1 minute', CURRENT_TIMESTAMP),
                    ('match_second', 'proj_multi', '{}'::json,
                     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO clip_plans (
                    clip_plan_id, match_id, mode, created_by, options,
                    created_at, updated_at
                ) VALUES
                    ('clip_first', 'match_first', 'MANUAL', 'usr_fixture',
                     '{}'::json, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                    ('clip_second', 'match_second', 'MANUAL', 'usr_fixture',
                     '{}'::json, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO media_assets (
                    asset_id, match_id, asset_type, file_path,
                    created_at, updated_at
                ) VALUES
                    ('asset_raw_1', 'match_first', 'RAW_VIDEO', 'one.mp4',
                     CURRENT_TIMESTAMP - INTERVAL '1 minute', CURRENT_TIMESTAMP),
                    ('asset_raw_2', 'match_first', 'RAW_VIDEO', 'two.mp4',
                     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
            )
        )


if __name__ == "__main__":
    unittest.main()
