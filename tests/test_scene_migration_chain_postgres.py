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
class SceneMigrationChainPostgresTest(unittest.TestCase):
    def test_empty_database_upgrades_through_scene_ai_head(self) -> None:
        engine = sa.create_engine(get_settings().DATABASE_URL)
        schema = f"kickclip_scene_chain_{uuid4().hex}"

        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
                connection.execute(
                    sa.text(f'SET LOCAL search_path TO "{schema}", public')
                )
                context = MigrationContext.configure(connection)
                scripts = ScriptDirectory.from_config(
                    Config("alembic.ini")
                )
                revisions = list(
                    scripts.walk_revisions(
                        base="base",
                        head="20260730_0016",
                    )
                )
                with Operations.context(context):
                    for revision in reversed(revisions):
                        revision.module.upgrade()

                inspector = sa.inspect(connection)
                self.assertIn(
                    "scene_target_selections",
                    inspector.get_table_names(),
                )
                self.assertIn(
                    "event_candidate_rankings",
                    inspector.get_table_names(),
                )
                self.assertIn(
                    "scene_ai_tasks",
                    inspector.get_table_names(),
                )
                selection_columns = {
                    column["name"]
                    for column in inspector.get_columns(
                        "scene_target_selections"
                    )
                }
                self.assertTrue(
                    {
                        "selection_artifact_root",
                        "target_selection_path",
                        "target_selection_sha256",
                        "target_reference_set_path",
                        "target_reference_set_sha256",
                        "earlier_proposals_path",
                        "earlier_proposals_sha256",
                        "earlier_decision_path",
                        "earlier_decision_sha256",
                    }.issubset(selection_columns)
                )
            finally:
                transaction.rollback()
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
