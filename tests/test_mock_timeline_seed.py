import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.domains.timeline.dev_context import (
    resolve_agent_match_id,
    select_dev_timeline_events,
)
from app.domains.timeline.mock_seed import (
    MOCK_SEED_METADATA_KEY,
    build_mock_timeline_rows,
)


class MockTimelineSeedTest(unittest.TestCase):
    def test_build_rows_rebinds_fixture_to_target_match(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "events": [
                {
                    "timeline_event_id": "E0001",
                    "match_id": "korjpn_2026",
                    "source_artifact_id": "video_korjpn_2026",
                    "source_job_id": "job_actionspotting_v1",
                    "event_type": "action_spotting",
                    "label": "Goal",
                    "half": 1,
                    "timestamp_sec": 120.0,
                    "start_sec": 113.0,
                    "end_sec": 133.0,
                    "duration_sec": 20.0,
                    "confidence": 0.9,
                    "highlight_score": 0.9,
                    "title": None,
                    "description": None,
                    "team_name": None,
                    "player_ids": [],
                    "metadata": {"raw_time_sec": 120.0},
                    "created_at": now,
                    "updated_at": now,
                }
            ]
        }

        rows = build_mock_timeline_rows(
            payload,
            target_match_id="match_dev",
            source_path=Path("fixture.json"),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_id"], "match_dev")
        self.assertIsNone(rows[0]["source_artifact_id"])
        self.assertIsNone(rows[0]["source_job_id"])
        self.assertEqual(
            rows[0]["metadata_"][MOCK_SEED_METADATA_KEY]["source_event_id"],
            "E0001",
        )

    def test_generated_ids_are_stable_per_target_match(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        event = {
            "timeline_event_id": "E0001",
            "match_id": "korjpn_2026",
            "source_artifact_id": None,
            "source_job_id": None,
            "event_type": "action_spotting",
            "label": "Shot",
            "half": 1,
            "timestamp_sec": 10.0,
            "start_sec": 5.0,
            "end_sec": 14.0,
            "duration_sec": 9.0,
            "confidence": 0.8,
            "highlight_score": 0.36,
            "title": None,
            "description": None,
            "team_name": None,
            "player_ids": [],
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        }

        first = build_mock_timeline_rows(
            {"events": [event]},
            target_match_id="match_dev",
            source_path=Path("fixture.json"),
        )
        second = build_mock_timeline_rows(
            {"events": [event]},
            target_match_id="match_dev",
            source_path=Path("fixture.json"),
        )

        self.assertEqual(
            first[0]["timeline_event_id"],
            second[0]["timeline_event_id"],
        )

    def test_agent_dev_match_overrides_request_only_in_mock_mode(self) -> None:
        mock_settings = SimpleNamespace(
            USE_MOCK_DATA=True,
            AGENT_DEV_MATCH_ID="match_fixed",
        )
        real_settings = SimpleNamespace(
            USE_MOCK_DATA=False,
            AGENT_DEV_MATCH_ID="match_fixed",
        )

        self.assertEqual(
            resolve_agent_match_id("match_requested", mock_settings),
            "match_fixed",
        )
        self.assertEqual(
            resolve_agent_match_id("match_requested", real_settings),
            "match_requested",
        )

    def test_fixed_dev_match_exposes_only_seeded_events(self) -> None:
        settings = SimpleNamespace(
            USE_MOCK_DATA=True,
            AGENT_DEV_MATCH_ID="match_fixed",
        )
        seeded = SimpleNamespace(
            metadata_={MOCK_SEED_METADATA_KEY: {"source_event_id": "E0001"}}
        )
        real = SimpleNamespace(metadata_={})

        selected = select_dev_timeline_events(
            [real, seeded],
            match_id="match_fixed",
            settings=settings,
        )

        self.assertEqual(selected, [seeded])


if __name__ == "__main__":
    unittest.main()
