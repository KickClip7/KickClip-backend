from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.domains.timeline.schema import TimelineEventRead


MOCK_SEED_METADATA_KEY = "kickclip_mock_seed"


def build_mock_timeline_rows(
    payload: dict[str, Any],
    *,
    target_match_id: str,
    source_path: Path,
) -> list[dict[str, Any]]:
    """Convert a timeline fixture into repeatable DB rows for one dev match."""
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("목업 timeline JSON의 events 필드는 배열이어야 합니다.")

    rows: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()

    for raw_event in raw_events:
        event = TimelineEventRead.model_validate(raw_event)
        source_event_id = event.timeline_event_id
        if source_event_id in seen_source_ids:
            raise ValueError(
                f"목업 timeline JSON에 중복 event ID가 있습니다: {source_event_id}"
            )
        seen_source_ids.add(source_event_id)

        digest = hashlib.sha256(
            f"{target_match_id}:{source_event_id}".encode("utf-8")
        ).hexdigest()[:24]
        metadata = {
            **event.metadata,
            MOCK_SEED_METADATA_KEY: {
                "source_event_id": source_event_id,
                "source_match_id": event.match_id,
                "source_path": str(source_path),
            },
        }

        rows.append(
            {
                "timeline_event_id": f"mock_evt_{digest}",
                "match_id": target_match_id,
                # Fixture IDs do not point at real artifact/job DB rows.
                "source_artifact_id": None,
                "source_job_id": None,
                "event_type": event.event_type,
                "label": event.label,
                "half": event.half,
                "timestamp_sec": event.timestamp_sec,
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "duration_sec": event.duration_sec,
                "confidence": event.confidence,
                "highlight_score": event.highlight_score,
                "title": event.title,
                "description": event.description,
                "team_name": event.team_name,
                "player_ids": event.player_ids,
                "metadata_": metadata,
            }
        )

    return rows
