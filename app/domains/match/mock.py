from __future__ import annotations

from typing import Any, Protocol


MOCK_MATCH_METADATA_KEY = "kickclip_mock_match"
MOCK_MATCH_SCHEMA_VERSION = 1


class MatchLike(Protocol):
    metadata_: dict[str, Any]


def build_mock_match_metadata(
    *,
    source_path: str,
    event_count: int,
) -> dict[str, Any]:
    """Return the metadata marker used to identify an explicitly seeded mock match.

    The marker is stored on ``Match.metadata_`` so authentication can grant shared
    development access only to matches that were intentionally created as fixtures.
    Real matches never receive this marker.
    """

    return {
        MOCK_MATCH_METADATA_KEY: {
            "schema_version": MOCK_MATCH_SCHEMA_VERSION,
            "source_path": source_path,
            "event_count": event_count,
        }
    }


def is_mock_match(match: MatchLike) -> bool:
    """Return whether a match was created by the mock fixture seeder."""

    marker = (match.metadata_ or {}).get(MOCK_MATCH_METADATA_KEY)
    return isinstance(marker, dict) and bool(marker)
