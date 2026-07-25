from __future__ import annotations

from typing import Protocol, TypeVar

from app.domains.timeline.mock_seed import MOCK_SEED_METADATA_KEY


class AgentDevSettings(Protocol):
    USE_MOCK_DATA: bool
    AGENT_DEV_MATCH_ID: str


class TimelineEventLike(Protocol):
    metadata_: dict


TimelineEventT = TypeVar("TimelineEventT", bound=TimelineEventLike)


def resolve_agent_match_id(match_id: str, settings: AgentDevSettings) -> str:
    """Use one configured match while developing against mock timeline data."""
    fixed_match_id = settings.AGENT_DEV_MATCH_ID.strip()
    if settings.USE_MOCK_DATA and fixed_match_id:
        return fixed_match_id
    return match_id


def select_dev_timeline_events(
    events: list[TimelineEventT],
    *,
    match_id: str,
    settings: AgentDevSettings,
) -> list[TimelineEventT]:
    """Expose only seeded fixture events for the configured development match."""
    fixed_match_id = settings.AGENT_DEV_MATCH_ID.strip()
    if not settings.USE_MOCK_DATA or not fixed_match_id or match_id != fixed_match_id:
        return events
    return [
        event
        for event in events
        if (event.metadata_ or {}).get(MOCK_SEED_METADATA_KEY)
    ]
