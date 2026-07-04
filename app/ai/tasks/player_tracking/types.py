from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.domains.timeline.model import TimelineEvent


@dataclass(frozen=True)
class PlayerTrackingInput:
    """Player tracking task input contract.

    Real detector/tracker implementations should depend on this object rather than
    directly touching DB models. This keeps PlayerTrackingTask stable when the
    dummy implementation is replaced by a model-backed implementation.
    """

    match_id: str
    analysis_job_id: str
    source_video_asset_id: str | None
    source_video_path: Path | None
    match_duration_sec: float | None
    home_team: str | None
    away_team: str | None
    events: list[TimelineEvent]
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedPlayerSeed:
    raw_player_id: str
    number: int | None
    display_name: str
    team_hint: str | None
    role: str | None
    confidence: float | None = None
    bbox: list[float] | None = None
    frame_index: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerTrackSeed:
    raw_player_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    linked_event_ids: list[str]
    summary: str
    confidence: float | None = None
    sample_points: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerTrackingRawResult:
    mode: str
    detected_players: list[DetectedPlayerSeed]
    tracks: list[PlayerTrackSeed]
    diagnostics: dict = field(default_factory=dict)


class PlayerDetector(Protocol):
    name: str

    def detect(self, task_input: PlayerTrackingInput) -> list[DetectedPlayerSeed]:
        ...


class PlayerTracker(Protocol):
    name: str

    def build_tracks(
        self,
        task_input: PlayerTrackingInput,
        detected_players: list[DetectedPlayerSeed],
    ) -> list[PlayerTrackSeed]:
        ...
