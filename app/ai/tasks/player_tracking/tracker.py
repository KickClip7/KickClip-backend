from __future__ import annotations

from app.ai.tasks.player_tracking.types import (
    DetectedPlayerSeed,
    PlayerTrackingInput,
    PlayerTrackSeed,
)


class DummyPlayerTracker:
    """Dummy player tracker.

    실제 bbox/track point는 생성하지 않는다.
    TimelineEvent와 연결된 주변 구간을 선수별 등장 구간으로 요약한다.
    """

    name = "dummy_player_tracker_v2"

    def build_tracks(
        self,
        task_input: PlayerTrackingInput,
        detected_players: list[DetectedPlayerSeed],
    ) -> list[PlayerTrackSeed]:
        tracks: list[PlayerTrackSeed] = []

        for player in detected_players:
            linked_events = [
                event
                for event in task_input.events
                if player.raw_player_id in (event.player_ids or [])
            ]

            if not linked_events:
                continue

            for event in linked_events:
                start_sec = max(float(event.start_sec) - 2.0, 0.0)
                end_sec = float(event.end_sec) + 2.0

                if task_input.match_duration_sec is not None:
                    end_sec = min(end_sec, task_input.match_duration_sec)

                duration_sec = max(end_sec - start_sec, 0.1)

                tracks.append(
                    PlayerTrackSeed(
                        raw_player_id=player.raw_player_id,
                        start_sec=round(start_sec, 3),
                        end_sec=round(end_sec, 3),
                        duration_sec=round(duration_sec, 3),
                        linked_event_ids=[event.timeline_event_id],
                        summary=f"{player.display_name} 관련 {event.label} 장면",
                        confidence=player.confidence,
                        metadata={
                            "source": self.name,
                            "linked_event_label": event.label,
                        },
                    )
                )

        return tracks


class UnimplementedRealPlayerTracker:
    """Placeholder for model-backed tracker."""

    name = "real_player_tracker_placeholder"

    def build_tracks(
        self,
        task_input: PlayerTrackingInput,
        detected_players: list[DetectedPlayerSeed],
    ) -> list[PlayerTrackSeed]:
        raise NotImplementedError(
            "Real player tracker is not implemented yet. "
            "Set player_tracking.mode=dummy until a tracker model is connected."
        )


def create_player_tracker(mode: str = "dummy"):
    normalized = (mode or "dummy").lower()
    if normalized == "dummy":
        return DummyPlayerTracker()
    if normalized in {"real", "model"}:
        return UnimplementedRealPlayerTracker()
    raise ValueError(f"Unsupported player tracker mode: {mode}")
