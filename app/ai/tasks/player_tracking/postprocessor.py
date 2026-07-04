from __future__ import annotations

from app.ai.tasks.player_tracking.types import DetectedPlayerSeed, PlayerTrackSeed


PLAYER_TRACKS_SCHEMA_VERSION = "player_tracks.v1"


class PlayerTrackingPostprocessor:
    """Normalize detected players/tracks to DB-friendly dictionaries and artifact schema."""

    name = "player_tracking_postprocessor_v1"

    def build_canonical_player_id(
        self,
        match_id: str,
        raw_player_id: str,
        number: int | None,
    ) -> str:
        suffix = match_id.replace("match_", "")[-6:]

        if number is not None:
            return f"player_{suffix}_{number:03d}"

        cleaned = raw_player_id.replace(" ", "_").replace("-", "_")
        return f"player_{suffix}_{cleaned}"

    def normalize_players(
        self,
        match_id: str,
        detected_players: list[DetectedPlayerSeed],
    ) -> tuple[list[dict], dict[str, str]]:
        players: list[dict] = []
        id_map: dict[str, str] = {}
        seen_canonical_ids: set[str] = set()

        for seed in detected_players:
            canonical_id = self.build_canonical_player_id(
                match_id=match_id,
                raw_player_id=seed.raw_player_id,
                number=seed.number,
            )
            id_map[seed.raw_player_id] = canonical_id

            if canonical_id in seen_canonical_ids:
                continue
            seen_canonical_ids.add(canonical_id)

            players.append(
                {
                    "player_id": canonical_id,
                    "match_id": match_id,
                    "display_name": seed.display_name,
                    "number": seed.number,
                    "team_name": seed.team_hint,
                    "role": seed.role,
                    "identity_status": "UNKNOWN",
                    "profile_source": "player_tracking_dummy" if seed.metadata.get("source", "").startswith("dummy") else "player_tracking_model",
                    "metadata": {
                        "raw_player_id": seed.raw_player_id,
                        "detector_confidence": seed.confidence,
                        "bbox": seed.bbox,
                        "frame_index": seed.frame_index,
                        **dict(seed.metadata or {}),
                    },
                }
            )

        return players, id_map

    def normalize_tracks(
        self,
        match_id: str,
        source_job_id: str,
        track_artifact_id: str | None,
        track_seeds: list[PlayerTrackSeed],
        id_map: dict[str, str],
    ) -> list[dict]:
        rows: list[dict] = []

        for seed in track_seeds:
            canonical_id = id_map.get(seed.raw_player_id)
            if canonical_id is None:
                continue

            rows.append(
                {
                    "match_id": match_id,
                    "player_id": canonical_id,
                    "source_job_id": source_job_id,
                    "start_sec": seed.start_sec,
                    "end_sec": seed.end_sec,
                    "duration_sec": seed.duration_sec,
                    "track_artifact_id": track_artifact_id,
                    "summary": seed.summary,
                    "linked_event_ids": seed.linked_event_ids,
                    "metadata": {
                        "raw_player_id": seed.raw_player_id,
                        "track_confidence": seed.confidence,
                        "sample_points": seed.sample_points,
                        **dict(seed.metadata or {}),
                    },
                }
            )

        return rows

    def build_artifact_payload(
        self,
        *,
        task_type: str,
        mode: str,
        analysis_job_id: str,
        match_id: str,
        source_video_asset_id: str | None,
        players: list[dict],
        track_seeds: list[PlayerTrackSeed],
        raw_to_canonical_id: dict[str, str],
        diagnostics: dict,
        created_at: str,
    ) -> dict:
        return {
            "schema_version": PLAYER_TRACKS_SCHEMA_VERSION,
            "task_type": task_type,
            "mode": mode,
            "analysis_job_id": analysis_job_id,
            "match_id": match_id,
            "source_video_asset_id": source_video_asset_id,
            "num_players": len(players),
            "num_tracks": len(track_seeds),
            "players": players,
            "tracks": [self.serialize_track_seed(seed) for seed in track_seeds],
            "raw_to_canonical_id": raw_to_canonical_id,
            "diagnostics": diagnostics,
            "created_at": created_at,
        }

    @staticmethod
    def serialize_track_seed(seed: PlayerTrackSeed) -> dict:
        return {
            "raw_player_id": seed.raw_player_id,
            "start_sec": seed.start_sec,
            "end_sec": seed.end_sec,
            "duration_sec": seed.duration_sec,
            "linked_event_ids": seed.linked_event_ids,
            "summary": seed.summary,
            "confidence": seed.confidence,
            "sample_points": seed.sample_points,
            "metadata": seed.metadata,
        }

    @staticmethod
    def track_seed_from_dict(item: dict) -> PlayerTrackSeed:
        return PlayerTrackSeed(
            raw_player_id=item["raw_player_id"],
            start_sec=item["start_sec"],
            end_sec=item["end_sec"],
            duration_sec=item["duration_sec"],
            linked_event_ids=item.get("linked_event_ids") or [],
            summary=item.get("summary") or "",
            confidence=item.get("confidence"),
            sample_points=item.get("sample_points") or [],
            metadata=item.get("metadata") or {},
        )


# Backward-compatible alias for imports written in 8회차.
DummyPlayerTrackingPostprocessor = PlayerTrackingPostprocessor
