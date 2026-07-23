from pathlib import Path


def _match_root(match_id: str) -> Path:
    safe_match_id = Path(match_id).name
    if safe_match_id != match_id or safe_match_id in {"", ".", ".."}:
        raise ValueError(f"Invalid match_id for workspace path: {match_id!r}")
    return Path("matches") / safe_match_id


def get_match_raw_video_subdir(match_id: str) -> Path:
    return _match_root(match_id) / "raw"


def get_match_preview_video_subdir(match_id: str) -> Path:
    return _match_root(match_id) / "previews"


def get_match_event_candidates_subdir(match_id: str) -> Path:
    return _match_root(match_id) / "action_spotting"


def get_match_player_tracks_subdir(match_id: str) -> Path:
    return _match_root(match_id) / "player_tracks"


def get_match_soccernet_features_subdir(match_id: str) -> Path:
    return _match_root(match_id) / "soccernet_features"


def get_match_soccernet_feature_chunks_subdir(match_id: str) -> Path:
    return get_match_soccernet_features_subdir(match_id) / "chunks"


def get_match_timeline_events_path(match_id: str) -> Path:
    return _match_root(match_id) / "timeline_events.json"
