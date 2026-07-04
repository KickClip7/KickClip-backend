FULL_MATCH_ANALYSIS = "FULL_MATCH_ANALYSIS"
HIGHLIGHT_SPOTTING = "HIGHLIGHT_SPOTTING"
PLAYER_TRACKING = "PLAYER_TRACKING"
BALL_TRACKING = "BALL_TRACKING"
TIMELINE_FUSION = "TIMELINE_FUSION"


SUPPORTED_JOB_TYPES = {
    FULL_MATCH_ANALYSIS,
    HIGHLIGHT_SPOTTING,
    PLAYER_TRACKING,
    BALL_TRACKING,
    TIMELINE_FUSION,
}


def normalize_job_type(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in SUPPORTED_JOB_TYPES:
        raise ValueError(f"Unsupported job_type: {value}")
    return normalized