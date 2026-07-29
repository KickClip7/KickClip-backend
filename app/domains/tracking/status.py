from enum import Enum


class TrackingBackendStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_MEMORY_REVIEW = "WAITING_MEMORY_REVIEW"
    WAITING_CROSS_SHOT_CONFIRMATION = "WAITING_CROSS_SHOT_CONFIRMATION"
    WAITING_SEGMENT_REVIEW = "WAITING_SEGMENT_REVIEW"
    COMPLETED = "COMPLETED"
    COMPLETED_SAFE_BLOCK = "COMPLETED_SAFE_BLOCK"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


WAITING_STATUSES = {
    TrackingBackendStatus.WAITING_MEMORY_REVIEW.value,
    TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value,
    TrackingBackendStatus.WAITING_SEGMENT_REVIEW.value,
}

TERMINAL_STATUSES = {
    TrackingBackendStatus.COMPLETED.value,
    TrackingBackendStatus.COMPLETED_SAFE_BLOCK.value,
    TrackingBackendStatus.FAILED.value,
    TrackingBackendStatus.CANCELLED.value,
}


def tracking_progress(status: str, current_stage: str | None = None) -> int:
    if status == TrackingBackendStatus.QUEUED.value:
        return 0
    if status in {
        TrackingBackendStatus.COMPLETED.value,
        TrackingBackendStatus.COMPLETED_SAFE_BLOCK.value,
    }:
        return 100
    stage = (current_stage or "").upper()
    stage_progress = {
        "INITIALIZATION": 10,
        "MEMORY": 25,
        "STAGE1": 40,
        "SEGMENT": 55,
        "STAGE2B": 65,
        "STAGE2D": 75,
        "STAGE2D1": 80,
        "STAGE2D2": 85,
        "FINALIZE": 95,
    }
    progress = next(
        (
            value
            for key, value in stage_progress.items()
            if key in stage
        ),
        15 if status == TrackingBackendStatus.RUNNING.value else 70,
    )
    return min(progress, 99)


def tracking_retryable(status: str, error_type: str | None) -> bool:
    if status != TrackingBackendStatus.FAILED.value:
        return False
    return error_type not in {
        "VIDEO_FILE_MISSING",
        "TRACKING_VALIDATION_ERROR",
        "TRACKING_INSTALLATION_UNAVAILABLE",
        "TRACKING_INPUT_INVALID",
        "TRACKING_CLIP_TOO_SHORT",
        "TRACKING_CLIP_TOO_LONG",
        "TRACKING_CLIP_DECODE_FAILED",
        "TRACKING_CLIP_METADATA_INVALID",
        "TRACKING_NO_VALID_INITIALIZATION_ANCHOR",
    }


def tracking_outcome(status: str, error_type: str | None) -> str:
    if status in WAITING_STATUSES:
        return "TRACKING_NEEDS_CONFIRMATION"
    if status == TrackingBackendStatus.COMPLETED.value:
        return "TRACKING_COMPLETED"
    if status == TrackingBackendStatus.COMPLETED_SAFE_BLOCK.value:
        return "TRACKING_COMPLETED_SAFE_BLOCK"
    if status == TrackingBackendStatus.FAILED.value:
        if (error_type or "").startswith("TRACKING_CLIP_") or error_type in {
            "TRACKING_INPUT_INVALID",
            "TRACKING_NO_VALID_INITIALIZATION_ANCHOR",
        }:
            return "TRACKING_INPUT_INVALID"
        return "TRACKING_RUNTIME_FAILED"
    return "TRACKING_RUNNING"
