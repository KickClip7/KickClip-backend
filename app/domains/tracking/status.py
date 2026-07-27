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
