QUEUED = "QUEUED"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELED = "CANCELED"

SUPPORTED_JOB_STATUSES = {
    QUEUED,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELED,
}


def normalize_job_status(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in SUPPORTED_JOB_STATUSES:
        raise ValueError(f"Unsupported job status: {value}")
    return normalized