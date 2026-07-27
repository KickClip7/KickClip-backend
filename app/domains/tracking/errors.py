from __future__ import annotations


class TrackingError(Exception):
    code = "TRACKING_ERROR"
    http_status = 500
    public_message = "Tracking request could not be completed."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.public_message)


class TrackingUnavailableError(TrackingError):
    code = "TRACKING_RUNTIME_UNAVAILABLE"
    http_status = 503
    public_message = "Target tracking runtime is unavailable."


class TrackingValidationError(TrackingError):
    code = "TRACKING_VALIDATION_ERROR"
    http_status = 422
    public_message = "Tracking request is invalid."


class TrackingConflictError(TrackingError):
    code = "TRACKING_STATE_CONFLICT"
    http_status = 409
    public_message = "Tracking job is not waiting for this action."


class TrackingArtifactNotFoundError(TrackingError):
    code = "TRACKING_ARTIFACT_NOT_FOUND"
    http_status = 404
    public_message = "Tracking artifact was not found."


class TrackingContractError(TrackingError):
    code = "TRACKING_OUTPUT_CONTRACT_ERROR"
    http_status = 500
    public_message = "Tracking runtime produced an invalid output contract."


class TrackingProcessTimeoutError(TrackingError):
    code = "TRACKING_PROCESS_TIMEOUT"
    http_status = 500
    public_message = "Tracking runtime exceeded its configured timeout."

