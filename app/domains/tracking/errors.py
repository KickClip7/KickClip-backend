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


class TrackingInputError(TrackingValidationError):
    code = "TRACKING_INPUT_INVALID"

    def __init__(
        self,
        message: str | None = None,
        *,
        diagnostics: dict | None = None,
    ):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class TrackingClipTooShortError(TrackingInputError):
    code = "TRACKING_CLIP_TOO_SHORT"
    public_message = "Tracking clip must be at least 10 seconds long."


class TrackingClipTooLongError(TrackingInputError):
    code = "TRACKING_CLIP_TOO_LONG"
    public_message = "Tracking clip must not exceed 30 seconds."


class TrackingClipDecodeFailedError(TrackingInputError):
    code = "TRACKING_CLIP_DECODE_FAILED"
    public_message = "Tracking clip frame 0 could not be decoded."


class TrackingClipMetadataInvalidError(TrackingInputError):
    code = "TRACKING_CLIP_METADATA_INVALID"
    public_message = "Tracking clip metadata is invalid."


class TrackingNoValidInitializationAnchorError(TrackingInputError):
    code = "TRACKING_NO_VALID_INITIALIZATION_ANCHOR"
    public_message = (
        "No stable player anchor leaves enough video for tracking. "
        "Select the player from an earlier frame."
    )


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
