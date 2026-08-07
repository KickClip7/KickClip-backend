from __future__ import annotations

from typing import Any


ACTION_SPOTTING_RUNTIME_UNAVAILABLE = "ACTION_SPOTTING_RUNTIME_UNAVAILABLE"
ACTION_SPOTTING_FAILED = "ACTION_SPOTTING_FAILED"
ACTION_SPOTTING_CHECKPOINT_MISMATCH = "ACTION_SPOTTING_CHECKPOINT_MISMATCH"
ACTION_SPOTTING_CLASS_ORDER_MISMATCH = "ACTION_SPOTTING_CLASS_ORDER_MISMATCH"
ACTION_SPOTTING_FEATURE_MISMATCH = "ACTION_SPOTTING_FEATURE_MISMATCH"
FEATURE_EXTRACTION_FAILED = "FEATURE_EXTRACTION_FAILED"
ACTION_SPOTTING_INFERENCE_POLICY_INCOMPLETE = (
    "ACTION_SPOTTING_INFERENCE_POLICY_INCOMPLETE"
)


class ActionSpottingError(RuntimeError):
    """Expected Action Spotting failure with a safe public error contract."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.diagnostics = diagnostics or {}
        self.retryable = retryable

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.public_message,
            "retryable": self.retryable,
        }

    def to_internal_dict(self) -> dict[str, Any]:
        return {
            **self.to_public_dict(),
            "diagnostics": self.diagnostics,
        }


def runtime_unavailable(message: str, **diagnostics: Any) -> ActionSpottingError:
    return ActionSpottingError(
        ACTION_SPOTTING_RUNTIME_UNAVAILABLE,
        message,
        diagnostics=diagnostics,
        retryable=True,
    )


def inference_failed(message: str, **diagnostics: Any) -> ActionSpottingError:
    return ActionSpottingError(
        ACTION_SPOTTING_FAILED,
        message,
        diagnostics=diagnostics,
        retryable=True,
    )


def checkpoint_mismatch(message: str, **diagnostics: Any) -> ActionSpottingError:
    return ActionSpottingError(
        ACTION_SPOTTING_CHECKPOINT_MISMATCH,
        message,
        diagnostics=diagnostics,
    )


def class_order_mismatch(message: str, **diagnostics: Any) -> ActionSpottingError:
    return ActionSpottingError(
        ACTION_SPOTTING_CLASS_ORDER_MISMATCH,
        message,
        diagnostics=diagnostics,
    )


def feature_mismatch(message: str, **diagnostics: Any) -> ActionSpottingError:
    return ActionSpottingError(
        ACTION_SPOTTING_FEATURE_MISMATCH,
        message,
        diagnostics=diagnostics,
        retryable=True,
    )


def feature_extraction_failed(
    message: str, **diagnostics: Any
) -> ActionSpottingError:
    return ActionSpottingError(
        FEATURE_EXTRACTION_FAILED,
        message,
        diagnostics=diagnostics,
        retryable=True,
    )


def inference_policy_incomplete(
    message: str, **diagnostics: Any
) -> ActionSpottingError:
    return ActionSpottingError(
        ACTION_SPOTTING_INFERENCE_POLICY_INCOMPLETE,
        message,
        diagnostics=diagnostics,
    )
