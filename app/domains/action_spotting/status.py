from __future__ import annotations

from typing import Any


def action_spotting_workflow_status(
    *,
    status: str,
    options: dict[str, Any] | None = None,
) -> str:
    options = options or {}
    explicit = options.get("action_spotting_state")
    if isinstance(explicit, str) and explicit.startswith("ACTION_SPOTTING_"):
        return explicit
    normalized = status.upper()
    if normalized == "QUEUED":
        return "ACTION_SPOTTING_QUEUED"
    if normalized == "RUNNING":
        return "ACTION_SPOTTING_RUNNING"
    if normalized == "COMPLETED":
        return "ACTION_SPOTTING_COMPLETED"
    if normalized == "FAILED":
        return "ACTION_SPOTTING_FAILED"
    return normalized


def action_spotting_public_error(
    options: dict[str, Any] | None,
) -> dict[str, Any] | None:
    value = (options or {}).get("action_spotting_error")
    if not isinstance(value, dict):
        return None
    return {
        key: value[key]
        for key in ("code", "message", "retryable", "diagnostics_artifact_id")
        if key in value
    }
