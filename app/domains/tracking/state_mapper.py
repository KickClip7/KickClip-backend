from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domains.tracking.errors import TrackingContractError
from app.domains.tracking.status import TrackingBackendStatus


@dataclass(frozen=True)
class TrackingStateMapping:
    backend_status: TrackingBackendStatus
    pipeline_status: str | None
    pipeline_decision: str | None
    pending_action_type: str | None
    pending_ambiguity_id: str | None
    current_stage: str | None
    error_type: str | None = None


def read_pipeline_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingContractError("pipeline_state.json is unreadable.") from exc
    if not isinstance(value, dict):
        raise TrackingContractError("pipeline_state.json must contain an object.")
    return value


def map_pipeline_state(
    state: Mapping[str, Any] | None,
    *,
    process_return_code: int | None,
    process_ended: bool = True,
) -> TrackingStateMapping:
    """Map JSON state to backend state; exit code is only a fallback signal."""

    if state is None:
        if process_ended:
            return TrackingStateMapping(
                backend_status=TrackingBackendStatus.FAILED,
                pipeline_status=None,
                pipeline_decision=None,
                pending_action_type=None,
                pending_ambiguity_id=None,
                current_stage=None,
                error_type="PIPELINE_STATE_MISSING",
            )
        return TrackingStateMapping(
            backend_status=TrackingBackendStatus.RUNNING,
            pipeline_status=None,
            pipeline_decision=None,
            pending_action_type=None,
            pending_ambiguity_id=None,
            current_stage=None,
        )

    pipeline_status = str(state.get("status") or "") or None
    decision = str(state.get("decision") or "") or None
    pending = state.get("pending_action")
    pending_mapping = pending if isinstance(pending, Mapping) else {}
    pending_type = str(pending_mapping.get("type") or "") or None
    if (
        pending_type is None
        and pending_mapping.get("backend_state")
        == "WAITING_CROSS_SHOT_CONFIRMATION"
    ):
        pending_type = "CROSS_SHOT_CONFIRMATION"
    ambiguity_id = str(
        pending_mapping.get("ambiguity_id")
        or pending_mapping.get("confirmation_id")
        or ""
    ) or None
    review_stage = str(pending_mapping.get("review_stage") or "") or None

    if pipeline_status in {
        "NEEDS_CONFIRMATION",
        "WAITING_CROSS_SHOT_CONFIRMATION",
    }:
        if pipeline_status == "WAITING_CROSS_SHOT_CONFIRMATION":
            pending_type = pending_type or "CROSS_SHOT_CONFIRMATION"
        waiting_status = {
            "MEMORY_REVIEW": TrackingBackendStatus.WAITING_MEMORY_REVIEW,
            "CROSS_SHOT_CONFIRMATION": (
                TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION
            ),
            "SEGMENT_VISUAL_REVIEW": TrackingBackendStatus.WAITING_SEGMENT_REVIEW,
            "PHASE1_INTERNAL_REVIEW": TrackingBackendStatus.WAITING_SEGMENT_REVIEW,
        }.get(pending_type or "")
        if waiting_status is None:
            return TrackingStateMapping(
                backend_status=TrackingBackendStatus.FAILED,
                pipeline_status=pipeline_status,
                pipeline_decision=decision,
                pending_action_type=pending_type,
                pending_ambiguity_id=ambiguity_id,
                current_stage=review_stage,
                error_type="UNKNOWN_PENDING_ACTION",
            )
        return TrackingStateMapping(
            backend_status=waiting_status,
            pipeline_status=pipeline_status,
            pipeline_decision=decision,
            pending_action_type=pending_type,
            pending_ambiguity_id=ambiguity_id,
            current_stage=review_stage or pending_type,
        )

    if pipeline_status == "COMPLETE":
        return TrackingStateMapping(
            backend_status=TrackingBackendStatus.COMPLETED,
            pipeline_status=pipeline_status,
            pipeline_decision=decision,
            pending_action_type=None,
            pending_ambiguity_id=None,
            current_stage=None,
        )

    if pipeline_status in {"COMPLETE_WITH_SAFE_BLOCK", "BLOCKED"}:
        return TrackingStateMapping(
            backend_status=TrackingBackendStatus.COMPLETED_SAFE_BLOCK,
            pipeline_status=pipeline_status,
            pipeline_decision=decision,
            pending_action_type=None,
            pending_ambiguity_id=None,
            current_stage=None,
            error_type=(
                "USER_REVIEW_REJECTED"
                if pipeline_status == "BLOCKED"
                else None
            ),
        )

    if pipeline_status in {"FAILED", "FATAL", "ERROR"}:
        return TrackingStateMapping(
            backend_status=TrackingBackendStatus.FAILED,
            pipeline_status=pipeline_status,
            pipeline_decision=decision,
            pending_action_type=pending_type,
            pending_ambiguity_id=ambiguity_id,
            current_stage=review_stage,
            error_type="PIPELINE_FATAL_ERROR",
        )

    if not process_ended and pipeline_status == "RUNNING":
        return TrackingStateMapping(
            backend_status=TrackingBackendStatus.RUNNING,
            pipeline_status=pipeline_status,
            pipeline_decision=decision,
            pending_action_type=pending_type,
            pending_ambiguity_id=ambiguity_id,
            current_stage=review_stage,
        )

    return TrackingStateMapping(
        backend_status=TrackingBackendStatus.FAILED,
        pipeline_status=pipeline_status,
        pipeline_decision=decision,
        pending_action_type=pending_type,
        pending_ambiguity_id=ambiguity_id,
        current_stage=review_stage,
        error_type="PIPELINE_ENDED_WITHOUT_TERMINAL_STATE",
    )
