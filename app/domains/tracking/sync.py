from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.state_mapper import TrackingStateMapping
from app.domains.tracking.status import TERMINAL_STATUSES, TrackingBackendStatus


def apply_pipeline_result(
    job: TrackingJob,
    *,
    state: Mapping[str, Any] | None,
    mapping: TrackingStateMapping,
    process_return_code: int | None,
    process_pid: int | None,
    artifacts: TrackingArtifactService,
) -> None:
    now = datetime.now(timezone.utc)
    job.status = mapping.backend_status.value
    job.pipeline_status = mapping.pipeline_status
    job.pipeline_decision = mapping.pipeline_decision
    job.pending_action_type = mapping.pending_action_type
    job.pending_ambiguity_id = mapping.pending_ambiguity_id
    job.current_stage = mapping.current_stage
    job.process_return_code = process_return_code
    job.process_pid = None
    job.runtime_finished_at = now
    job.queued_action = {}

    runtime_metadata = dict(job.runtime_metadata or {})
    runtime_metadata.update(
        {
            "last_process_pid": process_pid,
            "last_process_return_code": process_return_code,
            "last_pipeline_state_updated_at": (
                state.get("updated_at") if state else None
            ),
        }
    )
    job.runtime_metadata = runtime_metadata

    if state is not None:
        job.pipeline_version = _string_or_none(state.get("pipeline_version"))
        video = state.get("video")
        if isinstance(video, Mapping):
            job.video_sha256 = _string_or_none(video.get("sha256"))
        runtime = state.get("runtime")
        if isinstance(runtime, Mapping):
            runtime_metadata["tracking_runtime"] = dict(runtime)
            job.runtime_metadata = runtime_metadata

    job.artifact_index = artifacts.collect(job, state)
    root = artifacts.job_root(job)
    job.timeline_path = _existing_path(root / "target_timeline.json")
    job.summary_path = _existing_path(root / "pipeline_summary.json")
    job.tracking_preview_path = _existing_path(
        root / "full_frame_tracking_preview.mp4"
    )
    job.target_centered_preview_path = _existing_path(
        root / "target_centered_preview.mp4"
    )
    manifest_path = root / "pipeline_manifest.json"
    job.frozen_manifest_present = manifest_path.is_file()

    timeline = _read_optional_json(root / "target_timeline.json")
    if timeline is not None:
        job.schema_version = _string_or_none(timeline.get("schema_version"))
        job.pipeline_version = (
            _string_or_none(timeline.get("pipeline_version"))
            or job.pipeline_version
        )

    if mapping.error_type is not None:
        job.error_type = mapping.error_type
        if mapping.error_type == "USER_REVIEW_REJECTED":
            job.error_message = "User rejected the requested visual review."
        else:
            job.error_message = (
                "Tracking runtime did not produce a valid terminal state."
            )
    else:
        job.error_type = None
        job.error_message = None

    if job.status in TERMINAL_STATUSES:
        job.finished_at = now
    elif job.status == TrackingBackendStatus.RUNNING.value:
        job.finished_at = None


def mark_process_failed(
    job: TrackingJob,
    *,
    error_type: str,
    public_message: str,
    process_pid: int | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    job.status = TrackingBackendStatus.FAILED.value
    job.process_pid = None
    job.runtime_finished_at = now
    job.finished_at = now
    job.error_type = error_type
    job.error_message = public_message
    job.queued_action = {}
    metadata = dict(job.runtime_metadata or {})
    metadata["last_process_pid"] = process_pid
    job.runtime_metadata = metadata


def _existing_path(path: Path) -> str | None:
    return str(path.resolve()) if path.is_file() else None


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _string_or_none(value: object) -> str | None:
    text = str(value or "")
    return text or None
