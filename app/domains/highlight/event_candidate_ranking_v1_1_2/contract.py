from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    CandidateSequence,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.contract import (
    ResolvedEventContext,
    _first,
    _load_shots,
    resolve_event_context,
)
from app.domains.project.model import Project


APPROVED_SHOT_REVIEW_STATES = frozenset({"REVIEWED_PASS", "CONFIRMED"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ShotInterval:
    shot_id: str
    start_frame: int
    end_frame: int


def resolve_event_context_v112(
    db: Session,
    *,
    project: Project,
    revision_id: str,
    event_id: str,
    scene_id: str,
) -> ResolvedEventContext:
    resolved = resolve_event_context(
        db,
        project=project,
        revision_id=revision_id,
        event_id=event_id,
        scene_id=scene_id,
    )
    if resolved.scene_start_sec < 0:
        raise ValueError("Scene start must be non-negative.")
    if resolved.scene_end_sec <= resolved.scene_start_sec:
        raise ValueError("Scene bounds are invalid.")
    if not (
        resolved.scene_start_sec
        <= resolved.event_time_sec
        <= resolved.scene_end_sec
    ):
        raise ValueError("Event timestamp is outside the selected scene.")
    return resolved


def validate_scene_video_duration(
    resolved: ResolvedEventContext,
    *,
    video_fps: float,
    video_frame_count: int,
    tolerance_sec: float,
) -> dict[str, float]:
    if video_fps <= 0 or video_frame_count <= 0:
        raise ValueError("Scene video duration contract is unavailable.")
    if tolerance_sec < 0:
        raise ValueError("Scene video duration tolerance must be non-negative.")
    scene_duration = resolved.scene_end_sec - resolved.scene_start_sec
    video_duration = video_frame_count / video_fps
    difference = abs(scene_duration - video_duration)
    if difference > tolerance_sec:
        raise ValueError(
            "Scene bounds and scene video duration exceed allowed tolerance."
        )
    return {
        "scene_duration_sec": scene_duration,
        "video_duration_sec": video_duration,
        "difference_sec": difference,
        "tolerance_sec": tolerance_sec,
    }


def extract_manifest_candidate_sha(manifest: dict[str, Any]) -> str:
    candidates = [
        manifest.get("scene_candidates_sha256"),
        (manifest.get("scene_candidates") or {}).get("sha256")
        if isinstance(manifest.get("scene_candidates"), dict)
        else None,
        (manifest.get("artifacts") or {}).get("scene_candidates_sha256")
        if isinstance(manifest.get("artifacts"), dict)
        else None,
    ]
    files = manifest.get("files")
    if isinstance(files, dict):
        entry = files.get("scene_candidates.json")
        candidates.append(
            entry.get("sha256") if isinstance(entry, dict) else entry
        )
    elif isinstance(files, list):
        for entry in files:
            if (
                isinstance(entry, dict)
                and Path(str(entry.get("path") or "")).name
                == "scene_candidates.json"
            ):
                candidates.append(entry.get("sha256"))
    for value in candidates:
        normalized = str(value or "").strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized):
            return normalized
    raise ValueError(
        "Scene candidate manifest does not declare scene_candidates SHA-256."
    )


def verify_candidate_artifact_immutability(
    *,
    current_sha256: str,
    stored_discovery_sha256: str | None,
    manifest_declared_sha256: str,
) -> None:
    values = {
        str(current_sha256).lower(),
        str(stored_discovery_sha256 or "").lower(),
        str(manifest_declared_sha256).lower(),
    }
    if any(not _SHA256_PATTERN.fullmatch(value) for value in values):
        raise ValueError("Candidate artifact SHA-256 contract is incomplete.")
    if len(values) != 1:
        raise ValueError(
            "Current, discovery-time, and manifest candidate SHA-256 differ."
        )


def audit_shot_contract_v112(
    path: Path,
    *,
    candidates: tuple[CandidateSequence, ...],
    frame_count: int,
) -> tuple[dict[str, Any], tuple[ShotInterval, ...]]:
    rows = _load_shots(path)
    shots: dict[str, ShotInterval] = {}
    unapproved_shot_ids: list[str] = []
    for index, row in enumerate(rows):
        shot_id = str(_first(row, "shot_id", "id") or f"shot_{index}")
        if shot_id in shots:
            raise ValueError("Shot-boundary artifact contains duplicate shot IDs.")
        start = _first(row, "start_frame", "first_frame", "frame_start")
        end = _first(row, "end_frame", "last_frame", "frame_end")
        if start is None or end is None:
            raise ValueError("Shot-boundary row has no frame interval.")
        start, end = int(start), int(end)
        if start < 0 or end < start or end >= frame_count:
            raise ValueError("Shot-boundary interval is outside scene video.")
        review_state = str(
            _first(row, "review_status", "status", "boundary_status") or ""
        ).strip().upper()
        if review_state not in APPROVED_SHOT_REVIEW_STATES:
            unapproved_shot_ids.append(shot_id)
        shots[shot_id] = ShotInterval(shot_id, start, end)
    intervals = sorted(shots.values(), key=lambda item: item.start_frame)
    missing = 0
    duplicate = 0
    cursor = 0
    for interval in intervals:
        if interval.start_frame > cursor:
            missing += interval.start_frame - cursor
        elif interval.start_frame < cursor:
            duplicate += (
                min(interval.end_frame + 1, cursor)
                - interval.start_frame
            )
        cursor = max(cursor, interval.end_frame + 1)
    if cursor < frame_count:
        missing += frame_count - cursor
    unknown_candidates: list[str] = []
    membership_errors = 0
    for candidate in candidates:
        interval = shots.get(candidate.shot_id)
        if interval is None:
            unknown_candidates.append(candidate.candidate_id)
            continue
        membership_errors += sum(
            not (
                interval.start_frame
                <= observation.scene_local_frame
                <= interval.end_frame
            )
            for observation in candidate.observations
        )
    passed = (
        not unapproved_shot_ids
        and missing == 0
        and duplicate == 0
        and not unknown_candidates
        and membership_errors == 0
    )
    return (
        {
            "status": "PASS" if passed else "FAIL",
            "approved_review_states": sorted(
                APPROVED_SHOT_REVIEW_STATES
            ),
            "shot_count": len(shots),
            "unapproved_review_count": len(unapproved_shot_ids),
            "unapproved_shot_ids": unapproved_shot_ids,
            "frame_count": frame_count,
            "covered_frame_count": frame_count - missing,
            "missing_frame_count": missing,
            "duplicate_frame_count": duplicate,
            "unknown_candidate_shot_count": len(unknown_candidates),
            "unknown_candidate_ids": unknown_candidates,
            "observation_membership_error_count": membership_errors,
        },
        tuple(intervals),
    )


def shot_for_frame(
    intervals: tuple[ShotInterval, ...],
    frame: int,
) -> str | None:
    for interval in intervals:
        if interval.start_frame <= frame <= interval.end_frame:
            return interval.shot_id
    return None
