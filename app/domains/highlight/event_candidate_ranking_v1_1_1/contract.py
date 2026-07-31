from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    CandidateSequence,
    canonical_event_label,
)
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent


@dataclass(frozen=True)
class ResolvedEventContext:
    event_id: str
    event_label: str
    canonical_event_label: str | None
    event_time_sec: float
    event_confidence: float | None
    event_source_job_id: str
    event_source_artifact_id: str | None
    scene_id: str
    scene_start_sec: float
    scene_end_sec: float
    match_id: str
    project_id: str
    revision_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_event_context(
    db: Session,
    *,
    project: Project,
    revision_id: str,
    event_id: str,
    scene_id: str,
) -> ResolvedEventContext:
    revision = db.get(HighlightRevision, revision_id)
    if revision is None or revision.project_id != project.project_id:
        raise ValueError("Highlight revision is missing or outside the project.")
    if scene_id not in set(revision.selected_scene_ids or []):
        raise ValueError("Scene is not selected by the highlight revision.")
    event = db.get(TimelineEvent, event_id)
    scene = db.get(TimelineEvent, scene_id)
    if event is None or scene is None:
        raise ValueError("Event or scene TimelineEvent was not found.")
    if (
        event.match_id != project.match_id
        or scene.match_id != project.match_id
    ):
        raise ValueError("Event and scene must belong to the project Match.")
    if not revision.action_spotting_job_id:
        raise ValueError("Revision has no frozen Action Spotting job.")
    if event.source_job_id != revision.action_spotting_job_id:
        raise ValueError("Event does not belong to the revision Action Spotting job.")
    scene_metadata = scene.metadata_ or {}
    linked_event_ids = set(
        scene_metadata.get("source_event_ids")
        or scene_metadata.get("event_ids")
        or []
    )
    related = (
        event.timeline_event_id == scene.timeline_event_id
        or event.timeline_event_id in linked_event_ids
        or (
            event.source_job_id == scene.source_job_id
            and scene.start_sec <= event.timestamp_sec <= scene.end_sec
        )
    )
    if not related:
        raise ValueError("Event is not related to the selected scene.")
    return ResolvedEventContext(
        event_id=event.timeline_event_id,
        event_label=event.label,
        canonical_event_label=canonical_event_label(event.label),
        event_time_sec=float(event.timestamp_sec),
        event_confidence=(
            float(event.confidence) if event.confidence is not None else None
        ),
        event_source_job_id=event.source_job_id,
        event_source_artifact_id=event.source_artifact_id,
        scene_id=scene.timeline_event_id,
        scene_start_sec=float(scene.start_sec),
        scene_end_sec=float(scene.end_sec),
        match_id=project.match_id,
        project_id=project.project_id,
        revision_id=revision.revision_id,
    )


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _load_shots(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        for key in ("shots", "boundaries", "shot_boundaries"):
            if isinstance(document.get(key), list):
                return document[key]
    raise ValueError("Shot-boundary artifact has no supported shot array.")


def audit_shot_contract(
    path: Path,
    *,
    candidates: tuple[CandidateSequence, ...],
    frame_count: int,
) -> dict[str, Any]:
    rows = _load_shots(path)
    shots: dict[str, tuple[int, int]] = {}
    pending_review_count = 0
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
            raise ValueError("Shot-boundary interval is outside the scene video.")
        shots[shot_id] = (start, end)
        review_state = str(
            _first(row, "review_status", "status", "boundary_status") or ""
        ).upper()
        if "PENDING" in review_state or row.get("pending_review") is True:
            pending_review_count += 1
    intervals = sorted(shots.values())
    missing = 0
    duplicate = 0
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            missing += start - cursor
        elif start < cursor:
            duplicate += min(end + 1, cursor) - start
        cursor = max(cursor, end + 1)
    if cursor < frame_count:
        missing += frame_count - cursor
    unknown_candidate_shots: list[str] = []
    observation_membership_errors = 0
    for candidate in candidates:
        interval = shots.get(candidate.shot_id)
        if interval is None:
            unknown_candidate_shots.append(candidate.candidate_id)
            continue
        start, end = interval
        observation_membership_errors += sum(
            not start <= observation.scene_local_frame <= end
            for observation in candidate.observations
        )
    passed = (
        pending_review_count == 0
        and missing == 0
        and duplicate == 0
        and not unknown_candidate_shots
        and observation_membership_errors == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "shot_count": len(shots),
        "pending_review_count": pending_review_count,
        "frame_count": frame_count,
        "covered_frame_count": frame_count - missing,
        "missing_frame_count": missing,
        "duplicate_frame_count": duplicate,
        "unknown_candidate_shot_count": len(unknown_candidate_shots),
        "unknown_candidate_ids": unknown_candidate_shots,
        "observation_membership_error_count": observation_membership_errors,
    }
