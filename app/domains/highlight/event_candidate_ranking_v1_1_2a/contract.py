from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    CandidateSequence,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.contract import (
    _first,
    _load_shots,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.contract import (
    APPROVED_SHOT_REVIEW_STATES,
    ShotInterval,
)


def audit_shot_contract_v112a(
    path: Path,
    *,
    candidates: tuple[CandidateSequence, ...],
    frame_count: int,
) -> tuple[dict[str, Any], tuple[ShotInterval, ...]]:
    document = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() == ".json"
        else {}
    )
    automatic = (
        isinstance(document, dict)
        and document.get("artifact_type") == "AUTO_SHOT_BOUNDARIES"
        and document.get("boundary_origin") == "AUTO_DETECTED"
        and document.get("human_reviewed") is False
        and document.get("automatic_target_confirmation") is False
        and (document.get("structural_validation") or {}).get("status") == "PASS"
    )
    rows = _load_shots(path)
    shots: dict[str, ShotInterval] = {}
    unapproved_shot_ids: list[str] = []
    for index, row in enumerate(rows):
        shot_id = str(_first(row, "shot_id", "id") or f"shot_{index}")
        if shot_id in shots:
            raise ValueError("Shot-boundary artifact contains duplicate shot IDs.")
        start = _first(row, "start_frame", "first_frame", "frame_start")
        end = _first(
            row,
            "end_frame",
            "end_frame_inclusive",
            "last_frame",
            "frame_end",
        )
        if start is None or end is None:
            raise ValueError("Shot-boundary row has no frame interval.")
        start, end = int(start), int(end)
        if start < 0 or end < start or end >= frame_count:
            raise ValueError("Shot-boundary interval is outside scene video.")
        review_state = str(
            _first(
                row,
                "review_status",
                "review_state",
                "status",
                "boundary_status",
            )
            or ""
        ).strip().upper()
        if automatic and int(row.get("shot_index", -1)) != index:
            raise ValueError("Shot indexes are not contiguous.")
        if automatic and review_state in APPROVED_SHOT_REVIEW_STATES:
            raise ValueError(
                "Automatic shot boundaries must not impersonate review approval."
            )
        if not automatic and review_state not in APPROVED_SHOT_REVIEW_STATES:
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
            duplicate += min(interval.end_frame + 1, cursor) - interval.start_frame
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
            "approved_review_states": sorted(APPROVED_SHOT_REVIEW_STATES),
            "boundary_origin": (
                "AUTO_DETECTED" if automatic else "HUMAN_REVIEWED"
            ),
            "human_reviewed": not automatic,
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
