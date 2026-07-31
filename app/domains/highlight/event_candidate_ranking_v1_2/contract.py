from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APPROVED_SHOT_STATES = frozenset({"REVIEWED_PASS", "CONFIRMED"})


@dataclass(frozen=True)
class ReviewedShot:
    shot_id: str
    shot_index: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_reviewed_shots(document: dict[str, Any]) -> tuple[ReviewedShot, ...]:
    rows = document.get("shots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reviewed shot artifact has no shots.")
    shots: list[ReviewedShot] = []
    expected_frame = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Reviewed shot rows must be objects.")
        shot_id = str(row.get("shot_id") or "")
        shot_index = int(row.get("shot_index", -1))
        start_frame = int(row.get("start_frame", -1))
        end_frame = int(
            row.get(
                "end_frame_inclusive",
                row.get("end_frame", row.get("last_frame", -1)),
            )
        )
        start_time = float(row.get("start_time_sec", -1))
        end_time = float(row.get("end_time_sec", -1))
        state = str(
            row.get("review_state")
            or row.get("review_status")
            or row.get("status")
            or row.get("boundary_status")
            or ""
        ).strip().upper()
        if (
            not shot_id
            or shot_index != index
            or start_frame != expected_frame
            or end_frame < start_frame
            or start_time < 0
            or end_time <= start_time
        ):
            raise ValueError("Reviewed shot contract is invalid.")
        if state not in APPROVED_SHOT_STATES:
            raise ValueError(f"Shot is not approved: {shot_id}")
        shots.append(
            ReviewedShot(
                shot_id=shot_id,
                shot_index=shot_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_sec=start_time,
                end_time_sec=end_time,
            )
        )
        expected_frame = end_frame + 1
    frame_count = int((document.get("video") or {}).get("frame_count") or -1)
    if frame_count <= 0 or expected_frame != frame_count:
        raise ValueError("Reviewed shots do not cover the full scene.")
    return tuple(shots)


def validate_v112a_ranking(
    document: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    if (
        document.get("package")
        != "target_centric_tracking_event_candidate_ranking_v1_1_2a"
        or document.get("schema_version")
        != "kickclip.event_candidate_ranking.v1_1_2a"
        or document.get("automatic_target_confirmation") is not False
    ):
        raise ValueError("Source ranking is not frozen V1.1.2a shadow output.")
    rows = document.get("all_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source ranking has no global candidate rows.")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for expected_rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("Global ranking rows must be objects.")
        candidate_id = str(row.get("candidate_id") or "")
        rank = int(row.get("rank", -1))
        score = row.get("recommendation_score")
        shot_id = str(row.get("shot_id") or "")
        if (
            not candidate_id
            or candidate_id in seen
            or rank != expected_rank
            or not shot_id
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("Global ranking invariant is invalid.")
        seen.add(candidate_id)
        validated.append(row)
    return tuple(validated)


def global_ranking_fingerprint(rows: tuple[dict[str, Any], ...]) -> str:
    return canonical_sha256(
        [
            {
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "recommendation_score": float(row["recommendation_score"]),
            }
            for row in rows
        ]
    )

