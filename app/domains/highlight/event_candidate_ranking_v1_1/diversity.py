from __future__ import annotations

from typing import Any

import numpy as np


def _temporal_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if not left or not right:
        return 0.0
    start = max(left[0]["time_sec"], right[0]["time_sec"])
    end = min(left[-1]["time_sec"], right[-1]["time_sec"])
    union = max(left[-1]["time_sec"], right[-1]["time_sec"]) - min(
        left[0]["time_sec"], right[0]["time_sec"]
    )
    return max(0.0, end - start) / union if union > 0 else 0.0


def _trajectory_distance(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> float | None:
    right_by_frame = {row["frame"]: row for row in right}
    distances: list[float] = []
    for row in left:
        other = right_by_frame.get(row["frame"])
        if other is None:
            continue
        lbox, rbox = row["bbox_xyxy"], other["bbox_xyxy"]
        left_center = np.asarray(
            [(lbox[0] + lbox[2]) / 2, (lbox[1] + lbox[3]) / 2]
        )
        right_center = np.asarray(
            [(rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2]
        )
        distances.append(
            float(
                np.linalg.norm(
                    (left_center - right_center) / np.asarray([width, height])
                )
            )
        )
    return float(np.mean(distances)) if distances else None


def diverse_shortlist(
    rows: list[dict[str, Any]],
    *,
    size: int,
    width: int,
    height: int,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = policy["shortlist_diversity"]
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["recommendation_score"]),
            row["candidate_id"],
        ),
    )
    included: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    closeup_count = 0
    for row in ordered:
        exclusion: str | None = None
        for selected in included:
            if row["local_tracklet_id"] == selected["local_tracklet_id"]:
                exclusion = "DUPLICATE_LOCAL_TRACKLET"
                break
            same_shot = row["shot_id"] == selected["shot_id"]
            overlap = _temporal_overlap(row["_trajectory"], selected["_trajectory"])
            distance = _trajectory_distance(
                row["_trajectory"],
                selected["_trajectory"],
                width=width,
                height=height,
            )
            if (
                same_shot
                and overlap >= float(settings["fragment_time_overlap_threshold"])
                and distance is not None
                and distance <= float(settings["fragment_trajectory_distance_threshold"])
            ):
                exclusion = "OVERLAPPING_FRAGMENT_SAME_SHOT"
                break
        is_closeup_only = (
            row["raw_features"]["broadcast"]["first_post_event_closeup_delay_sec"]
            is not None
            and row["raw_features"]["temporal"]["visible_duration_before_event_sec"]
            == 0.0
        )
        if (
            exclusion is None
            and is_closeup_only
            and closeup_count >= int(settings["max_post_event_closeup_only"])
        ):
            exclusion = "POST_EVENT_CLOSEUP_DIVERSITY_CAP"
        if exclusion is None and len(included) < size:
            included.append(row)
            closeup_count += int(is_closeup_only)
            decisions.append(
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "INCLUDED",
                    "reason": "HIGHEST_AVAILABLE_DIVERSE_SCORE",
                }
            )
        else:
            decisions.append(
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "EXCLUDED",
                    "reason": exclusion or "SHORTLIST_CAPACITY_REACHED",
                }
            )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    for shortlist_rank, row in enumerate(included, start=1):
        row["shortlist_rank"] = shortlist_rank
    return included, decisions
