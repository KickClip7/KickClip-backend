from __future__ import annotations

from typing import Any

from app.domains.highlight.event_candidate_ranking_v1_1.diversity import (
    _temporal_overlap,
    _trajectory_distance,
)


def diverse_shortlist_v111(
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
        if row["reliability_state"] == "NO_RELIABLE_SHORTLIST":
            exclusion = "CANDIDATE_FEATURES_NOT_RELIABLE"
        for selected in included if exclusion is None else []:
            if (
                row["shot_id"],
                row["local_tracklet_id"],
            ) == (
                selected["shot_id"],
                selected["local_tracklet_id"],
            ):
                exclusion = "DUPLICATE_SHOT_LOCAL_TRACKLET"
                break
            same_shot = row["shot_id"] == selected["shot_id"]
            overlap = _temporal_overlap(
                row["_trajectory"], selected["_trajectory"]
            )
            distance = _trajectory_distance(
                row["_trajectory"],
                selected["_trajectory"],
                width=width,
                height=height,
            )
            if (
                same_shot
                and overlap
                >= float(settings["fragment_time_overlap_threshold"])
                and distance is not None
                and distance
                <= float(
                    settings["fragment_trajectory_distance_threshold"]
                )
            ):
                exclusion = "OVERLAPPING_FRAGMENT_SAME_SHOT"
                break
        is_closeup_only = (
            row["raw_features"]["broadcast"][
                "first_post_event_closeup_delay_sec"
            ]
            is not None
            and row["raw_features"]["temporal"][
                "visible_duration_before_event_sec"
            ]
            == 0.0
        )
        if (
            exclusion is None
            and is_closeup_only
            and closeup_count
            >= int(settings["max_post_event_closeup_only"])
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
                    "duplicate_scope": [
                        row["shot_id"],
                        row["local_tracklet_id"],
                    ],
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
