from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Iterable


# This policy applies ONLY to the user's first target-selection gallery.
# Cross-shot ambiguity review remains capped separately by the tracking runtime.
INITIAL_TARGET_GALLERY_POLICY_VERSION = "WIDE_TEMPORAL_DIVERSITY_R1"
INITIAL_TARGET_GALLERY_MAX_CANDIDATES = 15
INITIAL_TARGET_GALLERY_MAX_PER_TEMPORAL_BUCKET = 3
INITIAL_TARGET_GALLERY_BUCKET_SECONDS = 0.50


def _nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _candidate_time_sec(row: dict[str, Any]) -> float:
    raw_features = _nested_dict(row.get("raw_features"))
    temporal = _nested_dict(raw_features.get("temporal"))
    for key in (
        "first_candidate_time_sec",
        "last_candidate_time_sec",
    ):
        value = temporal.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, float(value))

    evidence = _nested_dict(row.get("feature_evidence"))
    frames = evidence.get("visual_frames") or evidence.get("temporal_frames") or []
    numeric_frames = [
        int(value)
        for value in frames
        if isinstance(value, (int, float)) and int(value) >= 0
    ]
    if numeric_frames:
        # Fallback only. The normal V1.1.2a document contains local seconds.
        return float(min(numeric_frames)) / 25.0
    return 0.0


def _bucket_key(row: dict[str, Any]) -> tuple[str, int]:
    shot_id = str(row.get("shot_id") or "UNKNOWN_SHOT")
    time_sec = _candidate_time_sec(row)
    bucket = int(math.floor(time_sec / INITIAL_TARGET_GALLERY_BUCKET_SECONDS + 1e-9))
    return shot_id, bucket


def select_initial_target_gallery(
    ranking: dict[str, Any],
    *,
    available_source_candidate_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a recall-oriented, temporally diverse initial target gallery.

    This deliberately does NOT alter the frozen V1.1.2a recommendation score or
    V1.2 shortlist. The V1.2 shortlist remains an immutable recommendation view.
    The user, however, needs enough temporal coverage to pick the actual target
    before tracking starts, so this product integration layer surfaces more
    already-ranked WIDE candidates across distinct sampled moments.
    """

    allowed_source_ids = (
        {str(value) for value in available_source_candidate_ids}
        if available_source_candidate_ids is not None
        else None
    )
    policy = _nested_dict(ranking.get("shortlist_policy"))
    eligible_states = {
        str(value)
        for value in policy.get("eligible_reliability_states") or ["READY", "PARTIAL_FEATURES"]
    }

    rows: list[dict[str, Any]] = []
    for raw in ranking.get("all_candidates") or []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "")
        if not candidate_id:
            continue
        if allowed_source_ids is not None and candidate_id not in allowed_source_ids:
            continue
        if str(raw.get("reliability_state") or "") not in eligible_states:
            continue
        rows.append(raw)

    rows.sort(
        key=lambda row: (
            int(row.get("rank") or 10**9),
            str(row.get("candidate_id") or ""),
        )
    )
    if not rows:
        return []

    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_bucket_key(row)].append(row)

    # Buckets with a strong global candidate are visited first, but we round-robin
    # through buckets so one event-near 3-frame burst cannot consume the gallery.
    bucket_order = sorted(
        buckets,
        key=lambda key: (
            min(int(row.get("rank") or 10**9) for row in buckets[key]),
            key[0],
            key[1],
        ),
    )
    for key in bucket_order:
        buckets[key].sort(
            key=lambda row: (
                int(row.get("rank") or 10**9),
                str(row.get("candidate_id") or ""),
            )
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for depth in range(INITIAL_TARGET_GALLERY_MAX_PER_TEMPORAL_BUCKET):
        for key in bucket_order:
            bucket_rows = buckets[key]
            if depth >= len(bucket_rows):
                continue
            row = bucket_rows[depth]
            candidate_id = str(row["candidate_id"])
            if candidate_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(candidate_id)
            if len(selected) >= INITIAL_TARGET_GALLERY_MAX_CANDIDATES:
                break
        if len(selected) >= INITIAL_TARGET_GALLERY_MAX_CANDIDATES:
            break

    # If the scene has only one/few temporal buckets, fill remaining slots by the
    # immutable global rank. This keeps the gallery useful without weakening the
    # cross-shot confirmation cap.
    if len(selected) < INITIAL_TARGET_GALLERY_MAX_CANDIDATES:
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(candidate_id)
            if len(selected) >= INITIAL_TARGET_GALLERY_MAX_CANDIDATES:
                break

    output: list[dict[str, Any]] = []
    for gallery_rank, row in enumerate(selected, start=1):
        copied = copy.deepcopy(row)
        copied["initial_target_gallery_rank"] = gallery_rank
        copied["initial_target_gallery_policy"] = INITIAL_TARGET_GALLERY_POLICY_VERSION
        copied["initial_target_temporal_bucket"] = {
            "shot_id": _bucket_key(row)[0],
            "bucket_index": _bucket_key(row)[1],
            "bucket_seconds": INITIAL_TARGET_GALLERY_BUCKET_SECONDS,
            "candidate_time_sec": _candidate_time_sec(row),
        }
        output.append(copied)
    return output
