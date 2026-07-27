from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256

from app.domains.tracking.errors import TrackingValidationError


TEST_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
UNCERTAIN_TARGET_STATES = {"SEARCHING", "AMBIGUOUS", "ABSENT"}


def generate_tracking_test_name(tracking_job_id: str) -> str:
    """Build a runner-safe server-owned test name."""

    compact = re.sub(r"[^A-Za-z0-9]", "", tracking_job_id)
    if not compact:
        compact = sha256(tracking_job_id.encode("utf-8")).hexdigest()[:24]
    name = f"tracking_{compact}"
    if not TEST_NAME_PATTERN.fullmatch(name):
        raise TrackingValidationError("Could not generate a safe tracking test name.")
    return name


def validate_bbox_xyxy(
    values: Sequence[float],
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[float]:
    if len(values) != 4:
        raise TrackingValidationError(
            "initial_bbox_xyxy must contain exactly four values."
        )
    bbox = [float(value) for value in values]
    if not all(math.isfinite(value) for value in bbox):
        raise TrackingValidationError("BBox coordinates must be finite numbers.")
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise TrackingValidationError("BBox must satisfy x1 < x2 and y1 < y2.")
    if width is not None and height is not None:
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise TrackingValidationError(
                "BBox must be inside the source video's pixel dimensions."
            )
    return bbox


def pending_review_candidate_ids(
    state: Mapping[str, object],
    ambiguity_id: str,
) -> set[str]:
    pending = state.get("pending_action")
    if not isinstance(pending, Mapping):
        return set()
    if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
        return set()
    if str(pending.get("ambiguity_id") or "") != ambiguity_id:
        return set()

    ambiguities = state.get("ambiguities")
    if not isinstance(ambiguities, list):
        return set()
    ambiguity = next(
        (
            item
            for item in ambiguities
            if isinstance(item, Mapping)
            and str(item.get("ambiguity_id") or "") == ambiguity_id
            and str(item.get("status") or "") == "PENDING"
        ),
        None,
    )
    if ambiguity is None:
        return set()
    candidates = ambiguity.get("review_candidates")
    if not isinstance(candidates, list):
        return set()
    return {
        str(item.get("candidate_id"))
        for item in candidates
        if isinstance(item, Mapping) and item.get("candidate_id")
    }


def build_action_key(*parts: object) -> str:
    return ":".join(str(part).strip() for part in parts)

