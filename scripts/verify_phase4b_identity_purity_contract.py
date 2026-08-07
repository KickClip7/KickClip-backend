from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PHASE4B_POLICY = (
    "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_"
    "R12_GROUP_CONFIDENCE_REVIEW_CATALOG"
)
PURITY_POLICY = "TARGET_AGNOSTIC_CONFIRMED_CHANGE_POINT_SEGMENTATION_R2"
SELECTION_POLICY = "CORROBORATION_GROUP_CONFIDENCE_REVIEW_CATALOG_R4"
RESCUE_POLICY = "GROUP_CONFIDENCE_ASSISTED_REVIEW_R5"
MAX_REVIEW_CANDIDATES = 3


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_attempt(report: Mapping[str, Any], shot_id: str | None) -> dict[str, Any]:
    attempts = _rows(report.get("attempts")) or [dict(report)]
    if shot_id is None:
        return attempts[-1]
    selected = next(
        (row for row in attempts if str(row.get("shot_id") or "") == shot_id),
        None,
    )
    if selected is None:
        raise ValueError(f"Shot attempt not found: {shot_id}")
    return selected


def _verify_hashed_file(
    path_value: object,
    sha_value: object,
    label: str,
    errors: list[str],
    *,
    required: bool,
) -> None:
    path_text = str(path_value or "")
    expected = str(sha_value or "")
    if not path_text:
        if required:
            errors.append(f"{label}_PATH_MISSING")
        return
    path = Path(path_text).resolve()
    if not path.is_file():
        errors.append(f"{label}_MISSING:{path}")
    elif len(expected) != 64 or _sha(path) != expected:
        errors.append(f"{label}_SHA_MISMATCH:{path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--shot-id", default=None)
    parser.add_argument("--require-split-parent", action="store_true")
    parser.add_argument("--require-review-candidate", action="store_true")
    args = parser.parse_args()

    root = args.job_root.resolve()
    state = _read(root / "pipeline_state.json")
    report = _read(root / "phase4b_first_cross_shot_report.json")
    attempt = _resolve_attempt(report, args.shot_id)
    errors: list[str] = []

    if report.get("policy") != PHASE4B_POLICY:
        errors.append("REPORT_POLICY_MISMATCH")
    if attempt.get("policy") != PHASE4B_POLICY:
        errors.append("ATTEMPT_POLICY_MISMATCH")
    if attempt.get("identity_purity_policy") != PURITY_POLICY:
        errors.append("IDENTITY_PURITY_POLICY_MISMATCH")
    if attempt.get("assisted_review_selection_policy") != SELECTION_POLICY:
        errors.append("SELECTION_POLICY_MISMATCH")
    if attempt.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")
    if attempt.get("candidate_link_created") is not False:
        errors.append("CANDIDATE_LINK_CREATED_BEFORE_REVIEW")

    parents = _rows(attempt.get("parent_tracklet_summaries"))
    rejected = _rows(attempt.get("rejected_identity_purity_segments"))
    review = _rows(attempt.get("review_candidates"))
    parent_count = int(
        attempt.get("parent_tracklet_count_before_purity_segmentation") or 0
    )
    split_count = int(attempt.get("split_parent_tracklet_count") or 0)
    segment_candidate_count = int(
        attempt.get("identity_segment_candidate_count") or 0
    )
    rejected_count = int(
        attempt.get("rejected_identity_purity_segment_count") or 0
    )
    review_count = int(attempt.get("review_candidate_count") or 0)

    if parent_count != len(parents):
        errors.append("PARENT_TRACKLET_COUNT_MISMATCH")
    actual_split_count = sum(row.get("split_parent") is True for row in parents)
    if split_count != actual_split_count:
        errors.append("SPLIT_PARENT_COUNT_MISMATCH")
    if rejected_count != len(rejected):
        errors.append("REJECTED_PURITY_SEGMENT_COUNT_MISMATCH")
    if review_count != len(review):
        errors.append("REVIEW_CANDIDATE_COUNT_PAYLOAD_MISMATCH")
    if len(review) > MAX_REVIEW_CANDIDATES:
        errors.append("REVIEW_LIMIT_EXCEEDED")
    if args.require_split_parent and split_count < 1:
        errors.append("SPLIT_PARENT_REQUIRED_BUT_MISSING")
    if args.require_review_candidate and not review:
        errors.append("REVIEW_CANDIDATE_REQUIRED_BUT_MISSING")

    candidate_parent: dict[str, str] = {}
    for parent in parents:
        parent_id = str(parent.get("parent_tracklet_id") or "")
        if not parent_id:
            errors.append("PARENT_TRACKLET_ID_MISSING")
        if parent.get("identity_purity_policy") != PURITY_POLICY:
            errors.append(f"PARENT_POLICY_MISMATCH:{parent_id}")
        if parent.get("target_memory_used_for_segmentation") is not False:
            errors.append(f"TARGET_MEMORY_USED_FOR_SEGMENTATION:{parent_id}")
        if int(parent.get("maximum_review_window_frames") or 0) != 0:
            errors.append(f"FORCED_TIME_WINDOW_SPLIT_PRESENT:{parent_id}")
        if "MAX_REVIEW_WINDOW" in {
            str(value) for value in parent.get("split_reasons") or []
        }:
            errors.append(f"MAX_REVIEW_WINDOW_REASON_FORBIDDEN:{parent_id}")
        segments = _rows(parent.get("segments"))
        if int(parent.get("segment_count") or 0) != len(segments):
            errors.append(f"PARENT_SEGMENT_COUNT_MISMATCH:{parent_id}")
        if parent.get("split_parent") is True and len(segments) < 2:
            errors.append(f"SPLIT_PARENT_WITHOUT_MULTIPLE_SEGMENTS:{parent_id}")
        for segment in segments:
            candidate_id = str(segment.get("candidate_id") or "")
            if not candidate_id:
                errors.append(f"SEGMENT_CANDIDATE_ID_MISSING:{parent_id}")
                continue
            candidate_parent[candidate_id] = parent_id
            if segment.get("policy") != PURITY_POLICY:
                errors.append(f"SEGMENT_POLICY_MISMATCH:{candidate_id}")
            if segment.get("target_memory_used_for_segmentation") is not False:
                errors.append(f"SEGMENT_TARGET_MEMORY_USED:{candidate_id}")
            if segment.get("automatic_target_confirmation") is not False:
                errors.append(f"SEGMENT_AUTO_CONFIRMATION_FORBIDDEN:{candidate_id}")
            start = int(segment.get("segment_start_frame") or 0)
            end = int(segment.get("segment_end_frame_inclusive") or -1)
            if end < start:
                errors.append(f"SEGMENT_FRAME_RANGE_INVALID:{candidate_id}")

    rejected_ids = {str(row.get("candidate_id") or "") for row in rejected}
    review_ids = {str(row.get("candidate_id") or "") for row in review}
    if rejected_ids & review_ids:
        errors.append("REJECTED_PURITY_SEGMENT_EXPOSED_FOR_REVIEW")

    for row in review:
        candidate_id = str(row.get("candidate_id") or "")
        purity = dict(row.get("identity_purity") or {})
        if purity.get("policy") != PURITY_POLICY:
            errors.append(f"REVIEW_PURITY_POLICY_MISMATCH:{candidate_id}")
        if purity.get("passed") is not True:
            errors.append(f"REVIEW_IDENTITY_PURITY_FAILED:{candidate_id}")
        if purity.get("target_memory_used_for_segmentation") is not False:
            errors.append(f"REVIEW_TARGET_MEMORY_USED_FOR_SEGMENTATION:{candidate_id}")
        parent_id = str(row.get("parent_tracklet_id") or purity.get("parent_tracklet_id") or "")
        if not parent_id:
            errors.append(f"REVIEW_PARENT_TRACKLET_ID_MISSING:{candidate_id}")
        elif candidate_parent.get(candidate_id) not in {None, parent_id}:
            errors.append(f"REVIEW_PARENT_TRACKLET_MISMATCH:{candidate_id}")
        if row.get("assisted_review_selection_policy") != SELECTION_POLICY:
            errors.append(f"REVIEW_SELECTION_POLICY_MISMATCH:{candidate_id}")
        if row.get("rescue_review_required") is True:
            if row.get("rescue_review_policy") != RESCUE_POLICY:
                errors.append(f"REVIEW_RESCUE_POLICY_MISMATCH:{candidate_id}")
        if row.get("automatic_target_confirmation") is True:
            errors.append(f"REVIEW_AUTO_CONFIRMATION_FORBIDDEN:{candidate_id}")

    # R10 must never join two segments from the same parent back into one
    # continuation group after the purity split.
    for group in _rows(attempt.get("continuation_groups")):
        member_ids = [str(value) for value in group.get("member_candidate_ids") or []]
        member_parents = [candidate_parent.get(value) for value in member_ids]
        known = [value for value in member_parents if value]
        if len(known) != len(set(known)):
            errors.append(
                f"SAME_PARENT_SEGMENTS_REMERGED:{group.get('group_id')}"
            )

    _verify_hashed_file(
        attempt.get("ranked_candidates_path"),
        attempt.get("ranked_candidates_sha256"),
        "RANKED_CANDIDATES",
        errors,
        required=True,
    )
    _verify_hashed_file(
        attempt.get("contact_sheet_path"),
        attempt.get("contact_sheet_sha256"),
        "CONTACT_SHEET",
        errors,
        required=True,
    )
    _verify_hashed_file(
        attempt.get("identity_purity_rejection_sheet_path"),
        attempt.get("identity_purity_rejection_sheet_sha256"),
        "IDENTITY_PURITY_REJECTION_SHEET",
        errors,
        required=rejected_count > 0,
    )

    pending = dict(state.get("pending_action") or {})
    if review:
        if state.get("status") != "NEEDS_CONFIRMATION":
            errors.append("STATE_NOT_WAITING_FOR_CONFIRMATION")
        if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
            errors.append("PENDING_ACTION_MISSING")
        pending_ids = {str(value) for value in pending.get("candidate_ids") or []}
        if pending_ids != review_ids:
            errors.append("PENDING_REVIEW_SET_MISMATCH")
        if pending.get("automatic_target_confirmation") is not False:
            errors.append("PENDING_AUTO_CONFIRMATION_FORBIDDEN")

    status = "PASS" if not errors else "FAIL"
    print(f"status={status}")
    print(f"shot_id={attempt.get('shot_id')}")
    print(f"phase4b_policy={attempt.get('policy')}")
    print(f"identity_purity_policy={attempt.get('identity_purity_policy')}")
    print(f"parent_tracklet_count={parent_count}")
    print(f"split_parent_tracklet_count={split_count}")
    print(f"identity_segment_candidate_count={segment_candidate_count}")
    print(f"rejected_identity_purity_segment_count={rejected_count}")
    print(f"review_candidate_count={len(review)}")
    print(f"review_candidate_ids={sorted(review_ids)}")
    print(f"pending_action_type={pending.get('type')}")
    print(
        "automatic_target_confirmation="
        f"{attempt.get('automatic_target_confirmation')}"
    )
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
