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
TARGET_CONTEXT_POLICY = "APPROVED_REFERENCE_TORSO_COLOR_SOFT_EVIDENCE_R1"
CORROBORATION_POLICY = (
    "INDEPENDENT_TRACKLET_SPATIOTEMPORAL_APPEARANCE_CORROBORATION_R1"
)
SELECTION_POLICY = "CORROBORATION_GROUP_CONFIDENCE_REVIEW_CATALOG_R4"
MAX_REVIEW = 3


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


def _attempt(report: Mapping[str, Any], shot_id: str | None) -> dict[str, Any]:
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


def _verify_file(
    path_value: object,
    sha_value: object,
    label: str,
    errors: list[str],
) -> None:
    path = Path(str(path_value or "")).resolve()
    expected = str(sha_value or "")
    if not path.is_file():
        errors.append(f"{label}_MISSING:{path}")
    elif len(expected) != 64 or _sha(path) != expected:
        errors.append(f"{label}_SHA_MISMATCH:{path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--shot-id", default=None)
    parser.add_argument("--require-review-candidate", action="store_true")
    parser.add_argument(
        "--require-corroborated-review-candidate", action="store_true"
    )
    args = parser.parse_args()

    root = args.job_root.resolve()
    state = _read(root / "pipeline_state.json")
    report = _read(root / "phase4b_first_cross_shot_report.json")
    attempt = _attempt(report, args.shot_id)
    errors: list[str] = []

    if report.get("policy") != PHASE4B_POLICY:
        errors.append("REPORT_POLICY_MISMATCH")
    if attempt.get("policy") != PHASE4B_POLICY:
        errors.append("ATTEMPT_POLICY_MISMATCH")
    if attempt.get("target_context_policy") != TARGET_CONTEXT_POLICY:
        errors.append("TARGET_CONTEXT_POLICY_MISMATCH")
    if attempt.get("target_context_is_soft_evidence_only") is not True:
        errors.append("TARGET_CONTEXT_NOT_SOFT")
    if attempt.get("corroboration_policy") != CORROBORATION_POLICY:
        errors.append("CORROBORATION_POLICY_MISMATCH")
    if attempt.get("assisted_review_selection_policy") != SELECTION_POLICY:
        errors.append("SELECTION_POLICY_MISMATCH")
    if attempt.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")
    if attempt.get("candidate_link_created") is not False:
        errors.append("CANDIDATE_LINK_CREATED_BEFORE_REVIEW")

    groups = _rows(attempt.get("corroboration_groups"))
    if int(attempt.get("corroboration_group_count") or 0) != len(groups):
        errors.append("CORROBORATION_GROUP_COUNT_MISMATCH")
    for group in groups:
        group_id = str(group.get("group_id") or "")
        if group.get("policy") != CORROBORATION_POLICY:
            errors.append(f"CORROBORATION_GROUP_POLICY_MISMATCH:{group_id}")
        member_ids = [str(value) for value in group.get("member_candidate_ids") or []]
        if int(group.get("member_count") or 0) != len(member_ids):
            errors.append(f"CORROBORATION_MEMBER_COUNT_MISMATCH:{group_id}")
        parent_ids = [
            str(value)
            for value in group.get("independent_parent_tracklet_ids") or []
        ]
        if int(group.get("independent_parent_count") or 0) != len(parent_ids):
            errors.append(f"CORROBORATION_PARENT_COUNT_MISMATCH:{group_id}")
        if group.get("corroborated") is True and len(parent_ids) < 2:
            errors.append(f"CORROBORATED_WITHOUT_TWO_PARENTS:{group_id}")

    review = _rows(attempt.get("review_candidates"))
    if int(attempt.get("review_candidate_count") or 0) != len(review):
        errors.append("REVIEW_CANDIDATE_COUNT_MISMATCH")
    if len(review) > MAX_REVIEW:
        errors.append("REVIEW_LIMIT_EXCEEDED")
    if args.require_review_candidate and not review:
        errors.append("REVIEW_CANDIDATE_REQUIRED_BUT_MISSING")

    corroborated_review = []
    for row in review:
        candidate_id = str(row.get("candidate_id") or "")
        if row.get("target_context_policy") != TARGET_CONTEXT_POLICY:
            errors.append(f"REVIEW_TARGET_CONTEXT_POLICY_MISMATCH:{candidate_id}")
        similarity = float(row.get("target_context_similarity") or 0.0)
        if not 0.0 <= similarity <= 1.0:
            errors.append(f"REVIEW_TARGET_CONTEXT_RANGE_INVALID:{candidate_id}")
        if row.get("target_context_is_soft_evidence_only") is not True:
            errors.append(f"REVIEW_TARGET_CONTEXT_NOT_SOFT:{candidate_id}")
        if row.get("corroboration_policy") != CORROBORATION_POLICY:
            errors.append(f"REVIEW_CORROBORATION_POLICY_MISMATCH:{candidate_id}")
        if row.get("automatic_target_confirmation") is True:
            errors.append(f"REVIEW_AUTO_CONFIRMATION_FORBIDDEN:{candidate_id}")
        if row.get("corroborated_by_independent_tracklet") is True:
            corroborated_review.append(row)
            if int(row.get("corroboration_independent_parent_count") or 0) < 2:
                errors.append(
                    f"REVIEW_CORROBORATION_PARENT_COUNT_INVALID:{candidate_id}"
                )
    if args.require_corroborated_review_candidate and not corroborated_review:
        errors.append("CORROBORATED_REVIEW_CANDIDATE_REQUIRED_BUT_MISSING")

    _verify_file(
        attempt.get("ranked_candidates_path"),
        attempt.get("ranked_candidates_sha256"),
        "RANKED_CANDIDATES",
        errors,
    )
    _verify_file(
        attempt.get("contact_sheet_path"),
        attempt.get("contact_sheet_sha256"),
        "CONTACT_SHEET",
        errors,
    )

    pending = dict(state.get("pending_action") or {})
    review_ids = {str(row.get("candidate_id") or "") for row in review}
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
    print(f"target_context_policy={attempt.get('target_context_policy')}")
    print(
        "target_context_is_soft_evidence_only="
        f"{attempt.get('target_context_is_soft_evidence_only')}"
    )
    print(f"corroboration_policy={attempt.get('corroboration_policy')}")
    print(f"corroboration_group_count={len(groups)}")
    print(f"review_candidate_count={len(review)}")
    print(
        "review_candidate_ids="
        f"{[row.get('candidate_id') for row in review]}"
    )
    print(
        "corroborated_review_candidate_ids="
        f"{[row.get('candidate_id') for row in corroborated_review]}"
    )
    print(f"pending_action_type={pending.get('type')}")
    print(f"automatic_target_confirmation={attempt.get('automatic_target_confirmation')}")
    print(f"errors={errors}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
