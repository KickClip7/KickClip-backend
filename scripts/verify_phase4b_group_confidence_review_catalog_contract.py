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
SELECTION_POLICY = "CORROBORATION_GROUP_CONFIDENCE_REVIEW_CATALOG_R4"
CATALOG_POLICY = "ALL_SAFE_PLAUSIBLE_EVIDENCE_GROUPS_R1"
PROMOTION_POLICY = "USER_SELECTED_SAFE_CANDIDATE_PROMOTION_R1"
GROUP_REPRESENTATIVE_POLICY = "IDENTITY_PURITY_CLEAN_RATIO_REPRESENTATIVE_R1"
GROUP_CONFIDENCE_POLICY = "INDEPENDENT_TRACKLET_GROUP_AGREEMENT_R1"
CORROBORATION_POLICY = (
    "INDEPENDENT_TRACKLET_SPATIOTEMPORAL_APPEARANCE_CORROBORATION_R1"
)
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
    parser.add_argument(
        "--require-candidate-id",
        default=None,
        help=(
            "Require a candidate id to be present in either the active review set "
            "or the safe review catalog. This is a generic verifier input, not a "
            "hard-coded sample rule."
        ),
    )
    parser.add_argument(
        "--require-candidate-in-active-review",
        action="store_true",
        help="When used with --require-candidate-id, require the candidate in Top-N review.",
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
    if attempt.get("assisted_review_selection_policy") != SELECTION_POLICY:
        errors.append("SELECTION_POLICY_MISMATCH")
    if attempt.get("review_catalog_policy") != CATALOG_POLICY:
        errors.append("CATALOG_POLICY_MISMATCH")
    if attempt.get("group_representative_policy") != GROUP_REPRESENTATIVE_POLICY:
        errors.append("GROUP_REPRESENTATIVE_POLICY_MISMATCH")
    if attempt.get("group_confidence_policy") != GROUP_CONFIDENCE_POLICY:
        errors.append("GROUP_CONFIDENCE_POLICY_MISMATCH")
    if attempt.get("manual_review_promotion_allowed") is not True:
        errors.append("MANUAL_REVIEW_PROMOTION_NOT_ALLOWED")
    if attempt.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")
    if attempt.get("candidate_link_created") is not False:
        errors.append("CANDIDATE_LINK_CREATED_BEFORE_REVIEW")

    review = _rows(attempt.get("review_candidates"))
    catalog = _rows(attempt.get("review_catalog_candidates"))
    review_ids = [str(row.get("candidate_id") or "") for row in review]
    catalog_ids = [str(row.get("candidate_id") or "") for row in catalog]
    if int(attempt.get("review_candidate_count") or 0) != len(review):
        errors.append("REVIEW_CANDIDATE_COUNT_MISMATCH")
    if int(attempt.get("review_catalog_candidate_count") or 0) != len(catalog):
        errors.append("REVIEW_CATALOG_COUNT_MISMATCH")
    if len(review) > MAX_REVIEW:
        errors.append("REVIEW_LIMIT_EXCEEDED")
    if args.require_review_candidate and not review:
        errors.append("REVIEW_CANDIDATE_REQUIRED_BUT_MISSING")
    if len(catalog_ids) != len(set(catalog_ids)):
        errors.append("REVIEW_CATALOG_DUPLICATE_IDS")
    if not set(review_ids).issubset(set(catalog_ids)):
        errors.append("ACTIVE_REVIEW_NOT_SUBSET_OF_CATALOG")

    catalog_ranks = [int(row.get("review_catalog_rank") or 0) for row in catalog]
    if catalog and catalog_ranks != list(range(1, len(catalog) + 1)):
        errors.append("REVIEW_CATALOG_RANKS_NOT_CONTIGUOUS")
    for row in catalog:
        candidate_id = str(row.get("candidate_id") or "")
        if row.get("review_catalog_policy") != CATALOG_POLICY:
            errors.append(f"CATALOG_ROW_POLICY_MISMATCH:{candidate_id}")
        if row.get("manual_review_promotion_allowed") is not True:
            errors.append(f"CATALOG_ROW_PROMOTION_DISABLED:{candidate_id}")
        if dict(row.get("role_confusion") or {}).get("passed") is not True:
            errors.append(f"CATALOG_ROLE_GATE_FAILED:{candidate_id}")
        if dict(row.get("identity_observability") or {}).get("passed") is not True:
            errors.append(f"CATALOG_OBSERVABILITY_GATE_FAILED:{candidate_id}")
        if dict(row.get("identity_purity") or {}).get("passed") is not True:
            errors.append(f"CATALOG_PURITY_GATE_FAILED:{candidate_id}")
        if dict(row.get("negative_review_gate") or {}).get("passed") is not True:
            errors.append(f"CATALOG_NEGATIVE_GATE_FAILED:{candidate_id}")
        if row.get("automatic_target_confirmation") is True:
            errors.append(f"CATALOG_AUTO_CONFIRMATION_FORBIDDEN:{candidate_id}")

    groups = _rows(attempt.get("corroboration_groups"))
    corroborated_catalog = [
        row for row in catalog if row.get("corroborated_by_independent_tracklet") is True
    ]
    corroborated_review = [
        row for row in review if row.get("corroborated_by_independent_tracklet") is True
    ]
    if args.require_corroborated_review_candidate and not corroborated_review:
        errors.append("CORROBORATED_REVIEW_CANDIDATE_REQUIRED_BUT_MISSING")

    for group in groups:
        group_id = str(group.get("group_id") or "")
        if group.get("policy") != CORROBORATION_POLICY:
            errors.append(f"CORROBORATION_POLICY_MISMATCH:{group_id}")
        if group.get("corroborated") is True:
            if group.get("representative_policy") != GROUP_REPRESENTATIVE_POLICY:
                errors.append(f"GROUP_REPRESENTATIVE_POLICY_MISMATCH:{group_id}")
            if group.get("group_confidence_policy") != GROUP_CONFIDENCE_POLICY:
                errors.append(f"GROUP_CONFIDENCE_POLICY_MISMATCH:{group_id}")
            representative_id = str(group.get("representative_candidate_id") or "")
            if representative_id not in catalog_ids:
                errors.append(f"GROUP_REPRESENTATIVE_NOT_IN_CATALOG:{group_id}")

    if corroborated_catalog and review:
        strongest = max(
            corroborated_catalog,
            key=lambda row: float(row.get("group_confidence_score") or 0.0),
        )
        if str(strongest.get("candidate_id") or "") not in review_ids:
            errors.append("STRONGEST_CORROBORATION_GROUP_NOT_IN_ACTIVE_REVIEW")

    required = str(args.require_candidate_id or "")
    if required:
        if args.require_candidate_in_active_review:
            if required not in review_ids:
                errors.append(f"REQUIRED_CANDIDATE_NOT_IN_ACTIVE_REVIEW:{required}")
        elif required not in catalog_ids and required not in review_ids:
            errors.append(f"REQUIRED_CANDIDATE_NOT_IN_REVIEW_OR_CATALOG:{required}")

    _verify_file(
        attempt.get("ranked_candidates_path"),
        attempt.get("ranked_candidates_sha256"),
        "RANKED_CANDIDATES",
        errors,
    )
    _verify_file(
        attempt.get("review_catalog_path"),
        attempt.get("review_catalog_sha256"),
        "REVIEW_CATALOG",
        errors,
    )
    _verify_file(
        attempt.get("review_catalog_contact_sheet_path"),
        attempt.get("review_catalog_contact_sheet_sha256"),
        "REVIEW_CATALOG_CONTACT_SHEET",
        errors,
    )

    pending = dict(state.get("pending_action") or {})
    if review:
        if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
            errors.append("PENDING_ACTION_MISSING")
        pending_ids = [str(value) for value in pending.get("candidate_ids") or []]
        promoted = str(pending.get("manual_review_promoted_candidate_id") or "")
        if promoted:
            if pending.get("review_promotion_policy") != PROMOTION_POLICY:
                errors.append("PENDING_PROMOTION_POLICY_MISMATCH")
            if pending_ids != [promoted]:
                errors.append("PROMOTED_PENDING_REVIEW_SET_MISMATCH")
        elif set(pending_ids) != set(review_ids):
            errors.append("PENDING_REVIEW_SET_MISMATCH")
        pending_catalog_ids = [
            str(value) for value in pending.get("review_catalog_candidate_ids") or []
        ]
        if pending_catalog_ids and pending_catalog_ids != catalog_ids:
            errors.append("PENDING_CATALOG_SET_MISMATCH")
        if pending.get("automatic_target_confirmation") is not False:
            errors.append("PENDING_AUTO_CONFIRMATION_FORBIDDEN")

    status = "PASS" if not errors else "FAIL"
    print(f"status={status}")
    print(f"shot_id={attempt.get('shot_id')}")
    print(f"phase4b_policy={attempt.get('policy')}")
    print(f"selection_policy={attempt.get('assisted_review_selection_policy')}")
    print(f"review_candidate_count={len(review)}")
    print(f"review_candidate_ids={review_ids}")
    print(f"review_catalog_candidate_count={len(catalog)}")
    print(f"corroborated_review_candidate_ids={[row.get('candidate_id') for row in corroborated_review]}")
    print(f"manual_review_promotion_allowed={attempt.get('manual_review_promotion_allowed')}")
    print(f"pending_action_type={pending.get('type')}")
    print(f"automatic_target_confirmation={attempt.get('automatic_target_confirmation')}")
    print(f"errors={errors}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
