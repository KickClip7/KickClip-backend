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
COMBINED_POLICY = (
    "PROVENANCE_ROLE_USER_HARD_DETECTOR_SOFT_IDENTITY_ROBUST_R4"
)
IDENTITY_POLICY = "SCALE_COMPATIBLE_CLUSTER_ROBUST_IDENTITY_NEGATIVE_R1"
USER_ROLE_POLICY = (
    "USER_CONFIRMED_NON_PLAYER_SCALE_COMPATIBLE_CLUSTER_ROBUST_HARD_GATE_R1"
)
DETECTOR_ROLE_POLICY = "DETECTOR_LABELED_STAFF_REFEREE_SOFT_EVIDENCE_R1"
ROLE_POLICY = (
    "USER_CONFIRMED_ROLE_SCALE_COMPATIBLE_HARD_DETECTOR_ROLE_SOFT_R2"
)
SELECTION_POLICY = "CORROBORATION_GROUP_CONFIDENCE_REVIEW_CATALOG_R4"
RESCUE_POLICY = "GROUP_CONFIDENCE_ASSISTED_REVIEW_R5"
MAX_REVIEW = 3
MIN_PLAUSIBLE_SCORE = 0.45
MIN_PLAUSIBLE_PROTOTYPE = 0.45


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rows(value: object) -> list[dict[str, Any]]:
    return (
        [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(value, list)
        else []
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _resolve_review_rows(
    selected: Mapping[str, Any],
    ranked: Mapping[str, Any],
    errors: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Resolve the immutable review set across R9 and corrected R9.1 reports.

    Early R9 attempt reports recorded review_candidate_count but omitted the
    review_candidates payload. The ranked artifact and pending action still
    contained the exact immutable candidate set. R9.1 writes the payload into
    every new attempt report and accepts the old artifact only when all three
    independent candidate-ID sources agree.
    """

    declared_count = int(selected.get("review_candidate_count") or 0)
    attempt_rows = _rows(selected.get("review_candidates"))
    ranked_rows = _rows(ranked.get("reviewable_candidates"))

    if attempt_rows:
        rows = attempt_rows
        source = "ATTEMPT_REPORT"
    elif declared_count > 0:
        rows = ranked_rows
        source = "RANKED_CANDIDATES_COMPAT_BACKFILL"
    else:
        rows = []
        source = "NONE"

    if len(rows) != declared_count:
        errors.append("REVIEW_CANDIDATE_COUNT_PAYLOAD_MISMATCH")

    row_ids = {str(row.get("candidate_id") or "") for row in rows}
    ranked_ids = {str(row.get("candidate_id") or "") for row in ranked_rows}
    if row_ids != ranked_ids:
        errors.append("RANKED_REVIEW_SET_MISMATCH")

    safe_gate = dict(selected.get("safe_gate") or {})
    declared_ids = {
        str(value)
        for value in (
            selected.get("selected_review_candidate_ids")
            or safe_gate.get("selected_review_candidate_ids")
            or []
        )
    }
    if declared_ids and declared_ids != row_ids:
        errors.append("DECLARED_REVIEW_SET_MISMATCH")

    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--shot-id", default=None)
    parser.add_argument("--require-review-candidate", action="store_true")
    args = parser.parse_args()

    root = args.job_root.resolve()
    state = _read(root / "pipeline_state.json")
    report = _read(root / "phase4b_first_cross_shot_report.json")
    attempts = _rows(report.get("attempts")) or [report]
    selected = attempts[-1]
    if args.shot_id:
        selected = next(
            (
                row
                for row in attempts
                if str(row.get("shot_id") or "") == args.shot_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Shot attempt not found: {args.shot_id}")

    errors: list[str] = []
    if report.get("policy") != PHASE4B_POLICY:
        errors.append("REPORT_POLICY_MISMATCH")
    if selected.get("policy") != PHASE4B_POLICY:
        errors.append("ATTEMPT_POLICY_MISMATCH")
    if selected.get("negative_memory_policy") != COMBINED_POLICY:
        errors.append("COMBINED_POLICY_MISMATCH")
    if selected.get("identity_negative_scoring_policy") != IDENTITY_POLICY:
        errors.append("IDENTITY_POLICY_MISMATCH")
    if selected.get("user_confirmed_role_scoring_policy") != USER_ROLE_POLICY:
        errors.append("USER_ROLE_POLICY_MISMATCH")
    if selected.get("detector_role_scoring_policy") != DETECTOR_ROLE_POLICY:
        errors.append("DETECTOR_ROLE_POLICY_MISMATCH")
    if selected.get("role_negative_scoring_policy") != ROLE_POLICY:
        errors.append("ROLE_POLICY_MISMATCH")
    if selected.get("assisted_review_selection_policy") != SELECTION_POLICY:
        errors.append("SELECTION_POLICY_MISMATCH")
    if selected.get("identity_purity_policy") != PURITY_POLICY:
        errors.append("IDENTITY_PURITY_POLICY_MISMATCH")
    if selected.get("detector_role_evidence_is_soft_only") is not True:
        errors.append("DETECTOR_ROLE_NOT_SOFT_ONLY")
    if selected.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")
    if selected.get("candidate_link_created") is not False:
        errors.append("CANDIDATE_LINK_CREATED_BEFORE_REVIEW")

    ranked_path = Path(str(selected.get("ranked_candidates_path") or "")).resolve()
    expected_sha = str(selected.get("ranked_candidates_sha256") or "")
    ranked: dict[str, Any] = {}
    if not ranked_path.is_file():
        errors.append(f"RANKED_CANDIDATES_MISSING:{ranked_path}")
    elif len(expected_sha) != 64 or _sha(ranked_path) != expected_sha:
        errors.append("RANKED_CANDIDATES_SHA_MISMATCH")
    else:
        ranked = _read(ranked_path)
        if ranked.get("assisted_review_selection_policy") != SELECTION_POLICY:
            errors.append("RANKED_SELECTION_POLICY_MISMATCH")
        if ranked.get("detector_role_evidence_is_soft_only") is not True:
            errors.append("RANKED_DETECTOR_ROLE_NOT_SOFT_ONLY")

    review_rows, review_candidate_source = _resolve_review_rows(
        selected, ranked, errors
    )
    rescue_rows = _rows(selected.get("rescue_review_candidates"))
    if len(review_rows) > MAX_REVIEW:
        errors.append("REVIEW_LIMIT_EXCEEDED")
    if args.require_review_candidate and not review_rows:
        errors.append("REVIEW_CANDIDATE_REQUIRED_BUT_MISSING")

    review_ids = {str(row.get("candidate_id") or "") for row in review_rows}
    rescue_ids = {str(row.get("candidate_id") or "") for row in rescue_rows}
    if not rescue_ids.issubset(review_ids):
        errors.append("RESCUE_NOT_SUBSET_OF_REVIEW")

    for row in review_rows:
        candidate_id = str(row.get("candidate_id") or "")
        gate = dict(row.get("negative_review_gate") or {})
        user_role_gate = dict(gate.get("user_confirmed_role_gate") or {})
        detector_gate = dict(gate.get("detector_role_soft_gate") or {})
        identity_gate = dict(gate.get("identity_negative_gate") or {})
        observability = dict(row.get("identity_observability") or {})
        purity = dict(row.get("identity_purity") or {})
        if observability.get("passed") is not True:
            errors.append(f"OBSERVABILITY_NOT_PASSED:{candidate_id}")
        if purity.get("policy") != PURITY_POLICY:
            errors.append(f"PURITY_POLICY_MISMATCH:{candidate_id}")
        if purity.get("passed") is not True:
            errors.append(f"IDENTITY_PURITY_NOT_PASSED:{candidate_id}")
        if purity.get("target_memory_used_for_segmentation") is not False:
            errors.append(f"TARGET_MEMORY_USED_FOR_SEGMENTATION:{candidate_id}")
        if user_role_gate.get("passed") is not True:
            errors.append(f"USER_ROLE_HARD_GATE_FAILED:{candidate_id}")
        if identity_gate.get("passed") is not True:
            errors.append(f"IDENTITY_GATE_FAILED:{candidate_id}")
        if detector_gate.get("passed") is not True:
            errors.append(f"DETECTOR_ROLE_SOFT_GATE_REJECTED:{candidate_id}")
        if float(row.get("retrieval_score") or 0.0) < MIN_PLAUSIBLE_SCORE:
            errors.append(f"BELOW_PLAUSIBLE_RETRIEVAL:{candidate_id}")
        if (
            float(row.get("prototype_target_similarity") or 0.0)
            < MIN_PLAUSIBLE_PROTOTYPE
        ):
            errors.append(f"BELOW_PLAUSIBLE_PROTOTYPE:{candidate_id}")
        if row.get("assisted_review_selection_policy") != SELECTION_POLICY:
            errors.append(f"CANDIDATE_SELECTION_POLICY_MISMATCH:{candidate_id}")
        if row.get("automatic_target_confirmation") is True:
            errors.append(f"CANDIDATE_AUTO_CONFIRMATION_FORBIDDEN:{candidate_id}")
        group_ids = [str(value) for value in row.get("continuation_group_member_ids") or []]
        if candidate_id not in group_ids:
            errors.append(f"CANDIDATE_NOT_IN_OWN_CONTINUATION_GROUP:{candidate_id}")
        if row.get("rescue_review_required") is True:
            if row.get("rescue_review_policy") != RESCUE_POLICY:
                errors.append(f"RESCUE_POLICY_MISMATCH:{candidate_id}")

    pending = dict(state.get("pending_action") or {})
    if review_rows:
        if state.get("status") != "NEEDS_CONFIRMATION":
            errors.append("STATE_NOT_WAITING_FOR_CONFIRMATION")
        if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
            errors.append("PENDING_ACTION_MISSING")
        pending_ids = {str(value) for value in pending.get("candidate_ids") or []}
        if pending_ids != review_ids:
            errors.append("PENDING_CANDIDATE_SET_MISMATCH")
        if pending.get("automatic_target_confirmation") is not False:
            errors.append("PENDING_AUTO_CONFIRMATION_FORBIDDEN")

    status = "PASS" if not errors else "FAIL"
    print(f"status={status}")
    print(f"shot_id={selected.get('shot_id')}")
    print(f"phase4b_policy={selected.get('policy')}")
    print(f"combined_negative_policy={selected.get('negative_memory_policy')}")
    print(
        "user_confirmed_role_scoring_policy="
        f"{selected.get('user_confirmed_role_scoring_policy')}"
    )
    print(
        "detector_role_scoring_policy="
        f"{selected.get('detector_role_scoring_policy')}"
    )
    print(
        "detector_role_evidence_is_soft_only="
        f"{selected.get('detector_role_evidence_is_soft_only')}"
    )
    print(
        "assisted_review_selection_policy="
        f"{selected.get('assisted_review_selection_policy')}"
    )
    print(f"identity_purity_policy={selected.get('identity_purity_policy')}")
    print(
        "split_parent_tracklet_count="
        f"{selected.get('split_parent_tracklet_count')}"
    )
    print(f"review_candidate_count={len(review_rows)}")
    print(f"review_candidate_source={review_candidate_source}")
    print(f"review_candidate_ids={sorted(review_ids)}")
    print(f"rescue_review_candidate_ids={sorted(rescue_ids)}")
    print(f"continuation_group_count={selected.get('continuation_group_count')}")
    print(f"pending_action_type={pending.get('type')}")
    print(
        "automatic_target_confirmation="
        f"{selected.get('automatic_target_confirmation')}"
    )
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
