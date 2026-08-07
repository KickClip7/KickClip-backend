from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPORT_SCHEMA = "kickclip.phase4b_multi_shot_search_report.v1"
ATTEMPT_SCHEMA = "kickclip.phase4b_shot_search_attempt.v1"
PHASE4B_POLICY = (
    "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_"
    "R12_GROUP_CONFIDENCE_REVIEW_CATALOG"
)
ROLE_FILTER_POLICY = "PLAYER_GOALKEEPER_ONLY_R2_WITH_ROLE_NEGATIVES"
OBSERVABILITY_POLICY = "SINGLE_PERSON_IDENTITY_OBSERVABILITY_R1"
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
TARGET_CONTEXT_POLICY = "APPROVED_REFERENCE_TORSO_COLOR_SOFT_EVIDENCE_R1"
CORROBORATION_POLICY = "INDEPENDENT_TRACKLET_SPATIOTEMPORAL_APPEARANCE_CORROBORATION_R1"
MAX_REVIEW = 3


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




def _resolve_review_rows(
    attempt: Mapping[str, Any],
    ranked: Mapping[str, Any],
    errors: list[str],
) -> tuple[list[dict[str, Any]], str]:
    declared_count = int(attempt.get("review_candidate_count") or 0)
    attempt_rows = _rows(attempt.get("review_candidates"))
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
    safe_gate = dict(attempt.get("safe_gate") or {})
    declared_ids = {
        str(value)
        for value in (
            attempt.get("selected_review_candidate_ids")
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
    args = parser.parse_args()

    root = args.job_root.resolve()
    state = _read(root / "pipeline_state.json")
    report = _read(root / "phase4b_first_cross_shot_report.json")
    attempts = _rows(report.get("attempts")) or [report]
    latest = attempts[-1]
    errors: list[str] = []

    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append("REPORT_SCHEMA_MISMATCH")
    if report.get("policy") != PHASE4B_POLICY:
        errors.append("REPORT_POLICY_MISMATCH")
    if report.get("status") != "PASS":
        errors.append("REPORT_STATUS_NOT_PASS")
    if report.get("automatic_target_confirmation") is not False:
        errors.append("REPORT_AUTO_CONFIRMATION_FORBIDDEN")
    if report.get("candidate_link_created") is not False:
        errors.append("REPORT_CANDIDATE_LINK_FORBIDDEN")

    attempted_ids = [str(value) for value in report.get("attempted_shot_ids") or []]
    attempt_ids = [str(row.get("shot_id") or "") for row in attempts]
    if attempted_ids != attempt_ids:
        errors.append("ATTEMPT_ID_SEQUENCE_MISMATCH")
    generations = [int(row.get("candidate_scoring_generation") or 0) for row in attempts]
    if generations != sorted(generations) or len(generations) != len(set(generations)):
        errors.append("GENERATION_SEQUENCE_INVALID")

    for attempt in attempts:
        if attempt.get("schema_version") != ATTEMPT_SCHEMA:
            errors.append("ATTEMPT_SCHEMA_MISMATCH")
        if attempt.get("policy") != PHASE4B_POLICY:
            errors.append("ATTEMPT_POLICY_MISMATCH")
        if attempt.get("status") != "PASS":
            errors.append("ATTEMPT_STATUS_NOT_PASS")
        if attempt.get("negative_memory_policy") != COMBINED_POLICY:
            errors.append("ATTEMPT_COMBINED_POLICY_MISMATCH")
        if attempt.get("identity_negative_scoring_policy") != IDENTITY_POLICY:
            errors.append("ATTEMPT_IDENTITY_POLICY_MISMATCH")
        if attempt.get("user_confirmed_role_scoring_policy") != USER_ROLE_POLICY:
            errors.append("ATTEMPT_USER_ROLE_POLICY_MISMATCH")
        if attempt.get("detector_role_scoring_policy") != DETECTOR_ROLE_POLICY:
            errors.append("ATTEMPT_DETECTOR_ROLE_POLICY_MISMATCH")
        if attempt.get("role_negative_scoring_policy") != ROLE_POLICY:
            errors.append("ATTEMPT_ROLE_POLICY_MISMATCH")
        if attempt.get("assisted_review_selection_policy") != SELECTION_POLICY:
            errors.append("ATTEMPT_SELECTION_POLICY_MISMATCH")
        if attempt.get("identity_purity_policy") != PURITY_POLICY:
            errors.append("ATTEMPT_IDENTITY_PURITY_POLICY_MISMATCH")
        if attempt.get("target_context_policy") != TARGET_CONTEXT_POLICY:
            errors.append("ATTEMPT_TARGET_CONTEXT_POLICY_MISMATCH")
        if attempt.get("target_context_is_soft_evidence_only") is not True:
            errors.append("ATTEMPT_TARGET_CONTEXT_NOT_SOFT")
        if attempt.get("corroboration_policy") != CORROBORATION_POLICY:
            errors.append("ATTEMPT_CORROBORATION_POLICY_MISMATCH")
        if int(attempt.get("corroboration_group_count") or 0) != len(
            _rows(attempt.get("corroboration_groups"))
        ):
            errors.append("ATTEMPT_CORROBORATION_COUNT_MISMATCH")
        parent_summaries = _rows(attempt.get("parent_tracklet_summaries"))
        if int(attempt.get("parent_tracklet_count_before_purity_segmentation") or 0) != len(parent_summaries):
            errors.append("ATTEMPT_PARENT_TRACKLET_COUNT_MISMATCH")
        if int(attempt.get("split_parent_tracklet_count") or 0) != sum(
            row.get("split_parent") is True for row in parent_summaries
        ):
            errors.append("ATTEMPT_SPLIT_PARENT_COUNT_MISMATCH")
        if int(attempt.get("rejected_identity_purity_segment_count") or 0) != len(
            _rows(attempt.get("rejected_identity_purity_segments"))
        ):
            errors.append("ATTEMPT_REJECTED_PURITY_COUNT_MISMATCH")
        if attempt.get("detector_role_evidence_is_soft_only") is not True:
            errors.append("ATTEMPT_DETECTOR_ROLE_NOT_SOFT")
        if attempt.get("automatic_target_confirmation") is not False:
            errors.append("ATTEMPT_AUTO_CONFIRMATION_FORBIDDEN")
        if attempt.get("candidate_link_created") is not False:
            errors.append("ATTEMPT_CANDIDATE_LINK_FORBIDDEN")
        if len(_rows(attempt.get("review_candidates"))) > MAX_REVIEW:
            errors.append("ATTEMPT_REVIEW_LIMIT_EXCEEDED")

    if latest.get("identity_observability_policy") != OBSERVABILITY_POLICY:
        errors.append("OBSERVABILITY_POLICY_MISMATCH")
    role_filter = dict(latest.get("candidate_role_filter") or {})
    if role_filter.get("policy_version") != ROLE_FILTER_POLICY:
        errors.append("ROLE_FILTER_POLICY_MISMATCH")
    if role_filter.get("class_mapping_validated") is not True:
        errors.append("ROLE_CLASS_MAPPING_NOT_VALIDATED")

    ranked_path = Path(str(latest.get("ranked_candidates_path") or "")).resolve()
    ranked: dict[str, Any] = {}
    expected_ranked_sha = str(latest.get("ranked_candidates_sha256") or "")
    if ranked_path.is_file() and len(expected_ranked_sha) == 64:
        if _sha(ranked_path) == expected_ranked_sha:
            ranked = _read(ranked_path)

    review_rows, review_candidate_source = _resolve_review_rows(
        latest, ranked, errors
    )
    for row in review_rows:
        candidate_id = str(row.get("candidate_id") or "")
        gate = dict(row.get("negative_review_gate") or {})
        if dict(row.get("identity_observability") or {}).get("passed") is not True:
            errors.append(f"REVIEW_OBSERVABILITY_FAILED:{candidate_id}")
        purity = dict(row.get("identity_purity") or {})
        if purity.get("policy") != PURITY_POLICY:
            errors.append(f"REVIEW_PURITY_POLICY_MISMATCH:{candidate_id}")
        if purity.get("passed") is not True:
            errors.append(f"REVIEW_IDENTITY_PURITY_FAILED:{candidate_id}")
        if purity.get("target_memory_used_for_segmentation") is not False:
            errors.append(f"REVIEW_TARGET_MEMORY_USED_FOR_SEGMENTATION:{candidate_id}")
        if dict(gate.get("user_confirmed_role_gate") or {}).get("passed") is not True:
            errors.append(f"REVIEW_USER_ROLE_GATE_FAILED:{candidate_id}")
        if dict(gate.get("identity_negative_gate") or {}).get("passed") is not True:
            errors.append(f"REVIEW_IDENTITY_GATE_FAILED:{candidate_id}")
        if dict(gate.get("detector_role_soft_gate") or {}).get("passed") is not True:
            errors.append(f"REVIEW_DETECTOR_SOFT_GATE_REJECTED:{candidate_id}")
        if row.get("target_context_policy") != TARGET_CONTEXT_POLICY:
            errors.append(f"REVIEW_TARGET_CONTEXT_POLICY_MISMATCH:{candidate_id}")
        similarity = float(row.get("target_context_similarity") or 0.0)
        if not 0.0 <= similarity <= 1.0:
            errors.append(f"REVIEW_TARGET_CONTEXT_RANGE_INVALID:{candidate_id}")
        if row.get("target_context_is_soft_evidence_only") is not True:
            errors.append(f"REVIEW_TARGET_CONTEXT_NOT_SOFT:{candidate_id}")
        if row.get("corroboration_policy") != CORROBORATION_POLICY:
            errors.append(f"REVIEW_CORROBORATION_POLICY_MISMATCH:{candidate_id}")

    _verify_file(
        latest.get("ranked_candidates_path"),
        latest.get("ranked_candidates_sha256"),
        "RANKED_CANDIDATES",
        errors,
    )
    _verify_file(
        latest.get("contact_sheet_path"),
        latest.get("contact_sheet_sha256"),
        "CONTACT_SHEET",
        errors,
    )
    _verify_file(
        latest.get("target_embeddings_path"),
        latest.get("target_embeddings_sha256"),
        "TARGET_EMBEDDINGS",
        errors,
    )
    _verify_file(
        latest.get("target_prototype_path"),
        latest.get("target_prototype_sha256"),
        "TARGET_PROTOTYPE",
        errors,
    )

    pending = dict(state.get("pending_action") or {})
    if review_rows:
        if state.get("status") != "NEEDS_CONFIRMATION":
            errors.append("STATE_NOT_NEEDS_CONFIRMATION")
        if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
            errors.append("PENDING_ACTION_MISSING")
        review_ids = {str(row.get("candidate_id") or "") for row in review_rows}
        pending_ids = {str(value) for value in pending.get("candidate_ids") or []}
        if pending_ids != review_ids:
            errors.append("PENDING_REVIEW_IDS_MISMATCH")
    else:
        if state.get("pending_action") is not None:
            errors.append("PENDING_ACTION_WITHOUT_REVIEW_CANDIDATES")

    status = "PASS" if not errors else "FAIL"
    print(f"status={status}")
    print(f"decision={report.get('decision')}")
    print(f"attempted_shot_ids={attempted_ids}")
    print(f"exhausted_shot_ids={report.get('exhausted_shot_ids')}")
    print(f"stop_reason={report.get('stop_reason')}")
    print(f"shot_id={latest.get('shot_id')}")
    print(f"memory_revision_id={latest.get('memory_revision_id')}")
    print(
        "candidate_scoring_generation="
        f"{latest.get('candidate_scoring_generation')}"
    )
    print(f"candidate_count={latest.get('candidate_count')}")
    print(f"review_candidate_count={len(review_rows)}")
    print(f"review_candidate_source={review_candidate_source}")
    print(
        "assisted_review_selection_policy="
        f"{latest.get('assisted_review_selection_policy')}"
    )
    print(f"identity_purity_policy={latest.get('identity_purity_policy')}")
    print(
        "parent_tracklet_count_before_purity_segmentation="
        f"{latest.get('parent_tracklet_count_before_purity_segmentation')}"
    )
    print(
        "split_parent_tracklet_count="
        f"{latest.get('split_parent_tracklet_count')}"
    )
    print(
        "identity_segment_candidate_count="
        f"{latest.get('identity_segment_candidate_count')}"
    )
    print(
        "rejected_identity_purity_segment_count="
        f"{latest.get('rejected_identity_purity_segment_count')}"
    )
    print(
        "detector_role_evidence_is_soft_only="
        f"{latest.get('detector_role_evidence_is_soft_only')}"
    )
    print(
        "continuation_group_count="
        f"{latest.get('continuation_group_count')}"
    )
    print(f"target_context_policy={latest.get('target_context_policy')}")
    print(
        "target_context_is_soft_evidence_only="
        f"{latest.get('target_context_is_soft_evidence_only')}"
    )
    print(f"corroboration_policy={latest.get('corroboration_policy')}")
    print(
        "corroboration_group_count="
        f"{latest.get('corroboration_group_count')}"
    )
    print(
        "selected_review_candidate_ids="
        f"{[row.get('candidate_id') for row in review_rows]}"
    )
    print(f"pending_action_type={pending.get('type')}")
    print(
        "automatic_target_confirmation="
        f"{latest.get('automatic_target_confirmation')}"
    )
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
