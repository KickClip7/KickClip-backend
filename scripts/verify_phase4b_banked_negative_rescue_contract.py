from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PHASE4B_POLICY = (
    "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R8_BANKED_NEGATIVE_RESCUE"
)
COMBINED_POLICY = "BANKED_ROLE_HARD_IDENTITY_ROBUST_RESCUE_R3"
IDENTITY_POLICY = "SCALE_COMPATIBLE_CLUSTER_ROBUST_IDENTITY_NEGATIVE_R1"
ROLE_POLICY = "PERSISTENT_AND_SAME_SHOT_ROLE_NEGATIVE_HARD_GATE_R1"
RESCUE_POLICY = "IDENTITY_NEGATIVE_FALSE_REJECT_RESCUE_R1"
MAX_RESCUE = 3


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--shot-id", default=None)
    args = parser.parse_args()

    root = args.job_root.resolve()
    state = _read(root / "pipeline_state.json")
    report = _read(root / "phase4b_first_cross_shot_report.json")
    attempts = _rows(report.get("attempts")) or [report]
    selected = attempts[-1]
    if args.shot_id:
        selected = next(
            (
                attempt
                for attempt in attempts
                if str(attempt.get("shot_id") or "") == args.shot_id
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
        errors.append("IDENTITY_SCORING_POLICY_MISMATCH")
    if selected.get("role_negative_scoring_policy") != ROLE_POLICY:
        errors.append("ROLE_SCORING_POLICY_MISMATCH")
    if selected.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")
    if selected.get("candidate_link_created") is not False:
        errors.append("CANDIDATE_LINK_CREATED_BEFORE_REVIEW")

    rescue = _rows(selected.get("rescue_review_candidates"))
    rescue_count = int(selected.get("rescue_review_candidate_count") or 0)
    if rescue_count != len(rescue):
        errors.append("RESCUE_COUNT_MISMATCH")
    if rescue_count > MAX_RESCUE:
        errors.append("RESCUE_LIMIT_EXCEEDED")
    if rescue_count and selected.get("rescue_review_policy") != RESCUE_POLICY:
        errors.append("RESCUE_POLICY_MISMATCH")

    for row in rescue:
        candidate_id = str(row.get("candidate_id") or "")
        gate = dict(row.get("negative_review_gate") or {})
        role_gate = dict(gate.get("role_negative_gate") or {})
        identity_gate = dict(gate.get("identity_negative_gate") or {})
        if row.get("rescue_review_required") is not True:
            errors.append(f"RESCUE_FLAG_MISSING:{candidate_id}")
        if row.get("rescue_review_policy") != RESCUE_POLICY:
            errors.append(f"RESCUE_POLICY_INVALID:{candidate_id}")
        if gate.get("passed") is not False:
            errors.append(f"RESCUE_COMBINED_GATE_NOT_FAILED:{candidate_id}")
        if role_gate.get("passed") is not True:
            errors.append(f"RESCUE_ROLE_GATE_NOT_PASSED:{candidate_id}")
        if identity_gate.get("passed") is not False:
            errors.append(f"RESCUE_IDENTITY_GATE_NOT_FAILED:{candidate_id}")
        if row.get("automatic_target_confirmation") is True:
            errors.append(f"RESCUE_AUTO_CONFIRMATION_FORBIDDEN:{candidate_id}")

    pending = dict(state.get("pending_action") or {})
    if rescue_count:
        if state.get("status") != "NEEDS_CONFIRMATION":
            errors.append("STATE_NOT_WAITING_FOR_RESCUE_REVIEW")
        if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
            errors.append("RESCUE_PENDING_ACTION_MISSING")
        pending_ids = {str(value) for value in pending.get("candidate_ids") or []}
        rescue_ids = {str(row.get("candidate_id") or "") for row in rescue}
        if not rescue_ids.issubset(pending_ids):
            errors.append("RESCUE_IDS_NOT_IN_PENDING_ACTION")

    ranked_path = Path(str(selected.get("ranked_candidates_path") or "")).resolve()
    expected_sha = str(selected.get("ranked_candidates_sha256") or "")
    if not ranked_path.is_file():
        errors.append(f"RANKED_CANDIDATES_MISSING:{ranked_path}")
    elif len(expected_sha) != 64 or _sha(ranked_path) != expected_sha:
        errors.append("RANKED_CANDIDATES_SHA_MISMATCH")
    else:
        ranked = _read(ranked_path)
        ranked_rescue = _rows(ranked.get("rescue_review_candidates"))
        if {
            str(row.get("candidate_id") or "") for row in ranked_rescue
        } != {
            str(row.get("candidate_id") or "") for row in rescue
        }:
            errors.append("RANKED_RESCUE_SET_MISMATCH")

    reopened = [
        str(value)
        for value in report.get("reopened_stale_terminal_shot_ids") or []
    ]
    attempted = [
        str(value)
        for value in report.get("attempted_shot_ids") or []
    ]
    if reopened and not set(reopened).issubset(set(attempted)):
        errors.append("REOPENED_SHOT_NOT_REATTEMPTED")

    status = "PASS" if not errors else "FAIL"
    print(f"status={status}")
    print(f"shot_id={selected.get('shot_id')}")
    print(f"phase4b_policy={selected.get('policy')}")
    print(f"combined_negative_policy={selected.get('negative_memory_policy')}")
    print(
        "identity_negative_scoring_policy="
        f"{selected.get('identity_negative_scoring_policy')}"
    )
    print(
        "role_negative_scoring_policy="
        f"{selected.get('role_negative_scoring_policy')}"
    )
    print(f"rescue_review_policy={selected.get('rescue_review_policy')}")
    print(f"rescue_review_candidate_count={rescue_count}")
    print(
        "rescue_review_candidate_ids="
        f"{[row.get('candidate_id') for row in rescue]}"
    )
    print(f"reopened_stale_terminal_shot_ids={reopened}")
    print(f"pending_action_type={pending.get('type')}")
    print(
        "automatic_target_confirmation="
        f"{selected.get('automatic_target_confirmation')}"
    )
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
