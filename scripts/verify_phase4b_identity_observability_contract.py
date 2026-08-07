from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


POLICY = "SINGLE_PERSON_IDENTITY_OBSERVABILITY_R1"
GROUP_STATUS = "SEARCH_EXHAUSTED_UNREVIEWABLE_GROUP_OCCLUSION"
VALID_REJECTED_CLASSIFICATIONS = {
    "UNREVIEWABLE_GROUP_OCCLUSION",
    "UNREVIEWABLE_LOW_IDENTITY_OBSERVABILITY",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def rows(value: object) -> list[dict[str, Any]]:
    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--shot-id", default=None)
    args = parser.parse_args()

    root = args.job_root.resolve()
    state = read_object(root / "pipeline_state.json")
    report = read_object(root / "phase4b_first_cross_shot_report.json")
    attempts = rows(report.get("attempts")) or [report]
    selected = None
    if args.shot_id:
        selected = next(
            (row for row in attempts if str(row.get("shot_id") or "") == args.shot_id),
            None,
        )
        if selected is None:
            raise ValueError(f"Shot attempt not found: {args.shot_id}")
    else:
        selected = attempts[-1]

    errors: list[str] = []
    shot_id = str(selected.get("shot_id") or "")
    if selected.get("identity_observability_policy") != POLICY:
        errors.append("IDENTITY_OBSERVABILITY_POLICY_MISMATCH")
    rejected = rows(selected.get("rejected_identity_observability_candidates"))
    for candidate in rejected:
        candidate_id = str(candidate.get("candidate_id") or "")
        evidence = dict(candidate.get("identity_observability") or {})
        if evidence.get("policy") != POLICY:
            errors.append(f"CANDIDATE_POLICY_MISMATCH:{candidate_id}")
        if evidence.get("passed") is not False:
            errors.append(f"REJECTED_CANDIDATE_PASSED:{candidate_id}")
        if str(evidence.get("classification") or "") not in VALID_REJECTED_CLASSIFICATIONS:
            errors.append(f"CANDIDATE_CLASSIFICATION_INVALID:{candidate_id}")
        if int(evidence.get("clean_frame_count") or 0) >= int(
            evidence.get("minimum_clean_frame_count") or 0
        ) and float(evidence.get("clean_frame_ratio") or 0.0) >= float(
            evidence.get("minimum_clean_frame_ratio") or 0.0
        ):
            errors.append(f"REJECTED_CANDIDATE_CLEAN_GATE_INCONSISTENT:{candidate_id}")
        candidate_dir = root / "phase4b_cross_shot" / shot_id / "candidates" / candidate_id
        if (candidate_dir / "candidate_embeddings.npy").exists():
            errors.append(f"REJECTED_CANDIDATE_WAS_EMBEDDED:{candidate_id}")
        if (candidate_dir / "candidate_prototype.npy").exists():
            errors.append(f"REJECTED_CANDIDATE_HAS_PROTOTYPE:{candidate_id}")

    ranked_path = Path(str(selected.get("ranked_candidates_path") or "")).resolve()
    if not ranked_path.is_file():
        errors.append(f"RANKED_CANDIDATES_MISSING:{ranked_path}")
    else:
        expected_sha = str(selected.get("ranked_candidates_sha256") or "")
        if len(expected_sha) != 64 or sha256_file(ranked_path) != expected_sha:
            errors.append("RANKED_CANDIDATES_SHA_MISMATCH")
        ranked = read_object(ranked_path)
        if ranked.get("identity_observability_policy") != POLICY:
            errors.append("RANKED_OBSERVABILITY_POLICY_MISMATCH")
        for candidate in rows(ranked.get("reviewable_candidates")):
            evidence = dict(candidate.get("identity_observability") or {})
            if evidence.get("passed") is not True:
                errors.append(
                    f"REVIEWABLE_CANDIDATE_OBSERVABILITY_NOT_PASSED:{candidate.get('candidate_id')}"
                )
            if int(candidate.get("embedded_crop_count") or 0) > int(
                evidence.get("clean_frame_count") or 0
            ):
                errors.append(
                    f"EMBEDDED_CROPS_EXCEED_CLEAN_FRAMES:{candidate.get('candidate_id')}"
                )

    shots = {
        str(row.get("shot_id") or ""): row
        for row in rows(state.get("shots"))
    }
    exhaustion_reason = str(selected.get("exhaustion_reason") or "")
    observed_status = str(shots.get(shot_id, {}).get("status") or "")
    if exhaustion_reason == "UNREVIEWABLE_GROUP_OCCLUSION":
        if observed_status != GROUP_STATUS:
            errors.append("GROUP_OCCLUSION_SHOT_STATUS_MISMATCH")
        if not rejected:
            errors.append("GROUP_OCCLUSION_WITHOUT_REJECTED_CANDIDATES")
        if any(
            str((row.get("identity_observability") or {}).get("classification") or "")
            != "UNREVIEWABLE_GROUP_OCCLUSION"
            for row in rejected
        ):
            errors.append("GROUP_OCCLUSION_ATTEMPT_CONTAINS_OTHER_CLASSIFICATION")

    pending = dict(state.get("pending_action") or {})
    if exhaustion_reason == "UNREVIEWABLE_GROUP_OCCLUSION" and pending.get("shot_id") == shot_id:
        errors.append("GROUP_OCCLUSION_SHOT_LEFT_PENDING")
    if selected.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")

    status = "PASS" if not errors else "FAIL"
    print(f"status={status}")
    print(f"shot_id={shot_id}")
    print(f"identity_observability_policy={selected.get('identity_observability_policy')}")
    print(f"raw_candidate_count={selected.get('identity_observability_raw_candidate_count')}")
    print(f"passed_candidate_count={selected.get('identity_observability_passed_candidate_count')}")
    print(f"rejected_candidate_count={len(rejected)}")
    print(f"exhaustion_reason={selected.get('exhaustion_reason')}")
    print(f"shot_status={observed_status}")
    print(f"automatic_target_confirmation={selected.get('automatic_target_confirmation')}")
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
