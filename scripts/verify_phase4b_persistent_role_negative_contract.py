from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POLICY = "USER_CONFIRMED_NON_PLAYER_PLUS_DETECTOR_ROLE_NEGATIVES_R1"
SCHEMA = "kickclip.phase4b_persistent_role_negative_memory.v1"
SHOT_STATUS = "SEARCH_EXHAUSTED_NON_PLAYER_ROLE"
COMBINED_POLICY = "CONFIRMED_IDENTITY_PLUS_PERSISTENT_AND_SAME_SHOT_ROLE_NEGATIVES_R2"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--ambiguity-id", default=None)
    args = parser.parse_args()

    job_root = args.job_root.resolve()
    state_path = job_root / "pipeline_state.json"
    report_path = job_root / "phase4b_first_cross_shot_report.json"
    state = read_object(state_path)
    report = read_object(report_path) if report_path.is_file() else {}
    runtime = state.get("runtime") if isinstance(state.get("runtime"), Mapping) else {}
    memory = (
        runtime.get("phase4b_persistent_role_negative_memory")
        if isinstance(runtime.get("phase4b_persistent_role_negative_memory"), Mapping)
        else {}
    )
    errors: list[str] = []

    if memory.get("policy") != POLICY:
        errors.append("persistent_role_policy")
    if not memory.get("revision_id"):
        errors.append("persistent_role_revision_id")
    if memory.get("user_confirmed_non_player_role") is not True:
        errors.append("persistent_role_user_confirmation")
    if memory.get("automatic_target_confirmation") is not False:
        errors.append("persistent_role_automatic_target_confirmation")
    if int(memory.get("embedding_count") or 0) < 1:
        errors.append("persistent_role_embedding_count")

    manifest_path = Path(str(memory.get("manifest_path") or "")).resolve()
    embeddings_path = Path(str(memory.get("embeddings_path") or "")).resolve()
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append("persistent_role_manifest_missing")
    elif sha256_file(manifest_path) != str(memory.get("manifest_sha256") or ""):
        errors.append("persistent_role_manifest_sha256")
    else:
        manifest = read_object(manifest_path)
        if manifest.get("schema_version") != SCHEMA:
            errors.append("persistent_role_manifest_schema")
        if manifest.get("policy") != POLICY:
            errors.append("persistent_role_manifest_policy")
        if manifest.get("decision") != "NONE_OF_THESE_NON_PLAYER_ROLE":
            errors.append("persistent_role_manifest_decision")
        if manifest.get("target_positive_memory_modified") is not False:
            errors.append("target_positive_memory_modified")
        if manifest.get("identity_negative_memory_modified") is not False:
            errors.append("identity_negative_memory_modified")
        if manifest.get("automatic_target_confirmation") is not False:
            errors.append("persistent_role_manifest_auto_confirmation")
        refs = [
            dict(row)
            for row in manifest.get("references") or []
            if isinstance(row, Mapping)
        ]
        if not any(
            row.get("source_kind") == "USER_CONFIRMED_NON_PLAYER_ROLE_CANDIDATE"
            and row.get("user_confirmed") is True
            for row in refs
        ):
            errors.append("user_confirmed_non_player_reference_missing")

    matrix = None
    if not embeddings_path.is_file():
        errors.append("persistent_role_embeddings_missing")
    elif sha256_file(embeddings_path) != str(memory.get("embeddings_sha256") or ""):
        errors.append("persistent_role_embeddings_sha256")
    else:
        with embeddings_path.open("rb") as stream:
            matrix = np.load(stream, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            errors.append("persistent_role_embeddings_shape")
        elif not np.isfinite(matrix).all():
            errors.append("persistent_role_embeddings_non_finite")
        elif int(matrix.shape[0]) != int(memory.get("embedding_count") or -1):
            errors.append("persistent_role_embeddings_count_mismatch")

    confirmations = [
        dict(row)
        for row in state.get("confirmations") or []
        if isinstance(row, Mapping)
        and row.get("decision") == "NONE_OF_THESE_NON_PLAYER_ROLE"
    ]
    if args.ambiguity_id:
        confirmations = [
            row
            for row in confirmations
            if str(row.get("ambiguity_id") or "") == args.ambiguity_id
        ]
    if not confirmations:
        errors.append("non_player_role_confirmation")
        confirmation = {}
    else:
        confirmation = confirmations[-1]

    shot_id = str(confirmation.get("shot_id") or memory.get("source_shot_id") or "")
    shot = next(
        (
            row
            for row in state.get("shots") or []
            if isinstance(row, Mapping) and str(row.get("shot_id") or "") == shot_id
        ),
        None,
    )
    if shot is None or str(shot.get("status") or "") != SHOT_STATUS:
        errors.append("non_player_role_shot_status")

    attempts = [
        dict(row)
        for row in report.get("attempts") or []
        if isinstance(row, Mapping)
    ]
    latest = attempts[-1] if attempts else report
    current_memory = (
        latest.get("persistent_role_negative_memory")
        if isinstance(latest.get("persistent_role_negative_memory"), Mapping)
        else {}
    )
    if report and current_memory.get("revision_id") != memory.get("revision_id"):
        errors.append("next_search_persistent_role_revision_mismatch")
    if report and latest.get("negative_memory_policy") != COMBINED_POLICY:
        errors.append("next_search_combined_negative_policy")
    if runtime.get("automatic_target_confirmation") is not False:
        errors.append("runtime_automatic_target_confirmation")

    pending = state.get("pending_action")
    print(f"status={'PASS' if not errors else 'FAIL'}")
    print(f"decision={state.get('decision')}")
    print(f"non_player_role_ambiguity_id={confirmation.get('ambiguity_id')}")
    print(f"non_player_role_shot_id={shot_id}")
    print(f"non_player_role_shot_status={shot.get('status') if shot else None}")
    print(f"persistent_role_policy={memory.get('policy')}")
    print(f"persistent_role_revision_id={memory.get('revision_id')}")
    print(f"persistent_role_embedding_count={memory.get('embedding_count')}")
    print(f"historical_detector_source_count={manifest.get('historical_detector_source_count')}")
    print(f"rejected_candidate_ids={memory.get('rejected_candidate_ids')}")
    print(f"next_shot_id={runtime.get('phase4b_next_shot_id')}")
    print(
        "pending_action_type="
        f"{pending.get('type') if isinstance(pending, Mapping) else None}"
    )
    print(f"automatic_target_confirmation={runtime.get('automatic_target_confirmation')}")
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
