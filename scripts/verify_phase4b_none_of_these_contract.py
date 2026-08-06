from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


POLICY = "USER_REJECTED_CROSS_SHOT_CANDIDATES_R1"
SHOT_STATUS = "SEARCH_EXHAUSTED_NONE_OF_THESE"


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
    state = read_object(state_path)
    runtime = state.get("runtime") if isinstance(state.get("runtime"), Mapping) else {}
    identity = (
        runtime.get("phase4b_identity_negative_memory")
        if isinstance(runtime.get("phase4b_identity_negative_memory"), Mapping)
        else {}
    )
    errors: list[str] = []

    if identity.get("policy") != POLICY:
        errors.append("identity_negative_policy")
    if not identity.get("revision_id"):
        errors.append("identity_negative_revision_id")
    if identity.get("user_confirmed_only") is not True:
        errors.append("identity_negative_user_confirmed_only")
    if identity.get("automatic_target_confirmation") is not False:
        errors.append("identity_negative_automatic_target_confirmation")
    if int(identity.get("embedding_count") or 0) < 1:
        errors.append("identity_negative_embedding_count")

    manifest_path = Path(str(identity.get("manifest_path") or "")).resolve()
    embeddings_path = Path(str(identity.get("embeddings_path") or "")).resolve()
    if not manifest_path.is_file():
        errors.append("identity_negative_manifest_missing")
    elif sha256_file(manifest_path) != str(identity.get("manifest_sha256") or ""):
        errors.append("identity_negative_manifest_sha256")
    if not embeddings_path.is_file():
        errors.append("identity_negative_embeddings_missing")
    elif sha256_file(embeddings_path) != str(identity.get("embeddings_sha256") or ""):
        errors.append("identity_negative_embeddings_sha256")

    confirmations = [
        dict(row)
        for row in state.get("confirmations") or []
        if isinstance(row, Mapping) and row.get("decision") == "NONE_OF_THESE"
    ]
    if args.ambiguity_id:
        confirmations = [
            row
            for row in confirmations
            if str(row.get("ambiguity_id") or "") == args.ambiguity_id
        ]
    if not confirmations:
        errors.append("none_of_these_confirmation")
        confirmation = {}
    else:
        confirmation = confirmations[-1]

    shot_id = str(confirmation.get("shot_id") or identity.get("source_shot_id") or "")
    shot = next(
        (
            row
            for row in state.get("shots") or []
            if isinstance(row, Mapping) and str(row.get("shot_id") or "") == shot_id
        ),
        None,
    )
    if shot is None or str(shot.get("status") or "") != SHOT_STATUS:
        errors.append("none_of_these_shot_status")
    if state.get("automatic_target_confirmation") is True:
        errors.append("state_automatic_target_confirmation")
    if runtime.get("automatic_target_confirmation") is not False:
        errors.append("runtime_automatic_target_confirmation")

    pending = state.get("pending_action")
    next_shot_id = runtime.get("phase4b_next_shot_id")
    print(f"status={'PASS' if not errors else 'FAIL'}")
    print(f"decision={state.get('decision')}")
    print(f"none_of_these_ambiguity_id={confirmation.get('ambiguity_id')}")
    print(f"none_of_these_shot_id={shot_id}")
    print(f"none_of_these_shot_status={shot.get('status') if shot else None}")
    print(f"identity_negative_policy={identity.get('policy')}")
    print(f"identity_negative_revision_id={identity.get('revision_id')}")
    print(f"identity_negative_embedding_count={identity.get('embedding_count')}")
    print(f"rejected_candidate_ids={identity.get('rejected_candidate_ids')}")
    print(f"next_shot_id={next_shot_id}")
    print(
        "pending_action_type="
        f"{pending.get('type') if isinstance(pending, Mapping) else None}"
    )
    print(f"automatic_target_confirmation={runtime.get('automatic_target_confirmation')}")
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
