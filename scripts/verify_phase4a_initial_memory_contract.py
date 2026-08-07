#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


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
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.job_root.expanduser().resolve()
    report_path = root / "phase4a_initial_target_memory_report.json"
    report = read_object(report_path)
    errors: list[str] = []
    if report.get("status") != "PASS":
        errors.append("report_status")
    if report.get("decision") != "AUTHORIZE_PHASE4A_INITIAL_MEMORY_REVIEW":
        errors.append("report_decision")
    if report.get("automatic_target_confirmation") is not False:
        errors.append("automatic_target_confirmation")
    if report.get("cross_shot_scoring_performed") is not False:
        errors.append("unexpected_cross_shot_scoring")

    memory_path = Path(str(report.get("memory_revision_path") or "")).resolve()
    expected_memory_sha = str(report.get("memory_revision_sha256") or "")
    if not memory_path.is_file() or sha256_file(memory_path) != expected_memory_sha:
        errors.append("memory_sha")
        memory: dict[str, Any] = {}
    else:
        memory = read_object(memory_path)
    references = rows(memory.get("references"))
    native = rows(memory.get("native_references"))
    active = rows(memory.get("active_references"))
    if len(native) < 3:
        errors.append("native_reference_count")
    if not active:
        errors.append("active_reference_count")
    if any(item.get("scoring_eligible") is not True for item in native):
        errors.append("native_scoring_eligibility")
    if any(item.get("scoring_eligible") is not False for item in active):
        errors.append("unreviewed_active_scoring_eligibility")
    if int(memory.get("reference_count") or -1) != len(references):
        errors.append("reference_count")
    if memory.get("automatic_target_confirmation") is not False:
        errors.append("memory_automatic_target_confirmation")
    if memory.get("cross_shot_scoring_performed") is not False:
        errors.append("memory_cross_shot_scoring")

    state_path = root / "pipeline_state.json"
    state = read_object(state_path) if state_path.is_file() else {}
    runtime = state.get("runtime") if isinstance(state.get("runtime"), Mapping) else {}
    state_integration_present = bool(
        state and runtime.get("phase4a_initial_memory_build_complete") is True
    )
    if (
        state_integration_present
        and str(runtime.get("memory_revision_sha256") or "") != expected_memory_sha
    ):
        errors.append("state_memory_sha")

    status = "PASS" if not errors else "FAIL"
    decision = (
        "AUTHORIZE_PHASE4A_MEMORY_REVIEW"
        if not errors
        else "BLOCK_PHASE4A_MEMORY_CONTRACT"
    )
    print(f"status={status}")
    print(f"decision={decision}")
    print(f"memory_revision_id={memory.get('memory_revision_id')}")
    print(f"native_reference_count={len(native)}")
    print(f"active_reference_count={len(active)}")
    print(f"total_reference_count={len(references)}")
    print("cross_shot_scoring_performed=False")
    print("automatic_target_confirmation=False")
    print(f"state_integration_present={state_integration_present}")
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
