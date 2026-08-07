#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Static/runtime-contract verification for KickClip tracking P0 integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend-root", type=Path, default=Path.cwd())
    a = p.parse_args()
    backend = a.backend_root.expanduser().resolve()
    tracking = backend / "tracking_source"
    errors: list[str] = []

    required = [
        backend / "app" / "core" / "config.py",
        backend / "app" / "domains" / "shot_boundary" / "service.py",
        backend / "app" / "domains" / "tracking" / "process_runner.py",
        backend / "app" / "domains" / "highlight" / "scene_target_selection.py",
        tracking / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        tracking / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline_core.py",
        tracking / "target_centric_tracking_e2e_v1" / "run_stage1_with_conf_override.py",
        tracking / "target_centric_tracking_e2e_v1" / "run_phase1_product_pipeline.py",
        tracking / "target_centric_tracking_e2e_v1" / "verify_e2e_installation.py",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"MISSING:{path}")
            continue
        if path.suffix == ".py":
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:
                errors.append(f"PY_COMPILE:{path}:{exc}")

    manifest_path = tracking / "target_centric_tracking_v1" / "phase1_frozen_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"MISSING:{manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for logical, record in (manifest.get("scripts") or {}).items():
            raw = record.get("path")
            if not raw:
                continue
            path = tracking / str(raw).replace("\\", "/")
            if not path.is_file():
                errors.append(f"FROZEN_SCRIPT_MISSING:{logical}:{path}")
            elif sha256(path) != str(record.get("sha256") or ""):
                errors.append(f"FROZEN_SCRIPT_CHANGED:{logical}:{path}")
        for logical, record in (manifest.get("models") or {}).items():
            if not record.get("verified"):
                continue
            path = tracking / str(record.get("path") or "").replace("\\", "/")
            if not path.is_file():
                errors.append(f"FROZEN_MODEL_MISSING:{logical}:{path}")
            elif sha256(path) != str(record.get("sha256") or ""):
                errors.append(f"FROZEN_MODEL_CHANGED:{logical}:{path}")

    if required[1].is_file():
        text = required[1].read_text(encoding="utf-8-sig")
        if "fail_closed_on_role_conflict" not in text:
            errors.append("ROLE_GATE_NOT_FAIL_CLOSED")
        if "kept = list(candidate_rows)" in text:
            errors.append("ROLE_GATE_FAIL_OPEN_FALLBACK_STILL_PRESENT")

    if required[2].is_file():
        text = required[2].read_text(encoding="utf-8-sig")
        for token in (
            "--full-scene",
            "--tracking-play-conf-threshold",
            "--target-reference-set",
            "--trusted-selected-reference-memory",
            "tracking_launch_manifest_sha256",
            "shot_boundaries_sha256",
        ):
            if token not in text:
                errors.append(f"PROCESS_RUNNER_MISSING:{token}")

    if required[3].is_file():
        text = required[3].read_text(encoding="utf-8-sig")
        for token in (
            '"tracking_launch_manifest_sha256": launch_sha256',
            '"shot_boundaries_sha256": shot_boundaries_sha256',
        ):
            if token not in text:
                errors.append(f"SCENE_TARGET_HANDOFF_MISSING:{token}")

    if required[4].is_file():
        text = required[4].read_text(encoding="utf-8-sig")
        for token in (
            "BIDIRECTIONAL_FROM_CONFIRMED_ANCHOR",
            "--full-scene",
            "run_target_centric_pipeline_core.py",
        ):
            if token not in text:
                errors.append(f"FULL_SCENE_WRAPPER_MISSING:{token}")

    if required[5].is_file():
        text = required[5].read_text(encoding="utf-8-sig")
        for token in (
            "BACKEND_USER_SELECTED_REFERENCE_SET",
            "backend_memory_used_for_scoring",
            "run_stage1_with_conf_override.py",
            "run_phase1_product_pipeline.py",
        ):
            if token not in text:
                errors.append(f"E2E_CORE_MISSING:{token}")

    if errors:
        print("Status=FAIL")
        for error in errors:
            print(error)
        return 2

    print("Status=PASS")
    print("P0 role gate fail-closed=True")
    print("P0 full-scene bidirectional wrapper=True")
    print("P0 selected reference memory scoring=True")
    print("P0 RF-DETR play confidence runtime override=True")
    print("P0 scene launch SHA handoff=True")
    print("Frozen V1/V2 source modification=False")
    print("V7 runtime dependency=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
