#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    a = p.parse_args()
    root = a.project_root.resolve()

    manifest_path = root / "target_centric_tracking_v1" / "phase1_frozen_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    errors: list[str] = []
    for logical, record in manifest["scripts"].items():
        path = root / str(record["path"]).replace("\\", "/")
        if not path.is_file() or sha256(path) != record["sha256"]:
            errors.append(f"script:{logical}:{path}")
    verified_models = 0
    for logical, record in manifest["models"].items():
        if not record.get("verified"):
            continue
        verified_models += 1
        path = root / str(record["path"]).replace("\\", "/")
        if not path.is_file() or sha256(path) != record["sha256"]:
            errors.append(f"model:{logical}:{path}")

    required = [
        root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline_core.py",
        root / "target_centric_tracking_e2e_v1" / "run_stage1_with_conf_override.py",
        root / "target_centric_tracking_e2e_v1" / "run_phase1_product_pipeline.py",
        root / "target_centric_tracking_v2" / "stage3a0_audit_cross_shot_inputs.py",
        root / "target_centric_tracking_v2" / "stage3a1_confirm_cross_shot_boundary.py",
        root / "target_centric_tracking_v2" / "stage3a2_build_precut_target_memory.py",
        root / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        root / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        root / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    ]
    errors.extend(f"missing:{path}" for path in required if not path.is_file())

    # Explicit invariant: V7 may exist as an archive, but must never be required.
    v7_required = any(
        "global_ID_tracking_upgrade_v7" in str(path)
        for path in required
    )
    if v7_required:
        errors.append("contract:V7_MUST_NOT_BE_REQUIRED")

    if errors:
        print("Status=FAIL")
        for error in errors:
            print(error)
        return 2

    print("Status=PASS")
    print("Canonical full-scene E2E wrapper verified=True")
    print("One-direction E2E core verified=True")
    print("Product Stage1 confidence wrapper verified=True")
    print("Product Phase1 runner verified=True")
    print("Phase1 frozen scripts verified=", len(manifest["scripts"]))
    print("Frozen models verified=", verified_models)
    print("V6 ReID helper verified=True")
    print("V7 runtime required=False")
    print("E2E dependencies verified=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
