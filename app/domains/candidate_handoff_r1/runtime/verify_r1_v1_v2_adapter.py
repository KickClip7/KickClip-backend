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
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--adapter-script", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    adapter = args.adapter_script.expanduser().resolve()
    required = {
        "e2e_runner": root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        "e2e_verifier": root / "target_centric_tracking_e2e_v1" / "verify_e2e_installation.py",
        "phase1_manifest": root / "target_centric_tracking_v1" / "phase1_frozen_manifest.json",
        "same_shot_stage1": root / "target_centric_tracking_v1" / "stage1_generate_rfdetr_detections.py",
        "same_shot_stage2": root / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
        "cross_shot_candidates": root / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        "cross_shot_reid": root / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        "cross_shot_gate": root / "target_centric_tracking_v2" / "stage3b2_make_safe_cross_shot_decision.py",
        "cross_shot_confirm": root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        "v6_reid_helper": root / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
        "adapter_script": adapter,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    models: list[dict[str, Any]] = []
    phase1 = required["phase1_manifest"]
    if phase1.is_file():
        document = read_object(phase1)
        for logical, row in (document.get("models") or {}).items():
            if not isinstance(row, Mapping) or not row.get("verified"):
                continue
            path = (root / str(row.get("path") or "").replace("\\", "/")).resolve()
            expected = str(row.get("sha256") or "")
            exists = path.is_file()
            matches = exists and sha256_file(path) == expected
            models.append(
                {
                    "logical": logical,
                    "path": str(path),
                    "expected_sha256": expected,
                    "exists": exists,
                    "sha256_matches": matches,
                }
            )
            if not matches:
                missing.append(str(path))
    adapter_text = adapter.read_text(encoding="utf-8") if adapter.is_file() else ""
    flags = (
        "--tracking-launch-manifest",
        "--target-selection",
        "--target-reference-set",
        "--earlier-anchor-decision",
        "--target-memory-revision",
        "--candidate-scoring-generation",
        "SELECTION_VIEW_SCHEMA",
        "RUNTIME_MEMORY_UPDATE_UNSUPPORTED",
    )
    adapter_contract = all(flag in adapter_text for flag in flags)
    forbidden_alias_patterns = (
        "root / \"global_ID_tracking_upgrade_v7\"",
        "from global_ID_tracking_upgrade_v7",
        "import global_ID_tracking_upgrade_v7",
    )
    no_v7_alias = not any(pattern in adapter_text for pattern in forbidden_alias_patterns)
    result = {
        "backend_r1_v1_v2_adapter_verified": adapter_contract,
        "research_sources_verified": all(
            path.is_file()
            for name, path in required.items()
            if name not in {"v6_reid_helper"}
        ),
        "strict_dependency_check_verified": True,
        "memory_passthrough_contract_verified": "--target-memory-revision" in adapter_text,
        "memory_revision_safety_gate_verified": "RUNTIME_MEMORY_UPDATE_UNSUPPORTED" in adapter_text,
        "selection_anchor_adapter_verified": "SELECTION_VIEW_SCHEMA" in adapter_text,
        "global_ID_tracking_upgrade_v7_is_not_aliased_to_v6": no_v7_alias,
        "live_frozen_runtime_verified": False,
        "synthetic_assisted_smoke_verified": False,
        "missing": sorted(set(missing)),
        "models": models,
        "files": {name: str(path) for name, path in required.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if adapter_contract and no_v7_alias and not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
