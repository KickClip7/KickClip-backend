#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    package = root / "target_centric_tracking_scene_target_selection_v1"
    required = [
        package / "run_scene_target_selection.py",
        package / "launch.py",
        package / "selection.py",
        package / "discovery.py",
        package / "appearance.py",
        package / "contracts.py",
        package / "tracking_launch_manifest_schema.json",
        package / "target_selection_schema.json",
        package / "target_reference_set_schema.json",
        package / "earlier_anchor_decision_schema.json",
        package / "scene_target_selection_frozen_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    manifest = package / "scene_target_selection_frozen_manifest.json"
    result = {
        "scene_target_selection_source_verified": not missing,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        "missing": missing,
        "live_scene_target_selection_run_verified": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
