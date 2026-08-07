#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent

BASELINE_TREES = {
    "target_centric_tracking_v1": (
        "212539ec704e50443c268b042d5f7cfe535f2c20e6f1a0c4f4e4e4df61f42652"
    ),
    "target_centric_tracking_v2": (
        "4b26dc752a46f6c8a85571d40a2862bdefb1a7e3f0ae2b89239806219428e9e8"
    ),
    "target_centric_tracking_v2_production": (
        "a68dd1693fc1775aa527bd87711a5f1506fb8b474a343e38b9f4f611194d4ff8"
    ),
    "target_centric_tracking_v2_production_r2": (
        "8834e793fdd68b0100ed5e9773c331b59a016c3ee7639eefe8f73b83796970be"
    ),
    "target_centric_tracking_v2_production_eval_r2": (
        "ef450b8b4e2268192e6bc84b60f7bcb8a0da19e6057b510d97638f298e342bf1"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts
        ),
        key=lambda item: str(item),
    ):
        # Match the pre-change audit contract: hash each absolute path record,
        # then hash their textual digest listing.
        audited_path = path.relative_to(REPOSITORY).as_posix()
        record = f"{sha256_file(path)}  {audited_path}\n".encode()
        digest.update(record)
    return digest.hexdigest()


def hash_paths(paths: list[Path], *, relative_to: Path) -> dict[str, str]:
    return {
        path.relative_to(relative_to).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    observed = {
        name: tree_sha256(ROOT / name)
        for name in BASELINE_TREES
    }
    if observed != BASELINE_TREES:
        raise RuntimeError(
            "Frozen baseline integrity failed: "
            + json.dumps(
                {"expected": BASELINE_TREES, "observed": observed},
                sort_keys=True,
            )
        )
    manifest_path = PACKAGE / "scene_target_selection_frozen_manifest.json"
    package_files = [
        path
        for path in PACKAGE.rglob("*")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path != manifest_path
            and path.suffix not in {".pyc"}
        )
    ]
    backend_paths = [
        REPOSITORY / "app/api/v1/highlights.py",
        REPOSITORY / "app/core/config.py",
        REPOSITORY / "app/db/models.py",
        REPOSITORY / "app/domains/highlight/model.py",
        REPOSITORY / "app/domains/highlight/repository.py",
        REPOSITORY / "app/domains/highlight/scene_target_selection.py",
        REPOSITORY / "app/domains/highlight/schema.py",
        REPOSITORY / "app/domains/tracking/process_runner.py",
        REPOSITORY / "app/domains/tracking/repository.py",
        REPOSITORY / "app/domains/tracking/service.py",
        REPOSITORY / "app/domains/tracking/state_mapper.py",
        REPOSITORY / "app/domains/tracking/validation.py",
        REPOSITORY / "app/domains/tracking/artifacts.py",
        REPOSITORY
        / "alembic/versions/20260730_0013_create_scene_target_selections.py",
    ]
    representative_root = (
        ROOT
        / "runs/scene_target_selection/representative_goal_scene_v1"
    )
    representative_paths = [
        representative_root / name
        for name in (
            "scene_candidate_manifest.json",
            "scene_candidates.json",
            "scene_candidates.csv",
            "candidate_gallery.json",
            "candidate_gallery.html",
            "target_selection_r0001.json",
            "target_selection.json",
            "target_reference_set.json",
            "earlier_candidate_proposals.json",
        )
        if (representative_root / name).is_file()
    ]
    result = {
        "schema_version": (
            "kickclip.scene_target_selection_frozen_manifest.v1"
        ),
        "package": "target_centric_tracking_scene_target_selection_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "mode": "ASSISTED_USER_SELECTION",
        "frozen_baseline_tree_sha256": observed,
        "frozen_manifests": {
            "production_r2": sha256_file(
                ROOT
                / "target_centric_tracking_v2_production_r2"
                / "v2_production_r2_frozen_manifest.json"
            ),
            "evaluation_r2": sha256_file(
                ROOT
                / "target_centric_tracking_v2_production_eval_r2"
                / "identity_evaluation_r2_frozen_manifest.json"
            ),
        },
        "checkpoints": {
            "rfdetr": sha256_file(
                ROOT / "weights/rfdetr/checkpoint_best_regular.pth"
            ),
            "sports_osnet": sha256_file(
                ROOT
                / "global_ID_tracking_upgrade_v6/third_party/"
                "Deep-EIoU/Deep-EIoU/checkpoints/"
                "sports_model.pth.tar-60"
            ),
        },
        "package_files": hash_paths(package_files, relative_to=PACKAGE),
        "backend_adapter": hash_paths(
            backend_paths, relative_to=REPOSITORY
        ),
        "ui_assets": {
            "gallery.py": sha256_file(PACKAGE / "gallery.py"),
            "candidate_gallery.html": sha256_file(
                representative_root / "candidate_gallery.html"
            ),
            "backend_api": sha256_file(
                REPOSITORY / "app/api/v1/highlights.py"
            ),
        },
        "fixture_expected_artifacts": hash_paths(
            representative_paths, relative_to=ROOT
        ),
        "dependency_lock_sha256": sha256_file(
            PACKAGE / "scene_target_selection_dependency_lock.json"
        ),
        "safety_contract": {
            "automatic_target_selection": False,
            "automatic_earlier_anchor_confirmation": False,
            "cross_shot_motion_identity_linking": False,
            "pre_anchor_bbox_count": 0,
            "ground_truth_used_for_ranking": False,
        },
        "representative_validation": {
            "video_frame_count": 749,
            "reviewed_shot_count": 11,
            "candidate_count": 170,
            "earlier_proposal_count": 3,
            "state": "WAITING_EARLIER_ANCHOR_CONFIRMATION",
            "human_identity_confirmed": False,
        },
    }
    write_json(manifest_path, result)
    print(sha256_file(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
