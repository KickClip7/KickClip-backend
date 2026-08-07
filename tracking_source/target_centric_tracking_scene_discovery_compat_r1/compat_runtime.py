from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


RUNTIME_NAME = "target_centric_tracking_scene_discovery_compat_r1"
RUNTIME_MODE = "COMPAT_R1"
SCENE_PACKAGE_NAME = "target_centric_tracking_scene_target_selection_v1"
SCENE_MANIFEST_NAME = "scene_target_selection_frozen_manifest.json"
COMPAT_MANIFEST_NAME = "scene_discovery_compat_r1_manifest.json"
APPROVED_REVIEW_STATES = frozenset({"REVIEWED_PASS", "CONFIRMED"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _path(value: str, label: str, *, file_required: bool) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute.")
    if file_required and not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def validate_cli_input(arguments: argparse.Namespace) -> dict[str, Any]:
    values = {
        "project_root": str(arguments.project_root),
        "scene_id": arguments.scene_id,
        "discovery_id": arguments.discovery_id,
        "video": str(arguments.video),
        "detections_csv": str(arguments.detections_csv),
        "shot_boundaries": str(arguments.shot_boundaries),
        "output_root": str(arguments.output_root),
    }
    if any(not str(values[key]).strip() for key in ("scene_id", "discovery_id")):
        raise ValueError("scene_id and discovery_id must be non-empty.")
    _path(values["project_root"], "project_root", file_required=False)
    _path(values["video"], "video", file_required=True)
    detections = _path(
        values["detections_csv"], "detections_csv", file_required=True
    )
    shots = _path(
        values["shot_boundaries"], "shot_boundaries", file_required=True
    )
    if detections.suffix.lower() != ".csv":
        raise ValueError("detections_csv must use the .csv suffix.")
    if shots.suffix.lower() != ".json":
        raise ValueError("shot_boundaries must use the .json suffix.")
    _path(values["output_root"], "output_root", file_required=False)
    return values


def audit_reviewed_shots(path: Path) -> dict[str, Any]:
    artifact = read_json(path)
    rows = artifact.get("shots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Shot boundary artifact has no shots.")
    unapproved: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Shot boundary rows must be objects.")
        shot_id = str(row.get("shot_id") or f"shot_{index:04d}")
        state = str(
            row.get("review_state")
            or row.get("review_status")
            or row.get("status")
            or row.get("boundary_status")
            or ""
        ).strip().upper()
        if state not in APPROVED_REVIEW_STATES:
            unapproved.append(shot_id)
    if unapproved:
        raise ValueError(
            "WAITING_SHOT_BOUNDARY_REVIEW: unapproved shots: "
            + ",".join(unapproved)
        )
    return {
        "approved_review_states": sorted(APPROVED_REVIEW_STATES),
        "reviewed_shot_count": len(rows),
        "unapproved_review_count": 0,
        "unapproved_shot_ids": [],
    }


def _load_frozen_module(
    package_root: Path, module_name: str, filename: str
):
    package_text = str(package_root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, package_text)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, package_root / filename
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {filename}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == package_text:
            sys.path.pop(0)


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path) if path.is_file() else None
    if not SHA256_PATTERN.fullmatch(str(expected)) or actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )


def verify_runtime_material(project_root: Path) -> dict[str, Any]:
    runtime_root = Path(__file__).resolve().parent
    manifest = read_json(runtime_root / COMPAT_MANIFEST_NAME)
    scene_root = project_root / SCENE_PACKAGE_NAME
    scene_manifest = scene_root / SCENE_MANIFEST_NAME
    _verify_sha(
        scene_manifest,
        manifest["original_scene_target_selection_manifest_sha256"],
        "Original Scene Target Selection manifest",
    )
    original = read_json(scene_manifest)
    package_files = original.get("package_files") or {}
    if len(package_files) != 21:
        raise RuntimeError("Original Scene package file count is not 21.")
    for relative, expected in package_files.items():
        _verify_sha(scene_root / relative, expected, f"Scene package {relative}")
    dependencies = manifest["discovery_dependencies"]
    for name, item in dependencies.items():
        _verify_sha(project_root / item["path"], item["sha256"], name)
    _verify_sha(
        project_root / manifest["rfdetr_checkpoint"]["path"],
        manifest["rfdetr_checkpoint"]["sha256"],
        "RF-DETR checkpoint",
    )
    for relative, expected in manifest["runtime_files"].items():
        _verify_sha(runtime_root / relative, expected, f"Runtime file {relative}")
    return {
        "original_scene_package_files_verified": 21,
        "compatibility_dependency_file_hashes_verified": True,
        "rfdetr_checkpoint_verified": True,
    }


def discover(arguments: argparse.Namespace) -> dict[str, Any]:
    values = validate_cli_input(arguments)
    project_root = Path(values["project_root"]).resolve()
    verification = verify_runtime_material(project_root)
    shot_audit = audit_reviewed_shots(Path(values["shot_boundaries"]))
    scene_root = project_root / SCENE_PACKAGE_NAME
    policy_path = scene_root / "scene_target_selection_policy.json"
    scene_manifest_path = scene_root / SCENE_MANIFEST_NAME
    policy = read_json(policy_path)
    discovery = _load_frozen_module(
        scene_root,
        "scene_discovery_compat_r1_frozen_discovery",
        "discovery.py",
    )
    output_root = Path(values["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result = discovery.discover_from_frozen_detections(
        project_root=project_root,
        scene_id=values["scene_id"],
        candidate_namespace=values["discovery_id"],
        video=Path(values["video"]).resolve(),
        detections_csv=Path(values["detections_csv"]).resolve(),
        shot_boundaries_path=Path(values["shot_boundaries"]).resolve(),
        output_root=output_root,
        policy=policy,
        candidate_policy_sha256=sha256_file(policy_path),
        package_manifest_sha256=sha256_file(scene_manifest_path),
    )
    candidates_path = output_root / "scene_candidates.json"
    candidate_sha = sha256_file(candidates_path)
    video = result["video"]
    cache_key = canonical_sha256(
        {
            "runtime": RUNTIME_NAME,
            "runtime_manifest_sha256": sha256_file(
                Path(__file__).resolve().parent / COMPAT_MANIFEST_NAME
            ),
            "discovery_id": values["discovery_id"],
            "scene_id": values["scene_id"],
            "video_sha256": video["sha256"],
            "shot_boundary_sha256": result["shot_boundary"]["sha256"],
            "detections_sha256": result["provenance"]["detections_sha256"],
            "candidate_policy_sha256": sha256_file(policy_path),
            "schema_version": result["schema_version"],
        }
    )
    manifest = {
        "schema_version": "kickclip.scene_candidate_manifest.compat_r1.v1",
        "runtime_mode": RUNTIME_MODE,
        "scene_id": values["scene_id"],
        "discovery_id": values["discovery_id"],
        "candidate_count": len(result["candidates"]),
        "shot_count": result["shot_boundary"]["shot_count"],
        "candidate_cache_key": cache_key,
        "scene_candidates_sha256": candidate_sha,
        "scene_candidates": {
            "path": "scene_candidates.json",
            "sha256": candidate_sha,
        },
        "files": {
            "scene_candidates.json": {
                "sha256": candidate_sha,
            }
        },
        "inputs": {
            "source_video_sha256": video["sha256"],
            "detections_sha256": result["provenance"]["detections_sha256"],
            "shot_boundaries_sha256": result["shot_boundary"]["sha256"],
        },
        "original_scene_target_selection_manifest_sha256": (
            sha256_file(scene_manifest_path)
        ),
        "compatibility_runtime_manifest_sha256": sha256_file(
            Path(__file__).resolve().parent / COMPAT_MANIFEST_NAME
        ),
        "policy_sha256": sha256_file(policy_path),
        "portable_artifacts": True,
        "absolute_paths_exposed": False,
        "automatic_target_selection": False,
        "reviewed_shot_audit": shot_audit,
        "runtime_verification": verification,
    }
    atomic_json(output_root / "scene_candidate_manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Scene Discovery Compatibility Runtime R1"
    )
    subparsers = root.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("discover")
    command.add_argument("--project-root", type=Path, required=True)
    command.add_argument("--scene-id", required=True)
    command.add_argument("--discovery-id", required=True)
    command.add_argument("--video", type=Path, required=True)
    command.add_argument("--detections-csv", type=Path, required=True)
    command.add_argument("--shot-boundaries", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    verify = subparsers.add_parser("verify-material")
    verify.add_argument("--project-root", type=Path, required=True)
    return root


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command == "verify-material":
        result = verify_runtime_material(arguments.project_root.resolve())
    else:
        result = discover(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

