from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


PACKAGE_NAME = "target_centric_tracking_scene_target_selection_v1"
MANIFEST_NAME = "scene_target_selection_frozen_manifest.json"
GOAL_A_BASELINES = (
    "target_centric_tracking_v1",
    "target_centric_tracking_v2",
)
DEPENDENCY_DISTRIBUTIONS = (
    "opencv-python",
    "numpy",
    "torch",
    "torchvision",
    "Pillow",
    "scipy",
    "pytest",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, *, repository: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ),
        key=str,
    )
    for path in files:
        audited_path = path.relative_to(repository).as_posix()
        digest.update(
            f"{sha256_file(path)}  {audited_path}\n".encode()
        )
    return digest.hexdigest()


def verify(project_root: Path) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    repository = root.parent
    package_root = (root / PACKAGE_NAME).resolve()
    manifest_path = package_root / MANIFEST_NAME
    failures: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return {
            "scene_target_selection_verified": False,
            "failures": [{"code": "MANIFEST_MISSING"}],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("package") != PACKAGE_NAME
        or not isinstance(manifest.get("package_files"), dict)
    ):
        failures.append({"code": "MANIFEST_CONTRACT_INVALID"})

    package_file_results: dict[str, dict[str, Any]] = {}
    for relative, expected in (manifest.get("package_files") or {}).items():
        path = (package_root / relative).resolve()
        actual = (
            sha256_file(path)
            if path.is_relative_to(package_root) and path.is_file()
            else None
        )
        passed = actual == expected
        package_file_results[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": passed,
        }
        if not passed:
            failures.append(
                {
                    "code": "PACKAGE_FILE_HASH_MISMATCH",
                    "path": relative,
                }
            )

    baseline_results: dict[str, dict[str, Any]] = {}
    expected_trees = manifest.get("frozen_baseline_tree_sha256") or {}
    for name in GOAL_A_BASELINES:
        path = (root / name).resolve()
        expected = expected_trees.get(name)
        actual = (
            tree_sha256(path, repository=repository)
            if path.is_dir()
            else None
        )
        passed = actual == expected
        baseline_results[name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": passed,
        }
        if not passed:
            failures.append(
                {
                    "code": "FROZEN_BASELINE_TREE_MISMATCH",
                    "path": name,
                }
            )

    checkpoint = (
        root / "weights" / "rfdetr" / "checkpoint_best_regular.pth"
    ).resolve()
    expected_checkpoint = (
        (manifest.get("checkpoints") or {}).get("rfdetr")
    )
    actual_checkpoint = (
        sha256_file(checkpoint) if checkpoint.is_file() else None
    )
    checkpoint_verified = actual_checkpoint == expected_checkpoint
    if not checkpoint_verified:
        failures.append({"code": "RFDETR_CHECKPOINT_HASH_MISMATCH"})

    dependency_lock_path = (
        package_root / "scene_target_selection_dependency_lock.json"
    )
    expected_lock_sha = manifest.get("dependency_lock_sha256")
    actual_lock_sha = (
        sha256_file(dependency_lock_path)
        if dependency_lock_path.is_file()
        else None
    )
    lock_sha_verified = actual_lock_sha == expected_lock_sha
    dependency_versions: dict[str, dict[str, Any]] = {}
    if lock_sha_verified:
        dependency_lock = json.loads(
            dependency_lock_path.read_text(encoding="utf-8")
        )
        expected_environment = (
            dependency_lock.get("validated_environment") or {}
        )
        expected_python = expected_environment.get("python")
        actual_python = platform.python_version()
        dependency_versions["python"] = {
            "expected": expected_python,
            "actual": actual_python,
            "verified": actual_python == expected_python,
        }
        for distribution in DEPENDENCY_DISTRIBUTIONS:
            expected = expected_environment.get(distribution)
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            dependency_versions[distribution] = {
                "expected": expected,
                "actual": actual,
                "verified": actual == expected,
            }
    else:
        failures.append({"code": "DEPENDENCY_LOCK_HASH_MISMATCH"})
    dependency_environment_verified = bool(
        dependency_versions
        and all(
            item["verified"] for item in dependency_versions.values()
        )
    )
    if not dependency_environment_verified:
        failures.append(
            {"code": "VALIDATED_DEPENDENCY_ENVIRONMENT_MISMATCH"}
        )

    import_verified = False
    package_integrity_failed = any(
        failure["code"]
        in {"MANIFEST_CONTRACT_INVALID", "PACKAGE_FILE_HASH_MISMATCH"}
        for failure in failures
    )
    if not package_integrity_failed:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        try:
            imported = importlib.import_module(PACKAGE_NAME)
            import_verified = (
                Path(imported.__file__).resolve().is_relative_to(
                    package_root
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "code": "PACKAGE_IMPORT_FAILED",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if sys.path and sys.path[0] == str(root):
                sys.path.pop(0)

    verified = not failures and import_verified
    return {
        "scene_target_selection_verified": verified,
        "scene_discovery_runtime_verified": verified,
        "package": PACKAGE_NAME,
        "manifest_sha256": sha256_file(manifest_path),
        "package_files": package_file_results,
        "frozen_baseline_trees": baseline_results,
        "rfdetr_checkpoint": {
            "expected_sha256": expected_checkpoint,
            "actual_sha256": actual_checkpoint,
            "verified": checkpoint_verified,
        },
        "dependency_lock": {
            "expected_sha256": expected_lock_sha,
            "actual_sha256": actual_lock_sha,
            "sha256_verified": lock_sha_verified,
            "environment_verified": dependency_environment_verified,
            "versions": dependency_versions,
        },
        "package_import_verified": import_verified,
        "r3_tracking_required": False,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    arguments = parser.parse_args()
    result = verify(Path(arguments.project_root))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["scene_target_selection_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
