from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME_ROOT))

import compat_runtime


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "")) or None


def _load_contract_tests():
    path = RUNTIME_ROOT / "tests" / "test_compat_runtime_contracts.py"
    spec = importlib.util.spec_from_file_location(
        "scene_discovery_compat_r1_contract_tests", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load compatibility contract tests.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_environment() -> dict[str, Any]:
    lock = compat_runtime.read_json(
        RUNTIME_ROOT / "compat_dependency_lock.json"
    )
    expected = lock["runtime_dependencies"]
    actual = {
        "python": platform.python_version(),
        "opencv": _module_version("cv2"),
        "numpy": _module_version("numpy"),
        "torch": _module_version("torch"),
        "torchvision": _module_version("torchvision"),
        "pillow": _module_version("PIL"),
        "scipy": _module_version("scipy"),
    }
    comparisons = {
        key: {
            "expected": value,
            "actual": actual.get(key),
            "verified": actual.get(key) == value,
        }
        for key, value in expected.items()
    }
    return {
        "compatibility_environment_verified": all(
            item["verified"] for item in comparisons.values()
        ),
        "runtime_dependencies": comparisons,
        "pytest_runtime_required": False,
    }


def verify(project_root: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    material: dict[str, Any] = {}
    environment: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    package_import_verified = False
    try:
        material = compat_runtime.verify_runtime_material(project_root)
    except Exception as exc:
        failures.append(
            {"code": "COMPATIBILITY_MATERIAL_FAILED", "detail": str(exc)}
        )
    try:
        environment = verify_environment()
        if not environment["compatibility_environment_verified"]:
            failures.append(
                {"code": "COMPATIBILITY_ENVIRONMENT_MISMATCH", "detail": ""}
            )
    except Exception as exc:
        failures.append(
            {"code": "COMPATIBILITY_ENVIRONMENT_FAILED", "detail": str(exc)}
        )
    try:
        project_text = str(project_root)
        sys.path.insert(0, project_text)
        imported = importlib.import_module(
            "target_centric_tracking_scene_target_selection_v1"
        )
        package_import_verified = Path(imported.__file__).resolve().is_relative_to(
            project_root / "target_centric_tracking_scene_target_selection_v1"
        )
    except Exception as exc:
        failures.append(
            {"code": "SCENE_PACKAGE_IMPORT_FAILED", "detail": str(exc)}
        )
    finally:
        if sys.path and sys.path[0] == str(project_root):
            sys.path.pop(0)
    work_root = Path(
        tempfile.mkdtemp(prefix="scene_discovery_compat_r1_")
    ).resolve()
    try:
        contract_tests = _load_contract_tests()
        tests = contract_tests.run_contract_tests(
            runtime_module=compat_runtime,
            project_root=project_root,
            work_root=work_root,
        )
    except Exception as exc:
        failures.append(
            {
                "code": "COMPATIBILITY_CONTRACT_TEST_FAILED",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    verified = (
        not failures
        and package_import_verified
        and bool(tests.get("synthetic_discovery_smoke"))
        and bool(tests.get("deterministic_repeat"))
        and environment.get("compatibility_environment_verified") is True
    )
    return {
        "scene_target_selection_verified": verified,
        "scene_discovery_frozen_runtime_verified": False,
        "scene_discovery_compat_runtime_verified": verified,
        "scene_discovery_runtime_verified": verified,
        "scene_discovery_runtime_mode": (
            "COMPAT_R1" if verified else "UNAVAILABLE"
        ),
        "runtime": compat_runtime.RUNTIME_NAME,
        "original_v1_v2_frozen_tree_reproduced": False,
        "material": material,
        "environment": environment,
        "package_import_verified": package_import_verified,
        "tests": tests,
        "automatic_target_selection": False,
        "r2_r3_required": False,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify(arguments.project_root.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["scene_discovery_compat_runtime_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

