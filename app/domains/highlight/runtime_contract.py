from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from app.core.config import Settings
from app.domains.tracking.process_runner import (
    get_tracking_process_registry,
    terminate_process_tree,
)
from app.domains.tracking.verifier import (
    configured_absolute_executable_path,
    configured_absolute_path,
)


JSON_SUFFIXES = {".json"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
REVIEW_ARTIFACT_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_runtime_file(
    output_root: Path,
    value: str | Path,
    *,
    allowed_suffixes: Iterable[str],
) -> Path:
    root = output_root.resolve()
    portable = Path(value)
    if portable.is_absolute():
        raise ValueError("Runtime artifact paths must be relative.")
    resolved = (root / portable).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Runtime artifact path escapes its immutable root.")
    suffixes = {suffix.lower() for suffix in allowed_suffixes}
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(
            f"Runtime artifact suffix is not allowed: {resolved.suffix}"
        )
    return resolved


def assert_expected_mime(path: Path, expected_prefix: str) -> None:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is None or not guessed.startswith(expected_prefix):
        raise ValueError(
            f"Runtime artifact MIME does not match {expected_prefix}: {path.name}"
        )


def load_runtime_json(output_root: Path, filename: str) -> tuple[dict[str, Any], Path]:
    path = validated_runtime_file(
        output_root,
        filename,
        allowed_suffixes=JSON_SUFFIXES,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Runtime JSON must contain an object: {filename}")
    return value, path


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Runtime artifact is outside the backend project root.")
    return resolved.relative_to(root).as_posix()


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


MUTABLE_SELECTION_FILENAMES = {
    "target_selection.json",
    "target_reference_set.json",
    "earlier_candidate_proposals.json",
    "earlier_anchor_decision.json",
    "tracking_launch_manifest.json",
}


@dataclass(frozen=True)
class SelectionWorkspace:
    discovery_root: Path
    staging_root: Path
    revision: int

    def finalize(self, selection_id: str) -> Path:
        final = (
            self.discovery_root
            / "selections"
            / f"r{self.revision:04d}_{selection_id}"
        ).resolve()
        selections_root = (self.discovery_root / "selections").resolve()
        if not final.is_relative_to(selections_root):
            raise ValueError("Selection artifact root escapes discovery.")
        if final.exists():
            raise ValueError("Immutable selection artifact root already exists.")
        self.staging_root.replace(final)
        return final


def prepare_selection_workspace(
    discovery_root: Path,
    *,
    revision: int,
    source_root: Path | None = None,
) -> SelectionWorkspace:
    discovery = discovery_root.resolve()
    source = (source_root or discovery_root).resolve()
    selections_root = (discovery / "selections").resolve()
    if not selections_root.is_relative_to(discovery):
        raise ValueError("Invalid selection workspace root.")
    selections_root.mkdir(parents=True, exist_ok=True)
    staging = (
        selections_root
        / f".building_r{revision:04d}_{uuid4().hex}"
    ).resolve()

    def ignore(_: str, names: list[str]) -> set[str]:
        ignored = set(MUTABLE_SELECTION_FILENAMES)
        ignored.add("selections")
        return ignored.intersection(names)

    shutil.copytree(
        source,
        staging,
        copy_function=_link_or_copy,
        ignore=ignore,
    )
    return SelectionWorkspace(
        discovery_root=discovery,
        staging_root=staging,
        revision=revision,
    )


class ConfiguredSceneRuntime:
    """Runs a frozen scene-selection CLI with explicit runtime configuration."""

    def __init__(
        self,
        settings: Settings,
        *,
        default_script: Path,
        configured_script: str | None = None,
        script_setting_name: str = "SCENE_TARGET_SELECTION_SCRIPT_PATH",
    ) -> None:
        self.settings = settings
        root_value = (
            settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
            or settings.TRACKING_PROJECT_ROOT
        )
        python_value = (
            settings.SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE
            or settings.TRACKING_PYTHON_EXECUTABLE
        )
        script_value = (
            configured_script
            if configured_script is not None
            else settings.SCENE_TARGET_SELECTION_SCRIPT_PATH
        )
        self.project_root = configured_absolute_path(
            root_value,
            "SCENE_TARGET_SELECTION_PROJECT_ROOT",
        )
        self.python = configured_absolute_executable_path(
            python_value,
            "SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE",
        )
        self.script = configured_absolute_path(
            script_value or str(default_script),
            script_setting_name,
        )
        if not self.script.is_relative_to(self.project_root):
            raise ValueError(
                "Scene target selection script must be under its project root."
            )

    def run(
        self,
        arguments: list[str],
        *,
        expect_json_stdout: bool = False,
    ) -> dict[str, Any] | None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self.script.parent)
            if not existing
            else f"{self.script.parent}{os.pathsep}{existing}"
        )
        creation_flags = 0
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            [str(self.python), str(self.script), *arguments],
            cwd=str(self.project_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            creationflags=creation_flags,
            **popen_kwargs,
        )
        registry = get_tracking_process_registry()
        registry.add(process)
        try:
            stdout, stderr = process.communicate(
                timeout=self.settings.SCENE_TARGET_SELECTION_PROCESS_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            terminate_process_tree(process)
            raise RuntimeError(
                "Scene target selection runtime exceeded its timeout."
            ) from exc
        except Exception:
            terminate_process_tree(process)
            raise
        finally:
            registry.discard(process)
        if process.returncode != 0:
            raise RuntimeError(
                "Scene target selection runtime failed: "
                + (stderr[-3000:] or stdout[-3000:])
            )
        if not expect_json_stdout:
            return None
        value = json.loads(stdout)
        if not isinstance(value, dict):
            raise RuntimeError("Scene target selection runtime returned invalid JSON.")
        return value


def redact_runtime_paths(
    value: Any,
    *,
    artifact_ids_by_path: dict[str, str],
) -> Any:
    """Replace runtime-owned paths with authenticated artifact references."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(child, str):
                normalized = child.replace("\\", "/")
                artifact_id = artifact_ids_by_path.get(normalized)
                if artifact_id:
                    result[key] = {
                        "artifact_id": artifact_id,
                        "url": f"/api/v1/artifacts/{artifact_id}/download",
                    }
                    continue
                if Path(child).is_absolute() or key.endswith("_path"):
                    continue
                if key.endswith("_artifact") or (
                    Path(normalized).suffix.lower()
                    in (REVIEW_ARTIFACT_SUFFIXES | JSON_SUFFIXES)
                    and "/" in normalized
                ):
                    continue
            result[key] = redact_runtime_paths(
                child,
                artifact_ids_by_path=artifact_ids_by_path,
            )
        return result
    if isinstance(value, list):
        return [
            redact_runtime_paths(
                child,
                artifact_ids_by_path=artifact_ids_by_path,
            )
            for child in value
        ]
    return value
