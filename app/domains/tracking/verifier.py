from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingInstallationStatus:
    enabled: bool
    available: bool
    checked_at: datetime
    code: str
    message: str
    verifier_return_code: int | None = None
    components: dict[str, bool] | None = None


def configured_absolute_path(value: str, setting_name: str) -> Path:
    text = value.strip()
    if not text:
        raise ValueError(f"{setting_name} is not configured")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{setting_name} must be an absolute path")
    return path.resolve()


def configured_absolute_executable_path(value: str, setting_name: str) -> Path:
    """Validate an executable path without resolving a virtualenv symlink."""

    text = value.strip()
    if not text:
        raise ValueError(f"{setting_name} is not configured")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{setting_name} must be an absolute path")
    return path


class TrackingInstallationVerifier:
    """Verify the single canonical target-centric E2E runtime.

    Product tracking is:
      target_centric_tracking_e2e_v1
        -> frozen target_centric_tracking_v1
        -> frozen target_centric_tracking_v2
        -> RF-DETR
        -> global_ID_tracking_upgrade_v6 Sports-OSNet helper

    global_ID_tracking_upgrade_v7 is intentionally not part of this graph.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._cached: TrackingInstallationStatus | None = None
        self._fingerprint: str | None = None

    def check(self, *, force: bool = False) -> TrackingInstallationStatus:
        fingerprint = self._settings_fingerprint()
        with self._lock:
            if (
                not force
                and self._cached is not None
                and self._fingerprint == fingerprint
            ):
                return self._cached
            status = self._run_check()
            self._cached = status
            self._fingerprint = fingerprint
            return status

    def _run_check(self) -> TrackingInstallationStatus:
        now = datetime.now(timezone.utc)
        base_components = {
            "CANONICAL_E2E_RUNTIME_VERIFIED": False,
            "FROZEN_V1_V2_DEPENDENCIES_VERIFIED": False,
            "V6_REID_RUNTIME_VERIFIED": False,
            "V7_RUNTIME_REQUIRED": False,
        }
        if not self.settings.TRACKING_ENABLED:
            return TrackingInstallationStatus(
                enabled=False,
                available=False,
                checked_at=now,
                code="TRACKING_DISABLED",
                message="Target tracking is disabled by configuration.",
                components=base_components,
            )

        try:
            project_root = configured_absolute_path(
                self.settings.TRACKING_PROJECT_ROOT,
                "TRACKING_PROJECT_ROOT",
            )
            python = configured_absolute_executable_path(
                self.settings.TRACKING_PYTHON_EXECUTABLE,
                "TRACKING_PYTHON_EXECUTABLE",
            )
            runner = configured_absolute_path(
                self.settings.TRACKING_E2E_SCRIPT_PATH,
                "TRACKING_E2E_SCRIPT_PATH",
            )
            verifier = configured_absolute_path(
                self.settings.TRACKING_VERIFY_SCRIPT_PATH,
                "TRACKING_VERIFY_SCRIPT_PATH",
            )
            configured_absolute_path(
                self.settings.TRACKING_OUTPUT_ROOT,
                "TRACKING_OUTPUT_ROOT",
            )
        except ValueError as exc:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="TRACKING_CONFIGURATION_INVALID",
                message=str(exc),
                components=base_components,
            )

        required = {
            "TRACKING_PROJECT_ROOT": (project_root, "directory"),
            "TRACKING_PYTHON_EXECUTABLE": (python, "file"),
            "TRACKING_E2E_SCRIPT_PATH": (runner, "file"),
            "TRACKING_VERIFY_SCRIPT_PATH": (verifier, "file"),
        }
        missing = [
            name
            for name, (path, kind) in required.items()
            if (kind == "file" and not path.is_file())
            or (kind == "directory" and not path.is_dir())
        ]
        if missing:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="TRACKING_RUNTIME_MISSING",
                message="Missing tracking runtime resource(s): " + ", ".join(missing),
                components=base_components,
            )
        if not runner.is_relative_to(project_root) or not verifier.is_relative_to(
            project_root
        ):
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="TRACKING_CONFIGURATION_INVALID",
                message="Tracking scripts must be located under TRACKING_PROJECT_ROOT.",
                components=base_components,
            )

        # Refuse to accidentally point the canonical setting at the old
        # tracking_runtime_adapter copy, an R3 adapter, or a Global-ID entry
        # point. TRACKING_PROJECT_ROOT is the tracking_source root, therefore
        # the canonical runner must live directly below its E2E package.
        expected_runner = (
            project_root
            / "target_centric_tracking_e2e_v1"
            / "run_target_centric_pipeline.py"
        ).resolve()
        expected_verifier = (
            project_root
            / "target_centric_tracking_e2e_v1"
            / "verify_e2e_installation.py"
        ).resolve()
        normalized_runner = runner.as_posix().lower()
        if (
            runner != expected_runner
            or verifier != expected_verifier
            or "r1_v1_v2_adapter" in normalized_runner
            or "global_id_tracking_upgrade_v7" in normalized_runner
        ):
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="TRACKING_CANONICAL_RUNNER_INVALID",
                message=(
                    "TRACKING_E2E_SCRIPT_PATH must point to the canonical "
                    "target_centric_tracking_e2e_v1 runner."
                ),
                components=base_components,
            )

        command = [
            str(python),
            str(verifier),
            "--project-root",
            str(project_root),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.settings.TRACKING_VERIFY_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("Tracking installation verifier could not run.")
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="TRACKING_VERIFIER_EXECUTION_FAILED",
                message="Tracking installation verifier could not complete.",
                components=base_components,
            )

        if completed.returncode != 0:
            logger.error(
                "Tracking verifier failed rc=%s stdout=%r stderr=%r",
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="TRACKING_VERIFICATION_FAILED",
                message=(
                    "Canonical target-centric tracking runtime verification failed. "
                    "See backend logs for verifier details."
                ),
                verifier_return_code=completed.returncode,
                components=base_components,
            )

        components = dict(base_components)
        components.update(
            {
                "CANONICAL_E2E_RUNTIME_VERIFIED": True,
                "FROZEN_V1_V2_DEPENDENCIES_VERIFIED": True,
                "V6_REID_RUNTIME_VERIFIED": True,
                "V7_RUNTIME_REQUIRED": False,
            }
        )
        return TrackingInstallationStatus(
            enabled=True,
            available=True,
            checked_at=now,
            code="TRACKING_AVAILABLE",
            message=(
                "Canonical target-centric E2E runtime and frozen V1/V2/V6 "
                "dependencies are verified."
            ),
            verifier_return_code=completed.returncode,
            components=components,
        )

    def _settings_fingerprint(self) -> str:
        values = (
            str(self.settings.TRACKING_ENABLED),
            self.settings.TRACKING_PROJECT_ROOT,
            self.settings.TRACKING_PYTHON_EXECUTABLE,
            self.settings.TRACKING_E2E_SCRIPT_PATH,
            self.settings.TRACKING_VERIFY_SCRIPT_PATH,
            self.settings.TRACKING_OUTPUT_ROOT,
        )
        return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


class SceneTargetTrackingInstallationVerifier:
    """Verify scene-selection inputs plus the canonical E2E tracker.

    The historical class name is retained because API/service imports depend on
    it. It no longer verifies or routes to an R3 tracking algorithm.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        generic: TrackingInstallationVerifier | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.generic = generic or get_tracking_verifier()
        self._lock = threading.Lock()
        self._cached: TrackingInstallationStatus | None = None
        self._fingerprint: str | None = None

    def check(self, *, force: bool = False) -> TrackingInstallationStatus:
        fingerprint = self._settings_fingerprint()
        with self._lock:
            if (
                not force
                and self._cached is not None
                and self._fingerprint == fingerprint
            ):
                return self._cached
            status = self._run_check()
            self._cached = status
            self._fingerprint = fingerprint
            return status

    def _run_check(self) -> TrackingInstallationStatus:
        now = datetime.now(timezone.utc)
        generic = self.generic.check()
        components = dict(generic.components or {})
        components.update(
            {
                "SCENE_TARGET_SELECTION_RUNTIME_VERIFIED": False,
                "CANONICAL_SCENE_TARGET_E2E_VERIFIED": False,
                # Backward-compatible diagnostic key. It is deliberately False:
                # no R3 wrapper is part of the product runtime anymore.
                "R3_WRAPPER_VERIFIED": False,
                "R2_ASSISTED_RUNTIME_VERIFIED": False,
                "FULL_TARGET_SELECTION_E2E_VERIFIED": False,
                "FULL_SCENE_SELECTION_TRACKING_E2E_VERIFIED": False,
            }
        )
        if not generic.available:
            return TrackingInstallationStatus(
                enabled=generic.enabled,
                available=False,
                checked_at=now,
                code="CANONICAL_E2E_RUNTIME_NOT_VERIFIED",
                message=generic.message,
                components=components,
            )

        try:
            selection_root = configured_absolute_path(
                (
                    self.settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
                    or self.settings.TRACKING_PROJECT_ROOT
                ),
                "SCENE_TARGET_SELECTION_PROJECT_ROOT",
            )
            selection_python = configured_absolute_executable_path(
                (
                    self.settings.SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE
                    or self.settings.TRACKING_PYTHON_EXECUTABLE
                ),
                "SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE",
            )
            selection_verifier = configured_absolute_path(
                self.settings.SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH,
                "SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH",
            )
            selection_manifest = configured_absolute_path(
                self.settings.SCENE_TARGET_SELECTION_MANIFEST_PATH,
                "SCENE_TARGET_SELECTION_MANIFEST_PATH",
            )
        except ValueError as exc:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_RUNTIME_CONFIGURATION_INVALID",
                message=str(exc),
                components=components,
            )

        required_files = [
            selection_python,
            selection_verifier,
            selection_manifest,
        ]
        if (
            not selection_root.is_dir()
            or any(not path.is_file() for path in required_files)
        ):
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_RUNTIME_MISSING",
                message="Scene target selection runtime resources are missing.",
                components=components,
            )

        expected = (
            self.settings.SCENE_TARGET_SELECTION_MANIFEST_SHA256.strip().lower()
        )
        actual = hashlib.sha256(selection_manifest.read_bytes()).hexdigest()
        if len(expected) != 64 or actual != expected:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_MANIFEST_HASH_MISMATCH",
                message="Scene target selection manifest SHA-256 verification failed.",
                components=components,
            )

        selection_result = self._run_verifier(
            [
                str(selection_python),
                str(selection_verifier),
                "--project-root",
                str(selection_root),
            ],
            cwd=selection_root,
        )
        if selection_result is None:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_SELECTION_VERIFICATION_FAILED",
                message="Scene target selection package verification failed.",
                components=components,
            )

        selection_verified = bool(
            selection_result.get(
                "scene_target_selection_verified",
                selection_result.get(
                    "scene_discovery_runtime_verified",
                    selection_result.get("status") == "PASS",
                ),
            )
        )
        if not selection_verified:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_SELECTION_VERIFICATION_FAILED",
                message="Scene target selection verifier did not confirm the package.",
                components=components,
            )

        components["SCENE_TARGET_SELECTION_RUNTIME_VERIFIED"] = True
        components["CANONICAL_SCENE_TARGET_E2E_VERIFIED"] = True
        # These compatibility keys mean the whole scene-selection -> canonical
        # tracker wiring is available; they do not imply a separate R3 algorithm.
        components["FULL_TARGET_SELECTION_E2E_VERIFIED"] = True
        components["FULL_SCENE_SELECTION_TRACKING_E2E_VERIFIED"] = True

        return TrackingInstallationStatus(
            enabled=True,
            available=True,
            checked_at=now,
            code="SCENE_TARGET_TRACKING_AVAILABLE",
            message=(
                "Scene target selection and the canonical target-centric E2E "
                "tracking runtime are verified. R2/R3/V7 tracking runtimes are not used."
            ),
            verifier_return_code=0,
            components=components,
        )

    def _run_verifier(
        self,
        command: list[str],
        *,
        cwd: Path,
    ) -> dict[str, object] | None:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.settings.TRACKING_VERIFY_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("Scene target runtime verifier could not run.")
            return None
        if completed.returncode != 0:
            logger.error(
                "Scene target verifier failed rc=%s stdout=%r stderr=%r",
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
            return None
        raw = (completed.stdout or "").strip()
        if not raw:
            return {"status": "PASS"}
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Some frozen verifiers print key=value lines rather than JSON.
            if "Status=PASS" in raw or "Status = PASS" in raw:
                return {"status": "PASS", "scene_target_selection_verified": True}
            return None
        return result if isinstance(result, dict) else None

    def _settings_fingerprint(self) -> str:
        names = (
            "SCENE_TARGET_SELECTION_PROJECT_ROOT",
            "SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE",
            "SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH",
            "SCENE_TARGET_SELECTION_MANIFEST_PATH",
            "SCENE_TARGET_SELECTION_MANIFEST_SHA256",
        )
        values = [
            self.generic._settings_fingerprint(),
            *[str(getattr(self.settings, name)) for name in names],
        ]
        return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


_verifier: TrackingInstallationVerifier | None = None
_verifier_lock = threading.RLock()
_scene_target_verifier: SceneTargetTrackingInstallationVerifier | None = None


def get_tracking_verifier() -> TrackingInstallationVerifier:
    global _verifier
    with _verifier_lock:
        if _verifier is None:
            _verifier = TrackingInstallationVerifier()
        return _verifier


def get_scene_target_tracking_verifier() -> (
    SceneTargetTrackingInstallationVerifier
):
    global _scene_target_verifier
    with _verifier_lock:
        if _scene_target_verifier is None:
            _scene_target_verifier = SceneTargetTrackingInstallationVerifier()
        return _scene_target_verifier
