from __future__ import annotations

import hashlib
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
    """Runs the ZIP verifier without making backend startup fatal."""

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
        if not self.settings.TRACKING_ENABLED:
            return TrackingInstallationStatus(
                enabled=False,
                available=False,
                checked_at=now,
                code="TRACKING_DISABLED",
                message="Target tracking is disabled by configuration.",
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
                    "Frozen tracking runtime verification failed. "
                    "See backend logs for the verifier details."
                ),
                verifier_return_code=completed.returncode,
            )

        return TrackingInstallationStatus(
            enabled=True,
            available=True,
            checked_at=now,
            code="TRACKING_AVAILABLE",
            message="Frozen tracking runtime verification passed.",
            verifier_return_code=completed.returncode,
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


_verifier: TrackingInstallationVerifier | None = None
_verifier_lock = threading.Lock()


def get_tracking_verifier() -> TrackingInstallationVerifier:
    global _verifier
    with _verifier_lock:
        if _verifier is None:
            _verifier = TrackingInstallationVerifier()
        return _verifier
