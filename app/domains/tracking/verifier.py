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
from app.core.paths import get_project_root
from app.domains.highlight.event_candidate_ranking_v1_1.verifier import (
    EventCandidateRankingV11Verifier,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.verifier import (
    EventCandidateRankingV111Verifier,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.verifier import (
    EventCandidateRankingV112Verifier,
)

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


class SceneTargetTrackingInstallationVerifier:
    """Strict verifier for selection-assisted R3 jobs."""

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
        components = {
            "SAME_SHOT_RUNTIME_VERIFIED": generic.available,
            "R2_ASSISTED_RUNTIME_VERIFIED": False,
            "SCENE_TARGET_SELECTION_RUNTIME_VERIFIED": False,
            "R3_WRAPPER_VERIFIED": False,
            "EVENT_RANKING_RUNTIME_VERIFIED": False,
            "FULL_TARGET_SELECTION_E2E_VERIFIED": False,
            "FULL_SCENE_SELECTION_TRACKING_E2E_VERIFIED": False,
            "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED": False,
            "EVENT_RANKING_SAFETY_RUNTIME_VERIFIED": False,
            "EVENT_RANKING_CONTRACT_RUNTIME_VERIFIED": False,
            "FULL_EVENT_RECOMMENDATION_E2E_VERIFIED": False,
        }
        if not generic.available:
            return TrackingInstallationStatus(
                enabled=generic.enabled,
                available=False,
                checked_at=now,
                code="SAME_SHOT_RUNTIME_NOT_VERIFIED",
                message=generic.message,
                components=components,
            )
        try:
            tracking_root = configured_absolute_path(
                self.settings.TRACKING_PROJECT_ROOT,
                "TRACKING_PROJECT_ROOT",
            )
            tracking_python = configured_absolute_executable_path(
                self.settings.TRACKING_PYTHON_EXECUTABLE,
                "TRACKING_PYTHON_EXECUTABLE",
            )
            r3_script = configured_absolute_path(
                self.settings.TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH,
                "TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH",
            )
            smoke_verifier = configured_absolute_path(
                self.settings.TRACKING_SCENE_SELECTION_VERIFY_SCRIPT_PATH,
                "TRACKING_SCENE_SELECTION_VERIFY_SCRIPT_PATH",
            )
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
            manifests = {
                "scene target selection": (
                    configured_absolute_path(
                        self.settings.SCENE_TARGET_SELECTION_MANIFEST_PATH,
                        "SCENE_TARGET_SELECTION_MANIFEST_PATH",
                    ),
                    self.settings.SCENE_TARGET_SELECTION_MANIFEST_SHA256,
                ),
                "R2": (
                    configured_absolute_path(
                        self.settings.TRACKING_R2_MANIFEST_PATH,
                        "TRACKING_R2_MANIFEST_PATH",
                    ),
                    self.settings.TRACKING_R2_MANIFEST_SHA256,
                ),
                "R3": (
                    configured_absolute_path(
                        self.settings.TRACKING_R3_MANIFEST_PATH,
                        "TRACKING_R3_MANIFEST_PATH",
                    ),
                    self.settings.TRACKING_R3_MANIFEST_SHA256,
                ),
            }
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
            tracking_python,
            selection_python,
            r3_script,
            smoke_verifier,
            selection_verifier,
            *(path for path, _ in manifests.values()),
        ]
        if (
            not tracking_root.is_dir()
            or not selection_root.is_dir()
            or any(not path.is_file() for path in required_files)
        ):
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_RUNTIME_MISSING",
                message="Selection-assisted R3 runtime resources are missing.",
                components=components,
            )
        if not r3_script.is_relative_to(tracking_root):
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="SCENE_TARGET_RUNTIME_CONFIGURATION_INVALID",
                message="R3 script must be under TRACKING_PROJECT_ROOT.",
                components=components,
            )
        for label, (path, expected) in manifests.items():
            expected = expected.strip().lower()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if len(expected) != 64 or actual != expected:
                return TrackingInstallationStatus(
                    enabled=True,
                    available=False,
                    checked_at=now,
                    code="SCENE_TARGET_MANIFEST_HASH_MISMATCH",
                    message=f"{label} manifest SHA-256 verification failed.",
                    components=components,
                )
        components["R2_ASSISTED_RUNTIME_VERIFIED"] = True
        components["SCENE_TARGET_SELECTION_RUNTIME_VERIFIED"] = True

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
        smoke_result = self._run_verifier(
            [
                str(tracking_python),
                str(smoke_verifier),
                "--project-root",
                str(tracking_root),
                "--r3-script",
                str(r3_script),
                "--synthetic-assisted-smoke",
            ],
            cwd=tracking_root,
        )
        if smoke_result is None:
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="R3_WRAPPER_VERIFICATION_FAILED",
                message="R3 wrapper synthetic assisted smoke failed.",
                components=components,
            )
        required_claims = {
            "r3_wrapper_verified",
            "sports_osnet_strict_loader_verified",
            "selection_schema_compatible",
            "reference_schema_compatible",
            "synthetic_assisted_smoke_verified",
        }
        if not all(smoke_result.get(key) is True for key in required_claims):
            return TrackingInstallationStatus(
                enabled=True,
                available=False,
                checked_at=now,
                code="R3_VERIFIER_CLAIMS_INCOMPLETE",
                message="R3 verifier did not prove every required capability.",
                components=components,
            )
        components["R3_WRAPPER_VERIFIED"] = True
        components["FULL_TARGET_SELECTION_E2E_VERIFIED"] = True
        components["FULL_SCENE_SELECTION_TRACKING_E2E_VERIFIED"] = True
        event_package_root = (
            get_project_root()
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1"
        )
        event_status = EventCandidateRankingV11Verifier(
            event_package_root
        ).check()
        safety_package_root = (
            get_project_root()
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_1"
        )
        safety_status = EventCandidateRankingV111Verifier(
            safety_package_root
        ).check()
        contract_package_root = (
            get_project_root()
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_2"
        )
        contract_status = EventCandidateRankingV112Verifier(
            contract_package_root
        ).check()
        components["EVENT_RANKING_SAFETY_RUNTIME_VERIFIED"] = (
            safety_status.event_ranking_safety_runtime_verified
        )
        components["EVENT_RANKING_CONTRACT_RUNTIME_VERIFIED"] = (
            contract_status.event_ranking_contract_runtime_verified
        )
        components["EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"] = (
            event_status.event_ranking_shadow_runtime_verified
            and safety_status.event_ranking_safety_runtime_verified
            and contract_status.event_ranking_contract_runtime_verified
        )
        components["EVENT_RANKING_RUNTIME_VERIFIED"] = components[
            "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"
        ]
        components["FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"] = False
        return TrackingInstallationStatus(
            enabled=True,
            available=True,
            checked_at=now,
            code="SCENE_TARGET_TRACKING_AVAILABLE",
            message="Selection-assisted R3 runtime verification passed.",
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
        try:
            result = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return None
        return result if isinstance(result, dict) else None

    def _settings_fingerprint(self) -> str:
        names = (
            "TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH",
            "TRACKING_SCENE_SELECTION_VERIFY_SCRIPT_PATH",
            "SCENE_TARGET_SELECTION_PROJECT_ROOT",
            "SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE",
            "SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH",
            "SCENE_TARGET_SELECTION_MANIFEST_PATH",
            "SCENE_TARGET_SELECTION_MANIFEST_SHA256",
            "TRACKING_R2_MANIFEST_PATH",
            "TRACKING_R2_MANIFEST_SHA256",
            "TRACKING_R3_MANIFEST_PATH",
            "TRACKING_R3_MANIFEST_SHA256",
        )
        values = [str(getattr(self.settings, name)) for name in names]
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
            _scene_target_verifier = (
                SceneTargetTrackingInstallationVerifier()
            )
        return _scene_target_verifier
