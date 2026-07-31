from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy.orm import Session

from app.core.paths import get_project_root
from app.domains.highlight.event_candidate_ranking_v1_1_2a.annotation_service import (
    EventAnnotationCompatibilityV112aService,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.backend_adapter import (
    EventCandidateRankingV112aBackendAdapter,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.verifier import (
    EventCandidateRankingV112aVerifier,
)
from app.domains.highlight.model import SceneAITask
from app.domains.highlight import scene_ai_task as scene_task_module
from app.domains.highlight import (
    scene_target_selection as scene_selection_module,
)
from app.domains.highlight.runtime_contract import ConfiguredSceneRuntime
from app.domains.tracking import verifier as tracking_verifier_module


EVENT_RANKING_V1_1_2A = (
    "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW"
)
_installed = False
_original_dispatch = scene_task_module.SceneAITaskExecutor._dispatch
_original_scene_selection_run = (
    scene_selection_module.SceneTargetSelectionService._run
)
_base_scene_verifier = (
    tracking_verifier_module.SceneTargetTrackingInstallationVerifier
)


class IntegratedSceneTargetTrackingInstallationVerifier(
    _base_scene_verifier
):
    def _check_scene_discovery_runtime(
        self,
    ) -> tuple[bool, str]:
        self._scene_discovery_runtime_details = {}
        try:
            runtime_root = (
                tracking_verifier_module.configured_absolute_path(
                    (
                        self.settings.SCENE_DISCOVERY_PROJECT_ROOT
                        or self.settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
                        or self.settings.TRACKING_PROJECT_ROOT
                    ),
                    "SCENE_DISCOVERY_PROJECT_ROOT",
                )
            )
            runtime_python = (
                tracking_verifier_module
                .configured_absolute_executable_path(
                    (
                        self.settings
                        .SCENE_DISCOVERY_PYTHON_EXECUTABLE
                        or self.settings
                        .SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE
                        or self.settings.TRACKING_PYTHON_EXECUTABLE
                    ),
                    "SCENE_DISCOVERY_PYTHON_EXECUTABLE",
                )
            )
            default_runner = (
                runtime_root
                / "target_centric_tracking_scene_target_selection_v1"
                / "run_scene_target_selection.py"
            )
            runner = tracking_verifier_module.configured_absolute_path(
                self.settings.SCENE_DISCOVERY_SCRIPT_PATH
                or self.settings.SCENE_TARGET_SELECTION_SCRIPT_PATH
                or str(default_runner),
                "SCENE_DISCOVERY_SCRIPT_PATH",
            )
            verifier = tracking_verifier_module.configured_absolute_path(
                self.settings.SCENE_DISCOVERY_VERIFY_SCRIPT_PATH
                or self.settings.SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH,
                "SCENE_DISCOVERY_VERIFY_SCRIPT_PATH",
            )
            manifest = tracking_verifier_module.configured_absolute_path(
                self.settings.SCENE_DISCOVERY_MANIFEST_PATH
                or self.settings.SCENE_TARGET_SELECTION_MANIFEST_PATH,
                "SCENE_DISCOVERY_MANIFEST_PATH",
            )
        except ValueError as exc:
            return False, str(exc)

        checkpoint = (
            runtime_root
            / "weights"
            / "rfdetr"
            / "checkpoint_best_regular.pth"
        ).resolve()
        required_files = (
            runtime_python,
            runner,
            verifier,
            manifest,
            checkpoint,
        )
        if (
            not runtime_root.is_dir()
            or any(not path.is_file() for path in required_files)
        ):
            return False, "Scene discovery runtime resources are missing."
        if not runner.is_relative_to(runtime_root):
            return (
                False,
                "Scene discovery script must be under "
                "SCENE_DISCOVERY_PROJECT_ROOT.",
            )
        expected_sha = (
            (
                self.settings.SCENE_DISCOVERY_MANIFEST_SHA256
                or self.settings
                .SCENE_TARGET_SELECTION_MANIFEST_SHA256
            )
            .strip()
            .lower()
        )
        actual_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if len(expected_sha) != 64 or actual_sha != expected_sha:
            return (
                False,
                "Scene target selection manifest SHA-256 verification "
                "failed.",
            )
        result = self._run_verifier(
            [
                str(runtime_python),
                str(verifier),
                "--project-root",
                str(runtime_root),
            ],
            cwd=runtime_root,
        )
        if (
            result is None
            or result.get(
                "scene_discovery_runtime_verified",
                result.get("scene_target_selection_verified"),
            )
            is not True
        ):
            return (
                False,
                "Scene target selection package verification failed.",
            )
        self._scene_discovery_runtime_details = dict(result)
        return True, "Scene discovery runtime verification passed."

    def _run_check(self):
        base = super()._run_check()
        discovery_verified, _ = (
            self._check_scene_discovery_runtime()
        )
        discovery_details = getattr(
            self, "_scene_discovery_runtime_details", {}
        )
        frozen_discovery_verified = bool(
            discovery_details.get(
                "scene_discovery_frozen_runtime_verified",
                discovery_verified
                and not discovery_details.get(
                    "scene_discovery_compat_runtime_verified", False
                ),
            )
        )
        compat_discovery_verified = bool(
            discovery_details.get(
                "scene_discovery_compat_runtime_verified", False
            )
        )
        discovery_mode = str(
            discovery_details.get("scene_discovery_runtime_mode")
            or ("FROZEN" if frozen_discovery_verified else "UNAVAILABLE")
        )
        package_root = (
            get_project_root()
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_2a"
        )
        compatibility = EventCandidateRankingV112aVerifier(
            package_root
        ).check()
        components = dict(base.components or {})
        components["SCENE_DISCOVERY_FROZEN_RUNTIME_VERIFIED"] = (
            frozen_discovery_verified
        )
        components["SCENE_DISCOVERY_COMPAT_RUNTIME_VERIFIED"] = (
            compat_discovery_verified
        )
        components["SCENE_DISCOVERY_RUNTIME_VERIFIED"] = (
            discovery_verified
        )
        components["SCENE_DISCOVERY_RUNTIME_MODE_COMPAT_R1"] = (
            discovery_mode == "COMPAT_R1"
        )
        components["EVENT_RANKING_COMPATIBILITY_RUNTIME_VERIFIED"] = (
            compatibility.event_ranking_compatibility_runtime_verified
        )
        ranking_verified = (
            compatibility.event_ranking_compatibility_runtime_verified
        )
        components["EVENT_RANKING_SAFETY_RUNTIME_VERIFIED"] = (
            ranking_verified
        )
        components["EVENT_RANKING_CONTRACT_RUNTIME_VERIFIED"] = (
            ranking_verified
        )
        components["EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"] = (
            ranking_verified
        )
        components["EVENT_RANKING_RUNTIME_VERIFIED"] = ranking_verified
        components["R3_TRACKING_RUNTIME_VERIFIED"] = bool(
            base.available
            and components.get("R3_WRAPPER_VERIFIED", False)
        )
        components["FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"] = False
        if (
            discovery_verified
            and ranking_verified
            and not components.get("R3_TRACKING_RUNTIME_VERIFIED", False)
        ):
            return replace(
                base,
                code="EVENT_RANKING_SHADOW_RUNTIME_VERIFIED",
                message=(
                    "Scene discovery and V1.1.2a event ranking shadow "
                    "runtimes passed; R3 tracking remains independently "
                    "unavailable."
                ),
                components=components,
            )
        return replace(base, components=components)


def _integrated_scene_selection_run(
    self: scene_selection_module.SceneTargetSelectionService,
    arguments: list[str],
) -> None:
    if (
        arguments
        and arguments[0] == "discover"
        and self.settings.SCENE_DISCOVERY_SCRIPT_PATH.strip()
    ):
        discovery_settings = self.settings.model_copy(
            update={
                "SCENE_TARGET_SELECTION_PROJECT_ROOT": (
                    self.settings.SCENE_DISCOVERY_PROJECT_ROOT
                    or self.settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
                ),
                "SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE": (
                    self.settings.SCENE_DISCOVERY_PYTHON_EXECUTABLE
                    or self.settings
                    .SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE
                ),
                "SCENE_TARGET_SELECTION_SCRIPT_PATH": (
                    self.settings.SCENE_DISCOVERY_SCRIPT_PATH
                ),
                "SCENE_TARGET_SELECTION_PROCESS_TIMEOUT_SECONDS": (
                    self.settings
                    .SCENE_DISCOVERY_PROCESS_TIMEOUT_SECONDS
                ),
            }
        )
        script = Path(
            self.settings.SCENE_DISCOVERY_SCRIPT_PATH
        ).expanduser().resolve()
        ConfiguredSceneRuntime(
            discovery_settings,
            default_script=script,
            configured_script=str(script),
            script_setting_name="SCENE_DISCOVERY_SCRIPT_PATH",
        ).run(arguments)
        return
    _original_scene_selection_run(self, arguments)


def _integrated_dispatch(
    self: scene_task_module.SceneAITaskExecutor,
    db: Session,
    task: SceneAITask,
) -> dict[str, Any]:
    if task.task_type == EVENT_RANKING_V1_1_2A:
        payload = dict(task.payload)
        user = db.get(scene_task_module.User, task.owner_id)
        project = db.get(scene_task_module.Project, task.project_id)
        if user is None or project is None:
            raise ValueError("Scene AI task ownership context is missing.")
        return EventCandidateRankingV112aBackendAdapter(db).run(
            project=project,
            user=user,
            revision_id=payload["revision_id"],
            shortlist_size=payload["shortlist_size"],
            resolved_event=payload["resolved_event"],
            freeze_material=payload["freeze_material"],
        )
    return _original_dispatch(self, db, task)


def install_v112a_integration(
    highlights_module: ModuleType | None = None,
) -> None:
    global _installed
    scene_task_module.EVENT_RANKING_V1_1_2A = EVENT_RANKING_V1_1_2A
    if not _installed:
        scene_task_module.SceneAITaskExecutor._dispatch = (
            _integrated_dispatch
        )
        scene_selection_module.SceneTargetSelectionService._run = (
            _integrated_scene_selection_run
        )
        tracking_verifier_module.SceneTargetTrackingInstallationVerifier = (
            IntegratedSceneTargetTrackingInstallationVerifier
        )
        tracking_verifier_module._scene_target_verifier = None
        _installed = True
    if highlights_module is not None:
        highlights_module.EventAnnotationCompatibilityService = (
            EventAnnotationCompatibilityV112aService
        )
