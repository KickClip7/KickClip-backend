from __future__ import annotations

import hashlib
import json
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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Shot-boundary artifact is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Shot-boundary artifact root must be a JSON object.")
    return value


def _is_automatic_boundary_document(document: dict[str, Any]) -> bool:
    return (
        str(document.get("artifact_type") or "") == "AUTO_SHOT_BOUNDARIES"
        or str(document.get("boundary_origin") or "") == "AUTO_DETECTED"
    )


def _validated_automatic_shots(document: dict[str, Any]) -> list[dict[str, Any]]:
    if str(document.get("boundary_origin") or "") != "AUTO_DETECTED":
        raise ValueError("Automatic shot-boundary artifact has invalid boundary_origin.")
    if document.get("human_reviewed") is not False:
        raise ValueError("Automatic shot boundaries must declare human_reviewed=false.")
    if document.get("automatic_target_confirmation") is True:
        raise ValueError("Automatic target identity confirmation is forbidden.")

    structural = document.get("structural_validation")
    if not isinstance(structural, dict) or structural.get("status") != "PASS":
        raise ValueError("Automatic shot-boundary structural gate is not PASS.")
    if structural.get("complete_event_window_coverage") is not True:
        raise ValueError("Automatic shots do not cover the complete event window.")
    if int(structural.get("gap_count") or 0) != 0:
        raise ValueError("Automatic shots contain frame gaps.")
    if int(structural.get("overlap_count") or 0) != 0:
        raise ValueError("Automatic shots contain frame overlaps.")

    raw_shots = document.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError("Automatic shot-boundary artifact has no shots.")

    frame_count = int(document.get("frame_count") or (document.get("video") or {}).get("frame_count") or 0)
    if frame_count <= 0:
        raise ValueError("Automatic shot-boundary artifact has invalid frame_count.")

    shots: list[dict[str, Any]] = []
    expected_start = 0
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_shots):
        if not isinstance(raw, dict):
            raise ValueError("Automatic shot entry must be an object.")
        shot_id = str(raw.get("shot_id") or f"shot_{index:04d}")
        if shot_id in seen_ids:
            raise ValueError(f"Duplicate automatic shot id: {shot_id}")
        seen_ids.add(shot_id)
        start = int(raw.get("start_frame") if raw.get("start_frame") is not None else -1)
        end_raw = raw.get("end_frame_inclusive", raw.get("end_frame"))
        end = int(end_raw if end_raw is not None else -1)
        if start != expected_start or end < start:
            raise ValueError(
                f"Automatic shot coverage is invalid at {shot_id}: start={start}, end={end}, expected_start={expected_start}."
            )
        if end >= frame_count:
            raise ValueError(f"Automatic shot exceeds scene frame_count: {shot_id}")
        expected_start = end + 1
        shots.append({**raw, "shot_id": shot_id, "start_frame": start, "end_frame": end, "end_frame_inclusive": end})
    if expected_start != frame_count:
        raise ValueError(
            f"Automatic shots end at frame {expected_start - 1}, expected {frame_count - 1}."
        )
    return shots


def _automatic_boundary_compatibility_projection(source: Path) -> Path:
    """Create an ephemeral legacy-runtime view of safe automatic boundaries.

    The old scene-discovery runtime only understands REVIEWED_PASS shot rows.
    Product policy no longer asks a human to approve camera cuts.  We therefore
    project a structurally validated AUTO_SHOT_BOUNDARIES artifact into the old
    row shape *only for the discovery subprocess*.  The source artifact remains
    immutable, human_reviewed stays false, and this projection is never stored
    as a human review decision or target-identity confirmation.
    """

    source = source.resolve()
    document = _read_json_object(source)
    if not _is_automatic_boundary_document(document):
        return source

    shots = _validated_automatic_shots(document)
    projected_shots: list[dict[str, Any]] = []
    for index, row in enumerate(shots):
        start = int(row["start_frame"])
        end = int(row["end_frame_inclusive"])
        projected_shots.append(
            {
                **row,
                "shot_index": int(row.get("shot_index", index)),
                "start_frame": start,
                "end_frame": end,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                # Legacy scene-discovery compatibility only.  This means the
                # camera-cut boundary passed the automatic structural gate; it
                # does NOT mean a person reviewed the cut or confirmed identity.
                "review_state": "REVIEWED_PASS",
                "review_status": "REVIEWED_PASS",
                "status": "REVIEWED_PASS",
                "boundary_state": "AUTO_DETECTED",
                "approval_source": "AUTOMATIC_STRUCTURAL_GATE",
                "human_reviewed": False,
            }
        )

    projection = {
        **document,
        "schema_version": "kickclip.auto_shot_boundaries.discovery_compat.v1",
        "artifact_type": "AUTO_SHOT_BOUNDARIES_DISCOVERY_COMPAT",
        "boundary_origin": "AUTO_DETECTED",
        "human_reviewed": False,
        "automatic_confirmation": False,
        "automatic_target_confirmation": False,
        "review_status": "AUTO_ACCEPTED_FOR_DISCOVERY",
        "compatibility_projection": True,
        "compatibility_purpose": "LEGACY_SCENE_DISCOVERY_ROW_STATUS_ONLY",
        "source_auto_shot_boundaries_path": str(source),
        "source_auto_shot_boundaries_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "diagnostics": {
            **(document.get("diagnostics") if isinstance(document.get("diagnostics"), dict) else {}),
            "review_required": False,
            "retrieval_authorized": True,
            "automatic_cut_acceptance": True,
            "human_cut_review_performed": False,
        },
        "review_contract": {
            "review_required": False,
            "retrieval_authorized": True,
            "automatic_confirmation": False,
            "automatic_cut_acceptance": True,
            "human_reviewed": False,
        },
        "shots": projected_shots,
    }
    projection.pop("content_sha256", None)
    canonical = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    projection["content_sha256"] = hashlib.sha256(canonical).hexdigest()

    root = source.parent / "_scene_discovery_compat"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{projection['content_sha256'][:24]}_shot_boundaries.json"
    if not target.is_file():
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return target.resolve()


def _project_automatic_boundaries_for_scene_discovery(arguments: list[str]) -> list[str]:
    if not arguments or arguments[0] != "discover":
        return list(arguments)
    projected = list(arguments)
    try:
        index = projected.index("--shot-boundaries")
    except ValueError:
        return projected
    if index + 1 >= len(projected):
        raise ValueError("--shot-boundaries requires a path.")
    source = Path(projected[index + 1]).expanduser().resolve()
    projected[index + 1] = str(_automatic_boundary_compatibility_projection(source))
    return projected


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
        components["CANONICAL_TRACKING_RUNTIME_VERIFIED"] = bool(
            base.available
            and components.get("CANONICAL_E2E_RUNTIME_VERIFIED", False)
        )
        # Compatibility key only; no R3 algorithm is part of product tracking.
        components["R3_TRACKING_RUNTIME_VERIFIED"] = False
        components["FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"] = False
        if (
            discovery_verified
            and ranking_verified
            and not components.get("CANONICAL_TRACKING_RUNTIME_VERIFIED", False)
        ):
            return replace(
                base,
                code="EVENT_RANKING_SHADOW_RUNTIME_VERIFIED",
                message=(
                    "Scene discovery and V1.1.2a event ranking shadow "
                    "runtimes passed; canonical target-centric tracking "
                    "remains independently unavailable."
                ),
                components=components,
            )
        return replace(base, components=components)


def _integrated_scene_selection_run(
    self: scene_selection_module.SceneTargetSelectionService,
    arguments: list[str],
) -> None:
    # Camera-cut approval is no longer a product interaction.  For discovery,
    # structurally valid automatic boundaries are converted to a temporary
    # legacy row-status view so older frozen/compat runtimes do not stop at
    # WAITING_SHOT_BOUNDARY_REVIEW.  Target identity confirmation remains
    # unchanged and is never inferred here.
    runtime_arguments = _project_automatic_boundaries_for_scene_discovery(
        arguments
    )
    if (
        runtime_arguments
        and runtime_arguments[0] == "discover"
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
                    self.settings.SCENE_DISCOVERY_PROCESS_TIMEOUT_SECONDS
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
        ).run(runtime_arguments)
        return
    _original_scene_selection_run(self, runtime_arguments)


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
