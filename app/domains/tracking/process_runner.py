from __future__ import annotations

import os
import hashlib
import json
import signal
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.domains.tracking.errors import (
    TrackingProcessTimeoutError,
    TrackingValidationError,
)
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.verifier import (
    configured_absolute_executable_path,
    configured_absolute_path,
)
from app.storage.local_storage import LocalStorage


@dataclass(frozen=True)
class TrackingProcessResult:
    command: tuple[str, ...]
    return_code: int
    stdout_log_path: Path
    stderr_log_path: Path
    process_pid: int


class TrackingProcessRegistry:
    """Tracks children so app shutdown can terminate the process tree."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[str]] = {}

    def add(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes[process.pid] = process

    def discard(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._processes.pop(process.pid, None)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            terminate_process_tree(process)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    shell=False,
                    capture_output=True,
                    check=False,
                    timeout=15,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()


class TrackingProcessRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        registry: TrackingProcessRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_tracking_process_registry()

    def build_new_command(
        self,
        job: TrackingJob,
        *,
        video_path: Path,
        overwrite: bool = False,
    ) -> list[str]:
        command = [
            str(self._python()),
            str(self._runner_script(job)),
            "--project-root",
            str(self._project_root()),
            "--video",
            str(video_path.resolve()),
            "--test-name",
            job.test_name,
            "--initial-bbox",
            *[str(value) for value in job.initial_bbox],
            "--device",
            job.device,
            "--reacquisition-mode",
            job.reacquisition_mode,
            "--output-root",
            str(self._output_root()),
            "--tracking-play-conf-threshold",
            str(float(self.settings.TRACKING_PLAY_CONF_THRESHOLD)),
        ]
        if self.settings.TRACKING_FULL_SCENE:
            command.append("--full-scene")
        self._append_scene_launch_arguments(command, job)
        if overwrite:
            command.append("--overwrite")
        if not self.settings.TRACKING_PREVIEW_ENABLED:
            command.append("--no-preview")
        return command

    def build_resume_command(
        self,
        job: TrackingJob,
        action: Mapping[str, Any],
    ) -> list[str]:
        command = [
            str(self._python()),
            str(self._runner_script(job)),
            "--project-root",
            str(self._project_root()),
            "--test-name",
            job.test_name,
            "--resume",
            "--device",
            job.device,
            "--reacquisition-mode",
            job.reacquisition_mode,
            "--output-root",
            str(self._output_root()),
            "--tracking-play-conf-threshold",
            str(float(self.settings.TRACKING_PLAY_CONF_THRESHOLD)),
        ]
        if self.settings.TRACKING_FULL_SCENE:
            command.append("--full-scene")
        kind = str(action.get("kind") or "")
        if kind == "review":
            stage = str(action["stage"])
            decision = str(action["decision"])
            flag = "--approve-review" if decision == "approve" else "--reject-review"
            command.extend([flag, stage])
        elif kind == "ambiguity":
            command.extend(["--ambiguity-id", str(action["ambiguity_id"])])
            if action.get("decision") == "absent":
                command.append("--confirm-absent")
            else:
                command.extend(
                    ["--confirmed-candidate", str(action["candidate_id"])]
                )
        elif kind == "none_of_these":
            command.extend(
                [
                    "--ambiguity-id",
                    str(action["ambiguity_id"]),
                    "--reject-all-candidates",
                ]
            )
        elif kind == "non_player_role":
            command.extend(
                [
                    "--ambiguity-id",
                    str(action["ambiguity_id"]),
                    "--reject-all-candidates-as-non-player-role",
                ]
            )
        elif kind == "candidate_rejected":
            command.extend(
                [
                    "--ambiguity-id",
                    str(action["ambiguity_id"]),
                    "--rejected-candidate",
                    str(action["candidate_id"]),
                ]
            )
        elif kind == "candidate_unreviewable":
            command.extend(
                [
                    "--ambiguity-id",
                    str(action["ambiguity_id"]),
                    "--unreviewable-candidate",
                    str(action["candidate_id"]),
                ]
            )
        elif kind != "recovery_resume":
            raise TrackingValidationError("Unsupported tracking resume action.")

        reviewer = str(action.get("reviewer") or "")
        note = str(action.get("note") or "")
        decision_artifact = str(action.get("decision_artifact_path") or "")
        decision_sha = str(action.get("decision_artifact_sha256") or "")
        if decision_artifact:
            decision_path = Path(decision_artifact)
            if not decision_path.is_absolute():
                decision_path = LocalStorage().resolve_path(decision_artifact)
            decision_path = decision_path.resolve()
            if not decision_path.is_file():
                raise TrackingValidationError(
                    "Candidate review decision artifact is missing."
                )
            actual_decision_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()
            if len(decision_sha) != 64 or actual_decision_sha != decision_sha:
                raise TrackingValidationError(
                    "Candidate review decision artifact hash mismatch."
                )
            command.extend(
                [
                    "--review-decision-artifact",
                    str(decision_path),
                    "--review-decision-sha256",
                    decision_sha,
                ]
            )
        if reviewer:
            command.extend(["--reviewer", reviewer])
        if note:
            command.extend(["--review-note", note])
        if not self.settings.TRACKING_PREVIEW_ENABLED:
            command.append("--no-preview")
        return command

    def command_for_job(
        self,
        job: TrackingJob,
        *,
        video_path: Path,
    ) -> list[str]:
        action = dict(job.queued_action or {})
        kind = str(action.get("kind") or "new")
        if kind == "new":
            return self.build_new_command(job, video_path=video_path)
        if kind == "recovery_restart":
            return self.build_new_command(
                job,
                video_path=video_path,
                overwrite=True,
            )
        return self.build_resume_command(job, action)

    def run(
        self,
        command: Sequence[str],
        *,
        test_name: str,
        on_start: Callable[[int], None] | None = None,
    ) -> TrackingProcessResult:
        output_root = self._output_root()
        log_dir = output_root / "_backend_process_logs" / test_name
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"

        creation_flags = 0
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        with stdout_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as stdout, stderr_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=str(self._project_root()),
                shell=False,
                stdout=stdout,
                stderr=stderr,
                text=True,
                creationflags=creation_flags,
                **popen_kwargs,
            )
            self.registry.add(process)
            try:
                if on_start is not None:
                    on_start(process.pid)
                return_code = process.wait(
                    timeout=self.settings.TRACKING_PROCESS_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired as exc:
                terminate_process_tree(process)
                raise TrackingProcessTimeoutError() from exc
            except Exception:
                terminate_process_tree(process)
                raise
            finally:
                self.registry.discard(process)

        return TrackingProcessResult(
            command=tuple(str(value) for value in command),
            return_code=return_code,
            stdout_log_path=stdout_path,
            stderr_log_path=stderr_path,
            process_pid=process.pid,
        )

    def output_directory(self, test_name: str) -> Path:
        output = (self._output_root() / test_name).resolve()
        if not output.is_relative_to(self._output_root()):
            raise TrackingValidationError("Invalid tracking output directory.")
        return output

    def _project_root(self) -> Path:
        return configured_absolute_path(
            self.settings.TRACKING_PROJECT_ROOT,
            "TRACKING_PROJECT_ROOT",
        )

    def _python(self) -> Path:
        # Preserve the explicitly configured executable path (including a
        # virtualenv symlink). Resolving it to the base interpreter can change
        # the runtime contract and bypass the virtualenv name used by operators.
        try:
            return configured_absolute_executable_path(
                self.settings.TRACKING_PYTHON_EXECUTABLE,
                "TRACKING_PYTHON_EXECUTABLE",
            )
        except ValueError as exc:
            raise TrackingValidationError(str(exc)) from exc

    @staticmethod
    def _scene_target_context(
        job: TrackingJob,
    ) -> Mapping[str, Any] | None:
        value = (job.runtime_metadata or {}).get("scene_target_selection")
        return value if isinstance(value, Mapping) else None

    @staticmethod
    def _read_json_object(path: Path, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackingValidationError(f"{label} is unreadable.") from exc
        if not isinstance(value, dict):
            raise TrackingValidationError(f"{label} must contain a JSON object.")
        return value

    @staticmethod
    def _bbox_matches(
        left: Sequence[float],
        right: Sequence[float],
        *,
        tolerance: float = 1.5,
    ) -> bool:
        if len(left) != 4 or len(right) != 4:
            return False
        return all(
            abs(float(a) - float(b)) <= tolerance
            for a, b in zip(left, right)
        )

    def _verified_scene_context(
        self,
        job: TrackingJob,
    ) -> tuple[Mapping[str, Any], dict[str, Path]] | None:
        scene_context = self._scene_target_context(job)
        if scene_context is None:
            return None

        selection_root_value = scene_context.get("selection_artifact_root")
        if not selection_root_value:
            raise TrackingValidationError(
                "Scene target selection artifact root is missing."
            )
        selection_root = Path(str(selection_root_value)).resolve()
        if not selection_root.is_dir():
            raise TrackingValidationError(
                "Scene target selection artifact root is missing."
            )

        required = {
            "tracking_launch_manifest_path": "tracking_launch_manifest_sha256",
            "shot_boundaries_path": "shot_boundaries_sha256",
            "target_selection_path": "target_selection_sha256",
            "target_reference_set_path": "target_reference_set_sha256",
            "earlier_anchor_decision_path": "earlier_anchor_decision_sha256",
        }
        resolved: dict[str, Path] = {}
        for key, sha_key in required.items():
            value = scene_context.get(key)
            if not value:
                raise TrackingValidationError(
                    f"Scene target selection is missing {key}."
                )
            path = Path(str(value)).resolve()
            if not path.is_file():
                raise TrackingValidationError(
                    f"Scene target selection file is missing: {key}."
                )
            if key != "shot_boundaries_path" and not path.is_relative_to(selection_root):
                raise TrackingValidationError(
                    "Scene target artifact escapes immutable root."
                )
            expected = str(scene_context.get(sha_key) or "").strip().lower()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if len(expected) != 64 or actual != expected:
                raise TrackingValidationError(
                    f"Scene target artifact hash mismatch: {key}."
                )
            resolved[key] = path

        return scene_context, resolved

    @staticmethod
    def _reviewed_cut_frames(
        boundaries: Mapping[str, Any],
        *,
        initial_frame: int,
    ) -> list[int]:
        rows = boundaries.get("shots")
        if not isinstance(rows, list):
            rows = boundaries.get("boundaries")
        if not isinstance(rows, list):
            rows = []

        cuts: set[int] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            candidate: int | None = None
            for key in ("start_frame", "cut_frame", "frame"):
                raw = row.get(key)
                if raw is None:
                    continue
                try:
                    candidate = int(raw)
                except (TypeError, ValueError):
                    candidate = None
                if candidate is not None:
                    break
            if candidate is not None and candidate > 0:
                cuts.add(candidate)
        return sorted(cuts)

    def _append_scene_launch_arguments(
        self,
        command: list[str],
        job: TrackingJob,
    ) -> None:
        """Bridge immutable scene selection into the canonical E2E runner.

        This method intentionally does not route to the legacy R1/R3 tracking
        adapter and does not pass R2/R3/V7 runtime arguments.
        """

        verified = self._verified_scene_context(job)
        if verified is None:
            return
        scene_context, paths = verified

        anchor = self._read_json_object(
            paths["earlier_anchor_decision_path"],
            "Earlier anchor decision",
        )
        try:
            initial_frame = int(anchor["anchor_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrackingValidationError(
                "Earlier anchor decision has no valid anchor_frame."
            ) from exc

        anchor_bbox = anchor.get("anchor_bbox_xyxy")
        if not isinstance(anchor_bbox, list) or len(anchor_bbox) != 4:
            raise TrackingValidationError(
                "Earlier anchor decision has no valid anchor_bbox_xyxy."
            )
        if not self._bbox_matches(job.initial_bbox, anchor_bbox):
            raise TrackingValidationError(
                "Tracking job bbox does not match the immutable selected anchor."
            )

        boundaries = self._read_json_object(
            paths["shot_boundaries_path"],
            "Shot boundaries",
        )
        cut_frames = self._reviewed_cut_frames(
            boundaries,
            initial_frame=initial_frame,
        )
        command.extend(["--initial-frame", str(initial_frame)])
        command.append("--allow-prestaged-output")
        if self.settings.TRACKING_USE_SELECTED_REFERENCE_MEMORY:
            command.extend(
                [
                    "--target-reference-set",
                    str(paths["target_reference_set_path"]),
                    "--target-reference-set-sha256",
                    str(scene_context["target_reference_set_sha256"]),
                    "--trusted-selected-reference-memory",
                ]
            )
        if cut_frames:
            command.extend(["--cut-frames", *[str(value) for value in cut_frames]])

    def _runner_script(self, job: TrackingJob) -> Path:
        # Every new/resume job uses the single canonical target-centric E2E
        # runtime. Scene selection contributes immutable input evidence only;
        # it never selects a different tracking algorithm.
        return configured_absolute_path(
            self.settings.TRACKING_E2E_SCRIPT_PATH,
            "TRACKING_E2E_SCRIPT_PATH",
        )

    def _output_root(self) -> Path:
        return configured_absolute_path(
            self.settings.TRACKING_OUTPUT_ROOT,
            "TRACKING_OUTPUT_ROOT",
        )


_process_registry = TrackingProcessRegistry()


def get_tracking_process_registry() -> TrackingProcessRegistry:
    return _process_registry
