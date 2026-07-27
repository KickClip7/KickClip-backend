from __future__ import annotations

import os
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
from app.domains.tracking.verifier import configured_absolute_path


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
            getattr(os, "killpg")(process.pid, signal.SIGTERM)
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
                getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
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
            str(self._runner_script()),
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
        ]
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
            str(self._runner_script()),
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
        ]
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
        elif kind != "recovery_resume":
            raise TrackingValidationError("Unsupported tracking resume action.")

        reviewer = str(action.get("reviewer") or "")
        note = str(action.get("note") or "")
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
        return configured_absolute_path(
            self.settings.TRACKING_PYTHON_EXECUTABLE,
            "TRACKING_PYTHON_EXECUTABLE",
        )

    def _runner_script(self) -> Path:
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
