from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, text

from app.core.config import Settings, get_settings
from app.ai.runtime.gpu_coordinator import claim_gpu_slot
from app.db.session import SessionLocal, engine
from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.errors import TrackingProcessTimeoutError
from app.domains.tracking.execution import LEGACY_EXECUTION_KIND
from app.domains.tracking.process_runner import (
    TrackingProcessRunner,
    get_tracking_process_registry,
)
from app.domains.tracking.repository import TrackingJobRepository
from app.domains.tracking.state_mapper import (
    map_pipeline_state,
    read_pipeline_state,
)
from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.sync import apply_pipeline_result, mark_process_failed
from app.domains.tracking.verifier import get_tracking_verifier
from app.storage.local_storage import LocalStorage


logger = logging.getLogger(__name__)
GPU_ADVISORY_LOCK_BASE = 0x4B49434B


class TrackingJobExecutor:
    """Durable DB-claimed local executor, replaceable by a queue worker later."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        execution_kind: str = LEGACY_EXECUTION_KIND,
        thread_name_prefix: str = "kickclip-tracking",
    ) -> None:
        self.settings = settings or get_settings()
        self.execution_kind = execution_kind
        self.thread_name_prefix = thread_name_prefix
        self._state_lock = threading.Lock()
        self._active_job_ids: set[str] = set()
        self._pool: ThreadPoolExecutor | None = None
        self._started = False
        self._stopping = threading.Event()

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._stopping.clear()

        installation = self._installation_status()
        if not installation.available:
            logger.info(
                "Tracking executor not started: %s (%s)",
                installation.code,
                installation.message,
            )
            return

        self._pool = ThreadPoolExecutor(
            max_workers=self.settings.TRACKING_MAX_CONCURRENT_JOBS,
            thread_name_prefix=self.thread_name_prefix,
        )
        recoverable = self._reconcile_recoverable_jobs()
        for tracking_job_id in recoverable:
            self.submit(tracking_job_id)

    def submit(self, tracking_job_id: str) -> bool:
        if not self._started:
            self.start()
        with self._state_lock:
            if (
                self._pool is None
                or self._stopping.is_set()
                or tracking_job_id in self._active_job_ids
            ):
                return False
            self._active_job_ids.add(tracking_job_id)
            pool = self._pool
        pool.submit(self._execute_and_release, tracking_job_id)
        return True

    def shutdown(self) -> None:
        self._stopping.set()
        get_tracking_process_registry().terminate_all()
        with self._state_lock:
            pool = self._pool
            self._pool = None
            self._started = False
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _execute_and_release(self, tracking_job_id: str) -> None:
        try:
            self._execute(tracking_job_id)
        finally:
            with self._state_lock:
                self._active_job_ids.discard(tracking_job_id)

    def _execute(self, tracking_job_id: str) -> None:
        db = SessionLocal()
        advisory: tuple[Connection, int] | None = None
        pid: int | None = None
        try:
            repository = TrackingJobRepository(db)
            if not repository.claim_queued(
                tracking_job_id, execution_kind=self.execution_kind
            ):
                db.rollback()
                return
            db.commit()

            job = repository.get_by_id(tracking_job_id)
            if job is None:
                return
            assert job is not None
            claimed_job = job
            now = datetime.now(timezone.utc)
            job.started_at = job.started_at or now
            job.runtime_started_at = now
            job.runtime_finished_at = None
            job.error_type = None
            job.error_message = None
            db.commit()

            with claim_gpu_slot():
                advisory = self._acquire_gpu_slot()
                if advisory is None and self._stopping.is_set():
                    mark_process_failed(
                        job,
                        error_type="TRACKING_EXECUTOR_SHUTDOWN",
                        public_message="Tracking runtime stopped during backend shutdown.",
                    )
                    db.commit()
                    return

                storage = LocalStorage()
                video_path = storage.resolve_path(job.media_asset.file_path)
                if not video_path.is_file():
                    mark_process_failed(
                        job,
                        error_type="VIDEO_FILE_MISSING",
                        public_message="Tracking source video is no longer available.",
                    )
                    db.commit()
                    return

                runner = TrackingProcessRunner(self.settings)
                command = runner.command_for_job(job, video_path=video_path)

                def on_start(process_pid: int) -> None:
                    nonlocal pid
                    pid = process_pid
                    claimed_job.process_pid = process_pid
                    db.commit()

                result = runner.run(
                    command,
                    test_name=job.test_name,
                    on_start=on_start,
                )
            artifacts = TrackingArtifactService(self.settings)
            self._stage_input_artifacts(job, artifacts)
            state = read_pipeline_state(Path(job.pipeline_state_path))
            mapping = map_pipeline_state(
                state,
                process_return_code=result.return_code,
                process_ended=True,
            )
            apply_pipeline_result(
                job,
                state=state,
                mapping=mapping,
                process_return_code=result.return_code,
                process_pid=result.process_pid,
                artifacts=artifacts,
            )
            self._after_pipeline_sync(db, job, state, mapping, artifacts)
            if mapping.backend_status == TrackingBackendStatus.FAILED:
                error_type, public_message = classify_runtime_failure(
                    result.stdout_log_path,
                    result.stderr_log_path,
                )
                job.error_type = error_type
                job.error_message = public_message
            metadata = dict(job.runtime_metadata or {})
            metadata["stdout_log_path"] = str(result.stdout_log_path)
            metadata["stderr_log_path"] = str(result.stderr_log_path)
            job.runtime_metadata = metadata
            db.commit()

        except TrackingProcessTimeoutError:
            db.rollback()
            job = TrackingJobRepository(db).get_by_id(tracking_job_id)
            if job is not None:
                self._reconcile_or_fail(
                    job,
                    process_return_code=None,
                    process_pid=pid,
                    fallback_type="TRACKING_PROCESS_TIMEOUT",
                    fallback_message=(
                        "Tracking runtime exceeded its configured timeout."
                    ),
                )
                db.commit()
            logger.exception("Tracking job timed out: %s", tracking_job_id)
        except Exception:
            db.rollback()
            logger.exception("Tracking job execution failed: %s", tracking_job_id)
            job = TrackingJobRepository(db).get_by_id(tracking_job_id)
            if job is not None:
                self._reconcile_or_fail(
                    job,
                    process_return_code=None,
                    process_pid=pid,
                    fallback_type="TRACKING_RUNTIME_FAILED",
                    fallback_message=(
                        "Tracking runtime failed. See backend process logs for details."
                    ),
                )
                db.commit()
        finally:
            if advisory is not None:
                self._release_gpu_slot(*advisory)
            db.close()

    def _installation_status(self):
        return get_tracking_verifier().check()

    def _after_pipeline_sync(
        self,
        db,
        job,
        state,
        mapping,
        artifacts,
    ) -> None:
        """Hook for execution-kind-specific DB synchronization."""

    @staticmethod
    def _stage_input_artifacts(
        job: Any,
        artifacts: TrackingArtifactService,
    ) -> None:
        metadata = job.runtime_metadata or {}
        configured = metadata.get("input_artifacts")
        if not isinstance(configured, Mapping):
            validation = metadata.get("input_validation")
            configured = (
                validation.get("artifact_paths")
                if isinstance(validation, Mapping)
                else None
            )
        if isinstance(configured, Mapping) and configured:
            artifacts.stage_input_validation_artifacts(job, configured)

    def _reconcile_or_fail(
        self,
        job,
        *,
        process_return_code: int | None,
        process_pid: int | None,
        fallback_type: str,
        fallback_message: str,
    ) -> None:
        try:
            state = read_pipeline_state(Path(job.pipeline_state_path))
            mapping = map_pipeline_state(
                state,
                process_return_code=process_return_code,
                process_ended=True,
            )
            if mapping.backend_status in {
                TrackingBackendStatus.COMPLETED,
                TrackingBackendStatus.COMPLETED_WITH_UNRESOLVED_GAPS,
                TrackingBackendStatus.COMPLETED_SAFE_BLOCK,
                TrackingBackendStatus.WAITING_MEMORY_REVIEW,
                TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION,
                TrackingBackendStatus.WAITING_SEGMENT_REVIEW,
            }:
                artifacts = TrackingArtifactService(self.settings)
                apply_pipeline_result(
                    job,
                    state=state,
                    mapping=mapping,
                    process_return_code=process_return_code,
                    process_pid=process_pid,
                    artifacts=artifacts,
                )
                self._after_pipeline_sync(None, job, state, mapping, artifacts)
                return
        except Exception:
            logger.exception(
                "Could not reconcile tracking state after process failure: %s",
                job.tracking_job_id,
            )
        mark_process_failed(
            job,
            error_type=fallback_type,
            public_message=fallback_message,
            process_pid=process_pid,
        )

    def _reconcile_recoverable_jobs(self) -> list[str]:
        db = SessionLocal()
        to_submit: list[str] = []
        try:
            repository = TrackingJobRepository(db)
            repository.quarantine_unroutable()
            for job in repository.list_recoverable(
                execution_kind=self.execution_kind
            ):
                state = read_pipeline_state(Path(job.pipeline_state_path))
                # A queued human decision intentionally precedes its runtime
                # acknowledgement. Do not let the stale pre-decision JSON state
                # overwrite the authoritative queued action during recovery.
                has_queued_action = bool(job.queued_action)
                if state is not None and not has_queued_action:
                    mapping = map_pipeline_state(
                        state,
                        process_return_code=job.process_return_code,
                        process_ended=False,
                    )
                    if mapping.backend_status != TrackingBackendStatus.RUNNING:
                        apply_pipeline_result(
                            job,
                            state=state,
                            mapping=mapping,
                            process_return_code=job.process_return_code,
                            process_pid=None,
                            artifacts=TrackingArtifactService(self.settings),
                        )
                        continue

                if job.status == TrackingBackendStatus.RUNNING.value:
                    if state is not None and state.get("memory"):
                        job.queued_action = {"kind": "recovery_resume"}
                    else:
                        job.queued_action = {"kind": "recovery_restart"}
                    job.status = TrackingBackendStatus.QUEUED.value
                    job.process_pid = None
                to_submit.append(job.tracking_job_id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Tracking startup reconciliation failed.")
        finally:
            db.close()
        return to_submit

    def _acquire_gpu_slot(self) -> tuple[Connection, int] | None:
        if engine.dialect.name != "postgresql":
            return None
        while not self._stopping.is_set():
            for slot in range(self.settings.TRACKING_MAX_CONCURRENT_JOBS):
                connection = engine.connect()
                lock_key = GPU_ADVISORY_LOCK_BASE + slot
                acquired = bool(
                    connection.scalar(
                        text("SELECT pg_try_advisory_lock(:key)"),
                        {"key": lock_key},
                    )
                )
                if acquired:
                    return connection, lock_key
                connection.close()
            time.sleep(0.5)
        return None

    @staticmethod
    def _release_gpu_slot(connection: Connection, lock_key: int) -> None:
        try:
            connection.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": lock_key},
            )
        finally:
            connection.close()


_executor: TrackingJobExecutor | None = None
_executor_lock = threading.Lock()


def get_tracking_executor() -> TrackingJobExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = TrackingJobExecutor()
        return _executor


def classify_runtime_failure(
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[str, str]:
    """Classify fatal diagnostics only after JSON state mapping has failed."""

    diagnostic = (
        _read_log_tail(stdout_path) + "\n" + _read_log_tail(stderr_path)
    ).lower()
    cuda_markers = (
        "cuda requested but unavailable",
        "cuda initialization",
        "cuda error",
        "cudnn",
        "no cuda gpus are available",
    )
    if any(marker in diagnostic for marker in cuda_markers):
        return (
            "CUDA_INITIALIZATION_FAILED",
            "Tracking CUDA runtime could not be initialized.",
        )
    model_markers = (
        "checkpoint is missing",
        "checkpoint is missing or changed",
        "error loading state_dict",
        "model initialization",
    )
    if any(marker in diagnostic for marker in model_markers):
        return (
            "MODEL_INITIALIZATION_FAILED",
            "A frozen tracking model could not be initialized.",
        )
    return (
        "PIPELINE_FATAL_ERROR",
        "Tracking pipeline terminated without a valid terminal JSON state.",
    )


def _read_log_tail(path: Path, limit: int = 64 * 1024) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            if size > limit:
                stream.seek(size - limit)
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
