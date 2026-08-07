from __future__ import annotations

import threading

from app.core.config import Settings
from app.domains.candidate_handoff_r1.runtime_sync import R1RuntimeStateSynchronizer
from app.domains.tracking.execution import R1_EXECUTION_KIND
from app.domains.tracking.executor import TrackingJobExecutor
from app.domains.tracking.verifier import get_scene_target_tracking_verifier


class R1TrackingRuntimeExecutor(TrackingJobExecutor):
    """Dedicated executor for EVENT_CANDIDATE_HANDOFF_R1 subprocess runs."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            settings,
            execution_kind=R1_EXECUTION_KIND,
            thread_name_prefix="kickclip-r1-tracking",
        )

    def _installation_status(self):
        return get_scene_target_tracking_verifier().check()

    def _after_pipeline_sync(self, db, job, state, mapping, artifacts) -> None:
        if db is None or state is None:
            return
        R1RuntimeStateSynchronizer(db).sync(job=job, state=state, mapping=mapping)


_executor: R1TrackingRuntimeExecutor | None = None
_executor_lock = threading.Lock()


def get_r1_tracking_executor() -> R1TrackingRuntimeExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = R1TrackingRuntimeExecutor()
        return _executor
