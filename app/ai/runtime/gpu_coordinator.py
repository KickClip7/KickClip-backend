from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator


# Action Spotting, RF-DETR player discovery, and target tracking share one
# in-process GPU slot. Tracking retains its existing PostgreSQL advisory lock as
# an additional cross-worker guard. This semaphore makes the local
# BackgroundTasks/TrackingExecutor paths deterministic and prevents the
# workloads from overlapping.
_GPU_SLOT = threading.BoundedSemaphore(value=1)


@contextmanager
def claim_gpu_slot() -> Iterator[None]:
    _GPU_SLOT.acquire()
    try:
        yield
    finally:
        _GPU_SLOT.release()
