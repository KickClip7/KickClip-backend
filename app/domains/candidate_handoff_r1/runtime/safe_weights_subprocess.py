from __future__ import annotations

import runpy
import sys
from pathlib import Path


ALLOWED_FROZEN_SCRIPTS = {
    "stage2b_run_same_shot_reentry_reacquisition.py",
    "stage3a2_build_precut_target_memory.py",
    "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
}


def register_safe_globals() -> None:
    """Keep weights_only=True while admitting checkpoint NumPy scalar metadata."""

    import numpy as np
    import torch

    safe = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
    ]
    for name in ("float16", "float32", "float64", "int32", "int64", "uint8"):
        safe.append(type(np.dtype(name)))
    torch.serialization.add_safe_globals(safe)


def main() -> int:
    if len(sys.argv) < 2:
        raise ValueError("A frozen script path is required")
    script = Path(sys.argv[1]).expanduser().resolve()
    if script.name not in ALLOWED_FROZEN_SCRIPTS or not script.is_file():
        raise ValueError(f"Frozen safe-weights script is not allowed: {script}")
    register_safe_globals()
    sys.argv = [str(script), *sys.argv[2:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
