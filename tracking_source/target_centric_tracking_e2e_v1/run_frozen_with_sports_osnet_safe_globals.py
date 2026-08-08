#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import _codecs
import runpy
import sys
from pathlib import Path
from typing import Any


def register_sports_osnet_safe_globals() -> dict[str, Any]:
    import numpy as np
    import torch

    serialization = getattr(torch, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        raise RuntimeError(
            "torch.serialization.add_safe_globals is required for the trusted "
            "Sports-OSNet compatibility wrapper"
        )

    numpy_core = getattr(np, "_core", None)
    multiarray = getattr(numpy_core, "multiarray", None)
    numpy_scalar = getattr(multiarray, "scalar", None)
    if numpy_scalar is None:
        legacy_core = getattr(np, "core", None)
        legacy_multiarray = getattr(legacy_core, "multiarray", None)
        numpy_scalar = getattr(legacy_multiarray, "scalar", None)
    if numpy_scalar is None:
        raise RuntimeError("NumPy multiarray.scalar is unavailable")

    entries: list[Any] = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (numpy_scalar, "numpy._core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        (_codecs.encode, "_codecs.encode"),
    ]

    float32_dtype_class = type(np.dtype(np.float32))
    entries.append(float32_dtype_class)

    numpy_dtypes = getattr(np, "dtypes", None)
    for dtype_name in ("Float64DType", "Float32DType"):
        dtype_class = getattr(numpy_dtypes, dtype_name, None) if numpy_dtypes is not None else None
        if dtype_class is not None:
            entries.append((dtype_class, f"numpy.dtypes.{dtype_name}"))

    add_safe_globals(entries)
    return {
        "status": "REGISTERED",
        "policy_version": "SPORTS_OSNET_NUMPY2_WEIGHTS_ONLY_SUBPROCESS_COMPAT_R1",
        "weights_only_false_allowed": False,
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_frozen_with_sports_osnet_safe_globals.py "
            "<frozen-script.py> [script args ...]"
        )
    script = Path(sys.argv[1]).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    contract = register_sports_osnet_safe_globals()
    print(
        "[SAFE-WEIGHTS] trusted Sports-OSNet globals registered; "
        "weights_only=False remains forbidden",
        flush=True,
    )
    print("[SAFE-WEIGHTS] policy=" + contract["policy_version"], flush=True)

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
