#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Product compatibility wrapper for frozen V1 Stage-1 RF-DETR inference.

The frozen Stage-1 source remains byte-for-byte unchanged.  The product backend
selected RF-DETR play threshold is injected only at runtime by assigning the
module-level ``CONF`` constant before calling the frozen module's ``main``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kickclip_frozen_stage1_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen Stage-1 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--frozen-stage1", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    args, remaining = parser.parse_known_args()

    source = args.frozen_stage1.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    threshold = float(args.confidence_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in (0, 1]")

    module = _load_module(source)
    if not hasattr(module, "CONF") or not callable(getattr(module, "main", None)):
        raise RuntimeError("Frozen Stage-1 runtime contract is incompatible")

    original = float(module.CONF)
    module.CONF = threshold
    print(
        "KickClip Stage-1 product compatibility wrapper: "
        f"frozen_CONF={original:.6f} runtime_CONF={threshold:.6f}",
        flush=True,
    )

    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(source), *remaining]
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"Stage-1 compatibility wrapper fatal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
