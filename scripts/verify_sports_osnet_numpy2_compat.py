#!/usr/bin/env python
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

EXPECTED_SHA256 = "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    backend_root = Path(__file__).resolve().parents[1]
    core_path = backend_root / "tracking_source/target_centric_tracking_e2e_v1/run_target_centric_pipeline_core.py"
    reid_path = backend_root / "tracking_source/global_ID_tracking_upgrade_v6/stage2b1_extract_frozen_tracking_reid_embeddings_v6.py"
    checkpoint = backend_root / "tracking_source/global_ID_tracking_upgrade_v6/third_party/Deep-EIoU/Deep-EIoU/checkpoints/sports_model.pth.tar-60"

    for path in (core_path, reid_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    actual_sha = sha256_file(checkpoint)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: expected={EXPECTED_SHA256} actual={actual_sha}")

    import numpy as np
    import torch

    core = load_module("kickclip_verify_product_core", core_path)
    reid = load_module("kickclip_verify_reid_safe", reid_path)

    unsafe = []
    getter = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if getter is not None:
        unsafe = sorted(set(getter(str(checkpoint))))

    compat = core._register_product_sports_osnet_safe_globals(torch)
    payload, contract = reid.safe_load_checkpoint(torch, checkpoint)
    state_dict = reid.checkpoint_state_dict(payload)

    if not state_dict:
        raise RuntimeError("Sports-OSNet state_dict is empty")
    if contract.get("checkpoint_loaded_with_weights_only") is not True:
        raise RuntimeError("weights_only safety contract was not preserved")
    if contract.get("unsafe_pickle_loading_used") is not False:
        raise RuntimeError("Unsafe pickle loading was used")

    print("status=PASS")
    print("decision=AUTHORIZE_SPORTS_OSNET_PRODUCT_LOAD")
    print(f"torch={torch.__version__}")
    print(f"numpy={np.__version__}")
    print(f"checkpoint_sha256={actual_sha}")
    print(f"unsafe_globals_before_allowlist={unsafe}")
    print(f"compat_policy={compat['policy_version']}")
    print(f"registered_safe_globals={compat['registered_safe_globals']}")
    print(f"state_dict_tensor_count={len(state_dict)}")
    print("weights_only=True")
    print("weights_only_false_allowed=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
