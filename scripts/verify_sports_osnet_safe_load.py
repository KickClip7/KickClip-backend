from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = (
    "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "kickclip_verify_sports_osnet_safe_loader",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    backend_root = Path(__file__).resolve().parents[1]
    helper = (
        backend_root
        / "tracking_source"
        / "global_ID_tracking_upgrade_v6"
        / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py"
    )
    checkpoint = (
        backend_root
        / "tracking_source"
        / "global_ID_tracking_upgrade_v6"
        / "third_party"
        / "Deep-EIoU"
        / "Deep-EIoU"
        / "checkpoints"
        / "sports_model.pth.tar-60"
    )

    if not helper.is_file():
        raise FileNotFoundError(helper)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Sports-OSNet checkpoint SHA-256 mismatch: "
            f"expected={EXPECTED_CHECKPOINT_SHA256} actual={checkpoint_sha}"
        )

    import torch

    module = load_module(helper)
    payload, contract = module.safe_load_checkpoint(torch, checkpoint)
    state_dict = module.checkpoint_state_dict(payload)

    if not state_dict:
        raise RuntimeError("Loaded checkpoint state_dict is empty.")
    if contract.get("checkpoint_loaded_with_weights_only") is not True:
        raise RuntimeError("weights_only safety contract was not preserved.")
    if contract.get("unsafe_pickle_loading_used") is not False:
        raise RuntimeError("Unsafe pickle loading was used.")
    if contract.get("safe_global_path_aliases_preserved") is not True:
        raise RuntimeError("Legacy NumPy path aliases were not preserved.")

    print("status=PASS")
    print("decision=AUTHORIZE_PHASE4B_REID_MODEL_LOAD")
    print(f"checkpoint_sha256={checkpoint_sha}")
    print(f"state_dict_tensor_count={len(state_dict)}")
    print(
        "checkpoint_loaded_with_weights_only="
        f"{contract['checkpoint_loaded_with_weights_only']}"
    )
    print(
        "safe_global_path_aliases_preserved="
        f"{contract['safe_global_path_aliases_preserved']}"
    )
    print(
        "unsafe_pickle_loading_used="
        f"{contract['unsafe_pickle_loading_used']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
