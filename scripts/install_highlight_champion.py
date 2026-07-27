from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


CHAMPION = "sampling_v1_loss_v2_ms_stem_v1"
EXPECTED_SHA256 = (
    "c3aa72c3d5be98fb8c6104818da69d2e8ac9a993696830fa1a0588cd58ffaee1"
)
DEFAULT_AI_ROOT = Path(r"D:\HAESUNG\prometheus\KickClip")
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the already-trained Champion bundle from KickClip."
    )
    parser.add_argument("--ai-root", type=Path, default=DEFAULT_AI_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.ai_root
    runtime_target = (
        BACKEND_ROOT / "storage" / "models" / "action_spotting" / CHAMPION
    )
    config_target = (
        BACKEND_ROOT / "configs" / "models" / "action_spotting" / CHAMPION
    )
    pairs = [
        (
            source_root
            / "storage"
            / f"checkpoints_{CHAMPION}"
            / "transformer_best.pt",
            runtime_target / "transformer_best.pt",
        ),
        (
            source_root
            / "storage"
            / f"checkpoints_{CHAMPION}"
            / "transformer_history.json",
            runtime_target / "transformer_history.json",
        ),
        (
            source_root
            / "storage"
            / "eval"
            / "validation"
            / f"transformer_{CHAMPION}_eval_t020.json",
            runtime_target / "valid_eval.json",
        ),
        (
            source_root / "configs" / "model_ms_stem_v1.yaml",
            config_target / "model.yaml",
        ),
        (
            source_root / "configs" / "train_model_ms_stem_v1.yaml",
            config_target / "train.yaml",
        ),
        (
            source_root
            / "storage"
            / f"checkpoints_{CHAMPION}"
            / "config_snapshot.yaml",
            config_target / "config_snapshot.yaml",
        ),
    ]
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and not args.overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    checkpoint = runtime_target / "transformer_best.pt"
    digest = _sha256(checkpoint)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={EXPECTED_SHA256}, actual={digest}"
        )
    print(f"Champion bundle ready: {CHAMPION}")
    print(f"checkpoint_sha256={digest}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
