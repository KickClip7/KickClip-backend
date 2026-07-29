from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


CHAMPION = "soccer_spotter_v9"
EXPECTED_SHA256 = (
    "b58db272f1feb43f548e0e8ee82df9b8bdbfe1e441fb14da0376f09ec056150c"
)
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the trained SoccerSpotter v9 checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to v9_best_model.pth.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    runtime_target = (
        BACKEND_ROOT / "storage" / "models" / "action_spotting" / CHAMPION
    )
    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = _sha256(source)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected={EXPECTED_SHA256}, actual={digest}"
        )
    checkpoint = runtime_target / "v9_best_model.pth"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"{checkpoint} already exists; pass --overwrite to replace it"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, checkpoint)
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
