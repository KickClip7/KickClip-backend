from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


TARGET_DIR = Path("storage/models/highlight_spotting/champion")
DEFAULT_CONFIG_DIR = Path("configs/models/action_spotting/best_soccer_model")


def _copy(src: str | None, dst: Path, overwrite: bool) -> str | None:
    if src is None:
        return None

    source = Path(src)
    if not source.exists():
        raise FileNotFoundError(source)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")

    shutil.copy2(source, dst)
    return dst.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy champion highlight model files into KickClip backend storage."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt/checkpoint.pt")
    parser.add_argument("--target-dir", default=TARGET_DIR.as_posix())
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR.as_posix())
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    checkpoint_path = target_dir / "best.pt"
    copied_path = _copy(args.checkpoint, checkpoint_path, args.overwrite)
    digest = _sha256(checkpoint_path)
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        checkpoint_path.unlink(missing_ok=True)
        raise ValueError(
            f"checkpoint sha256 mismatch: expected={args.expected_sha256}, actual={digest}"
        )

    config_dir = Path(args.config_dir)
    required_configs = [
        config_dir / "model.yaml",
        config_dir / "data.yaml",
        config_dir / "inference.yaml",
        config_dir / "label_map.json",
        config_dir / "manifest.json",
    ]
    missing_configs = [path.as_posix() for path in required_configs if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError("missing model config artifact(s): " + ", ".join(missing_configs))

    print(
        json.dumps(
            {
                "checkpoint_path": copied_path,
                "checkpoint_sha256": digest,
                "config_dir": config_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
