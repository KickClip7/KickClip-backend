from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TARGET_DIR = Path("storage/models/highlight_spotting/champion")


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
    parser.add_argument("--config", help="Path to the model config YAML/JSON used during training")
    parser.add_argument("--label-map", help="Path to label_map.json used during training/evaluation")
    parser.add_argument("--target-dir", default=TARGET_DIR.as_posix())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    copied = {
        "checkpoint_path": _copy(args.checkpoint, target_dir / "best.pt", args.overwrite),
        "config_path": _copy(args.config, target_dir / "config.yaml", args.overwrite) if args.config else None,
        "label_map_path": _copy(args.label_map, target_dir / "label_map.json", args.overwrite) if args.label_map else None,
    }

    manifest = {
        "note": "Set configs/model_registry.yaml highlight_spotting.champion.enabled=true only after the adapter and feature pipeline are ready.",
        **copied,
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nModel registry paths expected by 16회차 patch:")
    print("  checkpoint_path: storage/models/highlight_spotting/champion/best.pt")
    print("  config_path    : storage/models/highlight_spotting/champion/config.yaml")
    print("  label_map_path : storage/models/highlight_spotting/champion/label_map.json")


if __name__ == "__main__":
    main()
