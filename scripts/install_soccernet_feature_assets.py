from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


ASSETS = {
    "pca_512_TF2.pkl": {
        "url": "https://raw.githubusercontent.com/SoccerNet/sn-spotting/main/Features/pca_512_TF2.pkl",
        "sha256": "dbe1f6e1dcf3a178715656aad19e25c3d90b9f0d71857d09d99c0066421fc810",
    },
    "average_512_TF2.pkl": {
        "url": "https://raw.githubusercontent.com/SoccerNet/sn-spotting/main/Features/average_512_TF2.pkl",
        "sha256": "73c4b1c9e492929552e63edc87b8dc7cecc4fe6747d9106a5e4d56bc2760556e",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the official SoccerNet PCA512 feature assets."
    )
    parser.add_argument(
        "--target-dir",
        default="external/sn-spotting/Features",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, spec in ASSETS.items():
        target = target_dir / filename
        if target.exists() and not args.overwrite:
            _verify(target, spec["sha256"])
            print(f"[OK] existing {target}")
            continue

        temporary = target.with_suffix(target.suffix + ".download")
        try:
            with urllib.request.urlopen(spec["url"], timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            _verify(temporary, spec["sha256"])
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[OK] installed {target}")


def _verify(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path}: expected={expected_sha256}, actual={digest}"
        )


if __name__ == "__main__":
    main()
