from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_soccernet_feature_outputs.py <MATCH_ID>")
        return 2

    match_id = sys.argv[1]
    base = Path("storage") / "matches" / match_id / "features" / "soccernet_pca512"

    required = [
        base / "merged_feature.npy",
        base / "half1_feature.npy",
        base / "half2_feature.npy",
        base / "feature_metadata.json",
    ]

    chunks_dir = base / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npy")) if chunks_dir.exists() else []

    print(f"Feature directory: {base}")
    print(f"Chunks: {len(chunk_files)}")
    for path in chunk_files[:5]:
        print(f"  - {path}")
    if len(chunk_files) > 5:
        print("  ...")

    ok = True
    for path in required:
        exists = path.exists()
        ok = ok and exists
        print(f"{path.name}: {'OK' if exists else 'MISSING'}")

    for npy in required[:3]:
        if npy.exists():
            arr = np.load(npy, mmap_mode="r")
            print(f"{npy.name}: shape={arr.shape}, dtype={arr.dtype}")

    metadata_path = base / "feature_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        keys = [
            "merged_shape",
            "half1_shape",
            "half2_shape",
            "feature_fps",
            "split_sec",
            "split_index",
            "num_chunks",
            "chunk_sec",
            "source_asset_id",
        ]
        print("metadata summary:")
        for key in keys:
            print(f"  {key}: {metadata.get(key)}")

    return 0 if ok and chunk_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
