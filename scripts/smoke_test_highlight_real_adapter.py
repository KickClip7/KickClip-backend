from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.tasks.highlight_spotting.adapters.soccer_highlight_former import (
    SoccerHighlightFormerAdapter,
)
@dataclass(frozen=True)
class DummyFeatureBundle:
    path: Path
    shape: tuple[int, ...]
    dtype: str

    @property
    def available(self) -> bool:
        return True

    @property
    def layout(self) -> str:
        return "combined"

    @property
    def resolved_paths(self) -> list[Path]:
        return [self.path]

    @property
    def feature_infos(self) -> list[Any]:
        return [
            {
                "asset_id": "debug_feature_asset",
                "asset_type": "SOCCERNET_FEATURE",
                "path": self.path.as_posix(),
                "shape": self.shape,
                "dtype": self.dtype,
                "size_bytes": self.path.stat().st_size if self.path.exists() else None,
            }
        ]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "layout": self.layout,
            "asset_ids": ["debug_feature_asset"],
            "asset_types": ["SOCCERNET_FEATURE"],
            "paths": [self.path.as_posix()],
            "missing_asset_types": [],
            "validation_errors": [],
            "feature_shapes": [list(self.shape)],
            "feature_dtypes": [self.dtype],
            "feature_infos": [self.feature_infos[0]],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", required=True, help="Path to merged_feature.npy or other [T, 512] feature file")
    parser.add_argument("--model-dir", default="storage/models/highlight_spotting/champion")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    feature_path = Path(args.feature)
    array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    bundle = DummyFeatureBundle(
        path=feature_path,
        shape=tuple(int(v) for v in array.shape),
        dtype=str(array.dtype),
    )

    adapter = SoccerHighlightFormerAdapter.from_model_dir(args.model_dir)
    preflight = adapter.preflight()
    if not preflight.ready_for_real_adapter:
        raise SystemExit(json.dumps(preflight.to_metadata(), ensure_ascii=False, indent=2))

    predictions = adapter.predict(
        feature_bundle=bundle,  # type: ignore[arg-type]
        match_duration_sec=args.duration_sec,
        device=args.device,
    )

    payload = {
        "preflight": preflight.to_metadata(),
        "feature": bundle.to_metadata(),
        "num_predictions": len(predictions),
        "predictions": [prediction.to_postprocessor_input() for prediction in predictions[:20]],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote smoke report: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
