from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.tasks.highlight_spotting.adapters.soccer_spotter_v9 import (
    DEFAULT_CHAMPION_MODEL_DIR,
    SoccerSpotterV9Adapter,
)


class SmokeHalfFeatureBundle:
    def __init__(self, half1: Path, half2: Path) -> None:
        self.resolved_paths = [half1, half2]
        self.layout = "halves"
        self.assets = [object(), object()]
        self.missing_asset_types: list[str] = []
        self.validation_errors: list[str] = []
        self.feature_infos = [
            self._info("SOCCERNET_FEATURE_HALF1", half1),
            self._info("SOCCERNET_FEATURE_HALF2", half2),
        ]

    @property
    def available(self) -> bool:
        return all(
            info["shape"][1:] == [512] and info["shape"][0] > 0
            for info in self.feature_infos
        )

    def to_metadata(self, *, include_paths: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "available": self.available,
            "layout": self.layout,
            "asset_types": [info["asset_type"] for info in self.feature_infos],
            "feature_shapes": [info["shape"] for info in self.feature_infos],
        }
        if include_paths:
            value["paths"] = [path.as_posix() for path in self.resolved_paths]
        return value

    @staticmethod
    def _info(asset_type: str, path: Path) -> dict[str, Any]:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return {
            "asset_type": asset_type,
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--half1-feature", required=True)
    parser.add_argument("--half2-feature", required=True)
    parser.add_argument("--model-dir", default=DEFAULT_CHAMPION_MODEL_DIR)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = SmokeHalfFeatureBundle(
        Path(args.half1_feature),
        Path(args.half2_feature),
    )
    adapter = SoccerSpotterV9Adapter.from_model_dir(args.model_dir)
    preflight = adapter.preflight()
    if not preflight.ready_for_real_adapter:
        raise SystemExit(
            json.dumps(preflight.to_metadata(), ensure_ascii=False, indent=2)
        )
    predictions = adapter.predict(
        feature_bundle=bundle,  # type: ignore[arg-type]
        device=args.device,
    )
    print(
        json.dumps(
            {
                "preflight": preflight.to_metadata(),
                "feature": bundle.to_metadata(),
                "num_predictions": len(predictions),
                "predictions": [
                    item.to_postprocessor_input() for item in predictions[:20]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
