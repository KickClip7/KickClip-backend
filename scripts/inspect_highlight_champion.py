from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.tasks.highlight_spotting.adapters.soccer_highlight_former import (
    DEFAULT_CHAMPION_MODEL_DIR,
    SoccerHighlightFormerAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the checkpoint-paired Action Spotting Champion runtime."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_CHAMPION_MODEL_DIR,
        help=f"Champion model artifact directory. Default: {DEFAULT_CHAMPION_MODEL_DIR}",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path for the preflight report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SoccerHighlightFormerAdapter.from_model_dir(args.model_dir)
    report = adapter.preflight().to_metadata()

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\n[OK] wrote preflight report: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
