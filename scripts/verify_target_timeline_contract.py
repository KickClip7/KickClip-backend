#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PHASE3C_SOURCE = "REAL_FROZEN_STAGE2_DIRECTIONAL_TIMELINE"
REQUIRED = {
    "frame_index",
    "time_seconds",
    "shot_id",
    "state",
    "bbox_xyxy",
    "tracking_confidence",
    "identity_confidence",
    "identity_source",
    "review_required",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.timeline.read_text(encoding="utf-8-sig"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("frames is not an array")

    missing = Counter()
    phase3c_missing = Counter()
    for frame in frames:
        if not isinstance(frame, dict):
            missing["<non-object-frame>"] += 1
            continue
        fields = REQUIRED.difference(frame)
        missing.update(fields)
        if frame.get("phase3c_source") == PHASE3C_SOURCE:
            phase3c_missing.update(fields)

    print(f"frames={len(frames)}")
    print(f"missing_required_fields={dict(sorted(missing.items()))}")
    print(f"phase3c_missing_required_fields={dict(sorted(phase3c_missing.items()))}")
    if missing:
        return 2
    print("Status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
