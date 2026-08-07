#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate frozen Phase-1 final outputs without changing them."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

UNCERTAIN = {"LOST", "SEARCHING", "AMBIGUOUS", "ABSENT", "TERMINATED"}
CONFIRMED = {"INITIALIZING", "ACTIVE", "ACTIVE_LOW_CONFIDENCE", "REACQUIRED", "USER_CONFIRMED"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--test-name", required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/target_centric_tracking_v1"))
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    a = parse_args()
    root = a.project_root.expanduser().resolve()
    test_dir = resolve(root, a.output_root) / a.test_name
    timeline_path = test_dir / "final_target_timeline.json"
    observations_path = test_dir / "final_frame_observations.csv"
    crop_path = test_dir / "final_crop_trajectory.csv"
    audit_path = test_dir / "final_audit.json"
    report_path = test_dir / "final_report.md"
    for path in (timeline_path, observations_path, crop_path, audit_path, report_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    timeline = read_json(timeline_path)
    if timeline.get("schema_version") != "kickclip.phase1_final_target_timeline.v1":
        raise RuntimeError("Unexpected timeline schema")
    video = timeline.get("video") or {}
    frame_count = int(video.get("frame_count", 0))
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    frames = timeline.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count or frame_count <= 0:
        raise RuntimeError("Frame count mismatch")
    bbox_count = 0
    for index, frame in enumerate(frames):
        if int(frame.get("frame_index", -1)) != index:
            raise RuntimeError(f"Non-contiguous frame index: {index}")
        state = str(frame.get("state"))
        bbox = frame.get("bbox_xyxy")
        if state in UNCERTAIN and bbox is not None:
            raise RuntimeError(f"Uncertain state has bbox: {index}")
        if state in CONFIRMED and bbox is None:
            raise RuntimeError(f"Confirmed state lacks bbox: {index}")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise RuntimeError(f"Invalid bbox: {index}")
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise RuntimeError(f"Non-finite bbox: {index}")
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise RuntimeError(f"Out-of-bounds bbox: {index}")
            bbox_count += 1
        for key in ("tracking_confidence", "identity_confidence"):
            value = float(frame.get(key, -1))
            if not 0 <= value <= 1:
                raise RuntimeError(f"Invalid {key}: frame {index}")

    with observations_path.open(encoding="utf-8-sig", newline="") as f:
        observation_rows = list(csv.DictReader(f))
    with crop_path.open(encoding="utf-8-sig", newline="") as f:
        crop_rows = list(csv.DictReader(f))
    if len(observation_rows) != frame_count or len(crop_rows) != frame_count:
        raise RuntimeError("CSV row count mismatch")

    source_video = Path(video["path"])
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    if video.get("sha256") and sha256(source_video) != video["sha256"]:
        raise RuntimeError("Source video hash mismatch")

    audit = read_json(audit_path)
    if audit.get("status") != "PASS":
        raise RuntimeError("Final audit is not PASS")
    if int(audit["counts"]["bbox_frame_count"]) != bbox_count:
        raise RuntimeError("Audit bbox count mismatch")

    print("KickClip Phase-1 final output validation complete")
    print("Status      : PASS")
    print(f"Frames      : {frame_count}")
    print(f"BBox frames : {bbox_count}")
    print(f"Source stage: {timeline['provenance']['source_stage']}")
    print(f"Output      : {test_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Validation fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
