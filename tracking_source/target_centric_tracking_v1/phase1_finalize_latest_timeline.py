#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finalize the newest valid Phase-1 timeline into stable final_* artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

UNCERTAIN = {"LOST", "SEARCHING", "AMBIGUOUS", "ABSENT", "TERMINATED"}
CONFIRMED = {"INITIALIZING", "ACTIVE", "ACTIVE_LOW_CONFIDENCE", "REACQUIRED", "USER_CONFIRMED"}
OUTPUTS = (
    "final_target_timeline.json",
    "final_frame_observations.csv",
    "final_crop_trajectory.csv",
    "final_target_tracking_preview.mp4",
    "final_target_centered_preview.mp4",
    "final_reentry_episodes.json",
    "final_audit.json",
    "final_report.md",
)
TIMELINE_PRIORITY = (
    ("stage2d2", "stage2d2_target_timeline.json"),
    ("stage2d1", "stage2d1_target_timeline.json"),
    ("stage2d", "stage2d_target_timeline.json"),
    ("stage2b", "stage2b_target_timeline.json"),
    ("stage2c", "stage2c_target_timeline.json"),
    ("stage2", "target_timeline.json"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--test-name", required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/target_centric_tracking_v1"))
    p.add_argument("--target-screen-height-ratio", type=float, default=0.42)
    p.add_argument("--min-crop-height-ratio", type=float, default=0.38)
    p.add_argument("--center-alpha", type=float, default=0.22)
    p.add_argument("--scale-alpha", type=float, default=0.16)
    p.add_argument("--lost-return-alpha", type=float, default=0.08)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--overwrite-final", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare(test_dir: Path, overwrite: bool) -> None:
    existing = [test_dir / name for name in OUTPUTS if (test_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError("Final outputs exist; use --overwrite-final:\n" + "\n".join(map(str, existing)))
    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(path)
        path.unlink()


def select_timeline(test_dir: Path) -> tuple[str, Path, dict[str, Any]]:
    for stage, name in TIMELINE_PRIORITY:
        path = test_dir / name
        if not path.is_file():
            continue
        value = read_json(path)
        frames = value.get("frames")
        video = value.get("video")
        if isinstance(frames, list) and frames and isinstance(video, dict):
            return stage, path, value
    raise FileNotFoundError("No valid target timeline found in " + str(test_dir))


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_frames(raw: list[Any], frame_count: int, width: int, height: int) -> list[dict[str, Any]]:
    if len(raw) != frame_count:
        raise ValueError(f"Frame count mismatch: timeline={len(raw)} video={frame_count}")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or int(item.get("frame_index", -1)) != index:
            raise ValueError(f"Invalid frame row: {index}")
        row = dict(item)
        state = str(row.get("state", "SEARCHING"))
        bbox = row.get("bbox_xyxy")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox schema at {index}")
            bbox = [number(v) for v in bbox]
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Out-of-bounds bbox at {index}: {bbox}")
            row["bbox_xyxy"] = bbox
        if state in UNCERTAIN and bbox is not None:
            raise ValueError(f"Uncertain state has bbox at {index}")
        if state in CONFIRMED and bbox is None:
            raise ValueError(f"Confirmed state lacks bbox at {index}")
        row["state"] = state
        row["tracking_confidence"] = min(1.0, max(0.0, number(row.get("tracking_confidence"))))
        row["identity_confidence"] = min(1.0, max(0.0, number(row.get("identity_confidence"))))
        row.setdefault("selected_detection_id", None)
        row.setdefault("decision_reason", "")
        row.setdefault("review_required", False)
        result.append(row)
    return result


def bbox_center(box: list[float]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def crop_rect(center_x: float, center_y: float, crop_h: float, width: int, height: int) -> list[float]:
    aspect = width / height
    crop_h = max(2.0, min(float(height), crop_h))
    crop_w = crop_h * aspect
    if crop_w > width:
        crop_w = float(width)
        crop_h = crop_w / aspect
    x1 = center_x - crop_w / 2
    y1 = center_y - crop_h / 2
    x1 = min(max(0.0, x1), width - crop_w)
    y1 = min(max(0.0, y1), height - crop_h)
    return [x1, y1, x1 + crop_w, y1 + crop_h]


def build_crop_trajectory(
    frames: list[dict[str, Any]], width: int, height: int,
    target_ratio: float, min_crop_ratio: float, center_alpha: float,
    scale_alpha: float, lost_alpha: float,
) -> list[dict[str, Any]]:
    full_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    smooth_center = full_center.copy()
    smooth_height = float(height)
    rows: list[dict[str, Any]] = []
    for item in frames:
        bbox = item.get("bbox_xyxy")
        if bbox is not None:
            cx, cy = bbox_center(bbox)
            desired_center = np.array([cx, cy], dtype=np.float64)
            bbox_h = bbox[3] - bbox[1]
            desired_height = bbox_h / max(target_ratio, 1e-6)
            desired_height = min(float(height), max(height * min_crop_ratio, desired_height))
            smooth_center = (1 - center_alpha) * smooth_center + center_alpha * desired_center
            smooth_height = (1 - scale_alpha) * smooth_height + scale_alpha * desired_height
            source = "TARGET_BBOX"
        else:
            smooth_center = (1 - lost_alpha) * smooth_center + lost_alpha * full_center
            smooth_height = (1 - lost_alpha) * smooth_height + lost_alpha * float(height)
            source = "FULL_FRAME_FALLBACK"
        rect = crop_rect(float(smooth_center[0]), float(smooth_center[1]), smooth_height, width, height)
        rows.append({
            "frame_index": item["frame_index"],
            "time_seconds": item["frame_index"] / max(number(item.get("fps"), 0.0), 1.0),
            "state": item["state"],
            "crop_x1": f"{rect[0]:.4f}", "crop_y1": f"{rect[1]:.4f}",
            "crop_x2": f"{rect[2]:.4f}", "crop_y2": f"{rect[3]:.4f}",
            "crop_source": source,
            "target_bbox_present": bbox is not None,
        })
    return rows


def color_for_state(state: str) -> tuple[int, int, int]:
    if state in {"ACTIVE", "INITIALIZING", "USER_CONFIRMED"}:
        return (60, 220, 60)
    if state == "ACTIVE_LOW_CONFIDENCE":
        return (0, 200, 255)
    if state == "REACQUIRED":
        return (255, 80, 255)
    if state in {"OCCLUDED", "SEARCHING", "LOST"}:
        return (0, 170, 255)
    if state == "AMBIGUOUS":
        return (0, 80, 255)
    return (180, 180, 180)


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def render_videos(
    video: Path, tracking_path: Path, centered_path: Path,
    frames: list[dict[str, Any]], crop_rows: list[dict[str, Any]],
    fps: float, width: int, height: int, print_every: int,
) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    tracking_writer = open_writer(tracking_path, fps, width, height)
    centered_writer = open_writer(centered_path, fps, width, height)
    index = 0
    try:
        while index < len(frames):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {index}")
            item = frames[index]
            state = item["state"]
            annotated = image.copy()
            bbox = item.get("bbox_xyxy")
            color = color_for_state(state)
            if bbox is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"target_001 | {state} | t={item['tracking_confidence']:.2f} i={item['identity_confidence']:.2f}"
            cv2.rectangle(annotated, (0, 0), (min(width - 1, 620), 30), (0, 0, 0), -1)
            cv2.putText(annotated, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            tracking_writer.write(annotated)

            crop = crop_rows[index]
            x1 = int(round(float(crop["crop_x1"])))
            y1 = int(round(float(crop["crop_y1"])))
            x2 = int(round(float(crop["crop_x2"])))
            y2 = int(round(float(crop["crop_y2"])))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, max(x1 + 2, x2)), min(height, max(y1 + 2, y2))
            crop_image = image[y1:y2, x1:x2]
            centered = cv2.resize(crop_image, (width, height), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(centered, (0, 0), (min(width - 1, 430), 30), (0, 0, 0), -1)
            cv2.putText(centered, f"{state} | {crop['crop_source']}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            centered_writer.write(centered)
            index += 1
            if print_every > 0 and index % print_every == 0:
                print(f"[FINALIZER] frames={index}/{len(frames)}", flush=True)
    finally:
        cap.release()
        tracking_writer.release()
        centered_writer.release()
    if index != len(frames):
        raise RuntimeError("Rendered frame count mismatch")


def flatten_frame(item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item)
    bbox = row.get("bbox_xyxy")
    if bbox is None:
        row.update(bbox_x1="", bbox_y1="", bbox_x2="", bbox_y2="")
    else:
        row.update(bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3])
    row.pop("bbox_xyxy", None)
    for key, value in list(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return row


def build_report(audit: Mapping[str, Any]) -> str:
    counts = audit["counts"]
    return f"""# KickClip Phase 1 Frozen Final Report

- Status: `{audit['status']}`
- Source stage: `{audit['source_stage']}`
- Frame count: `{counts['frame_count']}`
- Bbox frames: `{counts['bbox_frame_count']}` ({counts['bbox_coverage_percent']:.2f}%)
- Reacquired frames: `{counts['state_counts'].get('REACQUIRED', 0)}`
- Review-required frames: `{counts['review_required_frame_count']}`
- Silent wrong-player switch: **must be assessed by visual review; never inferred from coverage alone**

## Frozen safety contract

- No threshold search in the finalizer.
- No detector or ReID inference in the finalizer.
- No bbox is fabricated when the selected timeline has no target observation.
- LOST / SEARCHING / AMBIGUOUS states remain explicit.
- Tracking trajectory and editing crop trajectory are separate outputs.
"""


def main() -> int:
    a = parse_args()
    root = a.project_root.expanduser().resolve()
    test_dir = resolve(root, a.output_root) / a.test_name
    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)
    prepare(test_dir, a.overwrite_final)
    source_stage, source_path, source = select_timeline(test_dir)
    video_info = source["video"]
    video = Path(video_info["path"]).expanduser()
    if not video.is_absolute():
        video = (root / video).resolve()
    else:
        video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError("Invalid video metadata")
    frames = normalize_frames(source["frames"], frame_count, width, height)
    for frame in frames:
        frame["fps"] = fps
    crop_rows = build_crop_trajectory(
        frames, width, height, a.target_screen_height_ratio, a.min_crop_height_ratio,
        a.center_alpha, a.scale_alpha, a.lost_return_alpha,
    )
    episodes = source.get("episodes") if isinstance(source.get("episodes"), list) else []
    final_timeline = {
        "schema_version": "kickclip.phase1_final_target_timeline.v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "FROZEN_PHASE1_FINALIZED",
        "target_id": str(source.get("target_id", "target_001")),
        "test_name": a.test_name,
        "video": {
            "path": str(video), "sha256": sha256(video), "width": width, "height": height,
            "fps": fps, "frame_count": frame_count,
        },
        "states": sorted(set(frame["state"] for frame in frames)),
        "episodes": episodes,
        "frames": frames,
        "provenance": {
            "source_stage": source_stage,
            "source_timeline": str(source_path),
            "source_timeline_sha256": sha256(source_path),
            "detector_inference": False,
            "reid_inference": False,
            "threshold_search": False,
            "crop_trajectory_separate_from_tracking": True,
        },
    }
    atomic_json(test_dir / "final_target_timeline.json", final_timeline)
    write_csv(test_dir / "final_frame_observations.csv", [flatten_frame(frame) for frame in frames])
    write_csv(test_dir / "final_crop_trajectory.csv", crop_rows)
    atomic_json(test_dir / "final_reentry_episodes.json", {
        "schema_version": "kickclip.phase1_final_reentry_episodes.v1",
        "source_stage": source_stage,
        "episodes": episodes,
    })
    if not a.no_video:
        render_videos(
            video, test_dir / "final_target_tracking_preview.mp4",
            test_dir / "final_target_centered_preview.mp4", frames, crop_rows,
            fps, width, height, a.print_every,
        )
    states = Counter(frame["state"] for frame in frames)
    bbox_count = sum(frame.get("bbox_xyxy") is not None for frame in frames)
    audit = {
        "schema_version": "kickclip.phase1_final_audit.v1",
        "status": "PASS",
        "decision": "AUTHORIZE_MANDATORY_FINAL_VISUAL_REVIEW",
        "source_stage": source_stage,
        "source_timeline": str(source_path),
        "counts": {
            "frame_count": frame_count,
            "bbox_frame_count": bbox_count,
            "bbox_coverage_percent": 100.0 * bbox_count / frame_count,
            "state_counts": dict(states),
            "review_required_frame_count": sum(bool(frame.get("review_required")) for frame in frames),
            "episode_count": len(episodes),
        },
        "invariants": {
            "frame_indices_contiguous": True,
            "uncertain_states_have_null_bbox": True,
            "confirmed_states_have_bbox": True,
            "tracking_and_crop_trajectories_separate": True,
            "detector_inference": False,
            "reid_inference": False,
            "threshold_search": False,
        },
        "outputs": {name: str(test_dir / name) for name in OUTPUTS if (test_dir / name).exists() or name.endswith(".mp4")},
    }
    atomic_json(test_dir / "final_audit.json", audit)
    atomic_text(test_dir / "final_report.md", build_report(audit))
    print("KickClip Phase-1 finalization complete")
    print(f"Source stage : {source_stage}")
    print(f"Frames       : {frame_count}")
    print(f"BBox frames  : {bbox_count} ({100.0 * bbox_count / frame_count:.2f}%)")
    print(f"States       : {dict(states)}")
    print(f"Output       : {test_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Finalizer fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
