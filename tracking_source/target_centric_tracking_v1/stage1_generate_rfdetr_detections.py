#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 1 RF-DETR detections.

Reads a PASS Stage-0 directory, reuses only load_rfdetr_model() and
normalize_detections() from the frozen legacy RF-DETR source, runs full-frame
RF-DETR inference, and audits the first-frame user target bbox.

No tracking, ReID, training, threshold search, GTA, or Global-ID linking.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Sequence

import cv2
import numpy as np

VERSION = "target-centric-v1-stage1-1.0.0"
STAGE = "stage1_generate_frozen_rfdetr_detections"
EXPECTED_CHECKPOINT_SHA256 = (
    "5d1d05cf78b6a777430430955d0e745ce5c61913c659b2e96ae5deddbd1a3a94"
)
EXPECTED_SOURCE_SHA256 = (
    "d8af67842b2c8197e812dc2374058fd5ea07750b1836393e5cc70ca276022555"
)
CONF = 0.25
MODEL_SIZE = "small"
CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "staff",
    4: "ball",
}
TARGET_CLASSES = {0, 1}
COLORS = {
    0: (50, 220, 50),
    1: (255, 80, 255),
    2: (0, 220, 255),
    3: (255, 220, 50),
    4: (0, 128, 255),
}
STAGE1_FILES = (
    "detections.csv",
    "initial_target_detection.json",
    "initial_target_detection_preview.jpg",
    "stage1_detection_summary.json",
    "stage1_detection_report.md",
    "rfdetr_detections_preview.mp4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--test-name",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "runs/target_centric_tracking_v1"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "weights/rfdetr/checkpoint_best_regular.pth"
        ),
    )
    parser.add_argument(
        "--detector-source",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-stage1",
        action="store_true",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (root / path).resolve()
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_sha256(path: Path) -> str:
    text = path.read_text(
        encoding="utf-8-sig"
    )
    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def atomic_text(
    path: Path,
    text: str,
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_json(
    path: Path,
    value: Any,
) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def validate_test_name(
    name: str,
) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if (
        not name
        or name in {".", ".."}
        or any(
            char not in allowed
            for char in name
        )
    ):
        raise ValueError(
            "Invalid --test-name"
        )

    return name


def prepare_outputs(
    test_dir: Path,
    overwrite: bool,
) -> None:
    existing = [
        test_dir / name
        for name in STAGE1_FILES
        if (test_dir / name).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Stage-1 outputs already exist; "
            "use --overwrite-stage1:\n"
            + "\n".join(
                map(str, existing)
            )
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(path)
        path.unlink()

    for path in test_dir.glob(
        "*.stage1.tmp*"
    ):
        if path.is_file():
            path.unlink()


def find_detector_source(
    root: Path,
    explicit: Optional[Path],
) -> tuple[Path, str]:
    if explicit is not None:
        candidates = [
            resolve(
                root,
                explicit,
            )
        ]
    else:
        candidates = [
            root
            / "global_ID_linker"
            / "stage1_rfdetr_hybrid_track_video.py",

            root
            / "backups"
            / "global_tracking_baseline_stage10_20260711"
            / "global_ID_linker"
            / "stage1_rfdetr_hybrid_track_video.py",

            root
            / "global_ID_tracking_upgrade_v6"
            / "stage1_rfdetr_hybrid_track_video.py",
        ]

        for base in (
            root / "global_ID_linker",
            root / "global_ID_tracking_upgrade_v6",
            root / "backups",
        ):
            if base.is_dir():
                candidates.extend(
                    base.rglob(
                        "stage1_rfdetr_hybrid_track_video.py"
                    )
                )

    seen: set[Path] = set()
    found: list[tuple[Path, str]] = []

    for candidate in candidates:
        candidate = candidate.resolve()

        if (
            candidate in seen
            or not candidate.is_file()
        ):
            continue

        seen.add(candidate)

        digest = canonical_sha256(
            candidate
        )

        found.append(
            (
                candidate,
                digest,
            )
        )

        if digest == EXPECTED_SOURCE_SHA256:
            return candidate, digest

    if explicit is not None and not found:
        raise FileNotFoundError(
            f"Detector source not found: "
            f"{candidates[0]}"
        )

    details = "\n".join(
        f"{path} -> {digest}"
        for path, digest in found
    )

    raise RuntimeError(
        "No detector source matched "
        "the frozen canonical SHA-256 "
        f"{EXPECTED_SOURCE_SHA256}.\n"
        f"{details}"
    )


def import_detector_source(
    path: Path,
    root: Path,
) -> ModuleType:
    for candidate in (
        root,
        root / "v1",
        root / "v1" / "rfdetr",
        path.parent,
    ):
        if (
            candidate.exists()
            and str(candidate)
            not in sys.path
        ):
            sys.path.insert(
                0,
                str(candidate),
            )

    spec = (
        importlib.util
        .spec_from_file_location(
            "kickclip_target_v1_frozen_rfdetr",
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Cannot import: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name in (
        "load_rfdetr_model",
        "normalize_detections",
    ):
        if not callable(
            getattr(
                module,
                name,
                None,
            )
        ):
            raise AttributeError(
                "Frozen detector source "
                f"lacks {name}(): {path}"
            )

    return module


def resolve_device(
    requested: str,
) -> tuple[str, Any]:
    import torch

    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but "
            "torch.cuda.is_available() "
            "is false"
        )

    return requested, torch


def bbox_iou(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    ax1, ay1, ax2, ay2 = map(
        float,
        a,
    )
    bx1, by1, bx2, by2 = map(
        float,
        b,
    )

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )
    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def target_coverage(
    target: Sequence[float],
    candidate: Sequence[float],
) -> float:
    tx1, ty1, tx2, ty2 = map(
        float,
        target,
    )
    cx1, cy1, cx2, cy2 = map(
        float,
        candidate,
    )

    intersection = (
        max(
            0.0,
            min(tx2, cx2)
            - max(tx1, cx1),
        )
        * max(
            0.0,
            min(ty2, cy2)
            - max(ty1, cy1),
        )
    )

    area = (
        max(0.0, tx2 - tx1)
        * max(0.0, ty2 - ty1)
    )

    return (
        intersection / area
        if area > 0
        else 0.0
    )


def center_inside(
    outer: Sequence[float],
    inner: Sequence[float],
) -> bool:
    ox1, oy1, ox2, oy2 = map(
        float,
        outer,
    )
    ix1, iy1, ix2, iy2 = map(
        float,
        inner,
    )

    center_x = (
        ix1 + ix2
    ) / 2.0
    center_y = (
        iy1 + iy2
    ) / 2.0

    return (
        ox1 <= center_x <= ox2
        and oy1 <= center_y <= oy2
    )


def center_distance(
    target: Sequence[float],
    candidate: Sequence[float],
) -> float:
    tx1, ty1, tx2, ty2 = map(
        float,
        target,
    )
    cx1, cy1, cx2, cy2 = map(
        float,
        candidate,
    )

    distance = math.hypot(
        (
            tx1
            + tx2
            - cx1
            - cx2
        )
        / 2.0,
        (
            ty1
            + ty2
            - cy1
            - cy2
        )
        / 2.0,
    )

    diagonal = math.hypot(
        tx2 - tx1,
        ty2 - ty1,
    )

    return (
        distance / diagonal
        if diagonal > 0
        else float("inf")
    )


def match_features(
    target: Sequence[float],
    candidate: Sequence[float],
) -> dict[str, Any]:
    iou = bbox_iou(
        target,
        candidate,
    )
    coverage = target_coverage(
        target,
        candidate,
    )

    candidate_center_in_target = (
        center_inside(
            target,
            candidate,
        )
    )

    target_center_in_candidate = (
        center_inside(
            candidate,
            target,
        )
    )

    distance = center_distance(
        target,
        candidate,
    )

    score = (
        0.70 * iou
        + 0.20 * coverage
        + 0.10
        * max(
            0.0,
            1.0
            - min(distance, 1.0),
        )
    )

    reliable = (
        iou >= 0.20
        or (
            coverage >= 0.60
            and target_center_in_candidate
        )
        or (
            iou >= 0.10
            and candidate_center_in_target
            and target_center_in_candidate
        )
    )

    return {
        "iou": round(iou, 6),
        "target_coverage": round(
            coverage,
            6,
        ),
        "candidate_center_inside_target": (
            candidate_center_in_target
        ),
        "target_center_inside_candidate": (
            target_center_in_candidate
        ),
        "normalized_center_distance": round(
            distance,
            6,
        ),
        "match_score": round(
            score,
            6,
        ),
        "reliable": bool(reliable),
    }


def select_initial_target(
    target_bbox: Sequence[float],
    detections: Sequence[
        dict[str, Any]
    ],
) -> tuple[
    Optional[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []

    for index, detection in enumerate(
        detections
    ):
        if (
            int(detection["class_id"])
            not in TARGET_CLASSES
        ):
            continue

        rows.append(
            {
                "detection_index": index,
                "class_id": int(
                    detection["class_id"]
                ),
                "class_name": str(
                    detection["class_name"]
                ),
                "confidence": round(
                    float(
                        detection["conf"]
                    ),
                    6,
                ),
                "bbox_xyxy": [
                    round(
                        float(value),
                        4,
                    )
                    for value in detection[
                        "xyxy"
                    ]
                ],
                **match_features(
                    target_bbox,
                    detection["xyxy"],
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["match_score"],
            row["iou"],
            row["confidence"],
        ),
        reverse=True,
    )

    return (
        rows[0]
        if rows
        else None
    ), rows


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1

    (
        width,
        height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    x = max(
        0,
        min(
            frame.shape[1]
            - width
            - 4,
            x,
        ),
    )
    y = max(
        height + 4,
        min(
            frame.shape[0]
            - baseline
            - 2,
            y,
        ),
    )

    cv2.rectangle(
        frame,
        (
            x,
            y - height - 4,
        ),
        (
            x + width + 4,
            y + baseline + 2,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x + 2,
            y - 2,
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_preview(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    detections: Sequence[
        dict[str, Any]
    ],
    target_bbox: Sequence[float],
    best: Optional[dict[str, Any]],
) -> np.ndarray:
    output = frame.copy()

    for index, detection in enumerate(
        detections
    ):
        class_id = int(
            detection["class_id"]
        )

        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in detection[
                "xyxy"
            ]
        ]

        color = COLORS.get(
            class_id,
            (220, 220, 220),
        )
        thickness = 1

        if (
            frame_index == 0
            and best is not None
            and index
            == int(
                best["detection_index"]
            )
        ):
            color = (
                (0, 0, 255)
                if best["reliable"]
                else (0, 128, 255)
            )
            thickness = 3

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        draw_label(
            output,
            (
                f"{detection['class_name']} "
                f"{float(detection['conf']):.2f}"
            ),
            x1,
            y1,
            color,
        )

    if frame_index == 0:
        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in target_bbox
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3,
        )

        draw_label(
            output,
            "USER TARGET",
            x1,
            y1 - 5,
            (255, 255, 255),
        )

    text = (
        "RF-DETR Stage 1"
        f" | frame={frame_index}"
        f" | t={frame_index / fps:.3f}s"
        f" | dets={len(detections)}"
    )

    cv2.rectangle(
        output,
        (0, 0),
        (
            output.shape[1],
            24,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def csv_fields() -> list[str]:
    return [
        "frame_index",
        "frame_1based",
        "time_ms",
        "detection_index",
        "detection_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "center_x",
        "center_y",
        "box_area",
        "box_area_ratio",
        "box_height_ratio",
        "is_target_candidate_class",
        "initial_target_iou",
        "initial_target_coverage",
        "initial_candidate_center_inside_target",
        "initial_target_center_inside_candidate",
        "initial_normalized_center_distance",
        "initial_match_score",
        "initial_match_reliable",
    ]


def detection_row(
    frame_index: int,
    fps: float,
    detection_index: int,
    detection: dict[str, Any],
    frame_width: int,
    frame_height: int,
    initial: Optional[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    x1, y1, x2, y2 = map(
        float,
        detection["xyxy"],
    )

    width = x2 - x1
    height = y2 - y1
    area = width * height
    empty = ""

    return {
        "frame_index": frame_index,
        "frame_1based": (
            frame_index + 1
        ),
        "time_ms": int(
            round(
                frame_index
                * 1000.0
                / fps
            )
        ),
        "detection_index": (
            detection_index
        ),
        "detection_id": (
            f"f{frame_index:06d}"
            f"_d{detection_index:03d}"
        ),
        "class_id": int(
            detection["class_id"]
        ),
        "class_name": str(
            detection["class_name"]
        ),
        "confidence": (
            f"{float(detection['conf']):.6f}"
        ),
        "x1": f"{x1:.4f}",
        "y1": f"{y1:.4f}",
        "x2": f"{x2:.4f}",
        "y2": f"{y2:.4f}",
        "width": f"{width:.4f}",
        "height": f"{height:.4f}",
        "center_x": (
            f"{(x1 + x2) / 2.0:.4f}"
        ),
        "center_y": (
            f"{(y1 + y2) / 2.0:.4f}"
        ),
        "box_area": f"{area:.4f}",
        "box_area_ratio": (
            f"{area / (frame_width * frame_height):.8f}"
        ),
        "box_height_ratio": (
            f"{height / frame_height:.8f}"
        ),
        "is_target_candidate_class": (
            int(
                detection["class_id"]
            )
            in TARGET_CLASSES
        ),
        "initial_target_iou": (
            empty
            if initial is None
            else f"{initial['iou']:.6f}"
        ),
        "initial_target_coverage": (
            empty
            if initial is None
            else (
                f"{initial['target_coverage']:.6f}"
            )
        ),
        "initial_candidate_center_inside_target": (
            empty
            if initial is None
            else initial[
                "candidate_center_inside_target"
            ]
        ),
        "initial_target_center_inside_candidate": (
            empty
            if initial is None
            else initial[
                "target_center_inside_candidate"
            ]
        ),
        "initial_normalized_center_distance": (
            empty
            if initial is None
            else (
                f"{initial['normalized_center_distance']:.6f}"
            )
        ),
        "initial_match_score": (
            empty
            if initial is None
            else (
                f"{initial['match_score']:.6f}"
            )
        ),
        "initial_match_reliable": (
            empty
            if initial is None
            else initial["reliable"]
        ),
    }


def report(
    summary: dict[str, Any],
    initialization: dict[str, Any],
) -> str:
    counts = summary["counts"]

    lines = [
        (
            "# KickClip Target-Centric "
            "Tracking V1 — Stage 1 "
            "RF-DETR Detections"
        ),
        "",
        f"- Status: `{summary['status']}`",
        (
            f"- Decision: "
            f"`{summary['decision']}`"
        ),
        (
            f"- Video: "
            f"`{summary['video']['path']}`"
        ),
        (
            f"- Geometry: "
            f"`{summary['video']['width']}"
            f"×{summary['video']['height']}`"
        ),
        (
            f"- FPS / frames: "
            f"`{summary['video']['fps']}` / "
            f"`{summary['video']['processed_frames']}`"
        ),
        (
            f"- Total detections: "
            f"`{counts['total_detections']}`"
        ),
        (
            f"- Processing speed: "
            f"`{summary['runtime']['processing_fps']:.3f} fps`"
        ),
        "",
        "## Frozen contract",
        "",
        (
            f"- RF-DETR Small, "
            f"confidence `{CONF}`"
        ),
        "- All five classes retained",
        (
            "- No extra NMS, scene filtering, "
            "class filtering, or bbox-size filtering"
        ),
        (
            "- Reused functions: "
            "`load_rfdetr_model`, "
            "`normalize_detections`"
        ),
        "",
        "## Initial target",
        "",
        (
            f"- User bbox: "
            f"`{initialization['authoritative_user_bbox_xyxy']}`"
        ),
        (
            f"- Status: "
            f"`{initialization['status']}`"
        ),
        (
            f"- Reliable anchor: "
            f"`{initialization['reliable_detector_anchor']}`"
        ),
        (
            f"- Best candidate: "
            f"`{initialization['best_candidate']}`"
        ),
        "",
        (
            "The user bbox remains authoritative; "
            "the detector candidate is an "
            "audit/association anchor only."
        ),
        "",
        "## Exclusions",
        "",
        (
            "- Tracking / ReID / training / "
            "threshold search / GTA / "
            "Global ID: `NONE`"
        ),
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    generated_at = (
        datetime.now(
            timezone.utc
        )
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise FileNotFoundError(root)

    test_name = validate_test_name(
        args.test_name
    )

    test_dir = (
        resolve(
            root,
            args.output_root,
        )
        / test_name
    )

    if not test_dir.is_dir():
        raise FileNotFoundError(
            "Stage-0 test directory "
            f"not found: {test_dir}"
        )

    manifest_path = (
        test_dir
        / "input_manifest.json"
    )
    audit_path = (
        test_dir
        / "audit.json"
    )
    target_path = (
        test_dir
        / "target_initialization.json"
    )

    for path in (
        manifest_path,
        audit_path,
        target_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                "Missing Stage-0 "
                f"artifact: {path}"
            )

    manifest = read_json(
        manifest_path
    )
    audit = read_json(
        audit_path
    )
    target = read_json(
        target_path
    )

    if (
        manifest.get("status")
        != "PASS"
        or audit.get("status")
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 0 must be PASS: "
            f"{manifest.get('status')} / "
            f"{audit.get('status')}"
        )

    prepare_outputs(
        test_dir,
        args.overwrite_stage1,
    )

    video_info = manifest.get(
        "video"
    )

    if not isinstance(
        video_info,
        dict,
    ):
        raise RuntimeError(
            "Missing Stage-0 "
            "video contract"
        )

    video = Path(
        str(video_info["path"])
    ).resolve()

    if (
        not video.is_file()
        or sha256_file(video)
        != str(video_info["sha256"])
    ):
        raise RuntimeError(
            "Video is missing or "
            "changed after Stage 0"
        )

    target_bbox = target.get(
        "bbox_xyxy"
    )

    if (
        not isinstance(
            target_bbox,
            list,
        )
        or len(target_bbox) != 4
    ):
        raise RuntimeError(
            "Missing Stage-0 "
            "bbox_xyxy"
        )

    target_bbox = [
        float(value)
        for value in target_bbox
    ]

    if not all(
        math.isfinite(value)
        for value in target_bbox
    ):
        raise ValueError(
            "Non-finite target bbox"
        )

    checkpoint = resolve(
        root,
        args.checkpoint,
    )

    checkpoint_sha = (
        sha256_file(checkpoint)
        if checkpoint.is_file()
        else ""
    )

    if (
        checkpoint_sha
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "RF-DETR checkpoint "
            "hash mismatch: "
            f"observed={checkpoint_sha}, "
            f"expected="
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )

    source, source_sha = (
        find_detector_source(
            root,
            args.detector_source,
        )
    )

    frozen = import_detector_source(
        source,
        root,
    )

    device, torch = resolve_device(
        args.device
    )

    model = frozen.load_rfdetr_model(
        checkpoint,
        device,
        MODEL_SIZE,
    )

    if not hasattr(
        model,
        "predict",
    ):
        raise RuntimeError(
            "Loaded RF-DETR model "
            "has no predict()"
        )

    capture = cv2.VideoCapture(
        str(video)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )
    declared_frames = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    )
    width = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )
    height = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )

    if (
        fps <= 0
        or declared_frames <= 0
        or width <= 0
        or height <= 0
    ):
        raise RuntimeError(
            "Invalid video metadata: "
            f"{width}x{height}, "
            f"fps={fps}, "
            f"frames={declared_frames}"
        )

    if (
        width,
        height,
    ) != (
        int(video_info["width"]),
        int(video_info["height"]),
    ):
        raise RuntimeError(
            "Video geometry changed "
            "after Stage 0"
        )

    csv_path = (
        test_dir
        / "detections.csv"
    )
    csv_temporary = (
        test_dir
        / "detections.csv.stage1.tmp"
    )

    preview_path = (
        test_dir
        / "rfdetr_detections_preview.mp4"
    )
    preview_temporary = (
        test_dir
        / "rfdetr_detections_preview.stage1.tmp.mp4"
    )

    writer: Optional[
        cv2.VideoWriter
    ] = None

    if not args.no_preview:
        writer = cv2.VideoWriter(
            str(preview_temporary),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Cannot create preview: "
                f"{preview_temporary}"
            )

    counts: Counter[int] = Counter()
    raw_types: Counter[str] = (
        Counter()
    )

    total_detections = 0
    zero_frames = 0
    zero_target_class_frames = 0
    processed = 0

    first_frame: Optional[
        np.ndarray
    ] = None
    first_detections: Optional[
        list[dict[str, Any]]
    ] = None

    best: Optional[
        dict[str, Any]
    ] = None
    ranked: list[
        dict[str, Any]
    ] = []

    with csv_temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields(),
            extrasaction="raise",
        )
        csv_writer.writeheader()

        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frame_index = processed

            if (
                frame.shape[1] != width
                or frame.shape[0]
                != height
            ):
                raise RuntimeError(
                    "Frame geometry "
                    f"changed at {frame_index}"
                )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            with torch.inference_mode():
                raw = model.predict(
                    rgb,
                    threshold=CONF,
                )

            raw_types[
                f"{type(raw).__module__}."
                f"{type(raw).__name__}"
            ] += 1

            detections = (
                frozen.normalize_detections(
                    raw,
                    frame_width=width,
                    frame_height=height,
                    conf_thresh=CONF,
                    class_id_offset=0,
                )
            )

            if not isinstance(
                detections,
                list,
            ):
                raise TypeError(
                    "normalize_detections() "
                    "did not return list"
                )

            if not detections:
                zero_frames += 1

            if not any(
                int(detection["class_id"])
                in TARGET_CLASSES
                for detection in detections
            ):
                zero_target_class_frames += 1

            initial_by_index: dict[
                int,
                dict[str, Any],
            ] = {}

            if frame_index == 0:
                first_frame = frame.copy()
                first_detections = (
                    detections
                )

                best, ranked = (
                    select_initial_target(
                        target_bbox,
                        detections,
                    )
                )

                initial_by_index = {
                    int(
                        item[
                            "detection_index"
                        ]
                    ): {
                        key: item[key]
                        for key in (
                            "iou",
                            "target_coverage",
                            "candidate_center_inside_target",
                            "target_center_inside_candidate",
                            "normalized_center_distance",
                            "match_score",
                            "reliable",
                        )
                    }
                    for item in ranked
                }

            for (
                detection_index,
                detection,
            ) in enumerate(detections):
                class_id = int(
                    detection["class_id"]
                )

                if (
                    class_id
                    not in CLASS_NAMES
                    or detection[
                        "class_name"
                    ]
                    != CLASS_NAMES[
                        class_id
                    ]
                ):
                    raise RuntimeError(
                        "Class mapping "
                        f"mismatch: {detection}"
                    )

                confidence = float(
                    detection["conf"]
                )

                x1, y1, x2, y2 = map(
                    float,
                    detection["xyxy"],
                )

                if (
                    confidence < CONF
                    or not (
                        0 <= x1 < x2
                        <= width - 1
                        and 0 <= y1 < y2
                        <= height - 1
                    )
                ):
                    raise RuntimeError(
                        "Invalid frozen "
                        f"detection: {detection}"
                    )

                csv_writer.writerow(
                    detection_row(
                        frame_index,
                        fps,
                        detection_index,
                        detection,
                        width,
                        height,
                        initial_by_index.get(
                            detection_index
                        ),
                    )
                )

                counts[class_id] += 1
                total_detections += 1

            if writer is not None:
                writer.write(
                    draw_preview(
                        frame,
                        frame_index,
                        fps,
                        detections,
                        target_bbox,
                        best,
                    )
                )

            processed += 1

            if (
                args.print_every > 0
                and (
                    processed
                    % args.print_every
                    == 0
                    or processed
                    == declared_frames
                )
            ):
                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    f"[INFO] "
                    f"frames={processed}/"
                    f"{declared_frames} "
                    f"detections="
                    f"{total_detections} "
                    f"speed="
                    f"{processed / max(elapsed, 1e-9):.2f} fps",
                    flush=True,
                )

        handle.flush()
        os.fsync(
            handle.fileno()
        )

    capture.release()

    if writer is not None:
        writer.release()

    if processed != declared_frames:
        raise RuntimeError(
            "Frame count mismatch: "
            f"{processed}/"
            f"{declared_frames}"
        )

    os.replace(
        csv_temporary,
        csv_path,
    )

    if writer is not None:
        if (
            not preview_temporary.is_file()
            or preview_temporary
            .stat()
            .st_size
            == 0
        ):
            raise RuntimeError(
                "Preview file is "
                "missing or empty"
            )

        os.replace(
            preview_temporary,
            preview_path,
        )

    reliable = bool(
        best is not None
        and best["reliable"]
    )

    decision = (
        "AUTHORIZE_STAGE2_TARGET_ASSOCIATION"
        if reliable
        else (
            "BLOCK_STAGE2_REVIEW_"
            "INITIAL_BBOX_OR_DETECTOR"
        )
    )

    initialization = {
        "schema_version": (
            "kickclip.target_centric."
            "initial_target_detection.v1"
        ),
        "stage": STAGE,
        "script_version": VERSION,
        "generated_at": generated_at,
        "status": (
            "MATCHED_RELIABLE_DETECTOR_ANCHOR"
            if reliable
            else (
                "NO_RELIABLE_"
                "DETECTOR_ANCHOR"
            )
        ),
        "decision": decision,
        "frame_index": 0,
        "time_ms": 0,
        "authoritative_source": (
            "USER_SELECTED_STAGE0_BBOX"
        ),
        "authoritative_user_bbox_xyxy": [
            round(
                value,
                4,
            )
            for value in target_bbox
        ],
        "user_bbox_replaced_by_detector": (
            False
        ),
        "reliable_detector_anchor": (
            reliable
        ),
        "best_candidate": best,
        "ranked_player_goalkeeper_candidates": (
            ranked
        ),
        "reliability_rule": (
            "iou>=0.20 OR "
            "(target_coverage>=0.60 "
            "AND target center inside candidate) "
            "OR (iou>=0.10 AND "
            "both centers contained)"
        ),
    }

    atomic_json(
        test_dir
        / "initial_target_detection.json",
        initialization,
    )

    if (
        first_frame is None
        or first_detections is None
    ):
        raise RuntimeError(
            "First frame was "
            "not processed"
        )

    initialization_preview = (
        draw_preview(
            first_frame,
            0,
            fps,
            first_detections,
            target_bbox,
            best,
        )
    )

    if not cv2.imwrite(
        str(
            test_dir
            / "initial_target_detection_preview.jpg"
        ),
        initialization_preview,
    ):
        raise RuntimeError(
            "Could not save initial "
            "target preview"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    summary = {
        "schema_version": (
            "kickclip.target_centric."
            "stage1_detection_summary.v1"
        ),
        "stage": STAGE,
        "script_version": VERSION,
        "generated_at": generated_at,
        "status": "PASS",
        "decision": decision,
        "test_name": test_name,
        "video": {
            "path": str(video),
            "sha256": (
                video_info["sha256"]
            ),
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frames": (
                declared_frames
            ),
            "processed_frames": (
                processed
            ),
            "duration_seconds": (
                processed / fps
            ),
        },
        "detector": {
            "model_size": MODEL_SIZE,
            "confidence_threshold": (
                CONF
            ),
            "class_id_offset": 0,
            "checkpoint": str(
                checkpoint
            ),
            "checkpoint_sha256": (
                checkpoint_sha
            ),
            "source": str(source),
            "source_canonical_sha256": (
                source_sha
            ),
            "device": device,
            "model_python_type": (
                f"{type(model).__module__}."
                f"{type(model).__name__}"
            ),
            "raw_output_types": dict(
                raw_types
            ),
            "all_five_classes_retained": (
                True
            ),
            "additional_nms_applied": (
                False
            ),
            "scene_mode_filter_applied": (
                False
            ),
            "class_filter_applied": (
                False
            ),
            "bbox_size_filter_applied": (
                False
            ),
        },
        "counts": {
            "total_detections": (
                total_detections
            ),
            "zero_detection_frames": (
                zero_frames
            ),
            "zero_player_goalkeeper_frames": (
                zero_target_class_frames
            ),
            "class_counts": {
                str(class_id): int(
                    counts.get(
                        class_id,
                        0,
                    )
                )
                for class_id in CLASS_NAMES
            },
        },
        "initial_target": (
            initialization
        ),
        "runtime": {
            "elapsed_seconds": (
                elapsed
            ),
            "processing_fps": (
                processed
                / max(
                    elapsed,
                    1e-9,
                )
            ),
        },
        "outputs": {
            "detections_csv": str(
                csv_path
            ),
            "detections_csv_sha256": (
                sha256_file(
                    csv_path
                )
            ),
            "initial_target_detection_json": str(
                test_dir
                / "initial_target_detection.json"
            ),
            "initial_target_detection_preview": str(
                test_dir
                / "initial_target_detection_preview.jpg"
            ),
            "preview_video": (
                str(preview_path)
                if writer is not None
                else None
            ),
            "preview_video_sha256": (
                sha256_file(
                    preview_path
                )
                if writer is not None
                else None
            ),
        },
        "training_performed": False,
        "tracking_performed": False,
        "reid_inference_performed": (
            False
        ),
        "global_linking_performed": (
            False
        ),
        "threshold_search_performed": (
            False
        ),
        "protected_v4_v7_outputs_modified": (
            False
        ),
    }

    atomic_json(
        test_dir
        / "stage1_detection_summary.json",
        summary,
    )

    atomic_text(
        test_dir
        / "stage1_detection_report.md",
        report(
            summary,
            initialization,
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V1 Stage 1 complete"
    )
    print(
        "Status               : PASS"
    )
    print(
        f"Decision             : "
        f"{decision}"
    )
    print(
        f"Frames               : "
        f"{processed}/"
        f"{declared_frames}"
    )
    print(
        f"Detections           : "
        f"{total_detections}"
    )
    print(
        "Player/GK detections : "
        f"{counts.get(0, 0) + counts.get(1, 0)}"
    )
    print(
        f"Initial anchor       : "
        f"{initialization['status']}"
    )

    if best is not None:
        print(
            "Initial best         : "
            f"{best['class_name']} "
            f"conf="
            f"{best['confidence']:.4f} "
            f"IoU={best['iou']:.4f} "
            f"score="
            f"{best['match_score']:.4f}"
        )

    print(
        f"Processing speed     : "
        f"{summary['runtime']['processing_fps']:.3f} fps"
    )
    print(
        "Training/tracking    : "
        "NONE / NONE"
    )
    print(
        "V4-V7 modification   : "
        "NONE"
    )
    print(
        f"Output               : "
        f"{test_dir}"
    )

    return (
        0
        if reliable
        else 3
    )


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "Stage 1 interrupted",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            "Stage 1 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)