#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-A0.

Cross-shot input and hard-cut candidate audit.

This stage:
- verifies the frozen Phase-1 manifest;
- validates the input video;
- lets the user select the initial target bbox;
- finds likely hard camera cuts;
- renders a before/after contact sheet.

It does not perform target tracking or cross-shot linking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


STAGE = "stage3a0_audit_cross_shot_inputs"
VERSION = "target-centric-v2-stage3a0-1.0.0"

OUTPUT_NAMES = (
    "stage3a0_summary.json",
    "stage3a0_cut_candidates.csv",
    "stage3a0_initial_target_preview.jpg",
    "stage3a0_cut_contact_sheet.jpg",
    "stage3a0_report.md",
)


@dataclass
class CutCandidate:
    rank: int
    cut_frame: int
    timestamp_seconds: float
    combined_score: float
    pixel_difference: float
    histogram_distance: float
    edge_change: float
    hard_cut_candidate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a Phase-2 cross-shot test video."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-name",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/target_centric_tracking_v2"),
    )

    parser.add_argument(
        "--phase1-manifest",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "phase1_frozen_manifest.json"
        ),
    )

    parser.add_argument(
        "--initial-bbox",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
    )

    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Select the target bbox interactively on the first frame.",
    )

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--max-cut-candidates",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--cut-score-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--nms-gap-frames",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()

    if value.is_absolute():
        return value.resolve()

    return (root / value).resolve()


def validate_test_name(value: str) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if (
        not value
        or value in {".", ".."}
        or any(character not in allowed for character in value)
    ):
        raise ValueError("Invalid --test-name")

    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(rows)

        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


def prepare_outputs(
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        output_dir / name
        for name in OUTPUT_NAMES
        if (output_dir / name).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Stage 3-A0 outputs already exist. "
            "Use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(path)

        path.unlink()


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


def verify_hash_entry(
    root: Path,
    path_value: Any,
    hash_value: Any,
) -> tuple[bool, str]:
    if not isinstance(path_value, str) or not path_value:
        return False, "PATH_NOT_RECORDED"

    if not isinstance(hash_value, str) or len(hash_value) != 64:
        return False, "SHA256_NOT_RECORDED"

    path = resolve(
        root,
        Path(path_value),
    )

    if not path.is_file():
        return False, f"MISSING:{path}"

    actual = sha256_file(path)

    if actual.lower() != hash_value.lower():
        return (
            False,
            f"HASH_MISMATCH:{path}",
        )

    return True, f"VERIFIED:{path}"


def verify_manifest(
    root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {
            "verified": False,
            "errors": [
                f"Manifest not found: {manifest_path}"
            ],
            "warnings": [],
            "script_count": 0,
            "model_results": {},
        }

    manifest = read_json(manifest_path)

    errors: list[str] = []
    warnings: list[str] = []

    scripts = manifest.get(
        "scripts",
        [],
    )

    script_count = (
        len(scripts)
        if isinstance(scripts, (list, dict))
        else 0
    )

    if script_count != 11:
        errors.append(
            "Frozen script count must be 11, "
            f"but manifest contains {script_count}."
        )

    model_results: dict[str, Any] = {}

    models = manifest.get(
        "models",
        {},
    )

    if not isinstance(models, dict):
        errors.append(
            "Manifest models section is invalid."
        )
        models = {}

    for model_name in (
        "rfdetr",
        "sports_osnet",
    ):
        model = models.get(
            model_name,
            {},
        )

        if not isinstance(model, dict):
            errors.append(
                f"Missing model record: {model_name}"
            )
            continue

        recorded_verified = bool(
            model.get(
                "verified",
                False,
            )
        )

        if not recorded_verified:
            errors.append(
                f"Manifest model is not verified: {model_name}"
            )

        path_value = model.get("path")
        hash_value = (
            model.get("sha256")
            or model.get("hash")
        )

        file_verified, message = verify_hash_entry(
            root,
            path_value,
            hash_value,
        )

        model_results[model_name] = {
            "manifest_verified": recorded_verified,
            "file_verified": file_verified,
            "message": message,
            "path": path_value,
            "sha256": hash_value,
        }

        if not file_verified:
            errors.append(
                f"{model_name}: {message}"
            )

    return {
        "verified": not errors,
        "errors": errors,
        "warnings": warnings,
        "script_count": script_count,
        "model_results": model_results,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def get_video_metadata(
    video_path: Path,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
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

    ok, first_frame = capture.read()
    capture.release()

    if not ok or first_frame is None:
        raise RuntimeError(
            "Cannot read the first video frame."
        )

    if width <= 0 or height <= 0:
        height, width = first_frame.shape[:2]

    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            f"Invalid video FPS: {fps}"
        )

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "declared_frame_count": declared_frames,
        "declared_duration_seconds": (
            declared_frames / fps
            if declared_frames > 0
            else None
        ),
        "first_frame": first_frame,
    }


def select_initial_bbox(
    first_frame: np.ndarray,
    supplied_bbox: Optional[
        Sequence[float]
    ],
    select_roi: bool,
) -> list[int]:
    height, width = first_frame.shape[:2]

    if supplied_bbox is not None:
        x1, y1, x2, y2 = (
            int(round(value))
            for value in supplied_bbox
        )

    elif select_roi:
        x, y, box_width, box_height = map(
            int,
            cv2.selectROI(
                "SELECT TARGET - drag bbox and press ENTER",
                first_frame,
                showCrosshair=True,
                fromCenter=False,
            ),
        )

        cv2.destroyAllWindows()

        if box_width <= 0 or box_height <= 0:
            raise RuntimeError(
                "Target ROI selection was cancelled."
            )

        x1 = x
        y1 = y
        x2 = x + box_width
        y2 = y + box_height

    else:
        raise ValueError(
            "Use either --initial-bbox "
            "or --select-roi."
        )

    x1 = max(
        0,
        min(
            width - 2,
            x1,
        ),
    )

    y1 = max(
        0,
        min(
            height - 2,
            y1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            width - 1,
            x2,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            height - 1,
            y2,
        ),
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def save_initial_preview(
    path: Path,
    first_frame: np.ndarray,
    bbox: Sequence[int],
) -> None:
    preview = first_frame.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label = (
        f"INITIAL TARGET "
        f"[{x1},{y1},{x2},{y2}]"
    )

    cv2.putText(
        preview,
        label,
        (
            max(
                4,
                x1,
            ),
            max(
                18,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(
        str(path),
        preview,
    ):
        raise RuntimeError(
            f"Failed to write preview: {path}"
        )


def resize_for_analysis(
    frame: np.ndarray,
    width: int,
) -> np.ndarray:
    original_height, original_width = (
        frame.shape[:2]
    )

    if original_width == width:
        return frame

    scale = width / original_width

    target_height = max(
        1,
        int(
            round(
                original_height * scale
            )
        ),
    )

    return cv2.resize(
        frame,
        (
            width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def hsv_histogram(
    frame: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        histogram,
        histogram,
        alpha=1.0,
        norm_type=cv2.NORM_L1,
    )

    return histogram


def edge_map(
    frame: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return cv2.Canny(
        gray,
        60,
        140,
    )


def robust_scale(
    values: np.ndarray,
) -> np.ndarray:
    if values.size == 0:
        return values.astype(
            np.float64
        )

    median = float(
        np.median(values)
    )

    percentile_95 = float(
        np.percentile(
            values,
            95,
        )
    )

    spread = max(
        1e-8,
        percentile_95 - median,
    )

    normalized = (
        values - median
    ) / spread

    return np.clip(
        normalized,
        0.0,
        1.5,
    ) / 1.5


def analyze_transitions(
    video_path: Path,
    analysis_width: int,
) -> tuple[
    int,
    list[dict[str, float]],
]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    ok, previous_full = capture.read()

    if not ok or previous_full is None:
        capture.release()

        raise RuntimeError(
            "Cannot read frame 0."
        )

    previous = resize_for_analysis(
        previous_full,
        analysis_width,
    )

    previous_gray = cv2.cvtColor(
        previous,
        cv2.COLOR_BGR2GRAY,
    )

    previous_hist = hsv_histogram(
        previous
    )

    previous_edges = edge_map(
        previous
    )

    transitions: list[
        dict[str, float]
    ] = []

    frame_index = 1

    while True:
        ok, full_frame = capture.read()

        if not ok:
            break

        current = resize_for_analysis(
            full_frame,
            analysis_width,
        )

        current_gray = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2GRAY,
        )

        current_hist = hsv_histogram(
            current
        )

        current_edges = edge_map(
            current
        )

        pixel_difference = float(
            np.mean(
                cv2.absdiff(
                    previous_gray,
                    current_gray,
                )
            )
            / 255.0
        )

        histogram_distance = float(
            cv2.compareHist(
                previous_hist,
                current_hist,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )

        edge_xor = cv2.bitwise_xor(
            previous_edges,
            current_edges,
        )

        edge_union = cv2.bitwise_or(
            previous_edges,
            current_edges,
        )

        union_pixels = float(
            np.count_nonzero(
                edge_union
            )
        )

        edge_change = (
            float(
                np.count_nonzero(
                    edge_xor
                )
            )
            / max(
                1.0,
                union_pixels,
            )
        )

        transitions.append(
            {
                "cut_frame": float(
                    frame_index
                ),
                "pixel_difference": (
                    pixel_difference
                ),
                "histogram_distance": (
                    histogram_distance
                ),
                "edge_change": (
                    edge_change
                ),
            }
        )

        previous_gray = current_gray
        previous_hist = current_hist
        previous_edges = current_edges

        frame_index += 1

    capture.release()

    return frame_index, transitions


def rank_cut_candidates(
    transitions: Sequence[
        Mapping[str, float]
    ],
    fps: float,
    threshold: float,
    nms_gap: int,
    maximum: int,
) -> tuple[
    list[CutCandidate],
    dict[str, float],
]:
    if not transitions:
        return [], {
            "pixel_median": 0.0,
            "histogram_median": 0.0,
            "edge_median": 0.0,
        }

    pixel = np.asarray(
        [
            row["pixel_difference"]
            for row in transitions
        ],
        dtype=np.float64,
    )

    histogram = np.asarray(
        [
            row["histogram_distance"]
            for row in transitions
        ],
        dtype=np.float64,
    )

    edge = np.asarray(
        [
            row["edge_change"]
            for row in transitions
        ],
        dtype=np.float64,
    )

    pixel_scaled = robust_scale(
        pixel
    )

    histogram_scaled = robust_scale(
        histogram
    )

    edge_scaled = robust_scale(
        edge
    )

    combined = (
        0.45 * pixel_scaled
        + 0.35 * histogram_scaled
        + 0.20 * edge_scaled
    )

    order = np.argsort(
        combined
    )[::-1]

    selected_indices: list[int] = []

    for index in order.tolist():
        cut_frame = int(
            transitions[
                index
            ][
                "cut_frame"
            ]
        )

        if any(
            abs(
                cut_frame
                - int(
                    transitions[
                        selected
                    ][
                        "cut_frame"
                    ]
                )
            )
            <= nms_gap
            for selected in selected_indices
        ):
            continue

        selected_indices.append(
            index
        )

        if len(selected_indices) >= maximum:
            break

    candidates: list[
        CutCandidate
    ] = []

    for rank, index in enumerate(
        selected_indices,
        start=1,
    ):
        cut_frame = int(
            transitions[
                index
            ][
                "cut_frame"
            ]
        )

        score = float(
            combined[
                index
            ]
        )

        candidates.append(
            CutCandidate(
                rank=rank,
                cut_frame=cut_frame,
                timestamp_seconds=(
                    cut_frame / fps
                ),
                combined_score=score,
                pixel_difference=float(
                    pixel[
                        index
                    ]
                ),
                histogram_distance=float(
                    histogram[
                        index
                    ]
                ),
                edge_change=float(
                    edge[
                        index
                    ]
                ),
                hard_cut_candidate=(
                    score >= threshold
                ),
            )
        )

    diagnostics = {
        "pixel_median": float(
            np.median(
                pixel
            )
        ),
        "pixel_p95": float(
            np.percentile(
                pixel,
                95,
            )
        ),
        "histogram_median": float(
            np.median(
                histogram
            )
        ),
        "histogram_p95": float(
            np.percentile(
                histogram,
                95,
            )
        ),
        "edge_median": float(
            np.median(
                edge
            )
        ),
        "edge_p95": float(
            np.percentile(
                edge,
                95,
            )
        ),
    }

    return candidates, diagnostics


def read_frame(
    video_path: Path,
    frame_index: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        max(
            0,
            frame_index,
        ),
    )

    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Cannot read frame {frame_index}"
        )

    return frame


def fit_tile(
    frame: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    height, width = frame.shape[:2]

    scale = min(
        tile_width / width,
        tile_height / height,
    )

    resized_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    resized = cv2.resize(
        frame,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            tile_height,
            tile_width,
            3,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        tile_width
        - resized_width
    ) // 2

    offset_y = (
        tile_height
        - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y + resized_height,
        offset_x:
        offset_x + resized_width,
    ] = resized

    return canvas


def draw_label(
    image: np.ndarray,
    text: str,
) -> None:
    cv2.rectangle(
        image,
        (0, 0),
        (
            image.shape[1] - 1,
            34,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def render_contact_sheet(
    video_path: Path,
    candidates: Sequence[
        CutCandidate
    ],
    output_path: Path,
) -> None:
    tile_width = 480
    tile_height = 285

    if not candidates:
        canvas = np.zeros(
            (
                tile_height,
                tile_width * 2,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            canvas,
            "NO CUT CANDIDATES",
            (40, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(output_path),
            canvas,
        )

        return

    rows: list[
        np.ndarray
    ] = []

    for candidate in candidates:
        before = read_frame(
            video_path,
            max(
                0,
                candidate.cut_frame - 1,
            ),
        )

        after = read_frame(
            video_path,
            candidate.cut_frame,
        )

        before_tile = fit_tile(
            before,
            tile_width,
            tile_height,
        )

        after_tile = fit_tile(
            after,
            tile_width,
            tile_height,
        )

        draw_label(
            before_tile,
            (
                f"#{candidate.rank} BEFORE "
                f"frame={candidate.cut_frame - 1}"
            ),
        )

        draw_label(
            after_tile,
            (
                f"AFTER frame={candidate.cut_frame} "
                f"time={candidate.timestamp_seconds:.2f}s "
                f"score={candidate.combined_score:.3f}"
            ),
        )

        rows.append(
            np.hstack(
                [
                    before_tile,
                    after_tile,
                ]
            )
        )

    sheet = np.vstack(
        rows
    )

    if not cv2.imwrite(
        str(output_path),
        sheet,
    ):
        raise RuntimeError(
            f"Cannot write contact sheet: {output_path}"
        )


def candidate_csv_row(
    candidate: CutCandidate,
) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "cut_frame": candidate.cut_frame,
        "timestamp_seconds": (
            f"{candidate.timestamp_seconds:.6f}"
        ),
        "combined_score": (
            f"{candidate.combined_score:.8f}"
        ),
        "pixel_difference": (
            f"{candidate.pixel_difference:.8f}"
        ),
        "histogram_distance": (
            f"{candidate.histogram_distance:.8f}"
        ),
        "edge_change": (
            f"{candidate.edge_change:.8f}"
        ),
        "hard_cut_candidate": (
            candidate.hard_cut_candidate
        ),
    }


def build_report(
    summary: Mapping[str, Any],
) -> str:
    video = summary["video"]
    cuts = summary["cut_detection"]
    bbox = summary["initial_target"]["bbox_xyxy"]

    return f"""# KickClip Target-Centric Tracking V2 — Stage 3-A0

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Video: `{video['path']}`
- Frames: `{video['actual_frame_count']}`
- FPS: `{video['fps']:.6f}`
- Duration: `{video['duration_seconds']:.3f} seconds`
- Initial target bbox: `{bbox}`
- Hard-cut candidates: `{cuts['hard_cut_candidate_count']}`
- Total ranked candidates: `{cuts['ranked_candidate_count']}`

## Scope

This stage only audits the cross-shot input and proposes hard-cut candidates.
It does not perform target tracking, ReID selection, or cross-shot linking.

## Next gate

Review `stage3a0_initial_target_preview.jpg` and
`stage3a0_cut_contact_sheet.jpg`. Confirm the true camera-cut frame and whether
the selected target appears before and after the cut.
"""


def main() -> int:
    arguments = parse_args()

    root = (
        arguments.project_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise FileNotFoundError(root)

    test_name = validate_test_name(
        arguments.test_name
    )

    video_path = resolve(
        root,
        arguments.video,
    )

    manifest_path = resolve(
        root,
        arguments.phase1_manifest,
    )

    output_dir = (
        resolve(
            root,
            arguments.output_root,
        )
        / test_name
    )

    if not video_path.is_file():
        raise FileNotFoundError(
            video_path
        )

    if arguments.analysis_width < 128:
        raise ValueError(
            "--analysis-width must be >= 128"
        )

    if arguments.max_cut_candidates < 1:
        raise ValueError(
            "--max-cut-candidates must be >= 1"
        )

    if not (
        0.0
        <= arguments.cut_score_threshold
        <= 1.0
    ):
        raise ValueError(
            "--cut-score-threshold must be within [0, 1]"
        )

    prepare_outputs(
        output_dir,
        arguments.overwrite,
    )

    manifest_audit = verify_manifest(
        root,
        manifest_path,
    )

    metadata = get_video_metadata(
        video_path
    )

    first_frame = metadata.pop(
        "first_frame"
    )

    bbox = select_initial_bbox(
        first_frame,
        arguments.initial_bbox,
        arguments.select_roi,
    )

    initial_preview_path = (
        output_dir
        / "stage3a0_initial_target_preview.jpg"
    )

    save_initial_preview(
        initial_preview_path,
        first_frame,
        bbox,
    )

    (
        actual_frame_count,
        transitions,
    ) = analyze_transitions(
        video_path,
        arguments.analysis_width,
    )

    (
        candidates,
        diagnostics,
    ) = rank_cut_candidates(
        transitions,
        metadata["fps"],
        arguments.cut_score_threshold,
        arguments.nms_gap_frames,
        arguments.max_cut_candidates,
    )

    contact_sheet_path = (
        output_dir
        / "stage3a0_cut_contact_sheet.jpg"
    )

    render_contact_sheet(
        video_path,
        candidates,
        contact_sheet_path,
    )

    candidate_rows = [
        candidate_csv_row(
            candidate
        )
        for candidate in candidates
    ]

    candidate_path = (
        output_dir
        / "stage3a0_cut_candidates.csv"
    )

    write_csv(
        candidate_path,
        candidate_rows,
        [
            "rank",
            "cut_frame",
            "timestamp_seconds",
            "combined_score",
            "pixel_difference",
            "histogram_distance",
            "edge_change",
            "hard_cut_candidate",
        ],
    )

    hard_candidates = [
        candidate
        for candidate in candidates
        if candidate.hard_cut_candidate
    ]

    duration_seconds = (
        actual_frame_count
        / metadata["fps"]
    )

    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_audit["verified"]:
        errors.extend(
            manifest_audit["errors"]
        )

    if actual_frame_count < 2:
        errors.append(
            "Video contains fewer than two readable frames."
        )

    if duration_seconds < 5.0:
        warnings.append(
            "Video is shorter than 5 seconds."
        )

    if len(hard_candidates) == 0:
        warnings.append(
            "No automatic hard-cut candidate exceeded "
            "the audit threshold."
        )

    if errors:
        status = "FAIL"
        decision = (
            "BLOCK_PHASE2_CROSS_SHOT_EXPERIMENT"
        )

    elif hard_candidates:
        status = "PASS"
        decision = (
            "AUTHORIZE_MANUAL_CUT_CANDIDATE_REVIEW"
        )

    else:
        status = "PASS"
        decision = (
            "REQUIRE_MANUAL_CUT_FRAME_CONFIRMATION"
        )

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "generated_at": (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "status": status,
        "decision": decision,
        "test_name": test_name,
        "errors": errors,
        "warnings": warnings,
        "phase1_freeze_manifest": (
            manifest_audit
        ),
        "video": {
            "path": str(
                video_path
            ),
            "sha256": sha256_file(
                video_path
            ),
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "declared_frame_count": (
                metadata[
                    "declared_frame_count"
                ]
            ),
            "actual_frame_count": (
                actual_frame_count
            ),
            "duration_seconds": (
                duration_seconds
            ),
        },
        "initial_target": {
            "bbox_xyxy": bbox,
            "preview": str(
                initial_preview_path
            ),
        },
        "cut_detection": {
            "method": (
                "HSV_HISTOGRAM_PIXEL_DIFFERENCE_EDGE_CHANGE"
            ),
            "analysis_width": (
                arguments.analysis_width
            ),
            "score_threshold": (
                arguments.cut_score_threshold
            ),
            "nms_gap_frames": (
                arguments.nms_gap_frames
            ),
            "ranked_candidate_count": (
                len(
                    candidates
                )
            ),
            "hard_cut_candidate_count": (
                len(
                    hard_candidates
                )
            ),
            "diagnostics": diagnostics,
            "candidates": [
                asdict(
                    candidate
                )
                for candidate in candidates
            ],
            "contact_sheet": str(
                contact_sheet_path
            ),
        },
        "safety_invariants": {
            "target_tracking_performed": False,
            "cross_shot_linking_performed": False,
            "reid_inference_performed": False,
            "phase1_files_modified": False,
            "threshold_search_performed": False,
        },
    }

    summary_path = (
        output_dir
        / "stage3a0_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3a0_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V2 Stage 3-A0 complete"
    )

    print(
        f"Status               : {status}"
    )

    print(
        f"Decision             : {decision}"
    )

    print(
        "Phase-1 freeze       : "
        f"{'VERIFIED' if manifest_audit['verified'] else 'FAIL'}"
    )

    print(
        "Video                : "
        f"{metadata['width']}x{metadata['height']} "
        f"{metadata['fps']:.6f} fps"
    )

    print(
        "Frames / duration    : "
        f"{actual_frame_count} / "
        f"{duration_seconds:.3f}s"
    )

    print(
        f"Initial bbox         : {bbox}"
    )

    print(
        "Cut candidates       : "
        f"{len(candidates)} ranked / "
        f"{len(hard_candidates)} hard"
    )

    for candidate in candidates:
        print(
            f"  #{candidate.rank} "
            f"frame={candidate.cut_frame} "
            f"time={candidate.timestamp_seconds:.3f}s "
            f"score={candidate.combined_score:.4f} "
            f"hard={candidate.hard_cut_candidate}"
        )

    print(
        f"Initial preview      : {initial_preview_path}"
    )

    print(
        f"Cut contact sheet    : {contact_sheet_path}"
    )

    print(
        f"Output               : {output_dir}"
    )

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "Stage 3-A0 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(130)

    except Exception as exc:
        print(
            "Stage 3-A0 fatal error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(2)