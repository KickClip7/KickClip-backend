#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 2-C.

Freeze the visually reviewed Stage 2-B result, preserve its raw target
observations, build a separate editing crop trajectory, and render a
same-geometry target-centered preview.

No tracking, detection, ReID, threshold search, Global ID, GTA, or V7 logic is
performed here. SEARCHING/AMBIGUOUS/LOST frames never receive a target bbox.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np


STAGE = "stage2c_finalize_target_timeline_and_render"
VERSION = "target-centric-v1-stage2c-1.0.0"

OUTPUT_NAMES = (
    "stage2c_final_target_timeline.json",
    "stage2c_final_frame_observations.csv",
    "stage2c_crop_trajectory.csv",
    "stage2c_target_centered_preview.mp4",
    "stage2c_audit.json",
    "stage2c_report.md",
)

ALL_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "OCCLUDED",
    "LOST",
    "SEARCHING",
    "AMBIGUOUS",
    "REACQUIRED",
    "ABSENT",
    "USER_CONFIRMED",
    "TERMINATED",
}

BBOX_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "OCCLUDED",
    "REACQUIRED",
    "USER_CONFIRMED",
}

UNCERTAIN_STATES = {
    "LOST",
    "SEARCHING",
    "AMBIGUOUS",
    "ABSENT",
    "TERMINATED",
}


@dataclass(frozen=True)
class CropPolicy:
    """Frozen initial crop policy for the Phase-1 preview."""

    # Target occupies about 38% of rendered output height.
    target_height_fraction: float = 0.38

    # Prevent narrow player bboxes from creating an excessively tight crop.
    target_width_fraction: float = 0.46

    # Maximum zoom is approximately 1 / 0.38 = 2.63x.
    min_crop_height_ratio: float = 0.38
    max_crop_height_ratio: float = 1.00

    # Low-pass smoothing.
    center_time_constant_seconds: float = 0.22
    zoom_time_constant_seconds: float = 0.30

    # Do not move the virtual camera for tiny target movements.
    deadzone_width_ratio: float = 0.07
    deadzone_height_ratio: float = 0.07

    # Hard velocity limits for editing trajectory.
    max_center_speed_diagonals_per_second: float = 1.00
    max_log_zoom_change_per_second: float = 0.95

    # Briefly keep the previous crop before returning to the full frame.
    uncertain_hold_seconds: float = 0.25

    # Slightly widen the crop during a short OCCLUDED state.
    occluded_zoom_out_factor: float = 1.08

    # Preview-only bbox overlay.
    bbox_overlay: bool = True


@dataclass
class CropFrame:
    frame_index: int
    time_ms: int
    tracking_state: str
    crop_mode: str
    target_bbox_available: bool

    crop_xyxy: list[float]
    center_x: float
    center_y: float
    crop_width: float
    crop_height: float
    zoom_factor: float

    target_center_x: Optional[float]
    target_center_y: Optional[float]
    target_width: Optional[float]
    target_height: Optional[float]

    center_speed_px_per_frame: float
    log_zoom_delta: float
    fallback_full_frame: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the reviewed Stage 2-B target timeline and render "
            "a separate target-centered crop preview."
        )
    )

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
        "--stage2b-visual-review",
        choices=(
            "PASS",
            "FAIL",
        ),
        required=True,
        help=(
            "Explicit visual-review result for the Stage 2-B "
            "reacquisition and post-reacquisition clips."
        ),
    )

    parser.add_argument(
        "--reviewer",
        default="USER",
    )

    parser.add_argument(
        "--review-note",
        default=(
            "Reacquisition and post-reacquisition review clips maintained "
            "the originally selected player without an observed switch."
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-stage2c",
        action="store_true",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
    )

    return parser.parse_args()


def resolve(
    root: Path,
    value: Path,
) -> Path:
    value = value.expanduser()

    if value.is_absolute():
        return value.resolve()

    return (
        root
        / value
    ).resolve()


def read_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:
        for chunk in iter(
            lambda: stream.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def atomic_text(
    path: Path,
    text: str,
) -> None:
    temporary = path.with_name(
        path.name
        + ".tmp"
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
            allow_nan=False,
        )
        + "\n",
    )


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows to write: {path}"
        )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0]
            ),
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

        stream.flush()
        os.fsync(
            stream.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def validate_test_name(
    value: str,
) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if (
        not value
        or value in {
            ".",
            "..",
        }
        or any(
            character
            not in allowed
            for character
            in value
        )
    ):
        raise ValueError(
            "Invalid --test-name"
        )

    return value


def prepare_outputs(
    test_dir: Path,
    overwrite: bool,
) -> None:
    existing = [
        test_dir
        / name
        for name
        in OUTPUT_NAMES
        if (
            test_dir
            / name
        ).exists()
    ]

    if (
        existing
        and not overwrite
    ):
        raise FileExistsError(
            "Stage 2-C outputs already exist. "
            "Use --overwrite-stage2c:\n"
            + "\n".join(
                str(path)
                for path
                in existing
            )
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(
                path
            )

        path.unlink()

    for path in test_dir.glob(
        "*.stage2c.tmp*"
    ):
        if path.is_file():
            path.unlink()


def bbox_valid(
    bbox: Optional[
        Sequence[float]
    ],
    width: int,
    height: int,
) -> bool:
    if (
        bbox is None
        or len(bbox) != 4
    ):
        return False

    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    return (
        all(
            math.isfinite(value)
            for value
            in (
                x1,
                y1,
                x2,
                y2,
            )
        )
        and 0 <= x1 < x2 <= width - 1
        and 0 <= y1 < y2 <= height - 1
    )


def bbox_parts(
    bbox: Sequence[float],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    return (
        (
            x1
            + x2
        )
        / 2,
        (
            y1
            + y2
        )
        / 2,
        x2
        - x1,
        y2
        - y1,
    )


def smoothing_alpha(
    fps: float,
    time_constant: float,
) -> float:
    if time_constant <= 0:
        return 1.0

    return (
        1.0
        - math.exp(
            -1.0
            / (
                fps
                * time_constant
            )
        )
    )


def desired_crop(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    policy: CropPolicy,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    (
        center_x,
        center_y,
        target_width,
        target_height,
    ) = bbox_parts(
        bbox
    )

    aspect_ratio = (
        frame_width
        / frame_height
    )

    crop_height_from_height = (
        target_height
        / policy.target_height_fraction
    )

    crop_width_from_target = (
        target_width
        / policy.target_width_fraction
    )

    crop_height_from_width = (
        crop_width_from_target
        / aspect_ratio
    )

    crop_height = max(
        crop_height_from_height,
        crop_height_from_width,
        frame_height
        * policy.min_crop_height_ratio,
    )

    crop_height = min(
        crop_height,
        frame_height
        * policy.max_crop_height_ratio,
    )

    crop_width = (
        crop_height
        * aspect_ratio
    )

    if crop_width > frame_width:
        crop_width = float(
            frame_width
        )

        crop_height = float(
            frame_height
        )

    return (
        center_x,
        center_y,
        crop_width,
        crop_height,
    )


def fit_crop(
    center_x: float,
    center_y: float,
    crop_width: float,
    crop_height: float,
    frame_width: int,
    frame_height: int,
) -> tuple[
    list[float],
    float,
    float,
]:
    crop_width = float(
        np.clip(
            crop_width,
            2.0,
            frame_width,
        )
    )

    crop_height = float(
        np.clip(
            crop_height,
            2.0,
            frame_height,
        )
    )

    center_x = float(
        np.clip(
            center_x,
            crop_width
            / 2,
            frame_width
            - crop_width
            / 2,
        )
    )

    center_y = float(
        np.clip(
            center_y,
            crop_height
            / 2,
            frame_height
            - crop_height
            / 2,
        )
    )

    crop = [
        center_x
        - crop_width
        / 2,
        center_y
        - crop_height
        / 2,
        center_x
        + crop_width
        / 2,
        center_y
        + crop_height
        / 2,
    ]

    return (
        crop,
        center_x,
        center_y,
    )


def limit_center(
    current: np.ndarray,
    proposed: np.ndarray,
    width: int,
    height: int,
    fps: float,
    policy: CropPolicy,
) -> np.ndarray:
    delta = (
        proposed
        - current
    )

    distance = float(
        np.linalg.norm(
            delta
        )
    )

    maximum = (
        policy
        .max_center_speed_diagonals_per_second
        * math.hypot(
            width,
            height,
        )
        / fps
    )

    if (
        distance > maximum
        and maximum > 0
    ):
        delta *= (
            maximum
            / distance
        )

    return (
        current
        + delta
    )


def limit_crop_height(
    current: float,
    proposed: float,
    fps: float,
    policy: CropPolicy,
) -> float:
    current = max(
        current,
        2.0,
    )

    proposed = max(
        proposed,
        2.0,
    )

    log_delta = math.log(
        proposed
        / current
    )

    maximum = (
        policy
        .max_log_zoom_change_per_second
        / fps
    )

    log_delta = float(
        np.clip(
            log_delta,
            -maximum,
            maximum,
        )
    )

    return (
        current
        * math.exp(
            log_delta
        )
    )


def normalize_frames(
    timeline: Mapping[str, Any],
    frame_count: int,
    width: int,
    height: int,
    fps: float,
) -> list[
    dict[str, Any]
]:
    source = timeline.get(
        "frames"
    )

    if (
        not isinstance(
            source,
            list,
        )
        or len(source)
        != frame_count
    ):
        source_count: Any = (
            len(source)
            if isinstance(
                source,
                list,
            )
            else "INVALID"
        )

        raise ValueError(
            "Stage 2-B timeline "
            "frame count mismatch: "
            f"{source_count}/"
            f"{frame_count}"
        )

    frames: list[
        dict[str, Any]
    ] = []

    for (
        expected_frame,
        raw,
    ) in enumerate(
        source
    ):
        if (
            not isinstance(
                raw,
                dict,
            )
            or int(
                raw.get(
                    "frame_index",
                    -1,
                )
            )
            != expected_frame
        ):
            raise ValueError(
                "Invalid Stage 2-B "
                f"frame at index "
                f"{expected_frame}"
            )

        state = str(
            raw.get(
                "state",
                "",
            )
        )

        if state not in ALL_STATES:
            raise ValueError(
                "Unsupported state "
                f"at frame "
                f"{expected_frame}: "
                f"{state}"
            )

        raw_bbox = raw.get(
            "bbox_xyxy"
        )

        if raw_bbox is None:
            bbox = None
        else:
            if (
                not isinstance(
                    raw_bbox,
                    list,
                )
                or len(raw_bbox)
                != 4
            ):
                raise ValueError(
                    "Invalid bbox structure "
                    f"at frame "
                    f"{expected_frame}"
                )

            bbox = [
                float(value)
                for value
                in raw_bbox
            ]

            if not bbox_valid(
                bbox,
                width,
                height,
            ):
                raise ValueError(
                    "Invalid bbox at frame "
                    f"{expected_frame}: "
                    f"{bbox}"
                )

        if (
            bbox is not None
            and state
            not in BBOX_STATES
        ):
            raise ValueError(
                f"State {state} "
                "must not contain bbox "
                f"at frame "
                f"{expected_frame}"
            )

        frames.append(
            {
                "frame_index": (
                    expected_frame
                ),
                "time_ms": int(
                    raw.get(
                        "time_ms",
                        round(
                            expected_frame
                            * 1000
                            / fps
                        ),
                    )
                ),
                "state": state,
                "bbox_xyxy": bbox,
                "bbox_source": (
                    raw.get(
                        "bbox_source"
                    )
                ),
                "selected_detection_id": (
                    raw.get(
                        "selected_detection_id"
                    )
                ),
                "tracking_confidence": float(
                    raw.get(
                        "tracking_confidence",
                        0.0,
                    )
                ),
                "identity_confidence": float(
                    raw.get(
                        "identity_confidence",
                        0.0,
                    )
                ),
                "decision_reason": (
                    raw.get(
                        "decision_reason"
                    )
                ),
                "target_similarity": (
                    raw.get(
                        "target_similarity"
                    )
                ),
                "target_max_similarity": (
                    raw.get(
                        "target_max_similarity"
                    )
                ),
                "negative_max_similarity": (
                    raw.get(
                        "negative_max_similarity"
                    )
                ),
                "negative_margin": (
                    raw.get(
                        "negative_margin"
                    )
                ),
                "candidate_margin": (
                    raw.get(
                        "candidate_margin"
                    )
                ),
                "review_required": bool(
                    raw.get(
                        "review_required",
                        False,
                    )
                ),
            }
        )

    return frames


def build_crop_trajectory(
    frames: Sequence[
        Mapping[str, Any]
    ],
    width: int,
    height: int,
    fps: float,
    policy: CropPolicy,
) -> list[CropFrame]:
    aspect_ratio = (
        width
        / height
    )

    full_center = np.array(
        [
            width
            / 2,
            height
            / 2,
        ],
        dtype=np.float64,
    )

    current_center = (
        full_center.copy()
    )

    current_height = float(
        height
    )

    center_alpha = (
        smoothing_alpha(
            fps,
            policy
            .center_time_constant_seconds,
        )
    )

    zoom_alpha = (
        smoothing_alpha(
            fps,
            policy
            .zoom_time_constant_seconds,
        )
    )

    uncertain_hold_frames = max(
        1,
        round(
            policy
            .uncertain_hold_seconds
            * fps
        ),
    )

    uncertain_run = 0

    result: list[
        CropFrame
    ] = []

    for raw in frames:
        frame_index = int(
            raw["frame_index"]
        )

        state = str(
            raw["state"]
        )

        bbox = raw[
            "bbox_xyxy"
        ]

        if bbox is None:
            target_center_x = None
            target_center_y = None
            target_width = None
            target_height = None
        else:
            (
                target_center_x,
                target_center_y,
                target_width,
                target_height,
            ) = bbox_parts(
                bbox
            )

        if state in UNCERTAIN_STATES:
            uncertain_run += 1
        else:
            uncertain_run = 0

        if (
            bbox is not None
            and state
            in {
                "INITIALIZING",
                "ACTIVE",
                "REACQUIRED",
                "USER_CONFIRMED",
            }
        ):
            (
                desired_center_x,
                desired_center_y,
                _,
                desired_height,
            ) = desired_crop(
                bbox,
                width,
                height,
                policy,
            )

            desired_center = np.array(
                [
                    desired_center_x,
                    desired_center_y,
                ],
                dtype=np.float64,
            )

            if state == "REACQUIRED":
                crop_mode = (
                    "REACQUIRED_EASING"
                )
            else:
                crop_mode = (
                    "TARGET_CENTERED"
                )

            fallback_full_frame = False

        elif (
            bbox is not None
            and state
            == "OCCLUDED"
        ):
            (
                desired_center_x,
                desired_center_y,
                _,
                desired_height,
            ) = desired_crop(
                bbox,
                width,
                height,
                policy,
            )

            desired_center = np.array(
                [
                    desired_center_x,
                    desired_center_y,
                ],
                dtype=np.float64,
            )

            desired_height = min(
                height,
                desired_height
                * policy
                .occluded_zoom_out_factor,
            )

            crop_mode = (
                "OCCLUDED_PREDICTION"
            )

            fallback_full_frame = False

        elif state == "OCCLUDED":
            desired_center = (
                current_center.copy()
            )

            desired_height = min(
                height,
                current_height
                * policy
                .occluded_zoom_out_factor,
            )

            crop_mode = (
                "OCCLUDED_HOLD"
            )

            fallback_full_frame = False

        elif (
            uncertain_run
            <= uncertain_hold_frames
        ):
            desired_center = (
                current_center.copy()
            )

            desired_height = (
                current_height
            )

            crop_mode = (
                "UNCERTAIN_HOLD"
            )

            fallback_full_frame = False

        else:
            desired_center = (
                full_center.copy()
            )

            desired_height = float(
                height
            )

            crop_mode = (
                "FULL_FRAME_FALLBACK"
            )

            fallback_full_frame = True

        current_width = (
            current_height
            * aspect_ratio
        )

        center_delta = (
            desired_center
            - current_center
        )

        inside_deadzone = (
            abs(
                float(
                    center_delta[0]
                )
            )
            <= current_width
            * policy
            .deadzone_width_ratio
            and abs(
                float(
                    center_delta[1]
                )
            )
            <= current_height
            * policy
            .deadzone_height_ratio
            and state
            not in {
                "REACQUIRED",
                "USER_CONFIRMED",
            }
        )

        if inside_deadzone:
            proposed_center = (
                current_center.copy()
            )
        else:
            proposed_center = (
                current_center
                + center_alpha
                * center_delta
            )

        proposed_center = (
            limit_center(
                current_center,
                proposed_center,
                width,
                height,
                fps,
                policy,
            )
        )

        proposed_height = (
            current_height
            + zoom_alpha
            * (
                desired_height
                - current_height
            )
        )

        proposed_height = (
            limit_crop_height(
                current_height,
                proposed_height,
                fps,
                policy,
            )
        )

        proposed_height = float(
            np.clip(
                proposed_height,
                height
                * policy
                .min_crop_height_ratio,
                height
                * policy
                .max_crop_height_ratio,
            )
        )

        proposed_width = (
            proposed_height
            * aspect_ratio
        )

        if proposed_width > width:
            proposed_width = float(
                width
            )

            proposed_height = float(
                height
            )

        (
            crop,
            fitted_center_x,
            fitted_center_y,
        ) = fit_crop(
            float(
                proposed_center[0]
            ),
            float(
                proposed_center[1]
            ),
            proposed_width,
            proposed_height,
            width,
            height,
        )

        fitted_center = np.array(
            [
                fitted_center_x,
                fitted_center_y,
            ],
            dtype=np.float64,
        )

        center_speed = float(
            np.linalg.norm(
                fitted_center
                - current_center
            )
        )

        log_zoom_delta = math.log(
            current_height
            / proposed_height
        )

        current_center = (
            fitted_center
        )

        current_height = (
            proposed_height
        )

        crop_width = (
            crop[2]
            - crop[0]
        )

        crop_height = (
            crop[3]
            - crop[1]
        )

        result.append(
            CropFrame(
                frame_index=(
                    frame_index
                ),
                time_ms=int(
                    raw["time_ms"]
                ),
                tracking_state=state,
                crop_mode=crop_mode,
                target_bbox_available=(
                    bbox is not None
                ),
                crop_xyxy=[
                    float(value)
                    for value
                    in crop
                ],
                center_x=(
                    fitted_center_x
                ),
                center_y=(
                    fitted_center_y
                ),
                crop_width=(
                    crop_width
                ),
                crop_height=(
                    crop_height
                ),
                zoom_factor=(
                    width
                    / crop_width
                ),
                target_center_x=(
                    target_center_x
                ),
                target_center_y=(
                    target_center_y
                ),
                target_width=(
                    target_width
                ),
                target_height=(
                    target_height
                ),
                center_speed_px_per_frame=(
                    center_speed
                ),
                log_zoom_delta=(
                    log_zoom_delta
                ),
                fallback_full_frame=(
                    fallback_full_frame
                ),
            )
        )

    return result


def observation_row(
    item: Mapping[str, Any],
) -> dict[str, Any]:
    bbox = item[
        "bbox_xyxy"
    ]

    row: dict[str, Any] = {
        "frame_index": (
            item["frame_index"]
        ),
        "frame_1based": (
            int(
                item["frame_index"]
            )
            + 1
        ),
        "time_ms": (
            item["time_ms"]
        ),
        "state": (
            item["state"]
        ),
        "bbox_source": (
            item["bbox_source"]
            or ""
        ),
        "selected_detection_id": (
            item[
                "selected_detection_id"
            ]
            or ""
        ),
        "tracking_confidence": (
            f"{float(item['tracking_confidence']):.6f}"
        ),
        "identity_confidence": (
            f"{float(item['identity_confidence']):.6f}"
        ),
        "decision_reason": (
            item["decision_reason"]
            or ""
        ),
        "review_required": (
            item["review_required"]
        ),
        "mask_ref": "",
    }

    for key in (
        "target_similarity",
        "target_max_similarity",
        "negative_max_similarity",
        "negative_margin",
        "candidate_margin",
    ):
        if item[key] is None:
            row[key] = ""
        else:
            row[key] = (
                f"{float(item[key]):.6f}"
            )

    for (
        index,
        axis,
    ) in enumerate(
        (
            "x1",
            "y1",
            "x2",
            "y2",
        )
    ):
        if bbox is None:
            row[
                f"bbox_{axis}"
            ] = ""
        else:
            row[
                f"bbox_{axis}"
            ] = (
                f"{bbox[index]:.4f}"
            )

    return row


def crop_row(
    item: CropFrame,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame_index": (
            item.frame_index
        ),
        "frame_1based": (
            item.frame_index
            + 1
        ),
        "time_ms": (
            item.time_ms
        ),
        "tracking_state": (
            item.tracking_state
        ),
        "crop_mode": (
            item.crop_mode
        ),
        "target_bbox_available": (
            item
            .target_bbox_available
        ),
        "crop_center_x": (
            f"{item.center_x:.4f}"
        ),
        "crop_center_y": (
            f"{item.center_y:.4f}"
        ),
        "crop_width": (
            f"{item.crop_width:.4f}"
        ),
        "crop_height": (
            f"{item.crop_height:.4f}"
        ),
        "zoom_factor": (
            f"{item.zoom_factor:.6f}"
        ),
        "target_center_x": (
            ""
            if item.target_center_x
            is None
            else
            f"{item.target_center_x:.4f}"
        ),
        "target_center_y": (
            ""
            if item.target_center_y
            is None
            else
            f"{item.target_center_y:.4f}"
        ),
        "target_width": (
            ""
            if item.target_width
            is None
            else
            f"{item.target_width:.4f}"
        ),
        "target_height": (
            ""
            if item.target_height
            is None
            else
            f"{item.target_height:.4f}"
        ),
        "center_speed_px_per_frame": (
            f"{item.center_speed_px_per_frame:.6f}"
        ),
        "log_zoom_delta": (
            f"{item.log_zoom_delta:.8f}"
        ),
        "fallback_full_frame": (
            item.fallback_full_frame
        ),
    }

    for (
        index,
        axis,
    ) in enumerate(
        (
            "x1",
            "y1",
            "x2",
            "y2",
        )
    ):
        row[
            f"crop_{axis}"
        ] = (
            f"{item.crop_xyxy[index]:.4f}"
        )

    return row


def build_segments(
    frames: Sequence[
        Mapping[str, Any]
    ],
    fps: float,
) -> list[
    dict[str, Any]
]:
    result: list[
        dict[str, Any]
    ] = []

    start = 0

    for index in range(
        1,
        len(frames)
        + 1,
    ):
        if (
            index < len(frames)
            and frames[index]["state"]
            == frames[start]["state"]
        ):
            continue

        part = frames[
            start:index
        ]

        result.append(
            {
                "segment_index": (
                    len(result)
                ),
                "state": (
                    part[0][
                        "state"
                    ]
                ),
                "start_frame": (
                    part[0][
                        "frame_index"
                    ]
                ),
                "end_frame_inclusive": (
                    part[-1][
                        "frame_index"
                    ]
                ),
                "start_ms": (
                    part[0][
                        "time_ms"
                    ]
                ),
                "end_ms_exclusive": (
                    round(
                        index
                        * 1000
                        / fps
                    )
                ),
                "frame_count": (
                    len(part)
                ),
                "bbox_frame_count": (
                    sum(
                        item[
                            "bbox_xyxy"
                        ]
                        is not None
                        for item
                        in part
                    )
                ),
                "mean_tracking_confidence": (
                    statistics.fmean(
                        float(
                            item[
                                "tracking_confidence"
                            ]
                        )
                        for item
                        in part
                    )
                ),
                "mean_identity_confidence": (
                    statistics.fmean(
                        float(
                            item[
                                "identity_confidence"
                            ]
                        )
                        for item
                        in part
                    )
                ),
                "review_required": (
                    any(
                        bool(
                            item[
                                "review_required"
                            ]
                        )
                        for item
                        in part
                    )
                ),
            }
        )

        start = index

    return result


def transformed_bbox(
    bbox: Sequence[float],
    crop: Sequence[float],
    width: int,
    height: int,
) -> list[int]:
    crop_x1, crop_y1, crop_x2, crop_y2 = map(
        float,
        crop,
    )

    crop_width = (
        crop_x2
        - crop_x1
    )

    crop_height = (
        crop_y2
        - crop_y1
    )

    values = [
        (
            bbox[0]
            - crop_x1
        )
        * width
        / crop_width,
        (
            bbox[1]
            - crop_y1
        )
        * height
        / crop_height,
        (
            bbox[2]
            - crop_x1
        )
        * width
        / crop_width,
        (
            bbox[3]
            - crop_y1
        )
        * height
        / crop_height,
    ]

    return [
        round(
            np.clip(
                values[0],
                0,
                width - 1,
            )
        ),
        round(
            np.clip(
                values[1],
                0,
                height - 1,
            )
        ),
        round(
            np.clip(
                values[2],
                0,
                width - 1,
            )
        ),
        round(
            np.clip(
                values[3],
                0,
                height - 1,
            )
        ),
    ]


def state_color(
    state: str,
) -> tuple[int, int, int]:
    colors = {
        "INITIALIZING": (
            255,
            255,
            255,
        ),
        "ACTIVE": (
            40,
            220,
            40,
        ),
        "OCCLUDED": (
            0,
            215,
            255,
        ),
        "LOST": (
            0,
            0,
            255,
        ),
        "SEARCHING": (
            255,
            160,
            0,
        ),
        "AMBIGUOUS": (
            180,
            0,
            255,
        ),
        "REACQUIRED": (
            255,
            255,
            0,
        ),
        "ABSENT": (
            150,
            150,
            150,
        ),
        "USER_CONFIRMED": (
            255,
            255,
            255,
        ),
        "TERMINATED": (
            100,
            100,
            100,
        ),
    }

    return colors.get(
        state,
        (
            255,
            255,
            255,
        ),
    )


def render_preview(
    video: Path,
    output: Path,
    frames: Sequence[
        Mapping[str, Any]
    ],
    trajectory: Sequence[
        CropFrame
    ],
    fps: float,
    width: int,
    height: int,
    policy: CropPolicy,
    print_every: int,
) -> None:
    temporary = output.with_name(
        output.stem
        + ".stage2c.tmp.mp4"
    )

    capture = cv2.VideoCapture(
        str(video)
    )

    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if (
        not capture.isOpened()
        or not writer.isOpened()
    ):
        capture.release()
        writer.release()

        raise RuntimeError(
            "Cannot create "
            "Stage 2-C preview"
        )

    frame_index = 0

    try:
        while True:
            ok, source = (
                capture.read()
            )

            if not ok:
                break

            if (
                frame_index
                >= len(frames)
            ):
                raise RuntimeError(
                    "Video has more "
                    "frames than the "
                    "final timeline"
                )

            item = frames[
                frame_index
            ]

            crop_item = trajectory[
                frame_index
            ]

            crop = (
                crop_item.crop_xyxy
            )

            x1 = max(
                0,
                min(
                    width - 2,
                    math.floor(
                        crop[0]
                    ),
                ),
            )

            y1 = max(
                0,
                min(
                    height - 2,
                    math.floor(
                        crop[1]
                    ),
                ),
            )

            x2 = max(
                x1 + 1,
                min(
                    width,
                    math.ceil(
                        crop[2]
                    ),
                ),
            )

            y2 = max(
                y1 + 1,
                min(
                    height,
                    math.ceil(
                        crop[3]
                    ),
                ),
            )

            region = source[
                y1:y2,
                x1:x2,
            ]

            if region.size == 0:
                raise RuntimeError(
                    "Empty crop at frame "
                    f"{frame_index}: "
                    f"{crop}"
                )

            rendered = cv2.resize(
                region,
                (
                    width,
                    height,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )

            state = str(
                item["state"]
            )

            color = state_color(
                state
            )

            bbox = item[
                "bbox_xyxy"
            ]

            if (
                policy.bbox_overlay
                and bbox is not None
            ):
                (
                    target_x1,
                    target_y1,
                    target_x2,
                    target_y2,
                ) = transformed_bbox(
                    bbox,
                    crop,
                    width,
                    height,
                )

                cv2.rectangle(
                    rendered,
                    (
                        target_x1,
                        target_y1,
                    ),
                    (
                        target_x2,
                        target_y2,
                    ),
                    color,
                    2,
                )

                cv2.putText(
                    rendered,
                    (
                        f"TARGET "
                        f"{state}"
                    ),
                    (
                        target_x1,
                        max(
                            20,
                            target_y1
                            - 5,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.rectangle(
                rendered,
                (
                    0,
                    0,
                ),
                (
                    width,
                    26,
                ),
                (
                    0,
                    0,
                    0,
                ),
                -1,
            )

            cv2.putText(
                rendered,
                (
                    "Stage 2-C"
                    f" | frame="
                    f"{frame_index}"
                    f" | state="
                    f"{state}"
                    f" | crop="
                    f"{crop_item.crop_mode}"
                    f" | zoom="
                    f"{crop_item.zoom_factor:.2f}x"
                ),
                (
                    6,
                    18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

            if state in UNCERTAIN_STATES:
                cv2.rectangle(
                    rendered,
                    (
                        0,
                        height - 26,
                    ),
                    (
                        width,
                        height,
                    ),
                    (
                        0,
                        0,
                        0,
                    ),
                    -1,
                )

                cv2.putText(
                    rendered,
                    (
                        "TARGET NOT CONFIRMED "
                        "- FULL VIEW FALLBACK"
                    ),
                    (
                        6,
                        height - 8,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            writer.write(
                rendered
            )

            frame_index += 1

            if (
                print_every > 0
                and frame_index
                % print_every
                == 0
            ):
                print(
                    "[STAGE2C PREVIEW] "
                    f"frames="
                    f"{frame_index}/"
                    f"{len(frames)}",
                    flush=True,
                )

    finally:
        capture.release()
        writer.release()

    if (
        frame_index
        != len(frames)
        or not temporary.is_file()
        or temporary.stat().st_size
        == 0
    ):
        raise RuntimeError(
            "Invalid Stage 2-C "
            f"preview: "
            f"{frame_index}/"
            f"{len(frames)}"
        )

    os.replace(
        temporary,
        output,
    )


def percentile_95(
    values: Sequence[float],
) -> float:
    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            0.95,
        )
    )


def build_report(
    audit: Mapping[str, Any],
) -> str:
    states = audit[
        "counts"
    ][
        "state_counts"
    ]

    crop = audit[
        "crop_metrics"
    ]

    review = audit[
        "visual_review"
    ]

    return f"""# KickClip Target-Centric Tracking V1 — Stage 2-C Finalization

- Status: `{audit['status']}`
- Decision: `{audit['decision']}`
- Visual review: `{review['result']}` by `{review['reviewer']}`
- Review note: {review['note']}

## Frozen result

```text
reviewed Stage 2-B target observations
→ final target timeline
→ independent crop trajectory
→ target-centered preview
INITIALIZING / ACTIVE: {states.get('INITIALIZING', 0)} / {states.get('ACTIVE', 0)}
OCCLUDED / SEARCHING: {states.get('OCCLUDED', 0)} / {states.get('SEARCHING', 0)}
AMBIGUOUS / REACQUIRED: {states.get('AMBIGUOUS', 0)} / {states.get('REACQUIRED', 0)}
Bbox frame ratio: {audit['counts']['bbox_frame_ratio']:.4%}
Crop center speed mean / p95: {crop['mean_center_speed_px_per_frame']:.4f} / {crop['p95_center_speed_px_per_frame']:.4f} px/frame
Zoom mean / max: {crop['mean_zoom_factor']:.4f} / {crop['max_zoom_factor']:.4f}x
Full-frame fallback frames: {crop['full_frame_fallback_frames']}

Tracking observations and crop trajectory remain separate. SEARCHING, AMBIGUOUS and LOST frames keep null target bboxes.

Open stage2c_target_centered_preview.mp4 and review crop comfort, excessive zoom and camera movement.
"""

def main() -> int:
    arguments = parse_args()

    started = (
        time.perf_counter()
    )

    if (
        arguments
        .stage2b_visual_review
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 2-C is blocked "
            "because Stage 2-B "
            "visual review is not PASS"
        )

    root = (
        arguments
        .project_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise FileNotFoundError(
            root
        )

    test_name = (
        validate_test_name(
            arguments.test_name
        )
    )

    test_dir = (
        resolve(
            root,
            arguments.output_root,
        )
        / test_name
    )

    paths = {
        "stage1": (
            test_dir
            / "stage1_detection_summary.json"
        ),
        "stage2": (
            test_dir
            / "stage2_association_summary.json"
        ),
        "stage2b_summary": (
            test_dir
            / "stage2b_reentry_summary.json"
        ),
        "stage2b_timeline": (
            test_dir
            / "stage2b_target_timeline.json"
        ),
        "stage2b_observations": (
            test_dir
            / "stage2b_frame_observations.csv"
        ),
        "stage2b_preview": (
            test_dir
            / "stage2b_reentry_preview.mp4"
        ),
        "review_reacquisition": (
            test_dir
            / "stage2b_review_reacquisition.mp4"
        ),
        "review_post_reacquisition": (
            test_dir
            / "stage2b_review_post_reacquisition.mp4"
        ),
    }

    for required_name in (
        "stage1",
        "stage2",
        "stage2b_summary",
        "stage2b_timeline",
        "stage2b_observations",
        "stage2b_preview",
    ):
        required_path = paths[
            required_name
        ]

        if not required_path.is_file():
            raise FileNotFoundError(
                f"Missing "
                f"{required_name}: "
                f"{required_path}"
            )

    prepare_outputs(
        test_dir,
        arguments
        .overwrite_stage2c,
    )

    stage1 = read_json(
        paths["stage1"]
    )

    stage2 = read_json(
        paths["stage2"]
    )

    stage2b_summary = read_json(
        paths[
            "stage2b_summary"
        ]
    )

    stage2b_timeline = read_json(
        paths[
            "stage2b_timeline"
        ]
    )

    if (
        stage1.get(
            "status"
        )
        != "PASS"
        or stage2.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 1 and Stage 2 "
            "must both be PASS"
        )

    if (
        stage2b_summary.get(
            "status"
        )
        != "PASS"
        or stage2b_summary.get(
            "decision"
        )
        != (
            "AUTHORIZE_MANDATORY_"
            "STAGE2B_REENTRY_"
            "VISUAL_REVIEW"
        )
    ):
        raise RuntimeError(
            "Stage 2-B was not "
            "authorized for visual review"
        )

    selected_tracklet = (
        stage2b_summary
        .get(
            "selection",
            {},
        )
        .get(
            "selected_tracklet_id"
        )
    )

    if not selected_tracklet:
        raise RuntimeError(
            "Stage 2-B has no "
            "selected re-entry tracklet"
        )

    video_info = (
        stage1["video"]
    )

    video = Path(
        video_info["path"]
    ).resolve()

    if (
        not video.is_file()
        or sha256_file(
            video
        )
        != video_info[
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Video is missing "
            "or changed"
        )

    width = int(
        video_info[
            "width"
        ]
    )

    height = int(
        video_info[
            "height"
        ]
    )

    fps = float(
        video_info[
            "fps"
        ]
    )

    frame_count = int(
        video_info[
            "processed_frames"
        ]
    )

    frames = normalize_frames(
        stage2b_timeline,
        frame_count,
        width,
        height,
        fps,
    )

    state_counts = Counter(
        item["state"]
        for item
        in frames
    )

    if (
        state_counts.get(
            "REACQUIRED",
            0,
        )
        < 1
    ):
        raise RuntimeError(
            "Stage 2-B timeline "
            "has no REACQUIRED frame"
        )

    policy = CropPolicy()

    trajectory = (
        build_crop_trajectory(
            frames,
            width,
            height,
            fps,
            policy,
        )
    )

    if (
        len(trajectory)
        != frame_count
    ):
        raise RuntimeError(
            "Crop trajectory "
            "frame count mismatch"
        )

    for (
        frame,
        crop_item,
    ) in zip(
        frames,
        trajectory,
    ):
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        ) = crop_item.crop_xyxy

        valid_crop = (
            -1e-4
            <= crop_x1
            < crop_x2
            <= width
            + 1e-4
            and -1e-4
            <= crop_y1
            < crop_y2
            <= height
            + 1e-4
        )

        if not valid_crop:
            raise RuntimeError(
                "Invalid crop at frame "
                f"{crop_item.frame_index}: "
                f"{crop_item.crop_xyxy}"
            )

        if (
            frame["state"]
            in UNCERTAIN_STATES
            and frame[
                "bbox_xyxy"
            ]
            is not None
        ):
            raise RuntimeError(
                "Stage 2-C fabricated "
                "an uncertain-state bbox"
            )

    observation_rows = [
        observation_row(
            item
        )
        for item
        in frames
    ]

    crop_rows = [
        crop_row(
            item
        )
        for item
        in trajectory
    ]

    observation_path = (
        test_dir
        / "stage2c_final_frame_observations.csv"
    )

    crop_path = (
        test_dir
        / "stage2c_crop_trajectory.csv"
    )

    write_csv(
        observation_path,
        observation_rows,
    )

    write_csv(
        crop_path,
        crop_rows,
    )

    review_artifacts: list[
        dict[str, Any]
    ] = []

    for review_name in (
        "review_reacquisition",
        "review_post_reacquisition",
    ):
        review_path = paths[
            review_name
        ]

        review_artifacts.append(
            {
                "path": str(
                    review_path
                ),
                "exists": (
                    review_path
                    .is_file()
                ),
                "sha256": (
                    sha256_file(
                        review_path
                    )
                    if review_path
                    .is_file()
                    else None
                ),
            }
        )

    visual_review = {
        "result": (
            arguments
            .stage2b_visual_review
        ),
        "reviewer": (
            arguments.reviewer
        ),
        "note": (
            arguments.review_note
        ),
        "selected_reentry_tracklet": (
            selected_tracklet
        ),
        "review_artifacts": (
            review_artifacts
        ),
    }

    timeline_path = (
        test_dir
        / "stage2c_final_target_timeline.json"
    )

    timeline = {
        "schema_version": (
            "kickclip.target_centric."
            "final_target_timeline.v1"
        ),
        "stage": STAGE,
        "script_version": VERSION,
        "generated_at": (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "status": (
            "FINALIZED_AFTER_"
            "VISUAL_REVIEW"
        ),
        "target_id": (
            "target_001"
        ),
        "test_name": (
            test_name
        ),
        "scope": (
            "single_camera_"
            "cut_free_shot"
        ),
        "primary_objective": (
            "MINIMIZE_SILENT_"
            "WRONG_PLAYER_SWITCH"
        ),
        "video": {
            "path": str(
                video
            ),
            "sha256": (
                video_info[
                    "sha256"
                ]
            ),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": (
                frame_count
            ),
            "duration_seconds": (
                frame_count
                / fps
            ),
        },
        "visual_review": (
            visual_review
        ),
        "tracking_contract": {
            "raw_tracking_source": str(
                paths[
                    "stage2b_timeline"
                ]
            ),
            "raw_tracking_source_sha256": (
                sha256_file(
                    paths[
                        "stage2b_timeline"
                    ]
                )
            ),
            "tracking_observations_modified": (
                False
            ),
            "new_tracking_inference": (
                False
            ),
            "new_reid_inference": (
                False
            ),
            "threshold_search": (
                False
            ),
            "global_linking": (
                False
            ),
            "v7_logic": (
                False
            ),
            "uncertain_state_bbox_policy": (
                "NULL"
            ),
            "mask_backend": (
                "NOT_RUN_BBOX_FIRST_MVP"
            ),
        },
        "crop_contract": {
            "trajectory_is_tracking_evidence": (
                False
            ),
            "output_geometry_preserved": (
                True
            ),
            "output_width": (
                width
            ),
            "output_height": (
                height
            ),
            "uncertain_state_render_policy": (
                "SHORT_HOLD_THEN_"
                "SMOOTH_FULL_FRAME_"
                "FALLBACK"
            ),
            "policy": asdict(
                policy
            ),
            "trajectory_csv": str(
                crop_path
            ),
        },
        "segments": (
            build_segments(
                frames,
                fps,
            )
        ),
        "frames": (
            frames
        ),
    }

    atomic_json(
        timeline_path,
        timeline,
    )

    preview_path = (
        test_dir
        / "stage2c_target_centered_preview.mp4"
    )

    if not arguments.no_preview:
        render_preview(
            video,
            preview_path,
            frames,
            trajectory,
            fps,
            width,
            height,
            policy,
            arguments.print_every,
        )

    center_speeds = [
        item
        .center_speed_px_per_frame
        for item
        in trajectory
    ]

    zoom_factors = [
        item.zoom_factor
        for item
        in trajectory
    ]

    log_zoom_deltas = [
        abs(
            item.log_zoom_delta
        )
        for item
        in trajectory
    ]

    crop_metrics = {
        "mean_center_speed_px_per_frame": (
            statistics.fmean(
                center_speeds
            )
        ),
        "p95_center_speed_px_per_frame": (
            percentile_95(
                center_speeds
            )
        ),
        "max_center_speed_px_per_frame": (
            max(
                center_speeds
            )
        ),
        "mean_zoom_factor": (
            statistics.fmean(
                zoom_factors
            )
        ),
        "max_zoom_factor": (
            max(
                zoom_factors
            )
        ),
        "mean_abs_log_zoom_delta": (
            statistics.fmean(
                log_zoom_deltas
            )
        ),
        "p95_abs_log_zoom_delta": (
            percentile_95(
                log_zoom_deltas
            )
        ),
        "full_frame_fallback_frames": (
            sum(
                item
                .fallback_full_frame
                for item
                in trajectory
            )
        ),
        "target_centered_frames": (
            sum(
                item.crop_mode
                in {
                    "TARGET_CENTERED",
                    "REACQUIRED_EASING",
                    "OCCLUDED_PREDICTION",
                    "OCCLUDED_HOLD",
                }
                for item
                in trajectory
            )
        ),
    }

    bbox_frames = sum(
        item[
            "bbox_xyxy"
        ]
        is not None
        for item
        in frames
    )

    uncertain_bbox_violations = sum(
        item["state"]
        in UNCERTAIN_STATES
        and item[
            "bbox_xyxy"
        ]
        is not None
        for item
        in frames
    )

    audit_path = (
        test_dir
        / "stage2c_audit.json"
    )

    audit = {
        "schema_version": (
            "kickclip.target_centric."
            "stage2c_audit.v1"
        ),
        "stage": STAGE,
        "script_version": VERSION,
        "generated_at": (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "status": "PASS",
        "decision": (
            "AUTHORIZE_PHASE1_"
            "FINAL_TARGET_CENTERED_"
            "PREVIEW_REVIEW"
        ),
        "test_name": (
            test_name
        ),
        "visual_review": (
            visual_review
        ),
        "counts": {
            "frame_count": (
                frame_count
            ),
            "bbox_frame_count": (
                bbox_frames
            ),
            "bbox_frame_ratio": (
                bbox_frames
                / frame_count
            ),
            "state_counts": dict(
                state_counts
            ),
            "uncertain_bbox_violations": (
                uncertain_bbox_violations
            ),
        },
        "crop_metrics": (
            crop_metrics
        ),
        "safety_invariants": {
            "stage2b_visual_review_pass": (
                True
            ),
            "stage2b_selected_tracklet_present": (
                True
            ),
            "tracking_observations_preserved": (
                True
            ),
            "uncertain_states_have_null_bbox": (
                uncertain_bbox_violations
                == 0
            ),
            "crop_trajectory_separate_from_tracking": (
                True
            ),
            "original_output_geometry_preserved": (
                True
            ),
            "no_new_tracking_inference": (
                True
            ),
            "no_new_reid_inference": (
                True
            ),
            "no_threshold_search": (
                True
            ),
            "no_global_linking": (
                True
            ),
            "no_v7_logic": (
                True
            ),
        },
        "inputs": {
            name: {
                "path": str(
                    path
                ),
                "sha256": (
                    sha256_file(
                        path
                    )
                ),
            }
            for (
                name,
                path,
            )
            in paths.items()
            if path.is_file()
        },
        "outputs": {
            "final_timeline": {
                "path": str(
                    timeline_path
                ),
                "sha256": (
                    sha256_file(
                        timeline_path
                    )
                ),
            },
            "final_frame_observations": {
                "path": str(
                    observation_path
                ),
                "sha256": (
                    sha256_file(
                        observation_path
                    )
                ),
            },
            "crop_trajectory": {
                "path": str(
                    crop_path
                ),
                "sha256": (
                    sha256_file(
                        crop_path
                    )
                ),
            },
            "preview": (
                None
                if arguments.no_preview
                else {
                    "path": str(
                        preview_path
                    ),
                    "sha256": (
                        sha256_file(
                            preview_path
                        )
                    ),
                }
            ),
        },
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
        "performance_certification": (
            "IDENTITY_REVIEW_PASS_"
            "FOR_REVIEWED_STAGE2B_CLIPS; "
            "FINAL_CROP_USABILITY_"
            "REVIEW_REQUIRED"
        ),
    }

    atomic_json(
        audit_path,
        audit,
    )

    atomic_text(
        test_dir
        / "stage2c_report.md",
        build_report(
            audit
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V1 Stage 2-C complete"
    )

    print(
        "Status                    : PASS"
    )

    print(
        "Decision                  : "
        "AUTHORIZE_PHASE1_FINAL_"
        "TARGET_CENTERED_PREVIEW_REVIEW"
    )

    print(
        f"Frames                    : "
        f"{frame_count}"
    )

    print(
        f"Bbox frames               : "
        f"{bbox_frames} "
        f"({bbox_frames / frame_count:.4%})"
    )

    print(
        f"States                    : "
        f"{dict(state_counts)}"
    )

    print(
        "Visual identity review    : "
        f"PASS ({arguments.reviewer})"
    )

    print(
        "Crop center speed mean/p95: "
        f"{crop_metrics['mean_center_speed_px_per_frame']:.3f} / "
        f"{crop_metrics['p95_center_speed_px_per_frame']:.3f} "
        "px/frame"
    )

    print(
        "Zoom mean/max             : "
        f"{crop_metrics['mean_zoom_factor']:.3f} / "
        f"{crop_metrics['max_zoom_factor']:.3f}x"
    )

    print(
        "Full-frame fallback frames: "
        f"{crop_metrics['full_frame_fallback_frames']}"
    )

    print(
        "Tracking/ReID inference   : "
        "NONE / NONE"
    )

    print(
        "Global linking / V7       : "
        "NONE / NONE"
    )

    print(
        f"Output                    : "
        f"{test_dir}"
    )

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "Stage 2-C interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 2-C fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )