#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 2 conservative association.

Input: Stage-0/Stage-1 artifacts in runs/target_centric_tracking_v1/<test_name>/
Output: target timeline, per-frame observations, candidate audit, and preview.

Safety objective: never force a low-confidence candidate. Uncertain frames become
OCCLUDED, then LOST. Same-shot recovery is short-lived and stricter than normal
continuity. No ReID, training, threshold search, GTA, Global ID, or V7 logic.
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
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

VERSION = "target-centric-v1-stage2-1.0.0"
STAGE = "stage2_conservative_single_shot_target_association"
TARGET_CLASSES = {0, 1}
STATES = {"INITIALIZING", "ACTIVE", "OCCLUDED", "LOST"}
OUTPUT_FILES = (
    "target_timeline.json",
    "frame_observations.csv",
    "stage2_candidate_audit.csv",
    "stage2_association_summary.json",
    "stage2_association_report.md",
    "target_tracking_preview.mp4",
)


@dataclass(frozen=True)
class Policy:
    max_occluded_seconds: float = 0.35
    max_recovery_seconds: float = 1.00
    min_detection_confidence: float = 0.25
    recovery_min_detection_confidence: float = 0.35
    active_min_area_ratio: float = 0.40
    active_max_area_ratio: float = 2.50
    active_min_aspect_ratio: float = 0.45
    active_max_aspect_ratio: float = 2.20
    active_max_center_distance: float = 0.95
    active_max_bottom_distance: float = 1.05
    active_min_predicted_iou: float = 0.05
    active_close_center_distance: float = 0.55
    active_close_bottom_distance: float = 0.70
    recovery_min_area_ratio: float = 0.50
    recovery_max_area_ratio: float = 2.00
    recovery_min_aspect_ratio: float = 0.55
    recovery_max_aspect_ratio: float = 1.85
    recovery_max_center_distance: float = 0.78
    recovery_max_bottom_distance: float = 0.88
    recovery_min_predicted_iou: float = 0.08
    recovery_close_center_distance: float = 0.42
    recovery_close_bottom_distance: float = 0.55
    active_accept_score: float = 0.56
    active_accept_margin: float = 0.09
    active_strong_iou: float = 0.45
    active_strong_score: float = 0.58
    active_strong_margin: float = 0.035
    recovery_accept_score: float = 0.66
    recovery_accept_margin: float = 0.15
    recovery_strong_iou: float = 0.52
    recovery_strong_score: float = 0.66
    recovery_strong_margin: float = 0.10
    velocity_alpha: float = 0.55
    max_center_velocity_diagonals_per_frame: float = 0.80
    max_log_size_velocity_per_frame: float = 0.12
    memory_update_identity_confidence: float = 0.72
    memory_update_detection_confidence: float = 0.45
    review_risk_margin: float = 0.11
    review_risk_center_jump: float = 0.70


@dataclass
class Detection:
    frame: int
    time_ms: int
    index: int
    detection_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


@dataclass
class Candidate:
    frame: int
    detection_id: str
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]
    predicted_iou: float
    last_iou: float
    center_distance: float
    bottom_distance: float
    area_ratio: float
    aspect_ratio: float
    area_similarity: float
    aspect_similarity: float
    center_similarity: float
    bottom_similarity: float
    score: float
    eligible: bool
    rejection_reasons: list[str]
    rank: int = 0
    selected: bool = False


@dataclass
class Observation:
    frame: int
    time_ms: int
    state: str
    transition: str
    reason: str
    mode: str
    bbox_source: str
    bbox: Optional[list[float]]
    predicted_bbox: Optional[list[float]]
    detection_id: Optional[str]
    detection_index: Optional[int]
    class_id: Optional[int]
    class_name: Optional[str]
    detection_confidence: Optional[float]
    tracking_confidence: float
    identity_confidence: float
    visibility: float
    top_candidate_id: Optional[str]
    top_score: Optional[float]
    second_candidate_id: Optional[str]
    second_score: Optional[float]
    margin: Optional[float]
    candidate_count: int
    eligible_count: int
    gap: int
    predicted_iou: Optional[float]
    last_iou: Optional[float]
    center_distance: Optional[float]
    bottom_distance: Optional[float]
    area_ratio: Optional[float]
    aspect_ratio: Optional[float]
    memory_updated: bool
    risk_flags: list[str]


class MotionMemory:
    def __init__(self, width: int, height: int, policy: Policy) -> None:
        self.width, self.height, self.policy = width, height, policy
        self.last_frame: Optional[int] = None
        self.last_bbox: Optional[list[float]] = None
        self.velocity_center = np.zeros(2, dtype=np.float64)
        self.velocity_log_size = np.zeros(2, dtype=np.float64)
        self.identity_history: deque[float] = deque(maxlen=12)
        self.accepted_updates = 0
        self.high_confidence_updates = 0

    @staticmethod
    def parts(box: Sequence[float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = map(float, box)
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        return (x1 + x2) / 2, (y1 + y2) / 2, w, h

    def initialize(self, frame: int, box: Sequence[float]) -> None:
        self.last_frame, self.last_bbox = frame, list(map(float, box))
        self.identity_history.append(1.0)
        self.accepted_updates = self.high_confidence_updates = 1

    def predict(self, frame: int) -> list[float]:
        if self.last_frame is None or self.last_bbox is None:
            raise RuntimeError("Motion memory is not initialized")

        gap = max(0, frame - self.last_frame)
        cx, cy, w, h = self.parts(self.last_bbox)
        center = np.array([cx, cy]) + self.velocity_center * gap
        size = np.exp(
            np.log(np.array([w, h]))
            + self.velocity_log_size * gap
        )

        w = float(np.clip(size[0], 2, self.width))
        h = float(np.clip(size[1], 2, self.height))

        return clip_bbox(
            [
                center[0] - w / 2,
                center[1] - h / 2,
                center[0] + w / 2,
                center[1] + h / 2,
            ],
            self.width,
            self.height,
        )

    def update(
        self,
        frame: int,
        box: Sequence[float],
        identity_conf: float,
        detection_conf: float,
    ) -> bool:
        new_box = list(map(float, box))

        if (
            self.last_frame is not None
            and self.last_bbox is not None
            and frame > self.last_frame
        ):
            gap = frame - self.last_frame
            ocx, ocy, ow, oh = self.parts(self.last_bbox)
            ncx, ncy, nw, nh = self.parts(new_box)

            center_v = np.array(
                [
                    (ncx - ocx) / gap,
                    (ncy - ocy) / gap,
                ],
                dtype=np.float64,
            )

            max_speed = (
                self.policy.max_center_velocity_diagonals_per_frame
                * max(math.hypot(ow, oh), 1)
            )
            speed = float(np.linalg.norm(center_v))

            if speed > max_speed > 0:
                center_v *= max_speed / speed

            size_v = np.array(
                [
                    math.log(nw / ow) / gap,
                    math.log(nh / oh) / gap,
                ],
                dtype=np.float64,
            )
            size_v = np.clip(
                size_v,
                -self.policy.max_log_size_velocity_per_frame,
                self.policy.max_log_size_velocity_per_frame,
            )

            a = self.policy.velocity_alpha
            self.velocity_center = (
                a * center_v
                + (1 - a) * self.velocity_center
            )
            self.velocity_log_size = (
                a * size_v
                + (1 - a) * self.velocity_log_size
            )

        self.last_frame, self.last_bbox = frame, new_box
        self.accepted_updates += 1
        self.identity_history.append(identity_conf)

        high = (
            identity_conf
            >= self.policy.memory_update_identity_confidence
            and detection_conf
            >= self.policy.memory_update_detection_confidence
        )

        if high:
            self.high_confidence_updates += 1

        return high

    def stable_identity(self) -> float:
        if not self.identity_history:
            return 0.0

        return float(
            statistics.fmean(
                self.identity_history
            )
        )


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservative target association "
            "from frozen Stage-1 detections"
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
        "--max-occluded-seconds",
        type=float,
        default=Policy.max_occluded_seconds,
    )
    parser.add_argument(
        "--max-recovery-seconds",
        type=float,
        default=Policy.max_recovery_seconds,
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-stage2",
        action="store_true",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()

    return (root / path).resolve()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_text(
    path: Path,
    text: str,
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )
    temp.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temp, path)


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


def prepare(
    test_dir: Path,
    overwrite: bool,
) -> None:
    existing = [
        test_dir / name
        for name in OUTPUT_FILES
        if (test_dir / name).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Stage-2 outputs exist; "
            "use --overwrite-stage2:\n"
            + "\n".join(
                map(str, existing)
            )
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(path)
        path.unlink()

    for path in test_dir.glob(
        "*.stage2.tmp*"
    ):
        if path.is_file():
            path.unlink()


def clip01(value: float) -> float:
    return float(
        np.clip(
            value,
            0,
            1,
        )
    )


def clip_bbox(
    box: Sequence[float],
    width: int,
    height: int,
) -> list[float]:
    x1, y1, x2, y2 = map(
        float,
        box,
    )
    box_width = max(
        x2 - x1,
        2,
    )
    box_height = max(
        y2 - y1,
        2,
    )

    center_x = np.clip(
        (x1 + x2) / 2,
        0,
        width - 1,
    )
    center_y = np.clip(
        (y1 + y2) / 2,
        0,
        height - 1,
    )

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    if x1 < 0:
        x2, x1 = x2 - x1, 0

    if y1 < 0:
        y2, y1 = y2 - y1, 0

    if x2 > width - 1:
        x1, x2 = (
            x1 - (x2 - width + 1),
            width - 1,
        )

    if y2 > height - 1:
        y1, y2 = (
            y1 - (y2 - height + 1),
            height - 1,
        )

    x1 = float(
        np.clip(
            x1,
            0,
            width - 2,
        )
    )
    y1 = float(
        np.clip(
            y1,
            0,
            height - 2,
        )
    )

    return [
        x1,
        y1,
        float(
            np.clip(
                x2,
                x1 + 1,
                width - 1,
            )
        ),
        float(
            np.clip(
                y2,
                y1 + 1,
                height - 1,
            )
        ),
    ]


def area(box: Sequence[float]) -> float:
    return (
        max(
            0,
            float(box[2]) - float(box[0]),
        )
        * max(
            0,
            float(box[3]) - float(box[1]),
        )
    )


def aspect(box: Sequence[float]) -> float:
    return (
        max(
            float(box[2]) - float(box[0]),
            1e-6,
        )
        / max(
            float(box[3]) - float(box[1]),
            1e-6,
        )
    )


def diagonal(
    box: Sequence[float],
) -> float:
    return max(
        math.hypot(
            float(box[2]) - float(box[0]),
            float(box[3]) - float(box[1]),
        ),
        1,
    )


def center(
    box: Sequence[float],
) -> tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2,
        (float(box[1]) + float(box[3])) / 2,
    )


def bottom(
    box: Sequence[float],
) -> tuple[float, float]:
    return (
        (float(box[0]) + float(box[2])) / 2,
        float(box[3]),
    )


def iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    intersection = (
        max(
            0,
            min(
                float(first[2]),
                float(second[2]),
            )
            - max(
                float(first[0]),
                float(second[0]),
            ),
        )
        * max(
            0,
            min(
                float(first[3]),
                float(second[3]),
            )
            - max(
                float(first[1]),
                float(second[1]),
            ),
        )
    )

    union = (
        area(first)
        + area(second)
        - intersection
    )

    if union <= 0:
        return 0

    return intersection / union


def point_distance(
    first: tuple[float, float],
    second: tuple[float, float],
    norm_box: Sequence[float],
) -> float:
    return (
        math.hypot(
            first[0] - second[0],
            first[1] - second[1],
        )
        / diagonal(norm_box)
    )


def ratio_similarity(
    ratio: float,
) -> float:
    if not math.isfinite(ratio) or ratio <= 0:
        return 0

    return math.exp(
        -abs(
            math.log(ratio)
        )
    )


def load_detections(
    path: Path,
    frames: int,
    width: int,
    height: int,
) -> tuple[
    dict[int, list[Detection]],
    int,
]:
    grouped: dict[
        int,
        list[Detection],
    ] = defaultdict(list)

    count = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "frame_index",
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
        }

        missing = (
            required
            - set(
                reader.fieldnames or []
            )
        )

        if missing:
            raise ValueError(
                "detections.csv "
                f"missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:
            frame = int(
                row["frame_index"]
            )
            box = [
                float(row[key])
                for key in (
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                )
            ]

            valid = (
                0 <= frame < frames
                and 0 <= box[0] < box[2] <= width - 1
                and 0 <= box[1] < box[3] <= height - 1
            )

            if not valid:
                raise ValueError(
                    "Invalid Stage-1 "
                    f"detection: {row}"
                )

            grouped[frame].append(
                Detection(
                    frame=frame,
                    time_ms=int(
                        row["time_ms"]
                    ),
                    index=int(
                        row["detection_index"]
                    ),
                    detection_id=(
                        row["detection_id"]
                    ),
                    class_id=int(
                        row["class_id"]
                    ),
                    class_name=(
                        row["class_name"]
                    ),
                    confidence=float(
                        row["confidence"]
                    ),
                    bbox=box,
                )
            )
            count += 1

    for items in grouped.values():
        items.sort(
            key=lambda item: item.index
        )

    return dict(grouped), count


def evaluate(
    detection: Detection,
    predicted: Sequence[float],
    last: Sequence[float],
    mode: str,
    policy: Policy,
) -> Candidate:
    predicted_iou = iou(
        predicted,
        detection.bbox,
    )
    last_iou = iou(
        last,
        detection.bbox,
    )
    center_distance = point_distance(
        center(predicted),
        center(detection.bbox),
        predicted,
    )
    bottom_distance = point_distance(
        bottom(predicted),
        bottom(detection.bbox),
        predicted,
    )

    area_ratio = (
        area(detection.bbox)
        / max(
            area(predicted),
            1e-6,
        )
    )
    aspect_ratio = (
        aspect(detection.bbox)
        / max(
            aspect(predicted),
            1e-6,
        )
    )

    area_similarity = ratio_similarity(
        area_ratio
    )
    aspect_similarity = ratio_similarity(
        aspect_ratio
    )
    center_similarity = math.exp(
        -1.8 * center_distance
    )
    bottom_similarity = math.exp(
        -1.55 * bottom_distance
    )

    score = (
        0.31 * predicted_iou
        + 0.14 * last_iou
        + 0.22 * center_similarity
        + 0.13 * bottom_similarity
        + 0.09 * area_similarity
        + 0.06 * aspect_similarity
        + 0.05
        * clip01(
            detection.confidence
        )
    )

    if mode == "RECOVERY":
        minimum_confidence = (
            policy
            .recovery_min_detection_confidence
        )
        area_minimum = (
            policy
            .recovery_min_area_ratio
        )
        area_maximum = (
            policy
            .recovery_max_area_ratio
        )
        aspect_minimum = (
            policy
            .recovery_min_aspect_ratio
        )
        aspect_maximum = (
            policy
            .recovery_max_aspect_ratio
        )
        center_maximum = (
            policy
            .recovery_max_center_distance
        )
        bottom_maximum = (
            policy
            .recovery_max_bottom_distance
        )
        minimum_iou = (
            policy
            .recovery_min_predicted_iou
        )
        close_center = (
            policy
            .recovery_close_center_distance
        )
        close_bottom = (
            policy
            .recovery_close_bottom_distance
        )
    else:
        minimum_confidence = (
            policy
            .min_detection_confidence
        )
        area_minimum = (
            policy
            .active_min_area_ratio
        )
        area_maximum = (
            policy
            .active_max_area_ratio
        )
        aspect_minimum = (
            policy
            .active_min_aspect_ratio
        )
        aspect_maximum = (
            policy
            .active_max_aspect_ratio
        )
        center_maximum = (
            policy
            .active_max_center_distance
        )
        bottom_maximum = (
            policy
            .active_max_bottom_distance
        )
        minimum_iou = (
            policy
            .active_min_predicted_iou
        )
        close_center = (
            policy
            .active_close_center_distance
        )
        close_bottom = (
            policy
            .active_close_bottom_distance
        )

    reasons: list[str] = []

    if detection.confidence < minimum_confidence:
        reasons.append(
            "LOW_DETECTION_CONFIDENCE"
        )

    if not area_minimum <= area_ratio <= area_maximum:
        reasons.append(
            "AREA_RATIO_OUT_OF_RANGE"
        )

    if not (
        aspect_minimum
        <= aspect_ratio
        <= aspect_maximum
    ):
        reasons.append(
            "ASPECT_RATIO_OUT_OF_RANGE"
        )

    if center_distance > center_maximum:
        reasons.append(
            "CENTER_DISTANCE_TOO_LARGE"
        )

    if bottom_distance > bottom_maximum:
        reasons.append(
            "BOTTOM_DISTANCE_TOO_LARGE"
        )

    spatial_continuity = (
        predicted_iou >= minimum_iou
        or (
            center_distance <= close_center
            and bottom_distance <= close_bottom
        )
    )

    if not spatial_continuity:
        reasons.append(
            "NO_SPATIAL_CONTINUITY"
        )

    return Candidate(
        frame=detection.frame,
        detection_id=detection.detection_id,
        index=detection.index,
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=detection.bbox,
        predicted_iou=predicted_iou,
        last_iou=last_iou,
        center_distance=center_distance,
        bottom_distance=bottom_distance,
        area_ratio=area_ratio,
        aspect_ratio=aspect_ratio,
        area_similarity=area_similarity,
        aspect_similarity=aspect_similarity,
        center_similarity=center_similarity,
        bottom_similarity=bottom_similarity,
        score=score,
        eligible=not reasons,
        rejection_reasons=reasons,
    )


def rank_candidates(
    candidates: list[Candidate],
) -> tuple[
    list[Candidate],
    Optional[Candidate],
    Optional[Candidate],
]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.eligible,
            candidate.score,
            candidate.predicted_iou,
            candidate.confidence,
        ),
        reverse=True,
    )

    for rank, candidate in enumerate(
        ranked,
        1,
    ):
        candidate.rank = rank

    eligible = [
        candidate
        for candidate in ranked
        if candidate.eligible
    ]

    return (
        ranked,
        eligible[0] if eligible else None,
        eligible[1]
        if len(eligible) > 1
        else None,
    )


def accept(
    top: Optional[Candidate],
    second: Optional[Candidate],
    mode: str,
    policy: Policy,
) -> tuple[
    bool,
    str,
    str,
    Optional[float],
]:
    if top is None:
        return (
            False,
            "NO_ELIGIBLE_CANDIDATE",
            "NONE",
            None,
        )

    margin = (
        top.score
        - (
            second.score
            if second
            else 0
        )
    )

    if mode == "RECOVERY":
        strong = (
            top.predicted_iou
            >= policy.recovery_strong_iou
            and top.score
            >= policy.recovery_strong_score
            and margin
            >= policy.recovery_strong_margin
        )
        normal = (
            top.score
            >= policy.recovery_accept_score
            and margin
            >= policy.recovery_accept_margin
        )

        if strong:
            return (
                True,
                "STRICT_RECOVERY_STRONG_OVERLAP",
                "STRICT_RECOVERY",
                margin,
            )

        if normal:
            return (
                True,
                "STRICT_RECOVERY_SCORE_AND_MARGIN",
                "STRICT_RECOVERY",
                margin,
            )

        if (
            top.score
            < policy.recovery_accept_score
        ):
            reason = (
                "RECOVERY_SCORE_TOO_LOW"
            )
        else:
            reason = (
                "RECOVERY_MARGIN_TOO_SMALL"
            )

        return (
            False,
            reason,
            "NONE",
            margin,
        )

    strong = (
        top.predicted_iou
        >= policy.active_strong_iou
        and top.score
        >= policy.active_strong_score
        and margin
        >= policy.active_strong_margin
    )
    normal = (
        top.score
        >= policy.active_accept_score
        and margin
        >= policy.active_accept_margin
    )

    if strong:
        return (
            True,
            "ACTIVE_STRONG_OVERLAP",
            "ACTIVE_CONTINUITY",
            margin,
        )

    if normal:
        return (
            True,
            "ACTIVE_SCORE_AND_MARGIN",
            "ACTIVE_CONTINUITY",
            margin,
        )

    if top.score < policy.active_accept_score:
        reason = "ACTIVE_SCORE_TOO_LOW"
    else:
        reason = "ACTIVE_MARGIN_TOO_SMALL"

    return (
        False,
        reason,
        "NONE",
        margin,
    )


def id_conf(
    candidate: Candidate,
    margin: float,
    mode: str,
) -> float:
    denominator = (
        0.30
        if mode == "RECOVERY"
        else 0.25
    )

    value = (
        0.58 * candidate.score
        + 0.24
        * clip01(
            margin / denominator
        )
        + 0.18
        * clip01(
            candidate.confidence
        )
    )

    if mode == "RECOVERY":
        value *= 0.93

    return clip01(value)


def trk_conf(
    candidate: Candidate,
    mode: str,
) -> float:
    value = (
        0.68 * candidate.score
        + 0.32
        * clip01(
            candidate.confidence
        )
    )

    if mode == "RECOVERY":
        value *= 0.95

    return clip01(value)


def run_association(
    detections: dict[int, list[Detection]],
    frames: int,
    fps: float,
    width: int,
    height: int,
    initial: dict[str, Any],
    policy: Policy,
    max_occluded: int,
    max_recovery: int,
    print_every: int,
) -> tuple[
    list[Observation],
    list[Candidate],
    dict[str, Any],
]:
    best = initial.get(
        "best_candidate"
    )

    if (
        not initial.get(
            "reliable_detector_anchor"
        )
        or not isinstance(best, dict)
    ):
        raise RuntimeError(
            "Stage 1 has no reliable "
            "initial detector anchor"
        )

    initial_id = (
        f"f000000_d"
        f"{int(best['detection_index']):03d}"
    )

    matches = [
        detection
        for detection
        in detections.get(0, [])
        if detection.detection_id
        == initial_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Cannot resolve initial "
            f"detector anchor {initial_id}: "
            f"found {len(matches)}"
        )

    anchor = matches[0]

    memory = MotionMemory(
        width,
        height,
        policy,
    )
    memory.initialize(
        0,
        anchor.bbox,
    )

    observations = [
        Observation(
            frame=0,
            time_ms=0,
            state="INITIALIZING",
            transition="NONE->INITIALIZING",
            reason=(
                "USER_SELECTED_RELIABLE_"
                "STAGE1_ANCHOR"
            ),
            mode="USER_INITIALIZATION",
            bbox_source=(
                "RFDETR_INITIAL_ANCHOR"
            ),
            bbox=anchor.bbox,
            predicted_bbox=anchor.bbox,
            detection_id=anchor.detection_id,
            detection_index=anchor.index,
            class_id=anchor.class_id,
            class_name=anchor.class_name,
            detection_confidence=(
                anchor.confidence
            ),
            tracking_confidence=clip01(
                anchor.confidence
            ),
            identity_confidence=1.0,
            visibility=clip01(
                anchor.confidence
            ),
            top_candidate_id=(
                anchor.detection_id
            ),
            top_score=float(
                best.get(
                    "match_score",
                    1,
                )
            ),
            second_candidate_id=None,
            second_score=None,
            margin=None,
            candidate_count=sum(
                detection.class_id
                in TARGET_CLASSES
                for detection
                in detections.get(0, [])
            ),
            eligible_count=1,
            gap=0,
            predicted_iou=1.0,
            last_iou=1.0,
            center_distance=0.0,
            bottom_distance=0.0,
            area_ratio=1.0,
            aspect_ratio=1.0,
            memory_updated=True,
            risk_flags=[],
        )
    ]

    audits: list[Candidate] = []

    states = Counter(
        {"INITIALIZING": 1}
    )
    reasons = Counter(
        {
            "USER_SELECTED_RELIABLE_"
            "STAGE1_ANCHOR": 1
        }
    )

    recoveries = 0
    rejected = 0
    risk_frames = 0
    previous = "INITIALIZING"

    for frame in range(
        1,
        frames,
    ):
        if (
            memory.last_frame is None
            or memory.last_bbox is None
        ):
            raise RuntimeError(
                "Motion memory lost"
            )

        gap = (
            frame
            - memory.last_frame
        )

        predicted = memory.predict(
            frame
        )
        mode = (
            "RECOVERY"
            if gap > 1
            else "ACTIVE"
        )
        recovery_allowed = (
            gap <= max_recovery
        )

        frame_detections = [
            detection
            for detection
            in detections.get(
                frame,
                [],
            )
            if detection.class_id
            in TARGET_CLASSES
        ]

        if recovery_allowed:
            evaluations = [
                evaluate(
                    detection,
                    predicted,
                    memory.last_bbox,
                    mode,
                    policy,
                )
                for detection
                in frame_detections
            ]
        else:
            evaluations = []

        (
            ranked,
            top,
            second,
        ) = rank_candidates(
            evaluations
        )

        (
            accepted,
            reason,
            acceptance_mode,
            margin,
        ) = accept(
            top,
            second,
            mode,
            policy,
        )

        if not recovery_allowed:
            accepted = False
            reason = (
                "RECOVERY_WINDOW_"
                "EXPIRED_STAY_LOST"
            )
            acceptance_mode = "NONE"
            margin = None
            top = None
            second = None

        selected_id = None
        selected_index = None
        selected_class = None
        selected_name = None
        detection_confidence = None
        tracking = 0.0
        identity = 0.0
        visibility = 0.0
        memory_updated = False
        risk: list[str] = []

        if (
            accepted
            and top is not None
            and margin is not None
        ):
            top.selected = True
            state = "ACTIVE"
            box = list(top.bbox)
            source = (
                "RFDETR_ASSOCIATED_DETECTION"
            )

            selected_id = top.detection_id
            selected_index = top.index
            selected_class = top.class_id
            selected_name = top.class_name
            detection_confidence = (
                top.confidence
            )
            tracking = trk_conf(
                top,
                mode,
            )
            identity = id_conf(
                top,
                margin,
                mode,
            )
            visibility = clip01(
                top.confidence
            )

            if (
                margin
                < policy.review_risk_margin
            ):
                risk.append(
                    "LOW_ACCEPTED_MARGIN"
                )

            if (
                top.center_distance
                > policy.review_risk_center_jump
            ):
                risk.append(
                    "LARGE_ACCEPTED_CENTER_JUMP"
                )

            if mode == "RECOVERY":
                risk.append(
                    "STRICT_SAME_SHOT_"
                    "RECOVERY_REVIEW"
                )
                recoveries += 1

            memory_updated = memory.update(
                frame,
                top.bbox,
                identity,
                top.confidence,
            )

            if risk:
                risk_frames += 1

        else:
            rejected += 1

            if gap <= max_occluded:
                state = "OCCLUDED"
                box = list(predicted)
                source = (
                    "MOTION_PREDICTION_"
                    "UNCONFIRMED"
                )
                decay = math.exp(
                    -gap
                    / max(
                        max_occluded,
                        1,
                    )
                )
                tracking = clip01(
                    0.55 * decay
                )
                identity = clip01(
                    memory.stable_identity()
                    * 0.65
                    * decay
                )
            else:
                state = "LOST"
                box = None
                source = "NONE"

            if (
                top
                and reason.endswith(
                    "MARGIN_TOO_SMALL"
                )
            ):
                risk.append(
                    "AMBIGUOUS_CANDIDATES_"
                    "REJECTED"
                )
            elif top:
                risk.append(
                    "LOW_CONFIDENCE_"
                    "CANDIDATE_REJECTED"
                )
            elif recovery_allowed:
                risk.append(
                    "NO_ELIGIBLE_CANDIDATE"
                )

        audits.extend(ranked)

        if previous != state:
            transition = (
                f"{previous}->{state}"
            )
        else:
            transition = (
                f"{state}->{state}"
            )

        observations.append(
            Observation(
                frame=frame,
                time_ms=round(
                    frame
                    * 1000
                    / fps
                ),
                state=state,
                transition=transition,
                reason=reason,
                mode=acceptance_mode,
                bbox_source=source,
                bbox=box,
                predicted_bbox=list(
                    predicted
                ),
                detection_id=selected_id,
                detection_index=(
                    selected_index
                ),
                class_id=selected_class,
                class_name=selected_name,
                detection_confidence=(
                    detection_confidence
                ),
                tracking_confidence=(
                    tracking
                ),
                identity_confidence=(
                    identity
                ),
                visibility=visibility,
                top_candidate_id=(
                    top.detection_id
                    if top
                    else None
                ),
                top_score=(
                    top.score
                    if top
                    else None
                ),
                second_candidate_id=(
                    second.detection_id
                    if second
                    else None
                ),
                second_score=(
                    second.score
                    if second
                    else None
                ),
                margin=margin,
                candidate_count=len(
                    frame_detections
                ),
                eligible_count=sum(
                    candidate.eligible
                    for candidate
                    in ranked
                ),
                gap=gap,
                predicted_iou=(
                    top.predicted_iou
                    if top
                    else None
                ),
                last_iou=(
                    top.last_iou
                    if top
                    else None
                ),
                center_distance=(
                    top.center_distance
                    if top
                    else None
                ),
                bottom_distance=(
                    top.bottom_distance
                    if top
                    else None
                ),
                area_ratio=(
                    top.area_ratio
                    if top
                    else None
                ),
                aspect_ratio=(
                    top.aspect_ratio
                    if top
                    else None
                ),
                memory_updated=(
                    memory_updated
                ),
                risk_flags=risk,
            )
        )

        states[state] += 1
        reasons[reason] += 1
        previous = state

        if (
            print_every > 0
            and (frame + 1)
            % print_every
            == 0
        ):
            print(
                "[ASSOCIATION] "
                f"frames={frame + 1}/"
                f"{frames} "
                f"ACTIVE="
                f"{states['ACTIVE']} "
                f"OCCLUDED="
                f"{states['OCCLUDED']} "
                f"LOST="
                f"{states['LOST']}",
                flush=True,
            )

    return (
        observations,
        audits,
        {
            "state_counts": dict(
                states
            ),
            "reason_counts": dict(
                reasons
            ),
            "strict_recovery_count": (
                recoveries
            ),
            "candidate_rejection_frames": (
                rejected
            ),
            "accepted_review_risk_frames": (
                risk_frames
            ),
            "high_confidence_memory_updates": (
                memory
                .high_confidence_updates
            ),
            "motion_memory_accepted_updates": (
                memory.accepted_updates
            ),
        },
    )


def obs_row(
    observation: Observation,
) -> dict[str, Any]:
    def bbox_columns(
        prefix: str,
        box: Optional[
            Sequence[float]
        ],
    ) -> dict[str, Any]:
        if box is None:
            values = [
                "",
                "",
                "",
                "",
            ]
        else:
            values = [
                f"{float(value):.4f}"
                for value in box
            ]

        keys = [
            f"{prefix}_{name}"
            for name in (
                "x1",
                "y1",
                "x2",
                "y2",
            )
        ]
        return dict(
            zip(
                keys,
                values,
            )
        )

    row = {
        "frame_index": observation.frame,
        "frame_1based": (
            observation.frame + 1
        ),
        "time_ms": observation.time_ms,
        "state": observation.state,
        "transition": (
            observation.transition
        ),
        "decision_reason": (
            observation.reason
        ),
        "acceptance_mode": (
            observation.mode
        ),
        "bbox_source": (
            observation.bbox_source
        ),
        "selected_detection_id": (
            observation.detection_id
            or ""
        ),
        "selected_detection_index": (
            ""
            if observation
            .detection_index
            is None
            else observation
            .detection_index
        ),
        "selected_class_id": (
            ""
            if observation.class_id
            is None
            else observation.class_id
        ),
        "selected_class_name": (
            observation.class_name
            or ""
        ),
        "detection_confidence": (
            ""
            if observation
            .detection_confidence
            is None
            else
            f"{observation.detection_confidence:.6f}"
        ),
        "tracking_confidence": (
            f"{observation.tracking_confidence:.6f}"
        ),
        "identity_confidence": (
            f"{observation.identity_confidence:.6f}"
        ),
        "visibility_proxy": (
            f"{observation.visibility:.6f}"
        ),
        "top_candidate_id": (
            observation.top_candidate_id
            or ""
        ),
        "top_score": (
            ""
            if observation.top_score
            is None
            else
            f"{observation.top_score:.6f}"
        ),
        "second_candidate_id": (
            observation
            .second_candidate_id
            or ""
        ),
        "second_score": (
            ""
            if observation.second_score
            is None
            else
            f"{observation.second_score:.6f}"
        ),
        "candidate_margin": (
            ""
            if observation.margin
            is None
            else
            f"{observation.margin:.6f}"
        ),
        "candidate_count": (
            observation.candidate_count
        ),
        "eligible_candidate_count": (
            observation.eligible_count
        ),
        "gap_from_last_confirmed": (
            observation.gap
        ),
        "predicted_iou": (
            ""
            if observation.predicted_iou
            is None
            else
            f"{observation.predicted_iou:.6f}"
        ),
        "last_iou": (
            ""
            if observation.last_iou
            is None
            else
            f"{observation.last_iou:.6f}"
        ),
        "normalized_center_distance": (
            ""
            if observation.center_distance
            is None
            else
            f"{observation.center_distance:.6f}"
        ),
        "normalized_bottom_distance": (
            ""
            if observation.bottom_distance
            is None
            else
            f"{observation.bottom_distance:.6f}"
        ),
        "area_ratio": (
            ""
            if observation.area_ratio
            is None
            else
            f"{observation.area_ratio:.6f}"
        ),
        "aspect_ratio_ratio": (
            ""
            if observation.aspect_ratio
            is None
            else
            f"{observation.aspect_ratio:.6f}"
        ),
        "memory_updated": (
            observation.memory_updated
        ),
        "review_risk_flags": "|".join(
            observation.risk_flags
        ),
    }

    row.update(
        bbox_columns(
            "bbox",
            observation.bbox,
        )
    )
    row.update(
        bbox_columns(
            "predicted_bbox",
            observation.predicted_bbox,
        )
    )

    return row


def candidate_row(
    candidate: Candidate,
) -> dict[str, Any]:
    return {
        "frame_index": candidate.frame,
        "detection_id": (
            candidate.detection_id
        ),
        "detection_index": (
            candidate.index
        ),
        "class_id": candidate.class_id,
        "class_name": (
            candidate.class_name
        ),
        "confidence": (
            f"{candidate.confidence:.6f}"
        ),
        "x1": (
            f"{candidate.bbox[0]:.4f}"
        ),
        "y1": (
            f"{candidate.bbox[1]:.4f}"
        ),
        "x2": (
            f"{candidate.bbox[2]:.4f}"
        ),
        "y2": (
            f"{candidate.bbox[3]:.4f}"
        ),
        "predicted_iou": (
            f"{candidate.predicted_iou:.6f}"
        ),
        "last_iou": (
            f"{candidate.last_iou:.6f}"
        ),
        "normalized_center_distance": (
            f"{candidate.center_distance:.6f}"
        ),
        "normalized_bottom_distance": (
            f"{candidate.bottom_distance:.6f}"
        ),
        "area_ratio": (
            f"{candidate.area_ratio:.6f}"
        ),
        "aspect_ratio_ratio": (
            f"{candidate.aspect_ratio:.6f}"
        ),
        "area_similarity": (
            f"{candidate.area_similarity:.6f}"
        ),
        "aspect_similarity": (
            f"{candidate.aspect_similarity:.6f}"
        ),
        "center_similarity": (
            f"{candidate.center_similarity:.6f}"
        ),
        "bottom_similarity": (
            f"{candidate.bottom_similarity:.6f}"
        ),
        "score": (
            f"{candidate.score:.6f}"
        ),
        "hard_gate_pass": (
            candidate.eligible
        ),
        "rank": candidate.rank,
        "selected": candidate.selected,
        "rejection_reasons": "|".join(
            candidate.rejection_reasons
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fallback_fields: list[str],
) -> None:
    if rows:
        fields = list(rows[0])
    else:
        fields = fallback_fields

    temp = path.with_name(
        path.name + ".stage2.tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp, path)


def segments(
    observations: Sequence[Observation],
    fps: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = 0

    for index in range(
        1,
        len(observations) + 1,
    ):
        if (
            index < len(observations)
            and observations[index].state
            == observations[start].state
        ):
            continue

        part = observations[
            start:index
        ]

        result.append(
            {
                "segment_index": len(
                    result
                ),
                "state": part[0].state,
                "start_frame": (
                    part[0].frame
                ),
                "end_frame_inclusive": (
                    part[-1].frame
                ),
                "start_ms": (
                    part[0].time_ms
                ),
                "end_ms_exclusive": round(
                    index
                    * 1000
                    / fps
                ),
                "frame_count": len(part),
                "bbox_frame_count": sum(
                    observation.bbox
                    is not None
                    for observation
                    in part
                ),
                "mean_tracking_confidence": (
                    statistics.fmean(
                        observation
                        .tracking_confidence
                        for observation
                        in part
                    )
                ),
                "mean_identity_confidence": (
                    statistics.fmean(
                        observation
                        .identity_confidence
                        for observation
                        in part
                    )
                ),
                "start_reason": (
                    part[0].reason
                ),
            }
        )

        start = index

    return result


def longest(
    flags: Sequence[bool],
) -> int:
    best = 0
    current = 0

    for flag in flags:
        if flag:
            current += 1
        else:
            current = 0

        best = max(
            best,
            current,
        )

    return best


def timeline(
    test_name: str,
    video: dict[str, Any],
    observations: Sequence[Observation],
    timeline_segments: list[
        dict[str, Any]
    ],
    policy: Policy,
    max_occluded: int,
    max_recovery: int,
) -> dict[str, Any]:
    frames = []

    for observation in observations:
        frames.append(
            {
                "frame_index": (
                    observation.frame
                ),
                "time_ms": (
                    observation.time_ms
                ),
                "state": (
                    observation.state
                ),
                "bbox_xyxy": (
                    None
                    if observation.bbox
                    is None
                    else [
                        round(
                            value,
                            4,
                        )
                        for value
                        in observation.bbox
                    ]
                ),
                "predicted_bbox_xyxy": (
                    None
                    if observation
                    .predicted_bbox
                    is None
                    else [
                        round(
                            value,
                            4,
                        )
                        for value
                        in observation
                        .predicted_bbox
                    ]
                ),
                "bbox_source": (
                    observation
                    .bbox_source
                ),
                "tracking_confidence": (
                    round(
                        observation
                        .tracking_confidence,
                        6,
                    )
                ),
                "identity_confidence": (
                    round(
                        observation
                        .identity_confidence,
                        6,
                    )
                ),
                "visibility": (
                    round(
                        observation
                        .visibility,
                        6,
                    )
                ),
                "selected_detection_id": (
                    observation
                    .detection_id
                ),
                "candidate_margin": (
                    None
                    if observation.margin
                    is None
                    else round(
                        observation.margin,
                        6,
                    )
                ),
                "decision_reason": (
                    observation.reason
                ),
                "review_risk_flags": (
                    observation.risk_flags
                ),
                "mask_ref": None,
            }
        )

    return {
        "schema_version": (
            "kickclip.target_centric."
            "target_timeline.v1"
        ),
        "stage": STAGE,
        "script_version": VERSION,
        "target_id": "target_001",
        "test_name": test_name,
        "video": video,
        "scope": (
            "single_camera_cut_free_shot"
        ),
        "state_enum_used": [
            "INITIALIZING",
            "ACTIVE",
            "OCCLUDED",
            "LOST",
        ],
        "primary_objective": (
            "MINIMIZE_SILENT_"
            "WRONG_PLAYER_SWITCH"
        ),
        "low_confidence_policy": (
            "OCCLUDED_OR_LOST_"
            "NEVER_FORCE_CANDIDATE"
        ),
        "mask_backend": (
            "NOT_RUN_BBOX_FIRST_MVP"
        ),
        "policy": {
            **asdict(policy),
            "max_occluded_frames": (
                max_occluded
            ),
            "max_recovery_frames": (
                max_recovery
            ),
        },
        "segments": (
            timeline_segments
        ),
        "frames": frames,
    }


def text_box(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        1,
    )

    x = max(
        0,
        min(
            frame.shape[1]
            - text_width
            - 7,
            x,
        ),
    )
    y = max(
        text_height + 5,
        min(
            frame.shape[0]
            - baseline
            - 3,
            y,
        ),
    )

    cv2.rectangle(
        frame,
        (
            x,
            y - text_height - 5,
        ),
        (
            x + text_width + 7,
            y + baseline + 3,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x + 3,
            y - 2,
        ),
        font,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_box(
    frame: np.ndarray,
    box: Sequence[float],
    color: tuple[int, int, int],
    thickness: int,
    label: str,
) -> None:
    x1, y1, x2, y2 = map(
        lambda value: round(
            float(value)
        ),
        box,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    text_box(
        frame,
        label,
        x1,
        y1 - 5,
        color,
    )


def render(
    video: Path,
    output: Path,
    observations: Sequence[Observation],
    detections: dict[int, list[Detection]],
    fps: float,
    width: int,
    height: int,
    print_every: int,
) -> None:
    temp = output.with_name(
        output.stem
        + ".stage2.tmp.mp4"
    )

    capture = cv2.VideoCapture(
        str(video)
    )
    writer = cv2.VideoWriter(
        str(temp),
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
            "Cannot open preview "
            "reader/writer"
        )

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
    }

    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            if frame_index >= len(
                observations
            ):
                raise RuntimeError(
                    "Video has more frames "
                    "than observations"
                )

            observation = observations[
                frame_index
            ]
            color = colors[
                observation.state
            ]

            if (
                observation.detection_id
                is None
                and observation
                .top_candidate_id
            ):
                top = next(
                    (
                        detection
                        for detection
                        in detections.get(
                            frame_index,
                            [],
                        )
                        if detection
                        .detection_id
                        == observation
                        .top_candidate_id
                    ),
                    None,
                )

                if top:
                    draw_box(
                        frame,
                        top.bbox,
                        (
                            110,
                            110,
                            110,
                        ),
                        1,
                        (
                            "TOP REJECTED "
                            "CANDIDATE"
                        ),
                    )

            if (
                observation.predicted_bbox
                is not None
                and observation.state
                != "ACTIVE"
            ):
                draw_box(
                    frame,
                    observation
                    .predicted_bbox,
                    (
                        255,
                        180,
                        0,
                    ),
                    1,
                    "MOTION PREDICTION",
                )

            if observation.bbox is not None:
                draw_box(
                    frame,
                    observation.bbox,
                    color,
                    3,
                    (
                        f"TARGET "
                        f"{observation.state} "
                        f"trk="
                        f"{observation.tracking_confidence:.2f} "
                        f"id="
                        f"{observation.identity_confidence:.2f}"
                    ),
                )

            header = (
                f"Stage 2 | "
                f"frame={frame_index} | "
                f"t="
                f"{observation.time_ms / 1000:.3f}s "
                f"| state="
                f"{observation.state} "
                f"| reason="
                f"{observation.reason}"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (
                    width,
                    25,
                ),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                header,
                (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )

            if (
                observation.top_score
                is None
            ):
                score_text = "NA"
            else:
                score_text = (
                    f"{observation.top_score:.3f}"
                )

            if observation.margin is None:
                margin_text = "NA"
            else:
                margin_text = (
                    f"{observation.margin:.3f}"
                )

            footer = (
                f"score={score_text} "
                f"margin={margin_text} "
                f"eligible="
                f"{observation.eligible_count}/"
                f"{observation.candidate_count}"
            )

            if observation.risk_flags:
                footer += (
                    " | REVIEW="
                    + ",".join(
                        observation.risk_flags
                    )
                )

            cv2.rectangle(
                frame,
                (
                    0,
                    height - 24,
                ),
                (
                    width,
                    height,
                ),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                footer,
                (
                    6,
                    height - 7,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1

            if (
                print_every > 0
                and frame_index
                % print_every
                == 0
            ):
                print(
                    "[PREVIEW] "
                    f"frames={frame_index}/"
                    f"{len(observations)}",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()

    if (
        frame_index
        != len(observations)
        or not temp.is_file()
        or temp.stat().st_size == 0
    ):
        raise RuntimeError(
            "Preview output invalid: "
            f"frames={frame_index}/"
            f"{len(observations)}"
        )

    os.replace(temp, output)


def report(
    summary: dict[str, Any],
) -> str:
    counts = summary["counts"]
    metrics = summary[
        "service_proxy_metrics"
    ]
    policy = summary["policy"]

    return f"""# KickClip Target-Centric Tracking V1 — Stage 2 Conservative Association

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Test: `{summary['test_name']}`
- Primary objective: `MINIMIZE_SILENT_WRONG_PLAYER_SWITCH`

## Policy

```text
RF-DETR player/goalkeeper candidates
→ motion prediction + spatial/shape/confidence evidence
→ score and candidate margin
→ uncertain candidate rejected
→ short uncertainty: OCCLUDED prediction
→ longer uncertainty: LOST with null bbox
Max OCCLUDED: {policy['max_occluded_frames']} frames
Max strict same-shot recovery: {policy['max_recovery_frames']} frames
ACTIVE score/margin: {policy['active_accept_score']} / {policy['active_accept_margin']}
RECOVERY score/margin: {policy['recovery_accept_score']} / {policy['recovery_accept_margin']}
ReID/training/threshold search/Global ID/GTA/V7: NONE
Results
INITIALIZING / ACTIVE: {counts['state_counts'].get('INITIALIZING', 0)} / {counts['state_counts'].get('ACTIVE', 0)}
OCCLUDED / LOST: {counts['state_counts'].get('OCCLUDED', 0)} / {counts['state_counts'].get('LOST', 0)}
Strict recoveries: {counts['strict_recovery_count']}
Rejected frames: {counts['candidate_rejection_frames']}
Accepted review-risk frames: {counts['accepted_review_risk_frames']}
ACTIVE ratio: {metrics['active_frame_ratio']:.4%}
Bbox-present ratio: {metrics['bbox_present_frame_ratio']:.4%}
Longest OCCLUDED / LOST: {metrics['longest_occluded_run_frames']} / {metrics['longest_lost_run_frames']} frames

These are operational proxies, not ground-truth identity metrics. Stage 2 authorizes mandatory visual review only.
Open target_tracking_preview.mp4 and verify that every green bbox stays on the selected player, overlap frames become OCCLUDED/LOST instead of switching, and LOST frames contain no fabricated bbox.
"""

def main() -> int:
    arguments = args()
    started = time.perf_counter()

    generated = (
        datetime.now(
            timezone.utc
        )
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    root = (
        arguments.project_root
        .expanduser()
        .resolve()
    )

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if not root.is_dir():
        raise FileNotFoundError(root)

    if (
        not arguments.test_name
        or any(
            character not in allowed
            for character
            in arguments.test_name
        )
    ):
        raise ValueError(
            "Invalid --test-name"
        )

    test_dir = (
        resolve(
            root,
            arguments.output_root,
        )
        / arguments.test_name
    )

    paths = {
        "manifest": (
            test_dir
            / "input_manifest.json"
        ),
        "audit": (
            test_dir
            / "audit.json"
        ),
        "detections": (
            test_dir
            / "detections.csv"
        ),
        "initial": (
            test_dir
            / "initial_target_detection.json"
        ),
        "stage1": (
            test_dir
            / "stage1_detection_summary.json"
        ),
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    manifest = read_json(
        paths["manifest"]
    )
    audit = read_json(
        paths["audit"]
    )
    initial = read_json(
        paths["initial"]
    )
    stage1 = read_json(
        paths["stage1"]
    )

    if (
        manifest.get("status")
        != "PASS"
        or audit.get("status")
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 0 must be PASS"
        )

    if (
        stage1.get("status")
        != "PASS"
        or stage1.get("decision")
        != (
            "AUTHORIZE_STAGE2_"
            "TARGET_ASSOCIATION"
        )
    ):
        raise RuntimeError(
            "Stage 1 did not "
            "authorize Stage 2"
        )

    if not initial.get(
        "reliable_detector_anchor"
    ):
        raise RuntimeError(
            "Stage 1 reliable "
            "detector anchor is false"
        )

    video_info = stage1["video"]
    video = Path(
        video_info["path"]
    ).resolve()

    if (
        not video.is_file()
        or sha256(video)
        != video_info["sha256"]
    ):
        raise RuntimeError(
            "Video missing or changed "
            "after Stage 1"
        )

    detection_hash = sha256(
        paths["detections"]
    )
    expected_hash = (
        stage1.get(
            "outputs",
            {},
        )
        .get(
            "detections_csv_sha256"
        )
    )

    if (
        expected_hash
        and detection_hash
        != expected_hash
    ):
        raise RuntimeError(
            "detections.csv changed "
            "after Stage 1"
        )

    frames = int(
        video_info["processed_frames"]
    )
    width = int(
        video_info["width"]
    )
    height = int(
        video_info["height"]
    )
    fps = float(
        video_info["fps"]
    )

    policy = Policy(
        max_occluded_seconds=float(
            arguments
            .max_occluded_seconds
        ),
        max_recovery_seconds=float(
            arguments
            .max_recovery_seconds
        ),
    )

    if (
        policy.max_occluded_seconds
        <= 0
        or policy.max_recovery_seconds
        < policy.max_occluded_seconds
    ):
        raise ValueError(
            "Invalid occluded/"
            "recovery duration"
        )

    max_occluded = max(
        1,
        round(
            policy.max_occluded_seconds
            * fps
        ),
    )
    max_recovery = max(
        max_occluded,
        round(
            policy.max_recovery_seconds
            * fps
        ),
    )

    prepare(
        test_dir,
        arguments.overwrite_stage2,
    )

    (
        detections,
        row_count,
    ) = load_detections(
        paths["detections"],
        frames,
        width,
        height,
    )

    expected_rows = int(
        stage1["counts"][
            "total_detections"
        ]
    )

    if row_count != expected_rows:
        raise RuntimeError(
            "Detection row mismatch: "
            f"{row_count}/"
            f"{expected_rows}"
        )

    (
        observations,
        candidates,
        diagnostics,
    ) = run_association(
        detections,
        frames,
        fps,
        width,
        height,
        initial,
        policy,
        max_occluded,
        max_recovery,
        arguments.print_every,
    )

    if (
        len(observations) != frames
        or any(
            observation.state
            not in STATES
            for observation
            in observations
        )
    ):
        raise RuntimeError(
            "Invalid observation output"
        )

    if any(
        observation.state == "LOST"
        and observation.bbox is not None
        for observation
        in observations
    ):
        raise RuntimeError(
            "LOST frame contains bbox"
        )

    if any(
        observation.state == "ACTIVE"
        and observation.detection_id
        is None
        for observation
        in observations
    ):
        raise RuntimeError(
            "ACTIVE frame "
            "lacks detection"
        )

    frame_rows = [
        obs_row(observation)
        for observation
        in observations
    ]
    candidate_rows = [
        candidate_row(candidate)
        for candidate
        in candidates
    ]

    frame_csv = (
        test_dir
        / "frame_observations.csv"
    )
    candidate_csv = (
        test_dir
        / "stage2_candidate_audit.csv"
    )

    write_csv(
        frame_csv,
        frame_rows,
        [],
    )

    fallback_candidate = Candidate(
        frame=0,
        detection_id="",
        index=0,
        class_id=0,
        class_name="",
        confidence=0,
        bbox=[0, 0, 1, 1],
        predicted_iou=0,
        last_iou=0,
        center_distance=0,
        bottom_distance=0,
        area_ratio=1,
        aspect_ratio=1,
        area_similarity=1,
        aspect_similarity=1,
        center_similarity=1,
        bottom_similarity=1,
        score=0,
        eligible=False,
        rejection_reasons=[],
    )

    write_csv(
        candidate_csv,
        candidate_rows,
        list(
            candidate_row(
                fallback_candidate
            )
        ),
    )

    timeline_segments = segments(
        observations,
        fps,
    )

    video_contract = {
        "path": str(video),
        "sha256": (
            video_info["sha256"]
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": (
            frames / fps
        ),
    }

    timeline_path = (
        test_dir
        / "target_timeline.json"
    )

    atomic_json(
        timeline_path,
        timeline(
            arguments.test_name,
            video_contract,
            observations,
            timeline_segments,
            policy,
            max_occluded,
            max_recovery,
        ),
    )

    preview = (
        test_dir
        / "target_tracking_preview.mp4"
    )

    if not arguments.no_preview:
        render(
            video,
            preview,
            observations,
            detections,
            fps,
            width,
            height,
            arguments.print_every,
        )

    state_counts = Counter(
        observation.state
        for observation
        in observations
    )

    active_count = (
        state_counts["ACTIVE"]
    )

    accepted_margins = [
        observation.margin
        for observation
        in observations
        if (
            observation.state
            == "ACTIVE"
            and observation.margin
            is not None
        )
    ]

    metrics = {
        "active_frame_ratio": (
            active_count / frames
        ),
        "bbox_present_frame_ratio": (
            sum(
                observation.bbox
                is not None
                for observation
                in observations
            )
            / frames
        ),
        "longest_occluded_run_frames": (
            longest(
                [
                    observation.state
                    == "OCCLUDED"
                    for observation
                    in observations
                ]
            )
        ),
        "longest_lost_run_frames": (
            longest(
                [
                    observation.state
                    == "LOST"
                    for observation
                    in observations
                ]
            )
        ),
        "minimum_accepted_margin": (
            min(accepted_margins)
            if accepted_margins
            else None
        ),
        "mean_accepted_margin": (
            statistics.fmean(
                accepted_margins
            )
            if accepted_margins
            else None
        ),
        "mean_active_tracking_confidence": (
            statistics.fmean(
                observation
                .tracking_confidence
                for observation
                in observations
                if observation.state
                == "ACTIVE"
            )
            if active_count
            else None
        ),
        "mean_active_identity_confidence": (
            statistics.fmean(
                observation
                .identity_confidence
                for observation
                in observations
                if observation.state
                == "ACTIVE"
            )
            if active_count
            else None
        ),
        "accepted_review_risk_frames": (
            diagnostics[
                "accepted_review_risk_frames"
            ]
        ),
        "ground_truth_identity_metrics_available": (
            False
        ),
        "silent_switch_rate_measured": (
            False
        ),
    }

    summary = {
        "schema_version": (
            "kickclip.target_centric."
            "stage2_association_summary.v1"
        ),
        "stage": STAGE,
        "script_version": VERSION,
        "generated_at": generated,
        "status": "PASS",
        "decision": (
            "AUTHORIZE_MANDATORY_"
            "STAGE2_VISUAL_REVIEW"
        ),
        "test_name": (
            arguments.test_name
        ),
        "video": video_contract,
        "input_contract": {
            "stage0_status": (
                manifest["status"]
            ),
            "stage1_status": (
                stage1["status"]
            ),
            "stage1_decision": (
                stage1["decision"]
            ),
            "detections_csv": str(
                paths["detections"]
            ),
            "detections_csv_sha256": (
                detection_hash
            ),
            "initial_detection_id": (
                observations[0]
                .detection_id
            ),
        },
        "policy": {
            **asdict(policy),
            "max_occluded_frames": (
                max_occluded
            ),
            "max_recovery_frames": (
                max_recovery
            ),
            "policy_origin": (
                "FROZEN_V1_INITIAL_"
                "CONSERVATIVE_POLICY_"
                "NO_THRESHOLD_SEARCH"
            ),
        },
        "counts": {
            "state_counts": dict(
                state_counts
            ),
            "state_segment_count": len(
                timeline_segments
            ),
            "target_candidate_evaluations": (
                len(candidates)
            ),
            **diagnostics,
        },
        "service_proxy_metrics": (
            metrics
        ),
        "safety_invariants": {
            "lost_frames_have_null_bbox": (
                True
            ),
            "active_frames_have_selected_detection": (
                True
            ),
            "low_confidence_candidate_forcing": (
                False
            ),
            "appearance_only_linking": (
                False
            ),
            "target_memory_updated_only_from_high_confidence_observations": (
                True
            ),
        },
        "outputs": {
            "target_timeline": str(
                timeline_path
            ),
            "target_timeline_sha256": (
                sha256(timeline_path)
            ),
            "frame_observations": str(
                frame_csv
            ),
            "frame_observations_sha256": (
                sha256(frame_csv)
            ),
            "candidate_audit": str(
                candidate_csv
            ),
            "candidate_audit_sha256": (
                sha256(candidate_csv)
            ),
            "preview": (
                None
                if arguments.no_preview
                else str(preview)
            ),
            "preview_sha256": (
                None
                if arguments.no_preview
                else sha256(preview)
            ),
        },
        "runtime": {
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
        },
        "training_performed": False,
        "reid_inference_performed": (
            False
        ),
        "threshold_search_performed": (
            False
        ),
        "global_linking_performed": (
            False
        ),
        "gta_performed": False,
        "v7_logic_used": False,
        "protected_v4_v7_outputs_modified": (
            False
        ),
        "performance_certification": (
            "NOT_CERTIFIED_REQUIRES_"
            "VISUAL_REVIEW_AND_LATER_"
            "GROUND_TRUTH_EVALUATION"
        ),
    }

    summary_path = (
        test_dir
        / "stage2_association_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        test_dir
        / "stage2_association_report.md",
        report(summary),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V1 Stage 2 complete"
    )
    print(
        "Status                 : PASS"
    )
    print(
        "Decision               : "
        "AUTHORIZE_MANDATORY_"
        "STAGE2_VISUAL_REVIEW"
    )
    print(
        f"Frames                 : "
        f"{frames}"
    )
    print(
        "INITIALIZING / ACTIVE  : "
        f"{state_counts['INITIALIZING']} / "
        f"{state_counts['ACTIVE']}"
    )
    print(
        "OCCLUDED / LOST        : "
        f"{state_counts['OCCLUDED']} / "
        f"{state_counts['LOST']}"
    )
    print(
        "Strict recoveries      : "
        f"{diagnostics['strict_recovery_count']}"
    )
    print(
        "Rejected frames        : "
        f"{diagnostics['candidate_rejection_frames']}"
    )
    print(
        "Accepted risk frames   : "
        f"{diagnostics['accepted_review_risk_frames']}"
    )
    print(
        f"ACTIVE ratio           : "
        f"{metrics['active_frame_ratio']:.4%}"
    )
    print(
        "Silent-switch metric   : "
        "NOT MEASURED "
        "(visual review required)"
    )
    print(
        "Training / ReID        : "
        "NONE / NONE"
    )
    print(
        "Global linking / V7    : "
        "NONE / NONE"
    )
    print(
        f"Output                 : "
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
            "Stage 2 interrupted",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            "Stage 2 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)