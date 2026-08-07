#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-B0.

Build target-agnostic local candidate tracklets inside one manually reviewed
post-cut shot.

Input shot:
    confirmed cross-shot cut frame <= frame < next camera-cut frame

This stage intentionally does not:
- compare candidates to target memory;
- run Sports OSNet;
- choose or link a target;
- update target memory;
- cross a camera cut;
- modify frozen Phase-1 files.

The output requires visual candidate-recall review. The reviewer may identify
which tracklet(s) show the selected target, but that label is evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


STAGE = "stage3b0_build_postcut_candidate_tracklets"
VERSION = "target-centric-v2-stage3b0-1.0.0"

OUTPUT_FILES = (
    "stage3b0_postcut_shot_contract.json",
    "stage3b0_tracklets.jsonl",
    "stage3b0_tracklet_assignments.csv",
    "stage3b0_tracklet_contact_sheet.jpg",
    "stage3b0_postcut_tracklets_preview.mp4",
    "stage3b0_summary.json",
    "stage3b0_report.md",
)

OUTPUT_DIRECTORIES = (
    "stage3b0_tracklet_strips",
)


@dataclass
class Observation:
    frame_index: int
    detection_id: str
    bbox_xyxy: list[float]
    confidence: float


@dataclass
class WorkingTrack:
    internal_id: int
    observations: list[Observation] = field(
        default_factory=list
    )

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    @property
    def previous(self) -> Optional[Observation]:
        if len(self.observations) < 2:
            return None

        return self.observations[-2]

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    def append(
        self,
        observation: Observation,
    ) -> None:
        if (
            self.observations
            and observation.frame_index
            <= self.last.frame_index
        ):
            raise ValueError(
                "Track observations must be chronological"
            )

        self.observations.append(
            observation
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
            "runs/target_centric_tracking_v2"
        ),
    )

    parser.add_argument(
        "--next-cut-frame",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--next-cut-review-status",
        choices=(
            "PASS",
            "FAIL",
        ),
        required=True,
    )

    parser.add_argument(
        "--memory-review-status",
        choices=(
            "PASS",
            "FAIL",
        ),
        required=True,
    )

    parser.add_argument(
        "--reviewer",
        default="USER",
    )

    parser.add_argument(
        "--review-note",
        default="",
    )

    parser.add_argument(
        "--stage2-helper",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage2_run_conservative_target_association.py"
        ),
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--minimum-tracklet-frames",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minimum-predicted-iou",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--maximum-center-distance",
        type=float,
        default=1.60,
    )

    parser.add_argument(
        "--minimum-area-ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--maximum-area-ratio",
        type=float,
        default=4.00,
    )

    parser.add_argument(
        "--minimum-match-score",
        type=float,
        default=0.17,
    )

    parser.add_argument(
        "--maximum-contact-sheet-tracklets",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


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
            character not in allowed
            for character in value
        )
    ):
        raise ValueError(
            "Invalid --test-name"
        )

    return value


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


def write_jsonl(
    path: Path,
    rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

        stream.flush()
        os.fsync(
            stream.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    fieldnames: Sequence[str],
) -> None:
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
                fieldnames
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


def load_module(
    name: str,
    path: Path,
) -> Any:
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Cannot import module: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def prepare_outputs(
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        output_dir
        / name
        for name in OUTPUT_FILES
        if (
            output_dir
            / name
        ).exists()
    ]

    existing.extend(
        output_dir
        / name
        for name in OUTPUT_DIRECTORIES
        if (
            output_dir
            / name
        ).exists()
    )

    if (
        existing
        and not overwrite
    ):
        raise FileExistsError(
            "Stage 3-B0 outputs already exist. "
            "Use --overwrite:\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )

    for path in existing:
        if path.is_dir():
            shutil.rmtree(
                path
            )
        else:
            path.unlink()


def bbox_area(
    bbox: Sequence[float],
) -> float:
    return (
        max(
            0.0,
            float(
                bbox[2]
                - bbox[0]
            ),
        )
        * max(
            0.0,
            float(
                bbox[3]
                - bbox[1]
            ),
        )
    )


def bbox_center(
    bbox: Sequence[float],
) -> tuple[
    float,
    float,
]:
    return (
        (
            float(
                bbox[0]
            )
            + float(
                bbox[2]
            )
        )
        / 2.0,
        (
            float(
                bbox[1]
            )
            + float(
                bbox[3]
            )
        )
        / 2.0,
    )


def bbox_diagonal(
    bbox: Sequence[float],
) -> float:
    return max(
        1.0,
        math.hypot(
            float(
                bbox[2]
                - bbox[0]
            ),
            float(
                bbox[3]
                - bbox[1]
            ),
        ),
    )


def bbox_iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    intersection_width = max(
        0.0,
        min(
            float(
                first[2]
            ),
            float(
                second[2]
            ),
        )
        - max(
            float(
                first[0]
            ),
            float(
                second[0]
            ),
        ),
    )

    intersection_height = max(
        0.0,
        min(
            float(
                first[3]
            ),
            float(
                second[3]
            ),
        )
        - max(
            float(
                first[1]
            ),
            float(
                second[1]
            ),
        ),
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    union = (
        bbox_area(
            first
        )
        + bbox_area(
            second
        )
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


def center_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    first_x, first_y = bbox_center(
        first
    )

    second_x, second_y = bbox_center(
        second
    )

    return (
        math.hypot(
            second_x
            - first_x,
            second_y
            - first_y,
        )
        / bbox_diagonal(
            first
        )
    )


def area_ratio(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return (
        bbox_area(
            second
        )
        / max(
            bbox_area(
                first
            ),
            1e-8,
        )
    )


def predict_bbox(
    track: WorkingTrack,
    frame_index: int,
) -> list[float]:
    last = track.last

    if track.previous is None:
        return list(
            last.bbox_xyxy
        )

    previous = track.previous

    previous_center = bbox_center(
        previous.bbox_xyxy
    )

    last_center = bbox_center(
        last.bbox_xyxy
    )

    source_gap = max(
        1,
        last.frame_index
        - previous.frame_index,
    )

    target_gap = max(
        1,
        frame_index
        - last.frame_index,
    )

    velocity_x = (
        last_center[0]
        - previous_center[0]
    ) / source_gap

    velocity_y = (
        last_center[1]
        - previous_center[1]
    ) / source_gap

    predicted_center_x = (
        last_center[0]
        + velocity_x
        * target_gap
    )

    predicted_center_y = (
        last_center[1]
        + velocity_y
        * target_gap
    )

    width = (
        last.bbox_xyxy[2]
        - last.bbox_xyxy[0]
    )

    height = (
        last.bbox_xyxy[3]
        - last.bbox_xyxy[1]
    )

    return [
        predicted_center_x
        - width
        / 2.0,
        predicted_center_y
        - height
        / 2.0,
        predicted_center_x
        + width
        / 2.0,
        predicted_center_y
        + height
        / 2.0,
    ]


def association_features(
    track: WorkingTrack,
    observation: Observation,
) -> dict[
    str,
    float,
]:
    predicted = predict_bbox(
        track,
        observation.frame_index,
    )

    predicted_iou = bbox_iou(
        predicted,
        observation.bbox_xyxy,
    )

    last_iou = bbox_iou(
        track.last.bbox_xyxy,
        observation.bbox_xyxy,
    )

    normalized_center_distance = center_distance(
        predicted,
        observation.bbox_xyxy,
    )

    current_area_ratio = area_ratio(
        track.last.bbox_xyxy,
        observation.bbox_xyxy,
    )

    motion_similarity = math.exp(
        -normalized_center_distance
    )

    score = (
        0.45
        * predicted_iou
        + 0.25
        * last_iou
        + 0.20
        * motion_similarity
        + 0.10
        * observation.confidence
    )

    return {
        "predicted_iou": (
            predicted_iou
        ),
        "last_iou": (
            last_iou
        ),
        "normalized_center_distance": (
            normalized_center_distance
        ),
        "area_ratio": (
            current_area_ratio
        ),
        "score": (
            score
        ),
    }


def greedy_assignment(
    valid_pairs: Sequence[
        tuple[
            float,
            int,
            int,
        ]
    ],
) -> list[
    tuple[
        int,
        int,
    ]
]:
    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()

    assignments: list[
        tuple[
            int,
            int,
        ]
    ] = []

    for _, track_index, detection_index in sorted(
        valid_pairs,
        reverse=True,
    ):
        if (
            track_index
            in assigned_tracks
            or detection_index
            in assigned_detections
        ):
            continue

        assigned_tracks.add(
            track_index
        )

        assigned_detections.add(
            detection_index
        )

        assignments.append(
            (
                track_index,
                detection_index,
            )
        )

    return assignments


def hungarian_assignment(
    active_track_count: int,
    observation_count: int,
    valid_pairs: Sequence[
        tuple[
            float,
            int,
            int,
        ]
    ],
) -> list[
    tuple[
        int,
        int,
    ]
]:
    if (
        active_track_count == 0
        or observation_count == 0
        or not valid_pairs
    ):
        return []

    try:
        from scipy.optimize import (
            linear_sum_assignment,
        )

    except Exception:
        return greedy_assignment(
            valid_pairs
        )

    very_large_cost = 1e6

    cost = np.full(
        (
            active_track_count,
            observation_count,
        ),
        very_large_cost,
        dtype=np.float64,
    )

    valid_lookup: set[
        tuple[
            int,
            int,
        ]
    ] = set()

    for (
        score,
        track_index,
        observation_index,
    ) in valid_pairs:
        cost[
            track_index,
            observation_index,
        ] = (
            -score
        )

        valid_lookup.add(
            (
                track_index,
                observation_index,
            )
        )

    track_indices, observation_indices = (
        linear_sum_assignment(
            cost
        )
    )

    assignments: list[
        tuple[
            int,
            int,
        ]
    ] = []

    for (
        track_index,
        observation_index,
    ) in zip(
        track_indices.tolist(),
        observation_indices.tolist(),
    ):
        if (
            track_index,
            observation_index,
        ) not in valid_lookup:
            continue

        assignments.append(
            (
                track_index,
                observation_index,
            )
        )

    return assignments


def build_tracklets(
    by_frame: Mapping[
        int,
        Sequence[Any],
    ],
    start_frame: int,
    end_frame_inclusive: int,
    max_age: int,
    minimum_predicted_iou: float,
    maximum_center_distance: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    minimum_match_score: float,
) -> tuple[
    list[WorkingTrack],
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    working_tracks: list[
        WorkingTrack
    ] = []

    next_internal_id = 1

    association_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for frame_index in range(
        start_frame,
        end_frame_inclusive
        + 1,
    ):
        detections = list(
            by_frame.get(
                frame_index,
                [],
            )
        )

        observations = [
            Observation(
                frame_index=frame_index,
                detection_id=str(
                    detection.detection_id
                ),
                bbox_xyxy=[
                    float(
                        value
                    )
                    for value
                    in detection.bbox
                ],
                confidence=float(
                    detection.confidence
                ),
            )
            for detection in detections
        ]

        active_tracks = [
            track
            for track
            in working_tracks
            if (
                frame_index
                - track.end_frame
            )
            <= (
                max_age
                + 1
            )
        ]

        valid_pairs: list[
            tuple[
                float,
                int,
                int,
            ]
        ] = []

        feature_lookup: dict[
            tuple[
                int,
                int,
            ],
            dict[
                str,
                float,
            ],
        ] = {}

        for track_index, track in enumerate(
            active_tracks
        ):
            for observation_index, observation in enumerate(
                observations
            ):
                features = association_features(
                    track,
                    observation,
                )

                feature_lookup[
                    (
                        track_index,
                        observation_index,
                    )
                ] = features

                geometry_pass = (
                    features[
                        "predicted_iou"
                    ]
                    >= minimum_predicted_iou
                    or features[
                        "normalized_center_distance"
                    ]
                    <= maximum_center_distance
                )

                area_pass = (
                    minimum_area_ratio
                    <= features[
                        "area_ratio"
                    ]
                    <= maximum_area_ratio
                )

                score_pass = (
                    features[
                        "score"
                    ]
                    >= minimum_match_score
                )

                if (
                    geometry_pass
                    and area_pass
                    and score_pass
                ):
                    valid_pairs.append(
                        (
                            features[
                                "score"
                            ],
                            track_index,
                            observation_index,
                        )
                    )

        assignments = hungarian_assignment(
            len(
                active_tracks
            ),
            len(
                observations
            ),
            valid_pairs,
        )

        matched_observations: set[
            int
        ] = set()

        for (
            track_index,
            observation_index,
        ) in assignments:
            track = active_tracks[
                track_index
            ]

            observation = observations[
                observation_index
            ]

            features = feature_lookup[
                (
                    track_index,
                    observation_index,
                )
            ]

            track.append(
                observation
            )

            matched_observations.add(
                observation_index
            )

            association_rows.append(
                {
                    "frame_index": (
                        frame_index
                    ),
                    "internal_track_id": (
                        track.internal_id
                    ),
                    "detection_id": (
                        observation.detection_id
                    ),
                    "association_type": (
                        "MATCHED_EXISTING_TRACK"
                    ),
                    "predicted_iou": (
                        features[
                            "predicted_iou"
                        ]
                    ),
                    "last_iou": (
                        features[
                            "last_iou"
                        ]
                    ),
                    "normalized_center_distance": (
                        features[
                            "normalized_center_distance"
                        ]
                    ),
                    "area_ratio": (
                        features[
                            "area_ratio"
                        ]
                    ),
                    "association_score": (
                        features[
                            "score"
                        ]
                    ),
                }
            )

        for observation_index, observation in enumerate(
            observations
        ):
            if observation_index in matched_observations:
                continue

            track = WorkingTrack(
                internal_id=next_internal_id,
            )

            next_internal_id += 1

            track.append(
                observation
            )

            working_tracks.append(
                track
            )

            association_rows.append(
                {
                    "frame_index": (
                        frame_index
                    ),
                    "internal_track_id": (
                        track.internal_id
                    ),
                    "detection_id": (
                        observation.detection_id
                    ),
                    "association_type": (
                        "NEW_TRACK"
                    ),
                    "predicted_iou": "",
                    "last_iou": "",
                    "normalized_center_distance": "",
                    "area_ratio": "",
                    "association_score": "",
                }
            )

    return (
        working_tracks,
        association_rows,
    )


def representative_observations(
    track: WorkingTrack,
    count: int = 3,
) -> list[Observation]:
    observations = track.observations

    if len(
        observations
    ) <= count:
        return list(
            observations
        )

    positions = np.linspace(
        0,
        len(
            observations
        )
        - 1,
        num=count,
    )

    selected_indices: list[int] = []

    for position in positions:
        index = int(
            round(
                float(
                    position
                )
            )
        )

        if index not in selected_indices:
            selected_indices.append(
                index
            )

    return [
        observations[
            index
        ]
        for index in selected_indices
    ]


def read_frame(
    video: Path,
    frame_index: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(
        str(
            video
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    ok, frame = capture.read()
    capture.release()

    if (
        not ok
        or frame is None
    ):
        raise RuntimeError(
            f"Cannot read frame {frame_index}"
        )

    return frame


def crop_bbox(
    frame: np.ndarray,
    bbox: Sequence[float],
    padding: float = 0.08,
) -> np.ndarray:
    frame_height, frame_width = (
        frame.shape[
            :2
        ]
    )

    x1, y1, x2, y2 = [
        float(
            value
        )
        for value in bbox
    ]

    padding_x = (
        x2
        - x1
    ) * padding

    padding_y = (
        y2
        - y1
    ) * padding

    left = max(
        0,
        int(
            math.floor(
                x1
                - padding_x
            )
        ),
    )

    top = max(
        0,
        int(
            math.floor(
                y1
                - padding_y
            )
        ),
    )

    right = min(
        frame_width,
        int(
            math.ceil(
                x2
                + padding_x
            )
        ),
    )

    bottom = min(
        frame_height,
        int(
            math.ceil(
                y2
                + padding_y
            )
        ),
    )

    if (
        right <= left
        or bottom <= top
    ):
        raise ValueError(
            f"Invalid crop bbox: {bbox}"
        )

    return frame[
        top:
        bottom,
        left:
        right,
    ].copy()


def fit_crop(
    crop: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    source_height, source_width = (
        crop.shape[
            :2
        ]
    )

    scale = min(
        width
        / source_width,
        height
        / source_height,
    )

    resized_width = max(
        1,
        int(
            round(
                source_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                source_height
                * scale
            )
        ),
    )

    resized = cv2.resize(
        crop,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    offset_x = (
        width
        - resized_width
    ) // 2

    offset_y = (
        height
        - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y
        + resized_height,
        offset_x:
        offset_x
        + resized_width,
    ] = resized

    return canvas


def make_tracklet_strip(
    video: Path,
    candidate_id: str,
    track: WorkingTrack,
    output_path: Path,
) -> np.ndarray:
    representatives = representative_observations(
        track,
        count=3,
    )

    tiles: list[
        np.ndarray
    ] = []

    for observation in representatives:
        frame = read_frame(
            video,
            observation.frame_index,
        )

        crop = crop_bbox(
            frame,
            observation.bbox_xyxy,
        )

        tile = fit_crop(
            crop,
            width=180,
            height=210,
        )

        cv2.rectangle(
            tile,
            (
                0,
                0,
            ),
            (
                tile.shape[
                    1
                ]
                - 1,
                32,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        cv2.putText(
            tile,
            (
                f"f={observation.frame_index} "
                f"c={observation.confidence:.2f}"
            ),
            (
                5,
                21,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        tiles.append(
            tile
        )

    while len(
        tiles
    ) < 3:
        tiles.append(
            np.zeros_like(
                tiles[
                    0
                ]
            )
        )

    strip = np.hstack(
        tiles
    )

    header_height = 44

    canvas = np.zeros(
        (
            strip.shape[
                0
            ]
            + header_height,
            strip.shape[
                1
            ],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        header_height:,
        :,
    ] = strip

    mean_confidence = float(
        np.mean(
            [
                observation.confidence
                for observation in track.observations
            ]
        )
    )

    label = (
        f"{candidate_id} "
        f"frames={track.start_frame}-{track.end_frame} "
        f"hits={len(track.observations)} "
        f"mean_conf={mean_confidence:.3f}"
    )

    cv2.putText(
        canvas,
        label,
        (
            7,
            28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        canvas,
    ):
        raise RuntimeError(
            f"Cannot write strip: {output_path}"
        )

    return canvas


def make_contact_sheet(
    strips: Sequence[np.ndarray],
    output_path: Path,
    columns: int = 2,
) -> None:
    if not strips:
        raise RuntimeError(
            "No tracklet strips available"
        )

    maximum_height = max(
        strip.shape[0]
        for strip in strips
    )

    maximum_width = max(
        strip.shape[1]
        for strip in strips
    )

    normalized: list[
        np.ndarray
    ] = []

    for strip in strips:
        canvas = np.zeros(
            (
                maximum_height,
                maximum_width,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            :strip.shape[0],
            :strip.shape[1],
        ] = strip

        normalized.append(
            canvas
        )

    row_count = int(
        math.ceil(
            len(
                normalized
            )
            / columns
        )
    )

    blank = np.zeros_like(
        normalized[
            0
        ]
    )

    normalized.extend(
        [
            blank
        ]
        * (
            row_count
            * columns
            - len(
                normalized
            )
        )
    )

    sheet = np.vstack(
        [
            np.hstack(
                normalized[
                    row_index
                    * columns:
                    (
                        row_index
                        + 1
                    )
                    * columns
                ]
            )
            for row_index in range(
                row_count
            )
        ]
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        sheet,
    ):
        raise RuntimeError(
            f"Cannot write contact sheet: {output_path}"
        )


def deterministic_color(
    candidate_number: int,
) -> tuple[
    int,
    int,
    int,
]:
    return (
        int(
            60
            + (
                candidate_number
                * 97
            )
            % 190
        ),
        int(
            60
            + (
                candidate_number
                * 57
            )
            % 190
        ),
        int(
            60
            + (
                candidate_number
                * 137
            )
            % 190
        ),
    )


def render_preview(
    video: Path,
    output_path: Path,
    start_frame: int,
    end_frame_inclusive: int,
    fps: float,
    width: int,
    height: int,
    assignments_by_frame: Mapping[
        int,
        Sequence[
            tuple[
                str,
                Observation,
            ]
        ],
    ],
    candidate_number_by_id: Mapping[
        str,
        int,
    ],
) -> None:
    capture = cv2.VideoCapture(
        str(
            video
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
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
        capture.release()

        raise RuntimeError(
            f"Cannot open video writer: {output_path}"
        )

    for frame_index in range(
        start_frame,
        end_frame_inclusive
        + 1,
    ):
        ok, frame = capture.read()

        if (
            not ok
            or frame is None
        ):
            writer.release()
            capture.release()

            raise RuntimeError(
                f"Cannot read frame {frame_index}"
            )

        for (
            candidate_id,
            observation,
        ) in assignments_by_frame.get(
            frame_index,
            [],
        ):
            candidate_number = (
                candidate_number_by_id[
                    candidate_id
                ]
            )

            color = deterministic_color(
                candidate_number
            )

            x1, y1, x2, y2 = [
                int(
                    round(
                        value
                    )
                )
                for value in observation.bbox_xyxy
            ]

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color,
                2,
            )

            cv2.putText(
                frame,
                candidate_id,
                (
                    max(
                        2,
                        x1,
                    ),
                    max(
                        20,
                        y1
                        - 6,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(
            frame,
            (
                0,
                0,
            ),
            (
                width
                - 1,
                39,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        cv2.putText(
            frame,
            (
                "Stage3-B0 TARGET-AGNOSTIC CANDIDATES "
                f"| frame={frame_index} "
                f"| shot={start_frame}-{end_frame_inclusive}"
            ),
            (
                8,
                26,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        writer.write(
            frame
        )

    writer.release()
    capture.release()


def build_report(
    summary: Mapping[
        str,
        Any,
    ],
) -> str:
    counts = summary[
        "counts"
    ]

    shot = summary[
        "postcut_shot"
    ]

    return f"""# KickClip Target-Centric Tracking V2 — Stage 3-B0

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Post-cut shot: `{shot['start_frame']}–{shot['end_frame_inclusive']}`
- Next camera cut: `{shot['next_cut_frame']}`
- Raw local tracks: `{counts['raw_local_track_count']}`
- Kept candidate tracklets: `{counts['kept_candidate_tracklet_count']}`
- Assigned detections: `{counts['kept_assignment_count']}`

## Review objective

Review the preview and contact sheet and determine whether the original selected
target appears in at least one candidate tracklet. Record all candidate IDs
showing the target because the target may be fragmented across multiple local
tracklets.

This manual identity label is candidate-recall evaluation only. It is not used
by the automatic candidate generator, and this stage performs no target-memory
comparison, ReID inference, target selection, or cross-shot linking.
"""


def main() -> int:
    arguments = parse_args()

    if (
        arguments.next_cut_review_status
        != "PASS"
    ):
        raise RuntimeError(
            "The next camera-cut review must be PASS"
        )

    if (
        arguments.memory_review_status
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 3-A2R2 target and negative "
            "memory visual review must be PASS"
        )

    if arguments.max_age < 0:
        raise ValueError(
            "--max-age cannot be negative"
        )

    if (
        arguments.minimum_tracklet_frames
        < 1
    ):
        raise ValueError(
            "--minimum-tracklet-frames must be positive"
        )

    if not (
        0.0
        <= arguments.minimum_predicted_iou
        <= 1.0
    ):
        raise ValueError(
            "Invalid --minimum-predicted-iou"
        )

    if arguments.maximum_center_distance <= 0:
        raise ValueError(
            "--maximum-center-distance must be positive"
        )

    if not (
        0.0
        < arguments.minimum_area_ratio
        <= 1.0
        <= arguments.maximum_area_ratio
    ):
        raise ValueError(
            "Invalid area-ratio bounds"
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

    test_name = validate_test_name(
        arguments.test_name
    )

    output_dir = (
        resolve(
            root,
            arguments.output_root,
        )
        / test_name
    )

    if not output_dir.is_dir():
        raise FileNotFoundError(
            output_dir
        )

    prepare_outputs(
        output_dir,
        arguments.overwrite,
    )

    stage3a0_path = (
        output_dir
        / "stage3a0_summary.json"
    )

    stage3a1_path = (
        output_dir
        / "stage3a1_summary.json"
    )

    stage3a2_path = (
        output_dir
        / "stage3a2_summary.json"
    )

    stage3a2r2_path = (
        output_dir
        / "stage3a2r2_summary.json"
    )

    pure_memory_path = (
        output_dir
        / "stage3a2r2_pure_target_memory.json"
    )

    for path in (
        stage3a0_path,
        stage3a1_path,
        stage3a2_path,
        stage3a2r2_path,
        pure_memory_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    stage3a0 = read_json(
        stage3a0_path
    )

    stage3a1 = read_json(
        stage3a1_path
    )

    stage3a2 = read_json(
        stage3a2_path
    )

    stage3a2r2 = read_json(
        stage3a2r2_path
    )

    pure_memory = read_json(
        pure_memory_path
    )

    expected_memory_decision = (
        "AUTHORIZE_MANDATORY_"
        "STAGE3A2R2_PURE_MEMORY_"
        "VISUAL_REVIEW"
    )

    if (
        stage3a2r2.get(
            "status"
        )
        != "PASS"
        or stage3a2r2.get(
            "decision"
        )
        != expected_memory_decision
    ):
        raise RuntimeError(
            "Stage 3-A2R2 did not authorize "
            "pure-memory visual review"
        )

    if (
        pure_memory.get(
            "status"
        )
        != (
            "PENDING_TARGET_AND_NEGATIVE_"
            "VISUAL_REVIEW"
        )
    ):
        raise RuntimeError(
            "Unexpected pure target-memory status"
        )

    first_cut_frame = int(
        stage3a1[
            "confirmed_cut"
        ][
            "cut_frame"
        ]
    )

    next_cut_frame = int(
        arguments.next_cut_frame
    )

    frame_count = int(
        stage3a1[
            "video"
        ][
            "frame_count"
        ]
    )

    if not (
        first_cut_frame
        < next_cut_frame
        < frame_count
    ):
        raise ValueError(
            "--next-cut-frame must be after "
            "the confirmed cross-shot cut and "
            "before the end of the video"
        )

    candidate_match = None

    raw_cut_candidates = (
        stage3a0.get(
            "cut_detection",
            {}
        )
        .get(
            "candidates",
            []
        )
    )

    for candidate in raw_cut_candidates:
        if (
            isinstance(
                candidate,
                dict,
            )
            and int(
                candidate.get(
                    "cut_frame",
                    -1,
                )
            )
            == next_cut_frame
        ):
            candidate_match = dict(
                candidate
            )

            break

    if candidate_match is None:
        raise RuntimeError(
            "The next cut frame does not exactly "
            "match a Stage 3-A0 candidate"
        )

    search_start_frame = (
        first_cut_frame
    )

    search_end_frame = (
        next_cut_frame
        - 1
    )

    video = Path(
        str(
            stage3a1[
                "video"
            ][
                "path"
            ]
        )
    ).resolve()

    if (
        not video.is_file()
        or sha256_file(
            video
        )
        != stage3a1[
            "video"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Video is missing or changed"
        )

    width = int(
        stage3a1[
            "video"
        ][
            "width"
        ]
    )

    height = int(
        stage3a1[
            "video"
        ][
            "height"
        ]
    )

    fps = float(
        stage3a1[
            "video"
        ][
            "fps"
        ]
    )

    source_contract = (
        stage3a2[
            "phase1_source"
        ]
    )

    detections_path = Path(
        str(
            source_contract[
                "detections"
            ]
        )
    ).resolve()

    if (
        not detections_path.is_file()
        or sha256_file(
            detections_path
        )
        != source_contract[
            "detections_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen RF-DETR detections "
            "are missing or changed"
        )

    stage2_helper_path = resolve(
        root,
        arguments.stage2_helper,
    )

    if not stage2_helper_path.is_file():
        raise FileNotFoundError(
            stage2_helper_path
        )

    stage2 = load_module(
        "kickclip_stage3b0_stage2",
        stage2_helper_path,
    )

    (
        by_frame,
        total_detection_count,
    ) = stage2.load_detections(
        detections_path,
        frame_count,
        width,
        height,
    )

    shot_detection_count = sum(
        len(
            by_frame.get(
                frame_index,
                [],
            )
        )
        for frame_index in range(
            search_start_frame,
            search_end_frame
            + 1,
        )
    )

    (
        raw_tracks,
        raw_association_rows,
    ) = build_tracklets(
        by_frame=by_frame,
        start_frame=search_start_frame,
        end_frame_inclusive=search_end_frame,
        max_age=arguments.max_age,
        minimum_predicted_iou=(
            arguments.minimum_predicted_iou
        ),
        maximum_center_distance=(
            arguments.maximum_center_distance
        ),
        minimum_area_ratio=(
            arguments.minimum_area_ratio
        ),
        maximum_area_ratio=(
            arguments.maximum_area_ratio
        ),
        minimum_match_score=(
            arguments.minimum_match_score
        ),
    )

    kept_tracks = [
        track
        for track in raw_tracks
        if len(
            track.observations
        )
        >= arguments.minimum_tracklet_frames
    ]

    kept_tracks.sort(
        key=lambda track: (
            track.start_frame,
            track.end_frame,
            track.internal_id,
        )
    )

    internal_to_candidate: dict[
        int,
        str,
    ] = {}

    candidate_number_by_id: dict[
        str,
        int,
    ] = {}

    for candidate_number, track in enumerate(
        kept_tracks,
        start=1,
    ):
        candidate_id = (
            f"postcut_track_"
            f"{candidate_number:04d}"
        )

        internal_to_candidate[
            track.internal_id
        ] = candidate_id

        candidate_number_by_id[
            candidate_id
        ] = candidate_number

    tracklet_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    assignment_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    assignments_by_frame: dict[
        int,
        list[
            tuple[
                str,
                Observation,
            ]
        ],
    ] = {}

    for track in kept_tracks:
        candidate_id = (
            internal_to_candidate[
                track.internal_id
            ]
        )

        confidences = [
            observation.confidence
            for observation in track.observations
        ]

        frame_indices = [
            observation.frame_index
            for observation in track.observations
        ]

        frame_gaps = [
            current
            - previous
            for previous, current in zip(
                frame_indices,
                frame_indices[1:],
            )
        ]

        representative_frames = [
            observation.frame_index
            for observation in (
                representative_observations(
                    track,
                    count=3,
                )
            )
        ]

        tracklet_rows.append(
            {
                "candidate_id": (
                    candidate_id
                ),
                "internal_track_id": (
                    track.internal_id
                ),
                "start_frame": (
                    track.start_frame
                ),
                "end_frame_inclusive": (
                    track.end_frame
                ),
                "span_frames": (
                    track.end_frame
                    - track.start_frame
                    + 1
                ),
                "detection_count": (
                    len(
                        track.observations
                    )
                ),
                "coverage_ratio": (
                    len(
                        track.observations
                    )
                    / (
                        track.end_frame
                        - track.start_frame
                        + 1
                    )
                ),
                "mean_confidence": float(
                    np.mean(
                        confidences
                    )
                ),
                "minimum_confidence": float(
                    np.min(
                        confidences
                    )
                ),
                "maximum_confidence": float(
                    np.max(
                        confidences
                    )
                ),
                "maximum_internal_gap": (
                    max(
                        frame_gaps
                    )
                    if frame_gaps
                    else 0
                ),
                "representative_frames": (
                    representative_frames
                ),
                "target_identity": (
                    "UNKNOWN_UNREVIEWED"
                ),
                "authorized_for_cross_shot_link": (
                    False
                ),
            }
        )

        for observation in track.observations:
            assignment_rows.append(
                {
                    "candidate_id": (
                        candidate_id
                    ),
                    "internal_track_id": (
                        track.internal_id
                    ),
                    "frame_index": (
                        observation.frame_index
                    ),
                    "detection_id": (
                        observation.detection_id
                    ),
                    "confidence": (
                        observation.confidence
                    ),
                    "x1": (
                        observation.bbox_xyxy[0]
                    ),
                    "y1": (
                        observation.bbox_xyxy[1]
                    ),
                    "x2": (
                        observation.bbox_xyxy[2]
                    ),
                    "y2": (
                        observation.bbox_xyxy[3]
                    ),
                }
            )

            assignments_by_frame.setdefault(
                observation.frame_index,
                [],
            ).append(
                (
                    candidate_id,
                    observation,
                )
            )

    tracklet_jsonl_path = (
        output_dir
        / "stage3b0_tracklets.jsonl"
    )

    assignment_csv_path = (
        output_dir
        / "stage3b0_tracklet_assignments.csv"
    )

    write_jsonl(
        tracklet_jsonl_path,
        tracklet_rows,
    )

    write_csv(
        assignment_csv_path,
        assignment_rows,
        (
            list(
                assignment_rows[
                    0
                ]
            )
            if assignment_rows
            else [
                "candidate_id",
                "frame_index",
            ]
        ),
    )

    strip_dir = (
        output_dir
        / "stage3b0_tracklet_strips"
    )

    strip_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    display_tracks = sorted(
        kept_tracks,
        key=lambda track: (
            len(
                track.observations
            ),
            (
                track.end_frame
                - track.start_frame
            ),
        ),
        reverse=True,
    )[
        :
        arguments.maximum_contact_sheet_tracklets
    ]

    strips: list[
        np.ndarray
    ] = []

    for track in display_tracks:
        candidate_id = (
            internal_to_candidate[
                track.internal_id
            ]
        )

        strip_path = (
            strip_dir
            / (
                f"{candidate_id}.jpg"
            )
        )

        strips.append(
            make_tracklet_strip(
                video,
                candidate_id,
                track,
                strip_path,
            )
        )

    contact_sheet_path = (
        output_dir
        / "stage3b0_tracklet_contact_sheet.jpg"
    )

    make_contact_sheet(
        strips,
        contact_sheet_path,
        columns=2,
    )

    preview_path = (
        output_dir
        / "stage3b0_postcut_tracklets_preview.mp4"
    )

    render_preview(
        video=video,
        output_path=preview_path,
        start_frame=search_start_frame,
        end_frame_inclusive=search_end_frame,
        fps=fps,
        width=width,
        height=height,
        assignments_by_frame=(
            assignments_by_frame
        ),
        candidate_number_by_id=(
            candidate_number_by_id
        ),
    )

    shot_contract_path = (
        output_dir
        / "stage3b0_postcut_shot_contract.json"
    )

    shot_contract = {
        "stage": STAGE,
        "version": VERSION,
        "test_name": test_name,
        "source_cross_shot_cut_frame": (
            first_cut_frame
        ),
        "postcut_shot": {
            "start_frame": (
                search_start_frame
            ),
            "end_frame_inclusive": (
                search_end_frame
            ),
            "frame_count": (
                search_end_frame
                - search_start_frame
                + 1
            ),
            "start_seconds": (
                search_start_frame
                / fps
            ),
            "end_seconds_exclusive": (
                next_cut_frame
                / fps
            ),
            "next_cut_frame": (
                next_cut_frame
            ),
            "next_cut_candidate_rank": (
                candidate_match.get(
                    "rank"
                )
            ),
            "next_cut_candidate_score": (
                candidate_match.get(
                    "combined_score"
                )
            ),
            "next_cut_review_status": (
                arguments.next_cut_review_status
            ),
        },
        "memory_review": {
            "status": (
                arguments.memory_review_status
            ),
            "reviewer": (
                arguments.reviewer
            ),
            "note": (
                arguments.review_note
            ),
        },
        "candidate_generation_policy": {
            "target_agnostic": True,
            "max_age": (
                arguments.max_age
            ),
            "minimum_tracklet_frames": (
                arguments.minimum_tracklet_frames
            ),
            "minimum_predicted_iou": (
                arguments.minimum_predicted_iou
            ),
            "maximum_center_distance": (
                arguments.maximum_center_distance
            ),
            "minimum_area_ratio": (
                arguments.minimum_area_ratio
            ),
            "maximum_area_ratio": (
                arguments.maximum_area_ratio
            ),
            "minimum_match_score": (
                arguments.minimum_match_score
            ),
            "training_performed": False,
            "reid_inference_performed": False,
            "target_memory_read_for_matching": False,
            "target_selection_performed": False,
            "cross_shot_linking_performed": False,
        },
    }

    atomic_json(
        shot_contract_path,
        shot_contract,
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
        "status": "PASS",
        "decision": (
            "AUTHORIZE_MANDATORY_"
            "STAGE3B0_CANDIDATE_RECALL_"
            "VISUAL_REVIEW"
        ),
        "test_name": (
            test_name
        ),
        "postcut_shot": (
            shot_contract[
                "postcut_shot"
            ]
        ),
        "memory_review": (
            shot_contract[
                "memory_review"
            ]
        ),
        "counts": {
            "full_video_detection_count": (
                total_detection_count
            ),
            "postcut_shot_detection_count": (
                shot_detection_count
            ),
            "raw_local_track_count": (
                len(
                    raw_tracks
                )
            ),
            "kept_candidate_tracklet_count": (
                len(
                    kept_tracks
                )
            ),
            "discarded_short_track_count": (
                len(
                    raw_tracks
                )
                - len(
                    kept_tracks
                )
            ),
            "kept_assignment_count": (
                len(
                    assignment_rows
                )
            ),
            "contact_sheet_tracklet_count": (
                len(
                    display_tracks
                )
            ),
        },
        "inputs": {
            "video": str(
                video
            ),
            "video_sha256": (
                sha256_file(
                    video
                )
            ),
            "detections": str(
                detections_path
            ),
            "detections_sha256": (
                sha256_file(
                    detections_path
                )
            ),
            "stage3a0_summary": str(
                stage3a0_path
            ),
            "stage3a1_summary": str(
                stage3a1_path
            ),
            "stage3a2r2_summary": str(
                stage3a2r2_path
            ),
            "pure_target_memory": str(
                pure_memory_path
            ),
            "stage2_helper": str(
                stage2_helper_path
            ),
            "stage2_helper_sha256": (
                sha256_file(
                    stage2_helper_path
                )
            ),
        },
        "outputs": {
            "shot_contract": str(
                shot_contract_path
            ),
            "tracklets": str(
                tracklet_jsonl_path
            ),
            "assignments": str(
                assignment_csv_path
            ),
            "tracklet_strips": str(
                strip_dir
            ),
            "contact_sheet": str(
                contact_sheet_path
            ),
            "preview": str(
                preview_path
            ),
        },
        "safety_invariants": {
            "target_memory_used_for_candidate_generation": (
                False
            ),
            "sports_osnet_inference_performed": (
                False
            ),
            "manual_target_identity_used_for_generation": (
                False
            ),
            "target_selected": (
                False
            ),
            "cross_shot_linking_performed": (
                False
            ),
            "candidate_tracklets_cross_camera_cut": (
                False
            ),
            "frozen_phase1_modified": (
                False
            ),
            "threshold_search_performed": (
                False
            ),
        },
    }

    summary_path = (
        output_dir
        / "stage3b0_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3b0_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric Tracking V2 "
        "Stage 3-B0 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        "Decision                       : "
        "AUTHORIZE_MANDATORY_"
        "STAGE3B0_CANDIDATE_RECALL_"
        "VISUAL_REVIEW"
    )

    print(
        f"Post-cut candidate shot        : "
        f"frames {search_start_frame}-"
        f"{search_end_frame}"
    )

    print(
        f"Next camera cut                : "
        f"frame {next_cut_frame}"
    )

    print(
        f"Post-cut detections            : "
        f"{shot_detection_count}"
    )

    print(
        f"Raw local tracks               : "
        f"{len(raw_tracks)}"
    )

    print(
        f"Kept candidate tracklets       : "
        f"{len(kept_tracks)}"
    )

    print(
        f"Kept assignments               : "
        f"{len(assignment_rows)}"
    )

    print(
        "Target memory comparison       : NONE"
    )

    print(
        "Sports OSNet inference         : NONE"
    )

    print(
        "Target selection               : NONE"
    )

    print(
        "Cross-shot linking             : NONE"
    )

    print(
        f"Tracklet contact sheet         : "
        f"{contact_sheet_path}"
    )

    print(
        f"Tracklet preview               : "
        f"{preview_path}"
    )

    print(
        f"Output                         : "
        f"{output_dir}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "Stage 3-B0 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-B0 fatal error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )