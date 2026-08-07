#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 2-D2.

Confirmed-track hysteresis for Stage 2-D1 temporal re-entry episodes.
Strict Stage 2-D1 acquisition decisions remain unchanged. After REACQUIRED,
a real RF-DETR detection on the already-selected geometric tracklet is kept
through short per-frame appearance drops. The track stops only after sustained
identity contradiction. No new detector/ReID inference or threshold search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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


STAGE = "stage2d2_confirmed_track_hysteresis"
VERSION = "target-centric-v1-stage2d2-1.0.0"

OUTPUT_NAMES = (
    "stage2d2_hysteresis_audit.csv",
    "stage2d2_frame_observations.csv",
    "stage2d2_target_timeline.json",
    "stage2d2_hysteresis_preview.mp4",
    "stage2d2_summary.json",
    "stage2d2_report.md",
)

UNCERTAIN_STATES = {
    "LOST",
    "SEARCHING",
    "AMBIGUOUS",
    "ABSENT",
    "TERMINATED",
}

CONFIRMED_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "REACQUIRED",
    "USER_CONFIRMED",
}


@dataclass(frozen=True)
class HysteresisPolicy:
    geometry_iou_min: float = 0.12
    geometry_center_distance_max: float = 0.90

    # 이 기준을 통과하면 appearance까지 안정적인 frame이다.
    # 통과하지 못해도 확정된 geometric tracklet이 이어지면 bbox를 유지한다.
    soft_target_similarity_min: float = 0.66
    soft_target_max_similarity_min: float = 0.70
    soft_negative_margin_min: float = 0.00

    # Appearance 반증과 geometry 붕괴가 동시에 지속될 때 사용하는 조건이다.
    strong_target_similarity_max: float = 0.64
    strong_target_max_similarity_max: float = 0.67
    strong_negative_margin_max: float = -0.10
    strong_geometry_required_frames: int = 4

    # Geometry가 자연스러워도 appearance 반증이 훨씬 강하게 지속되면 중단한다.
    extreme_target_similarity_max: float = 0.58
    extreme_negative_margin_max: float = -0.16
    extreme_required_frames: int = 6

    weak_identity_confidence_floor: float = 0.36
    weak_tracking_confidence_floor: float = 0.50

    # 실제 detection이 없는 frame에는 bbox를 만들지 않는다.
    max_missing_detection_gap: int = 3


@dataclass
class AuditRow:
    frame_index: int
    episode_index: int
    tracklet_id: str
    original_state: str
    output_state: str
    detection_id: str
    detection_present: bool

    temporal_iou: Optional[float]
    center_distance: Optional[float]
    geometry_continuous: bool

    target_similarity: Optional[float]
    target_max_similarity: Optional[float]
    negative_margin: Optional[float]

    soft_appearance_pass: bool
    strong_contradiction_frame: bool
    extreme_contradiction_frame: bool

    strong_contradiction_streak: int
    extreme_contradiction_streak: int

    hysteresis_applied: bool
    track_terminated: bool
    decision_reason: str


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
        "--stage2-helper",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage2_run_conservative_target_association.py"
        ),
    )

    parser.add_argument(
        "--stage2b-helper",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage2b_run_same_shot_reentry_reacquisition.py"
        ),
    )

    parser.add_argument(
        "--stage2d-helper",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage2d_run_iterative_multi_reentry.py"
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-stage2d2",
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
        Mapping[
            str,
            Any,
        ]
    ],
    fields: Sequence[str],
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
                fields
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
            f"Cannot import: {path}"
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
            "Stage 2-D2 outputs already exist. "
            "Use --overwrite-stage2d2:\n"
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


def optional_float(
    value: Any,
) -> Optional[float]:
    if (
        value is None
        or value == ""
    ):
        return None

    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        return None

    return result


def copy_and_validate_frames(
    timeline: Mapping[
        str,
        Any,
    ],
    frame_count: int,
    width: int,
    height: int,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    raw_frames = timeline.get(
        "frames"
    )

    if (
        not isinstance(
            raw_frames,
            list,
        )
        or len(
            raw_frames
        )
        != frame_count
    ):
        raise ValueError(
            "Stage 2-D1 frame count mismatch"
        )

    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for (
        index,
        raw,
    ) in enumerate(
        raw_frames
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
            != index
        ):
            raise ValueError(
                "Invalid Stage 2-D1 "
                f"frame {index}"
            )

        item = dict(
            raw
        )

        item[
            "target_similarity"
        ] = optional_float(
            item.get(
                "target_similarity"
            )
        )

        item[
            "target_max_similarity"
        ] = optional_float(
            item.get(
                "target_max_similarity"
            )
        )

        item[
            "negative_max_similarity"
        ] = optional_float(
            item.get(
                "negative_max_similarity"
            )
        )

        item[
            "negative_margin"
        ] = optional_float(
            item.get(
                "negative_margin"
            )
        )

        item[
            "candidate_margin"
        ] = optional_float(
            item.get(
                "candidate_margin"
            )
        )

        item[
            "tracking_confidence"
        ] = float(
            item.get(
                "tracking_confidence",
                0.0,
            )
        )

        item[
            "identity_confidence"
        ] = float(
            item.get(
                "identity_confidence",
                0.0,
            )
        )

        item[
            "review_required"
        ] = bool(
            item.get(
                "review_required",
                False,
            )
        )

        item[
            "hysteresis_status"
        ] = None

        item[
            "hysteresis_review_required"
        ] = False

        bbox = item.get(
            "bbox_xyxy"
        )

        if bbox is not None:
            bbox = [
                float(
                    value
                )
                for value
                in bbox
            ]

            (
                x1,
                y1,
                x2,
                y2,
            ) = bbox

            if not (
                0
                <= x1
                < x2
                <= width
                - 1
                and 0
                <= y1
                < y2
                <= height
                - 1
            ):
                raise ValueError(
                    "Invalid bbox at "
                    f"frame {index}: "
                    f"{bbox}"
                )

            item[
                "bbox_xyxy"
            ] = bbox

        if (
            item.get(
                "state"
            )
            in UNCERTAIN_STATES
            and bbox is not None
        ):
            raise ValueError(
                "Uncertain state "
                "has bbox at "
                f"frame {index}"
            )

        result.append(
            item
        )

    return result


def appearance_soft_pass(
    target: Optional[float],
    target_max: Optional[float],
    negative_margin: Optional[float],
    policy: HysteresisPolicy,
) -> bool:
    return (
        target is not None
        and target_max is not None
        and negative_margin is not None
        and target
        >= policy
        .soft_target_similarity_min
        and target_max
        >= policy
        .soft_target_max_similarity_min
        and negative_margin
        >= policy
        .soft_negative_margin_min
    )


def identity_confidence(
    target: Optional[float],
    target_max: Optional[float],
    negative_margin: Optional[float],
    floor: float,
) -> float:
    if (
        target is None
        or target_max is None
        or negative_margin is None
    ):
        return floor

    value = (
        0.58
        * target
        + 0.22
        * target_max
        + 0.20
        * max(
            0.0,
            min(
                1.0,
                negative_margin
                + 0.5,
            ),
        )
    )

    return max(
        floor,
        min(
            1.0,
            value,
        ),
    )


def apply_hysteresis(
    frames: list[
        dict[
            str,
            Any,
        ]
    ],
    episode: Mapping[
        str,
        Any,
    ],
    tracklet: Any,
    stage2: Any,
    policy: HysteresisPolicy,
) -> list[AuditRow]:
    episode_index = int(
        episode[
            "episode_index"
        ]
    )

    tracklet_id = str(
        episode[
            "selected_tracklet_id"
        ]
    )

    confirmation = int(
        episode[
            "confirmation_frame"
        ]
    )

    end_frame = int(
        episode[
            "selected_tracklet_end_frame"
        ]
    )

    detections = {
        int(
            detection.frame
        ): detection
        for detection
        in tracklet.detections
        if (
            confirmation
            <= int(
                detection.frame
            )
            <= end_frame
        )
    }

    audit: list[
        AuditRow
    ] = []

    previous_bbox: Optional[
        list[float]
    ] = None

    strong_streak = 0
    extreme_streak = 0
    missing_streak = 0
    terminated = False

    for frame_index in range(
        confirmation,
        end_frame
        + 1,
    ):
        original = frames[
            frame_index
        ]

        original_state = str(
            original[
                "state"
            ]
        )

        detection = detections.get(
            frame_index
        )

        target = optional_float(
            original.get(
                "target_similarity"
            )
        )

        target_max = optional_float(
            original.get(
                "target_max_similarity"
            )
        )

        negative_margin = optional_float(
            original.get(
                "negative_margin"
            )
        )

        if terminated:
            frames[
                frame_index
            ].update(
                state="SEARCHING",
                bbox_xyxy=None,
                bbox_source="NONE",
                selected_detection_id=None,
                tracking_confidence=0.0,
                identity_confidence=0.0,
                decision_reason=(
                    "HYSTERESIS_TRACK_"
                    "ALREADY_TERMINATED"
                ),
                review_required=True,
                hysteresis_status=(
                    "TERMINATED"
                ),
                hysteresis_review_required=True,
            )

            audit.append(
                AuditRow(
                    frame_index=(
                        frame_index
                    ),
                    episode_index=(
                        episode_index
                    ),
                    tracklet_id=(
                        tracklet_id
                    ),
                    original_state=(
                        original_state
                    ),
                    output_state=(
                        "SEARCHING"
                    ),
                    detection_id="",
                    detection_present=(
                        detection
                        is not None
                    ),
                    temporal_iou=None,
                    center_distance=None,
                    geometry_continuous=False,
                    target_similarity=(
                        target
                    ),
                    target_max_similarity=(
                        target_max
                    ),
                    negative_margin=(
                        negative_margin
                    ),
                    soft_appearance_pass=False,
                    strong_contradiction_frame=False,
                    extreme_contradiction_frame=False,
                    strong_contradiction_streak=(
                        strong_streak
                    ),
                    extreme_contradiction_streak=(
                        extreme_streak
                    ),
                    hysteresis_applied=False,
                    track_terminated=True,
                    decision_reason=(
                        "TRACK_ALREADY_TERMINATED"
                    ),
                )
            )

            continue

        if detection is None:
            missing_streak += 1
            strong_streak = 0
            extreme_streak = 0

            if (
                missing_streak
                <= policy
                .max_missing_detection_gap
            ):
                output_state = (
                    "OCCLUDED"
                )

                reason = (
                    "HYSTERESIS_SHORT_"
                    "MISSING_DETECTION_GAP"
                )

            else:
                output_state = (
                    "SEARCHING"
                )

                reason = (
                    "HYSTERESIS_MISSING_"
                    "DETECTION_GAP_EXPIRED"
                )

            frames[
                frame_index
            ].update(
                state=output_state,
                bbox_xyxy=None,
                bbox_source="NONE",
                selected_detection_id=None,
                tracking_confidence=0.0,
                identity_confidence=0.0,
                decision_reason=(
                    reason
                ),
                review_required=(
                    output_state
                    == "SEARCHING"
                ),
                hysteresis_status=(
                    "MISSING_DETECTION"
                ),
                hysteresis_review_required=(
                    output_state
                    == "SEARCHING"
                ),
            )

            audit.append(
                AuditRow(
                    frame_index=(
                        frame_index
                    ),
                    episode_index=(
                        episode_index
                    ),
                    tracklet_id=(
                        tracklet_id
                    ),
                    original_state=(
                        original_state
                    ),
                    output_state=(
                        output_state
                    ),
                    detection_id="",
                    detection_present=False,
                    temporal_iou=None,
                    center_distance=None,
                    geometry_continuous=False,
                    target_similarity=(
                        target
                    ),
                    target_max_similarity=(
                        target_max
                    ),
                    negative_margin=(
                        negative_margin
                    ),
                    soft_appearance_pass=False,
                    strong_contradiction_frame=False,
                    extreme_contradiction_frame=False,
                    strong_contradiction_streak=0,
                    extreme_contradiction_streak=0,
                    hysteresis_applied=False,
                    track_terminated=False,
                    decision_reason=(
                        reason
                    ),
                )
            )

            continue

        missing_streak = 0

        bbox = [
            float(
                value
            )
            for value
            in detection.bbox
        ]

        if previous_bbox is None:
            temporal_iou = None
            center_distance = None
            geometry_ok = True

        else:
            temporal_iou = float(
                stage2.iou(
                    previous_bbox,
                    bbox,
                )
            )

            center_distance = float(
                stage2.point_distance(
                    stage2.center(
                        previous_bbox
                    ),
                    stage2.center(
                        bbox
                    ),
                    previous_bbox,
                )
            )

            geometry_ok = (
                temporal_iou
                >= policy
                .geometry_iou_min
                or center_distance
                <= policy
                .geometry_center_distance_max
            )

        soft_pass = (
            appearance_soft_pass(
                target,
                target_max,
                negative_margin,
                policy,
            )
        )

        strong_frame = (
            target is not None
            and target_max is not None
            and negative_margin is not None
            and target
            < policy
            .strong_target_similarity_max
            and target_max
            < policy
            .strong_target_max_similarity_max
            and negative_margin
            < policy
            .strong_negative_margin_max
            and not geometry_ok
        )

        extreme_frame = (
            target is not None
            and negative_margin is not None
            and target
            < policy
            .extreme_target_similarity_max
            and negative_margin
            < policy
            .extreme_negative_margin_max
        )

        if strong_frame:
            strong_streak += 1
        else:
            strong_streak = 0

        if extreme_frame:
            extreme_streak += 1
        else:
            extreme_streak = 0

        terminate_now = (
            strong_streak
            >= policy
            .strong_geometry_required_frames
            or extreme_streak
            >= policy
            .extreme_required_frames
        )

        if terminate_now:
            terminated = True
            output_state = (
                "SEARCHING"
            )

            reason = (
                "SUSTAINED_IDENTITY_"
                "CONTRADICTION_TERMINATION"
            )

            hysteresis_applied = False

            frames[
                frame_index
            ].update(
                state=output_state,
                bbox_xyxy=None,
                bbox_source="NONE",
                selected_detection_id=None,
                tracking_confidence=0.0,
                identity_confidence=0.0,
                decision_reason=(
                    reason
                ),
                review_required=True,
                hysteresis_status=(
                    "TERMINATED"
                ),
                hysteresis_review_required=True,
            )

        else:
            if (
                frame_index
                == confirmation
            ):
                output_state = (
                    "REACQUIRED"
                )
            else:
                output_state = (
                    "ACTIVE"
                )

            weak = (
                not soft_pass
            )

            identity = (
                identity_confidence(
                    target,
                    target_max,
                    negative_margin,
                    policy
                    .weak_identity_confidence_floor,
                )
            )

            tracking = max(
                (
                    policy
                    .weak_tracking_confidence_floor
                    if weak
                    else 0.0
                ),
                min(
                    1.0,
                    0.65
                    * float(
                        detection.confidence
                    )
                    + 0.35
                    * identity,
                ),
            )

            if (
                frame_index
                == confirmation
            ):
                reason = (
                    "PRESERVED_STRICT_"
                    "STAGE2D1_REACQUISITION"
                )

                status = (
                    "STRICT_REACQUISITION"
                )

                review = True
                hysteresis_applied = False

            elif weak:
                reason = (
                    "CONFIRMED_TRACKLET_"
                    "HYSTERESIS_WEAK_"
                    "APPEARANCE_BBOX_KEPT"
                )

                status = (
                    "WEAK_APPEARANCE_"
                    "BBOX_KEPT"
                )

                review = True
                hysteresis_applied = True

            else:
                reason = (
                    "CONFIRMED_TRACKLET_"
                    "APPEARANCE_AND_"
                    "GEOMETRY_STABLE"
                )

                status = (
                    "STABLE"
                )

                review = False
                hysteresis_applied = False

            frames[
                frame_index
            ].update(
                state=output_state,
                bbox_xyxy=bbox,
                bbox_source=(
                    "RFDETR_CONFIRMED_"
                    "TRACKLET_HYSTERESIS"
                    if weak
                    else
                    "RFDETR_CONFIRMED_"
                    "TRACKLET_VERIFIED"
                ),
                selected_detection_id=(
                    detection
                    .detection_id
                ),
                tracking_confidence=(
                    tracking
                ),
                identity_confidence=(
                    identity
                ),
                decision_reason=(
                    reason
                ),
                review_required=(
                    review
                ),
                hysteresis_status=(
                    status
                ),
                hysteresis_review_required=(
                    weak
                ),
            )

        audit.append(
            AuditRow(
                frame_index=(
                    frame_index
                ),
                episode_index=(
                    episode_index
                ),
                tracklet_id=(
                    tracklet_id
                ),
                original_state=(
                    original_state
                ),
                output_state=(
                    output_state
                ),
                detection_id=(
                    ""
                    if terminate_now
                    else str(
                        detection
                        .detection_id
                    )
                ),
                detection_present=True,
                temporal_iou=(
                    temporal_iou
                ),
                center_distance=(
                    center_distance
                ),
                geometry_continuous=(
                    geometry_ok
                ),
                target_similarity=(
                    target
                ),
                target_max_similarity=(
                    target_max
                ),
                negative_margin=(
                    negative_margin
                ),
                soft_appearance_pass=(
                    soft_pass
                ),
                strong_contradiction_frame=(
                    strong_frame
                ),
                extreme_contradiction_frame=(
                    extreme_frame
                ),
                strong_contradiction_streak=(
                    strong_streak
                ),
                extreme_contradiction_streak=(
                    extreme_streak
                ),
                hysteresis_applied=(
                    hysteresis_applied
                ),
                track_terminated=(
                    terminate_now
                ),
                decision_reason=(
                    reason
                ),
            )
        )

        if not terminate_now:
            previous_bbox = (
                bbox
            )

    return audit


def extended_observation_row(
    item: Mapping[
        str,
        Any,
    ],
    stage2d: Any,
) -> dict[
    str,
    Any,
]:
    row = (
        stage2d
        .observation_csv_row(
            item
        )
    )

    row[
        "hysteresis_status"
    ] = (
        item.get(
            "hysteresis_status"
        )
        or ""
    )

    row[
        "hysteresis_review_required"
    ] = bool(
        item.get(
            "hysteresis_review_required",
            False,
        )
    )

    return row


def audit_csv_row(
    item: AuditRow,
) -> dict[
    str,
    Any,
]:
    row = asdict(
        item
    )

    for key in (
        "temporal_iou",
        "center_distance",
        "target_similarity",
        "target_max_similarity",
        "negative_margin",
    ):
        value = row[
            key
        ]

        if value is None:
            row[
                key
            ] = ""
        else:
            row[
                key
            ] = (
                f"{float(value):.8f}"
            )

    return row


def build_report(
    summary: Mapping[
        str,
        Any,
    ],
) -> str:
    counts = summary[
        "counts"
    ]

    states = counts[
        "state_counts"
    ]

    return f"""# KickClip Target-Centric Tracking V1 — Stage 2-D2

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Bbox frames before / after: `{counts['bbox_frames_before']}` / `{counts['bbox_frames_after']}`
- Recovered flicker frames: `{counts['hysteresis_bbox_recovered_frames']}`
- Terminated episodes: `{counts['terminated_episode_count']}`

## State counts

- INITIALIZING / ACTIVE: `{states.get('INITIALIZING', 0)}` / `{states.get('ACTIVE', 0)}`
- OCCLUDED / SEARCHING: `{states.get('OCCLUDED', 0)}` / `{states.get('SEARCHING', 0)}`
- AMBIGUOUS / REACQUIRED: `{states.get('AMBIGUOUS', 0)}` / `{states.get('REACQUIRED', 0)}`

Strict Stage 2-D1 acquisition decisions were preserved. Short per-frame
appearance drops no longer remove a real bbox from the already-confirmed
geometric tracklet. Termination requires sustained contradiction.
"""


def main() -> int:
    arguments = parse_args()

    started = (
        time.perf_counter()
    )

    root = (
        arguments
        .project_root
        .expanduser()
        .resolve()
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
        "detections": (
            test_dir
            / "detections.csv"
        ),
        "stage2b_summary": (
            test_dir
            / "stage2b_reentry_summary.json"
        ),
        "stage2d1_summary": (
            test_dir
            / "stage2d1_summary.json"
        ),
        "stage2d1_episodes": (
            test_dir
            / "stage2d1_reentry_episodes.json"
        ),
        "stage2d1_timeline": (
            test_dir
            / "stage2d1_target_timeline.json"
        ),
        "stage2d1_gallery": (
            test_dir
            / "stage2d1_target_gallery.json"
        ),
    }

    for (
        name,
        path,
    ) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: "
                f"{path}"
            )

    stage1 = read_json(
        paths[
            "stage1"
        ]
    )

    stage2b_summary = read_json(
        paths[
            "stage2b_summary"
        ]
    )

    stage2d1_summary = read_json(
        paths[
            "stage2d1_summary"
        ]
    )

    episodes_json = read_json(
        paths[
            "stage2d1_episodes"
        ]
    )

    timeline_json = read_json(
        paths[
            "stage2d1_timeline"
        ]
    )

    gallery_json = read_json(
        paths[
            "stage2d1_gallery"
        ]
    )

    if (
        stage1.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 1 must be PASS"
        )

    if (
        stage2d1_summary.get(
            "decision"
        )
        != (
            "AUTHORIZE_MANDATORY_"
            "STAGE2D1_TEMPORAL_"
            "REENTRY_VISUAL_REVIEW"
        )
    ):
        raise RuntimeError(
            "Unexpected Stage 2-D1 decision"
        )

    prepare_outputs(
        test_dir,
        arguments
        .overwrite_stage2d2,
    )

    stage2_path = resolve(
        root,
        arguments.stage2_helper,
    )

    stage2b_path = resolve(
        root,
        arguments.stage2b_helper,
    )

    stage2d_path = resolve(
        root,
        arguments.stage2d_helper,
    )

    for path in (
        stage2_path,
        stage2b_path,
        stage2d_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    stage2 = load_module(
        "kickclip_stage2d2_stage2",
        stage2_path,
    )

    stage2b = load_module(
        "kickclip_stage2d2_stage2b",
        stage2b_path,
    )

    stage2d = load_module(
        "kickclip_stage2d2_stage2d",
        stage2d_path,
    )

    video_info = stage1[
        "video"
    ]

    video = Path(
        video_info[
            "path"
        ]
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
            "Video missing or changed"
        )

    frame_count = int(
        video_info[
            "processed_frames"
        ]
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

    (
        by_frame,
        detection_count,
    ) = stage2.load_detections(
        paths[
            "detections"
        ],
        frame_count,
        width,
        height,
    )

    if (
        detection_count
        != int(
            stage1[
                "counts"
            ][
                "total_detections"
            ]
        )
    ):
        raise RuntimeError(
            "Detection count mismatch"
        )

    frames = (
        copy_and_validate_frames(
            timeline_json,
            frame_count,
            width,
            height,
        )
    )

    bbox_frames_before = sum(
        item.get(
            "bbox_xyxy"
        )
        is not None
        for item
        in frames
    )

    reviewed_end = int(
        stage2b_summary[
            "selection"
        ][
            "selected_candidate"
        ][
            "end_frame"
        ]
    )

    target_area = (
        statistics.median(
            stage2.area(
                item[
                    "bbox_xyxy"
                ]
            )
            for item
            in gallery_json[
                "target_gallery"
            ]
        )
    )

    tracklets = (
        stage2b
        .build_tracklets(
            by_frame,
            reviewed_end
            + 1,
            frame_count,
            target_area,
            stage2,
            stage2b.Policy(),
        )
    )

    tracklet_by_id = {
        tracklet.tracklet_id: (
            tracklet
        )
        for tracklet
        in tracklets
    }

    episodes = episodes_json.get(
        "episodes"
    )

    if not isinstance(
        episodes,
        list,
    ):
        raise RuntimeError(
            "Invalid Stage 2-D1 episodes"
        )

    selected = [
        episode
        for episode
        in episodes
        if (
            isinstance(
                episode,
                dict,
            )
            and int(
                episode.get(
                    "episode_index",
                    0,
                )
            )
            > 1
            and episode.get(
                "selected_tracklet_id"
            )
            and episode.get(
                "review_status"
            )
            == "PENDING_VISUAL_REVIEW"
        )
    ]

    if not selected:
        raise RuntimeError(
            "No Stage 2-D1 "
            "selected episode"
        )

    policy = (
        HysteresisPolicy()
    )

    audits: list[
        AuditRow
    ] = []

    for episode in selected:
        tracklet_id = str(
            episode[
                "selected_tracklet_id"
            ]
        )

        if (
            tracklet_id
            not in tracklet_by_id
        ):
            raise RuntimeError(
                "Cannot reconstruct "
                f"tracklet: {tracklet_id}"
            )

        print(
            "[STAGE2D2] "
            f"episode="
            f"{episode['episode_index']} "
            f"tracklet="
            f"{tracklet_id} "
            f"confirm="
            f"{episode['confirmation_frame']} "
            f"end="
            f"{episode['selected_tracklet_end_frame']}",
            flush=True,
        )

        audits.extend(
            apply_hysteresis(
                frames,
                episode,
                tracklet_by_id[
                    tracklet_id
                ],
                stage2,
                policy,
            )
        )

    for item in frames:
        state = str(
            item[
                "state"
            ]
        )

        bbox = item.get(
            "bbox_xyxy"
        )

        if (
            state
            in UNCERTAIN_STATES
            and bbox is not None
        ):
            raise RuntimeError(
                "Uncertain state "
                "contains bbox at "
                f"{item['frame_index']}"
            )

        if (
            state
            in CONFIRMED_STATES
            and bbox is None
        ):
            raise RuntimeError(
                "Confirmed state "
                "lacks bbox at "
                f"{item['frame_index']}"
            )

    audit_rows = [
        audit_csv_row(
            item
        )
        for item
        in audits
    ]

    audit_path = (
        test_dir
        / "stage2d2_hysteresis_audit.csv"
    )

    write_csv(
        audit_path,
        audit_rows,
        list(
            audit_rows[
                0
            ]
        ),
    )

    observation_rows = [
        extended_observation_row(
            item,
            stage2d,
        )
        for item
        in frames
    ]

    observation_path = (
        test_dir
        / "stage2d2_frame_observations.csv"
    )

    write_csv(
        observation_path,
        observation_rows,
        list(
            observation_rows[
                0
            ]
        ),
    )

    timeline_path = (
        test_dir
        / "stage2d2_target_timeline.json"
    )

    atomic_json(
        timeline_path,
        {
            "schema_version": (
                "kickclip.target_centric."
                "stage2d2_target_timeline.v1"
            ),
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
            "status": (
                "CONFIRMED_TRACK_"
                "HYSTERESIS_APPLIED"
            ),
            "target_id": (
                "target_001"
            ),
            "test_name": (
                test_name
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
                "width": (
                    width
                ),
                "height": (
                    height
                ),
                "fps": (
                    fps
                ),
                "frame_count": (
                    frame_count
                ),
            },
            "policy": {
                **asdict(
                    policy
                ),
                "strict_acquisition_source": (
                    "STAGE2D1_UNCHANGED"
                ),
                "new_detector_inference": (
                    False
                ),
                "new_reid_inference": (
                    False
                ),
                "new_memory_update": (
                    False
                ),
                "threshold_search": (
                    False
                ),
            },
            "episodes": (
                episodes
            ),
            "segments": (
                stage2d
                .build_segments(
                    frames,
                    fps,
                )
            ),
            "frames": (
                frames
            ),
        },
    )

    preview_path = (
        test_dir
        / "stage2d2_hysteresis_preview.mp4"
    )

    if not arguments.no_preview:
        stage2d.render_preview(
            video,
            preview_path,
            frames,
            fps,
            width,
            height,
            arguments.print_every,
        )

    states = Counter(
        item[
            "state"
        ]
        for item
        in frames
    )

    bbox_frames_after = sum(
        item.get(
            "bbox_xyxy"
        )
        is not None
        for item
        in frames
    )

    recovered = sum(
        item
        .hysteresis_applied
        for item
        in audits
    )

    missing = sum(
        not item
        .detection_present
        for item
        in audits
    )

    strong = sum(
        item
        .strong_contradiction_frame
        for item
        in audits
    )

    extreme = sum(
        item
        .extreme_contradiction_frame
        for item
        in audits
    )

    terminated = len(
        {
            item
            .episode_index
            for item
            in audits
            if item
            .track_terminated
        }
    )

    if terminated == 0:
        decision = (
            "AUTHORIZE_MANDATORY_"
            "STAGE2D2_HYSTERESIS_"
            "VISUAL_REVIEW"
        )

    else:
        decision = (
            "BLOCK_STAGE2D2_"
            "SUSTAINED_CONTRADICTION_"
            "DETECTED"
        )

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "PASS",
        "decision": (
            decision
        ),
        "test_name": (
            test_name
        ),
        "counts": {
            "frame_count": (
                frame_count
            ),
            "state_counts": dict(
                states
            ),
            "hysteresis_episode_count": (
                len(
                    selected
                )
            ),
            "hysteresis_audit_frame_count": (
                len(
                    audits
                )
            ),
            "bbox_frames_before": (
                bbox_frames_before
            ),
            "bbox_frames_after": (
                bbox_frames_after
            ),
            "hysteresis_bbox_recovered_frames": (
                recovered
            ),
            "missing_detection_frames": (
                missing
            ),
            "strong_contradiction_frames": (
                strong
            ),
            "extreme_contradiction_frames": (
                extreme
            ),
            "terminated_episode_count": (
                terminated
            ),
        },
        "policy": asdict(
            policy
        ),
        "selected_episodes": (
            selected
        ),
        "safety_invariants": {
            "strict_stage2d1_acquisition_preserved": (
                True
            ),
            "single_frame_appearance_failure_terminates_track": (
                False
            ),
            "sustained_contradiction_required": (
                True
            ),
            "missing_detection_bbox_fabrication": (
                False
            ),
            "new_detector_inference": (
                False
            ),
            "new_reid_inference": (
                False
            ),
            "new_memory_update": (
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
            "uncertain_states_have_null_bbox": (
                True
            ),
        },
        "outputs": {
            "audit": str(
                audit_path
            ),
            "observations": str(
                observation_path
            ),
            "timeline": str(
                timeline_path
            ),
            "preview": (
                None
                if arguments.no_preview
                else str(
                    preview_path
                )
            ),
        },
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
        "training": (
            False
        ),
        "detector_inference": (
            False
        ),
        "reid_inference": (
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
    }

    summary_path = (
        test_dir
        / "stage2d2_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        test_dir
        / "stage2d2_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V1 Stage 2-D2 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        f"Decision                       : "
        f"{decision}"
    )

    print(
        f"Hysteresis episodes            : "
        f"{len(selected)}"
    )

    print(
        "Bbox frames before / after     : "
        f"{bbox_frames_before} / "
        f"{bbox_frames_after}"
    )

    print(
        f"Recovered flicker frames       : "
        f"{recovered}"
    )

    print(
        f"Missing tracklet detections    : "
        f"{missing}"
    )

    print(
        "Strong/extreme contradictions : "
        f"{strong} / "
        f"{extreme}"
    )

    print(
        f"Terminated episodes            : "
        f"{terminated}"
    )

    print(
        f"States                         : "
        f"{dict(states)}"
    )

    print(
        "Detector/ReID inference        : "
        "NONE/NONE"
    )

    print(
        "Training/linking/V7            : "
        "NONE/NONE/NONE"
    )

    print(
        f"Output                         : "
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
            "Stage 2-D2 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 2-D2 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )