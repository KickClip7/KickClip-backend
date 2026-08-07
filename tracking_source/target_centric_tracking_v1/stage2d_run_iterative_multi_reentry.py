#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 2-D iterative multi-reentry.

Preserves the visually reviewed first Stage 2-B reacquisition, then repeatedly
searches every later same-shot re-entry episode until the video ends.

Safety contract:
- Stage 2-B thresholds are reused without relaxation or search.
- Appearance alone can never accept a candidate.
- Only previously visually reviewed observations form target memory.
- Newly automatic episodes do not update memory in this run.
- SEARCHING/AMBIGUOUS/LOST/ABSENT frames never receive a target bbox.
- Every new REACQUIRED episode requires visual review.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

STAGE = "stage2d_iterative_same_shot_multi_reentry"
VERSION = "target-centric-v1-stage2d-1.0.1"
EMBEDDING_DIM = 512

OUTPUT_NAMES = (
    "stage2d_target_gallery.json",
    "stage2d_target_embeddings.npy",
    "stage2d_negative_embeddings.npy",
    "stage2d_candidate_embeddings.npy",
    "stage2d_reentry_episodes.json",
    "stage2d_reentry_candidates.csv",
    "stage2d_frame_observations.csv",
    "stage2d_target_timeline.json",
    "stage2d_multi_reentry_preview.mp4",
    "stage2d_summary.json",
    "stage2d_report.md",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative same-shot multi-reentry after reviewed Stage 2-B."
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
        "--v6-reid-helper",
        type=Path,
        default=Path(
            "global_ID_tracking_upgrade_v6/"
            "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--deep-eiou-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cuda",
            "cpu",
        ),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-new-episodes",
        type=int,
        default=20,
        help=(
            "Safety cap only. Normal execution stops at video end."
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-stage2d",
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
            character
            not in allowed
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
                8
                * 1024
                * 1024
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


def atomic_npy(
    path: Path,
    value: np.ndarray,
) -> None:
    temporary = path.with_name(
        path.name
        + ".tmp.npy"
    )

    np.save(
        temporary,
        value,
        allow_pickle=False,
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
            "Stage 2-D outputs already exist. "
            "Use --overwrite-stage2d:\n"
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
        "*.stage2d.tmp*"
    ):
        if path.is_file():
            path.unlink()


def uniform_sample(
    items: Sequence[Any],
    count: int,
) -> list[Any]:
    if len(items) <= count:
        return list(
            items
        )

    indices = sorted(
        set(
            int(
                round(
                    value
                )
            )
            for value
            in np.linspace(
                0,
                len(items)
                - 1,
                count,
            )
        )
    )

    return [
        items[index]
        for index
        in indices
    ]


def normalize_seed_frames(
    timeline: Mapping[
        str,
        Any,
    ],
    frame_count: int,
    width: int,
    height: int,
    reviewed_search_start: int,
    reviewed_end: int,
    reviewed_tracklet_id: str,
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
        actual: Any = (
            len(
                raw_frames
            )
            if isinstance(
                raw_frames,
                list,
            )
            else "INVALID"
        )

        raise ValueError(
            "Stage 2-B timeline "
            "frame mismatch: "
            f"{actual}/"
            f"{frame_count}"
        )

    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for (
        expected_frame,
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
            != expected_frame
        ):
            raise ValueError(
                "Invalid Stage 2-B "
                f"frame at "
                f"{expected_frame}"
            )

        state = str(
            raw.get(
                "state",
                "",
            )
        )

        raw_bbox = raw.get(
            "bbox_xyxy"
        )

        if raw_bbox is None:
            bbox = None
        else:
            bbox = [
                float(
                    value
                )
                for value
                in raw_bbox
            ]

        if bbox is not None:
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
                    "Invalid bbox at frame "
                    f"{expected_frame}: "
                    f"{bbox}"
                )

        if (
            state
            in UNCERTAIN_STATES
            and bbox is not None
        ):
            raise ValueError(
                "Uncertain state "
                f"{state} "
                "contains bbox at frame "
                f"{expected_frame}"
            )

        in_reviewed_episode = (
            reviewed_search_start
            <= expected_frame
            <= reviewed_end
        )

        result.append(
            {
                "frame_index": (
                    expected_frame
                ),
                "time_ms": int(
                    raw.get(
                        "time_ms",
                        0,
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
                "review_required": (
                    False
                    if in_reviewed_episode
                    else bool(
                        raw.get(
                            "review_required",
                            False,
                        )
                    )
                ),
                "episode_index": (
                    1
                    if in_reviewed_episode
                    else None
                ),
                "episode_tracklet_id": (
                    reviewed_tracklet_id
                    if in_reviewed_episode
                    else None
                ),
                "episode_review_status": (
                    "REVIEWED_PASS"
                    if in_reviewed_episode
                    else None
                ),
            }
        )

    return result


def select_reviewed_gallery(
    frames: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    by_id: Mapping[
        str,
        Any,
    ],
    reviewed_confirmation: int,
    reviewed_end: int,
    gallery_size: int,
) -> list[Any]:
    before: list[Any] = []
    after: list[Any] = []

    for item in frames:
        detection_id = item.get(
            "selected_detection_id"
        )

        if (
            not detection_id
            or detection_id
            not in by_id
        ):
            continue

        frame = int(
            item[
                "frame_index"
            ]
        )

        state = str(
            item[
                "state"
            ]
        )

        tracking = float(
            item[
                "tracking_confidence"
            ]
        )

        identity = float(
            item[
                "identity_confidence"
            ]
        )

        if state == "INITIALIZING":
            before.append(
                by_id[
                    detection_id
                ]
            )

            continue

        if state not in {
            "ACTIVE",
            "REACQUIRED",
        }:
            continue

        if (
            tracking
            < 0.75
            or identity
            < 0.78
        ):
            continue

        if (
            frame
            < reviewed_confirmation
        ):
            before.append(
                by_id[
                    detection_id
                ]
            )

        elif (
            reviewed_confirmation
            <= frame
            <= reviewed_end
        ):
            after.append(
                by_id[
                    detection_id
                ]
            )

    first_count = max(
        1,
        gallery_size
        // 2,
    )

    selected = (
        uniform_sample(
            before,
            first_count,
        )
    )

    selected += uniform_sample(
        after,
        max(
            1,
            gallery_size
            - len(
                selected
            ),
        ),
    )

    result: list[Any] = []
    used: set[str] = set()

    for detection in selected:
        if (
            detection.detection_id
            in used
        ):
            continue

        result.append(
            detection
        )

        used.add(
            detection.detection_id
        )

    if not result:
        raise RuntimeError(
            "No reviewed target "
            "gallery observations"
        )

    return result


def candidate_row(
    episode_index: int,
    candidate: Any,
) -> dict[
    str,
    Any,
]:
    return {
        "episode_index": (
            episode_index
        ),
        "candidate_uid": (
            f"episode_"
            f"{episode_index:03d}:"
            f"{candidate.tracklet_id}"
        ),
        "tracklet_id": (
            candidate.tracklet_id
        ),
        "rank": (
            candidate.rank
        ),
        "selected": (
            candidate.selected
        ),
        "hard_gate_pass": (
            candidate.hard_gate_pass
        ),
        "start_frame": (
            candidate.start_frame
        ),
        "confirmation_frame": (
            candidate.confirmation_frame
        ),
        "end_frame": (
            candidate.end_frame
        ),
        "detection_count": (
            candidate.detection_count
        ),
        "confirmation_span": (
            candidate.confirmation_span
        ),
        "entry_edge_distance": (
            f"{candidate.entry_edge_distance:.8f}"
        ),
        "entry_score": (
            f"{candidate.entry_score:.8f}"
        ),
        "mean_detection_confidence": (
            f"{candidate.mean_detection_confidence:.8f}"
        ),
        "mean_temporal_iou": (
            f"{candidate.mean_temporal_iou:.8f}"
        ),
        "target_similarity": (
            f"{candidate.target_similarity:.8f}"
        ),
        "target_max_similarity": (
            f"{candidate.target_max_similarity:.8f}"
        ),
        "negative_max_similarity": (
            f"{candidate.negative_max_similarity:.8f}"
        ),
        "negative_margin": (
            f"{candidate.negative_margin:.8f}"
        ),
        "crop_consistency": (
            f"{candidate.crop_consistency:.8f}"
        ),
        "combined_score": (
            f"{candidate.combined_score:.8f}"
        ),
        "sample_detection_ids": (
            "|".join(
                candidate.sample_detection_ids
            )
        ),
        "rejection_reasons": (
            "|".join(
                candidate.rejection_reasons
            )
        ),
        "raw_candidate_json": (
            json.dumps(
                asdict(
                    candidate
                ),
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        ),
    }


def mark_searching(
    frames: list[
        dict[
            str,
            Any,
        ]
    ],
    start: int,
    end_exclusive: int,
    episode_index: int,
    reason: str,
    candidate_margin: Optional[float],
    plausible: Optional[Any],
) -> None:
    for frame in range(
        start,
        end_exclusive,
    ):
        ambiguous = (
            plausible is not None
            and plausible.start_frame
            <= frame
            <= plausible.end_frame
            and plausible.target_similarity
            >= 0.64
            and plausible.combined_score
            >= 0.66
        )

        frames[
            frame
        ] = {
            "frame_index": frame,
            "time_ms": (
                frames[
                    frame
                ][
                    "time_ms"
                ]
            ),
            "state": (
                "AMBIGUOUS"
                if ambiguous
                else "SEARCHING"
            ),
            "bbox_xyxy": None,
            "bbox_source": "NONE",
            "selected_detection_id": None,
            "tracking_confidence": 0.0,
            "identity_confidence": 0.0,
            "decision_reason": reason,
            "target_similarity": (
                plausible
                .target_similarity
                if ambiguous
                else None
            ),
            "target_max_similarity": (
                plausible
                .target_max_similarity
                if ambiguous
                else None
            ),
            "negative_max_similarity": (
                plausible
                .negative_max_similarity
                if ambiguous
                else None
            ),
            "negative_margin": (
                plausible
                .negative_margin
                if ambiguous
                else None
            ),
            "candidate_margin": (
                candidate_margin
            ),
            "review_required": (
                ambiguous
            ),
            "episode_index": (
                episode_index
            ),
            "episode_tracklet_id": (
                plausible
                .tracklet_id
                if plausible
                else None
            ),
            "episode_review_status": (
                "PENDING"
            ),
        }


def apply_verified_tracklet(
    frames: list[
        dict[
            str,
            Any,
        ]
    ],
    episode_index: int,
    selected: Any,
    verification: Mapping[
        int,
        Mapping[
            str,
            Any,
        ],
    ],
    candidate_margin: Optional[float],
    continuation_max_gap: int,
) -> tuple[
    int,
    int,
]:
    first_verified: Optional[
        int
    ] = None

    last_verified: Optional[
        int
    ] = None

    unverified_gap = 0

    for frame in range(
        selected.confirmation_frame,
        selected.end_frame
        + 1,
    ):
        verified = (
            verification.get(
                frame
            )
        )

        if (
            verified is not None
            and bool(
                verified[
                    "verified"
                ]
            )
        ):
            detection = (
                verified[
                    "detection"
                ]
            )

            unverified_gap = 0

            if first_verified is None:
                first_verified = (
                    frame
                )

            last_verified = (
                frame
            )

            target_similarity = float(
                verified[
                    "target_similarity"
                ]
            )

            target_max = float(
                verified[
                    "target_max_similarity"
                ]
            )

            negative_max = float(
                verified[
                    "negative_max_similarity"
                ]
            )

            negative_margin = float(
                verified[
                    "negative_margin"
                ]
            )

            identity = max(
                0.0,
                min(
                    1.0,
                    0.58
                    * target_similarity
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
                    ),
                ),
            )

            state = (
                "REACQUIRED"
                if frame
                == first_verified
                else "ACTIVE"
            )

            frames[
                frame
            ] = {
                "frame_index": (
                    frame
                ),
                "time_ms": (
                    frames[
                        frame
                    ][
                        "time_ms"
                    ]
                ),
                "state": state,
                "bbox_xyxy": [
                    float(
                        value
                    )
                    for value
                    in detection.bbox
                ],
                "bbox_source": (
                    "RFDETR_ITERATIVE_"
                    "REENTRY_TRACKLET_"
                    "APPEARANCE_VERIFIED"
                ),
                "selected_detection_id": (
                    detection.detection_id
                ),
                "tracking_confidence": min(
                    1.0,
                    0.65
                    * float(
                        detection.confidence
                    )
                    + 0.35
                    * identity,
                ),
                "identity_confidence": (
                    identity
                ),
                "decision_reason": (
                    "ITERATIVE_MULTI_EVIDENCE_"
                    "REENTRY_CONFIRMED"
                    if state
                    == "REACQUIRED"
                    else
                    "ITERATIVE_POST_REENTRY_"
                    "FRAME_VERIFIED"
                ),
                "target_similarity": (
                    target_similarity
                ),
                "target_max_similarity": (
                    target_max
                ),
                "negative_max_similarity": (
                    negative_max
                ),
                "negative_margin": (
                    negative_margin
                ),
                "candidate_margin": (
                    candidate_margin
                ),
                "review_required": (
                    state
                    == "REACQUIRED"
                ),
                "episode_index": (
                    episode_index
                ),
                "episode_tracklet_id": (
                    selected.tracklet_id
                ),
                "episode_review_status": (
                    "PENDING_VISUAL_REVIEW"
                ),
            }

        else:
            unverified_gap += 1

            state = (
                "OCCLUDED"
                if unverified_gap
                <= continuation_max_gap
                else "SEARCHING"
            )

            frames[
                frame
            ] = {
                "frame_index": (
                    frame
                ),
                "time_ms": (
                    frames[
                        frame
                    ][
                        "time_ms"
                    ]
                ),
                "state": state,
                "bbox_xyxy": None,
                "bbox_source": "NONE",
                "selected_detection_id": None,
                "tracking_confidence": 0.0,
                "identity_confidence": 0.0,
                "decision_reason": (
                    "ITERATIVE_POST_REENTRY_"
                    "SHORT_UNVERIFIED_GAP"
                    if state
                    == "OCCLUDED"
                    else
                    "ITERATIVE_POST_REENTRY_"
                    "VERIFICATION_LOST"
                ),
                "target_similarity": (
                    float(
                        verified[
                            "target_similarity"
                        ]
                    )
                    if verified
                    else None
                ),
                "target_max_similarity": (
                    float(
                        verified[
                            "target_max_similarity"
                        ]
                    )
                    if verified
                    else None
                ),
                "negative_max_similarity": (
                    float(
                        verified[
                            "negative_max_similarity"
                        ]
                    )
                    if verified
                    else None
                ),
                "negative_margin": (
                    float(
                        verified[
                            "negative_margin"
                        ]
                    )
                    if verified
                    else None
                ),
                "candidate_margin": (
                    candidate_margin
                ),
                "review_required": (
                    verified
                    is not None
                ),
                "episode_index": (
                    episode_index
                ),
                "episode_tracklet_id": (
                    selected.tracklet_id
                ),
                "episode_review_status": (
                    "PENDING_VISUAL_REVIEW"
                ),
            }

    if (
        first_verified is None
        or last_verified is None
    ):
        raise RuntimeError(
            "Selected episode "
            "produced no verified frame"
        )

    return (
        first_verified,
        last_verified,
    )


def observation_csv_row(
    item: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    bbox = item[
        "bbox_xyxy"
    ]

    row: dict[
        str,
        Any,
    ] = {
        "frame_index": (
            item[
                "frame_index"
            ]
        ),
        "frame_1based": (
            int(
                item[
                    "frame_index"
                ]
            )
            + 1
        ),
        "time_ms": (
            item[
                "time_ms"
            ]
        ),
        "state": (
            item[
                "state"
            ]
        ),
        "bbox_source": (
            item[
                "bbox_source"
            ]
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
            item[
                "decision_reason"
            ]
            or ""
        ),
        "target_similarity": "",
        "target_max_similarity": "",
        "negative_max_similarity": "",
        "negative_margin": "",
        "candidate_margin": "",
        "review_required": bool(
            item[
                "review_required"
            ]
        ),
        "episode_index": (
            ""
            if item.get(
                "episode_index"
            )
            is None
            else item[
                "episode_index"
            ]
        ),
        "episode_tracklet_id": (
            item.get(
                "episode_tracklet_id"
            )
            or ""
        ),
        "episode_review_status": (
            item.get(
                "episode_review_status"
            )
            or ""
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
        if item.get(
            key
        ) is not None:
            row[
                key
            ] = (
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


def build_segments(
    frames: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    fps: float,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    result: list[
        dict[
            str,
            Any,
        ]
    ] = []

    start = 0

    for index in range(
        1,
        len(
            frames
        )
        + 1,
    ):
        same = (
            index
            < len(
                frames
            )
            and frames[
                index
            ][
                "state"
            ]
            == frames[
                start
            ][
                "state"
            ]
            and frames[
                index
            ].get(
                "episode_index"
            )
            == frames[
                start
            ].get(
                "episode_index"
            )
        )

        if same:
            continue

        part = frames[
            start:index
        ]

        result.append(
            {
                "segment_index": (
                    len(
                        result
                    )
                ),
                "state": (
                    part[
                        0
                    ][
                        "state"
                    ]
                ),
                "episode_index": (
                    part[
                        0
                    ].get(
                        "episode_index"
                    )
                ),
                "episode_tracklet_id": (
                    part[
                        0
                    ].get(
                        "episode_tracklet_id"
                    )
                ),
                "episode_review_status": (
                    part[
                        0
                    ].get(
                        "episode_review_status"
                    )
                ),
                "start_frame": (
                    part[
                        0
                    ][
                        "frame_index"
                    ]
                ),
                "end_frame_inclusive": (
                    part[
                        -1
                    ][
                        "frame_index"
                    ]
                ),
                "start_ms": (
                    part[
                        0
                    ][
                        "time_ms"
                    ]
                ),
                "end_ms_exclusive": int(
                    round(
                        index
                        * 1000
                        / fps
                    )
                ),
                "frame_count": (
                    len(
                        part
                    )
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

        start = (
            index
        )

    return result


def render_preview(
    video: Path,
    output: Path,
    frames: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    fps: float,
    width: int,
    height: int,
    print_every: int,
) -> None:
    temporary = output.with_name(
        output.stem
        + ".stage2d.tmp.mp4"
    )

    capture = cv2.VideoCapture(
        str(
            video
        )
    )

    writer = cv2.VideoWriter(
        str(
            temporary
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

    if (
        not capture.isOpened()
        or not writer.isOpened()
    ):
        capture.release()
        writer.release()

        raise RuntimeError(
            "Cannot create "
            "Stage 2-D preview"
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
            130,
            130,
            130,
        ),
    }

    frame_index = 0

    try:
        while True:
            (
                ok,
                image,
            ) = capture.read()

            if not ok:
                break

            if (
                frame_index
                >= len(
                    frames
                )
            ):
                raise RuntimeError(
                    "Video has more "
                    "frames than "
                    "Stage 2-D timeline"
                )

            item = frames[
                frame_index
            ]

            state = str(
                item[
                    "state"
                ]
            )

            color = colors.get(
                state,
                (
                    255,
                    255,
                    255,
                ),
            )

            bbox = item[
                "bbox_xyxy"
            ]

            if bbox is not None:
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = [
                    round(
                        float(
                            value
                        )
                    )
                    for value
                    in bbox
                ]

                cv2.rectangle(
                    image,
                    (
                        x1,
                        y1,
                    ),
                    (
                        x2,
                        y2,
                    ),
                    color,
                    3,
                )

                cv2.putText(
                    image,
                    f"TARGET {state}",
                    (
                        x1,
                        max(
                            20,
                            y1
                            - 5,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            episode = item.get(
                "episode_index"
            )

            episode_text = (
                "-"
                if episode is None
                else str(
                    episode
                )
            )

            cv2.rectangle(
                image,
                (
                    0,
                    0,
                ),
                (
                    width,
                    27,
                ),
                (
                    0,
                    0,
                    0,
                ),
                -1,
            )

            cv2.putText(
                image,
                (
                    "Stage 2-D"
                    f" | frame="
                    f"{frame_index}"
                    f" | state="
                    f"{state}"
                    f" | episode="
                    f"{episode_text}"
                    f" | "
                    f"{item['decision_reason']}"
                ),
                (
                    6,
                    18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

            if bool(
                item.get(
                    "review_required"
                )
            ):
                cv2.rectangle(
                    image,
                    (
                        0,
                        height
                        - 26,
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
                    image,
                    (
                        "NEW AUTOMATIC REENTRY "
                        "- VISUAL REVIEW REQUIRED"
                    ),
                    (
                        6,
                        height
                        - 8,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (
                        0,
                        215,
                        255,
                    ),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(
                image
            )

            frame_index += 1

            if (
                print_every > 0
                and frame_index
                % print_every
                == 0
            ):
                print(
                    "[STAGE2D PREVIEW] "
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
        != len(
            frames
        )
        or not temporary.is_file()
        or temporary.stat().st_size
        == 0
    ):
        raise RuntimeError(
            "Invalid Stage 2-D preview: "
            f"{frame_index}/"
            f"{len(frames)}"
        )

    os.replace(
        temporary,
        output,
    )


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

    return f"""# KickClip Target-Centric Tracking V1 — Stage 2-D Multi-Reentry

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Reviewed seed episode: `1`
- New automatic episodes: `{counts['new_selected_episode_count']}`
- Total selected episodes: `{counts['total_selected_episode_count']}`

## State counts

- INITIALIZING / ACTIVE: `{states.get('INITIALIZING', 0)}` / `{states.get('ACTIVE', 0)}`
- OCCLUDED / SEARCHING: `{states.get('OCCLUDED', 0)}` / `{states.get('SEARCHING', 0)}`
- AMBIGUOUS / REACQUIRED: `{states.get('AMBIGUOUS', 0)}` / `{states.get('REACQUIRED', 0)}`

## Safety contract

- Stage 2-B thresholds were reused without relaxation.
- Appearance-only acceptance was prohibited.
- Only visually reviewed observations formed target memory.
- New automatic episodes did not update target memory.
- SEARCHING, AMBIGUOUS and LOST frames contain no target bbox.
- Every new REACQUIRED episode requires visual review.

Open `stage2d_multi_reentry_preview.mp4` and inspect every `REACQUIRED` frame, especially the second re-entry near frame 408.
"""


def main() -> int:
    arguments = (
        parse_args()
    )

    started = (
        time.perf_counter()
    )

    script_path = Path(
        __file__
    ).resolve()

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
        "stage2b_timeline": (
            test_dir
            / "stage2b_target_timeline.json"
        ),
        "stage2c_audit": (
            test_dir
            / "stage2c_audit.json"
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

    stage2b_timeline = read_json(
        paths[
            "stage2b_timeline"
        ]
    )

    stage2c_audit = read_json(
        paths[
            "stage2c_audit"
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
            "Unexpected Stage 2-B "
            "status/decision"
        )

    if (
        stage2c_audit.get(
            "status"
        )
        != "PASS"
        or stage2c_audit.get(
            "visual_review",
            {},
        ).get(
            "result"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 2-C must record "
            "Stage 2-B visual review PASS"
        )

    prepare_outputs(
        test_dir,
        arguments
        .overwrite_stage2d,
    )

    stage2_helper_path = resolve(
        root,
        arguments.stage2_helper,
    )

    stage2b_helper_path = resolve(
        root,
        arguments.stage2b_helper,
    )

    reid_helper_path = resolve(
        root,
        arguments.v6_reid_helper,
    )

    for helper_path in (
        stage2_helper_path,
        stage2b_helper_path,
        reid_helper_path,
    ):
        if not helper_path.is_file():
            raise FileNotFoundError(
                helper_path
            )

    stage2_helper = load_module(
        "kickclip_stage2d_stage2_helper",
        stage2_helper_path,
    )

    stage2b_helper = load_module(
        "kickclip_stage2d_stage2b_helper",
        stage2b_helper_path,
    )

    reid_helper = load_module(
        "kickclip_stage2d_reid_helper",
        reid_helper_path,
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
    ) = stage2_helper.load_detections(
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

    by_id = {
        detection.detection_id: (
            detection
        )
        for detections
        in by_frame.values()
        for detection
        in detections
    }

    reviewed = (
        stage2b_summary[
            "selection"
        ][
            "selected_candidate"
        ]
    )

    reviewed_tracklet_id = str(
        stage2b_summary[
            "selection"
        ][
            "selected_tracklet_id"
        ]
    )

    reviewed_search_start = int(
        stage2b_summary[
            "first_lost_frame"
        ]
    )

    reviewed_confirmation = int(
        reviewed[
            "confirmation_frame"
        ]
    )

    reviewed_end = int(
        reviewed[
            "end_frame"
        ]
    )

    frames = normalize_seed_frames(
        stage2b_timeline,
        frame_count,
        width,
        height,
        reviewed_search_start,
        reviewed_end,
        reviewed_tracklet_id,
    )

    policy = (
        stage2b_helper
        .Policy()
    )

    gallery = select_reviewed_gallery(
        frames,
        by_id,
        reviewed_confirmation,
        reviewed_end,
        policy.gallery_size,
    )

    target_area = (
        statistics.median(
            stage2_helper.area(
                item.bbox
            )
            for item
            in gallery
        )
    )

    negative_pool = (
        stage2b_helper
        .select_negative_pool(
            gallery,
            by_frame,
            stage2_helper,
            policy,
        )
    )

    checkpoint = (
        stage2b_helper
        .discover_checkpoint(
            root,
            arguments.checkpoint,
        )
    )

    (
        deep_eiou_root,
        models,
        reid_root,
    ) = (
        stage2b_helper
        .discover_deep_eiou(
            root,
            reid_helper,
            arguments.deep_eiou_root,
        )
    )

    import torch

    if (
        arguments.device
        == "auto"
    ):
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device_name = (
            arguments.device
        )

    if (
        device_name
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested "
            "but unavailable"
        )

    device = torch.device(
        device_name
    )

    if hasattr(
        reid_helper,
        "configure_determinism",
    ):
        reid_helper.configure_determinism(
            torch
        )

    (
        model,
        model_contract,
    ) = reid_helper.build_model(
        torch,
        models,
        checkpoint,
        device,
    )

    transform = (
        stage2b_helper
        .build_transform()
    )

    base_detections = {
        item.detection_id: item
        for item
        in (
            gallery
            + negative_pool
        )
    }

    base_crops = (
        stage2b_helper
        .collect_crops(
            video,
            list(
                base_detections
                .values()
            ),
            frame_count,
            width,
            height,
        )
    )

    embedding_by_id = (
        stage2b_helper
        .embed_crops(
            base_crops,
            model,
            transform,
            torch,
            device,
            arguments.batch_size,
        )
    )

    target_embeddings = [
        embedding_by_id[
            item.detection_id
        ]
        for item
        in gallery
    ]

    target_prototype = (
        stage2b_helper
        .l2_mean(
            target_embeddings
        )
    )

    negative_ranked = sorted(
        (
            (
                float(
                    embedding_by_id[
                        item.detection_id
                    ]
                    @ target_prototype
                ),
                item,
            )
            for item
            in negative_pool
        ),
        reverse=True,
        key=lambda pair: (
            pair[
                0
            ]
        ),
    )[
        :
        policy
        .negative_gallery_size
    ]

    negatives = [
        item
        for (
            _,
            item,
        )
        in negative_ranked
    ]

    negative_embeddings = [
        embedding_by_id[
            item.detection_id
        ]
        for item
        in negatives
    ]

    target_embedding_path = (
        test_dir
        / "stage2d_target_embeddings.npy"
    )

    negative_embedding_path = (
        test_dir
        / "stage2d_negative_embeddings.npy"
    )

    atomic_npy(
        target_embedding_path,
        np.stack(
            target_embeddings
        ).astype(
            np.float32
        ),
    )

    if negative_embeddings:
        negative_matrix = (
            np.stack(
                negative_embeddings
            ).astype(
                np.float32
            )
        )
    else:
        negative_matrix = (
            np.empty(
                (
                    0,
                    EMBEDDING_DIM,
                ),
                dtype=np.float32,
            )
        )

    atomic_npy(
        negative_embedding_path,
        negative_matrix,
    )

    gallery_path = (
        test_dir
        / "stage2d_target_gallery.json"
    )

    atomic_json(
        gallery_path,
        {
            "stage": STAGE,
            "version": VERSION,
            "memory_policy": (
                "REVIEWED_OBSERVATIONS_ONLY; "
                "NEW_AUTOMATIC_EPISODES_"
                "DO_NOT_UPDATE_MEMORY"
            ),
            "reviewed_seed_episode": {
                "episode_index": 1,
                "tracklet_id": (
                    reviewed_tracklet_id
                ),
                "search_start_frame": (
                    reviewed_search_start
                ),
                "confirmation_frame": (
                    reviewed_confirmation
                ),
                "end_frame": (
                    reviewed_end
                ),
                "visual_review": "PASS",
            },
            "target_gallery": [
                {
                    "detection_id": (
                        item.detection_id
                    ),
                    "frame": (
                        item.frame
                    ),
                    "bbox_xyxy": list(
                        item.bbox
                    ),
                    "confidence": (
                        item.confidence
                    ),
                }
                for item
                in gallery
            ],
            "negative_gallery": [
                {
                    "detection_id": (
                        item.detection_id
                    ),
                    "frame": (
                        item.frame
                    ),
                    "bbox_xyxy": list(
                        item.bbox
                    ),
                    "confidence": (
                        item.confidence
                    ),
                    "target_similarity": (
                        similarity
                    ),
                }
                for (
                    similarity,
                    item,
                )
                in negative_ranked
            ],
            "embedding_contract": {
                "architecture": (
                    "osnet_x1_0"
                ),
                "image_size_hw": [
                    256,
                    128,
                ],
                "dimension": (
                    EMBEDDING_DIM
                ),
                "channel_order": (
                    "RGB"
                ),
                "per_crop_l2": (
                    True
                ),
                "checkpoint": str(
                    checkpoint
                ),
                "checkpoint_sha256": (
                    sha256_file(
                        checkpoint
                    )
                ),
                "deep_eiou_root": str(
                    deep_eiou_root
                ),
                "reid_root": str(
                    reid_root
                ),
                "model_load_contract": (
                    model_contract
                ),
            },
        },
    )

    episodes: list[
        dict[
            str,
            Any,
        ]
    ] = [
        {
            "episode_index": 1,
            "source": (
                "PRESERVED_REVIEWED_STAGE2B"
            ),
            "review_status": (
                "PASS"
            ),
            "search_start_frame": (
                reviewed_search_start
            ),
            "candidate_start_frame": int(
                reviewed[
                    "start_frame"
                ]
            ),
            "confirmation_frame": (
                reviewed_confirmation
            ),
            "verified_end_frame": (
                reviewed_end
            ),
            "selected_tracklet_id": (
                reviewed_tracklet_id
            ),
            "selection_reason": (
                stage2b_summary[
                    "selection"
                ][
                    "reason"
                ]
            ),
            "candidate_margin": (
                stage2b_summary[
                    "selection"
                ][
                    "candidate_margin"
                ]
            ),
            "selected_candidate": (
                reviewed
            ),
        }
    ]

    candidate_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    candidate_embedding_blocks: list[
        np.ndarray
    ] = []

    search_cursor = (
        reviewed_end
        + 1
    )

    new_episode_count = 0

    while (
        search_cursor
        < frame_count
        and new_episode_count
        < arguments
        .max_new_episodes
    ):
        episode_index = (
            len(
                episodes
            )
            + 1
        )

        print(
            "[STAGE2D] "
            f"episode="
            f"{episode_index} "
            f"search_start="
            f"{search_cursor}",
            flush=True,
        )

        all_tracklets = (
            stage2b_helper
            .build_tracklets(
                by_frame,
                search_cursor,
                frame_count,
                target_area,
                stage2_helper,
                policy,
            )
        )

        edge_tracklets = [
            item
            for item
            in all_tracklets
            if (
                len(
                    item.detections
                )
                >= policy
                .track_min_detections
                and stage2b_helper
                .edge_distance(
                    item
                    .detections[
                        0
                    ]
                    .bbox,
                    width,
                    height,
                )
                <= policy
                .entry_edge_distance_max
            )
        ]

        if not edge_tracklets:
            mark_searching(
                frames,
                search_cursor,
                frame_count,
                episode_index,
                "NO_EDGE_REENTRY_TRACKLETS",
                None,
                None,
            )

            episodes.append(
                {
                    "episode_index": (
                        episode_index
                    ),
                    "source": (
                        "ITERATIVE_STAGE2D"
                    ),
                    "review_status": (
                        "NOT_APPLICABLE"
                    ),
                    "search_start_frame": (
                        search_cursor
                    ),
                    "selected_tracklet_id": None,
                    "selection_reason": (
                        "NO_EDGE_REENTRY_TRACKLETS"
                    ),
                    "candidate_margin": None,
                    "geometric_tracklet_count": (
                        len(
                            all_tracklets
                        )
                    ),
                    "edge_tracklet_count": 0,
                    "scored_candidate_count": 0,
                }
            )

            break

        sample_detections: dict[
            str,
            Any,
        ] = {}

        for tracklet in edge_tracklets:
            for detection in (
                stage2b_helper
                .sample_tracklet(
                    tracklet,
                    policy,
                )
            ):
                sample_detections[
                    detection
                    .detection_id
                ] = detection

        missing_samples = [
            detection
            for (
                detection_id,
                detection,
            )
            in sample_detections.items()
            if detection_id
            not in embedding_by_id
        ]

        if missing_samples:
            sample_crops = (
                stage2b_helper
                .collect_crops(
                    video,
                    missing_samples,
                    frame_count,
                    width,
                    height,
                )
            )

            embedding_by_id.update(
                stage2b_helper
                .embed_crops(
                    sample_crops,
                    model,
                    transform,
                    torch,
                    device,
                    arguments.batch_size,
                )
            )

        (
            candidates,
            candidate_embeddings,
        ) = (
            stage2b_helper
            .score_candidates(
                edge_tracklets,
                embedding_by_id,
                target_embeddings,
                target_prototype,
                negative_embeddings,
                width,
                height,
                stage2_helper,
                policy,
            )
        )

        (
            selected,
            selection_reason,
            selection_margin,
        ) = (
            stage2b_helper
            .choose_candidate(
                candidates,
                policy,
            )
        )

        for candidate in candidates:
            candidate_rows.append(
                candidate_row(
                    episode_index,
                    candidate,
                )
            )

        if (
            len(
                candidate_embeddings
            )
            > 0
        ):
            candidate_embedding_blocks.append(
                candidate_embeddings
            )

        plausible = (
            candidates[
                0
            ]
            if candidates
            else None
        )

        if selected is None:
            mark_searching(
                frames,
                search_cursor,
                frame_count,
                episode_index,
                selection_reason,
                selection_margin,
                plausible,
            )

            episodes.append(
                {
                    "episode_index": (
                        episode_index
                    ),
                    "source": (
                        "ITERATIVE_STAGE2D"
                    ),
                    "review_status": (
                        "NO_AUTOMATIC_SELECTION"
                    ),
                    "search_start_frame": (
                        search_cursor
                    ),
                    "selected_tracklet_id": None,
                    "selection_reason": (
                        selection_reason
                    ),
                    "candidate_margin": (
                        selection_margin
                    ),
                    "geometric_tracklet_count": (
                        len(
                            all_tracklets
                        )
                    ),
                    "edge_tracklet_count": (
                        len(
                            edge_tracklets
                        )
                    ),
                    "scored_candidate_count": (
                        len(
                            candidates
                        )
                    ),
                    "hard_gate_pass_candidate_count": (
                        sum(
                            item
                            .hard_gate_pass
                            for item
                            in candidates
                        )
                    ),
                    "top_candidate": (
                        asdict(
                            plausible
                        )
                        if plausible
                        else None
                    ),
                }
            )

            break

        tracklet_by_id = {
            item.tracklet_id: (
                item
            )
            for item
            in edge_tracklets
        }

        selected_tracklet = (
            tracklet_by_id[
                selected.tracklet_id
            ]
        )

        missing_selected = [
            detection
            for detection
            in selected_tracklet
            .detections
            if (
                detection.frame
                >= selected
                .confirmation_frame
                and detection
                .detection_id
                not in embedding_by_id
            )
        ]

        if missing_selected:
            selected_crops = (
                stage2b_helper
                .collect_crops(
                    video,
                    missing_selected,
                    frame_count,
                    width,
                    height,
                )
            )

            embedding_by_id.update(
                stage2b_helper
                .embed_crops(
                    selected_crops,
                    model,
                    transform,
                    torch,
                    device,
                    arguments.batch_size,
                )
            )

        verification = (
            stage2b_helper
            .verify_tracklet_frames(
                selected_tracklet,
                selected
                .confirmation_frame,
                embedding_by_id,
                target_embeddings,
                target_prototype,
                negative_embeddings,
                policy,
            )
        )

        mark_searching(
            frames,
            search_cursor,
            selected
            .confirmation_frame,
            episode_index,
            "CANDIDATE_CONFIRMATION_PENDING",
            selection_margin,
            None,
        )

        (
            first_verified,
            last_verified,
        ) = apply_verified_tracklet(
            frames,
            episode_index,
            selected,
            verification,
            selection_margin,
            policy
            .continuation_max_unverified_gap,
        )

        episodes.append(
            {
                "episode_index": (
                    episode_index
                ),
                "source": (
                    "ITERATIVE_STAGE2D"
                ),
                "review_status": (
                    "PENDING_VISUAL_REVIEW"
                ),
                "search_start_frame": (
                    search_cursor
                ),
                "candidate_start_frame": (
                    selected.start_frame
                ),
                "confirmation_frame": (
                    first_verified
                ),
                "selected_tracklet_end_frame": (
                    selected.end_frame
                ),
                "verified_end_frame": (
                    last_verified
                ),
                "selected_tracklet_id": (
                    selected.tracklet_id
                ),
                "selection_reason": (
                    selection_reason
                ),
                "candidate_margin": (
                    selection_margin
                ),
                "geometric_tracklet_count": (
                    len(
                        all_tracklets
                    )
                ),
                "edge_tracklet_count": (
                    len(
                        edge_tracklets
                    )
                ),
                "scored_candidate_count": (
                    len(
                        candidates
                    )
                ),
                "hard_gate_pass_candidate_count": (
                    sum(
                        item
                        .hard_gate_pass
                        for item
                        in candidates
                    )
                ),
                "selected_candidate": (
                    asdict(
                        selected
                    )
                ),
                "memory_updated_from_episode": (
                    False
                ),
            }
        )

        new_episode_count += 1

        next_cursor = max(
            selected.end_frame
            + 1,
            last_verified
            + 1,
        )

        if (
            next_cursor
            <= search_cursor
        ):
            raise RuntimeError(
                "Search cursor did not advance: "
                f"{search_cursor} "
                f"-> {next_cursor}"
            )

        search_cursor = (
            next_cursor
        )

    if (
        search_cursor
        < frame_count
        and new_episode_count
        >= arguments
        .max_new_episodes
    ):
        mark_searching(
            frames,
            search_cursor,
            frame_count,
            len(
                episodes
            )
            + 1,
            "MAX_NEW_EPISODE_SAFETY_CAP_REACHED",
            None,
            None,
        )

    for item in frames:
        state = str(
            item[
                "state"
            ]
        )

        bbox = item[
            "bbox_xyxy"
        ]

        if (
            state
            in UNCERTAIN_STATES
            and bbox is not None
        ):
            raise RuntimeError(
                "Uncertain state "
                f"{state} "
                "contains bbox at frame "
                f"{item['frame_index']}"
            )

        if (
            state
            in CONFIRMED_STATES
            and bbox is None
        ):
            raise RuntimeError(
                "Confirmed state "
                f"{state} "
                "lacks bbox at frame "
                f"{item['frame_index']}"
            )

    if candidate_embedding_blocks:
        candidate_matrix = (
            np.concatenate(
                candidate_embedding_blocks,
                axis=0,
            ).astype(
                np.float32
            )
        )
    else:
        candidate_matrix = (
            np.empty(
                (
                    0,
                    EMBEDDING_DIM,
                ),
                dtype=np.float32,
            )
        )

    candidate_embedding_path = (
        test_dir
        / "stage2d_candidate_embeddings.npy"
    )

    atomic_npy(
        candidate_embedding_path,
        candidate_matrix,
    )

    episodes_path = (
        test_dir
        / "stage2d_reentry_episodes.json"
    )

    atomic_json(
        episodes_path,
        {
            "schema_version": (
                "kickclip.target_centric."
                "stage2d_reentry_episodes.v1"
            ),
            "stage": STAGE,
            "version": VERSION,
            "test_name": (
                test_name
            ),
            "memory_policy": (
                "REVIEWED_GALLERY_FROZEN; "
                "NEW_AUTOMATIC_EPISODES_"
                "DO_NOT_UPDATE_MEMORY"
            ),
            "episodes": episodes,
        },
    )

    candidate_fields = [
        "episode_index",
        "candidate_uid",
        "tracklet_id",
        "rank",
        "selected",
        "hard_gate_pass",
        "start_frame",
        "confirmation_frame",
        "end_frame",
        "detection_count",
        "confirmation_span",
        "entry_edge_distance",
        "entry_score",
        "mean_detection_confidence",
        "mean_temporal_iou",
        "target_similarity",
        "target_max_similarity",
        "negative_max_similarity",
        "negative_margin",
        "crop_consistency",
        "combined_score",
        "sample_detection_ids",
        "rejection_reasons",
        "raw_candidate_json",
    ]

    candidate_path = (
        test_dir
        / "stage2d_reentry_candidates.csv"
    )

    write_csv(
        candidate_path,
        candidate_rows,
        candidate_fields,
    )

    observation_rows = [
        observation_csv_row(
            item
        )
        for item
        in frames
    ]

    observation_path = (
        test_dir
        / "stage2d_frame_observations.csv"
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
        / "stage2d_target_timeline.json"
    )

    atomic_json(
        timeline_path,
        {
            "schema_version": (
                "kickclip.target_centric."
                "stage2d_target_timeline.v1"
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
                "MULTI_REENTRY_TIMELINE_GENERATED"
            ),
            "target_id": (
                "target_001"
            ),
            "test_name": (
                test_name
            ),
            "scope": (
                "single_camera_cut_free_shot"
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
                "width": (
                    width
                ),
                "height": (
                    height
                ),
                "fps": fps,
                "frame_count": (
                    frame_count
                ),
                "duration_seconds": (
                    frame_count
                    / fps
                ),
            },
            "policy": {
                **asdict(
                    policy
                ),
                "threshold_origin": (
                    "UNCHANGED_STAGE2B_"
                    "FROZEN_INITIAL_POLICY"
                ),
                "memory_update_from_"
                "new_automatic_episode": (
                    False
                ),
                "max_new_episode_"
                "safety_cap": (
                    arguments
                    .max_new_episodes
                ),
            },
            "episodes": (
                episodes
            ),
            "segments": (
                build_segments(
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
        / "stage2d_multi_reentry_preview.mp4"
    )

    if not arguments.no_preview:
        render_preview(
            video,
            preview_path,
            frames,
            fps,
            width,
            height,
            arguments.print_every,
        )

    state_counts = Counter(
        item[
            "state"
        ]
        for item
        in frames
    )

    new_selected = [
        item
        for item
        in episodes
        if (
            item[
                "episode_index"
            ]
            > 1
            and item.get(
                "selected_tracklet_id"
            )
        )
    ]

    pending_review = [
        item
        for item
        in new_selected
        if item.get(
            "review_status"
        )
        == "PENDING_VISUAL_REVIEW"
    ]

    if pending_review:
        decision = (
            "AUTHORIZE_MANDATORY_"
            "STAGE2D_MULTI_REENTRY_"
            "VISUAL_REVIEW"
        )
    else:
        decision = (
            "SAFE_NO_ADDITIONAL_"
            "AUTOMATIC_REENTRY_"
            "REQUIRE_REVIEW_OR_UI"
        )

    summary = {
        "schema_version": (
            "kickclip.target_centric."
            "stage2d_summary.v1"
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
                state_counts
            ),
            "reviewed_seed_episode_count": 1,
            "new_selected_episode_count": (
                len(
                    new_selected
                )
            ),
            "pending_visual_review_"
            "episode_count": (
                len(
                    pending_review
                )
            ),
            "total_selected_episode_count": (
                1
                + len(
                    new_selected
                )
            ),
            "episode_record_count": (
                len(
                    episodes
                )
            ),
            "candidate_row_count": (
                len(
                    candidate_rows
                )
            ),
            "hard_gate_pass_candidate_count": (
                sum(
                    bool(
                        row[
                            "hard_gate_pass"
                        ]
                    )
                    for row
                    in candidate_rows
                )
            ),
            "bbox_frame_count": (
                sum(
                    item[
                        "bbox_xyxy"
                    ]
                    is not None
                    for item
                    in frames
                )
            ),
        },
        "reviewed_seed_episode": (
            episodes[
                0
            ]
        ),
        "new_selected_episodes": (
            new_selected
        ),
        "policy": {
            **asdict(
                policy
            ),
            "threshold_origin": (
                "UNCHANGED_STAGE2B_"
                "FROZEN_INITIAL_POLICY"
            ),
            "memory_gallery_source": (
                "ONLY_STAGE2C_"
                "VISUALLY_REVIEWED_"
                "OBSERVATIONS"
            ),
            "new_automatic_memory_update": (
                False
            ),
        },
        "safety_invariants": {
            "appearance_only_acceptance": (
                False
            ),
            "threshold_relaxation": (
                False
            ),
            "threshold_search": (
                False
            ),
            "new_automatic_memory_update": (
                False
            ),
            "uncertain_states_have_null_bbox": (
                True
            ),
            "camera_cut_handling": (
                False
            ),
            "global_linking": (
                False
            ),
            "gta": False,
            "v7_logic": False,
            "new_episode_visual_review_required": (
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
        },
        "script": {
            "path": str(
                script_path
            ),
            "sha256": (
                sha256_file(
                    script_path
                )
            ),
        },
        "helpers": {
            "stage2_helper": {
                "path": str(
                    stage2_helper_path
                ),
                "sha256": (
                    sha256_file(
                        stage2_helper_path
                    )
                ),
            },
            "stage2b_helper": {
                "path": str(
                    stage2b_helper_path
                ),
                "sha256": (
                    sha256_file(
                        stage2b_helper_path
                    )
                ),
            },
            "v6_reid_helper": {
                "path": str(
                    reid_helper_path
                ),
                "sha256": (
                    sha256_file(
                        reid_helper_path
                    )
                ),
            },
        },
        "checkpoint": {
            "path": str(
                checkpoint
            ),
            "sha256": (
                sha256_file(
                    checkpoint
                )
            ),
        },
        "outputs": {
            "target_gallery": {
                "path": str(
                    gallery_path
                ),
                "sha256": (
                    sha256_file(
                        gallery_path
                    )
                ),
            },
            "target_embeddings": {
                "path": str(
                    target_embedding_path
                ),
                "sha256": (
                    sha256_file(
                        target_embedding_path
                    )
                ),
            },
            "negative_embeddings": {
                "path": str(
                    negative_embedding_path
                ),
                "sha256": (
                    sha256_file(
                        negative_embedding_path
                    )
                ),
            },
            "candidate_embeddings": {
                "path": str(
                    candidate_embedding_path
                ),
                "sha256": (
                    sha256_file(
                        candidate_embedding_path
                    )
                ),
            },
            "episodes": {
                "path": str(
                    episodes_path
                ),
                "sha256": (
                    sha256_file(
                        episodes_path
                    )
                ),
            },
            "candidates": {
                "path": str(
                    candidate_path
                ),
                "sha256": (
                    sha256_file(
                        candidate_path
                    )
                ),
            },
            "frame_observations": {
                "path": str(
                    observation_path
                ),
                "sha256": (
                    sha256_file(
                        observation_path
                    )
                ),
            },
            "timeline": {
                "path": str(
                    timeline_path
                ),
                "sha256": (
                    sha256_file(
                        timeline_path
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
        "device": str(
            device
        ),
        "training": False,
        "threshold_search": False,
        "global_linking": False,
        "v7_logic": False,
        "performance_certification": (
            "NEW_AUTOMATIC_EPISODES_"
            "REQUIRE_VISUAL_REVIEW"
        ),
    }

    summary_path = (
        test_dir
        / "stage2d_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        test_dir
        / "stage2d_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V1 Stage 2-D complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        f"Decision                       : "
        f"{decision}"
    )

    print(
        f"Frames                         : "
        f"{frame_count}"
    )

    print(
        "Reviewed seed episodes         : 1"
    )

    print(
        "New selected episodes          : "
        f"{len(new_selected)}"
    )

    print(
        "Pending visual-review episodes : "
        f"{len(pending_review)}"
    )

    print(
        f"States                         : "
        f"{dict(state_counts)}"
    )

    for episode in new_selected:
        candidate = episode[
            "selected_candidate"
        ]

        margin = episode.get(
            "candidate_margin"
        )

        margin_text = (
            "NA"
            if margin is None
            else f"{float(margin):.4f}"
        )

        print(
            f"Episode "
            f"{episode['episode_index']}"
            f"                    : "
            f"search="
            f"{episode['search_start_frame']}, "
            f"entry="
            f"{episode['candidate_start_frame']}, "
            f"confirm="
            f"{episode['confirmation_frame']}, "
            f"end="
            f"{episode['verified_end_frame']}, "
            f"target="
            f"{float(candidate['target_similarity']):.4f}, "
            f"negative_margin="
            f"{float(candidate['negative_margin']):.4f}, "
            f"candidate_margin="
            f"{margin_text}"
        )

    print(
        "Memory update from new episodes: NONE"
    )

    print(
        "Training/linking/V7           : "
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
            "Stage 2-D interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 2-D fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )