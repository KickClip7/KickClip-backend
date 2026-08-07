#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-A2.

Build a visually reviewable pre-cut target memory from the frozen Phase-1
within-shot tracker. The script runs frozen Stage 0/1/2 on the original video,
uses only frames before the confirmed camera cut, selects stable target crops,
and extracts frozen Sports OSNet embeddings.

No post-cut candidate selection or cross-shot linking is performed here.
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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


STAGE = "stage3a2_build_precut_target_memory"
VERSION = "target-centric-v2-stage3a2-1.0.0"
EMBEDDING_DIM = 512

CONFIRMED_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "REACQUIRED",
    "USER_CONFIRMED",
}

OUTPUT_NAMES = (
    "stage3a2_target_memory.json",
    "stage3a2_target_embeddings.npy",
    "stage3a2_negative_embeddings.npy",
    "stage3a2_memory_candidates.csv",
    "stage3a2_memory_contact_sheet.jpg",
    "stage3a2_precut_tracking_preview.mp4",
    "stage3a2_summary.json",
    "stage3a2_report.md",
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
        "--phase1-output-root",
        type=Path,
        default=Path(
            "runs/target_centric_tracking_v1"
        ),
    )

    parser.add_argument(
        "--phase1-source-test-name",
        default=None,
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
        "--stage0-script",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage0_audit_inputs.py"
        ),
    )

    parser.add_argument(
        "--stage1-script",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage1_generate_rfdetr_detections.py"
        ),
    )

    parser.add_argument(
        "--stage2-script",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage2_run_conservative_target_association.py"
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
        "--sports-osnet-checkpoint",
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
        "--memory-window-frames",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--minimum-frame-gap",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gallery-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--negative-gallery-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--skip-phase1-run",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-phase1-source",
        action="store_true",
    )

    return parser.parse_args()


def validate_test_name(value: str) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if (
        not value
        or value in {".", ".."}
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
        for name in OUTPUT_NAMES
        if (
            output_dir
            / name
        ).exists()
    ]

    crop_dir = (
        output_dir
        / "stage3a2_memory_crops"
    )

    if crop_dir.exists():
        existing.append(
            crop_dir
        )

    if (
        existing
        and not overwrite
    ):
        raise FileExistsError(
            "Stage 3-A2 outputs exist. "
            "Use --overwrite:\n"
            + "\n".join(
                str(path)
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


def iter_manifest_records(
    value: Any,
) -> Iterable[
    dict[
        str,
        Any,
    ]
]:
    if isinstance(
        value,
        list,
    ):
        for item in value:
            if isinstance(
                item,
                dict,
            ):
                yield item

    elif isinstance(
        value,
        dict,
    ):
        for key, item in value.items():
            if isinstance(
                item,
                dict,
            ):
                record = dict(
                    item
                )

                record.setdefault(
                    "name",
                    key,
                )

                yield record


def verify_phase1_manifest(
    root: Path,
    path: Path,
) -> dict[
    str,
    Any,
]:
    manifest = read_json(
        path
    )

    errors: list[
        str
    ] = []

    verified_scripts = 0

    for record in iter_manifest_records(
        manifest.get(
            "scripts"
        )
    ):
        raw_path = (
            record.get(
                "path"
            )
            or record.get(
                "relative_path"
            )
        )

        expected = (
            record.get(
                "sha256"
            )
            or record.get(
                "hash"
            )
        )

        if (
            not raw_path
            or not expected
        ):
            errors.append(
                "Invalid script record: "
                f"{record.get('name', 'UNKNOWN')}"
            )

            continue

        file_path = resolve(
            root,
            Path(
                str(
                    raw_path
                )
            ),
        )

        if not file_path.is_file():
            errors.append(
                f"Missing frozen script: {file_path}"
            )

            continue

        actual = sha256_file(
            file_path
        )

        if (
            actual.lower()
            != str(
                expected
            ).lower()
        ):
            errors.append(
                "Frozen script hash mismatch: "
                f"{file_path}"
            )

            continue

        verified_scripts += 1

    models = manifest.get(
        "models",
        {},
    )

    model_results: dict[
        str,
        Any,
    ] = {}

    for name in (
        "rfdetr",
        "sports_osnet",
    ):
        record = (
            models.get(
                name,
                {},
            )
            if isinstance(
                models,
                dict,
            )
            else {}
        )

        raw_path = (
            record.get(
                "path"
            )
            if isinstance(
                record,
                dict,
            )
            else None
        )

        expected = (
            (
                record.get(
                    "sha256"
                )
                or record.get(
                    "hash"
                )
            )
            if isinstance(
                record,
                dict,
            )
            else None
        )

        recorded_verified = (
            bool(
                record.get(
                    "verified",
                    False,
                )
            )
            if isinstance(
                record,
                dict,
            )
            else False
        )

        file_verified = False
        actual = None
        resolved_path = None

        if raw_path:
            resolved_path = resolve(
                root,
                Path(
                    str(
                        raw_path
                    )
                ),
            )

            if resolved_path.is_file():
                actual = sha256_file(
                    resolved_path
                )

                file_verified = (
                    bool(
                        expected
                    )
                    and actual.lower()
                    == str(
                        expected
                    ).lower()
                )

        model_results[
            name
        ] = {
            "path": (
                str(
                    resolved_path
                )
                if resolved_path
                else raw_path
            ),
            "expected_sha256": (
                expected
            ),
            "actual_sha256": (
                actual
            ),
            "manifest_verified": (
                recorded_verified
            ),
            "file_verified": (
                file_verified
            ),
        }

        if (
            not recorded_verified
            or not file_verified
        ):
            errors.append(
                "Frozen model verification "
                f"failed: {name}"
            )

    if verified_scripts != 11:
        errors.append(
            "Expected 11 verified scripts, "
            f"got {verified_scripts}"
        )

    return {
        "verified": (
            not errors
        ),
        "errors": (
            errors
        ),
        "verified_script_count": (
            verified_scripts
        ),
        "models": (
            model_results
        ),
        "manifest_path": str(
            path
        ),
        "manifest_sha256": (
            sha256_file(
                path
            )
        ),
    }


def run_command(
    command: Sequence[
        str
    ],
    cwd: Path,
) -> None:
    print(
        "[STAGE3A2 RUN] "
        + subprocess.list2cmdline(
            list(
                command
            )
        ),
        flush=True,
    )

    result = subprocess.run(
        list(
            command
        ),
        cwd=str(
            cwd
        ),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen Phase-1 command failed "
            f"with exit code {result.returncode}: "
            + subprocess.list2cmdline(
                list(
                    command
                )
            )
        )


def run_phase1_source(
    root: Path,
    source_dir: Path,
    source_test_name: str,
    video: Path,
    bbox: Sequence[
        int
    ],
    stage0_script: Path,
    stage1_script: Path,
    stage2_script: Path,
    device: str,
    overwrite_source: bool,
) -> None:
    if (
        source_dir.exists()
        and any(
            source_dir.iterdir()
        )
    ):
        if not overwrite_source:
            raise FileExistsError(
                "Phase-1 source output already "
                f"exists: {source_dir}. "
                "Use --skip-phase1-run to reuse it "
                "or --overwrite-phase1-source to rebuild."
            )

        shutil.rmtree(
            source_dir
        )

    command0 = [
        sys.executable,
        str(
            stage0_script
        ),
        "--video",
        str(
            video
        ),
        "--test-name",
        source_test_name,
        "--initial-bbox",
        *(
            str(
                value
            )
            for value
            in bbox
        ),
        "--bbox-format",
        "xyxy_pixels",
    ]

    run_command(
        command0,
        root,
    )

    command1 = [
        sys.executable,
        str(
            stage1_script
        ),
        "--test-name",
        source_test_name,
        "--device",
        device,
    ]

    run_command(
        command1,
        root,
    )

    command2 = [
        sys.executable,
        str(
            stage2_script
        ),
        "--test-name",
        source_test_name,
    ]

    run_command(
        command2,
        root,
    )


def find_stage2_timeline(
    source_dir: Path,
    minimum_frames: int,
) -> Path:
    preferred = (
        "stage2_target_timeline.json",
        "target_timeline.json",
        "stage2_timeline.json",
    )

    for name in preferred:
        path = (
            source_dir
            / name
        )

        if path.is_file():
            value = read_json(
                path
            )

            frames = value.get(
                "frames"
            )

            if (
                isinstance(
                    frames,
                    list,
                )
                and len(
                    frames
                )
                >= minimum_frames
            ):
                return path

    candidates: list[
        Path
    ] = []

    for path in sorted(
        source_dir.glob(
            "*.json"
        )
    ):
        try:
            value = read_json(
                path
            )
        except Exception:
            continue

        frames = value.get(
            "frames"
        )

        if (
            not isinstance(
                frames,
                list,
            )
            or len(
                frames
            )
            < minimum_frames
        ):
            continue

        sample = (
            frames[
                0
            ]
            if frames
            else None
        )

        if (
            isinstance(
                sample,
                dict,
            )
            and "frame_index"
            in sample
            and "bbox_xyxy"
            in sample
        ):
            candidates.append(
                path
            )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "the Stage-2 timeline. Candidates: "
            + ", ".join(
                str(
                    path
                )
                for path
                in candidates
            )
        )

    return candidates[
        0
    ]


def find_detections_csv(
    source_dir: Path,
) -> Path:
    preferred = (
        source_dir
        / "detections.csv"
    )

    if preferred.is_file():
        return preferred

    candidates = sorted(
        source_dir.glob(
            "*detection*.csv"
        )
    )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "detections CSV: "
            + ", ".join(
                str(
                    path
                )
                for path
                in candidates
            )
        )

    return candidates[
        0
    ]


def optional_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(
        result
    ):
        return default

    return result


def match_detection(
    frame_item: Mapping[
        str,
        Any,
    ],
    detections: Sequence[
        Any
    ],
    by_id: Mapping[
        str,
        Any,
    ],
    stage2: Any,
) -> Optional[
    Any
]:
    detection_id = (
        frame_item.get(
            "selected_detection_id"
        )
    )

    if (
        detection_id
        and str(
            detection_id
        )
        in by_id
    ):
        return by_id[
            str(
                detection_id
            )
        ]

    bbox = frame_item.get(
        "bbox_xyxy"
    )

    if (
        bbox is None
        or not detections
    ):
        return None

    ranked = sorted(
        (
            (
                float(
                    stage2.iou(
                        bbox,
                        detection.bbox,
                    )
                ),
                detection,
            )
            for detection
            in detections
        ),
        key=lambda pair: (
            pair[
                0
            ]
        ),
        reverse=True,
    )

    if (
        not ranked
        or ranked[
            0
        ][
            0
        ]
        < 0.50
    ):
        return None

    return ranked[
        0
    ][
        1
    ]


def build_memory_candidates(
    frames: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    by_frame: Mapping[
        int,
        Sequence[
            Any
        ],
    ],
    by_id: Mapping[
        str,
        Any,
    ],
    cut_frame: int,
    window_frames: int,
    stage2: Any,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    start = max(
        0,
        cut_frame
        - window_frames,
    )

    raw: list[
        dict[
            str,
            Any,
        ]
    ] = []

    previous_bbox: Optional[
        Sequence[
            float
        ]
    ] = None

    for frame_index in range(
        start,
        cut_frame,
    ):
        item = frames[
            frame_index
        ]

        state = str(
            item.get(
                "state",
                "",
            )
        )

        bbox = item.get(
            "bbox_xyxy"
        )

        if (
            state
            not in CONFIRMED_STATES
            or bbox is None
        ):
            previous_bbox = None

            continue

        detection = match_detection(
            item,
            by_frame.get(
                frame_index,
                [],
            ),
            by_id,
            stage2,
        )

        if detection is None:
            previous_bbox = None

            continue

        area = float(
            stage2.area(
                detection.bbox
            )
        )

        tracking_confidence = (
            optional_float(
                item.get(
                    "tracking_confidence"
                ),
                0.0,
            )
        )

        if previous_bbox is None:
            temporal_iou = 1.0

        else:
            temporal_iou = float(
                stage2.iou(
                    previous_bbox,
                    detection.bbox,
                )
            )

        previous_bbox = (
            detection.bbox
        )

        raw.append(
            {
                "frame_index": (
                    frame_index
                ),
                "state": (
                    state
                ),
                "detection": (
                    detection
                ),
                "detection_id": str(
                    detection.detection_id
                ),
                "bbox_xyxy": [
                    float(
                        value
                    )
                    for value
                    in detection.bbox
                ],
                "detection_confidence": float(
                    detection.confidence
                ),
                "tracking_confidence": (
                    tracking_confidence
                ),
                "bbox_area": (
                    area
                ),
                "temporal_iou": (
                    temporal_iou
                ),
            }
        )

    if not raw:
        return []

    area_reference = max(
        1.0,
        float(
            np.percentile(
                [
                    item[
                        "bbox_area"
                    ]
                    for item
                    in raw
                ],
                90,
            )
        ),
    )

    denominator = max(
        1,
        cut_frame
        - start
        - 1,
    )

    for item in raw:
        area_score = min(
            1.0,
            item[
                "bbox_area"
            ]
            / area_reference,
        )

        recency = (
            item[
                "frame_index"
            ]
            - start
        ) / denominator

        item[
            "quality_score"
        ] = float(
            0.45
            * item[
                "detection_confidence"
            ]
            + 0.25
            * item[
                "tracking_confidence"
            ]
            + 0.15
            * area_score
            + 0.10
            * item[
                "temporal_iou"
            ]
            + 0.05
            * recency
        )

    return raw


def select_diverse_candidates(
    candidates: Sequence[
        dict[
            str,
            Any,
        ]
    ],
    gallery_size: int,
    minimum_gap: int,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    selected: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for gap in (
        minimum_gap,
        max(
            3,
            minimum_gap
            // 2,
        ),
        1,
    ):
        for item in sorted(
            candidates,
            key=lambda row: (
                row[
                    "quality_score"
                ],
                row[
                    "frame_index"
                ],
            ),
            reverse=True,
        ):
            if any(
                existing[
                    "detection_id"
                ]
                == item[
                    "detection_id"
                ]
                for existing
                in selected
            ):
                continue

            if any(
                abs(
                    existing[
                        "frame_index"
                    ]
                    - item[
                        "frame_index"
                    ]
                )
                < gap
                for existing
                in selected
            ):
                continue

            selected.append(
                item
            )

            if len(
                selected
            ) >= gallery_size:
                return sorted(
                    selected,
                    key=lambda row: (
                        row[
                            "frame_index"
                        ]
                    ),
                )

    return sorted(
        selected,
        key=lambda row: (
            row[
                "frame_index"
            ]
        ),
    )


def crop_detection(
    frame: np.ndarray,
    bbox: Sequence[
        float
    ],
    padding: float = 0.08,
) -> np.ndarray:
    height, width = (
        frame.shape[
            :2
        ]
    )

    x1, y1, x2, y2 = (
        float(
            value
        )
        for value
        in bbox
    )

    pad_x = (
        x2
        - x1
    ) * padding

    pad_y = (
        y2
        - y1
    ) * padding

    left = max(
        0,
        int(
            math.floor(
                x1
                - pad_x
            )
        ),
    )

    top = max(
        0,
        int(
            math.floor(
                y1
                - pad_y
            )
        ),
    )

    right = min(
        width,
        int(
            math.ceil(
                x2
                + pad_x
            )
        ),
    )

    bottom = min(
        height,
        int(
            math.ceil(
                y2
                + pad_y
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


def read_frames(
    video: Path,
    frame_indices: Iterable[
        int
    ],
) -> dict[
    int,
    np.ndarray,
]:
    wanted = sorted(
        set(
            int(
                value
            )
            for value
            in frame_indices
        )
    )

    result: dict[
        int,
        np.ndarray,
    ] = {}

    capture = cv2.VideoCapture(
        str(
            video
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video}"
        )

    for frame_index in wanted:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = capture.read()

        if (
            not ok
            or frame is None
        ):
            capture.release()

            raise RuntimeError(
                f"Cannot read frame {frame_index}"
            )

        result[
            frame_index
        ] = frame

    capture.release()

    return result


def fit_crop_tile(
    crop: np.ndarray,
    width: int = 240,
    height: int = 320,
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
        (
            height
            - 36
        )
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
        36
        + (
            (
                height
                - 36
            )
            - resized_height
        )
        // 2
    )

    canvas[
        offset_y:
        offset_y
        + resized_height,
        offset_x:
        offset_x
        + resized_width,
    ] = resized

    return canvas


def save_memory_crops_and_sheet(
    output_dir: Path,
    video: Path,
    selected: Sequence[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    crop_dir = (
        output_dir
        / "stage3a2_memory_crops"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    frames = read_frames(
        video,
        [
            item[
                "frame_index"
            ]
            for item
            in selected
        ],
    )

    tiles: list[
        np.ndarray
    ] = []

    records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for rank, item in enumerate(
        selected,
        start=1,
    ):
        crop = crop_detection(
            frames[
                item[
                    "frame_index"
                ]
            ],
            item[
                "bbox_xyxy"
            ],
        )

        filename = (
            f"target_memory_{rank:02d}_"
            f"frame_{item['frame_index']:06d}.jpg"
        )

        path = (
            crop_dir
            / filename
        )

        if not cv2.imwrite(
            str(
                path
            ),
            crop,
        ):
            raise RuntimeError(
                f"Cannot write crop: {path}"
            )

        tile = fit_crop_tile(
            crop
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
                35,
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
                f"#{rank} "
                f"f={item['frame_index']} "
                f"q={item['quality_score']:.3f}"
            ),
            (
                6,
                23,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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

        records.append(
            {
                "rank": (
                    rank
                ),
                "frame_index": (
                    item[
                        "frame_index"
                    ]
                ),
                "detection_id": (
                    item[
                        "detection_id"
                    ]
                ),
                "bbox_xyxy": (
                    item[
                        "bbox_xyxy"
                    ]
                ),
                "detection_confidence": (
                    item[
                        "detection_confidence"
                    ]
                ),
                "tracking_confidence": (
                    item[
                        "tracking_confidence"
                    ]
                ),
                "temporal_iou": (
                    item[
                        "temporal_iou"
                    ]
                ),
                "quality_score": (
                    item[
                        "quality_score"
                    ]
                ),
                "crop_path": str(
                    path
                ),
            }
        )

    columns = 4

    rows = math.ceil(
        len(
            tiles
        )
        / columns
    )

    blank = np.zeros_like(
        tiles[
            0
        ]
    )

    padded = (
        tiles
        + [
            blank
        ]
        * (
            rows
            * columns
            - len(
                tiles
            )
        )
    )

    sheet = np.vstack(
        [
            np.hstack(
                padded[
                    row
                    * columns:
                    (
                        row
                        + 1
                    )
                    * columns
                ]
            )
            for row in range(
                rows
            )
        ]
    )

    sheet_path = (
        output_dir
        / "stage3a2_memory_contact_sheet.jpg"
    )

    if not cv2.imwrite(
        str(
            sheet_path
        ),
        sheet,
    ):
        raise RuntimeError(
            "Cannot write contact sheet: "
            f"{sheet_path}"
        )

    return records


def render_precut_preview(
    video: Path,
    output_path: Path,
    frames: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    cut_frame: int,
    fps: float,
    width: int,
    height: int,
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

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            "Cannot open video writer: "
            f"{output_path}"
        )

    for frame_index in range(
        cut_frame
    ):
        ok, frame = capture.read()

        if (
            not ok
            or frame is None
        ):
            writer.release()
            capture.release()

            raise RuntimeError(
                "Cannot read preview frame "
                f"{frame_index}"
            )

        item = frames[
            frame_index
        ]

        bbox = item.get(
            "bbox_xyxy"
        )

        state = str(
            item.get(
                "state",
                "UNKNOWN",
            )
        )

        if bbox is not None:
            x1, y1, x2, y2 = (
                int(
                    round(
                        value
                    )
                )
                for value
                in bbox
            )

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
                (
                    0,
                    255,
                    255,
                ),
                2,
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
                34,
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
                "PRE-CUT TARGET MEMORY SOURCE "
                f"| frame={frame_index} "
                f"| state={state}"
            ),
            (
                8,
                23,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
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


def candidate_csv_row(
    item: Mapping[
        str,
        Any,
    ],
    selected_ids: set[
        str
    ],
) -> dict[
    str,
    Any,
]:
    return {
        "selected_for_memory": (
            item[
                "detection_id"
            ]
            in selected_ids
        ),
        "frame_index": (
            item[
                "frame_index"
            ]
        ),
        "state": (
            item[
                "state"
            ]
        ),
        "detection_id": (
            item[
                "detection_id"
            ]
        ),
        "bbox_xyxy": "|".join(
            f"{value:.3f}"
            for value
            in item[
                "bbox_xyxy"
            ]
        ),
        "detection_confidence": (
            f"{item['detection_confidence']:.8f}"
        ),
        "tracking_confidence": (
            f"{item['tracking_confidence']:.8f}"
        ),
        "bbox_area": (
            f"{item['bbox_area']:.3f}"
        ),
        "temporal_iou": (
            f"{item['temporal_iou']:.8f}"
        ),
        "quality_score": (
            f"{item['quality_score']:.8f}"
        ),
    }


def build_report(
    summary: Mapping[
        str,
        Any,
    ],
) -> str:
    counts = summary[
        "counts"
    ]

    return f"""# KickClip Target-Centric Tracking V2 — Stage 3-A2

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Confirmed cut frame: `{summary['confirmed_cut_frame']}`
- Frozen Phase-1 source test: `{summary['phase1_source']['test_name']}`
- Stable memory candidates: `{counts['stable_memory_candidate_count']}`
- Selected target crops: `{counts['selected_target_crop_count']}`
- Selected negative crops: `{counts['selected_negative_crop_count']}`

## Safety contract

Only frames before the manually confirmed camera cut were used. Target memory
was built from the frozen Phase-1 tracker and frozen Sports OSNet. The memory is
not authorized for post-cut linking until the contact sheet and pre-cut tracking
preview receive visual review PASS. No post-cut candidate was read or linked.
"""


def main() -> int:
    arguments = parse_args()

    started = time.perf_counter()

    root = (
        arguments.project_root
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

    stage3a0_path = (
        output_dir
        / "stage3a0_summary.json"
    )

    stage3a1_path = (
        output_dir
        / "stage3a1_summary.json"
    )

    for path in (
        stage3a0_path,
        stage3a1_path,
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

    if (
        stage3a1.get(
            "status"
        )
        != "PASS"
        or stage3a1.get(
            "decision"
        )
        != (
            "AUTHORIZE_STAGE3A2_"
            "PRE_CUT_TARGET_MEMORY_BUILD"
        )
    ):
        raise RuntimeError(
            "Stage 3-A1 must authorize Stage 3-A2"
        )

    prepare_outputs(
        output_dir,
        arguments.overwrite,
    )

    manifest_path = resolve(
        root,
        arguments.phase1_manifest,
    )

    manifest_audit = verify_phase1_manifest(
        root,
        manifest_path,
    )

    if not manifest_audit[
        "verified"
    ]:
        raise RuntimeError(
            "Frozen Phase-1 manifest "
            "verification failed: "
            + " | ".join(
                manifest_audit[
                    "errors"
                ]
            )
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
            "Stage 3-A1 video is missing or changed"
        )

    cut_frame = int(
        stage3a1[
            "confirmed_cut"
        ][
            "cut_frame"
        ]
    )

    initial_bbox = [
        int(
            round(
                value
            )
        )
        for value
        in stage3a0[
            "initial_target"
        ][
            "bbox_xyxy"
        ]
    ]

    source_test_name = (
        arguments.phase1_source_test_name
        or (
            f"{test_name}"
            "__stage3a2_precut"
        )
    )

    source_test_name = validate_test_name(
        source_test_name
    )

    source_dir = (
        resolve(
            root,
            arguments.phase1_output_root,
        )
        / source_test_name
    )

    stage0_script = resolve(
        root,
        arguments.stage0_script,
    )

    stage1_script = resolve(
        root,
        arguments.stage1_script,
    )

    stage2_script = resolve(
        root,
        arguments.stage2_script,
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

    for path in (
        stage0_script,
        stage1_script,
        stage2_script,
        stage2_helper_path,
        stage2b_helper_path,
        reid_helper_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    if not arguments.skip_phase1_run:
        run_phase1_source(
            root,
            source_dir,
            source_test_name,
            video,
            initial_bbox,
            stage0_script,
            stage1_script,
            stage2_script,
            arguments.device,
            arguments.overwrite_phase1_source,
        )

    elif not source_dir.is_dir():
        raise FileNotFoundError(
            source_dir
        )

    timeline_path = find_stage2_timeline(
        source_dir,
        cut_frame,
    )

    detections_path = find_detections_csv(
        source_dir
    )

    timeline = read_json(
        timeline_path
    )

    frames = timeline[
        "frames"
    ]

    if len(
        frames
    ) < cut_frame:
        raise RuntimeError(
            "Phase-1 timeline does not cover "
            "the pre-cut shot"
        )

    stage2 = load_module(
        "kickclip_stage3a2_stage2",
        stage2_helper_path,
    )

    stage2b = load_module(
        "kickclip_stage3a2_stage2b",
        stage2b_helper_path,
    )

    reid = load_module(
        "kickclip_stage3a2_reid",
        reid_helper_path,
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

    frame_count = int(
        stage3a1[
            "video"
        ][
            "frame_count"
        ]
    )

    (
        by_frame,
        detection_count,
    ) = stage2.load_detections(
        detections_path,
        frame_count,
        width,
        height,
    )

    by_id = {
        str(
            detection.detection_id
        ): detection
        for detections
        in by_frame.values()
        for detection
        in detections
    }

    candidates = build_memory_candidates(
        frames,
        by_frame,
        by_id,
        cut_frame,
        arguments.memory_window_frames,
        stage2,
    )

    selected = select_diverse_candidates(
        candidates,
        arguments.gallery_size,
        arguments.minimum_frame_gap,
    )

    if len(
        selected
    ) < 4:
        raise RuntimeError(
            "Insufficient stable pre-cut "
            "target observations: "
            f"{len(selected)}; at least 4 required"
        )

    gallery = [
        item[
            "detection"
        ]
        for item
        in selected
    ]

    policy = stage2b.Policy()

    negative_pool = (
        stage2b.select_negative_pool(
            gallery,
            by_frame,
            stage2,
            policy,
        )
    )

    manifest_models = (
        manifest_audit[
            "models"
        ]
    )

    if (
        arguments.sports_osnet_checkpoint
        is None
    ):
        checkpoint = Path(
            str(
                manifest_models[
                    "sports_osnet"
                ][
                    "path"
                ]
            )
        ).resolve()

    else:
        checkpoint = resolve(
            root,
            arguments.sports_osnet_checkpoint,
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )

    (
        deep_eiou_root,
        models,
        reid_root,
    ) = (
        stage2b.discover_deep_eiou(
            root,
            reid,
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
            "CUDA requested but unavailable"
        )

    device = torch.device(
        device_name
    )

    if hasattr(
        reid,
        "configure_determinism",
    ):
        reid.configure_determinism(
            torch
        )

    (
        model,
        model_contract,
    ) = reid.build_model(
        torch,
        models,
        checkpoint,
        device,
    )

    transform = (
        stage2b.build_transform()
    )

    all_for_embedding = {
        str(
            detection.detection_id
        ): detection
        for detection
        in (
            gallery
            + negative_pool
        )
    }

    crops = stage2b.collect_crops(
        video,
        list(
            all_for_embedding.values()
        ),
        frame_count,
        width,
        height,
    )

    embeddings = stage2b.embed_crops(
        crops,
        model,
        transform,
        torch,
        device,
        arguments.batch_size,
    )

    target_embeddings = [
        embeddings[
            str(
                detection.detection_id
            )
        ]
        for detection
        in gallery
    ]

    target_prototype = (
        stage2b.l2_mean(
            target_embeddings
        )
    )

    negative_ranked = sorted(
        (
            (
                float(
                    embeddings[
                        str(
                            detection.detection_id
                        )
                    ]
                    @ target_prototype
                ),
                detection,
            )
            for detection
            in negative_pool
        ),
        key=lambda pair: (
            pair[
                0
            ]
        ),
        reverse=True,
    )[
        :
        arguments.negative_gallery_size
    ]

    negative_detections = [
        detection
        for _,
        detection
        in negative_ranked
    ]

    negative_embeddings = [
        embeddings[
            str(
                detection.detection_id
            )
        ]
        for detection
        in negative_detections
    ]

    target_embedding_path = (
        output_dir
        / "stage3a2_target_embeddings.npy"
    )

    negative_embedding_path = (
        output_dir
        / "stage3a2_negative_embeddings.npy"
    )

    atomic_npy(
        target_embedding_path,
        np.stack(
            target_embeddings
        ).astype(
            np.float32
        ),
    )

    atomic_npy(
        negative_embedding_path,
        (
            np.stack(
                negative_embeddings
            ).astype(
                np.float32
            )
            if negative_embeddings
            else np.empty(
                (
                    0,
                    EMBEDDING_DIM,
                ),
                dtype=np.float32,
            )
        ),
    )

    crop_records = (
        save_memory_crops_and_sheet(
            output_dir,
            video,
            selected,
        )
    )

    selected_ids = {
        item[
            "detection_id"
        ]
        for item
        in selected
    }

    candidate_rows = [
        candidate_csv_row(
            item,
            selected_ids,
        )
        for item
        in candidates
    ]

    candidate_path = (
        output_dir
        / "stage3a2_memory_candidates.csv"
    )

    write_csv(
        candidate_path,
        candidate_rows,
        list(
            candidate_rows[
                0
            ]
        ),
    )

    preview_path = (
        output_dir
        / "stage3a2_precut_tracking_preview.mp4"
    )

    render_precut_preview(
        video,
        preview_path,
        frames,
        cut_frame,
        fps,
        width,
        height,
    )

    target_memory = {
        "stage": (
            STAGE
        ),
        "version": (
            VERSION
        ),
        "status": (
            "PENDING_VISUAL_REVIEW"
        ),
        "test_name": (
            test_name
        ),
        "confirmed_cut_frame": (
            cut_frame
        ),
        "source_frame_range": [
            max(
                0,
                cut_frame
                - arguments.memory_window_frames,
            ),
            cut_frame
            - 1,
        ],
        "memory_policy": {
            "source": (
                "FROZEN_PHASE1_"
                "CONFIRMED_TRACKING_ONLY"
            ),
            "post_cut_observations_used": (
                False
            ),
            "automatic_memory_update": (
                False
            ),
            "gallery_size": (
                len(
                    crop_records
                )
            ),
            "minimum_frame_gap": (
                arguments.minimum_frame_gap
            ),
        },
        "target_gallery": (
            crop_records
        ),
        "negative_gallery": [
            {
                "rank": (
                    rank
                ),
                "detection_id": str(
                    detection.detection_id
                ),
                "frame_index": int(
                    detection.frame
                ),
                "bbox_xyxy": [
                    float(
                        value
                    )
                    for value
                    in detection.bbox
                ],
                "target_similarity": (
                    similarity
                ),
            }
            for rank, (
                similarity,
                detection,
            )
            in enumerate(
                negative_ranked,
                start=1,
            )
        ],
        "embeddings": {
            "dimension": (
                EMBEDDING_DIM
            ),
            "target_path": str(
                target_embedding_path
            ),
            "negative_path": str(
                negative_embedding_path
            ),
            "target_prototype_l2_norm": (
                float(
                    np.linalg.norm(
                        target_prototype
                    )
                )
            ),
        },
        "model": {
            "architecture": (
                "osnet_x1_0"
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
            "load_contract": (
                model_contract
            ),
            "device": str(
                device
            ),
        },
    }

    memory_path = (
        output_dir
        / "stage3a2_target_memory.json"
    )

    atomic_json(
        memory_path,
        target_memory,
    )

    summary = {
        "stage": (
            STAGE
        ),
        "version": (
            VERSION
        ),
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
            "PASS"
        ),
        "decision": (
            "AUTHORIZE_MANDATORY_"
            "STAGE3A2_TARGET_MEMORY_"
            "VISUAL_REVIEW"
        ),
        "test_name": (
            test_name
        ),
        "confirmed_cut_frame": (
            cut_frame
        ),
        "phase1_freeze": (
            manifest_audit
        ),
        "phase1_source": {
            "test_name": (
                source_test_name
            ),
            "output_dir": str(
                source_dir
            ),
            "timeline": str(
                timeline_path
            ),
            "timeline_sha256": (
                sha256_file(
                    timeline_path
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
            "detection_count": (
                detection_count
            ),
        },
        "counts": {
            "precut_frame_count": (
                cut_frame
            ),
            "stable_memory_candidate_count": (
                len(
                    candidates
                )
            ),
            "selected_target_crop_count": (
                len(
                    selected
                )
            ),
            "negative_pool_count": (
                len(
                    negative_pool
                )
            ),
            "selected_negative_crop_count": (
                len(
                    negative_detections
                )
            ),
        },
        "outputs": {
            "target_memory": str(
                memory_path
            ),
            "target_embeddings": str(
                target_embedding_path
            ),
            "negative_embeddings": str(
                negative_embedding_path
            ),
            "candidate_audit": str(
                candidate_path
            ),
            "memory_contact_sheet": str(
                output_dir
                / "stage3a2_memory_contact_sheet.jpg"
            ),
            "precut_tracking_preview": str(
                preview_path
            ),
        },
        "safety_invariants": {
            "post_cut_frames_read_for_target_memory": (
                False
            ),
            "post_cut_candidate_selection": (
                False
            ),
            "cross_shot_linking": (
                False
            ),
            "automatic_memory_update": (
                False
            ),
            "frozen_phase1_verified": (
                True
            ),
            "frozen_osnet_verified": (
                True
            ),
            "threshold_search": (
                False
            ),
        },
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
    }

    summary_path = (
        output_dir
        / "stage3a2_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3a2_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V2 Stage 3-A2 complete"
    )

    print(
        "Status                       : PASS"
    )

    print(
        "Decision                     : "
        "AUTHORIZE_MANDATORY_"
        "STAGE3A2_TARGET_MEMORY_"
        "VISUAL_REVIEW"
    )

    print(
        f"Confirmed cut                : "
        f"frame {cut_frame}"
    )

    print(
        f"Frozen Phase-1 source        : "
        f"{source_test_name}"
    )

    print(
        f"Stable memory candidates     : "
        f"{len(candidates)}"
    )

    print(
        f"Selected target crops        : "
        f"{len(selected)}"
    )

    print(
        f"Selected negative crops      : "
        f"{len(negative_detections)}"
    )

    print(
        f"Sports OSNet                 : "
        f"{sha256_file(checkpoint)}"
    )

    print(
        "Post-cut observations used   : NONE"
    )

    print(
        "Cross-shot linking           : NONE"
    )

    print(
        f"Memory contact sheet         : "
        f"{output_dir / 'stage3a2_memory_contact_sheet.jpg'}"
    )

    print(
        f"Pre-cut tracking preview     : "
        f"{preview_path}"
    )

    print(
        f"Output                       : "
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
            "Stage 3-A2 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-A2 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )