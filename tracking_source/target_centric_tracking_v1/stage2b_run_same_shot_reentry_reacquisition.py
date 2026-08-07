#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 2-B same-shot re-entry.

This stage preserves Stage 2 and searches only after its first LOST frame.
Automatic reacquisition requires all of the following:
- a new player/goalkeeper tracklet entering from a frame edge,
- at least five detections in a short confirmation window,
- frozen Sports OSNet similarity to a pre-LOST target gallery,
- separation from simultaneous known-negative/confuser players,
- internally consistent candidate crops,
- a sufficient top-vs-second candidate margin.

If any condition fails, the state remains SEARCHING or AMBIGUOUS. Appearance
alone is never sufficient. No training, threshold sweep, Global ID, GTA, V7,
camera-cut handling, or action spotting is performed.
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

STAGE = "stage2b_same_shot_reentry_reacquisition"
VERSION = "target-centric-v1-stage2b-1.0.0"

EXPECTED_OSNET_SHA256 = (
    "8d5b2fd8763db34c2aad69810466adf4"
    "13f0426d9f8119d322227e0e639c5fbd"
)
EXPECTED_OSNET_BYTES = 30_393_613

TARGET_CLASSES = {0, 1}
EMBEDDING_DIM = 512


@dataclass(frozen=True)
class Policy:
    gallery_size: int = 8
    negative_gallery_size: int = 16
    required_negative_gallery_size: int = 4

    gallery_min_det_conf: float = 0.45
    gallery_min_tracking_conf: float = 0.75
    gallery_min_identity_conf: float = 0.78

    negative_min_det_conf: float = 0.35
    negative_max_target_iou: float = 0.15
    negative_min_target_center_distance: float = 0.65

    track_max_gap: int = 2
    track_min_detections: int = 5

    confirm_detections: int = 5
    confirm_max_span: int = 10

    candidate_samples: int = 6
    candidate_min_det_conf: float = 0.30
    candidate_min_mean_det_conf: float = 0.40
    candidate_area_ratio_min: float = 0.30
    candidate_area_ratio_max: float = 3.00

    entry_edge_distance_max: float = 0.12

    association_min_iou: float = 0.06
    association_max_center_distance: float = 0.85

    target_similarity_min: float = 0.72
    target_max_similarity_min: float = 0.78
    negative_margin_min: float = 0.04
    candidate_margin_min: float = 0.055
    crop_consistency_min: float = 0.70
    combined_score_min: float = 0.75

    plausible_target_similarity: float = 0.64
    plausible_combined_score: float = 0.66

    continuation_target_similarity_min: float = 0.66
    continuation_target_max_similarity_min: float = 0.70
    continuation_negative_margin_min: float = 0.00
    continuation_max_unverified_gap: int = 3


@dataclass
class Tracklet:
    tracklet_id: str
    detections: list[Any]
    missed: int = 0


@dataclass
class Candidate:
    tracklet_id: str
    start_frame: int
    confirmation_frame: int
    end_frame: int
    detection_count: int
    confirmation_span: int

    entry_edge_distance: float
    entry_score: float

    mean_detection_confidence: float
    mean_temporal_iou: float

    target_similarity: float
    target_max_similarity: float
    negative_max_similarity: float
    negative_margin: float
    crop_consistency: float
    combined_score: float

    sample_detection_ids: list[str]

    hard_gate_pass: bool
    rejection_reasons: list[str]

    rank: int = 0
    selected: bool = False


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
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-stage2b",
        action="store_true",
    )

    return parser.parse_args()


def resolve(
    root: Path,
    value: Path,
) -> Path:
    value = value.expanduser()

    if value.is_absolute():
        return value.resolve()

    return (root / value).resolve()


def load_module(
    name: str,
    path: Path,
) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def read_json(
    path: Path,
) -> dict[str, Any]:
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(
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
            allow_nan=False,
        )
        + "\n",
    )


def atomic_npy(
    path: Path,
    value: np.ndarray,
) -> None:
    temporary = path.with_name(
        path.name + ".tmp.npy"
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
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )

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

    os.replace(
        temporary,
        path,
    )


def prepare_outputs(
    test_dir: Path,
    overwrite: bool,
) -> None:
    names = (
        "stage2b_target_gallery.json",
        "stage2b_target_embeddings.npy",
        "stage2b_negative_embeddings.npy",
        "stage2b_candidate_embeddings.npy",
        "stage2b_reentry_candidates.csv",
        "stage2b_frame_observations.csv",
        "stage2b_target_timeline.json",
        "stage2b_reentry_summary.json",
        "stage2b_reentry_report.md",
        "stage2b_reentry_preview.mp4",
    )

    existing = [
        test_dir / name
        for name in names
        if (test_dir / name).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Stage 2-B outputs already exist. "
            "Use --overwrite-stage2b:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    for path in existing:
        path.unlink()


def discover_checkpoint(
    root: Path,
    explicit: Optional[Path],
) -> Path:
    if explicit is not None:
        search = [
            resolve(
                root,
                explicit,
            )
        ]
    else:
        search = []

        for directory in (
            root / "global_ID_tracking_upgrade_v6",
            root
            / "runs"
            / "global_ID_tracking_upgrade_v6",
            root / "weights",
            root / "backups",
        ):
            if directory.is_dir():
                search.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file()
                )

    for path in search:
        try:
            if (
                path.stat().st_size
                != EXPECTED_OSNET_BYTES
            ):
                continue
        except OSError:
            continue

        if (
            sha256_file(path)
            == EXPECTED_OSNET_SHA256
        ):
            return path.resolve()

    raise FileNotFoundError(
        "Frozen Sports OSNet checkpoint not found. "
        "Pass --checkpoint with its exact path."
    )


def discover_deep_eiou(
    root: Path,
    helper: Any,
    explicit: Optional[Path],
) -> tuple[Path, Any, Path]:
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(
            resolve(
                root,
                explicit,
            )
        )
    else:
        for directory in (
            root
            / "global_ID_tracking_upgrade_v6"
            / "third_party",
            root / "global_ID_tracking_upgrade_v6",
        ):
            if not directory.is_dir():
                continue

            for osnet in directory.rglob(
                "osnet.py"
            ):
                parents = list(
                    osnet.parents
                )

                for index in (3, 4):
                    if index < len(parents):
                        candidates.append(
                            parents[index]
                        )

    errors: list[str] = []

    for candidate in sorted(
        set(
            path.resolve()
            for path in candidates
            if path.exists()
        )
    ):
        try:
            models, reid_root = (
                helper.import_vendored_osnet(
                    candidate
                )
            )

            return (
                candidate,
                models,
                reid_root,
            )
        except Exception as exc:
            errors.append(
                f"{candidate}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    raise FileNotFoundError(
        "Deep-EIoU torchreid not found. "
        "Pass --deep-eiou-root.\n"
        + "\n".join(
            errors[:20]
        )
    )


def load_stage2_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = list(
            csv.DictReader(stream)
        )

    rows.sort(
        key=lambda row: int(
            row["frame_index"]
        )
    )

    for frame, row in enumerate(rows):
        if int(
            row["frame_index"]
        ) != frame:
            raise ValueError(
                "Stage 2 frame_observations.csv "
                "is not contiguous"
            )

    return rows


def optional_float(
    value: str,
) -> Optional[float]:
    if value.strip() == "":
        return None

    return float(value)


def uniform_sample(
    items: Sequence[Any],
    count: int,
) -> list[Any]:
    if len(items) <= count:
        return list(items)

    indices = sorted(
        set(
            int(round(value))
            for value in np.linspace(
                0,
                len(items) - 1,
                count,
            )
        )
    )

    return [
        items[index]
        for index in indices
    ]


def edge_distance(
    box: Sequence[float],
    width: int,
    height: int,
) -> float:
    return min(
        max(
            0.0,
            box[0],
        )
        / width,
        max(
            0.0,
            box[1],
        )
        / height,
        max(
            0.0,
            width - 1 - box[2],
        )
        / width,
        max(
            0.0,
            height - 1 - box[3],
        )
        / height,
    )


def select_target_gallery(
    rows: Sequence[dict[str, str]],
    by_id: Mapping[str, Any],
    first_lost: int,
    policy: Policy,
) -> list[Any]:
    eligible: list[Any] = []

    for row in rows[:first_lost]:
        if row["state"] not in {
            "INITIALIZING",
            "ACTIVE",
        }:
            continue

        detection_id = row[
            "selected_detection_id"
        ].strip()

        if (
            not detection_id
            or detection_id not in by_id
        ):
            continue

        if row["state"] == "INITIALIZING":
            eligible.append(
                by_id[detection_id]
            )
            continue

        det_conf = optional_float(
            row["detection_confidence"]
        )
        track_conf = optional_float(
            row["tracking_confidence"]
        )
        id_conf = optional_float(
            row["identity_confidence"]
        )

        if (
            det_conf is None
            or track_conf is None
            or id_conf is None
            or det_conf
            < policy.gallery_min_det_conf
            or track_conf
            < policy.gallery_min_tracking_conf
            or id_conf
            < policy.gallery_min_identity_conf
            or row[
                "review_risk_flags"
            ].strip()
        ):
            continue

        eligible.append(
            by_id[detection_id]
        )

    if not eligible:
        raise RuntimeError(
            "No high-confidence "
            "target gallery observations"
        )

    return uniform_sample(
        eligible,
        policy.gallery_size,
    )


def select_negative_pool(
    gallery: Sequence[Any],
    by_frame: Mapping[int, list[Any]],
    stage2_helper: Any,
    policy: Policy,
) -> list[Any]:
    scored: list[tuple[float, Any]] = []

    for target in gallery:
        for candidate in by_frame.get(
            target.frame,
            [],
        ):
            if (
                candidate.class_id
                not in TARGET_CLASSES
                or candidate.detection_id
                == target.detection_id
                or candidate.confidence
                < policy.negative_min_det_conf
            ):
                continue

            overlap = stage2_helper.iou(
                target.bbox,
                candidate.bbox,
            )

            distance = (
                stage2_helper.point_distance(
                    stage2_helper.center(
                        target.bbox
                    ),
                    stage2_helper.center(
                        candidate.bbox
                    ),
                    target.bbox,
                )
            )

            if (
                overlap
                > policy.negative_max_target_iou
            ):
                continue

            if (
                distance
                < policy
                .negative_min_target_center_distance
            ):
                continue

            score = (
                1.0 / (1.0 + distance)
                + 0.1
                * candidate.confidence
            )

            scored.append(
                (
                    score,
                    candidate,
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    result: list[Any] = []
    used: set[str] = set()

    for _, detection in scored:
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

        if (
            len(result)
            >= policy.negative_gallery_size
            * 2
        ):
            break

    return result


def predict_bbox(
    tracklet: Tracklet,
) -> Sequence[float]:
    last = tracklet.detections[-1]

    if len(tracklet.detections) < 2:
        return last.bbox

    previous = tracklet.detections[-2]

    source_gap = max(
        1,
        last.frame - previous.frame,
    )
    future_gap = (
        tracklet.missed + 1
    )

    last_center = (
        (
            last.bbox[0]
            + last.bbox[2]
        )
        / 2,
        (
            last.bbox[1]
            + last.bbox[3]
        )
        / 2,
    )

    previous_center = (
        (
            previous.bbox[0]
            + previous.bbox[2]
        )
        / 2,
        (
            previous.bbox[1]
            + previous.bbox[3]
        )
        / 2,
    )

    velocity_x = (
        last_center[0]
        - previous_center[0]
    ) / source_gap

    velocity_y = (
        last_center[1]
        - previous_center[1]
    ) / source_gap

    box_width = (
        last.bbox[2]
        - last.bbox[0]
    )
    box_height = (
        last.bbox[3]
        - last.bbox[1]
    )

    center_x = (
        last_center[0]
        + velocity_x
        * future_gap
    )
    center_y = (
        last_center[1]
        + velocity_y
        * future_gap
    )

    return [
        center_x - box_width / 2,
        center_y - box_height / 2,
        center_x + box_width / 2,
        center_y + box_height / 2,
    ]


def build_tracklets(
    by_frame: Mapping[int, list[Any]],
    first_lost: int,
    frame_count: int,
    target_area: float,
    stage2_helper: Any,
    policy: Policy,
) -> list[Tracklet]:
    from scipy.optimize import (
        linear_sum_assignment,
    )

    active: list[Tracklet] = []
    completed: list[Tracklet] = []

    next_id = 1

    for frame in range(
        first_lost,
        frame_count,
    ):
        detections = [
            item
            for item in by_frame.get(
                frame,
                [],
            )
            if item.class_id
            in TARGET_CLASSES
            and item.confidence
            >= policy.candidate_min_det_conf
            and (
                policy.candidate_area_ratio_min
                <= stage2_helper.area(
                    item.bbox
                )
                / max(
                    target_area,
                    1e-6,
                )
                <= policy
                .candidate_area_ratio_max
            )
        ]

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()

        if active and detections:
            cost = np.full(
                (
                    len(active),
                    len(detections),
                ),
                1e6,
                dtype=np.float64,
            )

            for row_index, tracklet in enumerate(
                active
            ):
                predicted = predict_bbox(
                    tracklet
                )

                for (
                    column_index,
                    detection,
                ) in enumerate(detections):
                    overlap = (
                        stage2_helper.iou(
                            predicted,
                            detection.bbox,
                        )
                    )

                    distance = (
                        stage2_helper
                        .point_distance(
                            stage2_helper
                            .center(predicted),
                            stage2_helper
                            .center(
                                detection.bbox
                            ),
                            predicted,
                        )
                    )

                    area_ratio = (
                        stage2_helper.area(
                            detection.bbox
                        )
                        / max(
                            stage2_helper.area(
                                predicted
                            ),
                            1e-6,
                        )
                    )

                    if not (
                        overlap
                        >= policy
                        .association_min_iou
                        or distance
                        <= policy
                        .association_max_center_distance
                    ):
                        continue

                    if not (
                        0.35
                        <= area_ratio
                        <= 2.85
                    ):
                        continue

                    size_similarity = math.exp(
                        -abs(
                            math.log(
                                max(
                                    area_ratio,
                                    1e-6,
                                )
                            )
                        )
                    )

                    cost[
                        row_index,
                        column_index,
                    ] = (
                        0.55
                        * (1 - overlap)
                        + 0.30
                        * min(
                            distance,
                            1.5,
                        )
                        / 1.5
                        + 0.15
                        * (
                            1
                            - size_similarity
                        )
                    )

            (
                row_indices,
                column_indices,
            ) = linear_sum_assignment(
                cost
            )

            for (
                row_index,
                column_index,
            ) in zip(
                row_indices.tolist(),
                column_indices.tolist(),
            ):
                if (
                    cost[
                        row_index,
                        column_index,
                    ]
                    >= 1e5
                ):
                    continue

                active[
                    row_index
                ].detections.append(
                    detections[
                        column_index
                    ]
                )
                active[
                    row_index
                ].missed = 0

                assigned_tracks.add(
                    row_index
                )
                assigned_detections.add(
                    column_index
                )

        for (
            index,
            tracklet,
        ) in enumerate(active):
            if index not in assigned_tracks:
                tracklet.missed += 1

        for (
            index,
            detection,
        ) in enumerate(detections):
            if (
                index
                in assigned_detections
            ):
                continue

            active.append(
                Tracklet(
                    tracklet_id=(
                        f"reentry_track_"
                        f"{next_id:04d}"
                    ),
                    detections=[
                        detection
                    ],
                )
            )

            next_id += 1

        remaining: list[Tracklet] = []

        for tracklet in active:
            if (
                tracklet.missed
                > policy.track_max_gap
            ):
                completed.append(
                    tracklet
                )
            else:
                remaining.append(
                    tracklet
                )

        active = remaining

    completed.extend(active)

    return sorted(
        completed,
        key=lambda item: (
            item.detections[0].frame,
            item.tracklet_id,
        ),
    )


def collect_crops(
    video: Path,
    detections: Sequence[Any],
    frame_count: int,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    requested: dict[
        int,
        list[Any],
    ] = defaultdict(list)

    for detection in detections:
        requested[
            detection.frame
        ].append(detection)

    capture = cv2.VideoCapture(
        str(video)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open {video}"
        )

    crops: dict[
        str,
        np.ndarray,
    ] = {}

    frame = 0

    try:
        while True:
            ok, image = capture.read()

            if not ok:
                break

            for detection in requested.get(
                frame,
                [],
            ):
                x1 = max(
                    0,
                    min(
                        width - 2,
                        math.floor(
                            detection.bbox[0]
                        ),
                    ),
                )
                y1 = max(
                    0,
                    min(
                        height - 2,
                        math.floor(
                            detection.bbox[1]
                        ),
                    ),
                )
                x2 = max(
                    x1 + 1,
                    min(
                        width,
                        math.ceil(
                            detection.bbox[2]
                        ),
                    ),
                )
                y2 = max(
                    y1 + 1,
                    min(
                        height,
                        math.ceil(
                            detection.bbox[3]
                        ),
                    ),
                )

                crop = image[
                    y1:y2,
                    x1:x2,
                ]

                if crop.size == 0:
                    raise RuntimeError(
                        "Empty crop: "
                        f"{detection.detection_id}"
                    )

                crops[
                    detection.detection_id
                ] = crop.copy()

            frame += 1

    finally:
        capture.release()

    if frame != frame_count:
        raise RuntimeError(
            "Video frame count changed: "
            f"{frame}/{frame_count}"
        )

    return crops


def build_transform() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    256,
                    128,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def embed_crops(
    crops: Mapping[
        str,
        np.ndarray,
    ],
    model: Any,
    transform: Any,
    torch: Any,
    device: Any,
    batch_size: int,
) -> dict[
    str,
    np.ndarray,
]:
    from PIL import Image

    keys = list(crops)
    result: dict[
        str,
        np.ndarray,
    ] = {}

    with torch.inference_mode():
        for start in range(
            0,
            len(keys),
            batch_size,
        ):
            batch_keys = keys[
                start : start + batch_size
            ]

            tensors = []

            for key in batch_keys:
                rgb = cv2.cvtColor(
                    crops[key],
                    cv2.COLOR_BGR2RGB,
                )

                tensors.append(
                    transform(
                        Image.fromarray(rgb)
                    )
                )

            batch = torch.stack(
                tensors
            ).to(
                device=device,
                dtype=torch.float32,
            )

            output = model(batch)

            if (
                not torch.is_tensor(output)
                or output.ndim != 2
                or output.shape[1]
                != EMBEDDING_DIM
            ):
                raise RuntimeError(
                    "Unexpected embedding "
                    f"output: "
                    f"{getattr(output, 'shape', None)}"
                )

            output = (
                torch.nn.functional
                .normalize(
                    output,
                    p=2,
                    dim=1,
                )
            )

            array = (
                output
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            for (
                key,
                vector,
            ) in zip(
                batch_keys,
                array,
            ):
                result[key] = (
                    vector.copy()
                )

    return result


def l2_mean(
    values: Sequence[np.ndarray],
) -> np.ndarray:
    vector = np.mean(
        np.stack(values),
        axis=0,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        vector
    )

    if (
        not math.isfinite(
            float(norm)
        )
        or norm <= 0
    ):
        raise RuntimeError(
            "Invalid prototype norm"
        )

    return (
        vector / norm
    ).astype(
        np.float32
    )


def consistency(
    values: Sequence[np.ndarray],
) -> float:
    if len(values) < 2:
        return 0.0

    matrix = np.stack(values)

    similarities = (
        matrix @ matrix.T
    )

    pairs = similarities[
        np.triu_indices(
            len(values),
            k=1,
        )
    ]

    return float(
        np.median(pairs)
    )


def sample_tracklet(
    tracklet: Tracklet,
    policy: Policy,
) -> list[Any]:
    early = tracklet.detections[
        : max(
            policy.confirm_detections
            + 3,
            8,
        )
    ]

    return uniform_sample(
        early,
        policy.candidate_samples,
    )


def score_candidates(
    tracklets: Sequence[Tracklet],
    embedding_by_id: Mapping[
        str,
        np.ndarray,
    ],
    target_embeddings: Sequence[
        np.ndarray
    ],
    target_prototype: np.ndarray,
    negative_embeddings: Sequence[
        np.ndarray
    ],
    width: int,
    height: int,
    stage2_helper: Any,
    policy: Policy,
) -> tuple[
    list[Candidate],
    np.ndarray,
]:
    results: list[Candidate] = []
    prototype_by_id: dict[
        str,
        np.ndarray,
    ] = {}

    for tracklet in tracklets:
        if (
            len(tracklet.detections)
            < policy.track_min_detections
        ):
            continue

        confirmation = (
            tracklet.detections[
                policy.confirm_detections
                - 1
            ]
        )

        samples = sample_tracklet(
            tracklet,
            policy,
        )

        sample_embeddings = [
            embedding_by_id[
                item.detection_id
            ]
            for item in samples
        ]

        prototype = l2_mean(
            sample_embeddings
        )

        prototype_by_id[
            tracklet.tracklet_id
        ] = prototype

        target_similarity = float(
            prototype
            @ target_prototype
        )

        target_max_similarity = max(
            float(
                prototype @ item
            )
            for item
            in target_embeddings
        )

        if negative_embeddings:
            negative_max_similarity = max(
                float(
                    prototype @ item
                )
                for item
                in negative_embeddings
            )
        else:
            negative_max_similarity = -1.0

        negative_margin = (
            target_similarity
            - negative_max_similarity
        )

        entry_distance = edge_distance(
            tracklet.detections[0].bbox,
            width,
            height,
        )

        entry_score = math.exp(
            -entry_distance / 0.08
        )

        mean_detection_confidence = (
            statistics.fmean(
                item.confidence
                for item
                in tracklet.detections[
                    : policy.confirm_detections
                ]
            )
        )

        temporal_ious = [
            stage2_helper.iou(
                first.bbox,
                second.bbox,
            )
            for first, second in zip(
                tracklet.detections[
                    : policy.confirm_detections
                ],
                tracklet.detections[
                    1 : policy.confirm_detections
                ],
            )
        ]

        if temporal_ious:
            mean_temporal_iou = (
                statistics.fmean(
                    temporal_ious
                )
            )
        else:
            mean_temporal_iou = 0.0

        crop_consistency = consistency(
            sample_embeddings
        )

        normalized_negative_margin = max(
            -1.0,
            min(
                1.0,
                negative_margin + 0.5,
            ),
        )

        combined_score = (
            0.50
            * target_similarity
            + 0.12
            * target_max_similarity
            + 0.13
            * normalized_negative_margin
            + 0.10
            * crop_consistency
            + 0.08
            * entry_score
            + 0.07
            * mean_detection_confidence
        )

        rejection_reasons: list[str] = []

        confirmation_span = (
            confirmation.frame
            - tracklet.detections[0].frame
            + 1
        )

        if (
            confirmation_span
            > policy.confirm_max_span
        ):
            rejection_reasons.append(
                "INSUFFICIENT_TEMPORAL_PERSISTENCE"
            )

        if (
            entry_distance
            > policy.entry_edge_distance_max
        ):
            rejection_reasons.append(
                "TRACKLET_DID_NOT_ENTER_FROM_EDGE"
            )

        if (
            mean_detection_confidence
            < policy
            .candidate_min_mean_det_conf
        ):
            rejection_reasons.append(
                "MEAN_DETECTION_CONFIDENCE_TOO_LOW"
            )

        if (
            target_similarity
            < policy.target_similarity_min
        ):
            rejection_reasons.append(
                "TARGET_SIMILARITY_TOO_LOW"
            )

        if (
            target_max_similarity
            < policy.target_max_similarity_min
        ):
            rejection_reasons.append(
                "TARGET_MAX_SIMILARITY_TOO_LOW"
            )

        if (
            len(negative_embeddings)
            < policy
            .required_negative_gallery_size
        ):
            rejection_reasons.append(
                "INSUFFICIENT_NEGATIVE_MEMORY"
            )

        if (
            negative_margin
            < policy.negative_margin_min
        ):
            rejection_reasons.append(
                "NEGATIVE_MARGIN_TOO_LOW"
            )

        if (
            crop_consistency
            < policy.crop_consistency_min
        ):
            rejection_reasons.append(
                "CROP_CONSISTENCY_TOO_LOW"
            )

        if (
            combined_score
            < policy.combined_score_min
        ):
            rejection_reasons.append(
                "COMBINED_SCORE_TOO_LOW"
            )

        results.append(
            Candidate(
                tracklet_id=(
                    tracklet.tracklet_id
                ),
                start_frame=(
                    tracklet
                    .detections[0]
                    .frame
                ),
                confirmation_frame=(
                    confirmation.frame
                ),
                end_frame=(
                    tracklet
                    .detections[-1]
                    .frame
                ),
                detection_count=len(
                    tracklet.detections
                ),
                confirmation_span=(
                    confirmation_span
                ),
                entry_edge_distance=(
                    entry_distance
                ),
                entry_score=entry_score,
                mean_detection_confidence=(
                    mean_detection_confidence
                ),
                mean_temporal_iou=(
                    mean_temporal_iou
                ),
                target_similarity=(
                    target_similarity
                ),
                target_max_similarity=(
                    target_max_similarity
                ),
                negative_max_similarity=(
                    negative_max_similarity
                ),
                negative_margin=(
                    negative_margin
                ),
                crop_consistency=(
                    crop_consistency
                ),
                combined_score=(
                    combined_score
                ),
                sample_detection_ids=[
                    item.detection_id
                    for item in samples
                ],
                hard_gate_pass=(
                    not rejection_reasons
                ),
                rejection_reasons=(
                    rejection_reasons
                ),
            )
        )

    results.sort(
        key=lambda item: (
            item.hard_gate_pass,
            item.combined_score,
            item.target_similarity,
            item.negative_margin,
        ),
        reverse=True,
    )

    for rank, item in enumerate(
        results,
        1,
    ):
        item.rank = rank

    if results:
        prototypes = np.stack(
            [
                prototype_by_id[
                    item.tracklet_id
                ]
                for item in results
            ]
        ).astype(
            np.float32
        )
    else:
        prototypes = np.empty(
            (
                0,
                EMBEDDING_DIM,
            ),
            dtype=np.float32,
        )

    return (
        results,
        prototypes,
    )


def choose_candidate(
    candidates: Sequence[Candidate],
    policy: Policy,
) -> tuple[
    Optional[Candidate],
    str,
    Optional[float],
]:
    if not candidates:
        return (
            None,
            "NO_REENTRY_CANDIDATES",
            None,
        )

    top = candidates[0]

    if len(candidates) > 1:
        second_score = (
            candidates[1]
            .combined_score
        )
    else:
        second_score = 0.0

    margin = (
        top.combined_score
        - second_score
    )

    if not top.hard_gate_pass:
        return (
            None,
            "TOP_CANDIDATE_FAILED_HARD_GATES",
            margin,
        )

    if (
        margin
        < policy.candidate_margin_min
    ):
        return (
            None,
            "TOP_VS_SECOND_MARGIN_TOO_SMALL",
            margin,
        )

    top.selected = True

    return (
        top,
        "HIGH_CONFIDENCE_MULTI_EVIDENCE_REENTRY",
        margin,
    )


def verify_tracklet_frames(
    tracklet: Tracklet,
    confirmation_frame: int,
    embedding_by_id: Mapping[
        str,
        np.ndarray,
    ],
    target_embeddings: Sequence[
        np.ndarray
    ],
    target_prototype: np.ndarray,
    negative_embeddings: Sequence[
        np.ndarray
    ],
    policy: Policy,
) -> dict[int, dict[str, Any]]:
    result: dict[
        int,
        dict[str, Any],
    ] = {}

    for detection in tracklet.detections:
        if (
            detection.frame
            < confirmation_frame
        ):
            continue

        vector = embedding_by_id[
            detection.detection_id
        ]

        target_similarity = float(
            vector @ target_prototype
        )

        target_max_similarity = max(
            float(
                vector @ item
            )
            for item
            in target_embeddings
        )

        if negative_embeddings:
            negative_max_similarity = max(
                float(
                    vector @ item
                )
                for item
                in negative_embeddings
            )
        else:
            negative_max_similarity = -1.0

        negative_margin = (
            target_similarity
            - negative_max_similarity
        )

        verified = (
            target_similarity
            >= policy
            .continuation_target_similarity_min
            and target_max_similarity
            >= policy
            .continuation_target_max_similarity_min
            and negative_margin
            >= policy
            .continuation_negative_margin_min
        )

        result[
            detection.frame
        ] = {
            "detection": detection,
            "target_similarity": (
                target_similarity
            ),
            "target_max_similarity": (
                target_max_similarity
            ),
            "negative_max_similarity": (
                negative_max_similarity
            ),
            "negative_margin": (
                negative_margin
            ),
            "verified": verified,
        }

    return result


def original_bbox(
    row: Mapping[str, str],
) -> Optional[list[float]]:
    values = [
        row[f"bbox_{axis}"]
        for axis in (
            "x1",
            "y1",
            "x2",
            "y2",
        )
    ]

    if any(
        value.strip() == ""
        for value in values
    ):
        return None

    return [
        float(value)
        for value in values
    ]


def build_output_rows(
    stage2_rows: Sequence[dict[str, str]],
    first_lost: int,
    selected: Optional[Candidate],
    reason: str,
    candidate_margin: Optional[float],
    selected_tracklet: Optional[Tracklet],
    verification: Mapping[
        int,
        dict[str, Any],
    ],
    plausible: Optional[Candidate],
    fps: float,
    policy: Policy,
) -> list[dict[str, Any]]:
    output: list[
        dict[str, Any]
    ] = []

    if selected is not None:
        confirmation = (
            selected.confirmation_frame
        )
    else:
        confirmation = None

    if selected_tracklet is not None:
        selected_frames = {
            item.frame
            for item
            in selected_tracklet.detections
        }
    else:
        selected_frames = set()

    unverified_gap = 0

    for frame, row in enumerate(
        stage2_rows
    ):
        base = {
            "frame_index": frame,
            "time_ms": int(
                round(
                    frame
                    * 1000
                    / fps
                )
            ),
            "candidate_margin": (
                candidate_margin
            ),
        }

        if frame < first_lost:
            output.append(
                {
                    **base,
                    "state": row["state"],
                    "bbox_xyxy": (
                        original_bbox(row)
                    ),
                    "bbox_source": (
                        row["bbox_source"]
                    ),
                    "selected_detection_id": (
                        row[
                            "selected_detection_id"
                        ]
                        or None
                    ),
                    "tracking_confidence": float(
                        row[
                            "tracking_confidence"
                        ]
                    ),
                    "identity_confidence": float(
                        row[
                            "identity_confidence"
                        ]
                    ),
                    "decision_reason": (
                        "PRESERVED_FROM_"
                        "STAGE2_PRE_LOST"
                    ),
                    "target_similarity": None,
                    "target_max_similarity": None,
                    "negative_max_similarity": None,
                    "negative_margin": None,
                    "review_required": bool(
                        row[
                            "review_risk_flags"
                        ].strip()
                    ),
                }
            )
            continue

        if (
            selected is None
            or confirmation is None
            or frame < confirmation
        ):
            ambiguous = (
                plausible is not None
                and plausible.confirmation_frame
                <= frame
                <= plausible.end_frame
                and plausible.target_similarity
                >= policy
                .plausible_target_similarity
                and plausible.combined_score
                >= policy
                .plausible_combined_score
            )

            output.append(
                {
                    **base,
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
                    "review_required": (
                        ambiguous
                    ),
                }
            )
            continue

        verified = verification.get(
            frame
        )

        if (
            verified
            and verified["verified"]
        ):
            unverified_gap = 0

            detection = verified[
                "detection"
            ]

            identity_confidence = max(
                0.0,
                min(
                    1.0,
                    0.58
                    * verified[
                        "target_similarity"
                    ]
                    + 0.22
                    * verified[
                        "target_max_similarity"
                    ]
                    + 0.20
                    * max(
                        0.0,
                        min(
                            1.0,
                            verified[
                                "negative_margin"
                            ]
                            + 0.5,
                        ),
                    ),
                ),
            )

            output.append(
                {
                    **base,
                    "state": (
                        "REACQUIRED"
                        if frame
                        == confirmation
                        else "ACTIVE"
                    ),
                    "bbox_xyxy": list(
                        detection.bbox
                    ),
                    "bbox_source": (
                        "RFDETR_REENTRY_"
                        "TRACKLET_APPEARANCE_"
                        "VERIFIED"
                    ),
                    "selected_detection_id": (
                        detection.detection_id
                    ),
                    "tracking_confidence": min(
                        1.0,
                        0.65
                        * detection.confidence
                        + 0.35
                        * identity_confidence,
                    ),
                    "identity_confidence": (
                        identity_confidence
                    ),
                    "decision_reason": (
                        "MULTI_EVIDENCE_"
                        "REENTRY_CONFIRMED"
                        if frame
                        == confirmation
                        else
                        "POST_REENTRY_"
                        "FRAME_VERIFIED"
                    ),
                    "target_similarity": (
                        verified[
                            "target_similarity"
                        ]
                    ),
                    "target_max_similarity": (
                        verified[
                            "target_max_similarity"
                        ]
                    ),
                    "negative_max_similarity": (
                        verified[
                            "negative_max_similarity"
                        ]
                    ),
                    "negative_margin": (
                        verified[
                            "negative_margin"
                        ]
                    ),
                    "review_required": (
                        frame
                        == confirmation
                    ),
                }
            )

        else:
            unverified_gap += 1

            if (
                frame
                in selected_frames
                and unverified_gap
                <= policy
                .continuation_max_unverified_gap
            ):
                state = "OCCLUDED"
            else:
                state = "SEARCHING"

            output.append(
                {
                    **base,
                    "state": state,
                    "bbox_xyxy": None,
                    "bbox_source": "NONE",
                    "selected_detection_id": None,
                    "tracking_confidence": 0.0,
                    "identity_confidence": 0.0,
                    "decision_reason": (
                        "POST_REENTRY_DETECTION_"
                        "FAILED_VERIFICATION"
                        if verified
                        else
                        "REACQUIRED_TRACKLET_"
                        "NOT_PRESENT"
                    ),
                    "target_similarity": (
                        verified[
                            "target_similarity"
                        ]
                        if verified
                        else None
                    ),
                    "target_max_similarity": (
                        verified[
                            "target_max_similarity"
                        ]
                        if verified
                        else None
                    ),
                    "negative_max_similarity": (
                        verified[
                            "negative_max_similarity"
                        ]
                        if verified
                        else None
                    ),
                    "negative_margin": (
                        verified[
                            "negative_margin"
                        ]
                        if verified
                        else None
                    ),
                    "review_required": (
                        verified is not None
                    ),
                }
            )

    return output


def candidate_csv_row(
    item: Candidate,
) -> dict[str, Any]:
    row = asdict(item)

    row["sample_detection_ids"] = (
        "|".join(
            item.sample_detection_ids
        )
    )

    row["rejection_reasons"] = (
        "|".join(
            item.rejection_reasons
        )
    )

    return row


def observation_csv_row(
    item: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(item)

    bbox = row.pop(
        "bbox_xyxy"
    )

    for index, axis in enumerate(
        (
            "x1",
            "y1",
            "x2",
            "y2",
        )
    ):
        row[f"bbox_{axis}"] = (
            ""
            if bbox is None
            else f"{bbox[index]:.4f}"
        )

    for key in (
        "tracking_confidence",
        "identity_confidence",
        "target_similarity",
        "target_max_similarity",
        "negative_max_similarity",
        "negative_margin",
        "candidate_margin",
    ):
        row[key] = (
            ""
            if row[key] is None
            else f"{row[key]:.6f}"
        )

    return row


def segments(
    rows: Sequence[Mapping[str, Any]],
    fps: float,
) -> list[dict[str, Any]]:
    result: list[
        dict[str, Any]
    ] = []

    start = 0

    for index in range(
        1,
        len(rows) + 1,
    ):
        if (
            index < len(rows)
            and rows[index]["state"]
            == rows[start]["state"]
        ):
            continue

        part = rows[
            start:index
        ]

        result.append(
            {
                "segment_index": len(
                    result
                ),
                "state": (
                    part[0]["state"]
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
                    part[0]["time_ms"]
                ),
                "end_ms_exclusive": int(
                    round(
                        index
                        * 1000
                        / fps
                    )
                ),
                "frame_count": len(
                    part
                ),
                "bbox_frame_count": sum(
                    item["bbox_xyxy"]
                    is not None
                    for item in part
                ),
            }
        )

        start = index

    return result


def render_preview(
    video: Path,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    fps: float,
    width: int,
    height: int,
) -> None:
    temporary = output.with_name(
        output.stem + ".tmp.mp4"
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
        raise RuntimeError(
            "Cannot create "
            "Stage 2-B preview"
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
    }

    frame = 0

    try:
        while True:
            ok, image = capture.read()

            if not ok:
                break

            row = rows[frame]
            state = row["state"]

            color = colors.get(
                state,
                (
                    0,
                    0,
                    255,
                ),
            )

            bbox = row["bbox_xyxy"]

            if bbox is not None:
                x1, y1, x2, y2 = [
                    round(value)
                    for value in bbox
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
                            y1 - 5,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.rectangle(
                image,
                (
                    0,
                    0,
                ),
                (
                    width,
                    25,
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
                    f"Stage 2-B | "
                    f"frame={frame} | "
                    f"{state} | "
                    f"{row['decision_reason']}"
                ),
                (
                    6,
                    17,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

            writer.write(image)
            frame += 1

    finally:
        capture.release()
        writer.release()

    if frame != len(rows):
        raise RuntimeError(
            "Preview frame mismatch: "
            f"{frame}/{len(rows)}"
        )

    os.replace(
        temporary,
        output,
    )


def main() -> int:
    arguments = parse_args()
    started = time.perf_counter()

    root = (
        arguments.project_root
        .expanduser()
        .resolve()
    )

    test_dir = (
        resolve(
            root,
            arguments.output_root,
        )
        / arguments.test_name
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
        "stage2": (
            test_dir
            / "stage2_association_summary.json"
        ),
        "observations": (
            test_dir
            / "frame_observations.csv"
        ),
        "timeline": (
            test_dir
            / "target_timeline.json"
        ),
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    stage1 = read_json(
        paths["stage1"]
    )
    stage2 = read_json(
        paths["stage2"]
    )

    if (
        stage1.get("decision")
        != "AUTHORIZE_STAGE2_TARGET_ASSOCIATION"
    ):
        raise RuntimeError(
            "Stage 1 did not "
            "authorize Stage 2"
        )

    if (
        stage2.get("decision")
        != "AUTHORIZE_MANDATORY_STAGE2_VISUAL_REVIEW"
    ):
        raise RuntimeError(
            "Unexpected Stage 2 decision"
        )

    prepare_outputs(
        test_dir,
        arguments.overwrite_stage2b,
    )

    stage2_helper_path = resolve(
        root,
        arguments.stage2_helper,
    )
    reid_helper_path = resolve(
        root,
        arguments.v6_reid_helper,
    )

    stage2_helper = load_module(
        "kickclip_stage2b_stage2_helper",
        stage2_helper_path,
    )
    reid_helper = load_module(
        "kickclip_stage2b_reid_helper",
        reid_helper_path,
    )

    video_info = stage1["video"]

    video = Path(
        video_info["path"]
    ).resolve()

    if (
        not video.is_file()
        or sha256_file(video)
        != video_info["sha256"]
    ):
        raise RuntimeError(
            "Video missing or changed"
        )

    frame_count = int(
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

    (
        by_frame,
        detection_count,
    ) = stage2_helper.load_detections(
        paths["detections"],
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
        detection.detection_id: detection
        for detections
        in by_frame.values()
        for detection
        in detections
    }

    stage2_rows = load_stage2_rows(
        paths["observations"]
    )

    lost_frames = [
        int(row["frame_index"])
        for row in stage2_rows
        if row["state"] == "LOST"
    ]

    if not lost_frames:
        raise RuntimeError(
            "Stage 2 has no LOST frame. "
            "Stage 2-B is unnecessary."
        )

    first_lost = min(
        lost_frames
    )

    policy = Policy()

    gallery = select_target_gallery(
        stage2_rows,
        by_id,
        first_lost,
        policy,
    )

    target_area = statistics.median(
        stage2_helper.area(
            item.bbox
        )
        for item in gallery
    )

    negative_pool = select_negative_pool(
        gallery,
        by_frame,
        stage2_helper,
        policy,
    )

    all_tracklets = build_tracklets(
        by_frame,
        first_lost,
        frame_count,
        target_area,
        stage2_helper,
        policy,
    )

    edge_tracklets = [
        item
        for item in all_tracklets
        if len(item.detections)
        >= policy.track_min_detections
        and edge_distance(
            item.detections[0].bbox,
            width,
            height,
        )
        <= policy
        .entry_edge_distance_max
    ]

    checkpoint = discover_checkpoint(
        root,
        arguments.checkpoint,
    )

    (
        deep_eiou_root,
        models,
        reid_root,
    ) = discover_deep_eiou(
        root,
        reid_helper,
        arguments.deep_eiou_root,
    )

    import torch

    if arguments.device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
    else:
        device_name = arguments.device

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
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

    transform = build_transform()

    first_pass_detections: dict[
        str,
        Any,
    ] = {
        item.detection_id: item
        for item in (
            gallery
            + negative_pool
        )
    }

    for tracklet in edge_tracklets:
        for detection in sample_tracklet(
            tracklet,
            policy,
        ):
            first_pass_detections[
                detection.detection_id
            ] = detection

    first_pass_crops = collect_crops(
        video,
        list(
            first_pass_detections.values()
        ),
        frame_count,
        width,
        height,
    )

    embedding_by_id = embed_crops(
        first_pass_crops,
        model,
        transform,
        torch,
        device,
        arguments.batch_size,
    )

    target_embeddings = [
        embedding_by_id[
            item.detection_id
        ]
        for item in gallery
    ]

    target_prototype = l2_mean(
        target_embeddings
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
            for item in negative_pool
        ),
        reverse=True,
        key=lambda pair: pair[0],
    )[
        : policy.negative_gallery_size
    ]

    negatives = [
        item
        for _, item
        in negative_ranked
    ]

    negative_embeddings = [
        embedding_by_id[
            item.detection_id
        ]
        for item in negatives
    ]

    (
        candidates,
        candidate_embeddings,
    ) = score_candidates(
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

    (
        selected,
        selection_reason,
        selection_margin,
    ) = choose_candidate(
        candidates,
        policy,
    )

    plausible = (
        candidates[0]
        if candidates
        else None
    )

    tracklet_by_id = {
        item.tracklet_id: item
        for item in edge_tracklets
    }

    if selected is not None:
        selected_tracklet = (
            tracklet_by_id[
                selected.tracklet_id
            ]
        )
    else:
        selected_tracklet = None

    verification: dict[
        int,
        dict[str, Any],
    ] = {}

    if (
        selected is not None
        and selected_tracklet is not None
    ):
        remaining = [
            item
            for item
            in selected_tracklet.detections
            if item.frame
            >= selected.confirmation_frame
            and item.detection_id
            not in embedding_by_id
        ]

        if remaining:
            remaining_crops = collect_crops(
                video,
                remaining,
                frame_count,
                width,
                height,
            )

            embedding_by_id.update(
                embed_crops(
                    remaining_crops,
                    model,
                    transform,
                    torch,
                    device,
                    arguments.batch_size,
                )
            )

        verification = verify_tracklet_frames(
            selected_tracklet,
            selected.confirmation_frame,
            embedding_by_id,
            target_embeddings,
            target_prototype,
            negative_embeddings,
            policy,
        )

    output_rows = build_output_rows(
        stage2_rows,
        first_lost,
        selected,
        selection_reason,
        selection_margin,
        selected_tracklet,
        verification,
        plausible,
        fps,
        policy,
    )

    target_embedding_path = (
        test_dir
        / "stage2b_target_embeddings.npy"
    )
    negative_embedding_path = (
        test_dir
        / "stage2b_negative_embeddings.npy"
    )
    candidate_embedding_path = (
        test_dir
        / "stage2b_candidate_embeddings.npy"
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
        negative_embedding_matrix = (
            np.stack(
                negative_embeddings
            ).astype(
                np.float32
            )
        )
    else:
        negative_embedding_matrix = (
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
        negative_embedding_matrix,
    )

    atomic_npy(
        candidate_embedding_path,
        candidate_embeddings,
    )

    target_gallery_path = (
        test_dir
        / "stage2b_target_gallery.json"
    )

    atomic_json(
        target_gallery_path,
        {
            "stage": STAGE,
            "version": VERSION,
            "first_lost_frame": (
                first_lost
            ),
            "target_gallery": [
                {
                    "detection_id": (
                        item.detection_id
                    ),
                    "frame": item.frame,
                    "bbox_xyxy": list(
                        item.bbox
                    ),
                    "confidence": (
                        item.confidence
                    ),
                }
                for item in gallery
            ],
            "negative_gallery": [
                {
                    "detection_id": (
                        item.detection_id
                    ),
                    "frame": item.frame,
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
                ) in negative_ranked
            ],
            "embedding_contract": {
                "architecture": (
                    "osnet_x1_0"
                ),
                "image_size_hw": [
                    256,
                    128,
                ],
                "dimension": 512,
                "channel_order": (
                    "RGB"
                ),
                "mean": [
                    0.485,
                    0.456,
                    0.406,
                ],
                "std": [
                    0.229,
                    0.224,
                    0.225,
                ],
                "per_crop_l2": True,
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

    candidate_rows = [
        candidate_csv_row(item)
        for item in candidates
    ]

    empty_candidate = Candidate(
        tracklet_id="",
        start_frame=0,
        confirmation_frame=0,
        end_frame=0,
        detection_count=0,
        confirmation_span=0,
        entry_edge_distance=0.0,
        entry_score=0.0,
        mean_detection_confidence=0.0,
        mean_temporal_iou=0.0,
        target_similarity=0.0,
        target_max_similarity=0.0,
        negative_max_similarity=0.0,
        negative_margin=0.0,
        crop_consistency=0.0,
        combined_score=0.0,
        sample_detection_ids=[],
        hard_gate_pass=False,
        rejection_reasons=[],
    )

    candidate_fields = list(
        candidate_csv_row(
            empty_candidate
        )
    )

    candidate_csv_path = (
        test_dir
        / "stage2b_reentry_candidates.csv"
    )

    write_csv(
        candidate_csv_path,
        candidate_rows,
        candidate_fields,
    )

    observation_rows = [
        observation_csv_row(item)
        for item in output_rows
    ]

    observation_csv_path = (
        test_dir
        / "stage2b_frame_observations.csv"
    )

    write_csv(
        observation_csv_path,
        observation_rows,
        list(
            observation_rows[0]
        ),
    )

    timeline_path = (
        test_dir
        / "stage2b_target_timeline.json"
    )

    atomic_json(
        timeline_path,
        {
            "stage": STAGE,
            "version": VERSION,
            "target_id": "target_001",
            "test_name": (
                arguments.test_name
            ),
            "video": {
                "path": str(video),
                "sha256": (
                    video_info["sha256"]
                ),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": (
                    frame_count
                ),
            },
            "policy": asdict(
                policy
            ),
            "selection": {
                "tracklet_id": (
                    selected.tracklet_id
                    if selected
                    else None
                ),
                "reason": (
                    selection_reason
                ),
                "candidate_margin": (
                    selection_margin
                ),
                "confirmation_frame": (
                    selected
                    .confirmation_frame
                    if selected
                    else None
                ),
            },
            "segments": segments(
                output_rows,
                fps,
            ),
            "frames": output_rows,
        },
    )

    preview_path = (
        test_dir
        / "stage2b_reentry_preview.mp4"
    )

    if not arguments.no_preview:
        render_preview(
            video,
            preview_path,
            output_rows,
            fps,
            width,
            height,
        )

    states = Counter(
        item["state"]
        for item in output_rows
    )

    if selected is not None:
        decision = (
            "AUTHORIZE_MANDATORY_"
            "STAGE2B_REENTRY_"
            "VISUAL_REVIEW"
        )
    else:
        decision = (
            "SAFE_NO_AUTOMATIC_"
            "REENTRY_REQUIRE_REVIEW_"
            "OR_LATER_UI"
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
        "decision": decision,
        "test_name": (
            arguments.test_name
        ),
        "first_lost_frame": (
            first_lost
        ),
        "selection": {
            "selected_tracklet_id": (
                selected.tracklet_id
                if selected
                else None
            ),
            "reason": (
                selection_reason
            ),
            "candidate_margin": (
                selection_margin
            ),
            "selected_candidate": (
                asdict(selected)
                if selected
                else None
            ),
        },
        "counts": {
            "target_gallery": len(
                gallery
            ),
            "negative_gallery": len(
                negatives
            ),
            "geometric_tracklets": len(
                all_tracklets
            ),
            "edge_tracklets": len(
                edge_tracklets
            ),
            "scored_candidates": len(
                candidates
            ),
            "hard_gate_pass_candidates": sum(
                item.hard_gate_pass
                for item in candidates
            ),
            "state_counts": dict(
                states
            ),
        },
        "policy": asdict(
            policy
        ),
        "hashes": {
            "detections_csv": (
                sha256_file(
                    paths["detections"]
                )
            ),
            "stage2_observations": (
                sha256_file(
                    paths["observations"]
                )
            ),
            "checkpoint": (
                sha256_file(
                    checkpoint
                )
            ),
            "stage2_helper": (
                sha256_file(
                    stage2_helper_path
                )
            ),
            "v6_reid_helper": (
                sha256_file(
                    reid_helper_path
                )
            ),
            "target_embeddings": (
                sha256_file(
                    target_embedding_path
                )
            ),
            "negative_embeddings": (
                sha256_file(
                    negative_embedding_path
                )
            ),
            "candidate_embeddings": (
                sha256_file(
                    candidate_embedding_path
                )
            ),
            "candidate_csv": (
                sha256_file(
                    candidate_csv_path
                )
            ),
            "observation_csv": (
                sha256_file(
                    observation_csv_path
                )
            ),
            "timeline": (
                sha256_file(
                    timeline_path
                )
            ),
        },
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
        "device": str(device),
        "training": False,
        "threshold_search": False,
        "global_linking": False,
        "v7_logic": False,
        "certification": (
            "REQUIRES_VISUAL_REVIEW"
        ),
    }

    if (
        not arguments.no_preview
        and preview_path.is_file()
    ):
        summary["hashes"][
            "preview"
        ] = sha256_file(
            preview_path
        )

    summary_path = (
        test_dir
        / "stage2b_reentry_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    report_path = (
        test_dir
        / "stage2b_reentry_report.md"
    )

    atomic_text(
        report_path,
        (
            "# Stage 2-B Same-Shot "
            "Re-entry\n\n"
            f"- Status: "
            f"`{summary['status']}`\n"
            f"- Decision: "
            f"`{decision}`\n"
            f"- First LOST frame: "
            f"`{first_lost}`\n"
            f"- Selected tracklet: "
            f"`{summary['selection']['selected_tracklet_id']}`\n\n"
            "Sports OSNet was used only "
            "with edge-entry, persistence, "
            "negative-memory, crop-consistency, "
            "and candidate-margin gates. "
            "Open `stage2b_reentry_preview.mp4` "
            "and verify every "
            "REACQUIRED/ACTIVE box.\n"
        ),
    )

    print(
        "KickClip Target-Centric Tracking "
        "V1 Stage 2-B complete"
    )
    print(
        "Status                    : PASS"
    )
    print(
        f"Decision                  : "
        f"{decision}"
    )
    print(
        f"First LOST frame          : "
        f"{first_lost}"
    )
    print(
        "Target / negative gallery : "
        f"{len(gallery)} / "
        f"{len(negatives)}"
    )
    print(
        "Geometric / edge tracks   : "
        f"{len(all_tracklets)} / "
        f"{len(edge_tracklets)}"
    )
    print(
        "Scored / passing          : "
        f"{len(candidates)} / "
        f"{sum(item.hard_gate_pass for item in candidates)}"
    )
    print(
        "Selected tracklet         : "
        f"{selected.tracklet_id if selected else 'NONE'}"
    )

    if selected is not None:
        print(
            "Selected evidence         : "
            f"target="
            f"{selected.target_similarity:.4f}, "
            f"target_max="
            f"{selected.target_max_similarity:.4f}, "
            f"negative_margin="
            f"{selected.negative_margin:.4f}, "
            f"candidate_margin="
            f"{selection_margin:.4f}"
        )
        print(
            "Confirmation frame        : "
            f"{selected.confirmation_frame}"
        )

    print(
        f"States                    : "
        f"{dict(states)}"
    )
    print(
        "Training/linking/V7       : "
        "NONE/NONE/NONE"
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
            "Stage 2-B interrupted",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            "Stage 2-B fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)