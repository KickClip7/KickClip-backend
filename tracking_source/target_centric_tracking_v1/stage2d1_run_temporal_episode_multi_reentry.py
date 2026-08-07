#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 2-D1.

Temporal episode-aware multi-reentry.

Changes from Stage 2-D:
- candidates from different times do not compete in one global ranking;
- immediate post-loss candidates must pass a short absence gate;
- an ambiguous/rejected episode does not stop later search;
- newly automatic episodes never update target memory in this run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


STAGE = "stage2d1_temporal_episode_aware_multi_reentry"
VERSION = "target-centric-v1-stage2d1-1.0.0"
EMBEDDING_DIM = 512

OUTPUT_NAMES = (
    "stage2d1_target_gallery.json",
    "stage2d1_target_embeddings.npy",
    "stage2d1_negative_embeddings.npy",
    "stage2d1_candidate_embeddings.npy",
    "stage2d1_reentry_episodes.json",
    "stage2d1_reentry_candidates.csv",
    "stage2d1_frame_observations.csv",
    "stage2d1_target_timeline.json",
    "stage2d1_multi_reentry_preview.mp4",
    "stage2d1_summary.json",
    "stage2d1_report.md",
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
            "Temporal episode-aware iterative same-shot re-entry."
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
        "--stage2d-helper",
        type=Path,
        default=Path(
            "target_centric_tracking_v1/"
            "stage2d_run_iterative_multi_reentry.py"
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
        "--temporal-start-gap-frames",
        type=int,
        default=12,
        help=(
            "Maximum start-frame gap within one temporal candidate episode."
        ),
    )

    parser.add_argument(
        "--temporal-confirmation-gap-frames",
        type=int,
        default=12,
        help=(
            "Maximum confirmation-frame gap within one temporal "
            "candidate episode."
        ),
    )

    parser.add_argument(
        "--minimum-absence-frames",
        type=int,
        default=12,
        help=(
            "Minimum confirmed absence before a candidate may be "
            "treated as a genuine re-entry."
        ),
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-stage2d1",
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
            "Stage 2-D1 outputs exist; "
            "use --overwrite-stage2d1:\n"
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
        "*.stage2d1.tmp*"
    ):
        if path.is_file():
            path.unlink()


def temporal_clusters(
    candidates: Sequence[Any],
    start_gap: int,
    confirmation_gap: int,
) -> list[
    list[Any]
]:
    """Cluster only candidates from the same approximate entry time."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start_frame,
            candidate.confirmation_frame,
            candidate.end_frame,
            candidate.tracklet_id,
        ),
    )

    clusters: list[
        list[Any]
    ] = []

    for candidate in ordered:
        if not clusters:
            clusters.append(
                [
                    candidate
                ]
            )

            continue

        current = clusters[
            -1
        ]

        anchor_start = min(
            item.start_frame
            for item
            in current
        )

        anchor_confirmation = min(
            item.confirmation_frame
            for item
            in current
        )

        same_episode = (
            candidate.start_frame
            - anchor_start
            <= start_gap
            or candidate.confirmation_frame
            - anchor_confirmation
            <= confirmation_gap
        )

        if same_episode:
            current.append(
                candidate
            )
        else:
            clusters.append(
                [
                    candidate
                ]
            )

    return clusters


def choose_local_candidate(
    candidates: Sequence[Any],
    search_start: int,
    minimum_absence: int,
    policy: Any,
) -> tuple[
    Optional[Any],
    str,
    Optional[float],
    list[Any],
]:
    """Apply the original Stage 2-B gates inside one temporal cluster."""

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.hard_gate_pass,
            candidate.combined_score,
            candidate.target_similarity,
            candidate.negative_margin,
        ),
        reverse=True,
    )

    qualified = [
        candidate
        for candidate
        in ranked
        if (
            candidate.start_frame
            - search_start
            >= minimum_absence
        )
    ]

    if not qualified:
        return (
            None,
            "MINIMUM_ABSENCE_BEFORE_REENTRY_NOT_MET",
            None,
            ranked,
        )

    top = qualified[
        0
    ]

    if len(
        qualified
    ) > 1:
        second_score = (
            qualified[
                1
            ].combined_score
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
            "TEMPORAL_EPISODE_TOP_FAILED_HARD_GATES",
            margin,
            ranked,
        )

    if (
        margin
        < policy
        .candidate_margin_min
    ):
        return (
            None,
            "TEMPORAL_EPISODE_MARGIN_TOO_SMALL",
            margin,
            ranked,
        )

    top.selected = True

    return (
        top,
        "TEMPORAL_EPISODE_MULTI_EVIDENCE_REENTRY",
        margin,
        ranked,
    )


def candidate_row(
    cluster_index: int,
    local_rank: int,
    search_start: int,
    minimum_absence: int,
    candidate: Any,
) -> dict[
    str,
    Any,
]:
    absence = (
        candidate.start_frame
        - search_start
    )

    return {
        "temporal_cluster_index": (
            cluster_index
        ),
        "local_rank": (
            local_rank
        ),
        "tracklet_id": (
            candidate.tracklet_id
        ),
        "selected": (
            candidate.selected
        ),
        "hard_gate_pass": (
            candidate.hard_gate_pass
        ),
        "minimum_absence_gate_pass": (
            absence
            >= minimum_absence
        ),
        "absence_frames_before_candidate": (
            absence
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
        "entry_edge_distance": (
            f"{candidate.entry_edge_distance:.8f}"
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

    states = counts[
        "state_counts"
    ]

    return f"""# Stage 2-D1 Temporal Episode-Aware Multi-Reentry

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Temporal clusters: `{counts['temporal_cluster_count']}`
- New selected episodes: `{counts['new_selected_episode_count']}`
- Pending visual reviews: `{counts['pending_visual_review_episode_count']}`

## State counts

- INITIALIZING / ACTIVE: `{states.get('INITIALIZING', 0)}` / `{states.get('ACTIVE', 0)}`
- OCCLUDED / SEARCHING: `{states.get('OCCLUDED', 0)}` / `{states.get('SEARCHING', 0)}`
- AMBIGUOUS / REACQUIRED: `{states.get('AMBIGUOUS', 0)}` / `{states.get('REACQUIRED', 0)}`

The global candidate competition was removed. Candidate margins are computed
only inside the same temporal entry episode. Immediate post-loss candidates
are blocked by the minimum-absence gate, and failed episodes no longer stop
later search. New automatic episodes do not update target memory.
"""


def main() -> int:
    arguments = (
        parse_args()
    )

    started = (
        time.perf_counter()
    )

    if min(
        arguments
        .temporal_start_gap_frames,
        arguments
        .temporal_confirmation_gap_frames,
    ) < 0:
        raise ValueError(
            "Temporal gaps must be >= 0"
        )

    if (
        arguments
        .minimum_absence_frames
        < 1
    ):
        raise ValueError(
            "--minimum-absence-frames must be >= 1"
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
        "stage2d_summary": (
            test_dir
            / "stage2d_summary.json"
        ),
    }

    for (
        name,
        path,
    ) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
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

    stage2d_summary = read_json(
        paths[
            "stage2d_summary"
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
            "decision"
        )
        != (
            "AUTHORIZE_MANDATORY_"
            "STAGE2B_REENTRY_"
            "VISUAL_REVIEW"
        )
    ):
        raise RuntimeError(
            "Unexpected Stage 2-B decision"
        )

    if (
        stage2c_audit.get(
            "visual_review",
            {},
        ).get(
            "result"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 2-C visual review must be PASS"
        )

    if (
        stage2d_summary.get(
            "decision"
        )
        != (
            "SAFE_NO_ADDITIONAL_"
            "AUTOMATIC_REENTRY_"
            "REQUIRE_REVIEW_OR_UI"
        )
    ):
        raise RuntimeError(
            "Unexpected Stage 2-D decision"
        )

    prepare_outputs(
        test_dir,
        arguments
        .overwrite_stage2d1,
    )

    helper_paths = {
        "stage2": resolve(
            root,
            arguments.stage2_helper,
        ),
        "stage2b": resolve(
            root,
            arguments.stage2b_helper,
        ),
        "stage2d": resolve(
            root,
            arguments.stage2d_helper,
        ),
        "reid": resolve(
            root,
            arguments.v6_reid_helper,
        ),
    }

    for path in helper_paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    stage2 = load_module(
        "kickclip_stage2d1_stage2",
        helper_paths[
            "stage2"
        ],
    )

    stage2b = load_module(
        "kickclip_stage2d1_stage2b",
        helper_paths[
            "stage2b"
        ],
    )

    stage2d = load_module(
        "kickclip_stage2d1_stage2d",
        helper_paths[
            "stage2d"
        ],
    )

    reid = load_module(
        "kickclip_stage2d1_reid",
        helper_paths[
            "reid"
        ],
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

    frames = (
        stage2d
        .normalize_seed_frames(
            stage2b_timeline,
            frame_count,
            width,
            height,
            reviewed_search_start,
            reviewed_end,
            reviewed_tracklet_id,
        )
    )

    policy = (
        stage2b
        .Policy()
    )

    gallery = (
        stage2d
        .select_reviewed_gallery(
            frames,
            by_id,
            reviewed_confirmation,
            reviewed_end,
            policy.gallery_size,
        )
    )

    target_area = (
        statistics.median(
            stage2.area(
                detection.bbox
            )
            for detection
            in gallery
        )
    )

    negative_pool = (
        stage2b
        .select_negative_pool(
            gallery,
            by_frame,
            stage2,
            policy,
        )
    )

    checkpoint = (
        stage2b
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
        stage2b
        .discover_deep_eiou(
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
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
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
        stage2b
        .build_transform()
    )

    base_detections = {
        detection.detection_id: (
            detection
        )
        for detection
        in (
            gallery
            + negative_pool
        )
    }

    base_crops = (
        stage2b
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

    embeddings = (
        stage2b
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
        embeddings[
            detection.detection_id
        ]
        for detection
        in gallery
    ]

    target_prototype = (
        stage2b
        .l2_mean(
            target_embeddings
        )
    )

    negative_ranked = sorted(
        (
            (
                float(
                    embeddings[
                        detection.detection_id
                    ]
                    @ target_prototype
                ),
                detection,
            )
            for detection
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
        detection
        for (
            _,
            detection,
        )
        in negative_ranked
    ]

    negative_embeddings = [
        embeddings[
            detection.detection_id
        ]
        for detection
        in negatives
    ]

    target_embedding_path = (
        test_dir
        / "stage2d1_target_embeddings.npy"
    )

    negative_embedding_path = (
        test_dir
        / "stage2d1_negative_embeddings.npy"
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
        / "stage2d1_target_gallery.json"
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
                        detection
                        .detection_id
                    ),
                    "frame": (
                        detection.frame
                    ),
                    "bbox_xyxy": list(
                        detection.bbox
                    ),
                    "confidence": (
                        detection
                        .confidence
                    ),
                }
                for detection
                in gallery
            ],
            "negative_gallery": [
                {
                    "detection_id": (
                        detection
                        .detection_id
                    ),
                    "frame": (
                        detection.frame
                    ),
                    "target_similarity": (
                        similarity
                    ),
                }
                for (
                    similarity,
                    detection,
                )
                in negative_ranked
            ],
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
    )

    search_start = (
        reviewed_end
        + 1
    )

    all_tracklets = (
        stage2b
        .build_tracklets(
            by_frame,
            search_start,
            frame_count,
            target_area,
            stage2,
            policy,
        )
    )

    edge_tracklets = [
        tracklet
        for tracklet
        in all_tracklets
        if (
            len(
                tracklet.detections
            )
            >= policy
            .track_min_detections
            and stage2b
            .edge_distance(
                tracklet
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

    samples: dict[
        str,
        Any,
    ] = {}

    for tracklet in edge_tracklets:
        for detection in (
            stage2b
            .sample_tracklet(
                tracklet,
                policy,
            )
        ):
            samples[
                detection.detection_id
            ] = detection

    missing_samples = [
        detection
        for (
            detection_id,
            detection,
        )
        in samples.items()
        if detection_id
        not in embeddings
    ]

    if missing_samples:
        embeddings.update(
            stage2b
            .embed_crops(
                stage2b
                .collect_crops(
                    video,
                    missing_samples,
                    frame_count,
                    width,
                    height,
                ),
                model,
                transform,
                torch,
                device,
                arguments.batch_size,
            )
        )

    (
        candidates,
        candidate_matrix,
    ) = (
        stage2b
        .score_candidates(
            edge_tracklets,
            embeddings,
            target_embeddings,
            target_prototype,
            negative_embeddings,
            width,
            height,
            stage2,
            policy,
        )
    )

    candidate_embedding_path = (
        test_dir
        / "stage2d1_candidate_embeddings.npy"
    )

    atomic_npy(
        candidate_embedding_path,
        candidate_matrix.astype(
            np.float32,
            copy=False,
        ),
    )

    clusters = temporal_clusters(
        candidates,
        arguments
        .temporal_start_gap_frames,
        arguments
        .temporal_confirmation_gap_frames,
    )

    tracklet_by_id = {
        tracklet.tracklet_id: (
            tracklet
        )
        for tracklet
        in edge_tracklets
    }

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
            "review_status": "PASS",
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

    selected_episodes: list[
        dict[
            str,
            Any,
        ]
    ] = []

    cursor = (
        search_start
    )

    for (
        position,
        cluster,
    ) in enumerate(
        clusters,
        start=1,
    ):
        cluster = [
            candidate
            for candidate
            in cluster
            if candidate.end_frame
            >= cursor
        ]

        if not cluster:
            continue

        cluster_start = max(
            cursor,
            min(
                candidate.start_frame
                for candidate
                in cluster
            ),
        )

        later_starts = [
            min(
                candidate.start_frame
                for candidate
                in later_cluster
            )
            for later_cluster
            in clusters[
                position:
            ]
            if (
                later_cluster
                and min(
                    candidate.start_frame
                    for candidate
                    in later_cluster
                )
                > cluster_start
            )
        ]

        if later_starts:
            next_start = min(
                later_starts
            )
        else:
            next_start = (
                frame_count
            )

        (
            selected,
            reason,
            margin,
            ranked,
        ) = choose_local_candidate(
            cluster,
            search_start,
            arguments
            .minimum_absence_frames,
            policy,
        )

        for (
            rank,
            candidate,
        ) in enumerate(
            ranked,
            start=1,
        ):
            candidate_rows.append(
                candidate_row(
                    position,
                    rank,
                    search_start,
                    arguments
                    .minimum_absence_frames,
                    candidate,
                )
            )

        if ranked:
            plausible = (
                ranked[
                    0
                ]
            )
        else:
            plausible = None

        episode_index = (
            len(
                episodes
            )
            + 1
        )

        print(
            "[STAGE2D1] "
            f"cluster={position} "
            f"window={cluster_start}:"
            f"{next_start} "
            f"candidates="
            f"{len(cluster)} "
            f"decision={reason}",
            flush=True,
        )

        if selected is None:
            stage2d.mark_searching(
                frames,
                cursor,
                next_start,
                episode_index,
                reason,
                margin,
                plausible,
            )

            episodes.append(
                {
                    "episode_index": (
                        episode_index
                    ),
                    "source": (
                        "TEMPORAL_STAGE2D1"
                    ),
                    "review_status": (
                        "NO_AUTOMATIC_SELECTION"
                    ),
                    "search_start_frame": (
                        cursor
                    ),
                    "temporal_cluster_start_frame": (
                        cluster_start
                    ),
                    "temporal_cluster_end_exclusive": (
                        next_start
                    ),
                    "selected_tracklet_id": None,
                    "selection_reason": (
                        reason
                    ),
                    "candidate_margin": (
                        margin
                    ),
                    "candidate_count": (
                        len(
                            cluster
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

            cursor = (
                next_start
            )

            continue

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
                not in embeddings
            )
        ]

        if missing_selected:
            embeddings.update(
                stage2b
                .embed_crops(
                    stage2b
                    .collect_crops(
                        video,
                        missing_selected,
                        frame_count,
                        width,
                        height,
                    ),
                    model,
                    transform,
                    torch,
                    device,
                    arguments.batch_size,
                )
            )

        verification = (
            stage2b
            .verify_tracklet_frames(
                selected_tracklet,
                selected
                .confirmation_frame,
                embeddings,
                target_embeddings,
                target_prototype,
                negative_embeddings,
                policy,
            )
        )

        stage2d.mark_searching(
            frames,
            cursor,
            selected
            .confirmation_frame,
            episode_index,
            "TEMPORAL_CANDIDATE_"
            "CONFIRMATION_PENDING",
            margin,
            None,
        )

        (
            first_verified,
            last_verified,
        ) = (
            stage2d
            .apply_verified_tracklet(
                frames,
                episode_index,
                selected,
                verification,
                margin,
                policy
                .continuation_max_unverified_gap,
            )
        )

        record = {
            "episode_index": (
                episode_index
            ),
            "source": (
                "TEMPORAL_STAGE2D1"
            ),
            "review_status": (
                "PENDING_VISUAL_REVIEW"
            ),
            "search_start_frame": (
                cursor
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
                reason
            ),
            "candidate_margin": (
                margin
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

        episodes.append(
            record
        )

        selected_episodes.append(
            record
        )

        cursor = max(
            selected.end_frame
            + 1,
            last_verified
            + 1,
        )

    if (
        cursor
        < frame_count
    ):
        stage2d.mark_searching(
            frames,
            cursor,
            frame_count,
            len(
                episodes
            )
            + 1,
            "NO_LATER_TEMPORAL_"
            "REENTRY_SELECTION",
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
                "lacks bbox at frame "
                f"{item['frame_index']}"
            )

    episodes_path = (
        test_dir
        / "stage2d1_reentry_episodes.json"
    )

    atomic_json(
        episodes_path,
        {
            "stage": STAGE,
            "version": VERSION,
            "temporal_policy": {
                "start_gap_frames": (
                    arguments
                    .temporal_start_gap_frames
                ),
                "confirmation_gap_frames": (
                    arguments
                    .temporal_confirmation_gap_frames
                ),
                "minimum_absence_frames": (
                    arguments
                    .minimum_absence_frames
                ),
            },
            "episodes": (
                episodes
            ),
        },
    )

    candidate_fields = [
        "temporal_cluster_index",
        "local_rank",
        "tracklet_id",
        "selected",
        "hard_gate_pass",
        "minimum_absence_gate_pass",
        "absence_frames_before_candidate",
        "start_frame",
        "confirmation_frame",
        "end_frame",
        "detection_count",
        "entry_edge_distance",
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
    ]

    candidate_path = (
        test_dir
        / "stage2d1_reentry_candidates.csv"
    )

    write_csv(
        candidate_path,
        candidate_rows,
        candidate_fields,
    )

    observation_rows = [
        stage2d
        .observation_csv_row(
            item
        )
        for item
        in frames
    ]

    observation_path = (
        test_dir
        / "stage2d1_frame_observations.csv"
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
        / "stage2d1_target_timeline.json"
    )

    atomic_json(
        timeline_path,
        {
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
                "TEMPORAL_MULTI_REENTRY_"
                "TIMELINE_GENERATED"
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
                "temporal_start_gap_frames": (
                    arguments
                    .temporal_start_gap_frames
                ),
                "temporal_confirmation_gap_frames": (
                    arguments
                    .temporal_confirmation_gap_frames
                ),
                "minimum_absence_frames": (
                    arguments
                    .minimum_absence_frames
                ),
                "new_automatic_memory_update": (
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
        / "stage2d1_multi_reentry_preview.mp4"
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

    if selected_episodes:
        decision = (
            "AUTHORIZE_MANDATORY_"
            "STAGE2D1_TEMPORAL_"
            "REENTRY_VISUAL_REVIEW"
        )
    else:
        decision = (
            "SAFE_NO_ADDITIONAL_"
            "TEMPORAL_REENTRY_"
            "REQUIRE_REVIEW_OR_UI"
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
            "temporal_cluster_count": (
                len(
                    clusters
                )
            ),
            "new_selected_episode_count": (
                len(
                    selected_episodes
                )
            ),
            "pending_visual_review_episode_count": (
                len(
                    selected_episodes
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
        "new_selected_episodes": (
            selected_episodes
        ),
        "safety_invariants": {
            "global_candidate_competition": (
                False
            ),
            "temporal_local_candidate_margin": (
                True
            ),
            "continue_after_ambiguous_episode": (
                True
            ),
            "minimum_absence_gate": (
                True
            ),
            "threshold_relaxation": (
                False
            ),
            "new_automatic_memory_update": (
                False
            ),
            "uncertain_states_have_null_bbox": (
                True
            ),
        },
        "outputs": {
            "episodes": str(
                episodes_path
            ),
            "candidates": str(
                candidate_path
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
        "device": str(
            device
        ),
        "training": (
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
        / "stage2d1_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        test_dir
        / "stage2d1_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V1 Stage 2-D1 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        f"Decision                       : "
        f"{decision}"
    )

    print(
        f"Temporal clusters              : "
        f"{len(clusters)}"
    )

    print(
        "New selected episodes          : "
        f"{len(selected_episodes)}"
    )

    print(
        f"States                         : "
        f"{dict(states)}"
    )

    for episode in selected_episodes:
        candidate = (
            episode[
                "selected_candidate"
            ]
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
            f"{float(episode['candidate_margin']):.4f}"
        )

    print(
        "Memory update from new episodes: NONE"
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
            "Stage 2-D1 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 2-D1 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )