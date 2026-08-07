#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-B1.

Rank target-agnostic post-cut candidate tracklets using the pure pre-cut target
memory and frozen Sports OSNet embeddings.

Manual candidate IDs supplied through --evaluation-target-candidate-ids are
used only after ranking to calculate retrieval metrics. They never influence
crop selection, embeddings, similarity scores, or rank order.

This stage does not automatically select or link a target.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


STAGE = "stage3b1_rank_postcut_candidates_with_frozen_reid"
VERSION = "target-centric-v2-stage3b1-1.0.0"
EMBEDDING_DIM = 512

OUTPUT_FILES = (
    "stage3b1_ranked_candidates.csv",
    "stage3b1_selected_observations.csv",
    "stage3b1_candidate_prototypes.npy",
    "stage3b1_candidate_embedding_index.json",
    "stage3b1_ranked_contact_sheet.jpg",
    "stage3b1_evaluation.json",
    "stage3b1_summary.json",
    "stage3b1_report.md",
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
        "--evaluation-target-candidate-ids",
        nargs="+",
        required=True,
        help=(
            "Manual target tracklet IDs used only for retrieval evaluation. "
            "They do not affect automatic ranking."
        ),
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
        "--max-crops-per-tracklet",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--minimum-crop-gap",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=2,
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


def read_jsonl(
    path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as stream:
        for line_number, raw_line in enumerate(
            stream,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            value = json.loads(
                line
            )

            if not isinstance(
                value,
                dict,
            ):
                raise TypeError(
                    f"Expected object at "
                    f"{path}:{line_number}"
                )

            rows.append(
                value
            )

    return rows


def read_csv(
    path: Path,
) -> list[
    dict[
        str,
        str,
    ]
]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(
                stream
            )
        )


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

    if (
        existing
        and not overwrite
    ):
        raise FileExistsError(
            "Stage 3-B1 outputs already exist. "
            "Use --overwrite:\n"
            + "\n".join(
                str(
                    path
                )
                for path in existing
            )
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(
                path
            )

        path.unlink()


def l2_normalize_rows(
    value: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        value,
        axis=1,
        keepdims=True,
    )

    if np.any(
        norms <= 1e-12
    ):
        raise RuntimeError(
            "Zero-norm embedding row detected"
        )

    return (
        value
        / norms
    )


def l2_normalize_vector(
    value: np.ndarray,
) -> np.ndarray:
    norm = float(
        np.linalg.norm(
            value
        )
    )

    if norm <= 1e-12:
        raise RuntimeError(
            "Zero-norm embedding vector detected"
        )

    return (
        value
        / norm
    )


def bbox_area(
    bbox: Sequence[
        float
    ],
) -> float:
    return (
        max(
            0.0,
            float(
                bbox[
                    2
                ]
                - bbox[
                    0
                ]
            ),
        )
        * max(
            0.0,
            float(
                bbox[
                    3
                ]
                - bbox[
                    1
                ]
            ),
        )
    )


def select_diverse_detections(
    detections: Sequence[
        Any
    ],
    maximum_count: int,
    minimum_gap: int,
) -> list[
    Any
]:
    if not detections:
        return []

    areas = np.asarray(
        [
            bbox_area(
                detection.bbox
            )
            for detection in detections
        ],
        dtype=np.float64,
    )

    area_reference = max(
        1.0,
        float(
            np.percentile(
                areas,
                90,
            )
        ),
    )

    ranked = sorted(
        detections,
        key=lambda detection: (
            0.70
            * float(
                detection.confidence
            )
            + 0.30
            * min(
                1.0,
                bbox_area(
                    detection.bbox
                )
                / area_reference,
            ),
            int(
                detection.frame
            ),
        ),
        reverse=True,
    )

    selected: list[
        Any
    ] = []

    for gap in (
        minimum_gap,
        max(
            1,
            minimum_gap
            // 2,
        ),
        1,
    ):
        for detection in ranked:
            if any(
                str(
                    existing.detection_id
                )
                == str(
                    detection.detection_id
                )
                for existing in selected
            ):
                continue

            if any(
                abs(
                    int(
                        existing.frame
                    )
                    - int(
                        detection.frame
                    )
                )
                < gap
                for existing in selected
            ):
                continue

            selected.append(
                detection
            )

            if len(
                selected
            ) >= maximum_count:
                return sorted(
                    selected,
                    key=lambda item: int(
                        item.frame
                    ),
                )

    return sorted(
        selected,
        key=lambda item: int(
            item.frame
        ),
    )


def top_mean(
    values: np.ndarray,
    count: int,
) -> float:
    if values.size == 0:
        raise ValueError(
            "Cannot aggregate an empty array"
        )

    actual = min(
        count,
        int(
            values.size
        ),
    )

    return float(
        np.mean(
            np.sort(
                values
            )[
                -actual:
            ]
        )
    )


def compute_candidate_metrics(
    candidate_embeddings: np.ndarray,
    target_gallery: np.ndarray,
    target_prototype: np.ndarray,
    negative_gallery: np.ndarray,
) -> dict[
    str,
    Any,
]:
    candidate_embeddings = (
        l2_normalize_rows(
            candidate_embeddings.astype(
                np.float32
            )
        )
    )

    candidate_prototype = (
        l2_normalize_vector(
            np.mean(
                candidate_embeddings,
                axis=0,
            ).astype(
                np.float32
            )
        )
    )

    target_pairwise = (
        candidate_embeddings
        @ target_gallery.T
    )

    crop_target_best = np.max(
        target_pairwise,
        axis=1,
    )

    if negative_gallery.shape[
        0
    ] > 0:
        negative_pairwise = (
            candidate_embeddings
            @ negative_gallery.T
        )

        crop_negative_best = np.max(
            negative_pairwise,
            axis=1,
        )

    else:
        crop_negative_best = np.full(
            (
                candidate_embeddings.shape[
                    0
                ],
            ),
            -1.0,
            dtype=np.float32,
        )

    crop_margin = (
        crop_target_best
        - crop_negative_best
    )

    prototype_target_similarity = float(
        candidate_prototype
        @ target_prototype
    )

    prototype_gallery_best = float(
        np.max(
            candidate_prototype
            @ target_gallery.T
        )
    )

    # Frozen retrieval ordering policy:
    # rank by the mean of the three strongest
    # crop-to-target-gallery similarities.
    #
    # Manual evaluation IDs are not read here.
    retrieval_score = top_mean(
        crop_target_best,
        3,
    )

    return {
        "prototype": (
            candidate_prototype
        ),
        "retrieval_score": (
            retrieval_score
        ),
        "prototype_target_similarity": (
            prototype_target_similarity
        ),
        "prototype_gallery_best": (
            prototype_gallery_best
        ),
        "crop_target_best_max": float(
            np.max(
                crop_target_best
            )
        ),
        "crop_target_best_median": float(
            np.median(
                crop_target_best
            )
        ),
        "crop_target_best_min": float(
            np.min(
                crop_target_best
            )
        ),
        "crop_negative_best_max": float(
            np.max(
                crop_negative_best
            )
        ),
        "crop_margin_mean": float(
            np.mean(
                crop_margin
            )
        ),
        "crop_margin_median": float(
            np.median(
                crop_margin
            )
        ),
        "crop_margin_min": float(
            np.min(
                crop_margin
            )
        ),
        "positive_margin_support_ratio": float(
            np.mean(
                crop_margin
                > 0.0
            )
        ),
    }


def fit_image(
    image: np.ndarray,
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
        image.shape[
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
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
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


def make_ranked_contact_sheet(
    ranked_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    strip_dir: Path,
    output_path: Path,
    columns: int,
) -> None:
    if not ranked_rows:
        raise RuntimeError(
            "No ranked candidates available"
        )

    cards: list[
        np.ndarray
    ] = []

    card_width = 620
    card_height = 320
    header_height = 66

    for row in ranked_rows:
        candidate_id = str(
            row[
                "candidate_id"
            ]
        )

        strip_path = (
            strip_dir
            / f"{candidate_id}.jpg"
        )

        if not strip_path.is_file():
            raise FileNotFoundError(
                strip_path
            )

        strip = cv2.imread(
            str(
                strip_path
            )
        )

        if strip is None:
            raise RuntimeError(
                f"Cannot read strip: {strip_path}"
            )

        card = np.zeros(
            (
                card_height,
                card_width,
                3,
            ),
            dtype=np.uint8,
        )

        fitted = fit_image(
            strip,
            card_width,
            card_height
            - header_height,
        )

        card[
            header_height:,
            :,
        ] = fitted

        line1 = (
            f"RANK #{row['retrieval_rank']} "
            f"{candidate_id} "
            f"score="
            f"{float(row['retrieval_score']):.4f}"
        )

        line2 = (
            "proto="
            f"{float(row['prototype_target_similarity']):.4f} "
            "margin_med="
            f"{float(row['crop_margin_median']):.4f} "
            "support="
            f"{float(row['positive_margin_support_ratio']):.3f}"
        )

        cv2.putText(
            card,
            line1,
            (
                8,
                25,
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

        cv2.putText(
            card,
            line2,
            (
                8,
                52,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        cards.append(
            card
        )

    row_count = int(
        math.ceil(
            len(
                cards
            )
            / columns
        )
    )

    blank = np.zeros_like(
        cards[
            0
        ]
    )

    cards.extend(
        [
            blank
        ]
        * (
            row_count
            * columns
            - len(
                cards
            )
        )
    )

    sheet = np.vstack(
        [
            np.hstack(
                cards[
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
            f"Cannot write contact sheet: "
            f"{output_path}"
        )


def build_evaluation(
    ranked_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    target_candidate_ids: Sequence[str],
    reviewer: str,
    review_note: str,
) -> dict[
    str,
    Any,
]:
    normalized_targets = list(
        dict.fromkeys(
            str(
                value
            )
            for value in target_candidate_ids
        )
    )

    rank_by_candidate = {
        str(
            row[
                "candidate_id"
            ]
        ): int(
            row[
                "retrieval_rank"
            ]
        )
        for row in ranked_rows
    }

    missing = [
        candidate_id
        for candidate_id in normalized_targets
        if candidate_id
        not in rank_by_candidate
    ]

    if missing:
        raise RuntimeError(
            "Evaluation-only target IDs are absent "
            "from Stage 3-B0 candidates: "
            + ", ".join(
                missing
            )
        )

    target_ranks = {
        candidate_id: (
            rank_by_candidate[
                candidate_id
            ]
        )
        for candidate_id in normalized_targets
    }

    target_count = len(
        normalized_targets
    )

    fragment_recall_at: dict[
        str,
        float,
    ] = {}

    any_target_at: dict[
        str,
        bool,
    ] = {}

    cutoffs = list(
        dict.fromkeys(
            (
                1,
                2,
                3,
                5,
                len(
                    ranked_rows
                ),
            )
        )
    )

    for cutoff in cutoffs:
        key = (
            f"at_{cutoff}"
        )

        retrieved = sum(
            rank
            <= cutoff
            for rank in target_ranks.values()
        )

        fragment_recall_at[
            key
        ] = (
            retrieved
            / target_count
        )

        any_target_at[
            key
        ] = (
            retrieved
            > 0
        )

    return {
        "label_source": (
            "USER_VISUAL_REVIEW"
        ),
        "label_usage": (
            "EVALUATION_ONLY_"
            "NOT_USED_FOR_RANKING"
        ),
        "reviewer": (
            reviewer
        ),
        "review_note": (
            review_note
        ),
        "target_candidate_ids": (
            normalized_targets
        ),
        "candidate_generation_recall": (
            1.0
        ),
        "target_ranks": (
            target_ranks
        ),
        "best_target_rank": min(
            target_ranks.values()
        ),
        "worst_target_rank": max(
            target_ranks.values()
        ),
        "fragment_recall": (
            fragment_recall_at
        ),
        "any_target_retrieved": (
            any_target_at
        ),
        "ranking_policy_read_manual_labels": (
            False
        ),
    }


def build_report(
    summary: Mapping[
        str,
        Any,
    ],
) -> str:
    evaluation = summary[
        "evaluation"
    ]

    top_candidates = summary[
        "top_candidates"
    ]

    lines = [
        (
            "# KickClip Target-Centric Tracking V2 "
            "— Stage 3-B1"
        ),
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        (
            "- Ranked candidate count: "
            f"`{summary['counts']['ranked_candidate_count']}`"
        ),
        (
            "- Evaluation target fragments: "
            f"`{evaluation['target_candidate_ids']}`"
        ),
        (
            "- Evaluation target ranks: "
            f"`{evaluation['target_ranks']}`"
        ),
        (
            "- Best target rank: "
            f"`{evaluation['best_target_rank']}`"
        ),
        (
            "- Worst target rank: "
            f"`{evaluation['worst_target_rank']}`"
        ),
        "",
        "## Automatic top candidates",
        "",
    ]

    for row in top_candidates:
        lines.append(
            f"- rank {row['retrieval_rank']}: "
            f"`{row['candidate_id']}` "
            f"score={row['retrieval_score']:.6f}, "
            f"prototype="
            f"{row['prototype_target_similarity']:.6f}, "
            f"median margin="
            f"{row['crop_margin_median']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Safety interpretation",
            "",
            (
                "The manual target candidate IDs were applied "
                "only after automatic ranking to calculate "
                "retrieval metrics."
            ),
            (
                "They did not influence crop selection, "
                "Sports OSNet embeddings, similarity scores, "
                "or rank order."
            ),
            "",
            (
                "This stage is retrieval-only. "
                "No target candidate is automatically selected, "
                "no tracklets are merged, and no cross-shot "
                "target link is created."
            ),
        ]
    )

    return (
        "\n".join(
            lines
        )
        + "\n"
    )


def main() -> int:
    arguments = parse_args()

    if arguments.batch_size < 1:
        raise ValueError(
            "--batch-size must be positive"
        )

    if (
        arguments.max_crops_per_tracklet
        < 1
    ):
        raise ValueError(
            "--max-crops-per-tracklet "
            "must be positive"
        )

    if (
        arguments.minimum_crop_gap
        < 1
    ):
        raise ValueError(
            "--minimum-crop-gap must be positive"
        )

    if (
        arguments.contact_sheet_columns
        < 1
    ):
        raise ValueError(
            "--contact-sheet-columns "
            "must be positive"
        )

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

    if not output_dir.is_dir():
        raise FileNotFoundError(
            output_dir
        )

    prepare_outputs(
        output_dir,
        arguments.overwrite,
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

    stage3b0_path = (
        output_dir
        / "stage3b0_summary.json"
    )

    tracklets_path = (
        output_dir
        / "stage3b0_tracklets.jsonl"
    )

    assignments_path = (
        output_dir
        / "stage3b0_tracklet_assignments.csv"
    )

    for path in (
        stage3a1_path,
        stage3a2_path,
        stage3a2r2_path,
        pure_memory_path,
        stage3b0_path,
        tracklets_path,
        assignments_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
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

    stage3b0 = read_json(
        stage3b0_path
    )

    tracklets = read_jsonl(
        tracklets_path
    )

    assignments = read_csv(
        assignments_path
    )

    if (
        stage3a2r2.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 3-A2R2 must be PASS"
        )

    if (
        stage3b0.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 3-B0 must be PASS"
        )

    if (
        stage3b0.get(
            "decision"
        )
        != (
            "AUTHORIZE_MANDATORY_"
            "STAGE3B0_CANDIDATE_RECALL_"
            "VISUAL_REVIEW"
        )
    ):
        raise RuntimeError(
            "Unexpected Stage 3-B0 decision"
        )

    target_embedding_path = Path(
        str(
            pure_memory[
                "embeddings"
            ][
                "target_path"
            ]
        )
    ).resolve()

    target_prototype_path = Path(
        str(
            pure_memory[
                "embeddings"
            ][
                "target_prototype_path"
            ]
        )
    ).resolve()

    negative_embedding_path = Path(
        str(
            pure_memory[
                "embeddings"
            ][
                "negative_path"
            ]
        )
    ).resolve()

    for path in (
        target_embedding_path,
        target_prototype_path,
        negative_embedding_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    target_gallery = np.load(
        target_embedding_path,
        allow_pickle=False,
    ).astype(
        np.float32
    )

    target_prototype = np.load(
        target_prototype_path,
        allow_pickle=False,
    ).astype(
        np.float32
    )

    negative_gallery = np.load(
        negative_embedding_path,
        allow_pickle=False,
    ).astype(
        np.float32
    )

    if (
        target_gallery.ndim
        != 2
        or target_gallery.shape[
            1
        ]
        != EMBEDDING_DIM
    ):
        raise RuntimeError(
            "Unexpected target gallery shape: "
            f"{target_gallery.shape}"
        )

    if target_prototype.shape != (
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            "Unexpected target prototype shape: "
            f"{target_prototype.shape}"
        )

    if (
        negative_gallery.ndim
        != 2
        or negative_gallery.shape[
            1
        ]
        != EMBEDDING_DIM
    ):
        raise RuntimeError(
            "Unexpected negative gallery shape: "
            f"{negative_gallery.shape}"
        )

    target_gallery = l2_normalize_rows(
        target_gallery
    )

    target_prototype = l2_normalize_vector(
        target_prototype
    )

    if negative_gallery.shape[
        0
    ] > 0:
        negative_gallery = (
            l2_normalize_rows(
                negative_gallery
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
            "Video is missing or changed"
        )

    frame_count = int(
        stage3a1[
            "video"
        ][
            "frame_count"
        ]
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

    detections_path = Path(
        str(
            stage3a2[
                "phase1_source"
            ][
                "detections"
            ]
        )
    ).resolve()

    if (
        not detections_path.is_file()
        or sha256_file(
            detections_path
        )
        != stage3a2[
            "phase1_source"
        ][
            "detections_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen detections are missing "
            "or changed"
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
        stage2_helper_path,
        stage2b_helper_path,
        reid_helper_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    stage2 = load_module(
        "kickclip_stage3b1_stage2",
        stage2_helper_path,
    )

    stage2b = load_module(
        "kickclip_stage3b1_stage2b",
        stage2b_helper_path,
    )

    reid = load_module(
        "kickclip_stage3b1_reid",
        reid_helper_path,
    )

    (
        by_frame,
        _,
    ) = stage2.load_detections(
        detections_path,
        frame_count,
        width,
        height,
    )

    detection_by_id = {
        str(
            detection.detection_id
        ): detection
        for detections in by_frame.values()
        for detection in detections
    }

    assignments_by_candidate: dict[
        str,
        list[
            Any
        ],
    ] = {}

    selected_observation_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for row in assignments:
        candidate_id = str(
            row[
                "candidate_id"
            ]
        )

        detection_id = str(
            row[
                "detection_id"
            ]
        )

        detection = detection_by_id.get(
            detection_id
        )

        if detection is None:
            raise RuntimeError(
                "Assignment detection is absent "
                "from frozen detections: "
                f"{detection_id}"
            )

        assignments_by_candidate.setdefault(
            candidate_id,
            [],
        ).append(
            detection
        )

    candidate_ids = [
        str(
            row[
                "candidate_id"
            ]
        )
        for row in tracklets
    ]

    if len(
        candidate_ids
    ) != len(
        set(
            candidate_ids
        )
    ):
        raise RuntimeError(
            "Duplicate candidate ID in "
            "Stage 3-B0 tracklets"
        )

    selected_by_candidate: dict[
        str,
        list[
            Any
        ],
    ] = {}

    all_selected_detections: dict[
        str,
        Any,
    ] = {}

    for candidate_id in candidate_ids:
        detections = assignments_by_candidate.get(
            candidate_id,
            [],
        )

        if not detections:
            raise RuntimeError(
                "Candidate has no assignments: "
                f"{candidate_id}"
            )

        selected = select_diverse_detections(
            detections,
            arguments.max_crops_per_tracklet,
            arguments.minimum_crop_gap,
        )

        if not selected:
            raise RuntimeError(
                "Candidate has no selected crops: "
                f"{candidate_id}"
            )

        selected_by_candidate[
            candidate_id
        ] = selected

        for selection_rank, detection in enumerate(
            selected,
            start=1,
        ):
            all_selected_detections[
                str(
                    detection.detection_id
                )
            ] = detection

            selected_observation_rows.append(
                {
                    "candidate_id": (
                        candidate_id
                    ),
                    "selection_rank_within_candidate": (
                        selection_rank
                    ),
                    "frame_index": int(
                        detection.frame
                    ),
                    "detection_id": str(
                        detection.detection_id
                    ),
                    "confidence": float(
                        detection.confidence
                    ),
                    "x1": float(
                        detection.bbox[
                            0
                        ]
                    ),
                    "y1": float(
                        detection.bbox[
                            1
                        ]
                    ),
                    "x2": float(
                        detection.bbox[
                            2
                        ]
                    ),
                    "y2": float(
                        detection.bbox[
                            3
                        ]
                    ),
                }
            )

    checkpoint = Path(
        str(
            pure_memory[
                "model"
            ][
                "checkpoint"
            ]
        )
    ).resolve()

    expected_checkpoint_hash = str(
        pure_memory[
            "model"
        ][
            "checkpoint_sha256"
        ]
    )

    if (
        not checkpoint.is_file()
        or sha256_file(
            checkpoint
        ).lower()
        != expected_checkpoint_hash.lower()
    ):
        raise RuntimeError(
            "Sports OSNet checkpoint is "
            "missing or changed"
        )

    (
        deep_eiou_root,
        models,
        reid_root,
    ) = stage2b.discover_deep_eiou(
        root,
        reid,
        arguments.deep_eiou_root,
    )

    import torch

    if arguments.device == "auto":
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

    crops = stage2b.collect_crops(
        video,
        list(
            all_selected_detections.values()
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

    automatic_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    prototype_by_candidate: dict[
        str,
        np.ndarray,
    ] = {}

    tracklet_contract_by_id = {
        str(
            row[
                "candidate_id"
            ]
        ): row
        for row in tracklets
    }

    for candidate_id in candidate_ids:
        selected = selected_by_candidate[
            candidate_id
        ]

        candidate_matrix = np.stack(
            [
                embeddings[
                    str(
                        detection.detection_id
                    )
                ]
                for detection in selected
            ]
        ).astype(
            np.float32
        )

        metrics = compute_candidate_metrics(
            candidate_matrix,
            target_gallery,
            target_prototype,
            negative_gallery,
        )

        prototype_by_candidate[
            candidate_id
        ] = metrics.pop(
            "prototype"
        )

        tracklet = tracklet_contract_by_id[
            candidate_id
        ]

        automatic_rows.append(
            {
                "candidate_id": (
                    candidate_id
                ),
                "start_frame": int(
                    tracklet[
                        "start_frame"
                    ]
                ),
                "end_frame_inclusive": int(
                    tracklet[
                        "end_frame_inclusive"
                    ]
                ),
                "tracklet_detection_count": int(
                    tracklet[
                        "detection_count"
                    ]
                ),
                "embedded_crop_count": int(
                    candidate_matrix.shape[
                        0
                    ]
                ),
                **metrics,
            }
        )

    automatic_rows.sort(
        key=lambda row: (
            float(
                row[
                    "retrieval_score"
                ]
            ),
            float(
                row[
                    "prototype_target_similarity"
                ]
            ),
            float(
                row[
                    "crop_margin_median"
                ]
            ),
            float(
                row[
                    "positive_margin_support_ratio"
                ]
            ),
        ),
        reverse=True,
    )

    for retrieval_rank, row in enumerate(
        automatic_rows,
        start=1,
    ):
        row[
            "retrieval_rank"
        ] = retrieval_rank

    ranked_rows = sorted(
        automatic_rows,
        key=lambda row: int(
            row[
                "retrieval_rank"
            ]
        ),
    )

    prototype_matrix = np.stack(
        [
            prototype_by_candidate[
                str(
                    row[
                        "candidate_id"
                    ]
                )
            ]
            for row in ranked_rows
        ]
    ).astype(
        np.float32
    )

    prototype_path = (
        output_dir
        / "stage3b1_candidate_prototypes.npy"
    )

    atomic_npy(
        prototype_path,
        prototype_matrix,
    )

    prototype_index = {
        "stage": (
            STAGE
        ),
        "version": (
            VERSION
        ),
        "candidate_order": [
            str(
                row[
                    "candidate_id"
                ]
            )
            for row in ranked_rows
        ],
        "shape": list(
            prototype_matrix.shape
        ),
        "embedding_dimension": (
            EMBEDDING_DIM
        ),
    }

    prototype_index_path = (
        output_dir
        / "stage3b1_candidate_embedding_index.json"
    )

    atomic_json(
        prototype_index_path,
        prototype_index,
    )

    ranked_csv_path = (
        output_dir
        / "stage3b1_ranked_candidates.csv"
    )

    write_csv(
        ranked_csv_path,
        ranked_rows,
        list(
            ranked_rows[
                0
            ]
        ),
    )

    selected_csv_path = (
        output_dir
        / "stage3b1_selected_observations.csv"
    )

    write_csv(
        selected_csv_path,
        selected_observation_rows,
        list(
            selected_observation_rows[
                0
            ]
        ),
    )

    strip_dir = Path(
        str(
            stage3b0[
                "outputs"
            ][
                "tracklet_strips"
            ]
        )
    ).resolve()

    if not strip_dir.is_dir():
        raise FileNotFoundError(
            strip_dir
        )

    contact_sheet_path = (
        output_dir
        / "stage3b1_ranked_contact_sheet.jpg"
    )

    make_ranked_contact_sheet(
        ranked_rows,
        strip_dir,
        contact_sheet_path,
        arguments.contact_sheet_columns,
    )

    evaluation = build_evaluation(
        ranked_rows,
        arguments.evaluation_target_candidate_ids,
        arguments.reviewer,
        arguments.review_note,
    )

    evaluation_path = (
        output_dir
        / "stage3b1_evaluation.json"
    )

    atomic_json(
        evaluation_path,
        evaluation,
    )

    top_candidates = [
        {
            "retrieval_rank": int(
                row[
                    "retrieval_rank"
                ]
            ),
            "candidate_id": str(
                row[
                    "candidate_id"
                ]
            ),
            "retrieval_score": float(
                row[
                    "retrieval_score"
                ]
            ),
            "prototype_target_similarity": float(
                row[
                    "prototype_target_similarity"
                ]
            ),
            "crop_margin_median": float(
                row[
                    "crop_margin_median"
                ]
            ),
            "positive_margin_support_ratio": float(
                row[
                    "positive_margin_support_ratio"
                ]
            ),
        }
        for row in ranked_rows[
            :
            min(
                5,
                len(
                    ranked_rows
                ),
            )
        ]
    ]

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
            "STAGE3B1_RETRIEVAL_"
            "RANKING_REVIEW"
        ),
        "test_name": (
            test_name
        ),
        "ranking_policy": {
            "primary_score": (
                "TOP3_MEAN_CROP_TO_"
                "TARGET_GALLERY_COSINE"
            ),
            "tie_breakers": [
                (
                    "TRACKLET_PROTOTYPE_TO_"
                    "TARGET_PROTOTYPE_COSINE"
                ),
                (
                    "MEDIAN_TARGET_MINUS_"
                    "NEGATIVE_MARGIN"
                ),
                (
                    "POSITIVE_MARGIN_"
                    "SUPPORT_RATIO"
                ),
            ],
            "manual_target_candidate_ids_used": (
                False
            ),
            "threshold_selection_performed": (
                False
            ),
            "calibration_status": (
                "DEVELOPMENT_ONLY_NOT_"
                "GENERALIZATION_VALIDATED"
            ),
        },
        "counts": {
            "ranked_candidate_count": (
                len(
                    ranked_rows
                )
            ),
            "embedded_detection_count": (
                len(
                    all_selected_detections
                )
            ),
            "target_gallery_size": int(
                target_gallery.shape[
                    0
                ]
            ),
            "negative_gallery_size": int(
                negative_gallery.shape[
                    0
                ]
            ),
            "evaluation_target_fragment_count": (
                len(
                    evaluation[
                        "target_candidate_ids"
                    ]
                )
            ),
        },
        "evaluation": (
            evaluation
        ),
        "top_candidates": (
            top_candidates
        ),
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
        "inputs": {
            "stage3a2r2_summary": str(
                stage3a2r2_path
            ),
            "pure_target_memory": str(
                pure_memory_path
            ),
            "target_embeddings": str(
                target_embedding_path
            ),
            "target_embeddings_sha256": (
                sha256_file(
                    target_embedding_path
                )
            ),
            "target_prototype": str(
                target_prototype_path
            ),
            "target_prototype_sha256": (
                sha256_file(
                    target_prototype_path
                )
            ),
            "negative_embeddings": str(
                negative_embedding_path
            ),
            "negative_embeddings_sha256": (
                sha256_file(
                    negative_embedding_path
                )
            ),
            "stage3b0_summary": str(
                stage3b0_path
            ),
            "tracklets": str(
                tracklets_path
            ),
            "assignments": str(
                assignments_path
            ),
            "detections": str(
                detections_path
            ),
        },
        "outputs": {
            "ranked_candidates": str(
                ranked_csv_path
            ),
            "selected_observations": str(
                selected_csv_path
            ),
            "candidate_prototypes": str(
                prototype_path
            ),
            "candidate_embedding_index": str(
                prototype_index_path
            ),
            "ranked_contact_sheet": str(
                contact_sheet_path
            ),
            "evaluation": str(
                evaluation_path
            ),
        },
        "safety_invariants": {
            "manual_labels_used_for_candidate_generation": (
                False
            ),
            "manual_labels_used_for_ranking": (
                False
            ),
            "manual_labels_used_for_evaluation_only": (
                True
            ),
            "target_candidate_automatically_selected": (
                False
            ),
            "tracklets_merged": (
                False
            ),
            "cross_shot_linking_performed": (
                False
            ),
            "target_memory_updated": (
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
        / "stage3b1_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3b1_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric Tracking V2 "
        "Stage 3-B1 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        "Decision                       : "
        "AUTHORIZE_MANDATORY_"
        "STAGE3B1_RETRIEVAL_"
        "RANKING_REVIEW"
    )

    print(
        f"Ranked candidate tracklets     : "
        f"{len(ranked_rows)}"
    )

    print(
        f"Embedded post-cut crops        : "
        f"{len(all_selected_detections)}"
    )

    print(
        "Evaluation target fragments    : "
        + ", ".join(
            evaluation[
                "target_candidate_ids"
            ]
        )
    )

    print(
        f"Evaluation target ranks        : "
        f"{evaluation['target_ranks']}"
    )

    print(
        f"Best / worst target rank       : "
        f"{evaluation['best_target_rank']} / "
        f"{evaluation['worst_target_rank']}"
    )

    print(
        "Automatic target selection     : NONE"
    )

    print(
        "Tracklet merge                  : NONE"
    )

    print(
        "Cross-shot linking             : NONE"
    )

    print(
        "Threshold search               : NONE"
    )

    print(
        "Top automatic candidates:"
    )

    for row in top_candidates:
        print(
            f"  #{row['retrieval_rank']} "
            f"{row['candidate_id']} "
            f"score={row['retrieval_score']:.6f} "
            f"proto="
            f"{row['prototype_target_similarity']:.6f} "
            f"margin_med="
            f"{row['crop_margin_median']:.6f} "
            f"support="
            f"{row['positive_margin_support_ratio']:.3f}"
        )

    print(
        f"Ranked contact sheet           : "
        f"{contact_sheet_path}"
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
            "Stage 3-B1 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-B1 fatal error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )