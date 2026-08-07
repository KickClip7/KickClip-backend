#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-B3.

Record an explicit user-confirmed target candidate after an AMBIGUOUS
cross-shot retrieval decision and select a stable anchor observation inside
that confirmed candidate.

The confirmed candidate ID is a manual fallback decision. The anchor frame is
selected deterministically only from observations belonging to that confirmed
candidate.

This stage does not:
- automatically choose a candidate;
- merge post-cut fragments;
- use the later target fragment as an operational link;
- run ReID inference;
- update target memory;
- execute Phase 1 tracking;
- create a silent automatic cross-shot link.

The resulting anchor requires visual review before Stage 3-C.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np


STAGE = "stage3b3_confirm_user_selected_cross_shot_anchor"
VERSION = "target-centric-v2-stage3b3-1.0.0"

OUTPUT_FILES = (
    "stage3b3_user_confirmation.json",
    "stage3b3_anchor_candidates.csv",
    "stage3b3_confirmed_anchor_review.jpg",
    "stage3b3_summary.json",
    "stage3b3_report.md",
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
        "--confirmed-candidate-id",
        required=True,
    )

    parser.add_argument(
        "--review-status",
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
        "--maximum-confirmable-rank",
        type=int,
        default=3,
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
                    f"Expected JSON object at "
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
        path.name + ".tmp"
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
        for name in OUTPUT_FILES
        if (
            output_dir / name
        ).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Stage 3-B3 outputs already exist. "
            "Use --overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(
                path
            )

        path.unlink()


def finite_float(
    value: Any,
    field_name: str,
) -> float:
    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"Non-finite {field_name}: {result}"
        )

    return result


def bbox_from_row(
    row: Mapping[
        str,
        Any,
    ],
) -> list[float]:
    return [
        finite_float(
            row["x1"],
            "x1",
        ),
        finite_float(
            row["y1"],
            "y1",
        ),
        finite_float(
            row["x2"],
            "x2",
        ),
        finite_float(
            row["y2"],
            "y2",
        ),
    ]


def bbox_area(
    bbox: Sequence[float],
) -> float:
    return (
        max(
            0.0,
            float(
                bbox[2] - bbox[0]
            ),
        )
        * max(
            0.0,
            float(
                bbox[3] - bbox[1]
            ),
        )
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

    if union <= 0.0:
        return 0.0

    return (
        intersection
        / union
    )


def normalized_interior_score(
    index: int,
    count: int,
) -> float:
    if count <= 1:
        return 0.0

    distance_to_edge = min(
        index,
        count - 1 - index,
    )

    maximum_distance = max(
        1.0,
        (
            count - 1
        )
        / 2.0,
    )

    return float(
        np.clip(
            distance_to_edge
            / maximum_distance,
            0.0,
            1.0,
        )
    )


def choose_anchor(
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> tuple[
    dict[
        str,
        Any,
    ],
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    if len(
        rows
    ) < 3:
        raise RuntimeError(
            "A confirmed tracklet must contain "
            "at least three observations"
        )

    ordered = sorted(
        (
            dict(
                row
            )
            for row in rows
        ),
        key=lambda row: int(
            row["frame_index"]
        ),
    )

    areas = np.asarray(
        [
            bbox_area(
                bbox_from_row(
                    row
                )
            )
            for row in ordered
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

    scored: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, row in enumerate(
        ordered
    ):
        bbox = bbox_from_row(
            row
        )

        confidence = finite_float(
            row["confidence"],
            "confidence",
        )

        area_score = min(
            1.0,
            bbox_area(
                bbox
            )
            / area_reference,
        )

        neighbor_ious: list[
            float
        ] = []

        if index > 0:
            previous_bbox = bbox_from_row(
                ordered[
                    index - 1
                ]
            )

            neighbor_ious.append(
                bbox_iou(
                    previous_bbox,
                    bbox,
                )
            )

        if index + 1 < len(
            ordered
        ):
            next_bbox = bbox_from_row(
                ordered[
                    index + 1
                ]
            )

            neighbor_ious.append(
                bbox_iou(
                    bbox,
                    next_bbox,
                )
            )

        continuity_score = (
            float(
                np.mean(
                    neighbor_ious
                )
            )
            if neighbor_ious
            else 0.0
        )

        interior_score = (
            normalized_interior_score(
                index,
                len(
                    ordered
                ),
            )
        )

        previous_gap = (
            int(
                row["frame_index"]
            )
            - int(
                ordered[
                    index - 1
                ][
                    "frame_index"
                ]
            )
            if index > 0
            else 0
        )

        next_gap = (
            int(
                ordered[
                    index + 1
                ][
                    "frame_index"
                ]
            )
            - int(
                row["frame_index"]
            )
            if index + 1
            < len(
                ordered
            )
            else 0
        )

        gap_penalty = (
            0.10
            if max(
                previous_gap,
                next_gap,
            )
            > 2
            else 0.0
        )

        anchor_quality_score = (
            0.45 * confidence
            + 0.25 * area_score
            + 0.20 * continuity_score
            + 0.10 * interior_score
            - gap_penalty
        )

        scored.append(
            {
                "candidate_id": str(
                    row["candidate_id"]
                ),
                "frame_index": int(
                    row["frame_index"]
                ),
                "detection_id": str(
                    row["detection_id"]
                ),
                "confidence": (
                    confidence
                ),
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "bbox_area": (
                    bbox_area(
                        bbox
                    )
                ),
                "area_score": (
                    area_score
                ),
                "continuity_score": (
                    continuity_score
                ),
                "interior_score": (
                    interior_score
                ),
                "previous_gap": (
                    previous_gap
                ),
                "next_gap": (
                    next_gap
                ),
                "gap_penalty": (
                    gap_penalty
                ),
                "anchor_quality_score": (
                    anchor_quality_score
                ),
                "selected_as_anchor": (
                    False
                ),
            }
        )

    ranked = sorted(
        scored,
        key=lambda row: (
            float(
                row[
                    "anchor_quality_score"
                ]
            ),
            float(
                row[
                    "confidence"
                ]
            ),
            float(
                row[
                    "continuity_score"
                ]
            ),
            -abs(
                int(
                    row[
                        "frame_index"
                    ]
                )
                - int(
                    np.median(
                        [
                            int(
                                item[
                                    "frame_index"
                                ]
                            )
                            for item in scored
                        ]
                    )
                )
            ),
        ),
        reverse=True,
    )

    selected = dict(
        ranked[0]
    )

    for row in scored:
        if (
            row["detection_id"]
            == selected["detection_id"]
        ):
            row[
                "selected_as_anchor"
            ] = True

    return (
        selected,
        scored,
    )


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


def fit_tile(
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
        image.shape[:2]
    )

    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(
        1,
        int(
            round(
                source_width * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                source_height * scale
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
        width - resized_width
    ) // 2

    offset_y = (
        height - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y + resized_height,
        offset_x:
        offset_x + resized_width,
    ] = resized

    return canvas


def draw_candidate_box(
    frame: np.ndarray,
    row: Mapping[
        str,
        Any,
    ],
    label: str,
) -> None:
    x1, y1, x2, y2 = [
        int(
            round(
                finite_float(
                    row[key],
                    key,
                )
            )
        )
        for key in (
            "x1",
            "y1",
            "x2",
            "y2",
        )
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
        (
            0,
            255,
            0,
        ),
        3,
    )

    cv2.putText(
        frame,
        label,
        (
            max(
                2,
                x1,
            ),
            max(
                22,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            0,
            255,
            0,
        ),
        2,
        cv2.LINE_AA,
    )


def label_tile(
    tile: np.ndarray,
    line1: str,
    line2: str,
) -> None:
    cv2.rectangle(
        tile,
        (
            0,
            0,
        ),
        (
            tile.shape[1] - 1,
            60,
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
        line1,
        (
            7,
            24,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        tile,
        line2,
        (
            7,
            49,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )


def make_review_sheet(
    video: Path,
    candidate_id: str,
    candidate_rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    anchor: Mapping[
        str,
        Any,
    ],
    strip_path: Path,
    output_path: Path,
) -> None:
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

    ordered = sorted(
        candidate_rows,
        key=lambda row: int(
            row["frame_index"]
        ),
    )

    anchor_index = next(
        index
        for index, row in enumerate(
            ordered
        )
        if str(
            row["detection_id"]
        )
        == str(
            anchor["detection_id"]
        )
    )

    previous_index = max(
        0,
        anchor_index - 2,
    )

    next_index = min(
        len(
            ordered
        )
        - 1,
        anchor_index + 2,
    )

    previous_row = ordered[
        previous_index
    ]

    anchor_row = ordered[
        anchor_index
    ]

    next_row = ordered[
        next_index
    ]

    tile_width = 640
    tile_height = 360

    strip_tile = fit_tile(
        strip,
        tile_width,
        tile_height,
    )

    label_tile(
        strip_tile,
        (
            f"USER-CONFIRMED CANDIDATE "
            f"{candidate_id}"
        ),
        (
            "Candidate identity confirmed manually; "
            "later fragments are not merged here."
        ),
    )

    frame_tiles: list[
        np.ndarray
    ] = []

    for title, row in (
        (
            "PREVIOUS OBSERVATION",
            previous_row,
        ),
        (
            "SELECTED ANCHOR",
            anchor_row,
        ),
        (
            "NEXT OBSERVATION",
            next_row,
        ),
    ):
        frame_index = int(
            row["frame_index"]
        )

        frame = read_frame(
            video,
            frame_index,
        )

        draw_candidate_box(
            frame,
            row,
            (
                f"{candidate_id} "
                f"frame={frame_index}"
            ),
        )

        tile = fit_tile(
            frame,
            tile_width,
            tile_height,
        )

        quality = (
            float(
                anchor[
                    "anchor_quality_score"
                ]
            )
            if title
            == "SELECTED ANCHOR"
            else float(
                row.get(
                    "anchor_quality_score",
                    0.0,
                )
            )
        )

        label_tile(
            tile,
            title,
            (
                f"frame={frame_index} "
                f"confidence="
                f"{float(row['confidence']):.3f} "
                f"anchor_quality={quality:.3f}"
            ),
        )

        frame_tiles.append(
            tile
        )

    sheet = np.vstack(
        (
            np.hstack(
                (
                    strip_tile,
                    frame_tiles[0],
                )
            ),
            np.hstack(
                (
                    frame_tiles[1],
                    frame_tiles[2],
                )
            ),
        )
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        sheet,
    ):
        raise RuntimeError(
            f"Cannot write review sheet: {output_path}"
        )


def build_report(
    summary: Mapping[
        str,
        Any,
    ],
) -> str:
    confirmation = summary[
        "user_confirmation"
    ]

    anchor = summary[
        "confirmed_anchor"
    ]

    return f"""# KickClip Target-Centric Tracking V2 — Stage 3-B3

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Confirmed candidate: `{confirmation['candidate_id']}`
- Retrieval rank: `{confirmation['retrieval_rank']}`
- Confirmation source: `{confirmation['source']}`
- Anchor frame: `{anchor['frame_index']}`
- Anchor detection: `{anchor['detection_id']}`
- Anchor confidence: `{anchor['confidence']:.6f}`
- Anchor bbox: `{anchor['bbox_xyxy']}`

## Operational interpretation

Stage 3-B2 returned `AMBIGUOUS`, so no automatic candidate was selected.
The user explicitly confirmed this candidate through the fallback path.

Only this candidate is authorized as the initial post-cut anchor. Other target
fragments are not merged manually. Stage 3-C must start from this anchor and
test whether frozen within-shot tracking can recover subsequent fragments.

The anchor review image must receive visual PASS before Stage 3-C runs.
"""


def main() -> int:
    arguments = parse_args()

    if arguments.review_status != "PASS":
        raise RuntimeError(
            "User candidate confirmation must be PASS"
        )

    if arguments.maximum_confirmable_rank < 1:
        raise ValueError(
            "--maximum-confirmable-rank must be positive"
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

    stage3b0_path = (
        output_dir
        / "stage3b0_summary.json"
    )

    stage3b1_path = (
        output_dir
        / "stage3b1_summary.json"
    )

    stage3b2_path = (
        output_dir
        / "stage3b2_summary.json"
    )

    ranked_path = (
        output_dir
        / "stage3b1_ranked_candidates.csv"
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
        stage3b0_path,
        stage3b1_path,
        stage3b2_path,
        ranked_path,
        tracklets_path,
        assignments_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    stage3b0 = read_json(
        stage3b0_path
    )

    stage3b1 = read_json(
        stage3b1_path
    )

    stage3b2 = read_json(
        stage3b2_path
    )

    ranked_rows = read_csv(
        ranked_path
    )

    tracklets = read_jsonl(
        tracklets_path
    )

    assignments = read_csv(
        assignments_path
    )

    if (
        stage3b2.get(
            "status"
        )
        != "PASS"
        or stage3b2.get(
            "decision"
        )
        != (
            "AUTHORIZE_STAGE3B3_"
            "USER_CONFIRMATION_FALLBACK"
        )
        or stage3b2.get(
            "operational_state"
        )
        != "AMBIGUOUS"
    ):
        raise RuntimeError(
            "Stage 3-B2 must authorize "
            "the AMBIGUOUS user-confirmation fallback"
        )

    confirmed_candidate_id = str(
        arguments.confirmed_candidate_id
    )

    ranked_by_id = {
        str(
            row["candidate_id"]
        ): row
        for row in ranked_rows
    }

    if (
        confirmed_candidate_id
        not in ranked_by_id
    ):
        raise RuntimeError(
            "Confirmed candidate is absent from "
            "Stage 3-B1 ranking: "
            f"{confirmed_candidate_id}"
        )

    ranked_row = ranked_by_id[
        confirmed_candidate_id
    ]

    retrieval_rank = int(
        ranked_row[
            "retrieval_rank"
        ]
    )

    if (
        retrieval_rank
        > arguments.maximum_confirmable_rank
    ):
        raise RuntimeError(
            "Confirmed candidate is outside "
            "the review fallback set: "
            f"rank={retrieval_rank}, "
            f"maximum={arguments.maximum_confirmable_rank}"
        )

    tracklet_by_id = {
        str(
            row["candidate_id"]
        ): row
        for row in tracklets
    }

    if (
        confirmed_candidate_id
        not in tracklet_by_id
    ):
        raise RuntimeError(
            "Confirmed candidate is absent from "
            "Stage 3-B0 tracklets"
        )

    candidate_rows = [
        row
        for row in assignments
        if str(
            row["candidate_id"]
        )
        == confirmed_candidate_id
    ]

    if len(
        candidate_rows
    ) < 3:
        raise RuntimeError(
            "Confirmed candidate has too few "
            "assignment observations"
        )

    anchor, anchor_rows = choose_anchor(
        candidate_rows
    )

    video = Path(
        str(
            stage3b0[
                "inputs"
            ][
                "video"
            ]
        )
    ).resolve()

    if (
        not video.is_file()
        or sha256_file(
            video
        )
        != stage3b0[
            "inputs"
        ][
            "video_sha256"
        ]
    ):
        raise RuntimeError(
            "Video is missing or changed"
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

    strip_path = (
        strip_dir
        / f"{confirmed_candidate_id}.jpg"
    )

    anchor_candidates_path = (
        output_dir
        / "stage3b3_anchor_candidates.csv"
    )

    write_csv(
        anchor_candidates_path,
        anchor_rows,
        list(
            anchor_rows[0]
        ),
    )

    review_path = (
        output_dir
        / "stage3b3_confirmed_anchor_review.jpg"
    )

    make_review_sheet(
        video,
        confirmed_candidate_id,
        anchor_rows,
        anchor,
        strip_path,
        review_path,
    )

    tracklet = tracklet_by_id[
        confirmed_candidate_id
    ]

    confirmation_contract = {
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
        "test_name": (
            test_name
        ),
        "operational_path": (
            "AMBIGUOUS_USER_CONFIRMATION_FALLBACK"
        ),
        "user_confirmation": {
            "candidate_id": (
                confirmed_candidate_id
            ),
            "review_status": (
                arguments.review_status
            ),
            "reviewer": (
                arguments.reviewer
            ),
            "review_note": (
                arguments.review_note
            ),
            "source": (
                "EXPLICIT_USER_VISUAL_CONFIRMATION"
            ),
            "retrieval_rank": (
                retrieval_rank
            ),
            "retrieval_score": finite_float(
                ranked_row[
                    "retrieval_score"
                ],
                "retrieval_score",
            ),
            "prototype_target_similarity": finite_float(
                ranked_row[
                    "prototype_target_similarity"
                ],
                "prototype_target_similarity",
            ),
            "tracklet_start_frame": int(
                tracklet[
                    "start_frame"
                ]
            ),
            "tracklet_end_frame_inclusive": int(
                tracklet[
                    "end_frame_inclusive"
                ]
            ),
            "tracklet_detection_count": int(
                tracklet[
                    "detection_count"
                ]
            ),
        },
        "confirmed_anchor": {
            "candidate_id": (
                confirmed_candidate_id
            ),
            "frame_index": int(
                anchor[
                    "frame_index"
                ]
            ),
            "detection_id": str(
                anchor[
                    "detection_id"
                ]
            ),
            "bbox_xyxy": [
                float(
                    anchor[
                        "x1"
                    ]
                ),
                float(
                    anchor[
                        "y1"
                    ]
                ),
                float(
                    anchor[
                        "x2"
                    ]
                ),
                float(
                    anchor[
                        "y2"
                    ]
                ),
            ],
            "confidence": float(
                anchor[
                    "confidence"
                ]
            ),
            "anchor_quality_score": float(
                anchor[
                    "anchor_quality_score"
                ]
            ),
            "selection_policy": (
                "CONFIDENCE_AREA_CONTINUITY_INTERIOR"
            ),
            "selection_scope": (
                "ONLY_WITHIN_USER_CONFIRMED_CANDIDATE"
            ),
        },
        "authorization": {
            "candidate_identity_user_confirmed": (
                True
            ),
            "anchor_visual_review_completed": (
                False
            ),
            "authorized_for_stage3c_restart": (
                False
            ),
            "automatic_candidate_selection": (
                False
            ),
            "cross_shot_link_source": (
                "USER_CONFIRMATION_FALLBACK"
            ),
            "later_fragments_manually_merged": (
                False
            ),
            "target_memory_updated": (
                False
            ),
        },
    }

    confirmation_path = (
        output_dir
        / "stage3b3_user_confirmation.json"
    )

    atomic_json(
        confirmation_path,
        confirmation_contract,
    )

    summary = {
        "stage": (
            STAGE
        ),
        "version": (
            VERSION
        ),
        "generated_at": (
            confirmation_contract[
                "generated_at"
            ]
        ),
        "status": (
            "PASS"
        ),
        "decision": (
            "AUTHORIZE_MANDATORY_"
            "STAGE3B3_CONFIRMED_ANCHOR_"
            "VISUAL_REVIEW"
        ),
        "test_name": (
            test_name
        ),
        "operational_state": (
            "USER_CONFIRMED_REACQUISITION_CANDIDATE"
        ),
        "user_confirmation": (
            confirmation_contract[
                "user_confirmation"
            ]
        ),
        "confirmed_anchor": (
            confirmation_contract[
                "confirmed_anchor"
            ]
        ),
        "counts": {
            "candidate_observation_count": (
                len(
                    candidate_rows
                )
            ),
            "anchor_candidate_count": (
                len(
                    anchor_rows
                )
            ),
            "manually_merged_later_fragment_count": (
                0
            ),
        },
        "inputs": {
            "stage3b0_summary": str(
                stage3b0_path
            ),
            "stage3b1_summary": str(
                stage3b1_path
            ),
            "stage3b2_summary": str(
                stage3b2_path
            ),
            "ranked_candidates": str(
                ranked_path
            ),
            "ranked_candidates_sha256": (
                sha256_file(
                    ranked_path
                )
            ),
            "tracklets": str(
                tracklets_path
            ),
            "tracklet_assignments": str(
                assignments_path
            ),
            "video": str(
                video
            ),
            "video_sha256": (
                sha256_file(
                    video
                )
            ),
        },
        "outputs": {
            "user_confirmation": str(
                confirmation_path
            ),
            "anchor_candidates": str(
                anchor_candidates_path
            ),
            "anchor_review": str(
                review_path
            ),
        },
        "safety_invariants": {
            "automatic_candidate_selection": (
                False
            ),
            "explicit_user_confirmation": (
                True
            ),
            "anchor_selected_only_within_"
            "confirmed_candidate": (
                True
            ),
            "postcut_track_0004_manually_merged": (
                False
            ),
            "phase1_tracking_executed": (
                False
            ),
            "cross_shot_timeline_link_created": (
                False
            ),
            "target_memory_updated": (
                False
            ),
            "frozen_phase1_modified": (
                False
            ),
        },
    }

    summary_path = (
        output_dir
        / "stage3b3_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3b3_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric Tracking V2 "
        "Stage 3-B3 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        "Decision                       : "
        "AUTHORIZE_MANDATORY_"
        "STAGE3B3_CONFIRMED_ANCHOR_"
        "VISUAL_REVIEW"
    )

    print(
        f"Confirmed candidate            : "
        f"{confirmed_candidate_id}"
    )

    print(
        f"Retrieval rank                 : "
        f"{retrieval_rank}"
    )

    print(
        f"Tracklet range                 : "
        f"frames "
        f"{tracklet['start_frame']}-"
        f"{tracklet['end_frame_inclusive']}"
    )

    print(
        f"Selected anchor frame          : "
        f"{anchor['frame_index']}"
    )

    print(
        f"Selected anchor detection      : "
        f"{anchor['detection_id']}"
    )

    print(
        f"Selected anchor bbox           : "
        f"[{anchor['x1']:.2f}, "
        f"{anchor['y1']:.2f}, "
        f"{anchor['x2']:.2f}, "
        f"{anchor['y2']:.2f}]"
    )

    print(
        f"Selected anchor confidence     : "
        f"{anchor['confidence']:.6f}"
    )

    print(
        f"Anchor quality score           : "
        f"{anchor['anchor_quality_score']:.6f}"
    )

    print(
        "Automatic candidate selection  : NONE"
    )

    print(
        "Later fragment manual merge    : NONE"
    )

    print(
        "Phase-1 restart                : NONE"
    )

    print(
        "Target memory update           : NONE"
    )

    print(
        f"Anchor review                  : "
        f"{review_path}"
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
            "Stage 3-B3 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-B3 fatal error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )