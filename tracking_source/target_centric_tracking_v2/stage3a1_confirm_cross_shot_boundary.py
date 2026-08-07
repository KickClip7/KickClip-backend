#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-A1.

Freeze one manually reviewed camera cut and produce a deterministic two-shot
manifest. No tracking, detector inference, ReID inference, or cross-shot link is
performed in this stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


STAGE = "stage3a1_confirm_cross_shot_boundary"
VERSION = "target-centric-v2-stage3a1-1.0.0"

OUTPUT_NAMES = (
    "stage3a1_confirmed_cut.json",
    "stage3a1_shot_manifest.csv",
    "stage3a1_cut_review.jpg",
    "stage3a1_summary.json",
    "stage3a1_report.md",
)


@dataclass(frozen=True)
class ShotRecord:
    shot_index: int
    shot_id: str
    start_frame: int
    end_frame_inclusive: int
    frame_count: int
    start_seconds: float
    end_seconds_exclusive: float
    role: str


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
        "--confirmed-cut-frame",
        type=int,
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
        required=True,
    )

    parser.add_argument(
        "--review-note",
        default="",
    )

    parser.add_argument(
        "--candidate-frame-tolerance",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--context-offset-frames",
        type=int,
        default=15,
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
        for name
        in OUTPUT_NAMES
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
            "Stage 3-A1 outputs already exist. "
            "Use --overwrite:\n"
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


def video_contract(
    path: Path,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(
        str(
            path
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
        )

    width = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )

    height = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    )

    capture.release()

    if (
        width <= 0
        or height <= 0
        or frame_count <= 1
        or not math.isfinite(
            fps
        )
        or fps <= 0
    ):
        raise RuntimeError(
            "Invalid video contract: "
            f"{width}x{height}, "
            f"fps={fps}, "
            f"frames={frame_count}"
        )

    return {
        "path": str(
            path
        ),
        "sha256": sha256_file(
            path
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": (
            frame_count
            / fps
        ),
    }


def read_frame(
    path: Path,
    frame_index: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(
        str(
            path
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
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
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    source_height, source_width = (
        frame.shape[
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
        frame,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
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


def label_tile(
    tile: np.ndarray,
    text: str,
) -> None:
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
            36,
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
        text,
        (
            8,
            24,
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


def render_review(
    video_path: Path,
    cut_frame: int,
    frame_count: int,
    offset: int,
    fps: float,
    output_path: Path,
) -> list[int]:
    indices = [
        max(
            0,
            cut_frame
            - offset,
        ),
        cut_frame
        - 1,
        cut_frame,
        min(
            frame_count
            - 1,
            cut_frame
            + offset,
        ),
    ]

    labels = (
        "PRE-CUT CONTEXT",
        "LAST FRAME OF SHOT 0",
        "FIRST FRAME OF SHOT 1",
        "POST-CUT CONTEXT",
    )

    tiles: list[
        np.ndarray
    ] = []

    for (
        frame_index,
        label,
    ) in zip(
        indices,
        labels,
    ):
        tile = fit_tile(
            read_frame(
                video_path,
                frame_index,
            ),
            640,
            360,
        )

        label_tile(
            tile,
            (
                f"{label} | "
                f"frame={frame_index} | "
                f"time={frame_index / fps:.3f}s"
            ),
        )

        tiles.append(
            tile
        )

    sheet = np.vstack(
        (
            np.hstack(
                tiles[
                    :2
                ]
            ),
            np.hstack(
                tiles[
                    2:
                ]
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
            "Cannot write review image: "
            f"{output_path}"
        )

    return indices


def build_report(
    summary: Mapping[
        str,
        Any,
    ],
) -> str:
    cut = summary[
        "confirmed_cut"
    ]

    return f"""# KickClip Target-Centric Tracking V2 — Stage 3-A1

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Confirmed cut frame: `{cut['cut_frame']}`
- Last frame of shot 0: `{cut['before_frame']}`
- First frame of shot 1: `{cut['after_frame']}`
- Cut time: `{cut['timestamp_seconds']:.6f}s`
- Review: `{cut['review_status']}` by `{cut['reviewer']}`

## Frozen shot split

- `shot_0000`: frames 0 through {cut['before_frame']}
- `shot_0001`: frames {cut['after_frame']} through {summary['video']['frame_count'] - 1}

This stage only freezes the manually reviewed camera cut. It does not perform
tracking, ReID inference, candidate selection, or cross-shot linking. Motion and
bbox continuity must be reset at the confirmed cut in subsequent stages.
"""


def main() -> int:
    arguments = parse_args()

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

    if not stage3a0_path.is_file():
        raise FileNotFoundError(
            stage3a0_path
        )

    stage3a0 = read_json(
        stage3a0_path
    )

    if (
        stage3a0.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 3-A0 must be PASS"
        )

    if stage3a0.get(
        "decision"
    ) not in {
        "AUTHORIZE_MANUAL_CUT_CANDIDATE_REVIEW",
        "REQUIRE_MANUAL_CUT_FRAME_CONFIRMATION",
    }:
        raise RuntimeError(
            "Unexpected Stage 3-A0 decision: "
            f"{stage3a0.get('decision')}"
        )

    if (
        arguments.review_status
        != "PASS"
    ):
        raise RuntimeError(
            "Manual cut review must be PASS "
            "before freezing the boundary"
        )

    prepare_outputs(
        output_dir,
        arguments.overwrite,
    )

    stage3a0_video = stage3a0.get(
        "video"
    )

    if not isinstance(
        stage3a0_video,
        dict,
    ):
        raise RuntimeError(
            "Stage 3-A0 video contract is missing"
        )

    video_path = Path(
        str(
            stage3a0_video[
                "path"
            ]
        )
    ).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            video_path
        )

    video = video_contract(
        video_path
    )

    if (
        video[
            "sha256"
        ]
        != stage3a0_video.get(
            "sha256"
        )
    ):
        raise RuntimeError(
            "Video hash changed after Stage 3-A0"
        )

    if (
        video[
            "frame_count"
        ]
        != int(
            stage3a0_video.get(
                "actual_frame_count",
                -1,
            )
        )
    ):
        raise RuntimeError(
            "Video frame count changed after Stage 3-A0"
        )

    cut_frame = (
        arguments.confirmed_cut_frame
    )

    if not (
        1
        <= cut_frame
        < video[
            "frame_count"
        ]
    ):
        raise ValueError(
            "--confirmed-cut-frame must be "
            f"within 1..{video['frame_count'] - 1}"
        )

    raw_candidates = (
        stage3a0.get(
            "cut_detection",
            {},
        )
        .get(
            "candidates",
            [],
        )
    )

    candidate = None

    for item in raw_candidates:
        if not isinstance(
            item,
            dict,
        ):
            continue

        candidate_frame = int(
            item.get(
                "cut_frame",
                -999999,
            )
        )

        if (
            abs(
                candidate_frame
                - cut_frame
            )
            <= arguments
            .candidate_frame_tolerance
        ):
            candidate = dict(
                item
            )

            break

    if candidate is None:
        raise RuntimeError(
            "Confirmed cut does not match a "
            "Stage 3-A0 ranked candidate within "
            "tolerance="
            f"{arguments.candidate_frame_tolerance}"
        )

    before_frame = (
        cut_frame
        - 1
    )

    after_frame = (
        cut_frame
    )

    fps = float(
        video[
            "fps"
        ]
    )

    review_path = (
        output_dir
        / "stage3a1_cut_review.jpg"
    )

    review_frames = render_review(
        video_path,
        cut_frame,
        int(
            video[
                "frame_count"
            ]
        ),
        max(
            1,
            arguments
            .context_offset_frames,
        ),
        fps,
        review_path,
    )

    shots = [
        ShotRecord(
            shot_index=0,
            shot_id="shot_0000",
            start_frame=0,
            end_frame_inclusive=(
                before_frame
            ),
            frame_count=(
                cut_frame
            ),
            start_seconds=0.0,
            end_seconds_exclusive=(
                cut_frame
                / fps
            ),
            role=(
                "PRE_CUT_TARGET_TRACKING_"
                "AND_MEMORY_SOURCE"
            ),
        ),
        ShotRecord(
            shot_index=1,
            shot_id="shot_0001",
            start_frame=(
                after_frame
            ),
            end_frame_inclusive=(
                int(
                    video[
                        "frame_count"
                    ]
                )
                - 1
            ),
            frame_count=(
                int(
                    video[
                        "frame_count"
                    ]
                )
                - cut_frame
            ),
            start_seconds=(
                cut_frame
                / fps
            ),
            end_seconds_exclusive=(
                int(
                    video[
                        "frame_count"
                    ]
                )
                / fps
            ),
            role=(
                "POST_CUT_CROSS_SHOT_"
                "REACQUISITION_SEARCH"
            ),
        ),
    ]

    shot_manifest_path = (
        output_dir
        / "stage3a1_shot_manifest.csv"
    )

    shot_rows = [
        asdict(
            shot
        )
        for shot
        in shots
    ]

    write_csv(
        shot_manifest_path,
        shot_rows,
        list(
            shot_rows[
                0
            ]
        ),
    )

    confirmed_cut = {
        "cut_frame": (
            cut_frame
        ),
        "before_frame": (
            before_frame
        ),
        "after_frame": (
            after_frame
        ),
        "timestamp_seconds": (
            cut_frame
            / fps
        ),
        "candidate_rank": (
            candidate.get(
                "rank"
            )
        ),
        "stage3a0_combined_score": (
            candidate.get(
                "combined_score"
            )
        ),
        "stage3a0_hard_cut_candidate": (
            candidate.get(
                "hard_cut_candidate"
            )
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
        "review_image": str(
            review_path
        ),
        "review_frames": (
            review_frames
        ),
        "cut_direction": (
            "WIDE_TO_TARGET_CLOSEUP"
        ),
        "post_cut_initial_occlusion_expected": (
            True
        ),
    }

    confirmed_cut_path = (
        output_dir
        / "stage3a1_confirmed_cut.json"
    )

    atomic_json(
        confirmed_cut_path,
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
            "test_name": (
                test_name
            ),
            "video": (
                video
            ),
            "confirmed_cut": (
                confirmed_cut
            ),
            "shots": [
                asdict(
                    shot
                )
                for shot
                in shots
            ],
        },
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
            "AUTHORIZE_STAGE3A2_"
            "PRE_CUT_TARGET_MEMORY_BUILD"
        ),
        "test_name": (
            test_name
        ),
        "video": (
            video
        ),
        "stage3a0_summary": str(
            stage3a0_path
        ),
        "stage3a0_summary_sha256": (
            sha256_file(
                stage3a0_path
            )
        ),
        "confirmed_cut": (
            confirmed_cut
        ),
        "shots": [
            asdict(
                shot
            )
            for shot
            in shots
        ],
        "outputs": {
            "confirmed_cut": str(
                confirmed_cut_path
            ),
            "shot_manifest": str(
                shot_manifest_path
            ),
            "cut_review": str(
                review_path
            ),
        },
        "safety_invariants": {
            "manual_cut_review_required": (
                True
            ),
            "camera_cut_motion_reset_required": (
                True
            ),
            "tracking_performed": (
                False
            ),
            "detector_inference_performed": (
                False
            ),
            "reid_inference_performed": (
                False
            ),
            "cross_shot_linking_performed": (
                False
            ),
            "phase1_files_modified": (
                False
            ),
            "threshold_search_performed": (
                False
            ),
        },
    }

    summary_path = (
        output_dir
        / "stage3a1_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3a1_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric "
        "Tracking V2 Stage 3-A1 complete"
    )

    print(
        "Status               : PASS"
    )

    print(
        "Decision             : "
        "AUTHORIZE_STAGE3A2_"
        "PRE_CUT_TARGET_MEMORY_BUILD"
    )

    print(
        f"Confirmed cut        : "
        f"frame {cut_frame} "
        f"({cut_frame / fps:.3f}s)"
    )

    print(
        f"Shot 0               : "
        f"frames 0-{before_frame} "
        f"({cut_frame} frames)"
    )

    print(
        f"Shot 1               : "
        f"frames {after_frame}-"
        f"{video['frame_count'] - 1} "
        f"({video['frame_count'] - cut_frame} frames)"
    )

    print(
        f"Candidate            : "
        f"rank #{candidate.get('rank')} "
        f"score={candidate.get('combined_score')}"
    )

    print(
        "Tracking/ReID/linking: "
        "NONE/NONE/NONE"
    )

    print(
        f"Review image         : "
        f"{review_path}"
    )

    print(
        f"Output               : "
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
            "Stage 3-A1 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-A1 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )