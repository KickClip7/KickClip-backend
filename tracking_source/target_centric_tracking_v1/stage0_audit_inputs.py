#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V1 - Stage 0 input audit.

No training or tracking inference is performed. This script validates the
single-shot input video, first-frame target bbox, frozen RF-DETR checkpoint,
reusable V6/V7 paths, and runtime environment. It writes only under
runs/target_centric_tracking_v1/<test_name>/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPT_VERSION = "target-centric-v1-stage0-1.0.0"
EXPECTED_RFDETR_SHA256 = (
    "5d1d05cf78b6a777430430955d0e745ce"
    "5c61913c659b2e96ae5deddbd1a3a94"
)
TARGET_STATES = [
    "INITIALIZING",
    "ACTIVE",
    "OCCLUDED",
    "LOST",
    "SEARCHING",
    "AMBIGUOUS",
    "REACQUIRED",
    "ABSENT",
    "USER_CONFIRMED",
    "TERMINATED",
]


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    code: str
    message: str
    path: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-name",
        required=True,
    )
    parser.add_argument(
        "--initial-bbox",
        type=float,
        nargs=4,
        required=True,
    )
    parser.add_argument(
        "--bbox-format",
        choices=(
            "xyxy_pixels",
            "xywh_pixels",
            "xyxy_normalized",
        ),
        default="xyxy_pixels",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path(
            "weights/rfdetr/checkpoint_best_regular.pth"
        ),
    )
    parser.add_argument(
        "--expected-detector-sha256",
        default=EXPECTED_RFDETR_SHA256,
        help=(
            "Pass an empty string to disable "
            "frozen-hash validation."
        ),
    )
    parser.add_argument(
        "--v6-code-dir",
        type=Path,
        default=Path(
            "global_ID_tracking_upgrade_v6"
        ),
    )
    parser.add_argument(
        "--v6-output-dir",
        type=Path,
        default=Path(
            "runs/global_ID_tracking_upgrade_v6"
        ),
    )
    parser.add_argument(
        "--v7-code-dir",
        type=Path,
        default=Path(
            "global_ID_tracking_upgrade_v7"
        ),
    )
    parser.add_argument(
        "--v7-output-dir",
        type=Path,
        default=Path(
            "runs/global_ID_tracking_upgrade_v7"
        ),
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path(
            "target_centric_tracking_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "runs/target_centric_tracking_v1"
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--min-bbox-side",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--overwrite-audit",
        action="store_true",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")


def resolve(
    root: Path,
    value: Path,
) -> Path:
    if value.is_absolute():
        return value.expanduser().resolve()

    return (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    value: Any,
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )
    temp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_text(
    path: Path,
    value: str,
) -> None:
    temp = path.with_name(
        path.name + ".tmp"
    )
    temp.write_text(
        value,
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temp, path)


def add(
    findings: list[Finding],
    severity: str,
    category: str,
    code: str,
    message: str,
    path: Optional[Path] = None,
) -> None:
    findings.append(
        Finding(
            severity=severity,
            category=category,
            code=code,
            message=message,
            path=(
                str(path)
                if path is not None
                else None
            ),
        )
    )


def status_of(
    findings: Sequence[Finding],
) -> str:
    if any(
        item.severity == "ERROR"
        for item in findings
    ):
        return "FAIL"

    if any(
        item.severity == "WARNING"
        for item in findings
    ):
        return "PASS_WITH_WARNINGS"

    return "PASS"


def validate_test_name(
    value: str,
) -> str:
    value = value.strip()

    valid = re.fullmatch(
        r"[A-Za-z0-9._-]+",
        value,
    )

    if not valid or value in {
        ".",
        "..",
    }:
        raise ValueError(
            "--test-name may contain only "
            "letters, numbers, dot, "
            "underscore, and hyphen"
        )

    return value


def module_available(
    name: str,
) -> bool:
    return (
        importlib.util.find_spec(name)
        is not None
    )


def read_video(
    path: Path,
) -> tuple[dict[str, Any], Any]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV could not open "
            "the video"
        )

    width = int(
        round(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )
    height = int(
        round(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )
    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )
    frames = int(
        round(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    )

    ok, first_frame = cap.read()
    cap.release()

    if (
        width <= 0
        or height <= 0
        or fps <= 0
        or frames <= 0
        or not ok
    ):
        raise RuntimeError(
            "Invalid video: "
            f"{width}x{height}, "
            f"fps={fps}, "
            f"frames={frames}, "
            f"first_frame={ok}"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": (
            path.stat().st_size
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": (
            frames / fps
        ),
    }, first_frame


def audit_bbox(
    values: Sequence[float],
    bbox_format: str,
    frame_width: int,
    frame_height: int,
    min_side: float,
) -> dict[str, Any]:
    a, b, c, d = map(
        float,
        values,
    )

    if bbox_format == "xyxy_pixels":
        xyxy = [
            a,
            b,
            c,
            d,
        ]
    elif bbox_format == "xywh_pixels":
        xyxy = [
            a,
            b,
            a + c,
            b + d,
        ]
    else:
        xyxy = [
            a * frame_width,
            b * frame_height,
            c * frame_width,
            d * frame_height,
        ]

    x1, y1, x2, y2 = xyxy

    finite = all(
        math.isfinite(value)
        for value in xyxy
    )
    valid_order = (
        x2 > x1
        and y2 > y1
    )

    width = x2 - x1
    height = y2 - y1

    area = (
        max(width, 0.0)
        * max(height, 0.0)
    )

    return {
        "source_format": bbox_format,
        "source_values": list(
            map(float, values)
        ),
        "bbox_xyxy_pixels": [
            round(value, 4)
            for value in xyxy
        ],
        "width": round(
            width,
            4,
        ),
        "height": round(
            height,
            4,
        ),
        "area_ratio": round(
            area
            / (
                frame_width
                * frame_height
            ),
            8,
        ),
        "finite": finite,
        "valid_order": valid_order,
        "inside_frame": (
            finite
            and x1 >= 0
            and y1 >= 0
            and x2 <= frame_width
            and y2 <= frame_height
        ),
        "minimum_size_ok": (
            width >= min_side
            and height >= min_side
        ),
    }


def discover_assets(
    root: Path,
    v6_code: Path,
    v6_output: Path,
) -> dict[str, list[str]]:
    result = {
        "rfdetr_scripts": [],
        "deep_eiou_paths": [],
        "sports_osnet_checkpoints": [],
        "v6_tracker_outputs": [],
    }

    search_roots = [
        path
        for path in (
            root,
            v6_code,
        )
        if path.is_dir()
    ]

    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue

            lower = str(path).lower()
            name = path.name.lower()

            if (
                path.suffix.lower() == ".py"
                and "rfdetr" in name
                and any(
                    token in name
                    for token in (
                        "detect",
                        "infer",
                        "track",
                        "video",
                        "stage1b",
                    )
                )
            ):
                result[
                    "rfdetr_scripts"
                ].append(str(path))

            if (
                "deep-eiou" in lower
                or "deep_eiou" in lower
            ):
                result[
                    "deep_eiou_paths"
                ].append(str(path))

            if (
                path.suffix.lower()
                in {
                    ".pth",
                    ".pt",
                    ".tar",
                    ".ckpt",
                }
                and (
                    "osnet" in name
                    or "sports" in name
                )
            ):
                result[
                    "sports_osnet_checkpoints"
                ].append(str(path))

    if v6_output.is_dir():
        for path in v6_output.rglob("*"):
            if not path.is_file():
                continue

            lower = str(path).lower()

            if any(
                token in lower
                for token in (
                    "deep_eiou_connect_only",
                    "stage1b6",
                    "local_track",
                    "tracklet",
                )
            ):
                result[
                    "v6_tracker_outputs"
                ].append(str(path))

                if (
                    len(
                        result[
                            "v6_tracker_outputs"
                        ]
                    )
                    >= 100
                ):
                    break

    return {
        key: sorted(
            set(values)
        )
        for key, values
        in result.items()
    }


def write_preview(
    frame: Any,
    bbox: Sequence[float],
    path: Path,
) -> bool:
    import cv2  # type: ignore

    image = frame.copy()

    x1, y1, x2, y2 = [
        int(round(value))
        for value in bbox
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        image,
        "TARGET INITIALIZATION",
        (
            max(0, x1),
            max(24, y1 - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return bool(
        cv2.imwrite(
            str(path),
            image,
        )
    )


def make_report(
    status: str,
    output_dir: Path,
    manifest: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    video = (
        manifest.get("video")
        or {}
    )
    target = (
        manifest.get(
            "target_initialization"
        )
        or {}
    )
    detector = manifest["detector"]

    lines = [
        "# KickClip Target-Centric "
        "Tracking V1 — Stage 0 Input Audit",
        "",
        f"- Status: `{status}`",
        f"- Script: `{SCRIPT_VERSION}`",
        f"- Output: `{output_dir}`",
        "- Training / inference: "
        "`NONE / NONE`",
        "- V4-V7 modification: `NONE`",
        "",
        "## Phase-1 frozen configuration",
        "",
        "```text",
        "RF-DETR player/goalkeeper detections",
        "→ target-conditioned "
        "conservative association",
        "→ motion/IoU continuity "
        "+ optional Sports OSNet support",
        "→ low confidence => LOST, "
        "never force another player",
        "→ bbox-first timeline; "
        "mask backend optional for "
        "the first smoke test",
        "```",
        "",
        "V6 Deep-EIoU may be reused "
        "for local association. "
        "GTA, V7 splitting, and "
        "Global-ID linking are excluded.",
        "",
        "## Input",
        "",
        f"- Video: `{video.get('path')}`",
        f"- Video SHA-256: "
        f"`{video.get('sha256')}`",
        "- Geometry / FPS / duration: "
        f"`{video.get('width')}"
        f"×{video.get('height')}` / "
        f"`{video.get('fps')}` / "
        f"`{video.get('duration_seconds')}s`",
        "- Target bbox XYXY: "
        f"`{target.get('bbox_xyxy_pixels')}`",
        f"- Detector: "
        f"`{detector.get('path')}`",
        "- Detector SHA-256 match: "
        f"`{detector.get('expected_hash_match')}`",
        "",
        "## Asset counts",
        "",
    ]

    for key, values in audit[
        "discovered_assets"
    ].items():
        lines.append(
            f"- `{key}`: **{len(values)}**"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )

    if audit["findings"]:
        lines.extend(
            [
                "| Severity | Category | "
                "Code | Message |",
                "|---|---|---|---|",
            ]
        )

        for item in audit["findings"]:
            message = item[
                "message"
            ].replace(
                "|",
                "\\|",
            )

            if item.get("path"):
                message += (
                    f" (`{item['path']}`)"
                )

            lines.append(
                f"| {item['severity']} "
                f"| {item['category']} "
                f"| {item['code']} "
                f"| {message} |"
            )
    else:
        lines.append(
            "- No findings."
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )

    if status == "FAIL":
        lines.append(
            "Phase 1 is blocked until "
            "ERROR findings are fixed."
        )
    else:
        lines.append(
            "Proceed to RF-DETR detection "
            "and conservative single-shot "
            "target tracking."
        )

    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []
    generated_at = now_iso()

    video: Optional[
        dict[str, Any]
    ] = None
    target: Optional[
        dict[str, Any]
    ] = None
    first_frame: Any = None

    try:
        test_name = validate_test_name(
            args.test_name
        )

        root = (
            args.project_root
            .expanduser()
            .resolve()
        )

        if not root.is_dir():
            raise FileNotFoundError(
                "Project root not found: "
                f"{root}"
            )

        video_path = resolve(
            root,
            args.video,
        )
        detector_path = resolve(
            root,
            args.detector_checkpoint,
        )

        v6_code = resolve(
            root,
            args.v6_code_dir,
        )
        v6_output = resolve(
            root,
            args.v6_output_dir,
        )
        v7_code = resolve(
            root,
            args.v7_code_dir,
        )
        v7_output = resolve(
            root,
            args.v7_output_dir,
        )

        code_dir = resolve(
            root,
            args.code_dir,
        )
        output_root = resolve(
            root,
            args.output_root,
        )
        output_dir = (
            output_root
            / test_name
        )

        for protected in (
            v6_code,
            v6_output,
            v7_code,
            v7_output,
        ):
            try:
                output_dir.relative_to(
                    protected
                )
            except ValueError:
                continue

            raise ValueError(
                "New output must not be "
                "inside protected path: "
                f"{protected}"
            )

        audit_files = [
            output_dir / name
            for name in (
                "input_manifest.json",
                "target_initialization.json",
                "audit.json",
                "audit_findings.csv",
                "report.md",
            )
        ]

        if (
            any(
                path.exists()
                for path in audit_files
            )
            and not args.overwrite_audit
        ):
            raise FileExistsError(
                "Stage-0 files already exist. "
                "Use --overwrite-audit "
                "to replace them."
            )

        code_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        (
            output_dir / "masks"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        packages = {
            name: module_available(name)
            for name in (
                "numpy",
                "cv2",
                "torch",
                "scipy",
                "PIL",
                "rfdetr",
            )
        }

        for name in (
            "numpy",
            "cv2",
        ):
            if not packages[name]:
                add(
                    findings,
                    "ERROR",
                    "environment",
                    "PACKAGE_MISSING",
                    "Required package "
                    f"not importable: {name}",
                )

        for name in (
            "torch",
            "scipy",
            "PIL",
            "rfdetr",
        ):
            if not packages[name]:
                add(
                    findings,
                    "WARNING",
                    "environment",
                    "PACKAGE_MISSING",
                    "Recommended package "
                    f"not importable: {name}",
                )

        if not video_path.is_file():
            add(
                findings,
                "ERROR",
                "video",
                "VIDEO_NOT_FOUND",
                "Input video does not exist",
                video_path,
            )
        elif packages["cv2"]:
            try:
                (
                    video,
                    first_frame,
                ) = read_video(
                    video_path
                )

                if not (
                    args.min_duration
                    <= video[
                        "duration_seconds"
                    ]
                    <= args.max_duration
                ):
                    add(
                        findings,
                        "WARNING",
                        "video",
                        "DURATION_OUTSIDE_RANGE",
                        "Recommended 10-30s; "
                        "actual "
                        f"{video['duration_seconds']:.3f}s",
                        video_path,
                    )

                target = audit_bbox(
                    args.initial_bbox,
                    args.bbox_format,
                    video["width"],
                    video["height"],
                    args.min_bbox_side,
                )

                if (
                    not target["finite"]
                    or not target[
                        "valid_order"
                    ]
                ):
                    add(
                        findings,
                        "ERROR",
                        "target",
                        "INVALID_BBOX",
                        "Bbox values or "
                        "order are invalid",
                    )

                if not target[
                    "inside_frame"
                ]:
                    add(
                        findings,
                        "ERROR",
                        "target",
                        "BBOX_OUTSIDE_FRAME",
                        "Bbox must be fully "
                        "inside the original "
                        "frame; no automatic "
                        "clipping",
                    )

                if not target[
                    "minimum_size_ok"
                ]:
                    add(
                        findings,
                        "ERROR",
                        "target",
                        "BBOX_TOO_SMALL",
                        "Each side must be "
                        "at least "
                        f"{args.min_bbox_side}px",
                    )

            except Exception as exc:
                add(
                    findings,
                    "ERROR",
                    "video",
                    "VIDEO_AUDIT_FAILED",
                    str(exc),
                    video_path,
                )

        detector = {
            "path": str(
                detector_path
            ),
            "exists": (
                detector_path.is_file()
            ),
            "size_bytes": (
                detector_path.stat().st_size
                if detector_path.is_file()
                else None
            ),
            "sha256": (
                sha256_file(
                    detector_path
                )
                if detector_path.is_file()
                else None
            ),
            "expected_sha256": (
                args.expected_detector_sha256
                or None
            ),
            "expected_hash_match": None,
        }

        if not detector_path.is_file():
            add(
                findings,
                "ERROR",
                "detector",
                "RFDETR_NOT_FOUND",
                "Frozen RF-DETR checkpoint "
                "is required",
                detector_path,
            )
        elif args.expected_detector_sha256:
            detector[
                "expected_hash_match"
            ] = (
                detector["sha256"].lower()
                == args
                .expected_detector_sha256
                .lower()
            )

            if not detector[
                "expected_hash_match"
            ]:
                add(
                    findings,
                    "ERROR",
                    "detector",
                    "RFDETR_HASH_MISMATCH",
                    "Checkpoint differs "
                    "from the frozen "
                    "V6/V7 contract",
                    detector_path,
                )

        for label, path in (
            (
                "V6_CODE",
                v6_code,
            ),
            (
                "V6_OUTPUT",
                v6_output,
            ),
            (
                "V7_CODE",
                v7_code,
            ),
            (
                "V7_OUTPUT",
                v7_output,
            ),
        ):
            if not path.exists():
                add(
                    findings,
                    "WARNING",
                    "reusable_assets",
                    f"{label}_NOT_FOUND",
                    "Existing asset path "
                    "was not found",
                    path,
                )

        assets = discover_assets(
            root,
            v6_code,
            v6_output,
        )

        if not assets[
            "rfdetr_scripts"
        ]:
            add(
                findings,
                "WARNING",
                "reusable_assets",
                "RFDETR_ADAPTER_NOT_FOUND",
                "No likely RF-DETR "
                "inference script "
                "was discovered",
            )

        if not assets[
            "sports_osnet_checkpoints"
        ]:
            add(
                findings,
                "WARNING",
                "reusable_assets",
                "SPORTS_OSNET_NOT_FOUND",
                "Optional for the first "
                "bbox-only smoke test",
            )

        cuda_available = None
        cuda_device = None
        torch_version = None

        if packages["torch"]:
            try:
                import torch  # type: ignore

                torch_version = str(
                    torch.__version__
                )
                cuda_available = bool(
                    torch.cuda.is_available()
                )

                cuda_device = (
                    torch.cuda
                    .get_device_name(0)
                    if cuda_available
                    else None
                )

                if not cuda_available:
                    add(
                        findings,
                        "WARNING",
                        "environment",
                        "CUDA_NOT_AVAILABLE",
                        "RF-DETR inference "
                        "may be slow",
                    )

            except Exception as exc:
                add(
                    findings,
                    "WARNING",
                    "environment",
                    "TORCH_AUDIT_FAILED",
                    str(exc),
                )

        ffmpeg = shutil.which(
            "ffmpeg"
        )

        if ffmpeg is None:
            add(
                findings,
                "WARNING",
                "environment",
                "FFMPEG_NOT_FOUND",
                "Preview rendering "
                "will use OpenCV",
            )

        environment = {
            "python_executable": (
                sys.executable
            ),
            "python_version": (
                sys.version.replace(
                    "\n",
                    " ",
                )
            ),
            "platform": (
                platform.platform()
            ),
            "conda_env": (
                os.environ.get(
                    "CONDA_DEFAULT_ENV"
                )
            ),
            "packages": packages,
            "torch_version": (
                torch_version
            ),
            "cuda_available": (
                cuda_available
            ),
            "cuda_device": (
                cuda_device
            ),
            "ffmpeg": ffmpeg,
        }

        preview_name = None

        if (
            first_frame is not None
            and target
            and target["inside_frame"]
        ):
            preview_name = (
                "target_initialization"
                "_preview.jpg"
            )

            if not write_preview(
                first_frame,
                target[
                    "bbox_xyxy_pixels"
                ],
                output_dir / preview_name,
            ):
                add(
                    findings,
                    "WARNING",
                    "target",
                    "PREVIEW_WRITE_FAILED",
                    "Could not write "
                    "initialization preview",
                )
                preview_name = None

        status = status_of(
            findings
        )

        manifest = {
            "schema_version": (
                "kickclip.target_centric"
                ".input_manifest.v1"
            ),
            "stage": (
                "stage0_input_audit"
            ),
            "script_version": (
                SCRIPT_VERSION
            ),
            "generated_at": (
                generated_at
            ),
            "status": status,
            "project_root": str(root),
            "test_name": test_name,
            "video": video,
            "target_initialization": (
                target
            ),
            "detector": detector,
            "paths": {
                "new_code_dir": (
                    str(code_dir)
                ),
                "new_output_root": (
                    str(output_root)
                ),
                "test_output_dir": (
                    str(output_dir)
                ),
                "v6_code_read_only": (
                    str(v6_code)
                ),
                "v6_output_read_only": (
                    str(v6_output)
                ),
                "v7_code_read_only": (
                    str(v7_code)
                ),
                "v7_output_read_only": (
                    str(v7_output)
                ),
            },
            "phase1_contract": {
                "scope": (
                    "single_shot_"
                    "target_tracking"
                ),
                "camera_cut_allowed": (
                    False
                ),
                "selection_frame_index": 0,
                "states": TARGET_STATES,
                "primary_metric": (
                    "silent_wrong_"
                    "player_switch"
                ),
                "low_confidence_policy": (
                    "LOST_NOT_"
                    "FORCED_SWITCH"
                ),
                "global_linking_enabled": (
                    False
                ),
                "gta_enabled": False,
                "sports_osnet_role": (
                    "support_only_optional"
                ),
                "mask_backend": (
                    "optional_bbox_"
                    "first_mvp"
                ),
                "preserve_original_geometry": (
                    True
                ),
            },
        }

        initialization = {
            "schema_version": (
                "kickclip.target_"
                "initialization.v1"
            ),
            "generated_at": (
                generated_at
            ),
            "status": (
                "INITIALIZING"
                if status != "FAIL"
                else "INVALID"
            ),
            "target_id": "target_001",
            "selection_frame_index": 0,
            "selection_time_ms": 0,
            "bbox_xyxy": (
                target[
                    "bbox_xyxy_pixels"
                ]
                if target
                else None
            ),
            "frame_geometry": {
                "width": (
                    video["width"]
                    if video
                    else None
                ),
                "height": (
                    video["height"]
                    if video
                    else None
                ),
            },
            "identity_memory_initialized": (
                False
            ),
            "appearance_embedding_initialized": (
                False
            ),
            "mask_initialized": False,
            "preview_path": (
                preview_name
            ),
        }

        audit = {
            "schema_version": (
                "kickclip.target_centric"
                ".audit.v1"
            ),
            "stage": (
                "stage0_input_audit"
            ),
            "script_version": (
                SCRIPT_VERSION
            ),
            "generated_at": (
                generated_at
            ),
            "status": status,
            "counts": {
                "errors": sum(
                    item.severity
                    == "ERROR"
                    for item in findings
                ),
                "warnings": sum(
                    item.severity
                    == "WARNING"
                    for item in findings
                ),
            },
            "findings": [
                asdict(item)
                for item in findings
            ],
            "environment": environment,
            "discovered_assets": assets,
            "protected_paths_modified": (
                False
            ),
            "training_performed": (
                False
            ),
            "tracking_inference_performed": (
                False
            ),
            "threshold_selection_performed": (
                False
            ),
        }

        write_json(
            output_dir
            / "input_manifest.json",
            manifest,
        )
        write_json(
            output_dir
            / "target_initialization.json",
            initialization,
        )
        write_json(
            output_dir
            / "audit.json",
            audit,
        )

        with (
            output_dir
            / "audit_findings.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "severity",
                    "category",
                    "code",
                    "message",
                    "path",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    asdict(item)
                    for item in findings
                ]
            )

        write_text(
            output_dir / "report.md",
            make_report(
                status,
                output_dir,
                manifest,
                audit,
            ),
        )

        print(
            "KickClip Target-Centric "
            "Tracking V1 Stage 0 "
            "audit complete"
        )
        print(
            f"Status             : {status}"
        )
        print(
            "Errors / warnings  : "
            f"{audit['counts']['errors']} / "
            f"{audit['counts']['warnings']}"
        )

        if video:
            print(
                "Video              : "
                f"{video['width']}x"
                f"{video['height']}, "
                f"{video['fps']:.4f} fps, "
                f"{video['duration_seconds']:.3f}s"
            )

        print(
            "RF-DETR hash match : "
            f"{detector['expected_hash_match']}"
        )
        print(
            "Training/inference : "
            "NONE / NONE"
        )
        print(
            "V4-V7 modification : NONE"
        )
        print(
            f"Output             : "
            f"{output_dir}"
        )

        if status == "FAIL":
            return 2

        return 0

    except Exception as exc:
        print(
            "Stage 0 fatal error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())