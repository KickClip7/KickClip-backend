#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-C0.

Run the frozen Phase-1 stages on a short, camera-cut-free clip whose only
Stage-0 warning is DURATION_OUTSIDE_RANGE.

The original Stage-0 run is preserved unchanged. This adapter creates a new,
derived Phase-1 run directory, copies the original Stage-0 artifacts, and
changes only the copied top-level status fields from PASS_WITH_WARNINGS to
PASS after validating the exact warning contract.

The compatibility authorization does not:
- modify frozen Phase-1 source code;
- modify the original Stage-0 output;
- add padded frames;
- use postcut_track_0004 or any manual same-player link as inference input;
- change tracking or ReID thresholds;
- modify target memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2


STAGE = "stage3c0_run_short_clip_phase1_compatibility"
VERSION = "target-centric-v2-stage3c0-1.0.0"

ALLOWED_WARNING = {
    "severity": "WARNING",
    "category": "video",
    "code": "DURATION_OUTSIDE_RANGE",
}

FROZEN_STAGE_SCRIPTS = (
    "stage1_generate_rfdetr_detections.py",
    "stage2_run_conservative_target_association.py",
    "stage2b_run_same_shot_reentry_reacquisition.py",
    "phase1_finalize_latest_timeline.py",
    "phase1_validate_outputs.py",
)

V2_OUTPUT_NAMES = (
    "stage3c0_short_clip_compatibility_authorization.json",
    "stage3c0_execution_log.txt",
    "stage3c0_summary.json",
    "stage3c0_report.md",
)

V2_OUTPUT_DIRECTORIES = (
    "stage3c0_source_stage0_snapshot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--v2-test-name",
        required=True,
    )

    parser.add_argument(
        "--source-stage0-test-name",
        required=True,
    )

    parser.add_argument(
        "--compat-test-name",
        required=True,
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
        "--reviewer",
        default="USER",
    )

    parser.add_argument(
        "--review-note",
        default="",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def validate_name(
    value: str,
    argument_name: str,
) -> str:
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
            f"Invalid {argument_name}: {value}"
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


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
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


def prepare_v2_outputs(
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        output_dir / name
        for name in V2_OUTPUT_NAMES
        if (
            output_dir / name
        ).exists()
    ]

    existing.extend(
        output_dir / name
        for name in V2_OUTPUT_DIRECTORIES
        if (
            output_dir / name
        ).exists()
    )

    if existing and not overwrite:
        raise FileExistsError(
            "Stage 3-C0 outputs already exist. "
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


def normalize_finding(
    row: Mapping[str, Any],
) -> dict[str, str]:
    return {
        str(key).strip().lower(): (
            ""
            if value is None
            else str(value).strip()
        )
        for key, value in row.items()
    }


def verify_duration_only_warning(
    findings_path: Path,
) -> dict[str, str]:
    findings = [
        normalize_finding(
            row
        )
        for row in read_csv(
            findings_path
        )
    ]

    findings = [
        row
        for row in findings
        if any(
            value
            for value in row.values()
        )
    ]

    if len(findings) != 1:
        raise RuntimeError(
            "Short-clip compatibility requires exactly "
            "one Stage-0 finding; found "
            f"{len(findings)}"
        )

    finding = findings[0]

    for key, expected in ALLOWED_WARNING.items():
        actual = finding.get(
            key,
            "",
        )

        if actual != expected:
            raise RuntimeError(
                "Unauthorized Stage-0 finding: "
                f"{key}={actual!r}, expected={expected!r}"
            )

    message = finding.get(
        "message",
        "",
    )

    if "actual" not in message.lower():
        raise RuntimeError(
            "Duration warning message is malformed: "
            f"{message}"
        )

    video_path = finding.get(
        "path",
        "",
    )

    if not video_path:
        raise RuntimeError(
            "Duration warning does not contain a video path"
        )

    return finding


def probe_video(
    video_path: Path,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(
            video_path
        )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
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

    if fps <= 0:
        raise RuntimeError(
            f"Invalid video FPS: {fps}"
        )

    duration = (
        frame_count
        / fps
    )

    return {
        "path": str(
            video_path
        ),
        "sha256": sha256_file(
            video_path
        ),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def verify_source_stage0_status(
    audit: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> None:
    audit_status = str(
        audit.get(
            "status",
            "",
        )
    )

    manifest_status = str(
        input_manifest.get(
            "status",
            "",
        )
    )

    if (
        audit_status != "PASS_WITH_WARNINGS"
        or manifest_status != "PASS_WITH_WARNINGS"
    ):
        raise RuntimeError(
            "Expected source Stage-0 statuses "
            "PASS_WITH_WARNINGS / PASS_WITH_WARNINGS; "
            f"found {audit_status} / {manifest_status}"
        )


def patch_derived_status(
    path: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(
        path
    )

    original_status = str(
        value.get(
            "status",
            "",
        )
    )

    if original_status != "PASS_WITH_WARNINGS":
        raise RuntimeError(
            f"Cannot authorize unexpected status "
            f"in {path}: {original_status}"
        )

    value[
        "status"
    ] = "PASS"

    value[
        "stage3c0_short_clip_compatibility"
    ] = dict(
        authorization
    )

    atomic_json(
        path,
        value,
    )

    return value


def run_command(
    command: Sequence[str],
    project_root: Path,
    log_lines: list[str],
) -> None:
    printable = subprocess.list2cmdline(
        list(
            command
        )
    )

    banner = (
        f"[STAGE3C0] {printable}\n"
    )

    print(
        banner,
        end="",
    )

    log_lines.append(
        banner
    )

    process = subprocess.Popen(
        list(
            command
        ),
        cwd=str(
            project_root
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
        )

        log_lines.append(
            line
        )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed ({return_code}): "
            f"{printable}"
        )


def build_report(
    summary: Mapping[str, Any],
) -> str:
    video = summary[
        "video"
    ]

    return f"""# KickClip Target-Centric Tracking V2 — Stage 3-C0

- Status: `{summary['status']}`
- Decision: `{summary['decision']}`
- Source Stage-0 test: `{summary['source_stage0_test_name']}`
- Derived compatibility test: `{summary['compat_test_name']}`
- Video frames: `{video['frame_count']}`
- Video duration: `{video['duration_seconds']:.6f}s`
- Allowed Stage-0 warning: `DURATION_OUTSIDE_RANGE`
- Padding frames used: `0`

## Compatibility authorization

The original Stage-0 directory was preserved unchanged. A separate derived
Phase-1 directory was created. Only the copied top-level status fields in
`audit.json` and `input_manifest.json` were changed from
`PASS_WITH_WARNINGS` to `PASS`.

Authorization was granted only after confirming that Stage 0 contained exactly
one finding: `WARNING / video / DURATION_OUTSIDE_RANGE`.

## Safety

No frozen Phase-1 Python source was modified. No tracking or ReID threshold was
changed. No later target fragment ID was supplied to inference. The result
requires visual review over local frames 0–81.
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

    v2_test_name = validate_name(
        arguments.v2_test_name,
        "--v2-test-name",
    )

    source_test_name = validate_name(
        arguments.source_stage0_test_name,
        "--source-stage0-test-name",
    )

    compat_test_name = validate_name(
        arguments.compat_test_name,
        "--compat-test-name",
    )

    if source_test_name == compat_test_name:
        raise ValueError(
            "Source and compatibility test names must differ"
        )

    phase1_code_dir = (
        root
        / "target_centric_tracking_v1"
    ).resolve()

    phase1_runs_dir = (
        root
        / "runs"
        / "target_centric_tracking_v1"
    ).resolve()

    source_dir = (
        phase1_runs_dir
        / source_test_name
    ).resolve()

    compat_dir = (
        phase1_runs_dir
        / compat_test_name
    ).resolve()

    v2_output_dir = (
        root
        / "runs"
        / "target_centric_tracking_v2"
        / v2_test_name
    ).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(
            source_dir
        )

    if not v2_output_dir.is_dir():
        raise FileNotFoundError(
            v2_output_dir
        )

    prepare_v2_outputs(
        v2_output_dir,
        arguments.overwrite,
    )

    if compat_dir.exists():
        if not arguments.overwrite:
            raise FileExistsError(
                "Compatibility Phase-1 output already exists. "
                "Use --overwrite:\n"
                f"{compat_dir}"
            )

        if not compat_dir.is_dir():
            raise NotADirectoryError(
                compat_dir
            )

        shutil.rmtree(
            compat_dir
        )

    source_audit_path = (
        source_dir
        / "audit.json"
    )

    source_manifest_path = (
        source_dir
        / "input_manifest.json"
    )

    source_findings_path = (
        source_dir
        / "audit_findings.csv"
    )

    for path in (
        source_audit_path,
        source_manifest_path,
        source_findings_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    source_audit = read_json(
        source_audit_path
    )

    source_manifest = read_json(
        source_manifest_path
    )

    verify_source_stage0_status(
        source_audit,
        source_manifest,
    )

    finding = verify_duration_only_warning(
        source_findings_path
    )

    video_path = Path(
        finding[
            "path"
        ]
    ).expanduser().resolve()

    video_contract = probe_video(
        video_path
    )

    if (
        video_contract[
            "duration_seconds"
        ]
        >= 10.0
    ):
        raise RuntimeError(
            "Compatibility authorization is only for "
            "a genuinely short clip; duration="
            f"{video_contract['duration_seconds']:.6f}s"
        )

    if video_contract[
        "frame_count"
    ] < 2:
        raise RuntimeError(
            "Short clip contains too few frames"
        )

    snapshot_dir = (
        v2_output_dir
        / "stage3c0_source_stage0_snapshot"
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in (
        source_audit_path,
        source_manifest_path,
        source_findings_path,
    ):
        shutil.copy2(
            path,
            snapshot_dir
            / path.name,
        )

    source_hashes_before = {
        path.name: sha256_file(
            path
        )
        for path in (
            source_audit_path,
            source_manifest_path,
            source_findings_path,
        )
    }

    shutil.copytree(
        source_dir,
        compat_dir,
    )

    generated_at = (
        datetime.now(
            timezone.utc
        )
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    compatibility_authorization = {
        "stage": STAGE,
        "version": VERSION,
        "authorized_at": generated_at,
        "reviewer": arguments.reviewer,
        "review_note": arguments.review_note,
        "source_stage0_test_name": (
            source_test_name
        ),
        "compat_test_name": (
            compat_test_name
        ),
        "allowed_warning": dict(
            ALLOWED_WARNING
        ),
        "warning_message": finding.get(
            "message",
            "",
        ),
        "source_stage0_modified": False,
        "derived_stage0_status_override": (
            "PASS_WITH_WARNINGS_TO_PASS"
        ),
        "padding_used": False,
        "threshold_change": False,
        "manual_later_fragment_link_used": False,
    }

    compat_audit_path = (
        compat_dir
        / "audit.json"
    )

    compat_manifest_path = (
        compat_dir
        / "input_manifest.json"
    )

    patch_derived_status(
        compat_audit_path,
        compatibility_authorization,
    )

    patch_derived_status(
        compat_manifest_path,
        compatibility_authorization,
    )

    source_hashes_after = {
        path.name: sha256_file(
            path
        )
        for path in (
            source_audit_path,
            source_manifest_path,
            source_findings_path,
        )
    }

    if source_hashes_before != source_hashes_after:
        raise RuntimeError(
            "Original Stage-0 artifacts changed unexpectedly"
        )

    authorization_payload = {
        "stage": STAGE,
        "version": VERSION,
        "generated_at": generated_at,
        "status": "AUTHORIZED",
        "source_stage0_test_name": (
            source_test_name
        ),
        "compat_test_name": (
            compat_test_name
        ),
        "allowed_finding": finding,
        "video": video_contract,
        "source_stage0": {
            "directory": str(
                source_dir
            ),
            "preserved_unchanged": True,
            "hashes_before": source_hashes_before,
            "hashes_after": source_hashes_after,
        },
        "derived_stage0": {
            "directory": str(
                compat_dir
            ),
            "audit_status": "PASS",
            "input_manifest_status": "PASS",
            "authorization": (
                compatibility_authorization
            ),
        },
        "safety_invariants": {
            "source_stage0_modified": False,
            "frozen_phase1_code_modified": False,
            "padded_frames_used": False,
            "tracking_threshold_changed": False,
            "reid_threshold_changed": False,
            "manual_same_player_link_used": False,
        },
    }

    authorization_path = (
        v2_output_dir
        / "stage3c0_short_clip_compatibility_authorization.json"
    )

    atomic_json(
        authorization_path,
        authorization_payload,
    )

    script_paths = {
        name: (
            phase1_code_dir
            / name
        ).resolve()
        for name in FROZEN_STAGE_SCRIPTS
    }

    for path in script_paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    log_lines: list[str] = []

    commands = (
        (
            sys.executable,
            str(
                script_paths[
                    "stage1_generate_rfdetr_detections.py"
                ]
            ),
            "--test-name",
            compat_test_name,
            "--device",
            arguments.device,
        ),
        (
            sys.executable,
            str(
                script_paths[
                    "stage2_run_conservative_target_association.py"
                ]
            ),
            "--test-name",
            compat_test_name,
        ),
        (
            sys.executable,
            str(
                script_paths[
                    "stage2b_run_same_shot_reentry_reacquisition.py"
                ]
            ),
            "--test-name",
            compat_test_name,
            "--device",
            arguments.device,
        ),
        (
            sys.executable,
            str(
                script_paths[
                    "phase1_finalize_latest_timeline.py"
                ]
            ),
            "--test-name",
            compat_test_name,
        ),
        (
            sys.executable,
            str(
                script_paths[
                    "phase1_validate_outputs.py"
                ]
            ),
            "--test-name",
            compat_test_name,
        ),
    )

    for command in commands:
        run_command(
            command,
            root,
            log_lines,
        )

    log_path = (
        v2_output_dir
        / "stage3c0_execution_log.txt"
    )

    atomic_text(
        log_path,
        "".join(
            log_lines
        ),
    )

    generated_files = sorted(
        str(
            path.relative_to(
                compat_dir
            )
        )
        for path in compat_dir.rglob(
            "*"
        )
        if path.is_file()
    )

    preview_files = sorted(
        str(
            path
        )
        for path in compat_dir.rglob(
            "*.mp4"
        )
        if path.is_file()
    )

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "generated_at": generated_at,
        "status": "PASS",
        "decision": (
            "AUTHORIZE_MANDATORY_"
            "STAGE3C0_SHORT_CLIP_"
            "VISUAL_REVIEW"
        ),
        "v2_test_name": (
            v2_test_name
        ),
        "source_stage0_test_name": (
            source_test_name
        ),
        "compat_test_name": (
            compat_test_name
        ),
        "video": video_contract,
        "compatibility": {
            "finding_count": 1,
            "allowed_warning": dict(
                ALLOWED_WARNING
            ),
            "source_stage0_preserved": True,
            "padding_used": False,
            "derived_status_override": (
                "PASS_WITH_WARNINGS_TO_PASS"
            ),
        },
        "frozen_scripts": {
            name: {
                "path": str(
                    path
                ),
                "sha256": sha256_file(
                    path
                ),
            }
            for name, path in script_paths.items()
        },
        "counts": {
            "executed_command_count": len(
                commands
            ),
            "generated_file_count": len(
                generated_files
            ),
            "preview_file_count": len(
                preview_files
            ),
        },
        "outputs": {
            "phase1_compatibility_run": str(
                compat_dir
            ),
            "authorization": str(
                authorization_path
            ),
            "execution_log": str(
                log_path
            ),
            "preview_files": preview_files,
        },
        "safety_invariants": {
            "original_stage0_modified": False,
            "frozen_phase1_code_modified": False,
            "padding_used": False,
            "only_original_short_clip_frames_used": True,
            "postcut_track_0004_used_for_inference": False,
            "manual_same_player_link_used": False,
            "threshold_search_performed": False,
            "target_memory_updated": False,
        },
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
    }

    summary_path = (
        v2_output_dir
        / "stage3c0_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        v2_output_dir
        / "stage3c0_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric Tracking V2 "
        "Stage 3-C0 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        "Decision                       : "
        "AUTHORIZE_MANDATORY_"
        "STAGE3C0_SHORT_CLIP_"
        "VISUAL_REVIEW"
    )

    print(
        f"Source Stage-0 test            : "
        f"{source_test_name}"
    )

    print(
        f"Compatibility test             : "
        f"{compat_test_name}"
    )

    print(
        f"Video frames                   : "
        f"{video_contract['frame_count']}"
    )

    print(
        f"Video duration                 : "
        f"{video_contract['duration_seconds']:.6f}s"
    )

    print(
        "Authorized warning             : "
        "DURATION_OUTSIDE_RANGE"
    )

    print(
        "Original Stage-0 modified      : NONE"
    )

    print(
        "Frozen Phase-1 code modified   : NONE"
    )

    print(
        "Padding frames                 : NONE"
    )

    print(
        "Manual later-fragment link     : NONE"
    )

    print(
        f"Preview files                  : "
        f"{len(preview_files)}"
    )

    for preview_path in preview_files:
        print(
            f"  {preview_path}"
        )

    print(
        f"Phase-1 output                 : "
        f"{compat_dir}"
    )

    print(
        f"Stage 3-C0 output              : "
        f"{v2_output_dir}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "Stage 3-C0 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-C0 fatal error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )