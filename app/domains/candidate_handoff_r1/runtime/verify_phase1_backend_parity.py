#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FIXTURE_SCHEMA = "kickclip.phase1_backend_parity_fixture.r1"
REPORT_SCHEMA = "kickclip.phase1_backend_parity_report.r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen experiment fixture directly and through the backend "
            "compatibility executor, then compare final Stage 2-D2 timelines."
        )
    )
    parser.add_argument("--backend-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tracking-project-root",
        type=Path,
        default=None,
        help=(
            "Tracking research root containing target_centric_tracking_v1. "
            "Defaults to the TRACKING_PROJECT_ROOT environment variable."
        ),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "app/domains/candidate_handoff_r1/runtime/"
            "phase1_backend_parity_fixture.json"
        ),
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--reference-test-name",
        default="backend_parity_reference_phase1_20s",
    )
    parser.add_argument(
        "--backend-test-name",
        default="backend_parity_compat_phase1_20s",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "runs/backend_phase1_parity/phase1_backend_parity_report.json"
        ),
    )
    parser.add_argument("--bbox-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--confidence-absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_executor(path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(
        "kickclip_phase1_compatibility_executor_parity",
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def phase1_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    discovered = shutil.which("ffmpeg", path=environment.get("PATH"))
    if discovered:
        environment["PATH"] = str(Path(discovered).resolve().parent) + os.pathsep + environment.get(
            "PATH", ""
        )
    return environment


def run(command: Sequence[str], *, cwd: Path) -> None:
    print("[PARITY] " + subprocess.list2cmdline(list(command)), flush=True)
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        shell=False,
        check=False,
        env=phase1_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Parity command failed ({completed.returncode}): "
            + subprocess.list2cmdline(list(command))
        )


def validate_fixture(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("Unexpected parity fixture schema.")
    bbox = value.get("initial_bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Fixture initial_bbox_xyxy must contain four values.")
    reviews = value.get("review_decisions")
    if not isinstance(reviews, dict) or not reviews:
        raise ValueError("Fixture review_decisions are required.")
    if value.get("review_provenance") != "PREVIOUSLY_HUMAN_REVIEWED_EXPERIMENT_FIXTURE":
        raise ValueError("Parity fixture must preserve explicit human-review provenance.")


def remove_test_output(root: Path, test_name: str) -> None:
    path = root / "runs" / "target_centric_tracking_v1" / test_name
    if path.exists():
        shutil.rmtree(path)


def direct_runner_command(
    *,
    root: Path,
    video: Path,
    test_name: str,
    fixture: Mapping[str, Any],
    device: str,
    overwrite: bool,
) -> list[str]:
    runner = root / "target_centric_tracking_v1" / "run_phase1_frozen_pipeline.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)

    command = [
        sys.executable,
        str(runner),
        "--project-root",
        str(root),
        "--video",
        str(video),
        "--test-name",
        test_name,
        "--initial-bbox",
        *[str(value) for value in fixture["initial_bbox_xyxy"]],
        "--bbox-format",
        "xyxy_pixels",
        "--device",
        device,
        "--reviewer",
        str(fixture["reviewer"]),
        "--review-note",
        str(fixture["review_note"]),
    ]
    argument_by_stage = {
        "STAGE2B": "--stage2b-visual-review",
        "STAGE2D": "--stage2d-visual-review",
        "STAGE2D1": "--stage2d1-visual-review",
        "STAGE2D2": "--stage2d2-visual-review",
    }
    for stage in ("STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"):
        decision = fixture["review_decisions"].get(stage)
        if decision:
            command.extend([argument_by_stage[stage], str(decision)])
    if overwrite:
        command.append("--overwrite")
    return command


def summarize_timeline(timeline: Mapping[str, Any]) -> dict[str, Any]:
    frames = timeline.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Timeline frames are missing.")
    states = [str(frame.get("state") or "") for frame in frames]
    state_counts = Counter(states)
    bbox_frames = [
        int(frame.get("frame_index", index))
        for index, frame in enumerate(frames)
        if frame.get("bbox_xyxy") is not None
    ]
    return {
        "schema_version": timeline.get("schema_version"),
        "frame_count": len(frames),
        "bbox_frame_count": len(bbox_frames),
        "state_counts": dict(sorted(state_counts.items())),
        "first_lost_frame": next(
            (index for index, state in enumerate(states) if state == "LOST"),
            None,
        ),
        "first_searching_frame": next(
            (index for index, state in enumerate(states) if state == "SEARCHING"),
            None,
        ),
        "reacquired_frames": [
            index for index, state in enumerate(states) if state == "REACQUIRED"
        ],
        "source_stage": (timeline.get("provenance") or {}).get("source_stage"),
        "video": dict(timeline.get("video") or {}),
    }


def compare_number(left: Any, right: Any, tolerance: float) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return left == right


def compare_timelines(
    reference: Mapping[str, Any],
    backend: Mapping[str, Any],
    *,
    bbox_tolerance: float,
    confidence_tolerance: float,
) -> dict[str, Any]:
    reference_frames = reference.get("frames")
    backend_frames = backend.get("frames")
    if not isinstance(reference_frames, list) or not isinstance(backend_frames, list):
        raise ValueError("Timeline frame arrays are required.")

    mismatches: list[dict[str, Any]] = []
    compared = min(len(reference_frames), len(backend_frames))
    for index in range(compared):
        left = reference_frames[index]
        right = backend_frames[index]
        reasons: list[str] = []
        for key in ("frame_index", "state", "selected_detection_id", "review_required"):
            if left.get(key) != right.get(key):
                reasons.append(key)

        left_bbox = left.get("bbox_xyxy")
        right_bbox = right.get("bbox_xyxy")
        if (left_bbox is None) != (right_bbox is None):
            reasons.append("bbox_presence")
        elif left_bbox is not None and right_bbox is not None:
            if len(left_bbox) != 4 or len(right_bbox) != 4:
                reasons.append("bbox_schema")
            elif any(
                not compare_number(lvalue, rvalue, bbox_tolerance)
                for lvalue, rvalue in zip(left_bbox, right_bbox)
            ):
                reasons.append("bbox_xyxy")

        for key in ("tracking_confidence", "identity_confidence"):
            if not compare_number(left.get(key), right.get(key), confidence_tolerance):
                reasons.append(key)

        if reasons:
            mismatches.append(
                {
                    "frame_index": index,
                    "reason_codes": reasons,
                    "reference": {
                        "state": left.get("state"),
                        "bbox_xyxy": left_bbox,
                        "tracking_confidence": left.get("tracking_confidence"),
                        "identity_confidence": left.get("identity_confidence"),
                    },
                    "backend": {
                        "state": right.get("state"),
                        "bbox_xyxy": right_bbox,
                        "tracking_confidence": right.get("tracking_confidence"),
                        "identity_confidence": right.get("identity_confidence"),
                    },
                }
            )
            if len(mismatches) >= 100:
                break

    return {
        "reference_frame_count": len(reference_frames),
        "backend_frame_count": len(backend_frames),
        "frame_count_match": len(reference_frames) == len(backend_frames),
        "compared_frame_count": compared,
        "mismatch_count_capped": len(mismatches),
        "first_mismatches": mismatches,
        "exact_semantic_match": (
            len(reference_frames) == len(backend_frames) and not mismatches
        ),
    }


def expected_checks(summary: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    expected = fixture.get("expected_final_metrics") or {}
    checks = {
        "frame_count": summary.get("frame_count") == expected.get("frame_count"),
        "bbox_frame_count": summary.get("bbox_frame_count")
        == expected.get("bbox_frame_count"),
        "state_counts": summary.get("state_counts") == expected.get("state_counts"),
        "source_stage": summary.get("source_stage") == expected.get("source_stage"),
    }
    return {
        "expected": expected,
        "observed": dict(summary),
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    arguments = parse_args()
    backend_root = arguments.backend_root.expanduser().resolve()
    configured_tracking_root = (
        arguments.tracking_project_root
        if arguments.tracking_project_root is not None
        else Path(os.environ["TRACKING_PROJECT_ROOT"])
        if os.environ.get("TRACKING_PROJECT_ROOT")
        else None
    )
    if configured_tracking_root is None:
        raise ValueError(
            "--tracking-project-root is required when TRACKING_PROJECT_ROOT is unset."
        )
    root = configured_tracking_root.expanduser().resolve()
    fixture_path = resolve(backend_root, arguments.fixture)
    fixture = read_json(fixture_path)
    validate_fixture(fixture)

    video = (
        resolve(root, arguments.video)
        if arguments.video is not None
        else resolve(root, Path(str(fixture["video_relative_path"])))
    )
    if not video.is_file():
        raise FileNotFoundError(video)

    if arguments.overwrite:
        remove_test_output(root, arguments.reference_test_name)
        remove_test_output(root, arguments.backend_test_name)

    reference_dir = root / "runs" / "target_centric_tracking_v1" / arguments.reference_test_name
    backend_dir = root / "runs" / "target_centric_tracking_v1" / arguments.backend_test_name

    reference_timeline_path = reference_dir / "final_target_timeline.json"
    if not reference_timeline_path.is_file() or arguments.overwrite:
        run(
            direct_runner_command(
                root=root,
                video=video,
                test_name=arguments.reference_test_name,
                fixture=fixture,
                device=arguments.device,
                overwrite=arguments.overwrite,
            ),
            cwd=root,
        )

    executor_path = (
        backend_root
        / "app"
        / "domains"
        / "candidate_handoff_r1"
        / "runtime"
        / "phase1_compatibility_executor.py"
    )
    executor = load_executor(executor_path)
    gate_path = (
        root
        / "runs"
        / "backend_phase1_parity"
        / "backend_compatibility_gate.json"
    )
    backend_timeline_path = backend_dir / "final_target_timeline.json"
    if not backend_timeline_path.is_file() or arguments.overwrite:
        executor.execute_phase1_compatibility(
            project_root=root,
            video=video,
            initial_bbox=fixture["initial_bbox_xyxy"],
            original_stage0_test_name=(
                arguments.backend_test_name + "__stage0_original"
            ),
            phase1_test_name=arguments.backend_test_name,
            compatibility_test_name=(
                arguments.backend_test_name + "__short_clip_compat"
            ),
            device=arguments.device,
            gate_path=gate_path,
            review_decisions=fixture["review_decisions"],
            reviewer=str(fixture["reviewer"]),
            review_note=str(fixture["review_note"]),
            allow_review_required=False,
            require_final_outputs=True,
            overwrite_phase1=arguments.overwrite,
        )

    reference_timeline = read_json(reference_timeline_path)
    backend_timeline = read_json(backend_timeline_path)
    reference_summary = summarize_timeline(reference_timeline)
    backend_summary = summarize_timeline(backend_timeline)
    comparison = compare_timelines(
        reference_timeline,
        backend_timeline,
        bbox_tolerance=arguments.bbox_absolute_tolerance,
        confidence_tolerance=arguments.confidence_absolute_tolerance,
    )
    reference_expected = expected_checks(reference_summary, fixture)
    backend_expected = expected_checks(backend_summary, fixture)

    status = (
        "PASS"
        if comparison["exact_semantic_match"]
        and reference_expected["pass"]
        and backend_expected["pass"]
        else "FAIL"
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": now(),
        "status": status,
        "decision": (
            "AUTHORIZE_PHASE2_SELECTED_SHOT_WORK"
            if status == "PASS"
            else "BLOCK_PHASE2_FIX_BACKEND_PARITY_FIRST"
        ),
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256(fixture_path),
            "video_path": str(video),
            "video_sha256": sha256(video),
            "initial_bbox_xyxy": fixture["initial_bbox_xyxy"],
            "review_decisions": fixture["review_decisions"],
            "review_provenance": fixture["review_provenance"],
            "automatic_review_decision_used": False,
        },
        "reference": {
            "test_name": arguments.reference_test_name,
            "timeline_path": str(reference_timeline_path),
            "timeline_sha256": sha256(reference_timeline_path),
            "summary": reference_summary,
            "expected_metric_check": reference_expected,
        },
        "backend": {
            "test_name": arguments.backend_test_name,
            "timeline_path": str(backend_timeline_path),
            "timeline_sha256": sha256(backend_timeline_path),
            "compatibility_gate_path": str(gate_path),
            "compatibility_gate_sha256": sha256(gate_path),
            "summary": backend_summary,
            "expected_metric_check": backend_expected,
        },
        "comparison": comparison,
        "tolerances": {
            "bbox_absolute": arguments.bbox_absolute_tolerance,
            "confidence_absolute": arguments.confidence_absolute_tolerance,
        },
    }
    report_path = resolve(root, arguments.report)
    atomic_json(report_path, report)

    print("KickClip Phase-1 backend parity verification complete")
    print(f"Status   : {status}")
    print(f"Decision : {report['decision']}")
    print(f"Frames   : {backend_summary['frame_count']}")
    print(f"BBox     : {backend_summary['bbox_frame_count']}")
    print(f"Stage    : {backend_summary['source_stage']}")
    print(f"Report   : {report_path}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(
            f"Parity verification fatal error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
