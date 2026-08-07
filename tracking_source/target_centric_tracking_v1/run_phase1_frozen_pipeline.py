#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the frozen KickClip Phase-1 target-centric tracking pipeline.

The runner never changes thresholds. It verifies the freeze manifest, executes
existing stage scripts, stops at mandatory human-review gates, and finalizes the
latest valid timeline into stable final_* artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_D = "SAFE_NO_ADDITIONAL_AUTOMATIC_REENTRY_REQUIRE_REVIEW_OR_UI"
SAFE_D1 = "SAFE_NO_ADDITIONAL_TEMPORAL_REENTRY_REQUIRE_REVIEW_OR_UI"
REVIEW_D1 = "AUTHORIZE_MANDATORY_STAGE2D1_TEMPORAL_REENTRY_VISUAL_REVIEW"
REVIEW_D2 = "AUTHORIZE_MANDATORY_STAGE2D2_HYSTERESIS_VISUAL_REVIEW"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--test-name", required=True)
    p.add_argument("--initial-bbox", nargs=4, type=float, required=True)
    p.add_argument("--bbox-format", choices=("xyxy_pixels", "xywh_pixels", "xyxy_normalized"), default="xyxy_pixels")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    p.add_argument("--manifest", type=Path, default=Path("target_centric_tracking_v1/phase1_frozen_manifest.json"))
    p.add_argument("--stage2b-visual-review", choices=("PASS", "FAIL"), default=None)
    p.add_argument("--stage2d-visual-review", choices=("PASS", "FAIL"), default=None)
    p.add_argument("--stage2d1-visual-review", choices=("PASS", "FAIL"), default=None)
    p.add_argument("--stage2d2-visual-review", choices=("PASS", "FAIL"), default=None)
    p.add_argument("--reviewer", default="USER")
    p.add_argument("--review-note", default="Frozen Phase-1 visual review")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-hash-check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_test_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not name or name in {".", ".."} or any(ch not in allowed for ch in name):
        raise ValueError("Invalid --test-name")
    return name


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    for logical, record in manifest["scripts"].items():
        if not record.get("path"):
            continue
        path = resolve(root, Path(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Frozen script missing ({logical}): {path}")
        observed = sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Frozen script changed ({logical}): {path}\nexpected={record['sha256']}\nobserved={observed}")
    for name, record in manifest["models"].items():
        if not record.get("verified"):
            continue
        path = resolve(root, Path(record["path"]))
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen model changed: {name}")
    for key in ("policy", "schema"):
        path = resolve(root, Path(manifest["contracts"][f"{key}_path"]))
        if sha256(path) != manifest["contracts"][f"{key}_sha256"]:
            raise RuntimeError(f"Frozen contract changed: {key}")
    return manifest


def script_path(root: Path, manifest: dict[str, Any], logical: str) -> Path:
    record = manifest["scripts"][logical]
    if not record.get("path"):
        raise FileNotFoundError(f"No frozen script for {logical}")
    return resolve(root, Path(record["path"]))


def command_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"[PHASE1] {command_text(cmd)}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {command_text(cmd)}")


def state_path(test_dir: Path) -> Path:
    return test_dir / "phase1_pipeline_state.json"


def write_state(test_dir: Path, status: str, **extra: Any) -> None:
    value = {
        "schema_version": "kickclip.phase1_pipeline_state.v1",
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": status,
        **extra,
    }
    test_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_path(test_dir).with_name("phase1_pipeline_state.json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, state_path(test_dir))


def selected_candidate(summary: dict[str, Any]) -> bool:
    selection = summary.get("selection") or {}
    return bool(selection.get("selected_tracklet_id") or selection.get("selected_candidate"))


def run_stage_if_missing(
    output_marker: Path,
    cmd: list[str],
    root: Path,
    dry_run: bool,
    overwrite: bool,
    overwrite_flag: str | None,
) -> None:
    if output_marker.exists() and not overwrite:
        print(f"[PHASE1] reuse: {output_marker}", flush=True)
        return
    if overwrite and overwrite_flag:
        cmd = [*cmd, overwrite_flag]
    run(cmd, root, dry_run)


def main() -> int:
    a = parse_args()
    root = a.project_root.expanduser().resolve()
    video = resolve(root, a.video)
    manifest_path = resolve(root, a.manifest)
    test_name = validate_test_name(a.test_name)
    if not video.is_file() and not a.dry_run:
        raise FileNotFoundError(video)
    manifest = read_json(manifest_path) if a.skip_hash_check else verify_manifest(root, manifest_path)
    test_dir = root / "runs" / "target_centric_tracking_v1" / test_name
    python = sys.executable

    stage0 = [python, str(script_path(root, manifest, "stage0")), "--video", str(video), "--test-name", test_name,
              "--initial-bbox", *[str(v) for v in a.initial_bbox], "--bbox-format", a.bbox_format]
    run_stage_if_missing(test_dir / "audit.json", stage0, root, a.dry_run, a.overwrite, "--overwrite-audit")

    stage1 = [python, str(script_path(root, manifest, "stage1")), "--test-name", test_name, "--device", a.device]
    run_stage_if_missing(test_dir / "stage1_detection_summary.json", stage1, root, a.dry_run, a.overwrite, "--overwrite-stage1")

    stage2 = [python, str(script_path(root, manifest, "stage2")), "--test-name", test_name]
    run_stage_if_missing(test_dir / "stage2_association_summary.json", stage2, root, a.dry_run, a.overwrite, "--overwrite-stage2")

    stage2b = [python, str(script_path(root, manifest, "stage2b")), "--test-name", test_name, "--device", a.device]
    run_stage_if_missing(test_dir / "stage2b_reentry_summary.json", stage2b, root, a.dry_run, a.overwrite, "--overwrite-stage2b")
    if a.dry_run:
        print("[PHASE1] dry-run stops before data-dependent review branching")
        return 0

    b_summary = read_json(test_dir / "stage2b_reentry_summary.json")
    if selected_candidate(b_summary):
        if a.stage2b_visual_review == "FAIL":
            write_state(test_dir, "BLOCKED", review_stage="STAGE2B", reason="USER_REVIEW_FAIL")
            raise RuntimeError("Stage 2-B visual review failed; pipeline blocked")
        if a.stage2b_visual_review != "PASS":
            write_state(test_dir, "REVIEW_REQUIRED", review_stage="STAGE2B",
                        preview=str(test_dir / "stage2b_reentry_preview.mp4"))
            print("Review required: Stage 2-B. Re-run with --stage2b-visual-review PASS or FAIL.")
            return 3

        stage2c = [python, str(script_path(root, manifest, "stage2c")), "--test-name", test_name,
                   "--stage2b-visual-review", "PASS", "--reviewer", a.reviewer, "--review-note", a.review_note]
        run_stage_if_missing(test_dir / "stage2c_audit.json", stage2c, root, False, a.overwrite, "--overwrite-stage2c")

        stage2d = [python, str(script_path(root, manifest, "stage2d")), "--test-name", test_name, "--device", a.device]
        run_stage_if_missing(test_dir / "stage2d_summary.json", stage2d, root, False, a.overwrite, "--overwrite-stage2d")
        d_summary = read_json(test_dir / "stage2d_summary.json")
        d_new = int((d_summary.get("counts") or {}).get("new_selected_episode_count", 0))
        if d_new > 0:
            if a.stage2d_visual_review == "FAIL":
                write_state(test_dir, "BLOCKED", review_stage="STAGE2D", reason="USER_REVIEW_FAIL")
                raise RuntimeError("Stage 2-D visual review failed; pipeline blocked")
            if a.stage2d_visual_review != "PASS":
                write_state(test_dir, "REVIEW_REQUIRED", review_stage="STAGE2D",
                            preview=str(test_dir / "stage2d_multi_reentry_preview.mp4"))
                print("Review required: Stage 2-D. Re-run with --stage2d-visual-review PASS or FAIL.")
                return 3
        elif d_summary.get("decision") == SAFE_D:
            stage2d1 = [python, str(script_path(root, manifest, "stage2d1")), "--test-name", test_name, "--device", a.device]
            run_stage_if_missing(test_dir / "stage2d1_summary.json", stage2d1, root, False, a.overwrite, "--overwrite-stage2d1")
            d1_summary = read_json(test_dir / "stage2d1_summary.json")
            if d1_summary.get("decision") == REVIEW_D1:
                if a.stage2d1_visual_review == "FAIL":
                    write_state(test_dir, "BLOCKED", review_stage="STAGE2D1", reason="USER_REVIEW_FAIL")
                    raise RuntimeError("Stage 2-D1 visual review failed; pipeline blocked")
                if a.stage2d1_visual_review != "PASS":
                    write_state(test_dir, "REVIEW_REQUIRED", review_stage="STAGE2D1",
                                preview=str(test_dir / "stage2d1_multi_reentry_preview.mp4"))
                    print("Review required: Stage 2-D1. Re-run with --stage2d1-visual-review PASS or FAIL.")
                    return 3
                stage2d2 = [python, str(script_path(root, manifest, "stage2d2")), "--test-name", test_name]
                run_stage_if_missing(test_dir / "stage2d2_summary.json", stage2d2, root, False, a.overwrite, "--overwrite-stage2d2")
                d2_summary = read_json(test_dir / "stage2d2_summary.json")
                if d2_summary.get("decision") == REVIEW_D2:
                    if a.stage2d2_visual_review == "FAIL":
                        write_state(test_dir, "BLOCKED", review_stage="STAGE2D2", reason="USER_REVIEW_FAIL")
                        raise RuntimeError("Stage 2-D2 visual review failed; pipeline blocked")
                    if a.stage2d2_visual_review != "PASS":
                        write_state(test_dir, "REVIEW_REQUIRED", review_stage="STAGE2D2",
                                    preview=str(test_dir / "stage2d2_hysteresis_preview.mp4"))
                        print("Review required: Stage 2-D2. Re-run with --stage2d2-visual-review PASS or FAIL.")
                        return 3
            elif d1_summary.get("decision") != SAFE_D1:
                raise RuntimeError(f"Unexpected Stage 2-D1 decision: {d1_summary.get('decision')}")
    else:
        print("[PHASE1] Stage 2-B made no automatic re-entry selection; finalizing safe timeline.")

    finalizer = script_path(root, manifest, "finalizer")
    cmd = [python, str(finalizer), "--test-name", test_name]
    if a.overwrite:
        cmd.append("--overwrite-final")
    run(cmd, root, False)

    validator = script_path(root, manifest, "validator")
    run([python, str(validator), "--test-name", test_name], root, False)
    write_state(test_dir, "COMPLETE", final_timeline=str(test_dir / "final_target_timeline.json"),
                final_preview=str(test_dir / "final_target_centered_preview.mp4"))
    print("KickClip frozen Phase-1 pipeline complete")
    print(f"Output: {test_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Frozen runner fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
