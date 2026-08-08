#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip product target-centric E2E entrypoint.

P0 product integration responsibilities:
- preserve frozen V1/V2 source trees;
- track the full Action-Spotting scene around a user-confirmed anchor by running
  the canonical one-direction core both forward and backward;
- pass the user-selected immutable reference gallery into identity scoring;
- pass the configured RF-DETR play confidence (0.15 by default) into the
  Stage-1 compatibility layer;
- preserve precision-first ambiguity behavior while allowing frozen safe-gate
  PASS reacquisition without human review, and merge both directions into one
  source-coordinate target timeline.

Without ``--full-scene`` this file delegates to the one-direction core for
backward compatibility.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import cv2

WRAPPER_SCHEMA = "kickclip.target_centric_full_scene.v1"
WRAPPER_VERSION = "1.2.1-product-assisted-intent"
TERMINAL = {
    "COMPLETE",
    "COMPLETE_WITH_SAFE_BLOCK",
    "COMPLETE_WITH_UNRESOLVED_GAPS",
    "BLOCKED",
}
RESUMABLE_CHILD_STATUS = "RUNNING"
MAX_INTERNAL_CONTINUATION_HOPS = 4
CONFIRMED_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "ACTIVE_LOW_CONFIDENCE",
    "OCCLUDED",
    "REACQUIRED",
    "USER_CONFIRMED",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def validate_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not value or value in {".", ".."} or any(ch not in allowed for ch in value):
        raise ValueError(f"Invalid test name: {value}")
    return value


def load_core(script: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kickclip_target_centric_e2e_core", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load E2E core: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KickClip full-scene target-centric product runner")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--test-name", required=True)
    p.add_argument("--initial-bbox", nargs=4, type=float, default=None)
    p.add_argument("--initial-frame", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    p.add_argument("--reacquisition-mode", choices=("assisted", "auto-safe"), default="assisted")
    p.add_argument("--cut-frames", nargs="*", type=int, default=None)
    p.add_argument("--phase1-manifest", type=Path, default=Path("target_centric_tracking_v1/phase1_frozen_manifest.json"))
    p.add_argument("--output-root", type=Path, default=Path("runs/target_centric_tracking_e2e_v1"))
    p.add_argument("--reviewer", default="USER")
    p.add_argument("--review-note", default="")
    p.add_argument("--backend-memory-revision", type=Path, default=None)
    p.add_argument("--backend-memory-sha256", default=None)
    p.add_argument("--backend-safe-weights-runner", type=Path, default=None)
    p.add_argument("--tracking-play-conf-threshold", type=float, default=0.15)
    p.add_argument("--target-reference-set", type=Path, default=None)
    p.add_argument("--target-reference-set-sha256", default=None)
    p.add_argument("--trusted-selected-reference-memory", action="store_true")
    p.add_argument("--full-scene", action="store_true")

    p.add_argument("--resume", action="store_true")
    p.add_argument("--ambiguity-id", default=None)
    decision = p.add_mutually_exclusive_group()
    decision.add_argument("--confirmed-candidate", default=None)
    decision.add_argument("--confirm-absent", action="store_true")
    decision.add_argument("--reject-all-candidates", action="store_true")
    decision.add_argument("--reject-all-candidates-as-non-player-role", action="store_true")
    decision.add_argument("--rejected-candidate", default=None)
    decision.add_argument("--unreviewable-candidate", default=None)
    review = p.add_mutually_exclusive_group()
    review.add_argument("--approve-review", choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"), default=None)
    review.add_argument("--reject-review", choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"), default=None)
    p.add_argument("--review-decision-artifact", type=Path, default=None)
    p.add_argument("--review-decision-sha256", default=None)
    p.add_argument("--allow-prestaged-output", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--print-every", type=int, default=50)
    return p.parse_args()


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }


def make_reverse_prefix(source: Path, output: Path, anchor: int, metadata: Mapping[str, Any]) -> None:
    expected = anchor + 1
    if output.is_file():
        existing = video_metadata(output)
        if int(existing["frame_count"]) == expected:
            return
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    width = int(metadata["width"])
    height = int(metadata["height"])
    fps = float(metadata["fps"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create reverse video: {output}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open source video: {source}")
    written = 0
    try:
        for original_index in range(anchor, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot read source frame {original_index} for reverse tracking")
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    if written != expected:
        raise RuntimeError(f"Reverse prefix frame mismatch: {written}/{expected}")


def normalized_cuts(cuts: Sequence[int] | None, frame_count: int) -> list[int]:
    return sorted({int(value) for value in (cuts or []) if 0 < int(value) < frame_count})


def shots_from_cuts(cuts: Sequence[int], frame_count: int) -> list[dict[str, Any]]:
    starts = [0, *normalized_cuts(cuts, frame_count)]
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = (starts[index + 1] - 1) if index + 1 < len(starts) else frame_count - 1
        rows.append(
            {
                "shot_index": index,
                "shot_id": f"shot_{index:04d}",
                "start_frame": start,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                "cut_in_frame": None if start == 0 else start,
                "cut_out_frame": None if end == frame_count - 1 else end + 1,
            }
        )
    return rows


def shot_id_for_frame(frame: int, shots: Sequence[Mapping[str, Any]]) -> str:
    for shot in shots:
        if int(shot["start_frame"]) <= frame <= int(shot["end_frame_inclusive"]):
            return str(shot["shot_id"])
    return str(shots[-1]["shot_id"])


def core_script(root: Path) -> Path:
    path = root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline_core.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_subprocess(command: list[str], root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[COMMAND] " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log.write(completed.stdout or "")
        log.flush()
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        return int(completed.returncode)


def delegate_non_full_scene(args: argparse.Namespace, root: Path) -> int:
    command = [sys.executable, str(core_script(root))]
    for token in sys.argv[1:]:
        if token != "--full-scene":
            command.append(token)
    return subprocess.run(command, cwd=str(root), check=False).returncode


def child_layout(parent: Path, test_name: str, direction: str) -> dict[str, Path | str]:
    child_name = f"{test_name}__{direction}"
    child_root = parent / "directions"
    return {
        "test_name": child_name,
        "output_root": child_root,
        "output_dir": child_root / child_name,
    }


def build_common_child_args(state: Mapping[str, Any], direction: str) -> list[str]:
    cfg = state["config"]
    result = [
        "--project-root", str(cfg["project_root"]),
        "--test-name", str(state["directions"][direction]["test_name"]),
        "--device", str(cfg["device"]),
        "--reacquisition-mode", str(cfg["reacquisition_mode"]),
        "--output-root", str(state["directions"][direction]["output_root"]),
        "--phase1-manifest", str(cfg["phase1_manifest"]),
        "--tracking-play-conf-threshold", str(cfg["tracking_play_conf_threshold"]),
        "--reviewer", str(cfg.get("reviewer") or "USER"),
    ]
    note = str(cfg.get("review_note") or "")
    if note:
        result.extend(["--review-note", note])
    if cfg.get("no_preview"):
        result.append("--no-preview")
    result.extend(["--print-every", str(cfg.get("print_every", 50))])
    reference = cfg.get("target_reference_set")
    if reference:
        result.extend([
            "--target-reference-set", str(reference["path"]),
            "--target-reference-set-sha256", str(reference["sha256"]),
        ])
        if cfg.get("trusted_selected_reference_memory"):
            result.append("--trusted-selected-reference-memory")
    if cfg.get("backend_safe_weights_runner"):
        result.extend(["--backend-safe-weights-runner", str(cfg["backend_safe_weights_runner"])])
    return result


def build_child_new_command(state: Mapping[str, Any], direction: str) -> list[str]:
    cfg = state["config"]
    command = [sys.executable, str(cfg["core_script"]), *build_common_child_args(state, direction)]
    if direction == "forward":
        command.extend([
            "--video", str(cfg["source_video"]),
            "--initial-bbox", *[str(v) for v in cfg["initial_bbox_xyxy"]],
            "--initial-frame", str(cfg["anchor_frame"]),
        ])
        cuts = cfg["cut_frames"]
    else:
        command.extend([
            "--video", str(cfg["reverse_video"]),
            "--initial-bbox", *[str(v) for v in cfg["initial_bbox_xyxy"]],
            "--initial-frame", "0",
        ])
        cuts = cfg["backward_cut_frames"]
    if cuts:
        command.extend(["--cut-frames", *[str(v) for v in cuts]])
    return command


def parent_to_child_ambiguity(value: str | None, direction: str) -> str | None:
    if value is None:
        return None
    prefix = f"{direction}:"
    return value[len(prefix):] if value.startswith(prefix) else value


def build_child_resume_command(state: Mapping[str, Any], direction: str, args: argparse.Namespace) -> list[str]:
    cfg = state["config"]
    command = [sys.executable, str(cfg["core_script"]), *build_common_child_args(state, direction), "--resume"]
    if args.approve_review:
        command.extend(["--approve-review", args.approve_review])
    if args.reject_review:
        command.extend(["--reject-review", args.reject_review])
    ambiguity = parent_to_child_ambiguity(args.ambiguity_id, direction)
    if ambiguity:
        command.extend(["--ambiguity-id", ambiguity])
    if args.confirmed_candidate:
        command.extend(["--confirmed-candidate", args.confirmed_candidate])
    elif args.confirm_absent:
        command.append("--confirm-absent")
    elif args.reject_all_candidates:
        command.append("--reject-all-candidates")
    elif args.reject_all_candidates_as_non_player_role:
        command.append("--reject-all-candidates-as-non-player-role")
    elif args.rejected_candidate:
        command.extend(["--rejected-candidate", args.rejected_candidate])
    elif args.unreviewable_candidate:
        command.extend(["--unreviewable-candidate", args.unreviewable_candidate])
    if args.review_decision_artifact:
        command.extend(["--review-decision-artifact", str(args.review_decision_artifact.expanduser().resolve())])
    if args.review_decision_sha256:
        command.extend(["--review-decision-sha256", args.review_decision_sha256])
    if args.reviewer:
        command.extend(["--reviewer", args.reviewer])
    if args.review_note:
        command.extend(["--review-note", args.review_note])
    return command


def build_child_continue_command(state: Mapping[str, Any], direction: str) -> list[str]:
    """Resume a durable RUNNING child without inventing a user decision."""
    cfg = state["config"]
    return [
        sys.executable,
        str(cfg["core_script"]),
        *build_common_child_args(state, direction),
        "--resume",
    ]


def read_child_state(state: Mapping[str, Any], direction: str) -> dict[str, Any] | None:
    path = Path(str(state["directions"][direction]["output_dir"])) / "pipeline_state.json"
    return read_json(path) if path.is_file() else None


def public_pending(direction: str, child_state: Mapping[str, Any]) -> dict[str, Any] | None:
    pending = child_state.get("pending_action")
    if not isinstance(pending, Mapping):
        return None
    value = dict(pending)
    value["direction"] = direction
    if value.get("ambiguity_id"):
        value["ambiguity_id"] = f"{direction}:{value['ambiguity_id']}"
    value["child_pending_action"] = dict(pending)
    return value


def initialize_parent(args: argparse.Namespace, root: Path, parent: Path) -> dict[str, Any]:
    if args.video is None or args.initial_bbox is None:
        raise ValueError("New full-scene run requires --video and --initial-bbox")
    source = resolve(root, args.video)
    if not source.is_file():
        raise FileNotFoundError(source)
    meta = video_metadata(source)
    anchor = int(args.initial_frame or 0)
    if anchor < 0 or anchor >= int(meta["frame_count"]):
        raise ValueError("--initial-frame is outside the source video")
    threshold = float(args.tracking_play_conf_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--tracking-play-conf-threshold must be in (0, 1]")
    bbox = [float(value) for value in args.initial_bbox]
    if not (0 <= bbox[0] < bbox[2] <= meta["width"] and 0 <= bbox[1] < bbox[3] <= meta["height"]):
        raise ValueError("--initial-bbox is outside the source frame")
    cuts = normalized_cuts(args.cut_frames, int(meta["frame_count"]))
    backward_cuts = sorted(anchor - cut + 1 for cut in cuts if 0 < cut <= anchor)

    prestaged_state: dict[str, Any] | None = None
    prestaged_state_path = parent / "pipeline_state.json"
    if parent.exists() and prestaged_state_path.exists() and not args.overwrite:
        existing = read_json(prestaged_state_path)
        existing_is_full_scene = bool(existing.get("config", {}).get("full_scene")) or str(
            existing.get("schema_version") or ""
        ) == WRAPPER_SCHEMA
        if existing_is_full_scene:
            raise FileExistsError(f"Tracking output already exists: {parent}")
        if not args.allow_prestaged_output:
            raise FileExistsError(f"Tracking output already exists: {parent}")

        execution_kind = str(existing.get("execution_kind") or "")
        pipeline_version = str(existing.get("pipeline_version") or "")
        if execution_kind != "EVENT_CANDIDATE_HANDOFF_R1" and not pipeline_version.startswith(
            "EVENT_CANDIDATE_HANDOFF_R1"
        ):
            raise RuntimeError(
                "--allow-prestaged-output may only reuse an EVENT_CANDIDATE_HANDOFF_R1 output directory"
            )
        prestaged_state = dict(existing)

    parent.mkdir(parents=True, exist_ok=True)
    if prestaged_state is not None:
        snapshot = parent / "backend_artifacts" / "prestaged_pipeline_state.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(snapshot, prestaged_state)

    if args.overwrite:
        for name in ("directions", "work", "ambiguity_candidates", "backend_artifacts"):
            path = parent / name
            if path.is_dir():
                shutil.rmtree(path)
        for name in (
            "pipeline_state.json", "pipeline_manifest.json", "pipeline_summary.json",
            "target_timeline.json", "target_timeline.csv", "target_segments.json",
            "ambiguities.json", "confirmations.json", "shot_boundaries.csv", "report.md",
            "full_frame_tracking_preview.mp4", "target_centered_preview.mp4",
        ):
            path = parent / name
            if path.is_file():
                path.unlink()

    reference = None
    if args.target_reference_set is not None:
        reference_path = resolve(root, args.target_reference_set)
        expected = str(args.target_reference_set_sha256 or "").strip().lower()
        if not reference_path.is_file() or len(expected) != 64 or sha256_file(reference_path).lower() != expected:
            raise RuntimeError("Target reference set is missing or changed")
        reference = {"path": str(reference_path), "sha256": expected}
    if args.trusted_selected_reference_memory and reference is None:
        raise ValueError("Trusted selected reference memory requires a reference set")

    reverse_video = parent / "work" / "input" / f"anchor_to_start_reversed_f{anchor:09d}.mp4"
    forward = child_layout(parent, args.test_name, "forward")
    backward = child_layout(parent, args.test_name, "backward")
    state = {
        "schema_version": WRAPPER_SCHEMA,
        "pipeline_version": WRAPPER_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "RUNNING",
        "decision": "FULL_SCENE_INITIALIZED",
        "test_name": args.test_name,
        "pending_action": None,
        "active_direction": "forward",
        "directions": {
            "forward": {**{k: str(v) for k, v in forward.items()}, "status": "PENDING"},
            "backward": {**{k: str(v) for k, v in backward.items()}, "status": "PENDING"},
        },
        "config": {
            "project_root": str(root),
            "core_script": str(core_script(root)),
            "source_video": str(source),
            "source_video_sha256": meta["sha256"],
            "source_video_metadata": meta,
            "anchor_frame": anchor,
            "initial_bbox_xyxy": bbox,
            "cut_frames": cuts,
            "backward_cut_frames": backward_cuts,
            "reverse_video": str(reverse_video),
            "device": args.device,
            "reacquisition_mode": args.reacquisition_mode,
            "phase1_manifest": str(resolve(root, args.phase1_manifest)),
            "tracking_play_conf_threshold": threshold,
            "target_reference_set": reference,
            "trusted_selected_reference_memory": bool(args.trusted_selected_reference_memory),
            "backend_safe_weights_runner": str(resolve(root, args.backend_safe_weights_runner)) if args.backend_safe_weights_runner else None,
            "reviewer": args.reviewer,
            "review_note": args.review_note,
            "no_preview": bool(args.no_preview),
            "print_every": int(args.print_every),
            "full_scene": True,
            "allow_prestaged_output": bool(args.allow_prestaged_output),
            "prestaged_handoff_state_preserved": prestaged_state is not None,
            "prestaged_handoff_state_path": (
                str(parent / "backend_artifacts" / "prestaged_pipeline_state.json")
                if prestaged_state is not None
                else None
            ),
        },
        "safety_contract": {
            "silent_wrong_player_switch": "FORBIDDEN",
            "camera_cut_resets_motion": True,
            "assisted_reacquisition": args.reacquisition_mode == "assisted",
            "safe_gate_pass_auto_reacquired": True,
            "ambiguous_only_requires_user_confirmation": True,
            "maximum_review_candidates_per_ambiguity": 3,
            "selected_reference_memory_used_for_scoring": bool(args.trusted_selected_reference_memory),
            "frozen_v1_v2_source_modified": False,
            "v7_runtime_dependency": False,
        },
    }
    atomic_json(parent / "pipeline_state.json", state)
    return state


def save_parent(parent: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    atomic_json(parent / "pipeline_state.json", state)


def transform_child_frame(row: Mapping[str, Any], direction: str, anchor: int, fps: float) -> dict[str, Any]:
    value = dict(row)
    child_index = int(row.get("local_frame_index", row.get("frame_index", 0)))
    original = int(row["frame_index"]) if direction == "forward" else anchor - child_index
    value["child_frame_index"] = child_index
    value["frame_index"] = original
    value["local_frame_index"] = original
    value["time_seconds"] = original / fps
    value["time_ms"] = int(round(1000.0 * original / fps))
    value["tracking_direction"] = direction.upper()
    if value.get("ambiguity_id"):
        value["ambiguity_id"] = f"{direction}:{value['ambiguity_id']}"
    if value.get("selected_detection_id"):
        value["selected_detection_id"] = f"{direction}:{value['selected_detection_id']}"
    return value


def transform_ambiguity(item: Mapping[str, Any], direction: str, anchor: int) -> dict[str, Any]:
    value = dict(item)
    raw_id = str(item.get("ambiguity_id") or "")
    value["ambiguity_id"] = f"{direction}:{raw_id}" if raw_id else raw_id
    value["direction"] = direction
    if direction == "backward":
        if item.get("start_frame") is not None and item.get("end_frame_inclusive") is not None:
            start_local = int(item["start_frame"])
            end_local = int(item["end_frame_inclusive"])
            value["start_frame"] = anchor - end_local
            value["end_frame_inclusive"] = anchor - start_local
    return value


def transform_confirmation(item: Mapping[str, Any], direction: str, anchor: int) -> dict[str, Any]:
    value = dict(item)
    if value.get("ambiguity_id"):
        value["ambiguity_id"] = f"{direction}:{value['ambiguity_id']}"
    value["direction"] = direction
    if item.get("anchor_frame") is not None and direction == "backward":
        value["anchor_frame"] = anchor - int(item["anchor_frame"])
    return value


def child_artifact(state: Mapping[str, Any], direction: str, filename: str) -> Path:
    return Path(str(state["directions"][direction]["output_dir"])) / filename


def collect_child_payload(state: Mapping[str, Any], direction: str, filename: str, key: str) -> list[dict[str, Any]]:
    path = child_artifact(state, direction, filename)
    if not path.is_file():
        return []
    payload = read_json(path)
    rows = payload.get(key)
    return [dict(item) for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []


def derive_segments(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start: int | None = None
    last: int | None = None
    for row in frames:
        confirmed = row.get("bbox_xyxy") is not None and str(row.get("state") or "") in CONFIRMED_STATES
        frame = int(row["frame_index"])
        if confirmed and start is None:
            start = last = frame
        elif confirmed:
            last = frame
        elif start is not None:
            segments.append({
                "segment_id": f"segment_{len(segments)+1:04d}",
                "start_frame": start,
                "end_frame_inclusive": int(last),
                "frame_count": int(last) - start + 1,
                "source": "MERGED_DIRECTIONAL_TARGET_TIMELINE",
            })
            start = last = None
    if start is not None:
        segments.append({
            "segment_id": f"segment_{len(segments)+1:04d}",
            "start_frame": start,
            "end_frame_inclusive": int(last),
            "frame_count": int(last) - start + 1,
            "source": "MERGED_DIRECTIONAL_TARGET_TIMELINE",
        })
    return segments


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        bbox = row.pop("bbox_xyxy", None)
        if bbox is None:
            row.update({"bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": ""})
        else:
            row.update({"bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3]})
        for key, val in list(row.items()):
            if isinstance(val, (dict, list)):
                row[key] = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        flattened.append(row)
    fields: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def merge_outputs(root: Path, parent: Path, state: dict[str, Any], render: bool) -> None:
    cfg = state["config"]
    meta = dict(cfg["source_video_metadata"])
    frame_count = int(meta["frame_count"])
    fps = float(meta["fps"])
    anchor = int(cfg["anchor_frame"])
    shots = shots_from_cuts(cfg["cut_frames"], frame_count)

    merged: dict[int, dict[str, Any]] = {}
    child_statuses: dict[str, str | None] = {}
    for direction in ("backward", "forward"):
        timeline_path = child_artifact(state, direction, "target_timeline.json")
        child_state = read_child_state(state, direction)
        child_statuses[direction] = str(child_state.get("status")) if child_state else None
        if not timeline_path.is_file():
            continue
        timeline = read_json(timeline_path)
        for raw in timeline.get("frames") or []:
            if not isinstance(raw, Mapping):
                continue
            row = transform_child_frame(raw, direction, anchor, fps)
            frame = int(row["frame_index"])
            if 0 <= frame < frame_count:
                # Forward wins at the duplicate user-confirmed anchor.
                if frame not in merged or direction == "forward":
                    merged[frame] = row

    frames: list[dict[str, Any]] = []
    for frame in range(frame_count):
        row = merged.get(frame)
        if row is None:
            row = {
                "frame_index": frame,
                "local_frame_index": frame,
                "time_seconds": frame / fps,
                "time_ms": int(round(1000.0 * frame / fps)),
                "state": "SEARCHING",
                "bbox_xyxy": None,
                "tracking_confidence": 0.0,
                "identity_confidence": 0.0,
                "identity_source": "NONE",
                "selected_detection_id": None,
                "decision_reason": "DIRECTION_NOT_YET_PROCESSED",
                "review_required": False,
                "ambiguity_id": None,
                "tracking_direction": "UNPROCESSED",
            }
        row["shot_id"] = shot_id_for_frame(frame, shots)
        frames.append(row)

    # The user-selected source anchor is immutable strongest evidence.
    anchor_row = frames[anchor]
    anchor_row.update({
        "state": "USER_CONFIRMED",
        "bbox_xyxy": list(cfg["initial_bbox_xyxy"]),
        "identity_confidence": 1.0,
        "identity_source": "USER_CONFIRMED_INITIAL_SELECTION",
        "decision_reason": "IMMUTABLE_USER_SELECTED_ANCHOR",
        "review_required": False,
        "ambiguity_id": None,
    })

    ambiguities: list[dict[str, Any]] = []
    confirmations: list[dict[str, Any]] = []
    for direction in ("backward", "forward"):
        child_ambiguities = collect_child_payload(
            state, direction, "ambiguities.json", "ambiguities"
        )
        child_confirmations = collect_child_payload(
            state, direction, "confirmations.json", "confirmations"
        )

        # A child can durably reach NEEDS_CONFIRMATION before the wrapper has
        # copied its ambiguity artifact into the parent contract.  The R1 DB
        # synchronizer intentionally requires pending_action.ambiguity_id to
        # resolve against parent pipeline_state.ambiguities.  Fall back to the
        # child's pipeline_state audit trail if an artifact is not present yet.
        child_state = read_child_state(state, direction)
        if not child_ambiguities and isinstance(child_state, Mapping):
            child_ambiguities = [
                dict(item)
                for item in (child_state.get("ambiguities") or [])
                if isinstance(item, Mapping)
            ]
        if not child_confirmations and isinstance(child_state, Mapping):
            child_confirmations = [
                dict(item)
                for item in (child_state.get("confirmations") or [])
                if isinstance(item, Mapping)
            ]

        ambiguities.extend(
            transform_ambiguity(item, direction, anchor)
            for item in child_ambiguities
        )
        confirmations.extend(
            transform_confirmation(item, direction, anchor)
            for item in child_confirmations
        )

    # R1RuntimeStateSynchronizer consumes pipeline_state.json, not only the
    # sibling ambiguities.json artifact.  Keep the parent JSON self-contained
    # so a durable CROSS_SHOT_CONFIRMATION can be synchronized atomically.
    state["ambiguities"] = ambiguities
    state["confirmations"] = confirmations

    pending = state.get("pending_action")
    if isinstance(pending, Mapping) and str(pending.get("type") or "") == "CROSS_SHOT_CONFIRMATION":
        pending_id = str(pending.get("ambiguity_id") or "")
        if not pending_id:
            raise RuntimeError("Parent CROSS_SHOT_CONFIRMATION has no ambiguity_id")
        if not any(str(item.get("ambiguity_id") or "") == pending_id for item in ambiguities):
            raise RuntimeError(
                "Parent pending ambiguity is absent from merged directional state: "
                + pending_id
            )

    # Persist the enriched parent contract before backend reconciliation.
    # Without this save the subprocess can correctly pause for review while the
    # backend sees only pending_action and fails referential-integrity sync.
    save_parent(parent, state)

    timeline = {
        "schema_version": "kickclip.target_centric_e2e.v1",
        "pipeline_version": WRAPPER_VERSION,
        "test_name": state["test_name"],
        "target_id": "target_001",
        "status": state["status"],
        "video": meta,
        "tracked_range": {
            "source_start_frame": 0,
            "source_end_frame_inclusive": frame_count - 1,
            "source_start_time_seconds": 0.0,
            "source_end_time_seconds": (frame_count - 1) / fps,
            "tracking_direction": "BIDIRECTIONAL_FROM_CONFIRMED_ANCHOR",
            "anchor_frame": anchor,
        },
        "shots": shots,
        "frames": frames,
        "ambiguities": ambiguities,
        "confirmations": confirmations,
        "provenance": {
            "full_scene_product_wrapper": True,
            "tracking_play_conf_threshold": cfg["tracking_play_conf_threshold"],
            "selected_reference_memory_used_for_scoring": bool(cfg["trusted_selected_reference_memory"]),
            "target_reference_set": cfg.get("target_reference_set"),
            "frozen_phase1_modified": False,
            "frozen_phase2_modified": False,
            "threshold_search_performed": False,
            "v7_runtime_dependency": False,
            "direction_statuses": child_statuses,
        },
    }
    atomic_json(parent / "target_timeline.json", timeline)
    write_csv(parent / "target_timeline.csv", frames)
    atomic_json(parent / "ambiguities.json", {"ambiguities": ambiguities})
    atomic_json(parent / "confirmations.json", {"confirmations": confirmations})
    segments = derive_segments(frames)
    atomic_json(parent / "target_segments.json", {"segments": segments})
    write_csv(
        parent / "shot_boundaries.csv",
        [
            {
                "shot_index": shot["shot_index"],
                "shot_id": shot["shot_id"],
                "start_frame": shot["start_frame"],
                "end_frame_inclusive": shot["end_frame_inclusive"],
                "frame_count": shot["frame_count"],
            }
            for shot in shots
        ],
    )

    counts = Counter(str(row.get("state") or "") for row in frames)
    bbox_count = sum(row.get("bbox_xyxy") is not None for row in frames)
    summary = {
        "schema_version": "kickclip.target_centric_e2e_summary.v1",
        "generated_at": now_iso(),
        "status": state["status"],
        "decision": state.get("decision"),
        "test_name": state["test_name"],
        "counts": {
            "frame_count": frame_count,
            "bbox_frame_count": bbox_count,
            "bbox_coverage_percent": 100.0 * bbox_count / max(1, frame_count),
            "state_counts": dict(counts),
            "shot_count": len(shots),
            "ambiguity_count": len(ambiguities),
            "confirmation_count": len(confirmations),
            "derived_segment_count": len(segments),
        },
        "pending_action": state.get("pending_action"),
        "full_scene": True,
        "anchor_frame": anchor,
        "direction_statuses": child_statuses,
        "tracking_play_conf_threshold": cfg["tracking_play_conf_threshold"],
        "selected_reference_memory_used_for_scoring": bool(cfg["trusted_selected_reference_memory"]),
        "safety_invariants": state["safety_contract"],
    }
    atomic_json(parent / "pipeline_summary.json", summary)

    manifest = {
        "schema_version": "kickclip.target_centric_full_scene_manifest.v1",
        "pipeline_version": WRAPPER_VERSION,
        "created_at": state["created_at"],
        "test_name": state["test_name"],
        "source_video": meta,
        "source_video_sha256": cfg["source_video_sha256"],
        "initial_bbox_xyxy": cfg["initial_bbox_xyxy"],
        "initial_frame": anchor,
        "tracking_direction": "BIDIRECTIONAL_FROM_CONFIRMED_ANCHOR",
        "tracking_play_conf_threshold": cfg["tracking_play_conf_threshold"],
        "target_reference_set": cfg.get("target_reference_set"),
        "selected_reference_memory_used_for_scoring": bool(cfg["trusted_selected_reference_memory"]),
        "core_script": {"path": cfg["core_script"], "sha256": sha256_file(Path(cfg["core_script"]))},
        "safety_contract": state["safety_contract"],
    }
    atomic_json(parent / "pipeline_manifest.json", manifest)
    atomic_text(
        parent / "report.md",
        "# KickClip Full-Scene Target-Centric Tracking\n\n"
        f"- Status: `{state['status']}`\n"
        f"- Decision: `{state.get('decision')}`\n"
        f"- Source frames: `{frame_count}`\n"
        f"- User anchor: `{anchor}`\n"
        f"- Tracking: `BIDIRECTIONAL_FROM_CONFIRMED_ANCHOR`\n"
        f"- RF-DETR play confidence: `{cfg['tracking_play_conf_threshold']}`\n"
        f"- Selected reference memory used for scoring: `{bool(cfg['trusted_selected_reference_memory'])}`\n"
        f"- BBox coverage: `{summary['counts']['bbox_coverage_percent']:.2f}%`\n\n"
        "Frozen V1/V2 source files are not modified. Camera-cut motion continuity is reset. "
        "Frozen safe-gate PASS links are auto-reacquired; only AMBIGUOUS cross-shot links "
        "require user confirmation (maximum three tracklet candidates).\n",
    )

    if render and state["status"] in TERMINAL:
        core = load_core(Path(cfg["core_script"]))
        core.render_previews(Path(cfg["source_video"]), parent, timeline, int(cfg.get("print_every", 50)))


def direction_ready(state: Mapping[str, Any], direction: str) -> bool:
    child = read_child_state(state, direction)
    return bool(child and str(child.get("status") or "") in TERMINAL)


def child_failure_diagnostic(parent: Path, direction: str) -> str | None:
    """Extract the most useful fatal diagnostic from a child direction log."""

    log_path = parent / "logs" / f"{direction}.log"
    if not log_path.is_file():
        return None
    try:
        lines = [
            line.strip()
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except OSError:
        return None
    if not lines:
        return None

    preferred_markers = (
        "e2e fatal error:",
        "fatal error:",
        "traceback (most recent call last):",
        "runtimeerror:",
        "file not found",
        "filenotfounderror:",
        "valueerror:",
    )
    lowered = [line.lower() for line in lines]
    for marker in preferred_markers:
        for index in range(len(lines) - 1, -1, -1):
            if marker in lowered[index]:
                if marker == "traceback (most recent call last):":
                    # The useful exception is normally the last non-empty line.
                    return lines[-1][:4000]
                return lines[index][:4000]
    return lines[-1][:4000]


def update_parent_from_child(parent: Path, state: dict[str, Any], direction: str) -> str:
    child = read_child_state(state, direction)
    if child is None:
        diagnostic = child_failure_diagnostic(parent, direction)
        message = f"{direction} child ended without pipeline_state.json"
        if diagnostic:
            message += f"; child diagnostic: {diagnostic}"
        raise RuntimeError(message)

    child_status = str(child.get("status") or "")
    pending = public_pending(direction, child)

    # Durability repair: a child may have materialized a pending ambiguity while
    # its status JSON still contains the preceding RUNNING checkpoint.  Never
    # discard a real pending action merely because the status field lagged.
    if child_status == RESUMABLE_CHILD_STATUS and pending is not None:
        child_status = "NEEDS_CONFIRMATION"

    state["directions"][direction]["status"] = child_status
    state["directions"][direction]["decision"] = child.get("decision")
    if child_status in {"NEEDS_CONFIRMATION", "WAITING_CROSS_SHOT_CONFIRMATION"}:
        if pending is None:
            raise RuntimeError(
                f"{direction} child requires confirmation but has no pending_action"
            )
        state["status"] = "NEEDS_CONFIRMATION"
        state["decision"] = f"{direction.upper()}_{child.get('decision') or 'REVIEW_REQUIRED'}"
        state["pending_action"] = pending
        state["active_direction"] = direction
    elif child_status in TERMINAL:
        state["pending_action"] = None
        state["status"] = "RUNNING"
        state["decision"] = f"{direction.upper()}_DIRECTION_COMPLETE"
    elif child_status == RESUMABLE_CHILD_STATUS:
        # RUNNING is a durable checkpoint, not a fatal terminal state.  The
        # wrapper owns the child process lifecycle, so if the subprocess has
        # exited at this checkpoint we immediately issue a neutral --resume.
        state["status"] = "RUNNING"
        state["decision"] = f"{direction.upper()}_CHILD_RUNNING_RESUMABLE"
        state["pending_action"] = None
        state["active_direction"] = direction
    else:
        state["status"] = "FAILED"
        state["decision"] = f"{direction.upper()}_UNEXPECTED_CHILD_STATUS_{child_status or 'MISSING'}"
        state["pending_action"] = None
    save_parent(parent, state)
    return child_status


def run_direction_new(root: Path, parent: Path, state: dict[str, Any], direction: str) -> int:
    if direction == "backward":
        cfg = state["config"]
        make_reverse_prefix(
            Path(cfg["source_video"]),
            Path(cfg["reverse_video"]),
            int(cfg["anchor_frame"]),
            cfg["source_video_metadata"],
        )
    command = build_child_new_command(state, direction)
    return run_subprocess(command, root, parent / "logs" / f"{direction}.log")


def run_direction_resume(root: Path, parent: Path, state: dict[str, Any], direction: str, args: argparse.Namespace) -> int:
    command = build_child_resume_command(state, direction, args)
    return run_subprocess(command, root, parent / "logs" / f"{direction}.log")


def run_direction_continue(
    root: Path,
    parent: Path,
    state: dict[str, Any],
    direction: str,
) -> int:
    command = build_child_continue_command(state, direction)
    return run_subprocess(command, root, parent / "logs" / f"{direction}.log")


def drain_resumable_child(
    root: Path,
    parent: Path,
    state: dict[str, Any],
    direction: str,
    initial_status: str,
) -> tuple[int, str]:
    """Drive an exited RUNNING checkpoint to review pause or terminal state.

    Core search normally reaches NEEDS_CONFIRMATION or a terminal state in one
    invocation.  This bounded recovery loop handles durable intermediate state
    after process interruption without converting it into a false fatal error.
    """
    status = initial_status
    return_code = 0
    hops = 0
    while status == RESUMABLE_CHILD_STATUS and hops < MAX_INTERNAL_CONTINUATION_HOPS:
        hops += 1
        state["decision"] = (
            f"{direction.upper()}_AUTO_RESUME_DURABLE_RUNNING_CHECKPOINT_{hops}"
        )
        save_parent(parent, state)
        return_code = run_direction_continue(root, parent, state, direction)
        status = update_parent_from_child(parent, state, direction)
        merge_outputs(root, parent, state, render=False)

    if status == RESUMABLE_CHILD_STATUS:
        state["status"] = "FAILED"
        state["decision"] = (
            f"{direction.upper()}_CHILD_STALLED_RUNNING_AFTER_"
            f"{MAX_INTERNAL_CONTINUATION_HOPS}_AUTO_RESUMES"
        )
        state["pending_action"] = None
        save_parent(parent, state)
        merge_outputs(root, parent, state, render=False)
        return 2, status

    return return_code, status


def finalize_status(state: dict[str, Any]) -> None:
    statuses = [str(state["directions"][d].get("status") or "") for d in ("forward", "backward")]
    if all(value == "COMPLETE" for value in statuses):
        state["status"] = "COMPLETE"
        state["decision"] = "FULL_SCENE_BIDIRECTIONAL_TARGET_TIMELINE_COMPLETE"
    elif any(value == "BLOCKED" for value in statuses):
        state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
        state["decision"] = "FULL_SCENE_COMPLETED_WITH_REVIEW_BLOCK"
    else:
        state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
        state["decision"] = "FULL_SCENE_COMPLETED_WITH_SAFE_UNRESOLVED_GAPS"
    state["pending_action"] = None
    state["active_direction"] = None


def print_status(parent: Path, state: Mapping[str, Any]) -> None:
    print("KickClip Target-Centric Full-Scene E2E")
    print(f"Status               : {state['status']}")
    print(f"Decision             : {state.get('decision')}")
    print(f"Tracking direction   : BIDIRECTIONAL_FROM_CONFIRMED_ANCHOR")
    print(f"Anchor frame         : {state['config']['anchor_frame']}")
    print(f"RF-DETR play conf    : {state['config']['tracking_play_conf_threshold']}")
    print(f"Reference scoring    : {state['config']['trusted_selected_reference_memory']}")
    print(f"Forward / backward   : {state['directions']['forward'].get('status')} / {state['directions']['backward'].get('status')}")
    if state.get("pending_action"):
        pending = state["pending_action"]
        print(f"Pending action       : {pending.get('type')}")
        print(f"Direction            : {pending.get('direction')}")
        if pending.get("ambiguity_id"):
            print(f"Ambiguity            : {pending.get('ambiguity_id')}")
        if pending.get("candidate_ids"):
            print(f"Candidates           : {', '.join(str(x) for x in pending.get('candidate_ids'))}")
        if pending.get("contact_sheet"):
            print(f"Contact sheet        : {pending.get('contact_sheet')}")
        if pending.get("preview"):
            print(f"Preview              : {pending.get('preview')}")
    print(f"Timeline             : {parent / 'target_timeline.json'}")
    print(f"Output               : {parent}")


def record_fatal_parent_state(exc: BaseException) -> None:
    """Best-effort terminal JSON for backend reconciliation after wrapper crashes."""

    try:
        args = parse_args()
    except BaseException:
        return
    if not bool(getattr(args, "full_scene", False)):
        return
    try:
        root = args.project_root.expanduser().resolve()
        test_name = validate_name(args.test_name)
        output_root = resolve(root, args.output_root)
        parent = output_root / test_name
        parent.mkdir(parents=True, exist_ok=True)
        state_path = parent / "pipeline_state.json"
        if state_path.is_file():
            try:
                state = read_json(state_path)
            except Exception:
                state = {}
        else:
            state = {}
        state.update(
            {
                "schema_version": state.get("schema_version") or WRAPPER_SCHEMA,
                "pipeline_version": state.get("pipeline_version") or WRAPPER_VERSION,
                "updated_at": now_iso(),
                "status": "FAILED",
                "decision": "FULL_SCENE_FATAL_ERROR",
                "pending_action": None,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "active_direction": state.get("active_direction"),
                },
            }
        )
        atomic_json(state_path, state)
    except BaseException:
        # Never mask the original tracking exception while recording diagnostics.
        return


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not args.full_scene:
        return delegate_non_full_scene(args, root)

    test_name = validate_name(args.test_name)
    output_root = resolve(root, args.output_root)
    parent = output_root / test_name

    if args.resume:
        state_path = parent / "pipeline_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = read_json(state_path)
        if not state.get("config", {}).get("full_scene"):
            raise RuntimeError("Existing job is not a full-scene product run")
        direction = str(state.get("active_direction") or "")
        if direction not in {"forward", "backward"}:
            raise RuntimeError("No resumable full-scene direction is active")
        return_code = run_direction_resume(root, parent, state, direction, args)
        child_status = update_parent_from_child(parent, state, direction)
        merge_outputs(root, parent, state, render=False)
        if child_status == RESUMABLE_CHILD_STATUS:
            return_code, child_status = drain_resumable_child(
                root, parent, state, direction, child_status
            )
        if child_status in {"NEEDS_CONFIRMATION", "WAITING_CROSS_SHOT_CONFIRMATION"}:
            print_status(parent, state)
            return 3
        if child_status not in TERMINAL:
            print_status(parent, state)
            return 2
        next_direction = "backward" if direction == "forward" else None
        if next_direction and not direction_ready(state, next_direction):
            state["active_direction"] = next_direction
            save_parent(parent, state)
            rc = run_direction_new(root, parent, state, next_direction)
            next_status = update_parent_from_child(parent, state, next_direction)
            merge_outputs(root, parent, state, render=False)
            if next_status == RESUMABLE_CHILD_STATUS:
                rc, next_status = drain_resumable_child(
                    root, parent, state, next_direction, next_status
                )
            if next_status in {"NEEDS_CONFIRMATION", "WAITING_CROSS_SHOT_CONFIRMATION"}:
                print_status(parent, state)
                return 3
            if next_status not in TERMINAL:
                print_status(parent, state)
                return 2 if rc != 0 else 2
        if direction_ready(state, "forward") and direction_ready(state, "backward"):
            finalize_status(state)
            save_parent(parent, state)
            merge_outputs(root, parent, state, render=not bool(state["config"].get("no_preview")))
            print_status(parent, state)
            return 0
        print_status(parent, state)
        return 3

    state = initialize_parent(args, root, parent)
    # Forward first preserves the current backend UX around the selected event
    # anchor. The backward half is then run from the same immutable target memory.
    rc = run_direction_new(root, parent, state, "forward")
    status = update_parent_from_child(parent, state, "forward")
    merge_outputs(root, parent, state, render=False)
    if status == RESUMABLE_CHILD_STATUS:
        rc, status = drain_resumable_child(
            root, parent, state, "forward", status
        )
    if status in {"NEEDS_CONFIRMATION", "WAITING_CROSS_SHOT_CONFIRMATION"}:
        print_status(parent, state)
        return 3
    if status not in TERMINAL:
        print_status(parent, state)
        return 2 if rc != 0 else 2

    state["active_direction"] = "backward"
    save_parent(parent, state)
    rc = run_direction_new(root, parent, state, "backward")
    status = update_parent_from_child(parent, state, "backward")
    merge_outputs(root, parent, state, render=False)
    if status == RESUMABLE_CHILD_STATUS:
        rc, status = drain_resumable_child(
            root, parent, state, "backward", status
        )
    if status in {"NEEDS_CONFIRMATION", "WAITING_CROSS_SHOT_CONFIRMATION"}:
        print_status(parent, state)
        return 3
    if status not in TERMINAL:
        print_status(parent, state)
        return 2 if rc != 0 else 2

    finalize_status(state)
    save_parent(parent, state)
    merge_outputs(root, parent, state, render=not args.no_preview)
    print_status(parent, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        record_fatal_parent_state(exc)
        print(f"Full-scene E2E fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
