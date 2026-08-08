#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip target-centric assisted end-to-end pipeline.

This orchestrator does not modify the frozen V1/V2 source trees. It reuses:
- V2 Stage 3-A0 cut detection;
- V2 Stage 3-A1/A2 pre-cut target-memory construction;
- V2 Stage 3-B0 tracklet generation helpers;
- V2 Stage 3-B1 frozen Sports-OSNet ranking helpers;
- V2 Stage 3-B2 frozen safety-gate values;
- V2 Stage 3-B3 anchor-selection helper;
- frozen V1 same-shot tracking/finalization scripts.

The default product mode is assisted-safe. Frozen Stage 3-B2 safe-gate PASS
candidates are automatically reacquired; only AMBIGUOUS candidates pause for
explicit user confirmation. Same-shot research visual-review checkpoints are
consumed internally by the product wrapper and are never exposed as user tasks.

Backend integration contract:
- V7 is not a runtime dependency.
- Scene-selection jobs may start from a confirmed source-video anchor frame.
- The frozen V1/V2 thresholds and ranking policies are never modified here.
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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

SCHEMA_VERSION = "kickclip.target_centric_e2e.v1"
STATE_VERSION = "kickclip.target_centric_e2e_state.v1"
PIPELINE_VERSION = "1.1.0-product-assisted-intent"

UNCERTAIN_STATES = {"LOST", "SEARCHING", "AMBIGUOUS", "ABSENT", "TERMINATED"}
CONFIRMED_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "ACTIVE_LOW_CONFIDENCE",
    "OCCLUDED",
    "REACQUIRED",
    "USER_CONFIRMED",
}

DEFAULT_V1_MANIFEST = Path("target_centric_tracking_v1/phase1_frozen_manifest.json")
DEFAULT_E2E_OUTPUT_ROOT = Path("runs/target_centric_tracking_e2e_v1")
DEFAULT_V2_OUTPUT_ROOT = Path("runs/target_centric_tracking_v2")
DEFAULT_V1_OUTPUT_ROOT = Path("runs/target_centric_tracking_v1")

# Frozen Stage 3-B0 defaults.
B0_POLICY = {
    "max_age": 8,
    "minimum_tracklet_frames": 3,
    "minimum_predicted_iou": 0.03,
    "maximum_center_distance": 1.6,
    "minimum_area_ratio": 0.25,
    "maximum_area_ratio": 4.0,
    "minimum_match_score": 0.17,
}

# Frozen Stage 3-B1 defaults.
B1_POLICY = {
    "max_crops_per_tracklet": 12,
    "minimum_crop_gap": 3,
}

# Frozen Stage 3-B2 defaults.
B2_POLICY = {
    "minimum_retrieval_score": 0.65,
    "minimum_prototype_similarity": 0.60,
    "minimum_median_margin": 0.0,
    "minimum_positive_margin_support": 0.50,
    "minimum_top1_top2_gap": 0.08,
    "minimum_plausible_score": 0.45,
    "minimum_plausible_prototype": 0.45,
    "review_candidate_count": 3,
}


# Product-side thin pre-filters. These do not modify the frozen V1/V2
# detector, Sports-OSNet ranking metrics, or Stage 3-B2 gate thresholds.
# They only remove candidates that are clearly incompatible before the frozen
# rank/gate output is presented to the user.
PRODUCT_PLAYER_CLASS_IDS = frozenset({0, 1})
TEAM_PROFILE_POLICY_VERSION = "UPPER_TORSO_HSV_COLOR_SIGNATURE_V1"
TEAM_PROFILE_TARGET_MAX_SAMPLES = 8
TEAM_PROFILE_MIN_TARGET_SAMPLES = 3
TEAM_PROFILE_MIN_CANDIDATE_SAMPLES = 2
TEAM_PREFILTER_MIN_BEST_SIMILARITY = 0.55
TEAM_PREFILTER_MAX_INCOMPATIBLE_SIMILARITY = 0.35
TEAM_PREFILTER_MIN_GAP_FROM_BEST = 0.25
ASSISTED_REVIEW_BATCH_SIZE = int(B2_POLICY["review_candidate_count"])
PRODUCT_NEGATIVE_MEMORY_MAX_REFERENCES = 16
PRODUCT_NEGATIVE_MEMORY_MIN_REFERENCES = 4
PRODUCT_NEGATIVE_MEMORY_MIN_CONFIDENCE = 0.25
PRODUCT_NEGATIVE_MEMORY_MIN_BOX_HEIGHT_PX = 24.0
PRODUCT_NEGATIVE_MEMORY_MAX_TARGET_IOU = 0.02
PRODUCT_NEGATIVE_MEMORY_MIN_TARGET_CENTER_DISTANCE = 0.55
PRODUCT_NEGATIVE_PERSON_CLASS_IDS = frozenset({0, 1, 2, 3})

# Product scene policy. KickClip intentionally tracks only on-field WIDE shots.
# Broadcast CLOSEUP and MIXED/transition shots are treated as non-trackable
# context: no cross-shot candidate ranking, no ambiguity, and no user review.
# This removes coach/staff close-up false positives from the target-tracking
# problem instead of attempting to solve role classification there.
PRODUCT_TRACKING_SCENE_POLICY_VERSION = "WIDE_ONLY_TARGET_TRACKING_R1"
PRODUCT_TRACKING_ALLOWED_SCENE_MODES = frozenset({"WIDE"})
PRODUCT_TRACKING_SCENE_PERSON_CLASS_IDS = frozenset({0, 1, 2, 3})
PRODUCT_TRACKING_SCENE_OBSERVATION_CONFIDENCE = 0.25
PRODUCT_TRACKING_SCENE_CLOSEUP_MIN_HEIGHT_RATIO = 0.35
PRODUCT_TRACKING_SCENE_CLOSEUP_MIN_AREA_RATIO = 0.08
PRODUCT_TRACKING_SCENE_CLOSEUP_RATIO_MIN = 0.70
PRODUCT_TRACKING_SCENE_WIDE_RATIO_MAX = 0.30


def _register_product_sports_osnet_safe_globals(torch: Any) -> dict[str, Any]:
    """Register the minimal NumPy dtype globals required by the frozen Sports-OSNet checkpoint.

    The checkpoint hash is verified by the caller before this compatibility hook
    is used.  We keep ``weights_only=True`` in the frozen loader and only add
    the exact legacy/modern NumPy globals observed in this trusted checkpoint.
    ``weights_only=False`` remains forbidden.
    """
    serialization = getattr(torch, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        raise RuntimeError(
            "torch.serialization.add_safe_globals is required for the frozen "
            "Sports-OSNet checkpoint on this PyTorch runtime"
        )

    numpy_core = getattr(np, "_core", None)
    multiarray = getattr(numpy_core, "multiarray", None)
    numpy_scalar = getattr(multiarray, "scalar", None)
    if numpy_scalar is None:
        legacy_core = getattr(np, "core", None)
        legacy_multiarray = getattr(legacy_core, "multiarray", None)
        numpy_scalar = getattr(legacy_multiarray, "scalar", None)
    if numpy_scalar is None:
        raise RuntimeError(
            "NumPy multiarray.scalar is unavailable; cannot authorize the "
            "trusted legacy Sports-OSNet checkpoint safely"
        )

    safe_entries: list[Any] = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (numpy_scalar, "numpy._core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
    ]
    registered_names = [
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar",
        "numpy.dtype",
    ]

    numpy_dtypes = getattr(np, "dtypes", None)
    for dtype_name in ("Float64DType", "Float32DType"):
        dtype_class = getattr(numpy_dtypes, dtype_name, None) if numpy_dtypes is not None else None
        if dtype_class is not None:
            qualified_name = f"numpy.dtypes.{dtype_name}"
            safe_entries.append((dtype_class, qualified_name))
            registered_names.append(qualified_name)

    # PyTorch 2.6+ accepts (callable_or_type, serialized_global_name) tuples.
    # The explicit legacy scalar alias is essential on NumPy 2.x because the
    # live object now reports ``numpy._core`` while the checkpoint stores
    # ``numpy.core.multiarray.scalar``.
    add_safe_globals(safe_entries)
    return {
        "status": "REGISTERED",
        "policy_version": "SPORTS_OSNET_NUMPY2_WEIGHTS_ONLY_COMPAT_R2",
        "registered_safe_globals": registered_names,
        "weights_only_false_allowed": False,
    }



def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _normalized_center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    reference = max(math.hypot(bx2 - bx1, by2 - by1), 1.0)
    return math.hypot(acx - bcx, acy - bcy) / reference


def _collect_initial_shot_negative_crops(
    *,
    video_path: Path,
    detections_csv: Path,
    timeline_path: Path,
    first_shot_end: int,
    tracking_threshold: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build conservative non-target memory from the user-selected initial shot.

    Only person detections on frames where the frozen Stage-2 timeline has a
    confirmed target bbox are eligible.  Detections too close to that target are
    excluded to prevent target contamination.  This is negative evidence only;
    it never changes target identity or frozen Stage-3B thresholds.
    """

    timeline = read_json(timeline_path)
    target_by_frame: dict[int, list[float]] = {}
    for raw in timeline.get("frames") or []:
        try:
            frame_index = int(raw.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if frame_index < 0 or frame_index > int(first_shot_end):
            continue
        if str(raw.get("state") or "") not in CONFIRMED_STATES:
            continue
        bbox = raw.get("bbox_xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            target_by_frame[frame_index] = [float(value) for value in bbox]

    if not target_by_frame:
        return {}, []

    candidates: list[dict[str, Any]] = []
    with detections_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            try:
                frame_index = int(raw["frame_index"])
                class_id = int(raw["class_id"])
                confidence = float(raw["confidence"])
                bbox = [float(raw[key]) for key in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                continue
            if frame_index not in target_by_frame:
                continue
            if class_id not in PRODUCT_NEGATIVE_PERSON_CLASS_IDS:
                continue
            if confidence < max(PRODUCT_NEGATIVE_MEMORY_MIN_CONFIDENCE, float(tracking_threshold)):
                continue
            if bbox[3] - bbox[1] < PRODUCT_NEGATIVE_MEMORY_MIN_BOX_HEIGHT_PX:
                continue
            target_bbox = target_by_frame[frame_index]
            if _bbox_iou_xyxy(bbox, target_bbox) > PRODUCT_NEGATIVE_MEMORY_MAX_TARGET_IOU:
                continue
            if _normalized_center_distance(bbox, target_bbox) < PRODUCT_NEGATIVE_MEMORY_MIN_TARGET_CENTER_DISTANCE:
                continue
            candidates.append(
                {
                    "frame_index": frame_index,
                    "detection_id": str(raw.get("detection_id") or f"f{frame_index:06d}"),
                    "class_id": class_id,
                    "class_name": str(raw.get("class_name") or "person"),
                    "confidence": confidence,
                    "bbox_xyxy": bbox,
                }
            )

    if not candidates:
        return {}, []

    # Keep at most two strong non-target people per frame, then spread samples
    # across time before filling remaining capacity.  This avoids a 16-crop
    # gallery that is just one adjacent burst of the same person.
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        by_frame.setdefault(int(row["frame_index"]), []).append(row)
    per_frame: list[dict[str, Any]] = []
    for frame_index in sorted(by_frame):
        rows = sorted(by_frame[frame_index], key=lambda item: float(item["confidence"]), reverse=True)
        per_frame.extend(rows[:2])

    selected: list[dict[str, Any]] = []
    for row in sorted(per_frame, key=lambda item: (-float(item["confidence"]), int(item["frame_index"]))):
        if any(abs(int(row["frame_index"]) - int(prev["frame_index"])) < 3 for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= PRODUCT_NEGATIVE_MEMORY_MAX_REFERENCES:
            break
    if len(selected) < PRODUCT_NEGATIVE_MEMORY_MAX_REFERENCES:
        selected_ids = {str(item["detection_id"]) for item in selected}
        for row in sorted(per_frame, key=lambda item: (-float(item["confidence"]), int(item["frame_index"]))):
            if str(row["detection_id"]) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(str(row["detection_id"]))
            if len(selected) >= PRODUCT_NEGATIVE_MEMORY_MAX_REFERENCES:
                break

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for negative target memory: {video_path}")
    crops: dict[str, np.ndarray] = {}
    provenance: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected, start=1):
            frame_index = int(row["frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = row["bbox_xyxy"]
            left, top = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
            right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            key = f"negative_{index:04d}"
            crops[key] = crop
            provenance.append(
                {
                    "reference_id": key,
                    "frame_index": frame_index,
                    "detection_id": row["detection_id"],
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "confidence": float(row["confidence"]),
                    "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
                    "selection_reason": "INITIAL_SHOT_CONFIRMED_TARGET_EXCLUSION_NEGATIVE",
                }
            )
    finally:
        capture.release()
    return crops, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the KickClip assisted target-centric E2E pipeline."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--initial-bbox", nargs=4, type=float, default=None)
    parser.add_argument(
        "--initial-frame",
        type=int,
        default=0,
        help=(
            "Source-video frame on which --initial-bbox was selected. "
            "Frame 0 preserves the original standalone contract. A positive "
            "value starts canonical tracking forward from that confirmed anchor."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--reacquisition-mode",
        choices=("assisted", "auto-safe"),
        default="assisted",
        help=(
            "assisted is the product-safe mode: frozen Stage 3-B2 PASS is "
            "automatically reacquired and only AMBIGUOUS shots require user "
            "confirmation. auto-safe is retained as an equivalent compatibility alias."
        ),
    )
    parser.add_argument(
        "--cut-frames",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional reviewed/source shot cut frames in ORIGINAL source-video "
            "coordinates. When --initial-frame is positive, cuts at or before "
            "the anchor are ignored and later cuts are translated internally."
        ),
    )
    parser.add_argument(
        "--phase1-manifest", type=Path, default=DEFAULT_V1_MANIFEST
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_E2E_OUTPUT_ROOT
    )
    parser.add_argument("--reviewer", default="USER")
    parser.add_argument("--review-note", default="")
    parser.add_argument("--backend-memory-revision", type=Path, default=None)
    parser.add_argument("--backend-memory-sha256", default=None)
    parser.add_argument("--backend-safe-weights-runner", type=Path, default=None)
    parser.add_argument(
        "--tracking-play-conf-threshold",
        type=float,
        default=0.15,
        help=(
            "Product RF-DETR confidence for play-shot tracking. The frozen V1 "
            "Stage-1 source is not modified; this value is injected by a product "
            "compatibility wrapper."
        ),
    )
    parser.add_argument("--target-reference-set", type=Path, default=None)
    parser.add_argument("--target-reference-set-sha256", default=None)
    parser.add_argument(
        "--trusted-selected-reference-memory",
        action="store_true",
        help=(
            "Use the immutable user-selected reference gallery as the positive "
            "identity memory. Intended only for backend-confirmed selections."
        ),
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ambiguity-id", default=None)
    decision_group = parser.add_mutually_exclusive_group()
    decision_group.add_argument("--confirmed-candidate", default=None)
    decision_group.add_argument("--confirm-absent", action="store_true")
    decision_group.add_argument("--reject-all-candidates", action="store_true")
    decision_group.add_argument(
        "--reject-all-candidates-as-non-player-role",
        action="store_true",
    )
    decision_group.add_argument("--rejected-candidate", default=None)
    decision_group.add_argument("--unreviewable-candidate", default=None)
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument(
        "--approve-review",
        choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"),
        default=None,
    )
    review_group.add_argument(
        "--reject-review",
        choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"),
        default=None,
    )

    parser.add_argument("--review-decision-artifact", type=Path, default=None)
    parser.add_argument("--review-decision-sha256", default=None)
    parser.add_argument(
        "--allow-prestaged-output",
        action="store_true",
        help=(
            "Allow backend-owned immutable prelaunch artifacts to already exist "
            "inside the job output directory. Canonical runtime artifacts are "
            "written alongside them; the directory is never deleted."
        ),
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def validate_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not value or value in {".", ".."} or any(ch not in allowed for ch in value):
        raise ValueError(f"Invalid test name: {value}")
    return value


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def normalized_relative_path(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fields = ordered or ["empty"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def run_command(
    command: Sequence[str],
    cwd: Path,
    log_path: Optional[Path] = None,
    accepted_codes: Sequence[int] = (0,),
) -> subprocess.CompletedProcess[str]:
    print(f"[E2E] {command_text(command)}", flush=True)
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"$ {command_text(command)}\n")
            stream.write(completed.stdout or "")
            if not (completed.stdout or "").endswith("\n"):
                stream.write("\n")
            stream.write(f"[returncode={completed.returncode}]\n\n")
    if completed.returncode not in accepted_codes:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command_text(command)}"
        )
    return completed


def verify_phase1_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    errors: list[str] = []
    for logical, record in (manifest.get("scripts") or {}).items():
        raw = record.get("path")
        if not raw:
            continue
        path = resolve(root, normalized_relative_path(str(raw)))
        if not path.is_file():
            errors.append(f"missing script {logical}: {path}")
            continue
        observed = sha256_file(path)
        expected = str(record.get("sha256") or "")
        if expected and observed.lower() != expected.lower():
            errors.append(f"changed script {logical}: {path}")
    for logical, record in (manifest.get("models") or {}).items():
        if not record.get("verified"):
            continue
        path = resolve(root, normalized_relative_path(str(record["path"])))
        if not path.is_file():
            errors.append(f"missing model {logical}: {path}")
            continue
        observed = sha256_file(path)
        expected = str(record.get("sha256") or record.get("expected_sha256") or "")
        if expected and observed.lower() != expected.lower():
            errors.append(f"changed model {logical}: {path}")
    if errors:
        raise RuntimeError("Frozen Phase-1 verification failed:\n" + "\n".join(errors))
    return manifest



def _portable_manifest_copy(value: Any, parent_key: str | None = None) -> Any:
    """Deep-copy manifest data while normalizing only path-valued fields.

    chr(92) is used deliberately instead of a backslash string literal so this
    code cannot regress through Python/source escaping while being patched or
    transported between Windows and POSIX systems.
    """
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if (
                key in {"path", "relative_path"}
                and isinstance(child, str)
                and child
            ):
                result[key] = child.replace(chr(92), "/")
            else:
                result[key] = _portable_manifest_copy(child, key)
        return result
    if isinstance(value, list):
        return [_portable_manifest_copy(item, parent_key) for item in value]
    return value


def _portable_manifest_path_values(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"path", "relative_path"} and isinstance(child, str):
                paths.append(child)
            else:
                paths.extend(_portable_manifest_path_values(child))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_portable_manifest_path_values(item))
    return paths


def materialize_portable_phase1_manifest(
    root: Path,
    output_dir: Path,
    source_manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Create a verified cross-platform derivative of the frozen manifest.

    The source frozen manifest is never edited.  Only path separators in the
    derived runtime copy are converted from Windows backslashes to forward
    slashes.  Hashes, thresholds, model records, and frozen scripts are
    unchanged.
    """
    portable = _portable_manifest_copy(dict(manifest))

    path_values = _portable_manifest_path_values(portable)
    remaining_windows_paths = [
        value for value in path_values if chr(92) in value
    ]
    if remaining_windows_paths:
        raise RuntimeError(
            "Portable Phase-1 manifest normalization failed; Windows "
            f"separators remain: {remaining_windows_paths[:5]}"
        )

    compatibility_root = output_dir / "work" / "runtime_compat"
    portable_path = (
        compatibility_root / "phase1_frozen_manifest_portable.json"
    )
    provenance_path = (
        compatibility_root
        / "phase1_frozen_manifest_portable_provenance.json"
    )

    atomic_json(portable_path, portable)

    # Read the exact bytes back from disk.  This catches any discrepancy
    # between the in-memory derivative and what frozen Stage 3-A0/A2 will read.
    disk_portable = read_json(portable_path)
    disk_paths = _portable_manifest_path_values(disk_portable)
    disk_windows_paths = [
        value for value in disk_paths if chr(92) in value
    ]
    if disk_windows_paths:
        raise RuntimeError(
            "Portable Phase-1 manifest written to disk still contains "
            f"Windows separators: {disk_windows_paths[:5]}"
        )

    # Fail closed: the portable derivative must resolve to the exact original
    # frozen files and SHA-256 values before any frozen V2 process receives it.
    verify_phase1_manifest(root, portable_path)

    model_paths = {
        logical: str(record.get("path") or "")
        for logical, record in (disk_portable.get("models") or {}).items()
        if isinstance(record, dict)
    }
    print(
        "[E2E] portable Phase-1 manifest verified: "
        f"rfdetr={model_paths.get('rfdetr', '')} "
        f"sports_osnet={model_paths.get('sports_osnet', '')}",
        flush=True,
    )

    atomic_json(
        provenance_path,
        {
            "schema_version": "kickclip.phase1_manifest_portability.v2",
            "created_at": now_iso(),
            "source_manifest_path": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "portable_manifest_path": str(portable_path.resolve()),
            "portable_manifest_sha256": sha256_file(portable_path),
            "transformation": "PATH_SEPARATOR_ONLY_RECURSIVE",
            "windows_backslash_to_forward_slash": True,
            "disk_readback_verified": True,
            "portable_path_count": len(disk_paths),
            "frozen_source_manifest_modified": False,
            "frozen_script_or_model_hash_modified": False,
        },
    )
    return portable_path


def manifest_script(root: Path, manifest: Mapping[str, Any], logical: str) -> Path:
    record = manifest["scripts"][logical]
    path = resolve(root, normalized_relative_path(str(record["path"])))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def manifest_model_is_verified(
    root: Path,
    manifest: Mapping[str, Any],
    logical: str,
) -> bool:
    """Return True only when the frozen manifest's model file and hash verify.

    Frozen Stage-0 performs legacy asset discovery outside the manifest contract.
    On product installations that discovery can emit SPORTS_OSNET_NOT_FOUND even
    though the exact frozen checkpoint has already been verified by the portable
    manifest.  Product compatibility may normalize that legacy discovery warning
    only when this authoritative check succeeds.
    """

    models = manifest.get("models")
    if not isinstance(models, Mapping):
        return False
    record = models.get(logical)
    if not isinstance(record, Mapping) or not bool(record.get("verified")):
        return False
    raw_path = str(record.get("path") or "").strip()
    if not raw_path:
        return False
    path = resolve(root, normalized_relative_path(raw_path))
    if not path.is_file():
        return False
    expected = str(
        record.get("sha256") or record.get("expected_sha256") or ""
    ).strip().lower()
    if not expected:
        return False
    return sha256_file(path).lower() == expected


def stage0_product_allowed_warning_codes(
    root: Path,
    manifest: Mapping[str, Any],
    device: str,
) -> frozenset[str]:
    """Warnings that are safe for the product integration layer to normalize.

    This does not modify frozen Stage-0 or its evidence.  CUDA absence is
    expected when the runtime was requested as auto/cpu (for example macOS,
    where the canonical tracking core currently falls back to CPU).  The
    Sports-OSNet discovery warning is ignored only if the exact checkpoint
    declared in the frozen manifest exists and its SHA-256 matches.
    """

    allowed = {"DURATION_OUTSIDE_RANGE"}
    requested_device = str(device or "").strip().lower()
    if requested_device in {"auto", "cpu"}:
        allowed.add("CUDA_NOT_AVAILABLE")
    if manifest_model_is_verified(root, manifest, "sports_osnet"):
        allowed.add("SPORTS_OSNET_NOT_FOUND")
    return frozenset(allowed)


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if width < 1 or height < 1 or fps <= 0 or frame_count < 1:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }


def clip_bbox(box: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(0.0, x1), width - 1.0)
    y1 = min(max(0.0, y1), height - 1.0)
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return [x1, y1, x2, y2]


def make_shots(frame_count: int, cut_frames: Sequence[int]) -> list[dict[str, Any]]:
    cuts = sorted(set(int(v) for v in cut_frames if 0 < int(v) < frame_count))
    boundaries = [0, *cuts, frame_count]
    shots: list[dict[str, Any]] = []
    for index, (start, end_exclusive) in enumerate(zip(boundaries, boundaries[1:])):
        shots.append(
            {
                "shot_index": index,
                "shot_id": f"shot_{index:04d}",
                "start_frame": start,
                "end_frame_inclusive": end_exclusive - 1,
                "frame_count": end_exclusive - start,
                "cut_in_frame": start if index > 0 else None,
                "cut_out_frame": end_exclusive if end_exclusive < frame_count else None,
                "status": "UNPROCESSED" if index > 0 else "INITIAL_TARGET_SHOT",
            }
        )
    return shots


def state_file(output_dir: Path) -> Path:
    return output_dir / "pipeline_state.json"


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    atomic_json(state_file(output_dir), state)


def load_state(output_dir: Path) -> dict[str, Any]:
    path = state_file(output_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def initialize_output_dir(
    output_dir: Path,
    overwrite: bool,
    *,
    allow_prestaged_output: bool = False,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        elif not allow_prestaged_output:
            raise FileExistsError(
                f"E2E output exists; use --resume or --overwrite: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "work").mkdir(exist_ok=True)
    (output_dir / "ambiguity_candidates").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)


def run_cut_audit(
    root: Path,
    output_dir: Path,
    test_name: str,
    video: Path,
    bbox: Sequence[float],
    phase1_manifest: Path,
) -> dict[str, Any]:
    audit_root = output_dir / "work" / "v2_audit_runs"
    audit_test_name = "cut_audit"
    audit_dir = audit_root / audit_test_name
    summary_path = audit_dir / "stage3a0_summary.json"

    # A previous launch may have written a terminal FAIL summary before the
    # canonical wrapper had a portable manifest.  Do not treat that stale
    # summary as reusable evidence; rerun Stage 3-A0 from clean derived output.
    if summary_path.is_file():
        existing_summary = read_json(summary_path)
        if existing_summary.get("status") != "PASS":
            shutil.rmtree(audit_dir, ignore_errors=True)

    if not summary_path.is_file():
        script = root / "target_centric_tracking_v2" / "stage3a0_audit_cross_shot_inputs.py"
        command = [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--video",
            str(video),
            "--test-name",
            audit_test_name,
            "--output-root",
            str(audit_root),
            "--phase1-manifest",
            str(phase1_manifest),
            "--initial-bbox",
            *[str(v) for v in bbox],
        ]
        run_command(command, root, output_dir / "logs" / "pipeline.log")
    summary = read_json(summary_path)
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Stage 3-A0 failed: {summary.get('decision')}")
    return summary


def choose_cut_frames(
    a0: Mapping[str, Any], explicit: Optional[Sequence[int]], frame_count: int
) -> list[int]:
    candidates = (a0.get("cut_detection") or {}).get("candidates") or []
    if explicit is not None and len(explicit) > 0:
        # Explicit boundaries are already-reviewed/authoritative shot boundaries
        # supplied by the backend. They do not need to coincide with the heuristic
        # Stage 3-A0 peak frame; identity is never inferred from this choice.
        cuts = sorted(set(int(v) for v in explicit))
    else:
        cuts = sorted(
            set(
                int(item["cut_frame"])
                for item in candidates
                if isinstance(item, dict) and bool(item.get("hard_cut_candidate"))
            )
        )
    return [v for v in cuts if 0 < v < frame_count]


def materialize_a0_for_v2_test(
    output_dir: Path,
    v2_dir: Path,
    a0: Mapping[str, Any],
) -> None:
    v2_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(v2_dir / "stage3a0_summary.json", dict(a0))
    source_dir = output_dir / "work" / "v2_audit_runs" / "cut_audit"
    for name in (
        "stage3a0_cut_candidates.csv",
        "stage3a0_cut_contact_sheet.jpg",
        "stage3a0_initial_target_preview.jpg",
        "stage3a0_report.md",
    ):
        source = source_dir / name
        target = v2_dir / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)



LEGACY_UNUSED_STAGE0_WARNING_CODES = frozenset(
    {
        "V7_CODE_NOT_FOUND",
        "V7_OUTPUT_NOT_FOUND",
        "V6_OUTPUT_NOT_FOUND",
    }
)


def _stage0_warning_codes(audit: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for item in audit.get("findings") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("severity") or "").upper() != "WARNING":
            continue
        values.append(str(item.get("code") or "UNKNOWN_WARNING"))
    return sorted(set(values))


def _materialize_phase1_stage0_compatibility(
    root: Path,
    output_dir: Path,
    *,
    raw_dir: Path,
    source_dir: Path,
    source_test_name: str,
    allowed_nonlegacy_warning_codes: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Clone immutable Stage-0 evidence into a derived product-compatible audit.

    Only historical-output/V7 warnings that are irrelevant to the selected
    target-centric runtime may be normalized. The original raw audit is kept
    byte-for-byte under ``raw_dir``. No V7 directory is created and no model
    or threshold check is bypassed.
    """

    audit = read_json(raw_dir / "audit.json")
    manifest = read_json(raw_dir / "input_manifest.json")
    warnings = _stage0_warning_codes(audit)
    errors = [
        str(item.get("code") or "UNKNOWN_ERROR")
        for item in audit.get("findings") or []
        if isinstance(item, Mapping)
        and str(item.get("severity") or "").upper() == "ERROR"
    ]
    nonlegacy = sorted(
        code for code in warnings if code not in LEGACY_UNUSED_STAGE0_WARNING_CODES
    )
    blocked_nonlegacy = sorted(
        code for code in nonlegacy
        if code not in allowed_nonlegacy_warning_codes
    )
    if errors or blocked_nonlegacy or str(audit.get("status") or "") not in {
        "PASS",
        "PASS_WITH_WARNINGS",
    }:
        raise RuntimeError(
            "Frozen Stage-0 blocked canonical E2E. "
            f"errors={sorted(set(errors))}, nonlegacy_warnings={blocked_nonlegacy}, "
            f"status={audit.get('status')}"
        )

    if source_dir.exists():
        shutil.rmtree(source_dir)
    shutil.copytree(raw_dir, source_dir)

    derived_audit = read_json(source_dir / "audit.json")
    derived_manifest = read_json(source_dir / "input_manifest.json")
    derived_initialization = read_json(source_dir / "target_initialization.json")

    # Remove only warnings that the compatibility contract explicitly
    # authorizes. Raw frozen Stage-0 evidence remains byte-for-byte preserved
    # under raw_dir; the derived copy is used solely to satisfy the unchanged
    # frozen Phase-1 runner contract without inventing V7 artifacts.
    normalized_codes = set(LEGACY_UNUSED_STAGE0_WARNING_CODES) | set(
        allowed_nonlegacy_warning_codes
    )
    retained_findings = [
        dict(item)
        for item in (derived_audit.get("findings") or [])
        if not (
            isinstance(item, Mapping)
            and str(item.get("code") or "") in normalized_codes
        )
    ]
    retained_warning_codes = sorted(
        {
            str(item.get("code") or "UNKNOWN_WARNING")
            for item in retained_findings
            if isinstance(item, Mapping)
            and str(item.get("severity") or "").upper() == "WARNING"
        }
    )
    derived_status = "PASS_WITH_WARNINGS" if retained_warning_codes else "PASS"
    derived_audit["findings"] = retained_findings
    derived_audit["status"] = derived_status
    counts = derived_audit.get("counts")
    if isinstance(counts, dict):
        counts["warnings"] = len(retained_warning_codes)
        counts["errors"] = 0
    derived_audit["compatibility"] = {
        "schema_version": "kickclip.stage0_legacy_unused_compatibility.v1",
        "source_audit_path": str((raw_dir / "audit.json").resolve()),
        "source_audit_sha256": sha256_file(raw_dir / "audit.json"),
        "normalized_warning_codes": sorted(
            code for code in warnings if code in normalized_codes
        ),
        "retained_warning_codes": retained_warning_codes,
        "v7_runtime_dependency": False,
        "frozen_stage0_modified": False,
        "synthetic_model_pass_created": False,
    }
    derived_manifest["status"] = derived_status
    derived_manifest["test_name"] = source_test_name
    paths = derived_manifest.get("paths")
    if isinstance(paths, dict):
        paths["test_output_dir"] = str(source_dir.resolve())
    phase1_contract = derived_manifest.get("phase1_contract")
    if isinstance(phase1_contract, dict):
        phase1_contract["legacy_v7_runtime_required"] = False
    derived_manifest["compatibility"] = dict(derived_audit["compatibility"])
    if str(derived_initialization.get("status") or "") == "INVALID":
        raise RuntimeError("Frozen Stage-0 target initialization is invalid")
    derived_initialization["status"] = "INITIALIZING"
    derived_initialization["compatibility"] = dict(derived_audit["compatibility"])

    atomic_json(source_dir / "audit.json", derived_audit)
    atomic_json(source_dir / "input_manifest.json", derived_manifest)
    atomic_json(source_dir / "target_initialization.json", derived_initialization)
    gate = {
        "schema_version": "kickclip.stage0_legacy_unused_compatibility_gate.v1",
        "created_at": now_iso(),
        "status": derived_status,
        "source_stage0_status": audit.get("status"),
        "source_audit_path": str((raw_dir / "audit.json").resolve()),
        "source_audit_sha256": sha256_file(raw_dir / "audit.json"),
        "source_manifest_path": str((raw_dir / "input_manifest.json").resolve()),
        "source_manifest_sha256": sha256_file(raw_dir / "input_manifest.json"),
        "normalized_warning_codes": sorted(
            code for code in warnings if code in normalized_codes
        ),
        "retained_warning_codes": retained_warning_codes,
        "allowed_codes": sorted(normalized_codes),
        "v7_runtime_dependency": False,
        "frozen_stage0_modified": False,
        "stage1_authorized": derived_status == "PASS",
    }
    atomic_json(source_dir / "compatibility_gate.json", gate)
    return gate


def ensure_precut_phase1_stage2_source(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    manifest: Mapping[str, Any],
    device: str,
) -> Path:
    """Build the real frozen Stage-1/2 evidence without requiring V7."""

    source_test_name = str(state["work"]["phase1_source_test_name"])
    raw_test_name = f"{source_test_name}__raw_stage0"
    phase1_root = resolve(root, DEFAULT_V1_OUTPUT_ROOT)
    raw_dir = phase1_root / raw_test_name
    source_dir = phase1_root / source_test_name

    stage1_summary = source_dir / "stage1_detection_summary.json"
    stage2_summary = source_dir / "stage2_association_summary.json"
    if stage1_summary.is_file() and stage2_summary.is_file():
        cached_stage1 = read_json(stage1_summary)
        observed_conf = float((cached_stage1.get("detector") or {}).get("confidence_threshold", -1.0))
        desired_conf = float(state.get("tracking_play_conf_threshold", 0.15))
        if abs(observed_conf - desired_conf) > 1e-9:
            raise RuntimeError(
                "Cached Stage-1 evidence uses a different RF-DETR threshold: "
                f"observed={observed_conf} desired={desired_conf}. "
                "Use a new test name or clear only this job's stale runtime output."
            )
        return source_dir

    raw_audit = raw_dir / "audit.json"
    if not raw_audit.is_file():
        command = [
            sys.executable,
            str(manifest_script(root, manifest, "stage0")),
            "--project-root",
            str(root),
            "--video",
            str(Path(state["video"]["path"]).resolve()),
            "--test-name",
            raw_test_name,
            "--initial-bbox",
            *[str(v) for v in state["initial_bbox_xyxy"]],
            "--bbox-format",
            "xyxy_pixels",
        ]
        run_command(command, root, output_dir / "logs" / "pipeline.log")

    allowed_stage0_warnings = stage0_product_allowed_warning_codes(
        root, manifest, device
    )
    _materialize_phase1_stage0_compatibility(
        root,
        output_dir,
        raw_dir=raw_dir,
        source_dir=source_dir,
        source_test_name=source_test_name,
        allowed_nonlegacy_warning_codes=allowed_stage0_warnings,
    )

    desired_conf = float(state.get("tracking_play_conf_threshold", 0.15))
    if not 0.0 < desired_conf <= 1.0:
        raise ValueError("tracking_play_conf_threshold must be in (0, 1]")
    stage1_wrapper = root / "target_centric_tracking_e2e_v1" / "run_stage1_with_conf_override.py"
    if not stage1_wrapper.is_file():
        raise FileNotFoundError(stage1_wrapper)
    run_command(
        [
            sys.executable,
            str(stage1_wrapper),
            "--frozen-stage1",
            str(manifest_script(root, manifest, "stage1")),
            "--confidence-threshold",
            str(desired_conf),
            "--project-root",
            str(root),
            "--test-name",
            source_test_name,
            "--device",
            device,
        ],
        root,
        output_dir / "logs" / "pipeline.log",
    )
    run_command(
        [
            sys.executable,
            str(manifest_script(root, manifest, "stage2")),
            "--project-root",
            str(root),
            "--test-name",
            source_test_name,
        ],
        root,
        output_dir / "logs" / "pipeline.log",
    )

    stage1 = read_json(stage1_summary)
    stage2 = read_json(stage2_summary)
    observed_conf = float((stage1.get("detector") or {}).get("confidence_threshold", -1.0))
    desired_conf = float(state.get("tracking_play_conf_threshold", 0.15))
    if abs(observed_conf - desired_conf) > 1e-9:
        raise RuntimeError(
            "Stage-1 detection cache uses a different RF-DETR threshold: "
            f"observed={observed_conf} desired={desired_conf}. "
            "Use a new tracking job/test name or explicitly overwrite stale runtime output."
        )
    if stage1.get("status") != "PASS" or stage2.get("status") != "PASS":
        raise RuntimeError("Frozen Stage-1/2 did not PASS")
    gate_path = source_dir / "compatibility_gate.json"
    gate = read_json(gate_path)
    gate.update(
        {
            "real_rfdetr_inference_performed": True,
            "real_same_shot_stage2_performed": True,
            "stage2_complete": True,
        }
    )
    atomic_json(gate_path, gate)
    return source_dir


def run_initial_memory_stage(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    manifest_path: Path,
    device: str,
    safe_weights_runner: Path | None,
) -> tuple[bool, Optional[str]]:
    """Run A1/A2. Return (success, error_message)."""
    first_cut = int(state["cut_frames"][0])
    v2_test_name = state["work"]["v2_memory_test_name"]
    v2_root = resolve(root, DEFAULT_V2_OUTPUT_ROOT)
    v2_dir = v2_root / v2_test_name
    materialize_a0_for_v2_test(output_dir, v2_dir, state["artifacts"]["stage3a0_summary_payload"])

    a1_summary = v2_dir / "stage3a1_summary.json"
    if not a1_summary.is_file():
        command = [
            sys.executable,
            str(root / "target_centric_tracking_v2" / "stage3a1_confirm_cross_shot_boundary.py"),
            "--project-root",
            str(root),
            "--test-name",
            v2_test_name,
            "--confirmed-cut-frame",
            str(first_cut),
            "--review-status",
            "PASS",
            "--reviewer",
            "E2E_AUTO_SEGMENTER",
            "--review-note",
            "Automatic hard-cut boundary accepted only to reset shot-local motion. Identity was not linked.",
        ]
        run_command(command, root, output_dir / "logs" / "pipeline.log")

    a2_summary = v2_dir / "stage3a2_summary.json"
    if a2_summary.is_file():
        return True, None

    source_test_name = state["work"]["phase1_source_test_name"]
    try:
        source_dir = ensure_precut_phase1_stage2_source(
            root,
            output_dir,
            state,
            read_json(manifest_path),
            device,
        )
    except Exception as exc:
        return False, f"Pre-cut frozen Stage-1/2 preparation failed: {type(exc).__name__}: {exc}"

    compatibility_gate = source_dir / "compatibility_gate.json"
    command = [
        sys.executable,
        str(root / "target_centric_tracking_v2" / "stage3a2_build_precut_target_memory.py"),
        "--project-root",
        str(root),
        "--test-name",
        v2_test_name,
        "--phase1-source-test-name",
        source_test_name,
        "--phase1-manifest",
        str(manifest_path),
        "--device",
        device,
        "--skip-phase1-run",
    ]
    gate = read_json(compatibility_gate)
    if (
        gate.get("stage1_authorized") is not True
        or gate.get("real_rfdetr_inference_performed") is not True
        or not (source_dir / "detections.csv").is_file()
    ):
        return False, "Canonical Phase-1 compatibility gate is invalid."
    if safe_weights_runner is not None:
        command = [
            sys.executable,
            str(safe_weights_runner),
            *command[1:],
        ]
    completed = run_command(
        command,
        root,
        output_dir / "logs" / "pipeline.log",
        accepted_codes=(0, 2),
    )
    if completed.returncode != 0:
        return False, completed.stdout or "Stage 3-A2 failed"
    return True, None


def _jersey_color_signature(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Return a compact upper-torso color signature for team compatibility.

    The signature is deliberately simple and training-free. Hue bins describe
    colorful jersey pixels while three achromatic bins retain white/gray/black
    kit evidence. Background is reduced by using the central upper torso only.
    """

    if crop_bgr is None or crop_bgr.ndim != 3 or crop_bgr.size == 0:
        return None
    height, width = crop_bgr.shape[:2]
    if height < 12 or width < 8:
        return None
    y1 = max(0, int(round(height * 0.12)))
    y2 = min(height, max(y1 + 2, int(round(height * 0.62))))
    x1 = max(0, int(round(width * 0.15)))
    x2 = min(width, max(x1 + 2, int(round(width * 0.85))))
    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    valid = val >= 28.0
    if int(valid.sum()) < 20:
        return None

    colorful = valid & (sat >= 45.0)
    hue_hist = np.zeros(12, dtype=np.float32)
    if colorful.any():
        indices = np.minimum(11, (hue[colorful] / 15.0).astype(np.int32))
        weights = 0.25 + (sat[colorful] / 255.0)
        hue_hist = np.bincount(indices, weights=weights, minlength=12).astype(np.float32)
        hue_hist /= max(float(hue_hist.sum()), 1e-12)

    achromatic = valid & (sat < 45.0)
    valid_count = max(float(valid.sum()), 1.0)
    achromatic_features = np.array(
        [
            float((achromatic & (val < 80.0)).sum()) / valid_count,
            float((achromatic & (val >= 80.0) & (val < 180.0)).sum()) / valid_count,
            float((achromatic & (val >= 180.0)).sum()) / valid_count,
        ],
        dtype=np.float32,
    )
    # Keep the feature directional: two strongly saturated but different jersey
    # hues should not look similar merely because both are "colorful".
    signature = np.concatenate([hue_hist, achromatic_features]).astype(np.float32)
    norm = float(np.linalg.norm(signature))
    if norm <= 1e-12:
        return None
    return signature / norm


def _aggregate_team_signatures(signatures: Sequence[np.ndarray]) -> np.ndarray | None:
    valid = [np.asarray(item, dtype=np.float32) for item in signatures if item is not None]
    if not valid:
        return None
    matrix = np.stack(valid).astype(np.float32)
    prototype = np.median(matrix, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-12:
        return None
    return prototype / norm


def _team_signature_similarity(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.dot(left, right))
    return max(0.0, min(1.0, value))


def _build_target_team_profile(
    *,
    video_path: Path,
    timeline_path: Path,
) -> dict[str, Any]:
    if not video_path.is_file() or not timeline_path.is_file():
        return {
            "policy_version": TEAM_PROFILE_POLICY_VERSION,
            "available": False,
            "sample_count": 0,
            "reason": "SOURCE_NOT_AVAILABLE",
        }
    timeline = read_json(timeline_path)
    frames = timeline.get("frames") or []
    candidates: list[tuple[float, int, list[float]]] = []
    for row in frames:
        if not isinstance(row, Mapping):
            continue
        bbox = row.get("bbox_xyxy")
        if (
            str(row.get("state") or "") not in CONFIRMED_STATES
            or not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            continue
        try:
            frame_index = int(row["frame_index"])
            values = [float(value) for value in bbox]
            tracking_conf = float(row.get("tracking_confidence") or 0.0)
            identity_conf = float(row.get("identity_confidence") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        area = max(0.0, values[2] - values[0]) * max(0.0, values[3] - values[1])
        quality = (0.45 * tracking_conf) + (0.45 * identity_conf) + (0.10 * math.log1p(area))
        candidates.append((quality, frame_index, values))
    if not candidates:
        return {
            "policy_version": TEAM_PROFILE_POLICY_VERSION,
            "available": False,
            "sample_count": 0,
            "reason": "NO_CONFIRMED_TARGET_BBOX",
        }

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[float, int, list[float]]] = []
    minimum_gap = 3
    for item in candidates:
        if any(abs(item[1] - prior[1]) < minimum_gap for prior in selected):
            continue
        selected.append(item)
        if len(selected) >= TEAM_PROFILE_TARGET_MAX_SAMPLES:
            break
    if not selected:
        selected = candidates[:1]

    capture = cv2.VideoCapture(str(video_path))
    signatures: list[np.ndarray] = []
    used_frames: list[int] = []
    try:
        if not capture.isOpened():
            return {
                "policy_version": TEAM_PROFILE_POLICY_VERSION,
                "available": False,
                "sample_count": 0,
                "reason": "VIDEO_OPEN_FAILED",
            }
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for _, frame_index, bbox in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            x1 = max(0, min(width - 2, int(math.floor(bbox[0]))))
            y1 = max(0, min(height - 2, int(math.floor(bbox[1]))))
            x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2]))))
            y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3]))))
            crop = frame[y1:y2, x1:x2]
            signature = _jersey_color_signature(crop)
            if signature is not None:
                signatures.append(signature)
                used_frames.append(frame_index)
    finally:
        capture.release()

    prototype = _aggregate_team_signatures(signatures)
    return {
        "policy_version": TEAM_PROFILE_POLICY_VERSION,
        "available": prototype is not None and len(signatures) >= TEAM_PROFILE_MIN_TARGET_SAMPLES,
        "sample_count": len(signatures),
        "source_frames": used_frames,
        "prototype": prototype.tolist() if prototype is not None else None,
        "reason": (
            "READY"
            if prototype is not None and len(signatures) >= TEAM_PROFILE_MIN_TARGET_SAMPLES
            else "INSUFFICIENT_STABLE_COLOR_SAMPLES"
        ),
    }


def _apply_team_compatibility_prefilter(
    ranked_rows: list[dict[str, Any]],
    *,
    selected_by_candidate: Mapping[str, Sequence[Any]],
    crops: Mapping[str, np.ndarray],
    target_profile: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = dict(target_profile or {})
    raw_prototype = profile.get("prototype")
    if not profile.get("available") or not isinstance(raw_prototype, list):
        for row in ranked_rows:
            row["team_compatibility_status"] = "NOT_AVAILABLE"
            row["team_compatibility_score"] = None
        return ranked_rows, {
            "policy_version": TEAM_PROFILE_POLICY_VERSION,
            "applied": False,
            "reason": str(profile.get("reason") or "TARGET_PROFILE_NOT_AVAILABLE"),
            "rejected_candidate_ids": [],
        }

    target = np.asarray(raw_prototype, dtype=np.float32)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        return ranked_rows, {
            "policy_version": TEAM_PROFILE_POLICY_VERSION,
            "applied": False,
            "reason": "TARGET_PROFILE_INVALID",
            "rejected_candidate_ids": [],
        }
    target = target / norm

    valid_scores: list[float] = []
    for row in ranked_rows:
        candidate_id = str(row["candidate_id"])
        signatures = [
            signature
            for detection in selected_by_candidate.get(candidate_id, [])
            if (signature := _jersey_color_signature(crops.get(str(detection.detection_id))))
            is not None
        ]
        prototype = _aggregate_team_signatures(signatures)
        if prototype is None or len(signatures) < TEAM_PROFILE_MIN_CANDIDATE_SAMPLES:
            row["team_compatibility_status"] = "INSUFFICIENT_EVIDENCE"
            row["team_compatibility_score"] = None
            row["team_profile_sample_count"] = len(signatures)
            continue
        score = _team_signature_similarity(target, prototype)
        row["team_compatibility_status"] = "EVALUATED"
        row["team_compatibility_score"] = round(score, 6)
        row["team_profile_sample_count"] = len(signatures)
        valid_scores.append(score)

    if len(valid_scores) < 2:
        return ranked_rows, {
            "policy_version": TEAM_PROFILE_POLICY_VERSION,
            "applied": False,
            "reason": "INSUFFICIENT_COMPARABLE_CANDIDATES",
            "rejected_candidate_ids": [],
        }

    best = max(valid_scores)
    rejected: list[str] = []
    kept: list[dict[str, Any]] = []
    for row in ranked_rows:
        score_value = row.get("team_compatibility_score")
        reject = False
        if score_value is not None:
            score = float(score_value)
            reject = (
                best >= TEAM_PREFILTER_MIN_BEST_SIMILARITY
                and score < TEAM_PREFILTER_MAX_INCOMPATIBLE_SIMILARITY
                and (best - score) >= TEAM_PREFILTER_MIN_GAP_FROM_BEST
            )
        if reject:
            row["team_compatibility_status"] = "CLEARLY_INCOMPATIBLE"
            rejected.append(str(row["candidate_id"]))
        else:
            kept.append(row)

    # Never erase the entire shot candidate set on color evidence alone.
    if not kept:
        for row in ranked_rows:
            if row.get("team_compatibility_status") == "CLEARLY_INCOMPATIBLE":
                row["team_compatibility_status"] = "INCONCLUSIVE_FAIL_OPEN"
        kept = ranked_rows
        rejected = []
        applied = False
        reason = "FILTER_WOULD_REMOVE_ALL_CANDIDATES"
    else:
        applied = bool(rejected)
        reason = "CLEAR_TEAM_MISMATCH_FILTERED" if rejected else "NO_CLEAR_TEAM_MISMATCH"

    return kept, {
        "policy_version": TEAM_PROFILE_POLICY_VERSION,
        "applied": applied,
        "reason": reason,
        "target_sample_count": int(profile.get("sample_count") or 0),
        "best_candidate_similarity": round(best, 6),
        "rejected_candidate_ids": rejected,
        "thresholds": {
            "minimum_best_similarity": TEAM_PREFILTER_MIN_BEST_SIMILARITY,
            "maximum_incompatible_similarity": TEAM_PREFILTER_MAX_INCOMPATIBLE_SIMILARITY,
            "minimum_gap_from_best": TEAM_PREFILTER_MIN_GAP_FROM_BEST,
        },
    }


def memory_paths(root: Path, state: Mapping[str, Any]) -> dict[str, Path]:
    v2_dir = resolve(root, DEFAULT_V2_OUTPUT_ROOT) / state["work"]["v2_memory_test_name"]
    summary = read_json(v2_dir / "stage3a2_summary.json")
    memory = read_json(v2_dir / "stage3a2_target_memory.json")
    return {
        "v2_dir": v2_dir,
        "summary": v2_dir / "stage3a2_summary.json",
        "memory": v2_dir / "stage3a2_target_memory.json",
        "target_embeddings": Path(memory["embeddings"]["target_path"]).resolve(),
        "negative_embeddings": Path(memory["embeddings"]["negative_path"]).resolve(),
        "contact_sheet": Path(summary["outputs"]["memory_contact_sheet"]).resolve(),
        "tracking_preview": Path(summary["outputs"]["precut_tracking_preview"]).resolve(),
        "source_timeline": Path(summary["phase1_source"]["timeline"]).resolve(),
        "source_detections": Path(summary["phase1_source"]["detections"]).resolve(),
        "source_output_dir": Path(summary["phase1_source"]["output_dir"]).resolve(),
    }


def prepare_memory_contract(root: Path, output_dir: Path, state: dict[str, Any]) -> None:
    paths = memory_paths(root, state)
    target = np.load(paths["target_embeddings"], allow_pickle=False).astype(np.float32)
    negative = np.load(paths["negative_embeddings"], allow_pickle=False).astype(np.float32)
    if target.ndim != 2 or target.shape[0] < 4:
        raise RuntimeError(f"Invalid target memory shape: {target.shape}")
    norms = np.linalg.norm(target, axis=1, keepdims=True)
    target = target / np.maximum(norms, 1e-12)
    prototype = target.mean(axis=0)
    prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
    prototype_path = output_dir / "target_memory_prototype.npy"
    temporary = prototype_path.with_name(prototype_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, prototype.astype(np.float32), allow_pickle=False)
    os.replace(temporary, prototype_path)
    state["memory"] = {
        "status": "VISUAL_REVIEW_PASS",
        "source_stage": "STAGE3A2",
        "target_embeddings": str(paths["target_embeddings"]),
        "target_embeddings_sha256": sha256_file(paths["target_embeddings"]),
        "negative_embeddings": str(paths["negative_embeddings"]),
        "negative_embeddings_sha256": sha256_file(paths["negative_embeddings"]),
        "target_prototype": str(prototype_path),
        "target_prototype_sha256": sha256_file(prototype_path),
        "contact_sheet": str(paths["contact_sheet"]),
        "tracking_preview": str(paths["tracking_preview"]),
        "source_timeline": str(paths["source_timeline"]),
        "source_detections": str(paths["source_detections"]),
        "source_output_dir": str(paths["source_output_dir"]),
        "approved_at": now_iso(),
    }
    state["memory"]["target_team_profile"] = _build_target_team_profile(
        video_path=Path(state["video"]["path"]).resolve(),
        timeline_path=paths["source_timeline"],
    )


def _write_reference_contact_sheet(
    output: Path,
    crops: Sequence[tuple[str, np.ndarray]],
) -> None:
    if not crops:
        return
    thumb_w, thumb_h = 220, 320
    cells: list[np.ndarray] = []
    for label, crop in crops:
        canvas = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        h, w = crop.shape[:2]
        scale = min((thumb_w - 12) / max(w, 1), (thumb_h - 42) / max(h, 1))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (thumb_w - nw) // 2
        y = 30 + (thumb_h - 30 - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        cv2.putText(
            canvas,
            label[:28],
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(canvas)
    columns = min(4, len(cells))
    rows = int(math.ceil(len(cells) / columns))
    blank = np.zeros_like(cells[0])
    row_images = []
    for row_index in range(rows):
        row = cells[row_index * columns:(row_index + 1) * columns]
        row = [*row, *[blank.copy() for _ in range(columns - len(row))]]
        row_images.append(np.hstack(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack(row_images)):
        raise RuntimeError(f"Failed to write reference contact sheet: {output}")


def prepare_trusted_selected_reference_memory(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    manifest_path: Path,
    device_name: str,
) -> None:
    """Seed identity memory from the immutable user-confirmed candidate gallery.

    This is a product integration layer only. Frozen V1/V2 files and thresholds
    remain untouched. The user-selected gallery supplies positive identity
    evidence. Conservative non-target person crops are mined only from the
    confirmed initial shot and are excluded whenever they overlap/approach the
    tracked target, recreating the target-vs-confuser memory used by the final
    research design without requiring a separate human memory review.
    """

    reference_contract = state.get("target_reference_set")
    if not isinstance(reference_contract, Mapping):
        raise RuntimeError("Trusted selected reference memory has no reference contract")
    reference_path = Path(str(reference_contract.get("path") or "")).expanduser().resolve()
    expected_sha = str(reference_contract.get("sha256") or "").strip().lower()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    actual_sha = sha256_file(reference_path).lower()
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise RuntimeError("Immutable target reference set hash mismatch")

    payload = read_json(reference_path)
    references = payload.get("references")
    if not isinstance(references, list) or len(references) < 1:
        raise RuntimeError("At least one user-confirmed target reference is required")

    crop_map: dict[str, np.ndarray] = {}
    crop_rows: list[tuple[str, np.ndarray]] = []
    reference_provenance: list[dict[str, Any]] = []
    signatures: list[np.ndarray] = []
    for index, raw in enumerate(references):
        if not isinstance(raw, Mapping):
            raise RuntimeError("Target reference entry is invalid")
        raw_crop_value = raw.get("path") or raw.get("crop_artifact")
        if not raw_crop_value:
            raise RuntimeError("Target reference entry has no crop artifact")
        raw_path = Path(str(raw_crop_value)).expanduser()
        crop_path = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (reference_path.parent / raw_path).resolve()
        )
        if not raw_path.is_absolute() and not crop_path.is_relative_to(reference_path.parent.resolve()):
            raise RuntimeError("Target reference crop escapes immutable reference root")
        if not crop_path.is_file():
            raise FileNotFoundError(crop_path)
        expected_crop_sha = str(raw.get("sha256") or raw.get("crop_sha256") or "").strip().lower()
        actual_crop_sha = sha256_file(crop_path).lower()
        if expected_crop_sha and (len(expected_crop_sha) != 64 or actual_crop_sha != expected_crop_sha):
            raise RuntimeError(f"Target reference crop hash mismatch: {crop_path}")
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None or crop.size == 0:
            raise RuntimeError(f"Target reference crop is unreadable: {crop_path}")
        key = f"reference_{index:04d}"
        crop_map[key] = crop
        crop_rows.append((f"{index + 1}:{raw.get('scale', 'unknown')}", crop))
        signature = _jersey_color_signature(crop)
        if signature is not None:
            signatures.append(signature)
        reference_provenance.append(
            {
                "reference_id": key,
                "frame_id": raw.get("frame_id", raw.get("frame_index")),
                "path": str(crop_path),
                "sha256": actual_crop_sha,
                "source_hash_was_declared": bool(expected_crop_sha),
                "scale": str(raw.get("scale") or raw.get("scale_class") or "unknown"),
                "source_candidate_id": raw.get("source_candidate_id"),
            }
        )

    manifest = read_json(manifest_path)
    source_dir = ensure_precut_phase1_stage2_source(
        root,
        output_dir,
        state,
        manifest,
        device_name,
    )
    timeline_candidates = [
        source_dir / "target_timeline.json",
        source_dir / "stage2_target_timeline.json",
    ]
    source_timeline = next((item for item in timeline_candidates if item.is_file()), None)
    if source_timeline is None:
        raise RuntimeError("Stage-2 source timeline is missing")
    source_detections = source_dir / "detections.csv"
    if not source_detections.is_file():
        raise RuntimeError("Stage-1 source detections are missing")

    paths = module_paths(root)
    stage2b = load_module("kickclip_selected_memory_stage2b", paths["stage2b"])
    reid = load_module("kickclip_selected_memory_reid", paths["reid"])
    checkpoint = resolve(
        root,
        normalized_relative_path(str(manifest["models"]["sports_osnet"]["path"])),
    )
    expected_checkpoint = str(manifest["models"]["sports_osnet"].get("sha256") or "")
    if not checkpoint.is_file() or (
        expected_checkpoint
        and sha256_file(checkpoint).lower() != expected_checkpoint.lower()
    ):
        raise RuntimeError("Frozen Sports OSNet checkpoint is missing or changed")

    _, models, _ = stage2b.discover_deep_eiou(root, reid, None)
    import torch

    selected_device = (
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if selected_device == "auto":
        selected_device = "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(selected_device)
    if hasattr(reid, "configure_determinism"):
        reid.configure_determinism(torch)
    safe_loader_compat = _register_product_sports_osnet_safe_globals(torch)
    model, model_contract = reid.build_model(torch, models, checkpoint, device)
    model_contract = dict(model_contract)
    model_contract["product_numpy_safe_globals_compat"] = safe_loader_compat
    transform = stage2b.build_transform()
    negative_crops, negative_provenance = _collect_initial_shot_negative_crops(
        video_path=Path(state["video"]["path"]).resolve(),
        detections_csv=source_detections,
        timeline_path=source_timeline,
        first_shot_end=int(state["shots"][0]["end_frame_inclusive"]),
        tracking_threshold=float(state.get("tracking_play_conf_threshold", 0.15)),
    )
    all_crops = {**crop_map, **negative_crops}
    embedded = stage2b.embed_crops(
        all_crops,
        model,
        transform,
        torch,
        device,
        16,
    )
    if len(embedded) != len(all_crops):
        raise RuntimeError("Sports OSNet did not embed every target/negative reference crop")
    target = np.stack([embedded[key] for key in crop_map], axis=0).astype(np.float32)
    norms = np.linalg.norm(target, axis=1, keepdims=True)
    target = target / np.maximum(norms, 1e-12)
    prototype = target.mean(axis=0).astype(np.float32)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
    if negative_crops:
        negative = np.stack([embedded[key] for key in negative_crops], axis=0).astype(np.float32)
        negative_norms = np.linalg.norm(negative, axis=1, keepdims=True)
        negative = negative / np.maximum(negative_norms, 1e-12)
    else:
        negative = np.empty((0, target.shape[1]), dtype=np.float32)
    negative_memory_ready = negative.shape[0] >= PRODUCT_NEGATIVE_MEMORY_MIN_REFERENCES
    if not negative_memory_ready:
        # Fewer than the minimum conservative negatives are not used for auto-safe
        # acceptance. Keep the saved matrix for audit, but mark the memory as
        # reference-only so cross-shot links continue to require review.
        negative = np.empty((0, target.shape[1]), dtype=np.float32)

    memory_dir = output_dir / "work" / "selected_reference_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    target_path = memory_dir / "target_embeddings.npy"
    negative_path = memory_dir / "negative_embeddings.npy"
    prototype_path = output_dir / "target_memory_prototype.npy"
    np.save(target_path, target, allow_pickle=False)
    np.save(negative_path, negative, allow_pickle=False)
    np.save(prototype_path, prototype, allow_pickle=False)
    contact_sheet = memory_dir / "selected_reference_contact_sheet.jpg"
    _write_reference_contact_sheet(contact_sheet, crop_rows)

    team_prototype = _aggregate_team_signatures(signatures)
    team_profile = {
        "policy_version": TEAM_PROFILE_POLICY_VERSION,
        "available": (
            team_prototype is not None
            and len(signatures) >= TEAM_PROFILE_MIN_TARGET_SAMPLES
        ),
        "sample_count": len(signatures),
        "source": "USER_SELECTED_REFERENCE_SET",
        "prototype": team_prototype.tolist() if team_prototype is not None else None,
        "reason": (
            "READY"
            if team_prototype is not None
            and len(signatures) >= TEAM_PROFILE_MIN_TARGET_SAMPLES
            else "INSUFFICIENT_REFERENCE_COLOR_SAMPLES"
        ),
    }

    state["memory"] = {
        "status": "USER_CONFIRMED_REFERENCE_MEMORY",
        "source_stage": "BACKEND_USER_SELECTED_REFERENCE_SET",
        "backend_memory_used_for_scoring": True,
        "reference_only_negative_memory": not negative_memory_ready,
        "negative_memory_source": (
            "INITIAL_SHOT_CONFIRMED_TARGET_EXCLUSION"
            if negative_memory_ready
            else "INSUFFICIENT_CONSERVATIVE_NEGATIVE_REFERENCES"
        ),
        "negative_reference_count": int(negative.shape[0]),
        "negative_references": negative_provenance if negative_memory_ready else [],
        "target_reference_set_path": str(reference_path),
        "target_reference_set_sha256": actual_sha,
        "reference_count": len(reference_provenance),
        "references": reference_provenance,
        "target_embeddings": str(target_path),
        "target_embeddings_sha256": sha256_file(target_path),
        "negative_embeddings": str(negative_path),
        "negative_embeddings_sha256": sha256_file(negative_path),
        "target_prototype": str(prototype_path),
        "target_prototype_sha256": sha256_file(prototype_path),
        "contact_sheet": str(contact_sheet),
        "tracking_preview": str(source_dir / "stage2_target_tracking_preview.mp4"),
        "source_timeline": str(source_timeline),
        "source_detections": str(source_detections),
        "source_output_dir": str(source_dir),
        "target_team_profile": team_profile,
        "sports_osnet_checkpoint": str(checkpoint),
        "sports_osnet_checkpoint_sha256": sha256_file(checkpoint),
        "sports_osnet_model_contract": model_contract,
        "approved_at": now_iso(),
        "automatic_target_confirmation": False,
    }
    state.setdefault("artifacts", {})["trusted_selected_reference_memory"] = {
        "target_reference_set": str(reference_path),
        "target_reference_set_sha256": actual_sha,
        "contact_sheet": str(contact_sheet),
        "positive_reference_count": len(reference_provenance),
        "negative_reference_count": int(negative.shape[0]),
        "negative_memory_source": (
            "INITIAL_SHOT_CONFIRMED_TARGET_EXCLUSION"
            if negative_memory_ready
            else "INSUFFICIENT_CONSERVATIVE_NEGATIVE_REFERENCES"
        ),
        "used_for_scoring": True,
        "frozen_v1_v2_modified": False,
    }


def module_paths(root: Path) -> dict[str, Path]:
    paths = {
        "stage2": root / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
        "stage2b": root / "target_centric_tracking_v1" / "stage2b_run_same_shot_reentry_reacquisition.py",
        "b0": root / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        "b1": root / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        "b3": root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        "reid": root / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_runtime(root: Path, state: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    paths = module_paths(root)
    stage2 = load_module("kickclip_e2e_stage2", paths["stage2"])
    stage2b = load_module("kickclip_e2e_stage2b", paths["stage2b"])
    b0 = load_module("kickclip_e2e_b0", paths["b0"])
    b1 = load_module("kickclip_e2e_b1", paths["b1"])
    b3 = load_module("kickclip_e2e_b3", paths["b3"])
    reid = load_module("kickclip_e2e_reid", paths["reid"])

    video = state["video"]
    detections_path = Path(state["memory"]["source_detections"]).resolve()
    by_frame, detection_count = stage2.load_detections(
        detections_path,
        int(video["frame_count"]),
        int(video["width"]),
        int(video["height"]),
    )
    detection_by_id = {
        str(detection.detection_id): detection
        for detections in by_frame.values()
        for detection in detections
    }

    manifest = read_json(Path(state['phase1_manifest']['path']).resolve())
    checkpoint = resolve(
        root,
        normalized_relative_path(str(manifest["models"]["sports_osnet"]["path"])),
    )
    expected = str(manifest["models"]["sports_osnet"].get("sha256") or "")
    if not checkpoint.is_file() or (expected and sha256_file(checkpoint).lower() != expected.lower()):
        raise RuntimeError("Frozen Sports OSNet checkpoint is missing or changed")

    deep_eiou_root, models, reid_root = stage2b.discover_deep_eiou(root, reid, None)
    import torch

    selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if selected_device == "auto":
        selected_device = "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(selected_device)
    if hasattr(reid, "configure_determinism"):
        reid.configure_determinism(torch)
    safe_loader_compat = _register_product_sports_osnet_safe_globals(torch)
    model, model_contract = reid.build_model(torch, models, checkpoint, device)
    model_contract = dict(model_contract)
    model_contract["product_numpy_safe_globals_compat"] = safe_loader_compat
    transform = stage2b.build_transform()

    target_gallery = np.load(state["memory"]["target_embeddings"], allow_pickle=False).astype(np.float32)
    negative_gallery = np.load(state["memory"]["negative_embeddings"], allow_pickle=False).astype(np.float32)
    target_prototype = np.load(state["memory"]["target_prototype"], allow_pickle=False).astype(np.float32)
    target_gallery = b1.l2_normalize_rows(target_gallery)
    target_prototype = b1.l2_normalize_vector(target_prototype)
    if negative_gallery.shape[0] > 0:
        negative_gallery = b1.l2_normalize_rows(negative_gallery)

    return {
        "stage2": stage2,
        "stage2b": stage2b,
        "b0": b0,
        "b1": b1,
        "b3": b3,
        "reid": reid,
        "torch": torch,
        "device": device,
        "model": model,
        "transform": transform,
        "model_contract": model_contract,
        "checkpoint": checkpoint,
        "deep_eiou_root": deep_eiou_root,
        "reid_root": reid_root,
        "by_frame": by_frame,
        "detection_by_id": detection_by_id,
        "detection_count": detection_count,
        "target_gallery": target_gallery,
        "target_prototype": target_prototype,
        "negative_gallery": negative_gallery,
    }


def gate_candidate_rows(ranked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not ranked:
        return {
            "operational_state": "SAFE_REJECTED_SEARCHING",
            "decision": "RETAIN_SEARCHING_NO_CANDIDATES",
            "automatically_proposed_candidate": None,
            "all_auto_gates_passed": False,
            "plausible_candidate_exists": False,
            "gates": {},
        }
    top1 = ranked[0]
    top2_score = float(ranked[1]["retrieval_score"]) if len(ranked) > 1 else float(top1["retrieval_score"])
    top1_score = float(top1["retrieval_score"])
    top1_proto = float(top1["prototype_target_similarity"])
    top1_margin = float(top1["crop_margin_median"])
    top1_support = float(top1["positive_margin_support_ratio"])
    gap = top1_score - top2_score if len(ranked) > 1 else 0.0
    gates = {
        "retrieval_score": {
            "value": top1_score,
            "required": f">={B2_POLICY['minimum_retrieval_score']}",
            "passed": top1_score >= B2_POLICY["minimum_retrieval_score"],
        },
        "prototype_similarity": {
            "value": top1_proto,
            "required": f">={B2_POLICY['minimum_prototype_similarity']}",
            "passed": top1_proto >= B2_POLICY["minimum_prototype_similarity"],
        },
        "median_target_negative_margin": {
            "value": top1_margin,
            "required": f">{B2_POLICY['minimum_median_margin']}",
            "passed": top1_margin > B2_POLICY["minimum_median_margin"],
        },
        "positive_margin_support": {
            "value": top1_support,
            "required": f">={B2_POLICY['minimum_positive_margin_support']}",
            "passed": top1_support >= B2_POLICY["minimum_positive_margin_support"],
        },
        "top1_top2_score_gap": {
            "value": gap,
            "required": f">={B2_POLICY['minimum_top1_top2_gap']}",
            "passed": len(ranked) > 1 and gap >= B2_POLICY["minimum_top1_top2_gap"],
        },
    }
    all_pass = all(bool(item["passed"]) for item in gates.values())
    plausible = (
        top1_score >= B2_POLICY["minimum_plausible_score"]
        and top1_proto >= B2_POLICY["minimum_plausible_prototype"]
    )
    if all_pass:
        state = "AUTO_REACQUIRED_CANDIDATE"
        decision = "AUTO_REACQUIRED_BY_FROZEN_SAFETY_GATE"
        proposed = str(top1["candidate_id"])
    elif plausible:
        state = "AMBIGUOUS"
        decision = "AUTHORIZE_USER_CONFIRMATION_FALLBACK"
        proposed = None
    else:
        state = "SAFE_REJECTED_SEARCHING"
        decision = "RETAIN_SEARCHING_NO_CROSS_SHOT_TARGET_SELECTED"
        proposed = None
    return {
        "operational_state": state,
        "decision": decision,
        "automatically_proposed_candidate": proposed,
        "all_auto_gates_passed": all_pass,
        "plausible_candidate_exists": plausible,
        "top_candidate": dict(top1),
        "second_candidate": dict(ranked[1]) if len(ranked) > 1 else None,
        "gates": gates,
    }



def _read_video_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video}")
    return image


def _materialize_candidate_evidence(
    *,
    output_dir: Path,
    state: Mapping[str, Any],
    shot: Mapping[str, Any],
    candidate_id: str,
    selected_detections: Sequence[Any],
    crops: Mapping[str, np.ndarray],
    strip_path: Path,
    shot_clip_path: Path,
) -> dict[str, Any]:
    """Create review-only evidence without changing candidate scoring.

    The immutable bundle exists so the backend can show the exact candidate the
    canonical E2E ranked and can build an audit/provenance memory *after* an
    explicit human confirmation.  None of these files are read by Stage 3-B1
    scoring or by the Stage 3-B2 safety gate.
    """

    if not selected_detections:
        raise RuntimeError(f"Candidate has no selected evidence: {candidate_id}")
    if not strip_path.is_file():
        raise FileNotFoundError(strip_path)
    if not shot_clip_path.is_file():
        raise FileNotFoundError(shot_clip_path)

    candidate_root = (
        output_dir
        / "work"
        / "shots"
        / str(shot["shot_id"])
        / "candidate_evidence"
        / candidate_id
    )
    candidate_root.mkdir(parents=True, exist_ok=False)
    crop_dir = candidate_root / "reference_crops"
    crop_dir.mkdir(parents=True, exist_ok=False)

    offset = source_frame_offset(state)
    references: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for ordinal, detection in enumerate(selected_detections, start=1):
        detection_id = str(detection.detection_id)
        crop = crops.get(detection_id)
        if crop is None or crop.size == 0:
            raise RuntimeError(f"Candidate crop missing: {candidate_id}/{detection_id}")
        local_frame = int(detection.frame)
        source_frame = offset + local_frame
        crop_path = crop_dir / f"{ordinal:02d}_f{source_frame:06d}_{detection_id}.jpg"
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"Cannot write candidate crop: {crop_path}")
        bbox = [float(value) for value in detection.bbox]
        references.append(
            {
                "frame": source_frame,
                "local_frame": local_frame,
                "path": crop_path.relative_to(candidate_root).as_posix(),
                "crop_sha256": sha256_file(crop_path),
                # Scale is provenance only.  Do not add a new scale threshold to
                # the frozen tracking/reacquisition policy.
                "scale_class": "unknown",
                "detection_id": detection_id,
            }
        )
        observations.append(
            {
                "candidate_id": candidate_id,
                "frame": source_frame,
                "local_frame": local_frame,
                "detection_id": detection_id,
                "confidence": float(detection.confidence),
                "bbox_xyxy": bbox,
            }
        )

    best = max(
        selected_detections,
        key=lambda detection: (
            float(detection.confidence),
            -int(detection.frame),
            str(detection.detection_id),
        ),
    )
    best_local_frame = int(best.frame)
    best_source_frame = offset + best_local_frame
    best_bbox = [float(value) for value in best.bbox]
    full_frame = _read_video_frame(Path(state["video"]["path"]), best_local_frame)
    x1, y1, x2, y2 = [int(round(value)) for value in best_bbox]
    height, width = full_frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        full_frame,
        f"{candidate_id} | source frame {best_source_frame}",
        (max(8, x1), max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    full_frame_path = candidate_root / "full_frame_context.jpg"
    if not cv2.imwrite(str(full_frame_path), full_frame):
        raise RuntimeError(f"Cannot write candidate context: {full_frame_path}")

    gallery_path = candidate_root / "reference_gallery.jpg"
    shutil.copy2(strip_path, gallery_path)

    manifest_path = candidate_root / "candidate_manifest.json"
    manifest = {
        "schema_version": "kickclip.canonical_e2e_candidate_evidence.v1",
        "candidate_id": candidate_id,
        "candidate_media_id": candidate_id,
        "shot_id": str(shot["shot_id"]),
        "tracklet_id": candidate_id,
        "source_frame_offset": offset,
        "automatic_target_confirmation": False,
        "target_identity_confirmed": False,
        "quality": {
            # This means that the review bundle comes from one frozen Stage-3B0
            # local tracklet.  It is NOT a claim that this tracklet is the target.
            # Target identity is established only by the explicit review action.
            "identity_pure": True,
            "identity_pure_scope": "SINGLE_FROZEN_STAGE3B0_LOCAL_TRACKLET_ONLY",
            "target_identity_requires_human_confirmation": True,
            "reviewability": "REVIEWABLE_ASSISTED_CONFIRMATION",
            "selected_best_frame": best_source_frame,
            "selected_reference_count": len(references),
        },
        "best_observation": {
            "candidate_id": candidate_id,
            "frame": best_source_frame,
            "local_frame": best_local_frame,
            "detection_id": str(best.detection_id),
            "confidence": float(best.confidence),
            "bbox_xyxy": best_bbox,
        },
        "reference_gallery": references,
        "observations": observations,
        "provenance": {
            "candidate_generator": "FROZEN_STAGE3B0_TRACKLET",
            "candidate_ranker": "FROZEN_STAGE3B1_SPORTS_OSNET",
            "safety_gate": "FROZEN_STAGE3B2_POLICY",
            "backend_memory_used_for_scoring": False,
            "v7_runtime_dependency": False,
        },
    }
    atomic_json(manifest_path, manifest)

    return {
        "status": "PENDING",
        "identity_pure": True,
        "identity_pure_scope": "LOCAL_TRACKLET_ONLY_NOT_TARGET_IDENTITY",
        "automatic_target_confirmation": False,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "full_frame_context_path": str(full_frame_path.resolve()),
        "full_frame_context_sha256": sha256_file(full_frame_path),
        "shot_clip_path": str(shot_clip_path.resolve()),
        "shot_clip_sha256": sha256_file(shot_clip_path),
        "reference_gallery_path": str(gallery_path.resolve()),
        "reference_gallery_sha256": sha256_file(gallery_path),
        "best_frame_path": str(full_frame_path.resolve()),
        "best_frame_sha256": sha256_file(full_frame_path),
        "score_evidence": {
            "scoring_policy": "FROZEN_STAGE3B1_SPORTS_OSNET",
            "memory_source": "FROZEN_STAGE3A2_USER_REVIEWED_PRECUT_MEMORY",
            "backend_memory_used_for_scoring": False,
            "automatic_target_confirmation": False,
            "v7_runtime_dependency": False,
        },
    }


def _tracking_scene_mode_for_shot(
    *,
    state: Mapping[str, Any],
    runtime: Mapping[str, Any],
    shot: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one runtime shot for the WIDE-only product tracking policy.

    Stage-1 detections already exist densely for the canonical E2E source, so
    this adds no detector inference. A frame votes CLOSEUP when any sufficiently
    confident person-role box occupies a large fraction of the frame. The shot
    aggregates those frame votes using the same 0.70/0.30 policy used by
    candidate discovery. MIXED is deliberately non-trackable.
    """

    width = max(1, int(state["video"]["width"]))
    height = max(1, int(state["video"]["height"]))
    frame_area = float(width * height)
    start = int(shot["start_frame"])
    end = int(shot["end_frame_inclusive"])
    frame_count = max(1, end - start + 1)
    closeup_votes = 0
    person_detection_count = 0
    max_height_ratios: list[float] = []
    max_area_ratios: list[float] = []

    for frame_index in range(start, end + 1):
        frame_max_height = 0.0
        frame_max_area = 0.0
        for detection in runtime["by_frame"].get(frame_index, []):
            if int(getattr(detection, "class_id", -1)) not in PRODUCT_TRACKING_SCENE_PERSON_CLASS_IDS:
                continue
            if float(getattr(detection, "confidence", 0.0)) < PRODUCT_TRACKING_SCENE_OBSERVATION_CONFIDENCE:
                continue
            bbox = getattr(detection, "bbox_xyxy", None)
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            height_ratio = box_height / float(height)
            area_ratio = (box_width * box_height) / frame_area
            frame_max_height = max(frame_max_height, height_ratio)
            frame_max_area = max(frame_max_area, area_ratio)
            person_detection_count += 1
        max_height_ratios.append(frame_max_height)
        max_area_ratios.append(frame_max_area)
        if (
            frame_max_height >= PRODUCT_TRACKING_SCENE_CLOSEUP_MIN_HEIGHT_RATIO
            or frame_max_area >= PRODUCT_TRACKING_SCENE_CLOSEUP_MIN_AREA_RATIO
        ):
            closeup_votes += 1

    closeup_ratio = closeup_votes / float(frame_count)
    if closeup_ratio >= PRODUCT_TRACKING_SCENE_CLOSEUP_RATIO_MIN:
        mode = "CLOSEUP"
        reason = "DENSE_STAGE1_CLOSEUP_MAJORITY"
    elif closeup_ratio <= PRODUCT_TRACKING_SCENE_WIDE_RATIO_MAX:
        mode = "WIDE"
        reason = "DENSE_STAGE1_WIDE_MAJORITY"
    else:
        mode = "MIXED"
        reason = "DENSE_STAGE1_MIXED_TRANSITION"

    ordered_height = sorted(max_height_ratios)
    ordered_area = sorted(max_area_ratios)
    middle = frame_count // 2
    if frame_count % 2:
        median_height = ordered_height[middle]
        median_area = ordered_area[middle]
    else:
        median_height = 0.5 * (ordered_height[middle - 1] + ordered_height[middle])
        median_area = 0.5 * (ordered_area[middle - 1] + ordered_area[middle])

    return {
        "policy_version": PRODUCT_TRACKING_SCENE_POLICY_VERSION,
        "mode": mode,
        "reason": reason,
        "tracking_allowed": mode in PRODUCT_TRACKING_ALLOWED_SCENE_MODES,
        "frame_count": frame_count,
        "closeup_vote_count": closeup_votes,
        "closeup_vote_ratio": round(closeup_ratio, 8),
        "person_detection_count": person_detection_count,
        "median_max_person_height_ratio": round(float(median_height), 8),
        "median_max_person_area_ratio": round(float(median_area), 8),
        "thresholds": {
            "observation_confidence": PRODUCT_TRACKING_SCENE_OBSERVATION_CONFIDENCE,
            "closeup_min_height_ratio": PRODUCT_TRACKING_SCENE_CLOSEUP_MIN_HEIGHT_RATIO,
            "closeup_min_area_ratio": PRODUCT_TRACKING_SCENE_CLOSEUP_MIN_AREA_RATIO,
            "closeup_ratio_min": PRODUCT_TRACKING_SCENE_CLOSEUP_RATIO_MIN,
            "wide_ratio_max": PRODUCT_TRACKING_SCENE_WIDE_RATIO_MAX,
        },
    }


def process_shot_candidates(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    runtime: Mapping[str, Any],
    shot: Mapping[str, Any],
) -> dict[str, Any]:
    b0 = runtime["b0"]
    b1 = runtime["b1"]
    stage2b = runtime["stage2b"]
    start = int(shot["start_frame"])
    end = int(shot["end_frame_inclusive"])
    shot_dir = output_dir / "work" / "shots" / str(shot["shot_id"])
    if shot_dir.exists():
        shutil.rmtree(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    scene_policy = _tracking_scene_mode_for_shot(
        state=state,
        runtime=runtime,
        shot=shot,
    )
    atomic_json(shot_dir / "tracking_scene_policy.json", scene_policy)
    if not bool(scene_policy["tracking_allowed"]):
        gate = {
            "operational_state": "SAFE_REJECTED_SEARCHING",
            "decision": "EXCLUDE_NON_WIDE_SHOT_FROM_TARGET_TRACKING",
            "automatically_proposed_candidate": None,
            "all_auto_gates_passed": False,
            "plausible_candidate_exists": False,
            "gates": {},
        }
        atomic_json(shot_dir / "safe_gate.json", gate)
        return {
            "shot_id": shot["shot_id"],
            "status": "EXCLUDED_NON_WIDE_TRACKING",
            "candidate_count": 0,
            "ranked_candidates": [],
            "gate": gate,
            "shot_dir": str(shot_dir),
            "tracking_scene_policy": scene_policy,
        }

    role_rejected_counts: Counter[str] = Counter()
    player_by_frame: dict[int, list[Any]] = {}
    for frame_index in range(start, end + 1):
        kept_detections: list[Any] = []
        for detection in runtime["by_frame"].get(frame_index, []):
            if int(getattr(detection, "class_id", -1)) in PRODUCT_PLAYER_CLASS_IDS:
                kept_detections.append(detection)
            else:
                role_name = str(getattr(detection, "class_name", "unknown") or "unknown")
                role_rejected_counts[role_name] += 1
        if kept_detections:
            player_by_frame[frame_index] = kept_detections

    raw_tracks, _ = b0.build_tracklets(
        by_frame=player_by_frame,
        start_frame=start,
        end_frame_inclusive=end,
        max_age=B0_POLICY["max_age"],
        minimum_predicted_iou=B0_POLICY["minimum_predicted_iou"],
        maximum_center_distance=B0_POLICY["maximum_center_distance"],
        minimum_area_ratio=B0_POLICY["minimum_area_ratio"],
        maximum_area_ratio=B0_POLICY["maximum_area_ratio"],
        minimum_match_score=B0_POLICY["minimum_match_score"],
    )
    kept = [
        track
        for track in raw_tracks
        if len(track.observations) >= B0_POLICY["minimum_tracklet_frames"]
    ]
    kept.sort(key=lambda track: (track.start_frame, track.end_frame, track.internal_id))

    strip_dir = shot_dir / "candidate_strips"
    strip_dir.mkdir(parents=True, exist_ok=True)

    role_prefilter_audit = {
        "policy_version": "PLAYER_GOALKEEPER_ONLY_BEFORE_FROZEN_STAGE3B0_V1",
        "allowed_class_ids": sorted(PRODUCT_PLAYER_CLASS_IDS),
        "rejected_detection_counts_by_role": dict(sorted(role_rejected_counts.items())),
        "frozen_stage3b0_thresholds_modified": False,
    }

    atomic_json(
        shot_dir / "stage3b1_input_manifest.json",
        {
            "schema_version": "kickclip.stage3b1_input_manifest.canonical_e2e_v1",
            "shot_id": str(shot["shot_id"]),
            "target_embeddings_path": str(state["memory"]["target_embeddings"]),
            "target_embeddings_sha256": str(state["memory"]["target_embeddings_sha256"]),
            "negative_embeddings_path": str(state["memory"]["negative_embeddings"]),
            "negative_embeddings_sha256": str(state["memory"]["negative_embeddings_sha256"]),
            "sports_osnet_checkpoint": str(runtime["checkpoint"]),
            "sports_osnet_checkpoint_sha256": sha256_file(Path(runtime["checkpoint"])),
            "memory_source": "FROZEN_STAGE3A2_USER_REVIEWED_PRECUT_MEMORY",
            "backend_memory_used_for_scoring": False,
            "v7_runtime_dependency": False,
        },
    )

    video_path = Path(state["video"]["path"]).resolve()
    candidates: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    candidate_detections: dict[str, list[Any]] = {}
    track_by_id: dict[str, Any] = {}

    for number, track in enumerate(kept, start=1):
        candidate_id = f"{shot['shot_id']}_track_{number:04d}"
        track_by_id[candidate_id] = track
        detections: list[Any] = []
        for observation in track.observations:
            detection = runtime["detection_by_id"].get(str(observation.detection_id))
            if detection is None:
                raise RuntimeError(f"Detection missing: {observation.detection_id}")
            detections.append(detection)
            assignments.append(
                {
                    "candidate_id": candidate_id,
                    "frame_index": int(observation.frame_index),
                    "detection_id": str(observation.detection_id),
                    "confidence": float(observation.confidence),
                    "x1": float(observation.bbox_xyxy[0]),
                    "y1": float(observation.bbox_xyxy[1]),
                    "x2": float(observation.bbox_xyxy[2]),
                    "y2": float(observation.bbox_xyxy[3]),
                }
            )
        candidate_detections[candidate_id] = detections
        candidates.append(
            {
                "candidate_id": candidate_id,
                "start_frame": int(track.start_frame),
                "end_frame_inclusive": int(track.end_frame),
                "detection_count": len(track.observations),
            }
        )
        b0.make_tracklet_strip(
            video_path,
            candidate_id,
            track,
            strip_dir / f"{candidate_id}.jpg",
        )

    write_csv(shot_dir / "candidate_assignments.csv", assignments)
    atomic_json(shot_dir / "candidate_tracklets.json", {"candidates": candidates})
    atomic_json(shot_dir / "role_prefilter_audit.json", role_prefilter_audit)

    if not candidates:
        return {
            "shot_id": shot["shot_id"],
            "status": "SAFE_REJECTED_SEARCHING",
            "candidate_count": 0,
            "ranked_candidates": [],
            "gate": gate_candidate_rows([]),
            "shot_dir": str(shot_dir),
            "tracking_scene_policy": scene_policy,
        }

    selected_by_candidate: dict[str, list[Any]] = {}
    all_selected: dict[str, Any] = {}
    for candidate_id, detections in candidate_detections.items():
        selected = b1.select_diverse_detections(
            detections,
            B1_POLICY["max_crops_per_tracklet"],
            B1_POLICY["minimum_crop_gap"],
        )
        selected_by_candidate[candidate_id] = selected
        for detection in selected:
            all_selected[str(detection.detection_id)] = detection

    crops = stage2b.collect_crops(
        video_path,
        list(all_selected.values()),
        int(state["video"]["frame_count"]),
        int(state["video"]["width"]),
        int(state["video"]["height"]),
    )
    embeddings = stage2b.embed_crops(
        crops,
        runtime["model"],
        runtime["transform"],
        runtime["torch"],
        runtime["device"],
        32,
    )

    # Materialize review evidence once per shot.  These artifacts are strictly
    # downstream of the frozen ranker inputs and do not participate in scores.
    shot_clip_path = shot_dir / "shot_context.mp4"
    extract_clip_cv2(
        video_path,
        shot_clip_path,
        start,
        end,
        state["video"],
    )
    candidate_evidence: dict[str, dict[str, Any]] = {}
    for candidate_id, selected in selected_by_candidate.items():
        candidate_evidence[candidate_id] = _materialize_candidate_evidence(
            output_dir=output_dir,
            state=state,
            shot=shot,
            candidate_id=candidate_id,
            selected_detections=selected,
            crops=crops,
            strip_path=strip_dir / f"{candidate_id}.jpg",
            shot_clip_path=shot_clip_path,
        )

    automatic_rows: list[dict[str, Any]] = []
    for item in candidates:
        candidate_id = str(item["candidate_id"])
        selected = selected_by_candidate[candidate_id]
        matrix = np.stack(
            [embeddings[str(detection.detection_id)] for detection in selected]
        ).astype(np.float32)
        metrics = b1.compute_candidate_metrics(
            matrix,
            runtime["target_gallery"],
            runtime["target_prototype"],
            runtime["negative_gallery"],
        )
        metrics.pop("prototype", None)
        automatic_rows.append(
            {
                "candidate_id": candidate_id,
                "start_frame": int(item["start_frame"]),
                "end_frame_inclusive": int(item["end_frame_inclusive"]),
                "tracklet_detection_count": int(item["detection_count"]),
                "embedded_crop_count": int(matrix.shape[0]),
                **metrics,
                **candidate_evidence[candidate_id],
            }
        )
    automatic_rows.sort(
        key=lambda row: (
            float(row["retrieval_score"]),
            float(row["prototype_target_similarity"]),
            float(row["crop_margin_median"]),
            float(row["positive_margin_support_ratio"]),
        ),
        reverse=True,
    )
    for frozen_rank, row in enumerate(automatic_rows, start=1):
        row["frozen_retrieval_rank"] = frozen_rank

    automatic_rows, team_prefilter_audit = _apply_team_compatibility_prefilter(
        automatic_rows,
        selected_by_candidate=selected_by_candidate,
        crops=crops,
        target_profile=(state.get("memory") or {}).get("target_team_profile"),
    )
    for rank, row in enumerate(automatic_rows, start=1):
        row["retrieval_rank"] = rank
    atomic_json(shot_dir / "team_prefilter_audit.json", team_prefilter_audit)

    write_csv(shot_dir / "ranked_candidates.csv", automatic_rows)
    contact_sheet = shot_dir / "ranked_candidates.jpg"
    b1.make_ranked_contact_sheet(
        automatic_rows,
        strip_dir,
        contact_sheet,
        2,
    )
    gate = gate_candidate_rows(automatic_rows)
    atomic_json(shot_dir / "safe_gate.json", gate)

    return {
        "shot_id": shot["shot_id"],
        "status": gate["operational_state"],
        "candidate_count": len(automatic_rows),
        "ranked_candidates": automatic_rows,
        "gate": gate,
        "shot_dir": str(shot_dir),
        "assignments": str(shot_dir / "candidate_assignments.csv"),
        "contact_sheet": str(contact_sheet),
        "role_prefilter": role_prefilter_audit,
        "team_prefilter": team_prefilter_audit,
        "tracking_scene_policy": scene_policy,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def candidate_anchor(root: Path, ambiguity: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    b3 = load_module(
        "kickclip_e2e_anchor",
        root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
    )
    rows = [
        row
        for row in read_csv_rows(Path(ambiguity["assignments"]))
        if str(row["candidate_id"]) == candidate_id
    ]
    if not rows:
        raise RuntimeError(f"Candidate has no assignments: {candidate_id}")
    selected, scored = b3.choose_anchor(rows)
    return {"selected": selected, "scored": scored}


def extract_clip_cv2(
    video: Path,
    output: Path,
    start_frame: int,
    end_frame_inclusive: int,
    metadata: Mapping[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(metadata["fps"]),
        (int(metadata["width"]), int(metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open clip writer: {output}")
    count = 0
    try:
        for _ in range(start_frame, end_frame_inclusive + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended while extracting {output}")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    expected = end_frame_inclusive - start_frame + 1
    if count != expected:
        raise RuntimeError(f"Clip frame mismatch: {count}/{expected}")


def _state_runtime_phase1_manifest(
    state: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Resolve the already-verified portable Phase-1 manifest for this job.

    Continuation segments must use the same cross-platform manifest derivative
    as the initial target-memory path. Falling back to the source manifest is
    allowed only for older persisted states that predate the portable-manifest
    field.
    """

    contract = state.get("phase1_manifest")
    if not isinstance(contract, Mapping):
        raise RuntimeError("Phase-1 manifest contract is missing from pipeline state")
    raw = str(
        contract.get("runtime_portable_path")
        or contract.get("path")
        or ""
    ).strip()
    if not raw:
        raise RuntimeError("Phase-1 manifest path is missing from pipeline state")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, read_json(path)


def _run_segment_stage1_stage2_if_missing(
    *,
    root: Path,
    output_dir: Path,
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    clip_path: Path,
    phase1_test_name: str,
    phase1_dir: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Run only frozen Stage-1/2 for a continuation segment when needed.

    This product wrapper deliberately checks Stage-2 before entering the
    re-entry chain. Frozen Stage-2B itself rejects clips with no LOST frame as
    "unnecessary"; short confirmed tail segments commonly have exactly that
    shape. In that case Stage-2 is already the complete same-shot result and
    running Stage-2B would manufacture a product runtime failure.
    """

    desired_conf = float(state.get("tracking_play_conf_threshold", 0.15))
    if not 0.0 < desired_conf <= 1.0:
        raise ValueError("tracking_play_conf_threshold must be in (0, 1]")

    stage1_summary_path = phase1_dir / "stage1_detection_summary.json"
    stage2_summary_path = phase1_dir / "stage2_association_summary.json"
    stage2_timeline_path = phase1_dir / "target_timeline.json"

    if not stage1_summary_path.is_file():
        stage1_wrapper = (
            root
            / "target_centric_tracking_e2e_v1"
            / "run_stage1_with_conf_override.py"
        )
        if not stage1_wrapper.is_file():
            raise FileNotFoundError(stage1_wrapper)
        run_command(
            [
                sys.executable,
                str(stage1_wrapper),
                "--frozen-stage1",
                str(manifest_script(root, manifest, "stage1")),
                "--confidence-threshold",
                str(desired_conf),
                "--project-root",
                str(root),
                "--test-name",
                phase1_test_name,
                "--device",
                device,
            ],
            root,
            output_dir / "logs" / "pipeline.log",
        )

    stage1 = read_json(stage1_summary_path)
    observed_conf = float(
        (stage1.get("detector") or {}).get("confidence_threshold", -1.0)
    )
    if abs(observed_conf - desired_conf) > 1e-9:
        raise RuntimeError(
            "Continuation Stage-1 cache uses a different RF-DETR threshold: "
            f"observed={observed_conf} desired={desired_conf}"
        )
    if stage1.get("status") != "PASS":
        raise RuntimeError(
            f"Continuation frozen Stage-1 did not PASS: {stage1.get('status')}"
        )

    if not stage2_summary_path.is_file():
        run_command(
            [
                sys.executable,
                str(manifest_script(root, manifest, "stage2")),
                "--project-root",
                str(root),
                "--test-name",
                phase1_test_name,
            ],
            root,
            output_dir / "logs" / "pipeline.log",
        )

    stage2 = read_json(stage2_summary_path)
    if stage2.get("status") != "PASS":
        raise RuntimeError(
            f"Continuation frozen Stage-2 did not PASS: {stage2.get('status')}"
        )
    if not stage2_timeline_path.is_file():
        raise RuntimeError(
            "Continuation frozen Stage-2 completed without target_timeline.json"
        )
    return stage1, stage2, stage2_timeline_path


def _stage2_requires_reentry(stage2_timeline_path: Path) -> bool:
    """Return True only when frozen Stage-2 actually entered LOST."""

    payload = read_json(stage2_timeline_path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(
            f"Invalid continuation Stage-2 timeline: {stage2_timeline_path}"
        )
    return any(
        isinstance(row, Mapping)
        and str(row.get("state") or "").upper() == "LOST"
        for row in frames
    )


def _materialize_stage2_as_final_segment_timeline(
    *,
    phase1_dir: Path,
    stage2_timeline_path: Path,
    clip_meta: Mapping[str, Any],
) -> Path:
    """Finalize a continuation that never became LOST.

    No identity inference is added here. The output is a byte-for-byte copy of
    the frozen Stage-2 target timeline plus a separate product provenance file.
    This avoids calling frozen Stage-2B, whose own contract says it is
    unnecessary when Stage-2 has no LOST frame.
    """

    final_timeline = phase1_dir / "final_target_timeline.json"
    shutil.copy2(stage2_timeline_path, final_timeline)
    atomic_json(
        phase1_dir / "segment_phase1_product_finalize.json",
        {
            "schema_version": "kickclip.segment_phase1_product_finalize.v1",
            "created_at": now_iso(),
            "status": "COMPLETE",
            "decision": "FINALIZE_FROZEN_STAGE2_NO_REENTRY_REQUIRED",
            "reason": "FROZEN_STAGE2_HAS_NO_LOST_FRAME",
            "clip_frame_count": int(clip_meta["frame_count"]),
            "clip_duration_seconds": float(clip_meta["duration_seconds"]),
            "source_timeline": str(stage2_timeline_path.resolve()),
            "source_timeline_sha256": sha256_file(stage2_timeline_path),
            "final_timeline": str(final_timeline.resolve()),
            "final_timeline_sha256": sha256_file(final_timeline),
            "frozen_stage1_modified": False,
            "frozen_stage2_modified": False,
            "stage2b_executed": False,
            "automatic_target_confirmation": False,
        },
    )
    return final_timeline


def run_segment_phase1(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    segment: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Run frozen Phase-1 for a confirmed continuation segment.

    Product compatibility rules are identical to the initial Stage-0 path:
    - DURATION_OUTSIDE_RANGE is a historical research-window warning;
    - CUDA_NOT_AVAILABLE is expected when product device is auto/cpu;
    - SPORTS_OSNET_NOT_FOUND is a legacy discovery warning only when the exact
      frozen manifest checkpoint exists and its SHA-256 verifies.

    A continuation whose frozen Stage-2 timeline never enters LOST is finalized
    directly from Stage-2. Frozen Stage-2B explicitly treats that case as
    unnecessary, so invoking the re-entry chain would be an orchestration bug.
    """

    video = Path(state["video"]["path"]).resolve()
    clip_path = Path(segment["clip_path"]).resolve()
    if not clip_path.is_file():
        extract_clip_cv2(
            video,
            clip_path,
            int(segment["global_start_frame"]),
            int(segment["global_end_frame_inclusive"]),
            state["video"],
        )
    clip_meta = video_metadata(clip_path)
    segment["clip"] = clip_meta

    phase1_test_name = str(segment["phase1_test_name"])
    raw_stage0_name = f"{phase1_test_name}__raw_stage0"
    phase1_root = resolve(root, DEFAULT_V1_OUTPUT_ROOT)
    raw_dir = phase1_root / raw_stage0_name
    phase1_dir = phase1_root / phase1_test_name

    runtime_manifest_path, runtime_manifest = _state_runtime_phase1_manifest(
        state
    )
    allowed_stage0_warnings = stage0_product_allowed_warning_codes(
        root,
        runtime_manifest,
        device,
    )

    if not (raw_dir / "audit.json").is_file():
        run_command(
            [
                sys.executable,
                str(manifest_script(root, runtime_manifest, "stage0")),
                "--project-root",
                str(root),
                "--video",
                str(clip_path),
                "--test-name",
                raw_stage0_name,
                "--initial-bbox",
                *[str(v) for v in segment["initial_bbox_xyxy"]],
                "--bbox-format",
                "xyxy_pixels",
            ],
            root,
            output_dir / "logs" / "pipeline.log",
        )

    # Rebuild the derived Stage-0 directory only before Stage-1 has started.
    # This also repairs a continuation that previously stopped at the old,
    # overly-strict Stage-0 compatibility gate.
    if not (phase1_dir / "stage1_detection_summary.json").is_file():
        _materialize_phase1_stage0_compatibility(
            root,
            output_dir,
            raw_dir=raw_dir,
            source_dir=phase1_dir,
            source_test_name=phase1_test_name,
            allowed_nonlegacy_warning_codes=allowed_stage0_warnings,
        )

    _, _, stage2_timeline = _run_segment_stage1_stage2_if_missing(
        root=root,
        output_dir=output_dir,
        state=state,
        manifest=runtime_manifest,
        clip_path=clip_path,
        phase1_test_name=phase1_test_name,
        phase1_dir=phase1_dir,
        device=device,
    )

    if not _stage2_requires_reentry(stage2_timeline):
        timeline = _materialize_stage2_as_final_segment_timeline(
            phase1_dir=phase1_dir,
            stage2_timeline_path=stage2_timeline,
            clip_meta=clip_meta,
        )
        return {
            "status": "COMPLETE",
            "decision": "FINALIZE_FROZEN_STAGE2_NO_REENTRY_REQUIRED",
            "phase1_output_dir": str(phase1_dir),
            "timeline": str(timeline),
            "preview": str(phase1_dir / "stage2_target_tracking_preview.mp4"),
            "centered_preview": "",
            "stage2b_executed": False,
        }

    runner = (
        root
        / "target_centric_tracking_e2e_v1"
        / "run_phase1_product_pipeline.py"
    )
    if not runner.is_file():
        raise FileNotFoundError(runner)
    command = [
        sys.executable,
        str(runner),
        "--project-root",
        str(root),
        "--video",
        str(clip_path),
        "--test-name",
        phase1_test_name,
        "--initial-bbox",
        *[str(v) for v in segment["initial_bbox_xyxy"]],
        "--bbox-format",
        "xyxy_pixels",
        "--device",
        device,
        "--tracking-play-conf-threshold",
        str(float(state.get("tracking_play_conf_threshold", 0.15))),
        "--manifest",
        str(runtime_manifest_path),
        "--reviewer",
        state.get("reviewer", "USER"),
        "--review-note",
        "E2E confirmed/auto-safe anchor continuation segment",
        "--product-runtime-auto-review",
    ]
    reviews = segment.setdefault("phase1_reviews", {})
    mapping = {
        "STAGE2B": "--stage2b-visual-review",
        "STAGE2D": "--stage2d-visual-review",
        "STAGE2D1": "--stage2d1-visual-review",
        "STAGE2D2": "--stage2d2-visual-review",
    }
    for stage, flag in mapping.items():
        if reviews.get(stage) in {"PASS", "FAIL"}:
            command.extend([flag, reviews[stage]])

    completed = run_command(
        command,
        root,
        output_dir / "logs" / "pipeline.log",
        accepted_codes=(0, 3),
    )
    if completed.returncode == 3:
        pipeline_state_path = phase1_dir / "phase1_pipeline_state.json"
        phase1_state = read_json(pipeline_state_path)
        return {
            "status": "REVIEW_REQUIRED",
            "review_stage": str(phase1_state.get("review_stage")),
            "phase1_output_dir": str(phase1_dir),
            "preview": str(phase1_state.get("preview") or ""),
            "timeline": str(phase1_dir / "final_target_timeline.json"),
        }

    timeline = phase1_dir / "final_target_timeline.json"
    if not timeline.is_file():
        raise RuntimeError("Frozen Phase-1 completed without final target timeline")
    return {
        "status": "COMPLETE",
        "phase1_output_dir": str(phase1_dir),
        "timeline": str(timeline),
        "preview": str(phase1_dir / "final_target_tracking_preview.mp4"),
        "centered_preview": str(phase1_dir / "final_target_centered_preview.mp4"),
        "stage2b_executed": True,
    }

def default_frame(frame_index: int, fps: float, shot_id: str) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "time_seconds": frame_index / fps,
        "shot_id": shot_id,
        "state": "SEARCHING",
        "bbox_xyxy": None,
        "tracking_confidence": 0.0,
        "identity_confidence": 0.0,
        "identity_source": "NONE",
        "selected_detection_id": None,
        "decision_reason": "NO_CONFIRMED_TARGET_OBSERVATION",
        "review_required": False,
        "ambiguity_id": None,
    }


def normalize_bbox(value: Any, width: int, height: int) -> Optional[list[float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return clip_bbox([float(v) for v in value], width, height)
    except (TypeError, ValueError):
        return None


def overlay_timeline(
    target: list[dict[str, Any]],
    source: Mapping[str, Any],
    global_start: int,
    global_end: int,
    identity_source: str,
    first_state: Optional[str] = None,
    first_identity_source: Optional[str] = None,
    first_reason: Optional[str] = None,
) -> None:
    width = int(source["video"]["width"])
    height = int(source["video"]["height"])
    raw_frames = source.get("frames") or []
    for local_index, raw in enumerate(raw_frames):
        global_index = global_start + local_index
        if global_index > global_end or global_index >= len(target):
            break
        state = str(raw.get("state") or "SEARCHING")
        bbox = normalize_bbox(raw.get("bbox_xyxy"), width, height)
        if state in UNCERTAIN_STATES:
            bbox = None
        if bbox is None and state in CONFIRMED_STATES:
            state = "SEARCHING"
        row = dict(target[global_index])
        row.update(
            {
                "state": state,
                "bbox_xyxy": bbox,
                "tracking_confidence": max(0.0, min(1.0, float(raw.get("tracking_confidence") or 0.0))),
                "identity_confidence": max(0.0, min(1.0, float(raw.get("identity_confidence") or 0.0))),
                "identity_source": identity_source,
                "selected_detection_id": raw.get("selected_detection_id"),
                "decision_reason": str(raw.get("decision_reason") or ""),
                "review_required": bool(raw.get("review_required", False)),
            }
        )
        target[global_index] = row
    if first_state and 0 <= global_start < len(target):
        row = target[global_start]
        if row.get("bbox_xyxy") is not None:
            row["state"] = first_state
            row["identity_confidence"] = 1.0
            row["identity_source"] = first_identity_source or identity_source
            row["decision_reason"] = first_reason or "CONFIRMED_SEGMENT_ANCHOR"



def source_frame_offset(state: Mapping[str, Any]) -> int:
    return int(state.get("source_frame_offset") or 0)


def externalize_shot(shot: Mapping[str, Any], offset: int) -> dict[str, Any]:
    value = dict(shot)
    value["local_start_frame"] = int(shot["start_frame"])
    value["local_end_frame_inclusive"] = int(shot["end_frame_inclusive"])
    value["start_frame"] = offset + int(shot["start_frame"])
    value["end_frame_inclusive"] = offset + int(shot["end_frame_inclusive"])
    if shot.get("cut_in_frame") is not None:
        value["cut_in_frame"] = offset + int(shot["cut_in_frame"])
    if shot.get("cut_out_frame") is not None:
        value["cut_out_frame"] = offset + int(shot["cut_out_frame"])
    return value


def externalize_ambiguity(item: Mapping[str, Any], offset: int) -> dict[str, Any]:
    value = dict(item)
    if item.get("start_frame") is not None:
        value["local_start_frame"] = int(item["start_frame"])
        value["start_frame"] = offset + int(item["start_frame"])
    if item.get("end_frame_inclusive") is not None:
        value["local_end_frame_inclusive"] = int(item["end_frame_inclusive"])
        value["end_frame_inclusive"] = offset + int(item["end_frame_inclusive"])
    return value


def externalize_confirmation(item: Mapping[str, Any], offset: int) -> dict[str, Any]:
    value = dict(item)
    if item.get("anchor_frame") is not None:
        value["local_anchor_frame"] = int(item["anchor_frame"])
        value["anchor_frame"] = offset + int(item["anchor_frame"])
    return value


def externalize_segment(item: Mapping[str, Any], offset: int) -> dict[str, Any]:
    value = dict(item)
    if item.get("global_start_frame") is not None:
        value["local_start_frame"] = int(item["global_start_frame"])
        value["global_start_frame"] = offset + int(item["global_start_frame"])
    if item.get("global_end_frame_inclusive") is not None:
        value["local_end_frame_inclusive"] = int(item["global_end_frame_inclusive"])
        value["global_end_frame_inclusive"] = offset + int(item["global_end_frame_inclusive"])
    anchor = value.get("anchor")
    if isinstance(anchor, Mapping) and anchor.get("frame_index") is not None:
        anchor_value = dict(anchor)
        anchor_value["local_frame_index"] = int(anchor["frame_index"])
        anchor_value["frame_index"] = offset + int(anchor["frame_index"])
        value["anchor"] = anchor_value
    return value


def build_global_timeline(root: Path, output_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    video = state["video"]
    fps = float(video["fps"])
    frame_count = int(video["frame_count"])
    shot_by_frame: list[str] = ["shot_0000"] * frame_count
    for shot in state["shots"]:
        for index in range(int(shot["start_frame"]), int(shot["end_frame_inclusive"]) + 1):
            shot_by_frame[index] = str(shot["shot_id"])
    frames = [default_frame(index, fps, shot_by_frame[index]) for index in range(frame_count)]

    for shot in state["shots"]:
        if shot.get("status") == "ABSENT_CONFIRMED":
            for index in range(int(shot["start_frame"]), int(shot["end_frame_inclusive"]) + 1):
                frames[index]["state"] = "ABSENT"
                frames[index]["decision_reason"] = "USER_CONFIRMED_TARGET_ABSENT_IN_SHOT"
                frames[index]["identity_source"] = "USER_CONFIRMED_ABSENT"
        elif shot.get("status") == "EXCLUDED_NON_WIDE_TRACKING":
            for index in range(int(shot["start_frame"]), int(shot["end_frame_inclusive"]) + 1):
                frames[index]["state"] = "SEARCHING"
                frames[index]["bbox_xyxy"] = None
                frames[index]["tracking_confidence"] = 0.0
                frames[index]["identity_confidence"] = 0.0
                frames[index]["selected_detection_id"] = None
                frames[index]["decision_reason"] = "NON_WIDE_SHOT_EXCLUDED_FROM_TARGET_TRACKING"
                frames[index]["identity_source"] = "WIDE_ONLY_SCENE_POLICY"
                frames[index]["review_required"] = False

    if state.get("memory") and state["memory"].get("source_timeline"):
        source = read_json(Path(state["memory"]["source_timeline"]))
        first_end = int(state["shots"][0]["end_frame_inclusive"])
        overlay_timeline(
            frames,
            source,
            0,
            first_end,
            "INITIAL_USER_SELECTION_PHASE1",
        )

    for segment in state.get("segments", []):
        if segment.get("status") != "ACCEPTED":
            continue
        timeline_path = Path(segment["timeline"])
        if not timeline_path.is_file():
            raise FileNotFoundError(timeline_path)
        source = read_json(timeline_path)
        auto_safe = str(segment.get("source") or "") == "AUTO_SAFE_CROSS_SHOT_CANDIDATE"
        overlay_timeline(
            frames,
            source,
            int(segment["global_start_frame"]),
            int(segment["global_end_frame_inclusive"]),
            (
                "PHASE1_FROM_AUTO_SAFE_REACQUIRED_ANCHOR"
                if auto_safe
                else "PHASE1_FROM_USER_CONFIRMED_ANCHOR"
            ),
            first_state="REACQUIRED" if auto_safe else "USER_CONFIRMED",
            first_identity_source=(
                "AUTO_SAFE_REACQUIRED" if auto_safe else "USER_CONFIRMED"
            ),
            first_reason=(
                "FROZEN_STAGE3B2_SAFE_GATE_AUTO_REACQUIRED"
                if auto_safe
                else "USER_CONFIRMED_CROSS_SHOT_ANCHOR"
            ),
        )

    ambiguity_by_shot = {
        str(item["shot_id"]): str(item["ambiguity_id"])
        for item in state.get("ambiguities", [])
        if item.get("status") == "PENDING"
    }
    for frame in frames:
        ambiguity_id = ambiguity_by_shot.get(str(frame["shot_id"]))
        if ambiguity_id and frame["state"] in {"SEARCHING", "ABSENT"}:
            frame["state"] = "AMBIGUOUS"
            frame["ambiguity_id"] = ambiguity_id
            frame["review_required"] = True
            frame["decision_reason"] = "CROSS_SHOT_CANDIDATE_CONFIRMATION_REQUIRED"

    offset = source_frame_offset(state)
    source_video = dict(state.get("source_video") or video)
    source_fps = float(source_video["fps"])
    output_frames: list[dict[str, Any]] = []
    for row in frames:
        value = dict(row)
        local_index = int(row["frame_index"])
        value["local_frame_index"] = local_index
        value["frame_index"] = offset + local_index
        value["time_seconds"] = (offset + local_index) / source_fps
        value["time_ms"] = int(round(1000.0 * value["time_seconds"]))
        output_frames.append(value)

    timeline = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "test_name": state["test_name"],
        "target_id": "target_001",
        "status": state["status"],
        "video": source_video,
        "tracked_range": {
            "source_start_frame": offset,
            "source_end_frame_inclusive": offset + frame_count - 1,
            "source_start_time_seconds": offset / source_fps,
            "source_end_time_seconds": (offset + frame_count - 1) / source_fps,
            "tracking_direction": state.get(
                "tracking_direction", "FORWARD_FROM_CONFIRMED_ANCHOR"
            ),
        },
        "shots": [externalize_shot(item, offset) for item in state["shots"]],
        "frames": output_frames,
        "ambiguities": [
            externalize_ambiguity(item, offset)
            for item in state.get("ambiguities", [])
        ],
        "confirmations": [
            externalize_confirmation(item, offset)
            for item in state.get("confirmations", [])
        ],
        "provenance": {
            "phase1_manifest": state["phase1_manifest"],
            "frozen_phase1_modified": False,
            "frozen_phase2_modified": False,
            "threshold_search_performed": False,
            "automatic_memory_update_from_unreviewed_episode": False,
            "reacquisition_mode": state["reacquisition_mode"],
            "source_frame_offset": offset,
            "v7_runtime_dependency": False,
        },
    }
    atomic_json(output_dir / "target_timeline.json", timeline)
    write_csv(
        output_dir / "target_timeline.csv",
        [flatten_frame(row) for row in output_frames],
    )
    return timeline


def flatten_frame(item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item)
    bbox = row.pop("bbox_xyxy", None)
    if bbox is None:
        row.update({"bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": ""})
    else:
        row.update(
            {
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
            }
        )
    for key, value in list(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return row


def crop_rect(center_x: float, center_y: float, crop_h: float, width: int, height: int) -> list[float]:
    aspect = width / height
    crop_h = max(2.0, min(float(height), crop_h))
    crop_w = crop_h * aspect
    if crop_w > width:
        crop_w = float(width)
        crop_h = crop_w / aspect
    x1 = min(max(0.0, center_x - crop_w / 2), width - crop_w)
    y1 = min(max(0.0, center_y - crop_h / 2), height - crop_h)
    return [x1, y1, x1 + crop_w, y1 + crop_h]


def build_crop_rows(frames: Sequence[Mapping[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    full_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    smooth_center = full_center.copy()
    smooth_height = float(height)
    rows: list[dict[str, Any]] = []
    for frame in frames:
        bbox = frame.get("bbox_xyxy")
        if bbox is not None:
            desired_center = np.asarray(
                [0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])],
                dtype=np.float64,
            )
            bbox_height = bbox[3] - bbox[1]
            desired_height = min(float(height), max(height * 0.38, bbox_height / 0.42))
            smooth_center = 0.78 * smooth_center + 0.22 * desired_center
            smooth_height = 0.84 * smooth_height + 0.16 * desired_height
            source = "TARGET_BBOX"
        else:
            smooth_center = 0.92 * smooth_center + 0.08 * full_center
            smooth_height = 0.92 * smooth_height + 0.08 * float(height)
            source = "FULL_FRAME_FALLBACK"
        rect = crop_rect(float(smooth_center[0]), float(smooth_center[1]), smooth_height, width, height)
        rows.append({"frame_index": frame["frame_index"], "crop_xyxy": rect, "crop_source": source})
    return rows


def state_color(state: str) -> tuple[int, int, int]:
    if state in {"ACTIVE", "INITIALIZING", "USER_CONFIRMED"}:
        return (60, 220, 60)
    if state == "REACQUIRED":
        return (255, 80, 255)
    if state == "AMBIGUOUS":
        return (0, 80, 255)
    if state in {"OCCLUDED", "SEARCHING", "LOST", "ABSENT"}:
        return (0, 170, 255)
    return (180, 180, 180)


def render_previews(
    video_path: Path,
    output_dir: Path,
    timeline: Mapping[str, Any],
    print_every: int,
) -> None:
    video = timeline["video"]
    width = int(video["width"])
    height = int(video["height"])
    fps = float(video["fps"])
    frames = timeline["frames"]
    crop_rows = build_crop_rows(frames, width, height)
    tracking_path = output_dir / "full_frame_tracking_preview.mp4"
    centered_path = output_dir / "target_centered_preview.mp4"
    tracking = cv2.VideoWriter(
        str(tracking_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    centered = cv2.VideoWriter(
        str(centered_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened() or not tracking.isOpened() or not centered.isOpened():
        capture.release()
        tracking.release()
        centered.release()
        raise RuntimeError("Cannot open E2E preview readers/writers")
    try:
        for index, frame_info in enumerate(frames):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {index}")
            state = str(frame_info["state"])
            color = state_color(state)
            annotated = image.copy()
            bbox = frame_info.get("bbox_xyxy")
            if bbox is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"target_001 | {state} | {frame_info['shot_id']}"
            cv2.rectangle(annotated, (0, 0), (min(width - 1, 700), 32), (0, 0, 0), -1)
            cv2.putText(annotated, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            tracking.write(annotated)

            rect = crop_rows[index]["crop_xyxy"]
            x1, y1, x2, y2 = [int(round(v)) for v in rect]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, max(x1 + 2, x2)), min(height, max(y1 + 2, y2))
            crop = image[y1:y2, x1:x2]
            view = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(view, (0, 0), (min(width - 1, 560), 32), (0, 0, 0), -1)
            cv2.putText(
                view,
                f"{state} | {crop_rows[index]['crop_source']}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                1,
                cv2.LINE_AA,
            )
            centered.write(view)
            if print_every > 0 and (index + 1) % print_every == 0:
                print(f"[E2E PREVIEW] frames={index + 1}/{len(frames)}", flush=True)
    finally:
        capture.release()
        tracking.release()
        centered.release()


def write_contract_outputs(output_dir: Path, state: dict[str, Any], timeline: Mapping[str, Any]) -> None:
    offset = source_frame_offset(state)
    atomic_json(
        output_dir / "ambiguities.json",
        {
            "source_frame_offset": offset,
            "ambiguities": [
                externalize_ambiguity(item, offset)
                for item in state.get("ambiguities", [])
            ],
        },
    )
    atomic_json(
        output_dir / "confirmations.json",
        {
            "source_frame_offset": offset,
            "confirmations": [
                externalize_confirmation(item, offset)
                for item in state.get("confirmations", [])
            ],
        },
    )
    atomic_json(
        output_dir / "target_segments.json",
        {
            "source_frame_offset": offset,
            "segments": [
                externalize_segment(item, offset)
                for item in state.get("segments", [])
            ],
        },
    )
    write_csv(
        output_dir / "shot_boundaries.csv",
        [
            {
                "shot_index": shot["shot_index"],
                "shot_id": shot["shot_id"],
                "local_start_frame": shot["start_frame"],
                "local_end_frame_inclusive": shot["end_frame_inclusive"],
                "start_frame": offset + int(shot["start_frame"]),
                "end_frame_inclusive": offset + int(shot["end_frame_inclusive"]),
                "frame_count": shot["frame_count"],
                "status": shot.get("status"),
            }
            for shot in state["shots"]
        ],
    )
    counts = Counter(str(row["state"]) for row in timeline["frames"])
    bbox_count = sum(row.get("bbox_xyxy") is not None for row in timeline["frames"])
    summary = {
        "schema_version": "kickclip.target_centric_e2e_summary.v1",
        "generated_at": now_iso(),
        "status": state["status"],
        "decision": state.get("decision"),
        "test_name": state["test_name"],
        "counts": {
            "frame_count": len(timeline["frames"]),
            "bbox_frame_count": bbox_count,
            "bbox_coverage_percent": 100.0 * bbox_count / max(1, len(timeline["frames"])),
            "state_counts": dict(counts),
            "shot_count": len(state["shots"]),
            "ambiguity_count": len(state.get("ambiguities", [])),
            "confirmation_count": len(state.get("confirmations", [])),
            "accepted_segment_count": sum(item.get("status") == "ACCEPTED" for item in state.get("segments", [])),
        },
        "pending_action": state.get("pending_action"),
        "outputs": {
            "timeline_json": str(output_dir / "target_timeline.json"),
            "timeline_csv": str(output_dir / "target_timeline.csv"),
            "tracking_preview": str(output_dir / "full_frame_tracking_preview.mp4"),
            "centered_preview": str(output_dir / "target_centered_preview.mp4"),
        },
        "safety_invariants": {
            "silent_wrong_player_link_allowed": False,
            "uncertain_states_keep_null_bbox": True,
            "user_confirmation_recorded": True,
            "frozen_v1_modified": False,
            "frozen_v2_modified": False,
            "threshold_search_performed": False,
            "memory_updated_from_unreviewed_episode": False,
            "v7_runtime_dependency": False,
        },
    }
    atomic_json(output_dir / "pipeline_summary.json", summary)
    report = f"""# KickClip Target-Centric E2E Report

- Status: `{state['status']}`
- Decision: `{state.get('decision')}`
- Frames: `{len(timeline['frames'])}`
- BBox frames: `{bbox_count}` ({summary['counts']['bbox_coverage_percent']:.2f}%)
- Shots: `{len(state['shots'])}`
- Ambiguities: `{len(state.get('ambiguities', []))}`
- User confirmations: `{len(state.get('confirmations', []))}`

## Safety contract

- Camera cuts reset shot-local motion continuity.
- Frozen Stage 3-B2 safe-gate PASS is auto-reacquired; only AMBIGUOUS shots require user confirmation.
- `SEARCHING`, `AMBIGUOUS`, and `ABSENT` remain explicit with null target bbox.
- Frozen V1/V2 code and thresholds are not modified.
- `global_ID_tracking_upgrade_v7` is not a runtime dependency.
- Target memory is created from pre-cut confirmed tracking and is not updated from unreviewed episodes.
"""
    atomic_text(output_dir / "report.md", report)


def update_all_outputs(root: Path, output_dir: Path, state: dict[str, Any], render: bool, print_every: int) -> None:
    timeline = build_global_timeline(root, output_dir, state)
    write_contract_outputs(output_dir, state, timeline)
    if render:
        render_previews(Path(state["video"]["path"]), output_dir, timeline, print_every)


def create_ambiguity(
    output_dir: Path,
    state: dict[str, Any],
    shot: dict[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    ambiguity_id = f"ambiguity_{len(state.get('ambiguities', [])) + 1:04d}"
    ranked = list(result["ranked_candidates"])
    review = ranked[: min(ASSISTED_REVIEW_BATCH_SIZE, len(ranked))]
    ambiguity_dir = output_dir / "ambiguity_candidates" / ambiguity_id
    ambiguity_dir.mkdir(parents=True, exist_ok=False)
    source_sheet = Path(str(result.get("contact_sheet") or ""))
    target_sheet = ambiguity_dir / "candidates.jpg"
    if source_sheet.is_file():
        shutil.copy2(source_sheet, target_sheet)
    atomic_json(ambiguity_dir / "ranked_candidates.json", {"candidates": ranked})
    ambiguity = {
        "ambiguity_id": ambiguity_id,
        "shot_id": shot["shot_id"],
        "shot_index": shot["shot_index"],
        "start_frame": shot["start_frame"],
        "end_frame_inclusive": shot["end_frame_inclusive"],
        "status": "PENDING",
        "operational_state": result["gate"]["operational_state"],
        "decision": result["gate"]["decision"],
        "recommended_candidate": result["gate"].get("automatically_proposed_candidate"),
        "review_candidates": [dict(item) for item in review],
        "all_ranked_candidates": [dict(item) for item in ranked],
        "all_ranked_candidate_ids": [str(item["candidate_id"]) for item in ranked],
        "review_batch_size": ASSISTED_REVIEW_BATCH_SIZE,
        "contact_sheet": str(target_sheet),
        "assignments": str(result["assignments"]),
        "safe_gate": result["gate"],
        "created_at": now_iso(),
    }
    state.setdefault("ambiguities", []).append(ambiguity)
    state["pending_action"] = {
        "type": "CROSS_SHOT_CONFIRMATION",
        "ambiguity_id": ambiguity_id,
        "shot_id": shot["shot_id"],
        "contact_sheet": str(target_sheet),
        "candidate_ids": [str(item["candidate_id"]) for item in review],
        "recommended_candidate": ambiguity["recommended_candidate"],
    }
    state["status"] = "NEEDS_CONFIRMATION"
    state["decision"] = "PAUSE_AT_CROSS_SHOT_AMBIGUITY"
    shot["status"] = "AMBIGUOUS_REVIEW_REQUIRED"
    return ambiguity


def new_segment_from_confirmation(
    output_dir: Path,
    state: dict[str, Any],
    shot: Mapping[str, Any],
    ambiguity: Mapping[str, Any],
    candidate_id: str,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    segment_number = len(state.get("segments", [])) + 1
    selected = anchor["selected"]
    start = int(selected["frame_index"])
    end = int(shot["end_frame_inclusive"])
    safe = validate_name(state["test_name"])
    segment = {
        "segment_id": f"segment_{segment_number:04d}",
        "shot_id": shot["shot_id"],
        "source": "USER_CONFIRMED_CROSS_SHOT_CANDIDATE",
        "ambiguity_id": ambiguity["ambiguity_id"],
        "candidate_id": candidate_id,
        "global_start_frame": start,
        "global_end_frame_inclusive": end,
        "initial_bbox_xyxy": [
            float(selected["x1"]),
            float(selected["y1"]),
            float(selected["x2"]),
            float(selected["y2"]),
        ],
        "anchor": dict(selected),
        "anchor_scored_observations": anchor["scored"],
        "clip_path": str(output_dir / "work" / "clips" / f"segment_{segment_number:04d}.mp4"),
        "source_stage0_test_name": f"e2e_{safe}__seg{segment_number:04d}_stage0",
        "phase1_test_name": f"e2e_{safe}__seg{segment_number:04d}_phase1",
        "v2_adapter_test_name": f"e2e_{safe}__seg{segment_number:04d}_adapter",
        "phase1_reviews": {},
        "status": "PENDING_PHASE1",
        "created_at": now_iso(),
    }
    state.setdefault("segments", []).append(segment)
    return segment



def new_segment_from_auto_safe(
    output_dir: Path,
    state: dict[str, Any],
    shot: Mapping[str, Any],
    candidate_id: str,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Start frozen Phase-1 from a Stage-3B2 safe-gate reacquisition.

    No ambiguity or human confirmation record is created.  The candidate was
    accepted only because every frozen cross-shot safety gate passed with a
    real negative gallery available.
    """
    segment_number = len(state.get("segments", [])) + 1
    selected = anchor["selected"]
    start = int(selected["frame_index"])
    end = int(shot["end_frame_inclusive"])
    safe = validate_name(state["test_name"])
    segment = {
        "segment_id": f"segment_{segment_number:04d}",
        "shot_id": shot["shot_id"],
        "source": "AUTO_SAFE_CROSS_SHOT_CANDIDATE",
        "ambiguity_id": None,
        "candidate_id": candidate_id,
        "global_start_frame": start,
        "global_end_frame_inclusive": end,
        "initial_bbox_xyxy": [
            float(selected["x1"]),
            float(selected["y1"]),
            float(selected["x2"]),
            float(selected["y2"]),
        ],
        "anchor": dict(selected),
        "anchor_scored_observations": anchor["scored"],
        "clip_path": str(output_dir / "work" / "clips" / f"segment_{segment_number:04d}.mp4"),
        "source_stage0_test_name": f"e2e_{safe}__seg{segment_number:04d}_stage0",
        "phase1_test_name": f"e2e_{safe}__seg{segment_number:04d}_phase1",
        "v2_adapter_test_name": f"e2e_{safe}__seg{segment_number:04d}_adapter",
        "phase1_reviews": {},
        "status": "PENDING_PHASE1",
        "created_at": now_iso(),
    }
    state.setdefault("segments", []).append(segment)
    shot["status"] = "AUTO_SAFE_REACQUIRED_PENDING_TRACKING"
    return segment



def record_review_decision_provenance(
    state: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    artifact = arguments.review_decision_artifact
    if artifact is None:
        return
    path = artifact.expanduser().resolve()
    expected = str(arguments.review_decision_sha256 or "").strip().lower()
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise RuntimeError("Review decision artifact is missing or changed")
    state.setdefault("review_decision_artifacts", []).append(
        {
            "path": str(path),
            "sha256": expected,
            "recorded_at": now_iso(),
            "used_for_scoring": False,
        }
    )


def _resolve_safe_candidate_rejection(
    state: dict[str, Any],
    ambiguity: dict[str, Any],
    shot: dict[str, Any],
    *,
    rejected_ids: Sequence[str],
    reason: str,
    reviewer: str,
    note: str,
) -> bool:
    rejected = {str(value) for value in rejected_ids if str(value)}
    review_rows = [
        dict(item) for item in (ambiguity.get("review_candidates") or [])
    ]
    for row in review_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in rejected:
            row["review_status"] = reason
    ambiguity["review_candidates"] = review_rows
    ambiguity.setdefault("rejected_candidate_ids", [])
    for candidate_id in sorted(rejected):
        if candidate_id not in ambiguity["rejected_candidate_ids"]:
            ambiguity["rejected_candidate_ids"].append(candidate_id)

    pending_ids = [
        str(item.get("candidate_id") or "")
        for item in review_rows
        if str(item.get("candidate_id") or "")
        and str(item.get("candidate_id") or "") not in rejected
        and str(item.get("review_status") or "") not in {
            "USER_REJECTED",
            "UNREVIEWABLE",
            "NON_PLAYER_ROLE",
        }
    ]
    state.setdefault("confirmations", []).append(
        {
            "ambiguity_id": ambiguity["ambiguity_id"],
            "shot_id": shot["shot_id"],
            "decision": reason,
            "candidate_ids": sorted(rejected),
            "reviewer": reviewer,
            "note": note,
            "confirmed_at": now_iso(),
        }
    )
    if pending_ids:
        state["pending_action"]["candidate_ids"] = pending_ids
        state["status"] = "NEEDS_CONFIRMATION"
        state["decision"] = "PAUSE_AT_CROSS_SHOT_AMBIGUITY"
        return False

    # Product contract: one ambiguity exposes at most the initial top-3
    # tracklets. If the user rejects that set, do not page through the rest of
    # the ranked pool; retain SEARCHING/ABSENT semantics and continue.
    ambiguity["status"] = (
        "ALL_CANDIDATES_NON_PLAYER_ROLE"
        if reason == "NON_PLAYER_ROLE"
        else "ALL_CANDIDATES_REJECTED"
    )
    ambiguity["resolved_at"] = now_iso()
    shot["status"] = "SAFE_REJECTED_SEARCHING"
    state["search"]["next_shot_index"] = int(shot["shot_index"]) + 1
    state["pending_action"] = None
    state["status"] = "RUNNING"
    state["decision"] = "SAFE_REJECT_ALL_CANDIDATES_CONTINUE_SEARCH"
    return True


def handle_pending_action(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    arguments: argparse.Namespace,
) -> bool:
    """Handle one user response. Return True when processing may continue."""
    pending = state.get("pending_action")
    if not pending:
        return True
    record_review_decision_provenance(state, arguments)
    pending_type = str(pending.get("type"))

    if pending_type == "MEMORY_REVIEW":
        if arguments.approve_review == "MEMORY":
            prepare_memory_contract(root, output_dir, state)
            # Backend memory revisions are provenance-only. Frozen Stage 3-B1
            # scoring uses the reviewed Stage 3-A2 target/negative embeddings.
            if arguments.backend_memory_revision is not None:
                path = arguments.backend_memory_revision.expanduser().resolve()
                expected = str(arguments.backend_memory_sha256 or "").strip().lower()
                if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
                    raise RuntimeError("Backend memory provenance artifact is missing or changed")
                state["memory"]["backend_memory_provenance"] = {
                    "path": str(path),
                    "sha256": expected,
                    "used_for_scoring": False,
                }
            state["shots"][0]["status"] = "TARGET_CONFIRMED_AND_TRACKED"
            state["pending_action"] = None
            state["status"] = "RUNNING"
            state["decision"] = "MEMORY_REVIEW_PASS_CONTINUE_SEARCH"
            return True
        if arguments.reject_review == "MEMORY":
            state["status"] = "BLOCKED"
            state["decision"] = "BLOCK_MEMORY_VISUAL_REVIEW_FAIL"
            state["pending_action"] = None
            return False
        raise RuntimeError(
            "Memory review is pending. Resume with --approve-review MEMORY or --reject-review MEMORY."
        )

    if pending_type == "CROSS_SHOT_CONFIRMATION":
        ambiguity_id = str(pending["ambiguity_id"])
        if arguments.ambiguity_id != ambiguity_id:
            raise RuntimeError(f"Expected --ambiguity-id {ambiguity_id}")
        ambiguity = next(
            item for item in state["ambiguities"] if item["ambiguity_id"] == ambiguity_id
        )
        shot = next(item for item in state["shots"] if item["shot_id"] == ambiguity["shot_id"])
        if arguments.confirm_absent:
            ambiguity["status"] = "CONFIRMED_ABSENT"
            ambiguity["resolved_at"] = now_iso()
            shot["status"] = "ABSENT_CONFIRMED"
            state.setdefault("confirmations", []).append(
                {
                    "ambiguity_id": ambiguity_id,
                    "shot_id": shot["shot_id"],
                    "decision": "TARGET_ABSENT",
                    "reviewer": arguments.reviewer,
                    "note": arguments.review_note,
                    "confirmed_at": now_iso(),
                }
            )
            state["search"]["next_shot_index"] = int(shot["shot_index"]) + 1
            state["pending_action"] = None
            state["status"] = "RUNNING"
            state["decision"] = "USER_CONFIRMED_ABSENT_CONTINUE_SEARCH"
            return True

        if arguments.reject_all_candidates or arguments.reject_all_candidates_as_non_player_role:
            ids = list(pending.get("candidate_ids") or [])
            return _resolve_safe_candidate_rejection(
                state,
                ambiguity,
                shot,
                rejected_ids=ids,
                reason=(
                    "NON_PLAYER_ROLE"
                    if arguments.reject_all_candidates_as_non_player_role
                    else "USER_REJECTED"
                ),
                reviewer=arguments.reviewer,
                note=arguments.review_note,
            )

        single_rejected = arguments.rejected_candidate or arguments.unreviewable_candidate
        if single_rejected:
            allowed_now = set(str(value) for value in (pending.get("candidate_ids") or []))
            if single_rejected not in allowed_now:
                raise RuntimeError(
                    f"Candidate is not pending for ambiguity {ambiguity_id}: {single_rejected}"
                )
            return _resolve_safe_candidate_rejection(
                state,
                ambiguity,
                shot,
                rejected_ids=[single_rejected],
                reason=(
                    "UNREVIEWABLE"
                    if arguments.unreviewable_candidate
                    else "USER_REJECTED"
                ),
                reviewer=arguments.reviewer,
                note=arguments.review_note,
            )

        candidate_id = arguments.confirmed_candidate
        if not candidate_id:
            raise RuntimeError(
                "Provide a candidate confirmation, target-absent decision, or safe rejection."
            )
        allowed = set(str(value) for value in (pending.get("candidate_ids") or []))
        if candidate_id not in allowed:
            raise RuntimeError(f"Candidate is not one of the presented review candidates for ambiguity {ambiguity_id}: {candidate_id}")
        anchor = candidate_anchor(root, ambiguity, candidate_id)
        ambiguity["status"] = "CANDIDATE_CONFIRMED_PENDING_SEGMENT_REVIEW"
        ambiguity["confirmed_candidate"] = candidate_id
        ambiguity["resolved_at"] = now_iso()
        segment = new_segment_from_confirmation(
            output_dir, state, shot, ambiguity, candidate_id, anchor
        )
        state.setdefault("confirmations", []).append(
            {
                "ambiguity_id": ambiguity_id,
                "shot_id": shot["shot_id"],
                "decision": "CANDIDATE_CONFIRMED",
                "candidate_id": candidate_id,
                "anchor_frame": int(anchor["selected"]["frame_index"]),
                "anchor_bbox_xyxy": segment["initial_bbox_xyxy"],
                "reviewer": arguments.reviewer,
                "note": arguments.review_note,
                "confirmed_at": now_iso(),
            }
        )
        state["pending_action"] = None
        state["status"] = "RUNNING"
        return True

    if pending_type in {"SEGMENT_VISUAL_REVIEW", "PHASE1_INTERNAL_REVIEW"}:
        segment_id = str(pending["segment_id"])
        segment = next(item for item in state["segments"] if item["segment_id"] == segment_id)
        review_stage = str(pending["review_stage"])
        if arguments.reject_review == review_stage:
            segment["status"] = "REJECTED_VISUAL_REVIEW"
            state["status"] = "BLOCKED"
            state["decision"] = f"BLOCK_{review_stage}_VISUAL_REVIEW_FAIL"
            state["pending_action"] = None
            return False
        if arguments.approve_review != review_stage:
            raise RuntimeError(
                f"Review {review_stage} is pending. Resume with --approve-review {review_stage} or --reject-review {review_stage}."
            )
        if review_stage == "SEGMENT":
            segment["status"] = "ACCEPTED"
            segment["timeline"] = str(pending["timeline"])
            segment["accepted_at"] = now_iso()
            shot = next(item for item in state["shots"] if item["shot_id"] == segment["shot_id"])
            shot["status"] = "TARGET_CONFIRMED_AND_TRACKED"
            state["search"]["next_shot_index"] = int(shot["shot_index"]) + 1
            state["pending_action"] = None
            state["status"] = "RUNNING"
            state["decision"] = "SEGMENT_VISUAL_REVIEW_PASS_CONTINUE_SEARCH"
            return True
        segment.setdefault("phase1_reviews", {})[review_stage] = "PASS"
        state["pending_action"] = None
        state["status"] = "RUNNING"
        return True

    raise RuntimeError(f"Unknown pending action: {pending_type}")


def continue_pending_segment(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    device: str,
) -> bool:
    pending_segments = [item for item in state.get("segments", []) if item.get("status") == "PENDING_PHASE1"]
    if not pending_segments:
        return True
    segment = pending_segments[-1]
    result = run_segment_phase1(root, output_dir, state, segment, device)
    segment["phase1_result"] = result
    if result["status"] == "REVIEW_REQUIRED":
        review_stage = str(result["review_stage"])
        action_type = "SEGMENT_VISUAL_REVIEW" if review_stage == "SEGMENT" else "PHASE1_INTERNAL_REVIEW"
        state["pending_action"] = {
            "type": action_type,
            "segment_id": segment["segment_id"],
            "review_stage": review_stage,
            "preview": result.get("preview"),
            "centered_preview": result.get("centered_preview"),
            "timeline": result.get("timeline"),
        }
        state["status"] = "NEEDS_CONFIRMATION"
        state["decision"] = f"PAUSE_FOR_{review_stage}_VISUAL_REVIEW"
        return False
    segment["status"] = "ACCEPTED"
    segment["timeline"] = str(result["timeline"])
    segment["accepted_at"] = now_iso()
    shot = next(item for item in state["shots"] if item["shot_id"] == segment["shot_id"])
    shot["status"] = "TARGET_CONFIRMED_AND_TRACKED"
    state["search"]["next_shot_index"] = int(shot["shot_index"]) + 1
    return True


def search_remaining_shots(
    root: Path,
    output_dir: Path,
    state: dict[str, Any],
    device: str,
) -> None:
    runtime = load_runtime(root, state, device)
    next_index = int(state["search"].get("next_shot_index", 1))
    while next_index < len(state["shots"]):
        shot = state["shots"][next_index]
        if shot.get("status") in {"ABSENT_CONFIRMED", "TARGET_CONFIRMED_AND_TRACKED"}:
            next_index += 1
            state["search"]["next_shot_index"] = next_index
            continue
        result = process_shot_candidates(root, output_dir, state, runtime, shot)
        state.setdefault("shot_search_results", {})[shot["shot_id"]] = result
        scene_policy = result.get("tracking_scene_policy") or {}
        if scene_policy and not bool(scene_policy.get("tracking_allowed", True)):
            shot["status"] = "EXCLUDED_NON_WIDE_TRACKING"
            shot["tracking_scene_mode"] = str(scene_policy.get("mode") or "MIXED")
            shot["tracking_scene_policy"] = str(
                scene_policy.get("policy_version") or PRODUCT_TRACKING_SCENE_POLICY_VERSION
            )
            next_index += 1
            state["search"]["next_shot_index"] = next_index
            state["decision"] = "NON_WIDE_SHOT_SKIPPED_CONTINUE_SEARCH"
            save_state(output_dir, state)
            continue
        operational = result["gate"]["operational_state"]
        if operational == "SAFE_REJECTED_SEARCHING":
            shot["status"] = "SAFE_REJECTED_SEARCHING"
            next_index += 1
            state["search"]["next_shot_index"] = next_index
            save_state(output_dir, state)
            continue
        if (
            operational == "AUTO_REACQUIRED_CANDIDATE"
            and state["reacquisition_mode"] in {"assisted", "auto-safe"}
            and not bool((state.get("memory") or {}).get("reference_only_negative_memory"))
        ):
            candidate_id = str(result["gate"]["automatically_proposed_candidate"])
            # AUTO-safe links are not ambiguities and are never shown as user
            # review cards. Reuse only the immutable assignments needed to pick
            # the Stage-3B3 anchor, then resume frozen Phase-1 within the shot.
            anchor = candidate_anchor(
                root,
                {"assignments": str(result["assignments"])},
                candidate_id,
            )
            segment = new_segment_from_auto_safe(
                output_dir, state, shot, candidate_id, anchor
            )
            segment["safe_gate"] = dict(result["gate"])
            state["pending_action"] = None
            state["status"] = "RUNNING"
            state["decision"] = "AUTO_SAFE_REACQUIRED_CONTINUE_SAME_SHOT_TRACKING"
            save_state(output_dir, state)
            if not continue_pending_segment(root, output_dir, state, device):
                return
            next_index = int(state["search"]["next_shot_index"])
            continue
        create_ambiguity(output_dir, state, shot, result)
        return
    state["status"] = "COMPLETE"
    state["decision"] = "E2E_TARGET_TIMELINE_COMPLETE"
    state["pending_action"] = None


def create_manifest(
    root: Path,
    output_dir: Path,
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    used_files: list[dict[str, Any]] = []
    for logical, record in (manifest.get("scripts") or {}).items():
        raw = record.get("path")
        if not raw:
            continue
        path = resolve(root, normalized_relative_path(str(raw)))
        used_files.append({"logical": f"phase1:{logical}", "path": str(path), "sha256": sha256_file(path)})
    for path in sorted((root / "target_centric_tracking_v2").glob("stage3*.py")):
        used_files.append({"logical": f"phase2:{path.name}", "path": str(path), "sha256": sha256_file(path)})
    payload = {
        "schema_version": "kickclip.target_centric_e2e_manifest.v1",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": now_iso(),
        "test_name": state["test_name"],
        "video": state["video"],
        "source_video": state.get("source_video", state["video"]),
        "source_frame_offset": int(state.get("source_frame_offset") or 0),
        "tracking_direction": state.get("tracking_direction", "FORWARD_FROM_CONFIRMED_ANCHOR"),
        "initial_bbox_xyxy": state["initial_bbox_xyxy"],
        "reacquisition_mode": state["reacquisition_mode"],
        "tracking_play_conf_threshold": float(state.get("tracking_play_conf_threshold", 0.15)),
        "selected_reference_memory": {
            "trusted": bool(state.get("trusted_selected_reference_memory")),
            "contract": state.get("target_reference_set"),
            "used_for_scoring": bool((state.get("memory") or {}).get("backend_memory_used_for_scoring", False)),
        },
        "frozen_files": used_files,
        "policies": {"stage3b0": B0_POLICY, "stage3b1": B1_POLICY, "stage3b2": B2_POLICY},
        "safety_contract": {
            "silent_wrong_player_switch": "FORBIDDEN",
            "assisted_mode_requires_confirmation": False,
            "ambiguous_cross_shot_requires_confirmation": True,
            "auto_safe_requires_real_negative_memory": True,
            "maximum_review_candidates_per_ambiguity": ASSISTED_REVIEW_BATCH_SIZE,
            "threshold_search": False,
            "frozen_source_modification": False,
            "v7_runtime_dependency": False,
        },
    }
    atomic_json(output_dir / "pipeline_manifest.json", payload)


def print_status(output_dir: Path, state: Mapping[str, Any]) -> None:
    print("KickClip Target-Centric E2E pipeline")
    print(f"Status               : {state['status']}")
    print(f"Decision             : {state.get('decision')}")
    print(f"Test                 : {state['test_name']}")
    print(f"Shots                : {len(state.get('shots', []))}")
    print(f"Confirmed segments   : {sum(item.get('status') == 'ACCEPTED' for item in state.get('segments', []))}")
    pending = state.get("pending_action")
    if pending:
        print(f"Pending action       : {pending.get('type')}")
        if pending.get("ambiguity_id"):
            print(f"Ambiguity            : {pending.get('ambiguity_id')}")
        if pending.get("candidate_ids"):
            print(f"Candidates           : {', '.join(pending.get('candidate_ids'))}")
        if pending.get("contact_sheet"):
            print(f"Contact sheet        : {pending.get('contact_sheet')}")
        if pending.get("review_stage"):
            print(f"Review stage         : {pending.get('review_stage')}")
        if pending.get("preview"):
            print(f"Preview              : {pending.get('preview')}")
    print(f"Timeline             : {output_dir / 'target_timeline.json'}")
    print(f"Output               : {output_dir}")


def main() -> int:
    arguments = parse_args()
    started = time.perf_counter()
    root = arguments.project_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if arguments.backend_safe_weights_runner is not None:
        runner = arguments.backend_safe_weights_runner.expanduser().resolve()
        if not runner.is_file():
            raise FileNotFoundError(runner)
        compatibility = load_module("kickclip_backend_safe_weights", runner)
        compatibility.register_safe_globals()
    test_name = validate_name(arguments.test_name)
    output_root = resolve(root, arguments.output_root)
    output_dir = output_root / test_name

    if arguments.resume:
        state = load_state(output_dir)
        if state["test_name"] != test_name:
            raise RuntimeError("State test name mismatch")
        if not handle_pending_action(root, output_dir, state, arguments):
            save_state(output_dir, state)
            update_all_outputs(root, output_dir, state, False, arguments.print_every)
            print_status(output_dir, state)
            return 2
        save_state(output_dir, state)
        if not continue_pending_segment(root, output_dir, state, arguments.device):
            save_state(output_dir, state)
            update_all_outputs(root, output_dir, state, False, arguments.print_every)
            print_status(output_dir, state)
            return 3
        search_remaining_shots(root, output_dir, state, arguments.device)
        state["runtime_seconds_accumulated"] = float(state.get("runtime_seconds_accumulated", 0.0)) + (time.perf_counter() - started)
        save_state(output_dir, state)
        render = state["status"] == "COMPLETE" and not arguments.no_preview
        update_all_outputs(root, output_dir, state, render, arguments.print_every)
        print_status(output_dir, state)
        return 0 if state["status"] == "COMPLETE" else 3

    if arguments.video is None or arguments.initial_bbox is None:
        raise ValueError("New run requires --video and --initial-bbox")
    if not 0.0 < float(arguments.tracking_play_conf_threshold) <= 1.0:
        raise ValueError("--tracking-play-conf-threshold must be in (0, 1]")
    if arguments.trusted_selected_reference_memory and arguments.target_reference_set is None:
        raise ValueError(
            "--trusted-selected-reference-memory requires --target-reference-set"
        )
    if arguments.target_reference_set is not None:
        reference_path = resolve(root, arguments.target_reference_set)
        expected_reference_sha = str(arguments.target_reference_set_sha256 or "").strip().lower()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        if len(expected_reference_sha) != 64 or sha256_file(reference_path).lower() != expected_reference_sha:
            raise RuntimeError("Target reference set is missing or changed")
    initialize_output_dir(
        output_dir,
        arguments.overwrite,
        allow_prestaged_output=arguments.allow_prestaged_output,
    )
    source_video_path = resolve(root, arguments.video)
    if not source_video_path.is_file():
        raise FileNotFoundError(source_video_path)
    source_metadata = video_metadata(source_video_path)
    initial_frame = int(arguments.initial_frame or 0)
    if initial_frame < 0 or initial_frame >= int(source_metadata["frame_count"]):
        raise ValueError(
            f"--initial-frame must be within source video: {initial_frame} / "
            f"{source_metadata['frame_count']}"
        )
    bbox = clip_bbox(
        arguments.initial_bbox,
        int(source_metadata["width"]),
        int(source_metadata["height"]),
    )

    if initial_frame > 0:
        effective_video_path = output_dir / "work" / "input" / f"anchor_forward_f{initial_frame:09d}.mp4"
        extract_clip_cv2(
            source_video_path,
            effective_video_path,
            initial_frame,
            int(source_metadata["frame_count"]) - 1,
            source_metadata,
        )
    else:
        effective_video_path = source_video_path
    metadata = video_metadata(effective_video_path)

    manifest_path = resolve(root, arguments.phase1_manifest)
    manifest = verify_phase1_manifest(root, manifest_path)
    runtime_manifest_path = materialize_portable_phase1_manifest(
        root,
        output_dir,
        manifest_path,
        manifest,
    )

    a0 = run_cut_audit(
        root,
        output_dir,
        test_name,
        effective_video_path,
        bbox,
        runtime_manifest_path,
    )
    explicit_local_cuts = None
    if arguments.cut_frames:
        explicit_local_cuts = [
            int(value) - initial_frame
            for value in arguments.cut_frames
            if int(value) > initial_frame
        ]
    cut_frames = choose_cut_frames(
        a0,
        explicit_local_cuts,
        int(metadata["frame_count"]),
    )
    shots = make_shots(int(metadata["frame_count"]), cut_frames)
    safe = validate_name(test_name)
    state: dict[str, Any] = {
        "schema_version": STATE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "RUNNING",
        "decision": "INITIALIZED",
        "test_name": test_name,
        "reviewer": arguments.reviewer,
        "review_note": arguments.review_note,
        "video": metadata,
        "source_video": source_metadata,
        "source_frame_offset": initial_frame,
        "tracking_direction": "FORWARD_FROM_CONFIRMED_ANCHOR",
        "initial_bbox_xyxy": bbox,
        "phase1_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "runtime_portable_path": str(runtime_manifest_path),
            "runtime_portable_sha256": sha256_file(runtime_manifest_path),
            "runtime_portability_transform": "PATH_SEPARATOR_ONLY",
            "source_manifest_modified": False,
        },
        "reacquisition_mode": arguments.reacquisition_mode,
        "tracking_play_conf_threshold": float(arguments.tracking_play_conf_threshold),
        "trusted_selected_reference_memory": bool(arguments.trusted_selected_reference_memory),
        "target_reference_set": (
            {
                "path": str(resolve(root, arguments.target_reference_set)),
                "sha256": str(arguments.target_reference_set_sha256 or "").strip().lower(),
            }
            if arguments.target_reference_set is not None
            else None
        ),
        "cut_frames": cut_frames,
        "shots": shots,
        "memory": None,
        "search": {"next_shot_index": 1},
        "segments": [],
        "ambiguities": [],
        "confirmations": [],
        "shot_search_results": {},
        "pending_action": None,
        "work": {
            "v2_memory_test_name": f"e2e_{safe}__memory",
            "phase1_source_test_name": f"e2e_{safe}__precut_source",
        },
        "artifacts": {"stage3a0_summary_payload": a0},
        "runtime_seconds_accumulated": 0.0,
    }
    create_manifest(root, output_dir, state, manifest)

    if not cut_frames:
        # No-cut case: use the frozen Phase-1 runner directly as one segment.
        segment = {
            "segment_id": "segment_0001",
            "shot_id": "shot_0000",
            "source": "INITIAL_USER_SELECTION_NO_CUT",
            "global_start_frame": 0,
            "global_end_frame_inclusive": int(metadata["frame_count"]) - 1,
            "initial_bbox_xyxy": bbox,
            "clip_path": str(effective_video_path),
            "source_stage0_test_name": f"e2e_{safe}__nocut_stage0",
            "phase1_test_name": f"e2e_{safe}__nocut_phase1",
            "v2_adapter_test_name": f"e2e_{safe}__nocut_adapter",
            "phase1_reviews": {},
            "status": "PENDING_PHASE1",
            "created_at": now_iso(),
        }
        state["segments"].append(segment)
        save_state(output_dir, state)
        if not continue_pending_segment(root, output_dir, state, arguments.device):
            save_state(output_dir, state)
            update_all_outputs(root, output_dir, state, False, arguments.print_every)
            print_status(output_dir, state)
            return 3
        state["status"] = "COMPLETE"
        state["decision"] = "E2E_NO_CUT_TIMELINE_COMPLETE"
        save_state(output_dir, state)
        update_all_outputs(root, output_dir, state, not arguments.no_preview, arguments.print_every)
        print_status(output_dir, state)
        return 0

    if state.get("trusted_selected_reference_memory"):
        prepare_trusted_selected_reference_memory(
            root,
            output_dir,
            state,
            runtime_manifest_path,
            arguments.device,
        )
        create_manifest(root, output_dir, state, manifest)
        state["shots"][0]["status"] = "INITIAL_TARGET_TRACKING_FROM_USER_REFERENCE"
        state["search"]["next_shot_index"] = 1
        state["decision"] = "USER_SELECTED_REFERENCE_MEMORY_READY"
        state["pending_action"] = None
        save_state(output_dir, state)
        search_remaining_shots(root, output_dir, state, arguments.device)
        state["runtime_seconds_accumulated"] = time.perf_counter() - started
        save_state(output_dir, state)
        render = state["status"] == "COMPLETE" and not arguments.no_preview
        update_all_outputs(root, output_dir, state, render, arguments.print_every)
        print_status(output_dir, state)
        return 0 if state["status"] == "COMPLETE" else 3

    success, error = run_initial_memory_stage(
        root,
        output_dir,
        state,
        runtime_manifest_path,
        arguments.device,
        arguments.backend_safe_weights_runner,
    )
    if not success:
        state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
        state["decision"] = "SAFE_BLOCK_NO_STABLE_PRECUT_TARGET_MEMORY"
        state["memory_error"] = error
        state["shots"][0]["status"] = "INITIAL_TARGET_TRACKING_ONLY"
        for shot in state["shots"][1:]:
            shot["status"] = "SEARCHING_NO_MEMORY"
        # The source Stage-2 timeline may still exist and is used only before the first cut.
        source_dir = resolve(root, DEFAULT_V1_OUTPUT_ROOT) / state["work"]["phase1_source_test_name"]
        timeline_candidates = [source_dir / "target_timeline.json", source_dir / "stage2_target_timeline.json"]
        source_timeline = next((path for path in timeline_candidates if path.is_file()), None)
        if source_timeline:
            state["memory"] = {"source_timeline": str(source_timeline)}
        save_state(output_dir, state)
        update_all_outputs(root, output_dir, state, not arguments.no_preview, arguments.print_every)
        print_status(output_dir, state)
        return 0

    paths = memory_paths(root, state)
    state["pending_action"] = {
        "type": "MEMORY_REVIEW",
        "review_stage": "MEMORY",
        "contact_sheet": str(paths["contact_sheet"]),
        "preview": str(paths["tracking_preview"]),
    }
    state["status"] = "NEEDS_CONFIRMATION"
    state["decision"] = "PAUSE_FOR_PRECUT_TARGET_MEMORY_VISUAL_REVIEW"
    state["shots"][0]["status"] = "INITIAL_TARGET_TRACKING_MEMORY_REVIEW_REQUIRED"
    state["runtime_seconds_accumulated"] = time.perf_counter() - started
    save_state(output_dir, state)
    # Preserve a partial timeline immediately.
    temporary_memory = {
        "source_timeline": str(paths["source_timeline"]),
    }
    state["memory"] = temporary_memory
    save_state(output_dir, state)
    update_all_outputs(root, output_dir, state, False, arguments.print_every)
    print_status(output_dir, state)
    return 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"E2E fatal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
