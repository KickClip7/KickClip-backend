#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "kickclip.r1_v1_v2_adapter_state.v1"
PIPELINE_VERSION = "EVENT_CANDIDATE_HANDOFF_R1_REAL_V1_V2_ADAPTER"
SELECTION_VIEW_SCHEMA = "kickclip.r1_selection_anchor_view.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate immutable R1 selection artifacts and invoke the supplied "
            "target-centric V1/V2 research runtime without inventing production_r3."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--initial-bbox", nargs=4, type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--reacquisition-mode", default="assisted")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tracking-launch-manifest", type=Path, required=True)
    parser.add_argument("--shot-boundaries", type=Path, required=True)
    parser.add_argument("--target-selection", type=Path, required=True)
    parser.add_argument("--target-reference-set", type=Path, required=True)
    parser.add_argument("--earlier-anchor-decision", type=Path, required=True)
    parser.add_argument("--target-memory-revision", type=Path, default=None)
    parser.add_argument("--target-memory-sha256", default=None)
    parser.add_argument("--candidate-scoring-generation", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ambiguity-id", default=None)
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--confirmed-candidate", default=None)
    decision.add_argument("--confirm-absent", action="store_true")
    decision.add_argument("--rejected-candidate", default=None)
    decision.add_argument("--unreviewable-candidate", default=None)
    review = parser.add_mutually_exclusive_group()
    review.add_argument(
        "--approve-review",
        choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"),
        default=None,
    )
    review.add_argument(
        "--reject-review",
        choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"),
        default=None,
    )
    parser.add_argument("--reviewer", default="USER")
    parser.add_argument("--review-note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--contract-test-skip-phase1-compatibility",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def required_research_files(root: Path) -> dict[str, Path]:
    return {
        "e2e_runner": root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        "e2e_verifier": root / "target_centric_tracking_e2e_v1" / "verify_e2e_installation.py",
        "phase1_runner": root / "target_centric_tracking_v1" / "run_phase1_frozen_pipeline.py",
        "phase1_manifest": root / "target_centric_tracking_v1" / "phase1_frozen_manifest.json",
        "phase1_stage0": root / "target_centric_tracking_v1" / "stage0_audit_inputs.py",
        "same_shot_stage1": root / "target_centric_tracking_v1" / "stage1_generate_rfdetr_detections.py",
        "same_shot_stage2": root / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
        "short_clip_compatibility": root / "target_centric_tracking_v2" / "stage3c0_run_short_clip_phase1_compatibility.py",
        "cross_shot_b0": root / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        "cross_shot_b1": root / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        "cross_shot_b2": root / "target_centric_tracking_v2" / "stage3b2_make_safe_cross_shot_decision.py",
        "cross_shot_b3": root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        "v6_reid_helper": root / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    }


def validate_input_file(path: Path, expected_sha: str | None, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return digest


def dependency_report(root: Path) -> dict[str, Any]:
    files = required_research_files(root)
    phase1_manifest_path = files["phase1_manifest"]
    models: list[dict[str, Any]] = []
    if phase1_manifest_path.is_file():
        manifest = read_object(phase1_manifest_path)
        for logical, record in (manifest.get("models") or {}).items():
            if not isinstance(record, Mapping) or not record.get("verified"):
                continue
            model_path = root / str(record.get("path") or "").replace("\\", "/")
            models.append(
                {
                    "logical": logical,
                    "path": str(model_path),
                    "expected_sha256": record.get("sha256"),
                    "exists": model_path.is_file(),
                    "sha256_matches": (
                        model_path.is_file()
                        and sha256_file(model_path) == str(record.get("sha256") or "")
                    ),
                }
            )
    missing = [str(path) for path in files.values() if not path.is_file()]
    missing.extend(
        row["path"]
        for row in models
        if not row["exists"] or not row["sha256_matches"]
    )
    return {
        "files": {name: str(path) for name, path in files.items()},
        "models": models,
        "missing": sorted(set(missing)),
        "global_ID_tracking_upgrade_v7_is_not_aliased_to_v6": True,
    }


def write_blocked_state(
    output_dir: Path,
    *,
    decision: str,
    failure_code: str,
    message: str,
    launch: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    extra_runtime: Mapping[str, Any] | None = None,
) -> None:
    state = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "updated_at": now_iso(),
        "status": "COMPLETE_WITH_SAFE_BLOCK",
        "decision": decision,
        "failure_code": failure_code,
        "message": message,
        "execution_kind": "EVENT_CANDIDATE_HANDOFF_R1",
        "pending_action": None,
        "shots": [],
        "ambiguities": [],
        "confirmations": [],
        "runtime": {
            "integration_path": "B.ADD_THIN_BACKEND_ADAPTER_TO_V1_V2_STAGES",
            "provided_e2e_runner_used": False,
            "selection_aware_video_adapter_used": False,
            "synthetic_tracking_used": False,
            "observation_copy_used_as_success": False,
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
            "launch_manifest_sha256": sha256_file(Path(str(launch["path"]))),
            "dependency_report": dependencies,
            **dict(extra_runtime or {}),
        },
    }
    atomic_json(output_dir / "pipeline_state.json", state)
    atomic_json(
        output_dir / "pipeline_summary.json",
        {
            "status": state["status"],
            "decision": decision,
            "failure_code": failure_code,
            "active_or_reacquired_bbox_frames": 0,
            "unresolved_shot_count": None,
            "preview_generated": False,
            "full_event_recommendation_e2e": "NOT_RUN",
        },
    )


def invoke(command: Sequence[str], cwd: Path) -> int:
    completed = subprocess.run(list(command), cwd=str(cwd), shell=False, check=False)
    return int(completed.returncode)


def _safe_name(value: object) -> str:
    text = str(value or "unknown")
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in text)


def _video_metadata(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {path}")
    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    if (
        metadata["width"] < 1
        or metadata["height"] < 1
        or metadata["fps"] <= 0
        or metadata["frame_count"] < 1
    ):
        raise RuntimeError(f"Invalid source video metadata: {path}")
    metadata["duration_seconds"] = metadata["frame_count"] / metadata["fps"]
    return metadata


def _write_video_range(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    end_frame_inclusive: int,
    metadata: Mapping[str, Any],
) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(metadata["fps"]),
        (int(metadata["width"]), int(metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video writer: {temporary}")
    count = 0
    try:
        for _ in range(start_frame, end_frame_inclusive + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended while materializing: {output}")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    expected = end_frame_inclusive - start_frame + 1
    if count != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Materialized video frame mismatch: {count}/{expected}")
    temporary.replace(output)


def _read_shots(boundaries: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = boundaries.get("shots") or boundaries.get("boundaries") or []
    shots: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            continue
        start = int(raw.get("start_frame", -1))
        end = int(raw.get("end_frame_inclusive", raw.get("end_frame", -1)))
        if start < 0 or end < start:
            raise ValueError("Reviewed shot boundary is invalid.")
        shots.append(
            {
                **dict(raw),
                "shot_id": str(raw.get("shot_id") or f"reviewed_shot_{index:04d}"),
                "start_frame": start,
                "end_frame_inclusive": end,
            }
        )
    shots.sort(key=lambda row: (int(row["start_frame"]), int(row["end_frame_inclusive"])))
    if not shots:
        raise ValueError("Reviewed shot boundaries are empty.")
    return shots


def materialize_selection_view(
    *,
    output_dir: Path,
    launch: Mapping[str, Any],
    target_selection: Mapping[str, Any],
    boundaries: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    source_video = Path(str((launch.get("source_video") or {}).get("path") or "")).resolve()
    validate_input_file(
        source_video,
        str((launch.get("source_video") or {}).get("sha256") or ""),
        "source video",
    )
    metadata = _video_metadata(source_video)
    anchor_frame = int(target_selection["best_anchor_frame"])
    selected_shot_id = str(target_selection["shot_id"])
    shots = _read_shots(boundaries)
    selected_index = next(
        (index for index, row in enumerate(shots) if row["shot_id"] == selected_shot_id),
        None,
    )
    if selected_index is None:
        raise ValueError("Selected shot is absent from reviewed boundaries.")
    selected = shots[selected_index]
    if not int(selected["start_frame"]) <= anchor_frame <= int(selected["end_frame_inclusive"]):
        raise ValueError("Selected anchor is outside its reviewed shot.")

    workspace = output_dir / "_selection_anchor_view"
    workspace.mkdir(parents=True, exist_ok=True)
    suffix_video = workspace / "selection_anchor_suffix.mp4"
    same_shot_video = workspace / "selected_shot_pre_cut.mp4"
    view_path = workspace / "selection_view.json"
    expected_view = {
        "source_video_sha256": metadata["sha256"],
        "source_offset_frame": anchor_frame,
        "source_frame_count": metadata["frame_count"],
        "selected_shot_id": selected_shot_id,
    }
    reusable = False
    if view_path.is_file() and suffix_video.is_file() and not overwrite:
        observed = read_object(view_path)
        reusable = all(observed.get(key) == value for key, value in expected_view.items())
    if not reusable:
        _write_video_range(
            source_video,
            suffix_video,
            start_frame=anchor_frame,
            end_frame_inclusive=int(metadata["frame_count"]) - 1,
            metadata=metadata,
        )
        _write_video_range(
            source_video,
            same_shot_video,
            start_frame=anchor_frame,
            end_frame_inclusive=int(selected["end_frame_inclusive"]),
            metadata=metadata,
        )
    elif not same_shot_video.is_file():
        _write_video_range(
            source_video,
            same_shot_video,
            start_frame=anchor_frame,
            end_frame_inclusive=int(selected["end_frame_inclusive"]),
            metadata=metadata,
        )

    future = shots[selected_index:]
    shot_map: list[dict[str, Any]] = []
    for raw_index, shot in enumerate(future):
        local_start = 0 if raw_index == 0 else int(shot["start_frame"]) - anchor_frame
        local_end = int(shot["end_frame_inclusive"]) - anchor_frame
        shot_map.append(
            {
                "raw_shot_id": f"shot_{raw_index:04d}",
                "raw_shot_index": raw_index,
                "source_shot_id": shot["shot_id"],
                "source_start_frame": int(shot["start_frame"]),
                "source_end_frame_inclusive": int(shot["end_frame_inclusive"]),
                "processed_source_start_frame": max(anchor_frame, int(shot["start_frame"])),
                "local_start_frame": local_start,
                "local_end_frame_inclusive": local_end,
            }
        )
    translated_cuts = [
        int(row["local_start_frame"])
        for row in shot_map[1:]
        if int(row["local_start_frame"]) > 0
    ]
    view = {
        "schema_version": SELECTION_VIEW_SCHEMA,
        "created_at": now_iso(),
        "source_video": metadata,
        "source_video_sha256": metadata["sha256"],
        "source_offset_frame": anchor_frame,
        "source_frame_count": metadata["frame_count"],
        "selected_shot_id": selected_shot_id,
        "selected_shot_start_frame": int(selected["start_frame"]),
        "selected_shot_end_frame_inclusive": int(selected["end_frame_inclusive"]),
        "unresolved_prefix_frame_count": anchor_frame - int(selected["start_frame"]),
        "selection_view_video": {
            "path": str(suffix_video),
            "sha256": sha256_file(suffix_video),
            **{key: value for key, value in _video_metadata(suffix_video).items() if key not in {"path", "sha256"}},
        },
        "same_shot_pre_cut_source": {
            "path": str(same_shot_video),
            "sha256": sha256_file(same_shot_video),
            "source_start_frame": anchor_frame,
            "source_end_frame_inclusive": int(selected["end_frame_inclusive"]),
            **{
                key: value
                for key, value in _video_metadata(same_shot_video).items()
                if key not in {"path", "sha256"}
            },
        },
        "cross_shot_search_source": {
            "path": str(suffix_video),
            "sha256": sha256_file(suffix_video),
            "source_start_frame": int(selected["end_frame_inclusive"]) + 1,
            "reviewed_shot_count": max(0, len(future) - 1),
        },
        "translated_cut_frames": translated_cuts,
        "shot_map": shot_map,
        "frame_zero_fallback_used": False,
        "actual_anchor_materialized_as_view_frame_zero": True,
    }
    atomic_json(view_path, view)
    return view


def _shot_map_by_raw(view: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["raw_shot_id"]): dict(row)
        for row in view.get("shot_map") or []
        if isinstance(row, Mapping)
    }


def _shot_for_global_frame(view: Mapping[str, Any], frame_index: int) -> str:
    for row in view.get("shot_map") or []:
        if not isinstance(row, Mapping):
            continue
        if int(row["source_start_frame"]) <= frame_index <= int(row["source_end_frame_inclusive"]):
            return str(row["source_shot_id"])
    return str(view.get("selected_shot_id") or "")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_full_frame(source: Path, frame_index: int, output: Path) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index}: {source}")
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Cannot write full-frame evidence: {output}")


def _write_bbox_crop(
    source: Path, frame_index: int, bbox: Sequence[float], output: Path
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read ACTIVE frame {frame_index}: {source}")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid ACTIVE crop bbox at frame {frame_index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame[y1:y2, x1:x2]):
        raise RuntimeError(f"Cannot write ACTIVE crop: {output}")


def build_initial_memory_revision(
    *,
    root: Path,
    output_dir: Path,
    raw_state: Mapping[str, Any],
    launch: Mapping[str, Any],
    view: Mapping[str, Any],
) -> tuple[Path, str] | None:
    pending = raw_state.get("pending_action")
    if not isinstance(pending, Mapping) or pending.get("type") != "MEMORY_REVIEW":
        return None
    memory = raw_state.get("memory")
    timeline_value = memory.get("source_timeline") if isinstance(memory, Mapping) else None
    if not timeline_value:
        raise RuntimeError("MEMORY_REVIEW is missing the same-shot source timeline.")
    timeline_path = Path(str(timeline_value)).resolve()
    timeline_sha = validate_input_file(timeline_path, None, "same-shot timeline")
    timeline = read_object(timeline_path)
    active = [
        dict(row)
        for row in timeline.get("frames") or []
        if isinstance(row, Mapping)
        and str(row.get("state") or "").upper() in {"ACTIVE", "REACQUIRED"}
        and isinstance(row.get("bbox_xyxy"), list)
        and len(row["bbox_xyxy"]) == 4
    ]
    if not active:
        raise RuntimeError("Same-shot Phase-1 produced no real ACTIVE/REACQUIRED frames.")
    selected_rows = [active[index] for index in sorted({0, len(active) // 2, len(active) - 1})]
    source = Path(str(view["source_video"]["path"])).resolve()
    offset = int(view["source_offset_frame"])
    active_references: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        local_frame = int(row["frame_index"])
        source_frame = local_frame + offset
        crop = output_dir / "initial_target_memory" / "active_crops" / (
            f"active_{index:02d}_frame_{source_frame:06d}.jpg"
        )
        _write_bbox_crop(source, source_frame, row["bbox_xyxy"], crop)
        active_references.append(
            {
                "kind": "SAME_SHOT_ACTIVE_CROP",
                "frame_id": source_frame,
                "runtime_local_frame_id": local_frame,
                "path": str(crop),
                "sha256": sha256_file(crop),
                "scale_class": "same-shot-active",
                "state": str(row["state"]),
                "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
            }
        )

    target_reference_path = Path(str(launch["target_reference_set"]["path"])).resolve()
    target_reference = read_object(target_reference_path)
    native_references: list[dict[str, Any]] = []
    for raw in target_reference.get("references") or []:
        if not isinstance(raw, Mapping):
            continue
        path = Path(str(raw.get("path") or "")).resolve()
        digest = validate_input_file(path, str(raw.get("sha256") or ""), "native reference")
        native_references.append(
            {
                "kind": "IMMUTABLE_REVIEW_BUNDLE_REFERENCE",
                "frame_id": int(raw["frame_id"]),
                "path": str(path),
                "sha256": digest,
                "scale_class": str(raw.get("scale") or "unknown"),
            }
        )
    if not native_references:
        raise RuntimeError("Immutable target reference set is empty.")

    work = raw_state.get("work") if isinstance(raw_state.get("work"), Mapping) else {}
    v2_name = str(work.get("v2_memory_test_name") or "")
    runtime_memory_path = root / "runs" / "target_centric_tracking_v2" / v2_name / "stage3a2_target_memory.json"
    runtime_memory = read_object(runtime_memory_path)
    scoring_inputs: dict[str, dict[str, Any]] = {}
    embeddings = runtime_memory.get("embeddings") if isinstance(runtime_memory.get("embeddings"), Mapping) else {}
    for logical, key in (("target_embeddings", "target_path"), ("negative_embeddings", "negative_path")):
        path = Path(str(embeddings.get(key) or "")).resolve()
        scoring_inputs[logical] = {"path": str(path), "sha256": validate_input_file(path, None, logical)}

    target_selection = read_object(Path(str(launch["target_selection"]["path"])).resolve())
    selection_id = str(target_selection["selection_id"])
    candidate_id = str(target_selection["selected_candidate_id"])
    seed = hashlib.sha256(
        f"{selection_id}:{candidate_id}:{timeline_sha}".encode("utf-8")
    ).hexdigest()[:16]
    revision_id = f"ecmem_initial_{seed}"
    document = {
        "schema_version": "kickclip.initial_target_memory_revision.r1",
        "immutable": True,
        "memory_revision_id": revision_id,
        "selection_id": selection_id,
        "candidate_id": candidate_id,
        "source_shot_id": str(target_selection["shot_id"]),
        "source_tracklet_id": str(target_selection["tracklet_id"]),
        "same_shot_timeline_path": str(timeline_path),
        "same_shot_timeline_sha256": timeline_sha,
        "native_references": native_references,
        "active_references": active_references,
        "references": native_references + active_references,
        "reference_count": len(native_references) + len(active_references),
        "active_or_reacquired_frame_count": len(active),
        "runtime_scoring_inputs": scoring_inputs,
        "runtime_memory_source_path": str(runtime_memory_path),
        "runtime_memory_source_sha256": sha256_file(runtime_memory_path),
        "automatic_target_confirmation": False,
    }
    path = output_dir / "initial_target_memory" / f"{revision_id}.json"
    atomic_json(path, document)
    return path, sha256_file(path)


def _candidate_evidence(
    *,
    root: Path,
    output_dir: Path,
    raw_dir: Path,
    raw_state: Mapping[str, Any],
    view: Mapping[str, Any],
    ambiguity: Mapping[str, Any],
    candidate: Mapping[str, Any],
    memory_path: Path | None,
    memory_sha: str | None,
    generation: int,
) -> dict[str, Any]:
    ambiguity_id = str(ambiguity["ambiguity_id"])
    candidate_id = str(candidate["candidate_id"])
    raw_shot_id = str(ambiguity["shot_id"])
    mapping = _shot_map_by_raw(view).get(raw_shot_id)
    if mapping is None:
        raise RuntimeError(f"No reviewed-shot mapping for runtime shot: {raw_shot_id}")
    assignments_path = Path(str(ambiguity.get("assignments") or "")).resolve()
    rows = [
        row for row in _read_csv_rows(assignments_path)
        if str(row.get("candidate_id") or "") == candidate_id
    ]
    if not rows:
        raise RuntimeError(f"Candidate has no runtime assignments: {candidate_id}")
    anchor_module = _load_module(
        "kickclip_r1_evidence_anchor",
        root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
    )
    selected, scored = anchor_module.choose_anchor(rows)
    offset = int(view["source_offset_frame"])
    observations = []
    for row in rows:
        local_frame = int(row["frame_index"])
        observations.append(
            {
                "frame_index": local_frame + offset,
                "runtime_local_frame_index": local_frame,
                "detection_id": str(row.get("detection_id") or ""),
                "confidence": float(row.get("confidence") or 0.0),
                "bbox_xyxy": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
            }
        )
    anchor_local = int(selected["frame_index"])
    anchor_global = anchor_local + offset
    evidence_dir = output_dir / "runtime_candidate_evidence" / _safe_name(ambiguity_id) / _safe_name(candidate_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source_video = Path(str(view["source_video"]["path"])).resolve()
    full_frame = evidence_dir / "full_frame_context.jpg"
    if not full_frame.is_file():
        _write_full_frame(source_video, anchor_global, full_frame)
    shot_clip = output_dir / "runtime_candidate_evidence" / "shots" / f"{_safe_name(mapping['source_shot_id'])}.mp4"
    if not shot_clip.is_file():
        _write_video_range(
            source_video,
            shot_clip,
            start_frame=int(mapping["source_start_frame"]),
            end_frame_inclusive=int(mapping["source_end_frame_inclusive"]),
            metadata=view["source_video"],
        )
    strip = raw_dir / "work" / "shots" / raw_shot_id / "candidate_strips" / f"{candidate_id}.jpg"
    gallery = evidence_dir / "reference_gallery.jpg"
    if strip.is_file():
        shutil.copy2(strip, gallery)
    else:
        contact_sheet = Path(str(ambiguity.get("contact_sheet") or "")).resolve()
        if not contact_sheet.is_file():
            raise RuntimeError(f"Candidate visual evidence is missing: {candidate_id}")
        shutil.copy2(contact_sheet, gallery)

    backend_memory = read_object(memory_path) if memory_path and memory_path.is_file() else {}
    runtime_memory = raw_state.get("memory") if isinstance(raw_state.get("memory"), Mapping) else {}
    manifest = {
        "schema_version": "kickclip.runtime_candidate_manifest.r1",
        "candidate_id": candidate_id,
        "ambiguity_id": ambiguity_id,
        "shot_id": str(mapping["source_shot_id"]),
        "runtime_shot_id": raw_shot_id,
        "tracklet_id": candidate_id,
        "identity_pure": True,
        "identity_purity_source": "FROZEN_STAGE3B0_LOCAL_TRACKLET_POLICY",
        "start_frame": int(candidate.get("start_frame", rows[0]["frame_index"])) + offset,
        "end_frame_inclusive": int(candidate.get("end_frame_inclusive", rows[-1]["frame_index"])) + offset,
        "best_observation": {
            "frame_index": anchor_global,
            "runtime_local_frame_index": anchor_local,
            "bbox_xyxy": [
                float(selected["x1"]),
                float(selected["y1"]),
                float(selected["x2"]),
                float(selected["y2"]),
            ],
            "stage3b3_scored_observations": scored,
        },
        "observations": observations,
        "ranking_metrics": dict(candidate),
        "evidence": {
            "full_frame_context_path": str(full_frame),
            "full_frame_context_sha256": sha256_file(full_frame),
            "shot_clip_path": str(shot_clip),
            "shot_clip_sha256": sha256_file(shot_clip),
            "reference_gallery_path": str(gallery),
            "reference_gallery_sha256": sha256_file(gallery),
            "assignments_path": str(assignments_path),
            "assignments_sha256": sha256_file(assignments_path),
        },
        "score_evidence": {
            "candidate_scoring_generation": generation,
            "runtime_memory_source": dict(runtime_memory),
            "backend_memory_revision_path": str(memory_path) if memory_path else None,
            "backend_memory_revision_sha256": memory_sha,
            "backend_memory_reference_count": len(backend_memory.get("references") or []),
            "backend_memory_used_by_provided_e2e_scoring": bool(
                runtime_memory.get("backend_memory_used_by_provided_e2e_scoring")
            ),
        },
        "automatic_target_confirmation": False,
    }
    manifest_path = evidence_dir / "candidate_manifest.json"
    manifest_sha = None
    atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    return {
        **dict(candidate),
        "candidate_id": candidate_id,
        "shot_id": str(mapping["source_shot_id"]),
        "runtime_shot_id": raw_shot_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "full_frame_context_path": str(full_frame),
        "full_frame_context_sha256": sha256_file(full_frame),
        "shot_clip_path": str(shot_clip),
        "shot_clip_sha256": sha256_file(shot_clip),
        "reference_gallery_path": str(gallery),
        "reference_gallery_sha256": sha256_file(gallery),
        "best_frame": anchor_global,
        "best_bbox_xyxy": manifest["best_observation"]["bbox_xyxy"],
        "score_evidence": manifest["score_evidence"],
        "status": "PENDING",
    }


def _normalize_timeline(
    *,
    raw_timeline: Path,
    output_timeline: Path,
    view: Mapping[str, Any],
) -> None:
    raw = read_object(raw_timeline)
    source = view["source_video"]
    frame_count = int(source["frame_count"])
    fps = float(source["fps"])
    frames = [
        {
            "frame_index": index,
            "time_seconds": index / fps,
            "shot_id": _shot_for_global_frame(view, index),
            "state": "SEARCHING",
            "bbox_xyxy": None,
            "tracking_confidence": 0.0,
            "identity_confidence": 0.0,
            "identity_source": "NONE",
            "selected_detection_id": None,
            "decision_reason": "OUTSIDE_SELECTION_ANCHOR_VIEW",
            "review_required": False,
            "ambiguity_id": None,
        }
        for index in range(frame_count)
    ]
    offset = int(view["source_offset_frame"])
    for raw_row in raw.get("frames") or []:
        if not isinstance(raw_row, Mapping):
            continue
        local = int(raw_row.get("frame_index", -1))
        global_index = local + offset
        if not 0 <= global_index < frame_count:
            continue
        row = dict(raw_row)
        row["frame_index"] = global_index
        row["time_seconds"] = global_index / fps
        row["shot_id"] = _shot_for_global_frame(view, global_index)
        row["runtime_local_frame_index"] = local
        frames[global_index] = row
    result = {
        **raw,
        "video": dict(source),
        "frames": frames,
        "provenance": {
            **(dict(raw.get("provenance") or {}) if isinstance(raw.get("provenance"), Mapping) else {}),
            "selection_anchor_view": {
                "source_offset_frame": offset,
                "unresolved_prefix_frame_count": int(view.get("unresolved_prefix_frame_count") or 0),
                "frame_zero_fallback_used": False,
            },
            "observation_copy_used_as_tracking_success": False,
            "synthetic_tracking_used": False,
        },
    }
    atomic_json(output_timeline, result)


def normalize_state(
    *,
    root: Path,
    output_dir: Path,
    raw_dir: Path,
    launch: Mapping[str, Any],
    view: Mapping[str, Any],
    memory_path: Path | None,
    memory_sha: str | None,
    generation: int,
) -> None:
    raw_state_path = raw_dir / "pipeline_state.json"
    if not raw_state_path.is_file():
        raise FileNotFoundError(raw_state_path)
    raw = read_object(raw_state_path)
    initial_memory = build_initial_memory_revision(
        root=root,
        output_dir=output_dir,
        raw_state=raw,
        launch=launch,
        view=view,
    )
    effective_memory_path = memory_path or (initial_memory[0] if initial_memory else None)
    effective_memory_sha = memory_sha or (initial_memory[1] if initial_memory else None)
    mapping = _shot_map_by_raw(view)
    normalized_shots: list[dict[str, Any]] = []
    for raw_shot in raw.get("shots") or []:
        if not isinstance(raw_shot, Mapping):
            continue
        raw_id = str(raw_shot.get("shot_id") or "")
        source = mapping.get(raw_id)
        row = dict(raw_shot)
        if source:
            row.update(
                {
                    "runtime_shot_id": raw_id,
                    "shot_id": source["source_shot_id"],
                    "start_frame": source["source_start_frame"],
                    "end_frame_inclusive": source["source_end_frame_inclusive"],
                    "processed_start_frame": source["processed_source_start_frame"],
                }
            )
            if (
                int(view.get("unresolved_prefix_frame_count") or 0) > 0
                and int(source["raw_shot_index"]) == 0
                and str(row.get("status") or "").upper() in {
                    "TARGET_CONFIRMED_AND_TRACKED",
                    "ACCEPTED",
                    "TRACKED",
                }
            ):
                row["runtime_forward_status"] = row["status"]
                row["status"] = "UNRESOLVED_SELECTION_PREFIX"
        normalized_shots.append(row)

    normalized_ambiguities: list[dict[str, Any]] = []
    for raw_ambiguity in raw.get("ambiguities") or []:
        if not isinstance(raw_ambiguity, Mapping):
            continue
        ambiguity = dict(raw_ambiguity)
        raw_shot_id = str(ambiguity.get("shot_id") or "")
        source = mapping.get(raw_shot_id)
        if source:
            ambiguity["runtime_shot_id"] = raw_shot_id
            ambiguity["shot_id"] = source["source_shot_id"]
            ambiguity["start_frame"] = source["source_start_frame"]
            ambiguity["end_frame_inclusive"] = source["source_end_frame_inclusive"]
        candidates = []
        for candidate in ambiguity.get("review_candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            candidates.append(
                _candidate_evidence(
                    root=root,
                    output_dir=output_dir,
                    raw_dir=raw_dir,
                    raw_state=raw,
                    view=view,
                    ambiguity=raw_ambiguity,
                    candidate=candidate,
                    memory_path=effective_memory_path,
                    memory_sha=effective_memory_sha,
                    generation=generation,
                )
            )
        if candidates:
            ambiguity["review_candidates"] = candidates
        normalized_ambiguities.append(ambiguity)

    pending = dict(raw.get("pending_action") or {}) if isinstance(raw.get("pending_action"), Mapping) else None
    if pending:
        raw_shot_id = str(pending.get("shot_id") or "")
        source = mapping.get(raw_shot_id)
        if source:
            pending["runtime_shot_id"] = raw_shot_id
            pending["shot_id"] = source["source_shot_id"]

    state = dict(raw)
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "execution_kind": "EVENT_CANDIDATE_HANDOFF_R1",
            "video": dict(view["source_video"]),
            "shots": normalized_shots,
            "runtime": {
                **(dict(raw.get("runtime") or {}) if isinstance(raw.get("runtime"), Mapping) else {}),
                "integration_path": "B.ADD_THIN_BACKEND_ADAPTER_TO_V1_V2_STAGES",
                "provided_e2e_runner_used": True,
                "selection_aware_video_adapter_used": True,
                "selection_anchor_view_path": str(output_dir / "_selection_anchor_view" / "selection_view.json"),
                "selection_anchor_view_sha256": sha256_file(output_dir / "_selection_anchor_view" / "selection_view.json"),
                "source_offset_frame": int(view["source_offset_frame"]),
                "unresolved_prefix_frame_count": int(view.get("unresolved_prefix_frame_count") or 0),
                "reviewed_shot_boundaries_translated": True,
                "synthetic_tracking_used": False,
                "observation_copy_used_as_success": False,
                "frame_zero_fallback_used": False,
                "automatic_target_confirmation": False,
                "memory_revision_path": (
                    str(effective_memory_path) if effective_memory_path else None
                ),
                "memory_revision_sha256": effective_memory_sha,
                "reference_count": (
                    len((read_object(effective_memory_path).get("references") or []))
                    if effective_memory_path and effective_memory_path.is_file()
                    else len(read_object(Path(str(launch["target_reference_set"]["path"]))).get("references") or [])
                ),
                "candidate_scoring_generation": generation,
                "backend_memory_used_by_provided_e2e_scoring": bool(
                    (raw.get("memory") or {}).get(
                        "backend_memory_used_by_provided_e2e_scoring"
                    )
                    if isinstance(raw.get("memory"), Mapping)
                    else False
                ),
            },
            "pending_action": pending,
            "ambiguities": normalized_ambiguities,
        }
    )
    if str(state.get("status") or "") == "COMPLETE" and int(view.get("unresolved_prefix_frame_count") or 0) > 0:
        state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
        state["decision"] = "E2E_COMPLETE_WITH_UNRESOLVED_SELECTION_PREFIX"
    atomic_json(output_dir / "pipeline_state.json", state)

    raw_timeline = raw_dir / "target_timeline.json"
    if raw_timeline.is_file():
        _normalize_timeline(
            raw_timeline=raw_timeline,
            output_timeline=output_dir / "target_timeline.json",
            view=view,
        )
    for name in (
        "target_timeline.csv",
        "pipeline_summary.json",
        "pipeline_manifest.json",
        "full_frame_tracking_preview.mp4",
        "target_centered_preview.mp4",
    ):
        source = raw_dir / name
        target = output_dir / name
        if source.is_file():
            shutil.copy2(source, target)
    summary_path = output_dir / "pipeline_summary.json"
    summary = read_object(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "status": state.get("status"),
            "decision": state.get("decision"),
            "selection_anchor_view": True,
            "source_offset_frame": int(view["source_offset_frame"]),
            "unresolved_prefix_frame_count": int(view.get("unresolved_prefix_frame_count") or 0),
            "backend_memory_used_by_provided_e2e_scoring": state["runtime"][
                "backend_memory_used_by_provided_e2e_scoring"
            ],
            "full_event_recommendation_e2e": "NOT_RUN",
        }
    )
    atomic_json(summary_path, summary)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_path = args.tracking_launch_manifest.resolve()
    launch = read_object(launch_path)
    launch_wrapper = {"path": str(launch_path)}
    validate_input_file(launch_path, None, "tracking launch manifest")
    validate_input_file(args.target_selection, launch["target_selection"]["sha256"], "target selection")
    validate_input_file(args.target_reference_set, launch["target_reference_set"]["sha256"], "target reference set")
    validate_input_file(args.earlier_anchor_decision, launch["earlier_anchor_decision"]["sha256"], "earlier anchor decision")
    validate_input_file(args.shot_boundaries, launch["shot_boundaries"]["sha256"], "reviewed shot boundaries")
    target_selection = read_object(args.target_selection)
    anchor_decision = read_object(args.earlier_anchor_decision)
    boundaries = read_object(args.shot_boundaries)
    if anchor_decision.get("frame_zero_fallback_used") is not False:
        raise ValueError("FRAME_ZERO_FALLBACK_FORBIDDEN")
    if target_selection.get("automatic_target_confirmation") is not False:
        raise ValueError("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")

    memory_sha: str | None = None
    if args.target_memory_revision is not None:
        memory_sha = validate_input_file(
            args.target_memory_revision,
            args.target_memory_sha256,
            "target memory revision",
        )

    dependencies = dependency_report(root)
    if args.verify_only:
        print(json.dumps(dependencies, ensure_ascii=False, indent=2))
        return 0 if not dependencies["missing"] else 2
    if dependencies["missing"]:
        write_blocked_state(
            output_dir,
            decision="BLOCK_MISSING_FROZEN_RUNTIME_DEPENDENCY",
            failure_code="MISSING_FROZEN_RUNTIME_DEPENDENCY",
            message="The supplied research source is present, but required helper/model files are missing.",
            launch=launch_wrapper,
            dependencies=dependencies,
        )
        return 2

    # A confirmed candidate creates a new immutable backend memory revision. The
    # supplied E2E resume CLI continues directly into the next search while still
    # using its original pre-cut embedding gallery. Running it would falsely claim
    # that the new memory drove scoring, so this adapter blocks that resume contract.
    if args.resume and args.confirmed_candidate and args.target_memory_revision is not None:
        write_blocked_state(
            output_dir,
            decision="BLOCK_SUPPLIED_E2E_CANNOT_APPLY_CONFIRMED_MEMORY_BEFORE_NEXT_SEARCH",
            failure_code="RUNTIME_MEMORY_UPDATE_UNSUPPORTED",
            message=(
                "The supplied E2E runner does not load a backend memory revision before "
                "searching the next reviewed shot. A selection-aware post-confirmation "
                "memory update stage is required."
            ),
            launch=launch_wrapper,
            dependencies=dependencies,
            extra_runtime={
                "memory_revision_path": str(args.target_memory_revision.resolve()),
                "memory_revision_sha256": memory_sha,
                "candidate_scoring_generation": args.candidate_scoring_generation,
                "backend_memory_used_by_provided_e2e_scoring": False,
            },
        )
        return 2

    if args.resume and (args.rejected_candidate or args.unreviewable_candidate):
        write_blocked_state(
            output_dir,
            decision="BLOCK_PROVIDED_E2E_REJECTION_RESUME_UNSUPPORTED",
            failure_code="RUNTIME_REJECTION_RESUME_UNSUPPORTED",
            message="DIFFERENT_PLAYER/UNREVIEWABLE requires a selection-aware service resume adapter.",
            launch=launch_wrapper,
            dependencies=dependencies,
        )
        return 2

    view_path = output_dir / "_selection_anchor_view" / "selection_view.json"
    if args.resume:
        if not view_path.is_file():
            raise FileNotFoundError(f"Selection anchor view is missing for resume: {view_path}")
        view = read_object(view_path)
    else:
        if args.video is None or args.initial_bbox is None:
            raise ValueError("New run requires video and initial bbox.")
        view = materialize_selection_view(
            output_dir=output_dir,
            launch=launch,
            target_selection=target_selection,
            boundaries=boundaries,
            overwrite=args.overwrite,
        )

    e2e = Path(dependencies["files"]["e2e_runner"])
    raw_root = output_dir / "_provided_e2e_output"
    if not args.resume and not args.contract_test_skip_phase1_compatibility:
        compatibility_module = _load_module(
            "kickclip_phase1_compatibility_executor",
            Path(__file__).with_name("phase1_compatibility_executor.py"),
        )
        safe = _safe_name(args.test_name)
        gate_path = output_dir / "phase1_compatibility_gate.json"
        compatibility_module.execute_phase1_compatibility(
            project_root=root,
            video=Path(str(view["same_shot_pre_cut_source"]["path"])),
            initial_bbox=args.initial_bbox,
            original_stage0_test_name=f"e2e_{safe}__precut_stage0_original",
            phase1_test_name=f"e2e_{safe}__precut_source",
            compatibility_test_name=f"e2e_{safe}__precut_compatibility",
            device="cpu" if args.device == "mps" else args.device,
            gate_path=gate_path,
        )
    command = [
        sys.executable,
        str(e2e),
        "--project-root",
        str(root),
        "--test-name",
        args.test_name,
        "--device",
        "cpu" if args.device == "mps" else args.device,
        "--reacquisition-mode",
        args.reacquisition_mode,
        "--output-root",
        str(raw_root),
        "--backend-safe-weights-runner",
        str(Path(__file__).with_name("safe_weights_subprocess.py").resolve()),
    ]
    if args.resume:
        command.append("--resume")
        if args.target_memory_revision is not None and memory_sha:
            command.extend(
                [
                    "--backend-memory-revision",
                    str(args.target_memory_revision.resolve()),
                    "--backend-memory-sha256",
                    memory_sha,
                ]
            )
        if args.ambiguity_id:
            command.extend(["--ambiguity-id", args.ambiguity_id])
        if args.approve_review:
            command.extend(["--approve-review", args.approve_review])
        elif args.reject_review:
            command.extend(["--reject-review", args.reject_review])
        elif args.confirm_absent:
            command.append("--confirm-absent")
        elif args.confirmed_candidate:
            command.extend(["--confirmed-candidate", args.confirmed_candidate])
    else:
        suffix_video = Path(str(view["selection_view_video"]["path"])).resolve()
        command.extend(["--video", str(suffix_video), "--initial-bbox", *map(str, args.initial_bbox)])
        cut_frames = [int(value) for value in view.get("translated_cut_frames") or []]
        if cut_frames:
            command.extend(["--cut-frames", *map(str, cut_frames)])
        if args.overwrite:
            command.append("--overwrite")
    if args.reviewer:
        command.extend(["--reviewer", args.reviewer])
    if args.review_note:
        command.extend(["--review-note", args.review_note])
    if args.no_preview:
        command.append("--no-preview")

    rc = invoke(command, root)
    raw_dir = raw_root / args.test_name
    if raw_dir.is_dir():
        normalize_state(
            root=root,
            output_dir=output_dir,
            raw_dir=raw_dir,
            launch=launch,
            view=view,
            memory_path=args.target_memory_revision.resolve() if args.target_memory_revision else None,
            memory_sha=memory_sha,
            generation=args.candidate_scoring_generation,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
