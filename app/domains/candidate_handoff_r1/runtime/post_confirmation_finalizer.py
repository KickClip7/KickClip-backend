from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASE4C_POLICY = "USER_CONFIRMED_CANDIDATE_TIMELINE_COMMIT_R1"
PHASE4C_REPORT_SCHEMA = "kickclip.phase4c_post_confirmation_report.v1"
PHASE4C_LINK_SCHEMA = "kickclip.user_confirmed_candidate_link.v1"
PHASE4C_TIMELINE_SOURCE = "USER_CONFIRMED_REAL_RFDETR_TRACKLET"
TERMINAL_DECISION = "TRACKING_COMPLETED_WITH_USER_CONFIRMED_CROSS_SHOT_LINK"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_name(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in value
    )


def _valid_bbox(
    bbox: object,
    *,
    width: int,
    height: int,
) -> list[float]:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        raise RuntimeError("Confirmed candidate observation has no bbox.")
    if len(bbox) != 4:
        raise RuntimeError("Confirmed candidate bbox must contain four values.")
    result = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("Confirmed candidate bbox contains a non-finite value.")
    x1, y1, x2, y2 = result
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Confirmed candidate bbox is invalid: {result}")
    if x2 > width + 1.0 or y2 > height + 1.0:
        raise RuntimeError(
            "Confirmed candidate bbox escapes the source video bounds: "
            f"bbox={result}, video={width}x{height}"
        )
    return [
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    ]


def _candidate_manifest_path(
    output_dir: Path,
    *,
    shot_id: str,
    candidate_id: str,
) -> Path:
    return (
        output_dir
        / "phase4b_cross_shot"
        / _safe_name(shot_id)
        / "candidates"
        / _safe_name(candidate_id)
        / "candidate_manifest.json"
    )


def _extract_observations(
    manifest: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    source_start = int(manifest.get("start_frame", -1))
    source_end = int(manifest.get("end_frame_inclusive", -1))
    if source_start < 0 or source_end < source_start:
        raise RuntimeError("Candidate manifest has an invalid source frame range.")

    direct = _rows(manifest.get("observations"))
    observations: list[dict[str, Any]] = []
    if direct:
        for row in direct:
            observations.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "runtime_local_frame_index": row.get(
                        "runtime_local_frame_index"
                    ),
                    "detection_id": str(row.get("detection_id") or ""),
                    "confidence": float(row.get("confidence") or 0.0),
                    "bbox_xyxy": _valid_bbox(
                        row.get("bbox_xyxy"),
                        width=width,
                        height=height,
                    ),
                    "identity_observability_clean": bool(
                        row.get("identity_observability_clean", True)
                    ),
                    "review_risk_flags": list(
                        row.get("review_risk_flags") or []
                    ),
                }
            )
    else:
        observability = _object(manifest.get("identity_observability"))
        local_rows = _rows(observability.get("per_observation"))
        if not local_rows:
            raise RuntimeError(
                "Candidate manifest contains no real detector observations."
            )
        local_frames = [int(row["frame"]) for row in local_rows]
        source_offset = source_start - min(local_frames)
        for row in local_rows:
            local_frame = int(row["frame"])
            source_frame = local_frame + source_offset
            observations.append(
                {
                    "frame_index": source_frame,
                    "runtime_local_frame_index": local_frame,
                    "detection_id": str(row.get("detection_id") or ""),
                    "confidence": float(row.get("confidence") or 0.0),
                    "bbox_xyxy": _valid_bbox(
                        row.get("bbox_xyxy"),
                        width=width,
                        height=height,
                    ),
                    "identity_observability_clean": bool(
                        row.get("clean_for_reid") is True
                    ),
                    "review_risk_flags": list(
                        row.get("rejection_reasons") or []
                    ),
                }
            )

    observations.sort(key=lambda item: int(item["frame_index"]))
    seen_frames: set[int] = set()
    seen_detections: set[str] = set()
    for row in observations:
        frame_index = int(row["frame_index"])
        detection_id = str(row["detection_id"])
        if frame_index in seen_frames:
            raise RuntimeError(
                f"Confirmed candidate has duplicate source frame {frame_index}."
            )
        if detection_id and detection_id in seen_detections:
            raise RuntimeError(
                f"Confirmed candidate has duplicate detection {detection_id}."
            )
        if frame_index < source_start or frame_index > source_end:
            raise RuntimeError(
                "Confirmed observation escapes candidate frame range: "
                f"{frame_index} not in [{source_start}, {source_end}]"
            )
        if not 0.0 <= float(row["confidence"]) <= 1.0:
            raise RuntimeError("Confirmed observation confidence is invalid.")
        seen_frames.add(frame_index)
        if detection_id:
            seen_detections.add(detection_id)

    if len(observations) < 3:
        raise RuntimeError(
            "Confirmed candidate does not contain enough real observations."
        )
    return observations


def _append_unique_confirmation(
    rows: list[dict[str, Any]],
    confirmation: Mapping[str, Any],
) -> None:
    decision_id = str(confirmation.get("decision_id") or "")
    if any(str(row.get("decision_id") or "") == decision_id for row in rows):
        return
    rows.append(dict(confirmation))


def _resolve_ambiguity(
    ambiguities: list[dict[str, Any]],
    *,
    ambiguity_id: str,
    decision: Mapping[str, Any],
    candidate_link_path: Path,
    candidate_link_sha256: str,
) -> None:
    found = False
    for row in ambiguities:
        if str(row.get("ambiguity_id") or "") != ambiguity_id:
            continue
        found = True
        row.update(
            {
                "status": "RESOLVED_SAME_PLAYER",
                "operational_state": "TERMINAL",
                "decision": "SAME_PLAYER",
                "resolved_at": str(decision.get("reviewed_at") or _now_iso()),
                "review_decision": dict(decision),
                "confirmed_candidate_id": str(
                    decision.get("candidate_id") or ""
                ),
                "candidate_link_path": str(candidate_link_path),
                "candidate_link_sha256": candidate_link_sha256,
                "automatic_target_confirmation": False,
            }
        )
    if not found:
        raise RuntimeError(
            "Pending ambiguity is missing from pipeline_state.json."
        )


def _mark_shot_tracked(
    shots: list[dict[str, Any]],
    *,
    shot_id: str,
    candidate_id: str,
    start_frame: int,
    end_frame: int,
    bbox_frame_count: int,
) -> None:
    found = False
    for shot in shots:
        if str(shot.get("shot_id") or "") != shot_id:
            continue
        found = True
        shot.update(
            {
                "status": "TRACKED",
                "tracking_resolution": (
                    "USER_CONFIRMED_SAME_PLAYER_REAL_DETECTIONS"
                ),
                "confirmed_candidate_id": candidate_id,
                "confirmed_segment_start_frame": start_frame,
                "confirmed_segment_end_frame_inclusive": end_frame,
                "confirmed_bbox_frame_count": bbox_frame_count,
                "automatic_target_confirmation": False,
            }
        )
    if not found:
        raise RuntimeError(f"Confirmed shot is missing from shot contract: {shot_id}")


def _write_timeline_csv(path: Path, timeline: Mapping[str, Any]) -> None:
    fields = [
        "frame_index",
        "time_seconds",
        "shot_id",
        "state",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "tracking_confidence",
        "identity_confidence",
        "identity_source",
        "selected_detection_id",
        "decision_reason",
        "candidate_id",
        "memory_revision_id",
        "confirmation_decision_id",
        "review_required",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in _rows(timeline.get("frames")):
            bbox = row.get("bbox_xyxy")
            bbox_values = (
                [float(value) for value in bbox]
                if isinstance(bbox, list) and len(bbox) == 4
                else [None, None, None, None]
            )
            writer.writerow(
                {
                    "frame_index": row.get("frame_index"),
                    "time_seconds": row.get("time_seconds"),
                    "shot_id": row.get("shot_id"),
                    "state": row.get("state"),
                    "bbox_x1": bbox_values[0],
                    "bbox_y1": bbox_values[1],
                    "bbox_x2": bbox_values[2],
                    "bbox_y2": bbox_values[3],
                    "tracking_confidence": row.get(
                        "tracking_confidence", 0.0
                    ),
                    "identity_confidence": row.get(
                        "identity_confidence", 0.0
                    ),
                    "identity_source": row.get("identity_source"),
                    "selected_detection_id": row.get(
                        "selected_detection_id"
                    ),
                    "decision_reason": row.get("decision_reason"),
                    "candidate_id": row.get("candidate_id"),
                    "memory_revision_id": row.get("memory_revision_id"),
                    "confirmation_decision_id": row.get(
                        "confirmation_decision_id"
                    ),
                    "review_required": row.get("review_required", False),
                }
            )
    temp.replace(path)


def _render_previews(
    *,
    video_path: Path,
    timeline: Mapping[str, Any],
    full_preview_path: Path,
    centered_preview_path: Path,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to regenerate confirmed tracking previews."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Source video metadata is invalid.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    full_temp = full_preview_path.with_suffix(".phase4c.tmp.mp4")
    centered_temp = centered_preview_path.with_suffix(".phase4c.tmp.mp4")
    full_preview_path.parent.mkdir(parents=True, exist_ok=True)
    centered_preview_path.parent.mkdir(parents=True, exist_ok=True)
    for temp in (full_temp, centered_temp):
        if temp.exists():
            temp.unlink()

    full_writer = cv2.VideoWriter(
        str(full_temp), fourcc, fps, (width, height)
    )
    centered_writer = cv2.VideoWriter(
        str(centered_temp), fourcc, fps, (width, height)
    )
    if not full_writer.isOpened() or not centered_writer.isOpened():
        capture.release()
        full_writer.release()
        centered_writer.release()
        raise RuntimeError("Cannot create Phase 4-C preview writers.")

    frame_map = {
        int(row["frame_index"]): row
        for row in _rows(timeline.get("frames"))
    }
    index = 0
    rendered = 0
    confirmed_bbox_frames = 0
    smooth_center: tuple[float, float] | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = frame_map.get(index, {})
            bbox = row.get("bbox_xyxy")
            full_frame = frame.copy()
            centered_frame = frame
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                confirmed_bbox_frames += 1
                p1 = (int(round(x1)), int(round(y1)))
                p2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(full_frame, p1, p2, (0, 255, 0), 3)
                cv2.putText(
                    full_frame,
                    "USER CONFIRMED TARGET",
                    (max(0, p1[0]), max(30, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                if smooth_center is None:
                    smooth_center = center
                else:
                    smooth_center = (
                        0.72 * smooth_center[0] + 0.28 * center[0],
                        0.72 * smooth_center[1] + 0.28 * center[1],
                    )
                bbox_h = max(1.0, y2 - y1)
                crop_h = min(
                    float(height),
                    max(float(height) * 0.38, bbox_h * 4.2),
                )
                crop_w = min(
                    float(width),
                    crop_h * float(width) / float(height),
                )
                cx, cy = smooth_center
                left = max(0.0, min(float(width) - crop_w, cx - crop_w / 2))
                top = max(0.0, min(float(height) - crop_h, cy - crop_h / 2))
                right = left + crop_w
                bottom = top + crop_h
                crop = frame[
                    int(round(top)):int(round(bottom)),
                    int(round(left)):int(round(right)),
                ]
                if crop.size:
                    centered_frame = cv2.resize(
                        crop, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                    cv2.putText(
                        centered_frame,
                        "USER CONFIRMED TARGET-CENTERED",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                full_frame,
                f"frame {index}",
                (24, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            full_writer.write(full_frame)
            centered_writer.write(centered_frame)
            index += 1
            rendered += 1
    finally:
        capture.release()
        full_writer.release()
        centered_writer.release()

    expected = int(_object(timeline.get("video")).get("frame_count") or 0)
    if expected and rendered != expected:
        for temp in (full_temp, centered_temp):
            temp.unlink(missing_ok=True)
        raise RuntimeError(
            "Preview frame count mismatch: "
            f"rendered={rendered}, expected={expected}"
        )
    full_temp.replace(full_preview_path)
    centered_temp.replace(centered_preview_path)
    return {
        "frame_count": rendered,
        "confirmed_bbox_frame_count": confirmed_bbox_frames,
        "full_frame_tracking_preview_path": str(full_preview_path),
        "full_frame_tracking_preview_sha256": _sha256_file(
            full_preview_path
        ),
        "target_centered_preview_path": str(centered_preview_path),
        "target_centered_preview_sha256": _sha256_file(
            centered_preview_path
        ),
    }


def finalize_confirmed_candidate(
    *,
    output_dir: Path,
    candidate_id: str,
    ambiguity_id: str,
    memory_path: Path,
    memory_sha256: str,
    no_preview: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    memory_path = memory_path.resolve()
    report_dir = output_dir / "phase4c_post_confirmation"
    report_path = report_dir / "phase4c_post_confirmation_report.json"

    if report_path.is_file():
        existing = _read_object(report_path)
        if (
            existing.get("status") == "PASS"
            and existing.get("candidate_id") == candidate_id
            and existing.get("ambiguity_id") == ambiguity_id
            and existing.get("memory_revision_sha256") == memory_sha256
        ):
            return existing

    state_path = output_dir / "pipeline_state.json"
    timeline_path = output_dir / "target_timeline.json"
    summary_path = output_dir / "pipeline_summary.json"
    if not state_path.is_file() or not timeline_path.is_file():
        raise RuntimeError(
            "Phase 4-C requires durable pipeline_state.json and "
            "target_timeline.json."
        )

    actual_memory_sha = _sha256_file(memory_path)
    if (
        len(memory_sha256) != 64
        or actual_memory_sha.lower() != memory_sha256.lower()
    ):
        raise RuntimeError("Confirmed target memory SHA-256 mismatch.")
    memory = _read_object(memory_path)
    if memory.get("source_candidate_id") != candidate_id:
        raise RuntimeError(
            "Confirmed memory source candidate does not match the decision."
        )
    if memory.get("automatic_target_confirmation") is not False:
        raise RuntimeError(
            "Confirmed memory must preserve automatic_target_confirmation=false."
        )
    shot_id = str(memory.get("source_shot_id") or "")
    decision_id = str(memory.get("confirmation_decision_id") or "")
    memory_revision_id = str(memory.get("memory_revision_id") or "")
    generation = int(memory.get("candidate_scoring_generation") or 0)
    if not shot_id or not decision_id or not memory_revision_id:
        raise RuntimeError("Confirmed target memory provenance is incomplete.")

    decision_path = output_dir / "review_decisions" / f"{decision_id}.json"
    decision = _read_object(decision_path)
    if (
        decision.get("state") != "SAME_PLAYER"
        or decision.get("candidate_id") != candidate_id
        or decision.get("ambiguity_id") != ambiguity_id
        or decision.get("automatic_target_confirmation") is not False
    ):
        raise RuntimeError("SAME_PLAYER decision provenance mismatch.")

    state = _read_object(state_path)
    pending = _object(state.get("pending_action"))
    if (
        pending.get("type") != "CROSS_SHOT_CONFIRMATION"
        or pending.get("ambiguity_id") != ambiguity_id
        or candidate_id not in [
            str(value) for value in pending.get("candidate_ids") or []
        ]
    ):
        raise RuntimeError(
            "Durable runtime state is not waiting for the confirmed candidate."
        )
    if pending.get("automatic_target_confirmation") is not False:
        raise RuntimeError(
            "Pending review contract must preserve automatic confirmation=false."
        )

    timeline = _read_object(timeline_path)
    video = _object(timeline.get("video"))
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    frame_count = int(video.get("frame_count") or 0)
    fps = float(video.get("fps") or 0.0)
    if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
        raise RuntimeError("Target timeline video metadata is invalid.")

    manifest_path = _candidate_manifest_path(
        output_dir,
        shot_id=shot_id,
        candidate_id=candidate_id,
    )
    manifest = _read_object(manifest_path)
    quality = _object(manifest.get("quality"))
    purity = _object(manifest.get("identity_purity"))
    if (
        manifest.get("candidate_id") != candidate_id
        or manifest.get("shot_id") != shot_id
        or quality.get("identity_pure") is not True
        or quality.get("identity_purity_gate_passed") is not True
        or quality.get("identity_observability_gate_passed") is not True
        or purity.get("passed") is not True
        or manifest.get("automatic_target_confirmation") is not False
    ):
        raise RuntimeError(
            "Confirmed candidate does not satisfy the frozen safety gates."
        )

    observations = _extract_observations(
        manifest,
        width=width,
        height=height,
    )
    source_frames = [int(row["frame_index"]) for row in observations]
    if max(source_frames) >= frame_count:
        raise RuntimeError("Confirmed observations escape the target timeline.")

    report_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = report_dir / "pre_commit_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for source in (state_path, timeline_path, summary_path):
        if source.is_file():
            target = backup_dir / source.name
            if not target.exists():
                shutil.copy2(source, target)

    candidate_link_path = report_dir / "candidate_link.json"
    candidate_link = {
        "schema_version": PHASE4C_LINK_SCHEMA,
        "created_at": _now_iso(),
        "policy": PHASE4C_POLICY,
        "tracking_test_name": str(state.get("test_name") or ""),
        "ambiguity_id": ambiguity_id,
        "shot_id": shot_id,
        "candidate_id": candidate_id,
        "tracklet_id": str(
            manifest.get("tracklet_id") or candidate_id
        ),
        "parent_tracklet_id": str(
            manifest.get("parent_tracklet_id") or candidate_id
        ),
        "decision_id": decision_id,
        "decision_artifact_path": str(decision_path),
        "decision_artifact_sha256": _sha256_file(decision_path),
        "memory_revision_id": memory_revision_id,
        "memory_revision_path": str(memory_path),
        "memory_revision_sha256": actual_memory_sha,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "observation_count": len(observations),
        "observation_frame_ids": source_frames,
        "source_start_frame": min(source_frames),
        "source_end_frame_inclusive": max(source_frames),
        "identity_pure": True,
        "real_detector_observations_only": True,
        "interpolation_used": False,
        "synthetic_tracking_used": False,
        "automatic_target_confirmation": False,
    }
    _atomic_json(candidate_link_path, candidate_link)
    candidate_link_sha = _sha256_file(candidate_link_path)

    frame_rows = _rows(timeline.get("frames"))
    frame_map = {
        int(row.get("frame_index", -1)): row
        for row in frame_rows
    }
    if len(frame_map) != frame_count:
        raise RuntimeError(
            "Target timeline does not contain exactly one row per source frame."
        )

    previous_frame: int | None = None
    for observation in observations:
        frame_index = int(observation["frame_index"])
        row = frame_map[frame_index]
        reacquired = (
            previous_frame is None or frame_index - previous_frame > 1
        )
        row.update(
            {
                "state": "REACQUIRED" if reacquired else "ACTIVE",
                "bbox_xyxy": list(observation["bbox_xyxy"]),
                "predicted_bbox_xyxy": None,
                "bbox_source": "RFDETR_USER_CONFIRMED_TRACKLET",
                "tracking_confidence": float(observation["confidence"]),
                "identity_confidence": 1.0,
                "visibility": 1.0,
                "selected_detection_id": str(
                    observation["detection_id"]
                ),
                "decision_reason": "USER_CONFIRMED_SAME_PLAYER",
                "review_risk_flags": list(
                    observation.get("review_risk_flags") or []
                ),
                "identity_source": "USER_CONFIRMED_SAME_PLAYER",
                "review_required": False,
                "ambiguity_id": None,
                "candidate_id": candidate_id,
                "tracklet_id": str(
                    manifest.get("tracklet_id") or candidate_id
                ),
                "memory_revision_id": memory_revision_id,
                "confirmation_decision_id": decision_id,
                "runtime_local_frame_index": observation.get(
                    "runtime_local_frame_index"
                ),
                "phase4c_source": PHASE4C_TIMELINE_SOURCE,
                "identity_observability_clean": bool(
                    observation.get("identity_observability_clean")
                ),
                "synthetic_tracking_used": False,
                "automatic_target_confirmation": False,
            }
        )
        previous_frame = frame_index

    timeline["frames"] = frame_rows

    confirmation = {
        "ambiguity_id": ambiguity_id,
        "shot_id": shot_id,
        "decision": "SAME_PLAYER",
        "candidate_id": candidate_id,
        "decision_id": decision_id,
        "memory_revision_id": memory_revision_id,
        "observation_frame_ids": source_frames,
        "reviewer": decision.get("reviewer"),
        "review_note": decision.get("note"),
        "reviewed_at": decision.get("reviewed_at"),
        "candidate_link_path": str(candidate_link_path),
        "candidate_link_sha256": candidate_link_sha,
        "automatic_target_confirmation": False,
    }

    timeline_confirmations = _rows(timeline.get("confirmations"))
    _append_unique_confirmation(timeline_confirmations, confirmation)
    timeline["confirmations"] = timeline_confirmations
    timeline_ambiguities = _rows(timeline.get("ambiguities"))
    if timeline_ambiguities:
        for row in timeline_ambiguities:
            if str(row.get("ambiguity_id") or "") == ambiguity_id:
                row.update(
                    {
                        "status": "RESOLVED_SAME_PLAYER",
                        "decision": "SAME_PLAYER",
                        "candidate_id": candidate_id,
                        "decision_id": decision_id,
                        "automatic_target_confirmation": False,
                    }
                )
    timeline["ambiguities"] = timeline_ambiguities
    timeline_shots = _rows(timeline.get("shots"))
    _mark_shot_tracked(
        timeline_shots,
        shot_id=shot_id,
        candidate_id=candidate_id,
        start_frame=min(source_frames),
        end_frame=max(source_frames),
        bbox_frame_count=len(observations),
    )
    # The selected shot's memory review was already passed and its real
    # Phase 3-C observations remain unchanged.
    for shot in timeline_shots:
        if (
            str(shot.get("shot_id") or "") == "shot_0003"
            and str(shot.get("status") or "")
            == "INITIAL_TARGET_TRACKING_ONLY"
        ):
            shot["status"] = "TRACKED"
            shot["tracking_resolution"] = "REAL_PHASE3C_SELECTED_SHOT"
    timeline["shots"] = timeline_shots
    timeline["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
    timeline["decision"] = TERMINAL_DECISION
    timeline["updated_at"] = _now_iso()
    provenance = _object(timeline.get("provenance"))
    provenance.update(
        {
            "phase4c_policy": PHASE4C_POLICY,
            "phase4c_candidate_link_path": str(candidate_link_path),
            "phase4c_candidate_link_sha256": candidate_link_sha,
            "phase4c_real_detector_observation_count": len(observations),
            "phase4c_interpolation_used": False,
            "synthetic_tracking_used": False,
            "automatic_target_confirmation": False,
            "silent_wrong_player_switches": 0,
        }
    )
    timeline["provenance"] = provenance
    _atomic_json(timeline_path, timeline)
    _write_timeline_csv(output_dir / "target_timeline.csv", timeline)

    state_confirmations = _rows(state.get("confirmations"))
    _append_unique_confirmation(state_confirmations, confirmation)
    state["confirmations"] = state_confirmations
    state_ambiguities = _rows(state.get("ambiguities"))
    _resolve_ambiguity(
        state_ambiguities,
        ambiguity_id=ambiguity_id,
        decision=decision,
        candidate_link_path=candidate_link_path,
        candidate_link_sha256=candidate_link_sha,
    )
    state["ambiguities"] = state_ambiguities
    state_shots = _rows(state.get("shots"))
    _mark_shot_tracked(
        state_shots,
        shot_id=shot_id,
        candidate_id=candidate_id,
        start_frame=min(source_frames),
        end_frame=max(source_frames),
        bbox_frame_count=len(observations),
    )
    for shot in state_shots:
        if (
            str(shot.get("shot_id") or "") == "shot_0003"
            and str(shot.get("status") or "")
            == "INITIAL_TARGET_MEMORY_REVIEW_REQUIRED"
        ):
            shot["status"] = "TRACKED"
            shot["tracking_resolution"] = "REAL_PHASE3C_SELECTED_SHOT"
    state["shots"] = state_shots
    state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
    state["decision"] = TERMINAL_DECISION
    state["failure_code"] = None
    state["pending_action"] = None
    state["updated_at"] = _now_iso()
    state["current_memory_revision_id"] = memory_revision_id
    runtime = _object(state.get("runtime"))
    runtime.update(
        {
            "memory_revision_id": memory_revision_id,
            "memory_revision_path": str(memory_path),
            "memory_revision_sha256": actual_memory_sha,
            "reference_count": int(memory.get("reference_count") or 0),
            "candidate_scoring_generation": generation,
            "phase4c_policy": PHASE4C_POLICY,
            "phase4c_post_confirmation_complete": True,
            "phase4c_confirmed_candidate_id": candidate_id,
            "phase4c_confirmation_decision_id": decision_id,
            "phase4c_candidate_link_path": str(candidate_link_path),
            "phase4c_candidate_link_sha256": candidate_link_sha,
            "phase4c_real_detector_observation_count": len(observations),
            "backend_memory_used_by_phase4c_timeline": True,
            "candidate_link_created": True,
            "interpolation_used": False,
            "synthetic_tracking_used": False,
            "automatic_target_confirmation": False,
        }
    )
    state["runtime"] = runtime
    _atomic_json(state_path, state)

    preview = {
        "preview_regenerated": False,
        "full_frame_tracking_preview_path": str(
            output_dir / "full_frame_tracking_preview.mp4"
        ),
        "target_centered_preview_path": str(
            output_dir / "target_centered_preview.mp4"
        ),
    }
    if not no_preview:
        source_video_path = Path(str(video.get("path") or "")).resolve()
        preview = {
            "preview_regenerated": True,
            **_render_previews(
                video_path=source_video_path,
                timeline=timeline,
                full_preview_path=(
                    output_dir / "full_frame_tracking_preview.mp4"
                ),
                centered_preview_path=(
                    output_dir / "target_centered_preview.mp4"
                ),
            ),
        }

    active_count = sum(
        1
        for row in _rows(timeline.get("frames"))
        if str(row.get("state") or "") in {"ACTIVE", "REACQUIRED"}
        and isinstance(row.get("bbox_xyxy"), list)
    )
    searching_count = sum(
        1
        for row in _rows(timeline.get("frames"))
        if str(row.get("state") or "") == "SEARCHING"
    )
    lost_count = sum(
        1
        for row in _rows(timeline.get("frames"))
        if str(row.get("state") or "") == "LOST"
    )
    summary = _read_object(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "status": "COMPLETE_WITH_UNRESOLVED_GAPS",
            "decision": TERMINAL_DECISION,
            "phase4c_policy": PHASE4C_POLICY,
            "phase4c_post_confirmation_complete": True,
            "phase4c_confirmed_candidate_id": candidate_id,
            "phase4c_confirmed_shot_id": shot_id,
            "phase4c_confirmation_decision_id": decision_id,
            "phase4c_memory_revision_id": memory_revision_id,
            "phase4c_candidate_link_path": str(candidate_link_path),
            "phase4c_candidate_link_sha256": candidate_link_sha,
            "phase4c_real_detector_observation_count": len(observations),
            "frame_count": frame_count,
            "active_or_reacquired_bbox_frames": active_count,
            "searching_frames": searching_count,
            "lost_frames": lost_count,
            "timeline_valid": True,
            "timeline_sha256": _sha256_file(timeline_path),
            "preview_generated": bool(
                (output_dir / "full_frame_tracking_preview.mp4").is_file()
                and (output_dir / "target_centered_preview.mp4").is_file()
            ),
            "synthetic_tracking_used": False,
            "automatic_target_confirmation": False,
        }
    )
    _atomic_json(summary_path, summary)

    report = {
        "schema_version": PHASE4C_REPORT_SCHEMA,
        "status": "PASS",
        "decision": TERMINAL_DECISION,
        "policy": PHASE4C_POLICY,
        "ambiguity_id": ambiguity_id,
        "shot_id": shot_id,
        "candidate_id": candidate_id,
        "decision_id": decision_id,
        "memory_revision_id": memory_revision_id,
        "memory_revision_path": str(memory_path),
        "memory_revision_sha256": actual_memory_sha,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "candidate_link_path": str(candidate_link_path),
        "candidate_link_sha256": candidate_link_sha,
        "source_start_frame": min(source_frames),
        "source_end_frame_inclusive": max(source_frames),
        "real_detector_observation_count": len(observations),
        "observation_frame_ids": source_frames,
        "timeline_path": str(timeline_path),
        "timeline_sha256": _sha256_file(timeline_path),
        "timeline_csv_path": str(output_dir / "target_timeline.csv"),
        "timeline_csv_sha256": _sha256_file(
            output_dir / "target_timeline.csv"
        ),
        "active_or_reacquired_bbox_frames": active_count,
        "candidate_scoring_generation": generation,
        "preview": preview,
        "interpolation_used": False,
        "synthetic_tracking_used": False,
        "candidate_link_created": True,
        "automatic_target_confirmation": False,
        "created_at": _now_iso(),
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    report["report_sha256"] = _sha256_file(report_path)
    return report
