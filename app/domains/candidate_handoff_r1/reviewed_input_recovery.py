from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


APPROVED_REVIEW_STATES = frozenset({"REVIEWED_PASS", "CONFIRMED"})


class ReviewedInputRecoveryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RecoveredReviewedInputBundle:
    source_root: Path
    scene_video_path: Path
    reviewed_shots_path: Path
    detections_path: Path
    scene_video_sha256: str
    reviewed_shots_sha256: str
    detections_sha256: str
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int
    shot_count: int
    source_match_similarity: float

    @property
    def identity(self) -> tuple[str, str, str]:
        return (
            self.scene_video_sha256,
            self.reviewed_shots_sha256,
            self.detections_sha256,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _shot_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("shots", "boundaries", "shot_boundaries"):
        value = document.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return list(value)
    raise ValueError("Reviewed shot artifact has no supported shot array.")


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Scene video is not decodable.")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if width < 1 or height < 1 or fps <= 0 or frame_count < 1:
        raise ValueError("Scene video metadata is invalid.")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
    }


def _validate_shots(
    document: dict[str, Any],
    *,
    scene_id: str,
    frame_count: int,
) -> int:
    declared_scene_id = str(document.get("scene_id") or "")
    if declared_scene_id and declared_scene_id != scene_id:
        raise ValueError("Reviewed shot artifact scene_id differs from the requested scene.")
    rows = _shot_rows(document)
    intervals: list[tuple[int, int]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        shot_id = str(_first(row, "shot_id", "id") or f"shot_{index:04d}")
        if shot_id in seen_ids:
            raise ValueError("Reviewed shot artifact contains duplicate shot IDs.")
        seen_ids.add(shot_id)
        start = _first(row, "start_frame", "first_frame", "frame_start")
        end = _first(
            row,
            "end_frame_inclusive",
            "end_frame",
            "last_frame",
            "frame_end",
        )
        if start is None or end is None:
            raise ValueError("Reviewed shot artifact has an incomplete frame interval.")
        start_frame, end_frame = int(start), int(end)
        if start_frame < 0 or end_frame < start_frame or end_frame >= frame_count:
            raise ValueError("Reviewed shot interval is outside the scene video.")
        state = str(
            _first(row, "review_status", "review_state", "status", "boundary_status")
            or ""
        ).strip().upper()
        if state not in APPROVED_REVIEW_STATES:
            raise ValueError(f"Shot {shot_id} is not human-reviewed: {state or 'MISSING'}")
        intervals.append((start_frame, end_frame))
    intervals.sort()
    cursor = 0
    for start_frame, end_frame in intervals:
        if start_frame != cursor:
            raise ValueError("Reviewed shots do not provide exact, non-overlapping frame coverage.")
        cursor = end_frame + 1
    if cursor != frame_count:
        raise ValueError("Reviewed shots do not cover the complete scene video.")
    return len(intervals)


def _validate_detections(path: Path, *, frame_count: int) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        frame_key = next(
            (
                key
                for key in (
                    "frame_index",
                    "scene_local_frame_index",
                    "scene_local_frame",
                    "frame",
                )
                if key in fieldnames
            ),
            None,
        )
        if frame_key is None:
            raise ValueError("Frozen detections have no scene-local frame column.")
        bbox_columns = {"x1", "y1", "x2", "y2"}
        if not bbox_columns.issubset(fieldnames) and not ({"bbox_xyxy", "bbox"} & fieldnames):
            raise ValueError("Frozen detections have no supported bbox columns.")
        row_count = 0
        for row in reader:
            frame_index = int(float(str(row.get(frame_key) or "-1")))
            if frame_index < 0 or frame_index >= frame_count:
                raise ValueError("Frozen detection frame is outside the scene video.")
            row_count += 1
        if row_count < 1:
            raise ValueError("Frozen detections are empty.")


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> Any:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    return frame if ok else None


def _frame_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    size = (96, 54)
    left_gray = cv2.cvtColor(cv2.resize(left, size), cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(cv2.resize(right, size), cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(left_gray, right_gray)
    return max(0.0, 1.0 - float(difference.mean()) / 255.0)


def _source_match_similarity(
    *,
    source_video_path: Path,
    scene_video_path: Path,
    source_start_sec: float,
    source_end_sec: float,
    source_fps: float,
    scene_fps: float,
    scene_frame_count: int,
) -> float:
    source = cv2.VideoCapture(str(source_video_path))
    scene = cv2.VideoCapture(str(scene_video_path))
    if not source.isOpened() or not scene.isOpened():
        source.release()
        scene.release()
        raise ValueError("Source or scene video is not decodable.")
    try:
        sample_count = min(7, scene_frame_count)
        if sample_count <= 1:
            local_frames = [0]
        else:
            local_frames = sorted(
                {
                    round(index * (scene_frame_count - 1) / (sample_count - 1))
                    for index in range(sample_count)
                }
            )
        similarities: list[float] = []
        max_source_frame = int(round(source_end_sec * source_fps))
        for local_frame in local_frames:
            scene_frame = _read_frame(scene, local_frame)
            source_time = source_start_sec + local_frame / scene_fps
            expected_source_frame = int(round(source_time * source_fps))
            best = 0.0
            for offset in (-2, -1, 0, 1, 2):
                source_frame_index = expected_source_frame + offset
                if source_frame_index < 0 or source_frame_index > max_source_frame:
                    continue
                best = max(
                    best,
                    _frame_similarity(
                        scene_frame,
                        _read_frame(source, source_frame_index),
                    ),
                )
            similarities.append(best)
    finally:
        source.release()
        scene.release()
    return float(statistics.median(similarities)) if similarities else 0.0


def _candidate_roots(storage_root: Path, scene_id: str) -> Iterable[Path]:
    base = (storage_root / "candidate_pipeline_inputs").resolve()
    if not base.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in base.glob(f"*/*/{scene_id}")
        if path.is_dir() and path.resolve().is_relative_to(base)
    )


def find_recoverable_reviewed_input_bundle(
    *,
    storage_root: Path,
    scene_id: str,
    source_video_path: Path,
    source_video_sha256: str,
    source_start_sec: float,
    source_end_sec: float,
    source_fps: float,
) -> RecoveredReviewedInputBundle:
    storage_root = storage_root.resolve()
    source_video_path = source_video_path.resolve()
    if not source_video_path.is_file() or sha256_file(source_video_path) != source_video_sha256:
        raise ReviewedInputRecoveryError(
            "SOURCE_VIDEO_NOT_READY",
            "The current immutable Action Spotting source video is missing or changed.",
        )
    if source_fps <= 0 or source_start_sec < 0 or source_end_sec <= source_start_sec:
        raise ReviewedInputRecoveryError(
            "FRAME_MAPPING_NOT_READY",
            "The current event interval or source FPS is invalid.",
        )

    valid: list[RecoveredReviewedInputBundle] = []
    rejected: list[str] = []
    for root in _candidate_roots(storage_root, scene_id):
        scene_video = root / "scene.mp4"
        reviewed_shots = root / "reviewed_shots.json"
        detections = root / "detections.csv"
        if not all(path.is_file() for path in (scene_video, reviewed_shots, detections)):
            continue
        try:
            metadata = _video_metadata(scene_video)
            scene_sha = sha256_file(scene_video)
            document = _load_object(reviewed_shots)
            declared_video = document.get("video") or {}
            declared_sha = str(declared_video.get("sha256") or "")
            if declared_sha != scene_sha:
                raise ValueError("Reviewed shots do not declare the colocated scene video SHA-256.")
            duration_tolerance = max(0.35, 3.0 / float(metadata["fps"]))
            expected_duration = source_end_sec - source_start_sec
            if abs(float(metadata["duration_sec"]) - expected_duration) > duration_tolerance:
                raise ValueError("Scene video duration differs from the current event interval.")
            shot_count = _validate_shots(
                document,
                scene_id=scene_id,
                frame_count=int(metadata["frame_count"]),
            )
            _validate_detections(detections, frame_count=int(metadata["frame_count"]))
            similarity = _source_match_similarity(
                source_video_path=source_video_path,
                scene_video_path=scene_video,
                source_start_sec=source_start_sec,
                source_end_sec=source_end_sec,
                source_fps=source_fps,
                scene_fps=float(metadata["fps"]),
                scene_frame_count=int(metadata["frame_count"]),
            )
            if similarity < 0.93:
                raise ValueError(
                    f"Scene video does not match the current event source frames ({similarity:.4f})."
                )
            valid.append(
                RecoveredReviewedInputBundle(
                    source_root=root,
                    scene_video_path=scene_video,
                    reviewed_shots_path=reviewed_shots,
                    detections_path=detections,
                    scene_video_sha256=scene_sha,
                    reviewed_shots_sha256=sha256_file(reviewed_shots),
                    detections_sha256=sha256_file(detections),
                    duration_sec=float(metadata["duration_sec"]),
                    fps=float(metadata["fps"]),
                    width=int(metadata["width"]),
                    height=int(metadata["height"]),
                    frame_count=int(metadata["frame_count"]),
                    shot_count=shot_count,
                    source_match_similarity=similarity,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append(f"{root}: {exc}")

    if not valid:
        detail = f" Checked {len(rejected)} stored bundle(s)." if rejected else ""
        reasons = (
            " Rejections: " + " | ".join(rejected[:3])
            if rejected
            else ""
        )
        raise ReviewedInputRecoveryError(
            "REVIEWED_SHOT_BOUNDARIES_NOT_READY",
            "No DB artifact or safely recoverable human-reviewed bundle exists for this exact event source."
            + detail
            + reasons,
        )
    identities = {bundle.identity for bundle in valid}
    if len(identities) != 1:
        raise ReviewedInputRecoveryError(
            "AMBIGUOUS_REVIEWED_INPUT_RECOVERY",
            "Multiple different human-reviewed input bundles match the same event source; explicit review is required.",
        )
    valid.sort(key=lambda bundle: bundle.source_root.stat().st_mtime, reverse=True)
    return valid[0]
