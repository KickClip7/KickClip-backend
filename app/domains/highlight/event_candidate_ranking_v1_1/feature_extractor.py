from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contract import CandidateObservation, CandidateSequence


BALL_SIGNAL_AVAILABLE = "AVAILABLE"
BALL_SIGNAL_LOW_COVERAGE = "LOW_COVERAGE"
BALL_SIGNAL_UNAVAILABLE = "UNAVAILABLE"
BALL_SIGNAL_UNAVAILABLE_CODE = "BALL_SIGNAL_UNAVAILABLE"
BALL_SIGNAL_LOW_COVERAGE_CODE = "BALL_SIGNAL_LOW_COVERAGE"


@dataclass(frozen=True)
class Detection:
    frame: int
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float | None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.split(",")]
    if isinstance(value, dict):
        value = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def _detection_from_row(row: dict[str, Any]) -> Detection | None:
    frame = next(
        (
            row.get(key)
            for key in ("global_frame", "global_frame_index", "frame_index", "frame")
            if row.get(key) is not None
        ),
        None,
    )
    label = next(
        (
            row.get(key)
            for key in ("class_name", "label", "class", "category")
            if row.get(key) is not None
        ),
        None,
    )
    box = _bbox(
        row.get("bbox_xyxy")
        or row.get("bbox")
        or {
            "x1": row.get("x1"),
            "y1": row.get("y1"),
            "x2": row.get("x2"),
            "y2": row.get("y2"),
        }
    )
    if frame is None or label is None or box is None:
        return None
    confidence = row.get("confidence")
    return Detection(
        frame=int(frame),
        label=str(label).strip().lower(),
        bbox_xyxy=box,
        confidence=float(confidence) if confidence not in (None, "") else None,
    )


def load_detections(path: Path) -> tuple[Detection, ...]:
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, list):
            rows = document
        elif isinstance(document, dict):
            rows = document.get("detections") or document.get("items") or []
        else:
            rows = []
    return tuple(
        detection
        for row in rows
        if isinstance(row, dict)
        for detection in [_detection_from_row(row)]
        if detection is not None
    )


def _center(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(
        [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
        dtype=float,
    )


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _direction_changes(vectors: list[np.ndarray]) -> float | None:
    if len(vectors) < 2:
        return None
    changes: list[float] = []
    for previous, current in zip(vectors, vectors[1:]):
        denom = float(np.linalg.norm(previous) * np.linalg.norm(current))
        if denom <= 1e-9:
            continue
        cosine = float(np.clip(np.dot(previous, current) / denom, -1.0, 1.0))
        changes.append(math.acos(cosine) / math.pi)
    return _mean(changes)


class VideoFrameReader:
    def __init__(self, path: Path) -> None:
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError("Source video cannot be opened.")
        self.cache: dict[int, np.ndarray] = {}

    def get(self, frame_index: int) -> np.ndarray | None:
        if frame_index in self.cache:
            return self.cache[frame_index]
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.cache[frame_index] = frame
        return frame

    def close(self) -> None:
        self.capture.release()


class RawFeatureExtractor:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        event_scene_local_sec: float,
        event_window_before_sec: float,
        event_window_after_sec: float,
        closeup_area_ratio: float,
        ball_labels: set[str],
        ball_low_coverage_threshold: float,
    ) -> None:
        self.width = width
        self.height = height
        self.event_time = event_scene_local_sec
        self.window_start = event_scene_local_sec - event_window_before_sec
        self.window_end = event_scene_local_sec + event_window_after_sec
        self.closeup_area_ratio = closeup_area_ratio
        self.ball_labels = {value.lower() for value in ball_labels}
        self.ball_low_coverage_threshold = ball_low_coverage_threshold

    def extract_all(
        self,
        *,
        candidates: tuple[CandidateSequence, ...],
        detections: tuple[Detection, ...],
        video_path: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        by_frame: dict[int, list[tuple[str, CandidateObservation]]] = defaultdict(list)
        for candidate in candidates:
            for observation in candidate.observations:
                by_frame[observation.global_frame].append(
                    (candidate.candidate_id, observation)
                )
        balls_by_frame: dict[int, list[Detection]] = defaultdict(list)
        for detection in detections:
            if detection.label in self.ball_labels:
                balls_by_frame[detection.frame].append(detection)
        reader = VideoFrameReader(video_path)
        try:
            rows = [
                self.extract_candidate(
                    candidate,
                    by_frame=by_frame,
                    balls_by_frame=balls_by_frame,
                    reader=reader,
                )
                for candidate in candidates
            ]
        finally:
            reader.close()
        group_counts = {
            group: sum(
                1
                for row in rows
                if row["feature_availability"][group]["available"]
            )
            for group in (
                "temporal",
                "visual",
                "motion",
                "broadcast",
                "ball",
                "field_context",
            )
        }
        total = len(rows)
        summary = {
            "candidate_count": total,
            "actual_candidate_observations_available": sum(
                len(candidate.observations) for candidate in candidates
            ),
            "group_available_candidate_counts": group_counts,
            "group_coverage": {
                key: (value / total if total else None)
                for key, value in group_counts.items()
            },
            "ball_detection_count": sum(len(value) for value in balls_by_frame.values()),
        }
        return rows, summary

    def extract_candidate(
        self,
        candidate: CandidateSequence,
        *,
        by_frame: dict[int, list[tuple[str, CandidateObservation]]],
        balls_by_frame: dict[int, list[Detection]],
        reader: VideoFrameReader,
    ) -> dict[str, Any]:
        observations = candidate.observations
        times = [item.scene_local_time_sec for item in observations]
        closest = min(observations, key=lambda item: abs(item.scene_local_time_sec - self.event_time))
        before = [time for time in times if time <= self.event_time]
        after = [time for time in times if time >= self.event_time]
        duration = max(0.0, times[-1] - times[0])
        in_window = [
            item for item in observations
            if self.window_start <= item.scene_local_time_sec <= self.window_end
        ]

        areas = [_area(item.bbox_xyxy) / (self.width * self.height) for item in observations]
        image_center = np.asarray([self.width / 2.0, self.height / 2.0])
        half_diagonal = float(np.linalg.norm(image_center))
        center_scores = [
            max(0.0, 1.0 - float(np.linalg.norm(_center(item.bbox_xyxy) - image_center)) / half_diagonal)
            for item in observations
        ]
        border_clips = [
            float(
                item.bbox_xyxy[0] <= 0
                or item.bbox_xyxy[1] <= 0
                or item.bbox_xyxy[2] >= self.width - 1
                or item.bbox_xyxy[3] >= self.height - 1
            )
            for item in observations
        ]
        overlaps = [
            max(
                [
                    _iou(item.bbox_xyxy, other.bbox_xyxy)
                    for other_id, other in by_frame[item.global_frame]
                    if other_id != candidate.candidate_id
                ]
                or [0.0]
            )
            for item in observations
        ]
        sharpness: list[float] = []
        sharpness_frames: list[int] = []
        for item in observations:
            frame = reader.get(item.global_frame)
            if frame is None:
                continue
            x1 = max(0, min(self.width - 1, int(item.bbox_xyxy[0])))
            y1 = max(0, min(self.height - 1, int(item.bbox_xyxy[1])))
            x2 = max(x1 + 1, min(self.width, int(math.ceil(item.bbox_xyxy[2]))))
            y2 = max(y1 + 1, min(self.height, int(math.ceil(item.bbox_xyxy[3]))))
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                sharpness.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()))
                sharpness_frames.append(item.global_frame)

        velocities: list[np.ndarray] = []
        velocity_times: list[float] = []
        for previous, current in zip(observations, observations[1:]):
            delta = current.scene_local_time_sec - previous.scene_local_time_sec
            if delta <= 0:
                continue
            normalized = (
                (_center(current.bbox_xyxy) - _center(previous.bbox_xyxy))
                / np.asarray([self.width, self.height])
                / delta
            )
            velocities.append(normalized)
            velocity_times.append((current.scene_local_time_sec + previous.scene_local_time_sec) / 2.0)
        speeds = [float(np.linalg.norm(value)) for value in velocities]
        accelerations = [
            float(np.linalg.norm(current - previous))
            for previous, current in zip(velocities, velocities[1:])
        ]

        post_closeups = [
            item
            for item, area in zip(observations, areas)
            if item.scene_local_time_sec >= self.event_time
            and area >= self.closeup_area_ratio
        ]
        post_shots = {
            item.shot_id
            for item in observations
            if item.scene_local_time_sec >= self.event_time
        }

        ball_distances: list[tuple[CandidateObservation, float, Detection]] = []
        for item in observations:
            player_foot = np.asarray(
                [
                    (item.bbox_xyxy[0] + item.bbox_xyxy[2]) / 2.0,
                    item.bbox_xyxy[1] + 0.85 * (item.bbox_xyxy[3] - item.bbox_xyxy[1]),
                ]
            )
            for ball in balls_by_frame.get(item.global_frame, []):
                distance = float(
                    np.linalg.norm(_center(ball.bbox_xyxy) - player_foot)
                    / math.hypot(self.width, self.height)
                )
                ball_distances.append((item, distance, ball))
        covered_frames = {row[0].global_frame for row in ball_distances}
        ball_coverage = len(covered_frames) / len(observations)
        if not balls_by_frame:
            ball_state = BALL_SIGNAL_UNAVAILABLE
            ball_risks = [BALL_SIGNAL_UNAVAILABLE_CODE]
        elif ball_coverage < self.ball_low_coverage_threshold:
            ball_state = BALL_SIGNAL_LOW_COVERAGE
            ball_risks = [BALL_SIGNAL_LOW_COVERAGE_CODE]
        else:
            ball_state = BALL_SIGNAL_AVAILABLE
            ball_risks = []
        pre_ball = [
            distance
            for item, distance, _ in ball_distances
            if item.scene_local_time_sec <= self.event_time
        ]
        pre_ball_change = (
            pre_ball[-1] - pre_ball[0] if len(pre_ball) >= 2 else None
        )
        ball_centers = [
            _center(ball.bbox_xyxy)
            for _, _, ball in sorted(
                ball_distances, key=lambda row: row[0].scene_local_time_sec
            )
        ]
        ball_vectors = [
            current - previous
            for previous, current in zip(ball_centers, ball_centers[1:])
        ]

        visual_available = bool(sharpness)
        motion_available = bool(velocities)
        ball_available = ball_state == BALL_SIGNAL_AVAILABLE
        features = {
            "temporal": {
                "first_candidate_time_sec": times[0],
                "last_candidate_time_sec": times[-1],
                "event_window_overlap": len(in_window) / len(observations),
                "visible_duration_before_event_sec": (
                    max(before) - min(before) if len(before) >= 2 else 0.0
                ),
                "visible_duration_after_event_sec": (
                    max(after) - min(after) if len(after) >= 2 else 0.0
                ),
                "representative_time_distance_sec": abs(
                    closest.scene_local_time_sec - self.event_time
                ),
                "closest_observation_to_event": {
                    "global_frame": closest.global_frame,
                    "scene_local_time_sec": closest.scene_local_time_sec,
                    "distance_sec": abs(
                        closest.scene_local_time_sec - self.event_time
                    ),
                },
                "local_track_duration_ratio": duration
                / max(1e-9, self.window_end - self.window_start),
            },
            "visual": {
                "bbox_area_ratio": _mean(areas),
                "center_proximity": _mean(center_scores),
                "crop_sharpness": _mean(sharpness),
                "border_clipping_ratio": _mean(border_clips),
                "overlap_occlusion_ratio": _mean(overlaps),
                "bbox_scale_change": (
                    (areas[-1] - areas[0]) / areas[0] if areas[0] > 0 else None
                ),
            },
            "motion": {
                "center_velocity": _mean(speeds),
                "acceleration": _mean(accelerations),
                "direction_change": _direction_changes(velocities),
                "pre_event_motion_magnitude": _mean(
                    [
                        speed
                        for speed, time in zip(speeds, velocity_times)
                        if time < self.event_time
                    ]
                ),
                "post_event_motion_magnitude": _mean(
                    [
                        speed
                        for speed, time in zip(speeds, velocity_times)
                        if time >= self.event_time
                    ]
                ),
            },
            "broadcast": {
                "first_post_event_closeup_delay_sec": (
                    post_closeups[0].scene_local_time_sec - self.event_time
                    if post_closeups
                    else None
                ),
                "post_event_closeup_duration_sec": (
                    post_closeups[-1].scene_local_time_sec
                    - post_closeups[0].scene_local_time_sec
                    if len(post_closeups) >= 2
                    else (0.0 if post_closeups else None)
                ),
                "post_event_shots_with_appearance": len(post_shots),
                "repeated_post_event_focus": (
                    len(post_shots) / max(1, len({item.shot_id for item in observations}))
                    if post_shots
                    else 0.0
                ),
            },
            "ball": {
                "state": ball_state,
                "coverage": ball_coverage if balls_by_frame else None,
                "foot_region_distance": (
                    _mean([row[1] for row in ball_distances])
                    if ball_available
                    else None
                ),
                "pre_event_distance_change": (
                    pre_ball_change if ball_available else None
                ),
                "velocity": (
                    _mean([float(np.linalg.norm(value)) for value in ball_vectors])
                    if ball_available and ball_vectors
                    else None
                ),
                "direction_change": (
                    _direction_changes(ball_vectors)
                    if ball_available
                    else None
                ),
            },
            "field_context": None,
        }
        return {
            "candidate_id": candidate.candidate_id,
            "shot_id": candidate.shot_id,
            "shot_index": candidate.shot_index,
            "local_tracklet_id": candidate.local_tracklet_id,
            "trackability_score": candidate.trackability_score,
            "raw_features": features,
            "feature_availability": {
                "temporal": {
                    "available": True,
                    "coverage": 1.0,
                    "source": "scene_candidates.json:observations",
                },
                "visual": {
                    "available": visual_available,
                    "coverage": len(sharpness) / len(observations),
                    "source": "source_video+scene_candidates.json:observations",
                },
                "motion": {
                    "available": motion_available,
                    "coverage": len(velocities) / max(1, len(observations) - 1),
                    "source": "scene_candidates.json:observations",
                },
                "broadcast": {
                    "available": True,
                    "coverage": 1.0,
                    "source": "shot_id+scene_candidates.json:observations",
                },
                "ball": {
                    "available": ball_available,
                    "coverage": ball_coverage if balls_by_frame else None,
                    "source": "detections_artifact",
                    "state": ball_state,
                },
                "field_context": {
                    "available": False,
                    "coverage": None,
                    "source": None,
                    "reason": "FIELD_CONTEXT_NOT_IMPLEMENTED",
                },
            },
            "feature_evidence": {
                "temporal_frames": [
                    observations[0].global_frame,
                    closest.global_frame,
                    observations[-1].global_frame,
                ],
                "visual_frames": sharpness_frames,
                "motion_frames": [
                    item.global_frame for item in observations
                ],
                "broadcast_frames": [
                    item.global_frame for item in post_closeups
                ],
                "ball_frames": sorted(covered_frames),
            },
            "risk_codes": ball_risks,
            "_trajectory": [
                {
                    "frame": item.global_frame,
                    "time_sec": item.scene_local_time_sec,
                    "bbox_xyxy": list(item.bbox_xyxy),
                }
                for item in observations
            ],
        }
