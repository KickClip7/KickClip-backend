from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    CandidateObservation,
    CandidateSequence,
)
from app.domains.highlight.event_candidate_ranking_v1_1.feature_extractor import (
    BALL_SIGNAL_AVAILABLE,
    BALL_SIGNAL_LOW_COVERAGE,
    BALL_SIGNAL_LOW_COVERAGE_CODE,
    BALL_SIGNAL_UNAVAILABLE,
    BALL_SIGNAL_UNAVAILABLE_CODE,
    Detection,
    RawFeatureExtractor,
    _center,
    _direction_changes,
    _mean,
)

from .frame_reader import BoundedVideoFrameReader


def _nested(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _valid_ball(
    detection: Detection,
    *,
    width: int,
    height: int,
    minimum_confidence: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> bool:
    box = detection.bbox_xyxy
    area_ratio = (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
        / (width * height)
    )
    return (
        (detection.confidence or 0.0) >= minimum_confidence
        and min_area_ratio <= area_ratio <= max_area_ratio
        and box[0] >= 0
        and box[1] >= 0
        and box[2] <= width
        and box[3] <= height
    )


def select_ball_trajectory(
    detections: tuple[Detection, ...],
    *,
    width: int,
    height: int,
    policy: dict[str, Any],
) -> dict[int, Detection]:
    candidates: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        if _valid_ball(
            detection,
            width=width,
            height=height,
            minimum_confidence=float(policy["minimum_confidence"]),
            min_area_ratio=float(policy["min_area_ratio"]),
            max_area_ratio=float(policy["max_area_ratio"]),
        ):
            candidates[detection.frame].append(detection)
    selected: dict[int, Detection] = {}
    previous: Detection | None = None
    previous_frame: int | None = None
    diagonal = math.hypot(width, height)
    for frame in sorted(candidates):
        rows = candidates[frame]
        if (
            previous is None
            or previous_frame is None
            or frame - previous_frame > int(policy["continuity_reset_gap_frames"])
        ):
            winner = max(rows, key=lambda row: row.confidence or 0.0)
        else:
            winner = max(
                rows,
                key=lambda row: (
                    float(row.confidence or 0.0)
                    - float(policy["continuity_distance_penalty"])
                    * float(
                        np.linalg.norm(
                            _center(row.bbox_xyxy)
                            - _center(previous.bbox_xyxy)
                        )
                        / diagonal
                    )
                ),
            )
        selected[frame] = winner
        previous = winner
        previous_frame = frame
    return selected


class RawFeatureExtractorV111(RawFeatureExtractor):
    def __init__(
        self,
        *,
        canonical_event_label: str,
        safety_policy: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canonical_event_label = canonical_event_label
        self.safety_policy = safety_policy

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
        ball_detections = tuple(
            detection
            for detection in detections
            if detection.label in self.ball_labels
        )
        selected_ball = select_ball_trajectory(
            ball_detections,
            width=self.width,
            height=self.height,
            policy=self.safety_policy["ball_selection"],
        )
        balls_by_frame = {
            frame: [detection] for frame, detection in selected_ball.items()
        }
        reader = BoundedVideoFrameReader(
            video_path,
            max_cached_frames=int(
                self.safety_policy["frame_decoding"]["max_cached_frames"]
            ),
        )
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
            decoding_stats = reader.stats()
        finally:
            reader.close()
        groups = (
            "temporal",
            "visual",
            "motion",
            "broadcast",
            "ball",
            "field_context",
        )
        group_counts = {
            group: sum(
                row["feature_availability"][group]["available"]
                for row in rows
            )
            for group in groups
        }
        reliability_counts = {
            state: sum(row["reliability_state"] == state for row in rows)
            for state in (
                "READY",
                "PARTIAL_FEATURES",
                "NO_RELIABLE_SHORTLIST",
            )
        }
        return rows, {
            "candidate_count": len(rows),
            "actual_candidate_observations_available": sum(
                len(candidate.observations) for candidate in candidates
            ),
            "group_available_candidate_counts": group_counts,
            "group_coverage": {
                group: (
                    group_counts[group] / len(rows) if rows else None
                )
                for group in groups
            },
            "ball_detection_audit": {
                "raw_matching_detection_count": len(ball_detections),
                "selected_valid_frame_count": len(selected_ball),
                "one_detection_per_frame": True,
            },
            "frame_decoding": decoding_stats,
            "reliability_counts": reliability_counts,
        }

    def extract_candidate(self, candidate: CandidateSequence, **kwargs: Any):
        row = super().extract_candidate(candidate, **kwargs)
        observations = candidate.observations
        balls_by_frame: dict[int, list[Detection]] = kwargs["balls_by_frame"]
        self._correct_broadcast(row, observations)
        self._correct_ball(row, observations, balls_by_frame)
        self._add_completeness(row)
        return row

    def _correct_broadcast(
        self,
        row: dict[str, Any],
        observations: tuple[CandidateObservation, ...],
    ) -> None:
        broadcast = row["raw_features"]["broadcast"]
        broadcast.pop("repeated_post_event_focus", None)
        post_count = sum(
            observation.scene_local_time_sec >= self.event_time
            for observation in observations
        )
        broadcast["local_post_event_focus"] = post_count / len(observations)
        broadcast["cross_shot_repeated_focus"] = None
        availability = row["feature_availability"]["broadcast"]
        availability["local_post_event_focus_available"] = True
        availability["cross_shot_repeated_focus_available"] = False
        availability["cross_shot_reason"] = (
            "CROSS_SHOT_IDENTITY_NOT_ESTABLISHED"
        )

    def _correct_ball(
        self,
        row: dict[str, Any],
        observations: tuple[CandidateObservation, ...],
        balls_by_frame: dict[int, list[Detection]],
    ) -> None:
        policy = self.safety_policy["ball_selection"]
        diagonal = math.hypot(self.width, self.height)
        evidence: list[tuple[CandidateObservation, Detection, float]] = []
        for observation in observations:
            selected = balls_by_frame.get(observation.global_frame)
            if not selected:
                continue
            ball = selected[0]
            player_foot = np.asarray(
                [
                    (observation.bbox_xyxy[0] + observation.bbox_xyxy[2]) / 2,
                    observation.bbox_xyxy[1]
                    + 0.85
                    * (
                        observation.bbox_xyxy[3]
                        - observation.bbox_xyxy[1]
                    ),
                ]
            )
            distance = float(
                np.linalg.norm(_center(ball.bbox_xyxy) - player_foot)
                / diagonal
            )
            evidence.append((observation, ball, distance))
        coverage = len(evidence) / len(observations)
        continuity_pairs = 0
        possible_pairs = max(0, len(observations) - 1)
        evidence_by_frame = {
            observation.global_frame: (observation, ball, distance)
            for observation, ball, distance in evidence
        }
        for previous, current in zip(observations, observations[1:]):
            left = evidence_by_frame.get(previous.global_frame)
            right = evidence_by_frame.get(current.global_frame)
            if left is None or right is None:
                continue
            frame_delta = max(1, current.global_frame - previous.global_frame)
            displacement = float(
                np.linalg.norm(_center(right[1].bbox_xyxy) - _center(left[1].bbox_xyxy))
                / diagonal
                / frame_delta
            )
            if displacement <= float(policy["max_normalized_displacement_per_frame"]):
                continuity_pairs += 1
        continuity = (
            continuity_pairs / possible_pairs if possible_pairs else None
        )
        if not balls_by_frame:
            state = BALL_SIGNAL_UNAVAILABLE
            risk = BALL_SIGNAL_UNAVAILABLE_CODE
        elif (
            coverage < float(policy["minimum_candidate_coverage"])
            or continuity is None
            or continuity < float(policy["minimum_trajectory_continuity"])
        ):
            state = BALL_SIGNAL_LOW_COVERAGE
            risk = BALL_SIGNAL_LOW_COVERAGE_CODE
        else:
            state = BALL_SIGNAL_AVAILABLE
            risk = None
        event_near = [
            item
            for item in evidence
            if abs(item[0].scene_local_time_sec - self.event_time)
            <= float(policy["event_near_window_sec"])
        ]
        pre_event = [
            item for item in evidence
            if item[0].scene_local_time_sec <= self.event_time
        ]
        distances = [item[2] for item in event_near]
        trend = None
        if state == BALL_SIGNAL_AVAILABLE and len(pre_event) >= 2:
            times = np.asarray(
                [item[0].scene_local_time_sec for item in pre_event]
            )
            values = np.asarray([item[2] for item in pre_event])
            if float(np.ptp(times)) > 0:
                trend = float(np.polyfit(times, values, 1)[0])
        ball_vectors: list[np.ndarray] = []
        ball_speeds: list[float] = []
        for previous, current in zip(evidence, evidence[1:]):
            delta = (
                current[0].scene_local_time_sec
                - previous[0].scene_local_time_sec
            )
            if delta <= 0:
                continue
            vector = (
                _center(current[1].bbox_xyxy)
                - _center(previous[1].bbox_xyxy)
            ) / np.asarray([self.width, self.height]) / delta
            ball_vectors.append(vector)
            ball_speeds.append(float(np.linalg.norm(vector)))
        available = state == BALL_SIGNAL_AVAILABLE
        event_near_median = (
            float(np.median(distances))
            if available and distances
            else None
        )
        row["raw_features"]["ball"] = {
            "state": state,
            "coverage": coverage if balls_by_frame else None,
            "trajectory_continuity": continuity if balls_by_frame else None,
            "event_near_min_foot_distance": (
                min(distances) if available and distances else None
            ),
            "event_near_median_foot_distance": (
                event_near_median
            ),
            "pre_event_distance_trend": trend if available else None,
            "foot_region_distance": event_near_median,
            "pre_event_distance_change": trend if available else None,
            "velocity": _mean(ball_speeds) if available else None,
            "direction_change": (
                _direction_changes(ball_vectors) if available else None
            ),
        }
        row["feature_availability"]["ball"] = {
            "available": available,
            "coverage": coverage if balls_by_frame else None,
            "trajectory_continuity": continuity if balls_by_frame else None,
            "source": "detections_artifact:selected_one_per_frame",
            "state": state,
        }
        row["feature_evidence"]["ball_frames"] = [
            item[0].global_frame for item in evidence
        ]
        row["risk_codes"] = [
            code
            for code in row.get("risk_codes", [])
            if code
            not in {
                BALL_SIGNAL_UNAVAILABLE_CODE,
                BALL_SIGNAL_LOW_COVERAGE_CODE,
            }
        ]
        if risk:
            row["risk_codes"].append(risk)

    def _add_completeness(self, row: dict[str, Any]) -> None:
        config = self.safety_policy["feature_completeness"]
        weights = config["features"]
        available = 0.0
        total = 0.0
        for path, weight in weights.items():
            total += float(weight)
            if _nested(row["raw_features"], path) is not None:
                available += float(weight)
        score = available / total if total else 0.0
        critical = config["critical_by_event"][self.canonical_event_label]
        missing = [
            path
            for path in critical
            if _nested(row["raw_features"], path) is None
        ]
        if score >= float(config["ready_threshold"]) and not missing:
            state = "READY"
        elif score >= float(config["partial_threshold"]):
            state = "PARTIAL_FEATURES"
        else:
            state = "NO_RELIABLE_SHORTLIST"
        row["feature_completeness_score"] = score
        row["critical_features_missing"] = missing
        row["reliability_state"] = state
        if missing:
            row["risk_codes"].append("CRITICAL_FEATURES_MISSING")
