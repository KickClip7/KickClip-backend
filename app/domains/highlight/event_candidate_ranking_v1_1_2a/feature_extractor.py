from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
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
    _center,
    _direction_changes,
    _mean,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.feature_extractor import (
    RawFeatureExtractorV111,
    _valid_ball,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.frame_reader import (
    BoundedVideoFrameReader,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.contract import (
    ShotInterval,
    shot_for_frame,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.feature_extractor import (
    RawFeatureExtractorV112,
)


@dataclass(frozen=True)
class ShotBallSelection:
    detections: dict[tuple[str, int], Detection]
    segment_ids: dict[tuple[str, int], int]
    reset_count: int


def select_ball_trajectory_segments_by_shot(
    detections: tuple[Detection, ...],
    *,
    intervals: tuple[ShotInterval, ...],
    width: int,
    height: int,
    policy: dict[str, Any],
) -> ShotBallSelection:
    grouped: dict[tuple[str, int], list[Detection]] = defaultdict(list)
    for detection in detections:
        shot_id = shot_for_frame(intervals, detection.frame)
        if shot_id is None:
            continue
        if _valid_ball(
            detection,
            width=width,
            height=height,
            minimum_confidence=float(policy["minimum_confidence"]),
            min_area_ratio=float(policy["min_area_ratio"]),
            max_area_ratio=float(policy["max_area_ratio"]),
        ):
            grouped[(shot_id, detection.frame)].append(detection)
    selected: dict[tuple[str, int], Detection] = {}
    segment_ids: dict[tuple[str, int], int] = {}
    diagonal = math.hypot(width, height)
    reset_count = 0
    for interval in intervals:
        previous: Detection | None = None
        previous_frame: int | None = None
        segment_id = 0
        frames = sorted(
            frame
            for shot_id, frame in grouped
            if shot_id == interval.shot_id
        )
        for frame in frames:
            rows = grouped[(interval.shot_id, frame)]
            gap_reset = (
                previous is None
                or previous_frame is None
                or frame - previous_frame
                > int(policy["continuity_reset_gap_frames"])
            )
            if gap_reset:
                if previous is not None:
                    segment_id += 1
                    reset_count += 1
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
            key = (interval.shot_id, frame)
            selected[key] = winner
            segment_ids[key] = segment_id
            previous = winner
            previous_frame = frame
    return ShotBallSelection(selected, segment_ids, reset_count)


class RawFeatureExtractorV112a(RawFeatureExtractorV112):
    def extract_all(
        self,
        *,
        candidates: tuple[CandidateSequence, ...],
        detections: tuple[Detection, ...],
        video_path: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        by_frame: dict[
            int, list[tuple[str, CandidateObservation]]
        ] = defaultdict(list)
        for candidate in candidates:
            for observation in candidate.observations:
                by_frame[observation.global_frame].append(
                    (candidate.candidate_id, observation)
                )
        matching = tuple(
            detection
            for detection in detections
            if detection.label in self.ball_labels
        )
        selection = select_ball_trajectory_segments_by_shot(
            matching,
            intervals=self.shot_intervals,
            width=self.width,
            height=self.height,
            policy=self.safety_policy["ball_selection"],
        )
        flat_by_frame = {
            frame: [detection]
            for (_, frame), detection in selection.detections.items()
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
                    balls_by_frame=flat_by_frame,
                    shot_balls=selection.detections,
                    ball_segment_ids=selection.segment_ids,
                    reader=reader,
                )
                for candidate in candidates
            ]
            decoding = reader.stats()
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
        return rows, {
            "candidate_count": len(rows),
            "actual_candidate_observations_available": sum(
                len(candidate.observations) for candidate in candidates
            ),
            "group_available_candidate_counts": group_counts,
            "group_coverage": {
                group: group_counts[group] / len(rows) if rows else None
                for group in groups
            },
            "ball_detection_audit": {
                "raw_matching_detection_count": len(matching),
                "selected_valid_shot_frame_count": len(
                    selection.detections
                ),
                "key_contract": "shot_id+frame",
                "trajectory_reset_at_shot_boundary": True,
                "continuity_gap_reset_count": selection.reset_count,
                "one_detection_per_shot_frame": True,
            },
            "frame_decoding": decoding,
            "reliability_counts": {
                state: sum(
                    row["reliability_state"] == state for row in rows
                )
                for state in (
                    "READY",
                    "PARTIAL_FEATURES",
                    "NO_RELIABLE_SHORTLIST",
                )
            },
        }

    def extract_candidate(
        self,
        candidate: CandidateSequence,
        **kwargs: Any,
    ) -> dict[str, Any]:
        shot_balls = kwargs.pop("shot_balls")
        segment_ids = kwargs.pop("ball_segment_ids")
        row = RawFeatureExtractorV111.extract_candidate(
            self,
            candidate,
            **kwargs,
        )
        self._correct_ball_v112a(
            row,
            candidate,
            shot_balls,
            segment_ids,
        )
        broadcast = row["raw_features"]["broadcast"]
        broadcast["post_event_shots_with_appearance"] = None
        broadcast["post_event_shots_with_appearance_deprecated"] = {
            "deprecated": True,
            "reason": "SHOT_LOCAL_CANDIDATE_HAS_NO_CROSS_SHOT_IDENTITY",
        }
        row["feature_availability"]["broadcast"][
            "post_event_shots_with_appearance_available"
        ] = False
        row["feature_availability"]["broadcast"][
            "post_event_shots_with_appearance_reason"
        ] = "SHOT_LOCAL_CANDIDATE_HAS_NO_CROSS_SHOT_IDENTITY"
        self._add_completeness(row)
        row["risk_codes"] = sorted(set(row["risk_codes"]))
        return row

    def _correct_ball_v112a(
        self,
        row: dict[str, Any],
        candidate: CandidateSequence,
        shot_balls: dict[tuple[str, int], Detection],
        segment_ids: dict[tuple[str, int], int],
    ) -> None:
        policy = self.safety_policy["ball_selection"]
        diagonal = math.hypot(self.width, self.height)
        evidence: list[
            tuple[CandidateObservation, Detection, float, int]
        ] = []
        for observation in candidate.observations:
            key = (candidate.shot_id, observation.global_frame)
            ball = shot_balls.get(key)
            if ball is None:
                continue
            foot = np.asarray(
                [
                    (
                        observation.bbox_xyxy[0]
                        + observation.bbox_xyxy[2]
                    )
                    / 2,
                    observation.bbox_xyxy[1]
                    + 0.85
                    * (
                        observation.bbox_xyxy[3]
                        - observation.bbox_xyxy[1]
                    ),
                ]
            )
            distance = float(
                np.linalg.norm(_center(ball.bbox_xyxy) - foot) / diagonal
            )
            evidence.append(
                (observation, ball, distance, segment_ids[key])
            )
        coverage = len(evidence) / len(candidate.observations)
        passing_pairs = 0
        valid_pairs = 0
        reset_gap_pairs = 0
        vectors: list[np.ndarray] = []
        speeds: list[float] = []
        for previous, current in zip(evidence, evidence[1:]):
            frame_gap = current[0].global_frame - previous[0].global_frame
            if current[3] != previous[3]:
                reset_gap_pairs += 1
                continue
            valid_pairs += 1
            frame_delta = max(1, frame_gap)
            displacement = float(
                np.linalg.norm(
                    _center(current[1].bbox_xyxy)
                    - _center(previous[1].bbox_xyxy)
                )
                / diagonal
                / frame_delta
            )
            passing_pairs += int(
                displacement
                <= float(
                    policy["max_normalized_displacement_per_frame"]
                )
            )
            time_delta = (
                current[0].scene_local_time_sec
                - previous[0].scene_local_time_sec
            )
            if time_delta > 0:
                vector = (
                    _center(current[1].bbox_xyxy)
                    - _center(previous[1].bbox_xyxy)
                ) / np.asarray([self.width, self.height]) / time_delta
                vectors.append(vector)
                speeds.append(float(np.linalg.norm(vector)))
        continuity = passing_pairs / valid_pairs if valid_pairs else None
        if not shot_balls:
            state = BALL_SIGNAL_UNAVAILABLE
            risk = BALL_SIGNAL_UNAVAILABLE_CODE
        elif (
            coverage < float(policy["minimum_candidate_coverage"])
            or continuity is None
            or continuity
            < float(policy["minimum_trajectory_continuity"])
        ):
            state = BALL_SIGNAL_LOW_COVERAGE
            risk = BALL_SIGNAL_LOW_COVERAGE_CODE
        else:
            state = BALL_SIGNAL_AVAILABLE
            risk = None
        event_near = [
            item
            for item in evidence
            if candidate.shot_id == self.event_shot_id
            and abs(item[0].scene_local_time_sec - self.event_time)
            <= float(policy["event_near_window_sec"])
        ]
        if event_near:
            event_segment = min(
                event_near,
                key=lambda item: abs(
                    item[0].scene_local_time_sec - self.event_time
                ),
            )[3]
            event_near = [
                item for item in event_near if item[3] == event_segment
            ]
        pre_event = [
            item
            for item in event_near
            if item[0].scene_local_time_sec <= self.event_time
        ]
        distances = [item[2] for item in event_near]
        available = state == BALL_SIGNAL_AVAILABLE
        trend = None
        if available and len(pre_event) >= 2:
            times = np.asarray(
                [item[0].scene_local_time_sec for item in pre_event]
            )
            values = np.asarray([item[2] for item in pre_event])
            centered_times = times - float(np.mean(times))
            denominator = float(np.dot(centered_times, centered_times))
            if denominator > 0:
                trend = float(
                    np.dot(
                        centered_times,
                        values - float(np.mean(values)),
                    )
                    / denominator
                )
        median_distance = (
            float(np.median(distances))
            if available and distances
            else None
        )
        row["raw_features"]["ball"] = {
            "state": state,
            "candidate_frame_coverage": coverage
            if shot_balls
            else None,
            "selected_ball_observation_count": len(evidence),
            "valid_continuity_pair_count": valid_pairs,
            "reset_gap_pair_count": reset_gap_pairs,
            "continuity_passing_pair_count": passing_pairs,
            "trajectory_continuity": continuity
            if shot_balls
            else None,
            "event_near_min_foot_distance": (
                min(distances) if available and distances else None
            ),
            "event_near_median_foot_distance": median_distance,
            "pre_event_distance_trend": trend if available else None,
            "velocity": _mean(speeds) if available else None,
            "direction_change": (
                _direction_changes(vectors) if available else None
            ),
            "coverage": coverage if shot_balls else None,
            "foot_region_distance": median_distance,
            "pre_event_distance_change": trend if available else None,
        }
        row["feature_availability"]["ball"] = {
            "available": available,
            "candidate_frame_coverage": coverage
            if shot_balls
            else None,
            "trajectory_continuity": continuity
            if shot_balls
            else None,
            "source": (
                "detections_artifact:"
                "selected_by_shot_id+frame+reset_segment"
            ),
            "state": state,
        }
        row["feature_evidence"]["ball_shot_frames"] = [
            {
                "shot_id": candidate.shot_id,
                "global_frame": item[0].global_frame,
                "trajectory_segment_id": item[3],
            }
            for item in evidence
        ]
        row["feature_evidence"]["ball_frames"] = [
            item[0].global_frame for item in evidence
        ]
        row["risk_codes"] = [
            code
            for code in row["risk_codes"]
            if code
            not in {
                BALL_SIGNAL_UNAVAILABLE_CODE,
                BALL_SIGNAL_LOW_COVERAGE_CODE,
            }
        ]
        if risk:
            row["risk_codes"].append(risk)
