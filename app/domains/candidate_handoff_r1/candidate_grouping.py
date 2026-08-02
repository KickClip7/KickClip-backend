from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .artifacts import canonical_sha256
from .media_cache import SharedFrameCache
from .work_metrics import CandidatePreparationWorkMetrics


CANDIDATE_GROUPING_SCHEMA_VERSION = "kickclip.candidate_grouping.r1"
CANDIDATE_GROUPING_POLICY_VERSION = "STRICT_FRAGMENT_GROUPING_R1"


@dataclass(frozen=True)
class CandidateSignature:
    candidate_id: str
    source_candidate_id: str
    shot_id: str
    shortlist_rank: int
    global_rank: int
    observations: tuple[dict[str, Any], ...]
    histogram: np.ndarray

    @property
    def first_frame(self) -> int:
        return int(self.observations[0]["mapped_frame_index"])

    @property
    def last_frame(self) -> int:
        return int(self.observations[-1]["mapped_frame_index"])

    @property
    def by_frame(self) -> dict[int, dict[str, Any]]:
        return {
            int(row["mapped_frame_index"]): row
            for row in self.observations
        }


@dataclass(frozen=True)
class PairDecision:
    matched: bool
    reason_codes: tuple[str, ...]
    evidence: dict[str, Any]


def _clip_bbox(
    bbox: Iterable[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    values = [float(value) for value in bbox]
    if len(values) != 4:
        raise ValueError("bbox_xyxy must contain four values.")
    x1, y1, x2, y2 = values
    x1i = max(0, min(width - 1, int(math.floor(x1))))
    y1i = max(0, min(height - 1, int(math.floor(y1))))
    x2i = max(x1i + 1, min(width, int(math.ceil(x2))))
    y2i = max(y1i + 1, min(height, int(math.ceil(y2))))
    return x1i, y1i, x2i, y2i


def _histogram(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        crop,
        (64, 128),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256],
    ).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm > 0 else histogram


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(
        np.linalg.norm(left) * np.linalg.norm(right)
    )
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(1.0, x2 - x1) * max(1.0, y2 - y1)


def _bbox_height(bbox: Iterable[float]) -> float:
    values = [float(value) for value in bbox]
    return max(1.0, values[3] - values[1])


def _bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(
    left: Iterable[float],
    right: Iterable[float],
) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_distance(
    left: Iterable[float],
    right: Iterable[float],
) -> float:
    normalization = max(
        _bbox_height(left),
        _bbox_height(right),
        1.0,
    )
    return math.dist(
        _bbox_center(left),
        _bbox_center(right),
    ) / normalization


def _sample_observations(
    observations: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if len(observations) <= 3:
        return observations
    indices = sorted({0, len(observations) // 2, len(observations) - 1})
    return tuple(observations[index] for index in indices)


def _mapped_observations(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    frame_offset = int(candidate.get("frame_offset", 0))
    mapped: list[dict[str, Any]] = []
    for raw in candidate.get("observations") or []:
        mapped.append(
            {
                "mapped_frame_index": (
                    int(raw["frame_index"]) + frame_offset
                ),
                "source_frame_index": int(raw["frame_index"]),
                "bbox_xyxy": [
                    float(value)
                    for value in raw["bbox_xyxy"]
                ],
                "confidence": float(raw.get("confidence", 0.0)),
            }
        )
    mapped.sort(key=lambda row: int(row["mapped_frame_index"]))
    if not mapped:
        raise ValueError(
            f"Candidate has no observations: {candidate.get('candidate_id')}"
        )
    return tuple(mapped)


def _signature_histogram(
    *,
    observations: tuple[dict[str, Any], ...],
    frames: dict[int, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for observation in _sample_observations(observations):
        frame_index = int(observation["mapped_frame_index"])
        frame = frames.get(frame_index)
        if frame is None:
            continue
        x1, y1, x2, y2 = _clip_bbox(
            observation["bbox_xyxy"],
            width,
            height,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        values.append(_histogram(crop))
    if not values:
        return np.zeros((24 * 16,), dtype=np.float32)
    mean = np.mean(np.stack(values), axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


def _endpoint_velocity(
    observations: tuple[dict[str, Any], ...],
    *,
    from_end: bool,
) -> tuple[float, float]:
    if len(observations) < 2:
        return (0.0, 0.0)
    left, right = (
        observations[-2],
        observations[-1],
    ) if from_end else (
        observations[0],
        observations[1],
    )
    frame_delta = max(
        1,
        int(right["mapped_frame_index"])
        - int(left["mapped_frame_index"]),
    )
    left_center = _bbox_center(left["bbox_xyxy"])
    right_center = _bbox_center(right["bbox_xyxy"])
    return (
        (right_center[0] - left_center[0]) / frame_delta,
        (right_center[1] - left_center[1]) / frame_delta,
    )


def _pair_decision(
    left: CandidateSignature,
    right: CandidateSignature,
) -> PairDecision:
    if left.shot_id != right.shot_id:
        return PairDecision(False, (), {"reason": "DIFFERENT_SHOT"})

    appearance_similarity = _cosine(
        left.histogram,
        right.histogram,
    )
    left_by_frame = left.by_frame
    right_by_frame = right.by_frame
    common_frames = sorted(set(left_by_frame) & set(right_by_frame))

    if common_frames:
        ious = [
            _bbox_iou(
                left_by_frame[frame]["bbox_xyxy"],
                right_by_frame[frame]["bbox_xyxy"],
            )
            for frame in common_frames
        ]
        center_distances = [
            _normalized_center_distance(
                left_by_frame[frame]["bbox_xyxy"],
                right_by_frame[frame]["bbox_xyxy"],
            )
            for frame in common_frames
        ]
        median_iou = float(np.median(ious))
        minimum_iou = float(min(ious))
        maximum_center_distance = float(max(center_distances))
        matched = bool(
            median_iou >= 0.72
            and minimum_iou >= 0.45
            and maximum_center_distance <= 0.45
            and appearance_similarity >= 0.86
        )
        return PairDecision(
            matched,
            ("OVERLAPPING_DUPLICATE_GEOMETRY",) if matched else (),
            {
                "comparison_mode": "OVERLAPPING_FRAMES",
                "common_frame_count": len(common_frames),
                "median_iou": round(median_iou, 6),
                "minimum_iou": round(minimum_iou, 6),
                "maximum_normalized_center_distance": round(
                    maximum_center_distance,
                    6,
                ),
                "appearance_similarity": round(
                    appearance_similarity,
                    6,
                ),
            },
        )

    first, second = (
        (left, right)
        if left.last_frame < right.first_frame
        else (right, left)
    )
    gap_frames = second.first_frame - first.last_frame - 1
    if gap_frames < 0 or gap_frames > 3:
        return PairDecision(
            False,
            (),
            {
                "comparison_mode": "DISJOINT_FRAMES",
                "gap_frames": gap_frames,
                "appearance_similarity": round(
                    appearance_similarity,
                    6,
                ),
            },
        )

    first_last = first.observations[-1]
    second_first = second.observations[0]
    endpoint_distance = _normalized_center_distance(
        first_last["bbox_xyxy"],
        second_first["bbox_xyxy"],
    )
    scale_jump = abs(
        math.log(
            _bbox_area(second_first["bbox_xyxy"])
            / _bbox_area(first_last["bbox_xyxy"])
        )
    )
    velocity = _endpoint_velocity(
        first.observations,
        from_end=True,
    )
    frame_delta = max(
        1,
        second.first_frame - first.last_frame,
    )
    first_center = _bbox_center(first_last["bbox_xyxy"])
    predicted_center = (
        first_center[0] + velocity[0] * frame_delta,
        first_center[1] + velocity[1] * frame_delta,
    )
    actual_center = _bbox_center(second_first["bbox_xyxy"])
    predicted_distance = math.dist(
        predicted_center,
        actual_center,
    ) / max(
        _bbox_height(first_last["bbox_xyxy"]),
        _bbox_height(second_first["bbox_xyxy"]),
        1.0,
    )

    matched = bool(
        appearance_similarity >= 0.92
        and endpoint_distance <= 0.75
        and predicted_distance <= 0.90
        and scale_jump <= 0.45
    )
    return PairDecision(
        matched,
        ("ADJACENT_FRAGMENT_CONTINUITY",) if matched else (),
        {
            "comparison_mode": "ADJACENT_FRAGMENTS",
            "gap_frames": gap_frames,
            "endpoint_normalized_center_distance": round(
                endpoint_distance,
                6,
            ),
            "predicted_normalized_center_distance": round(
                predicted_distance,
                6,
            ),
            "scale_jump": round(scale_jump, 6),
            "appearance_similarity": round(
                appearance_similarity,
                6,
            ),
        },
    )


def build_candidate_grouping(
    *,
    video_path: Path,
    source_video_sha256: str,
    shortlist_patch_id: str,
    ranking_id: str,
    candidates: list[dict[str, Any]],
    shared_frame_cache: SharedFrameCache,
    metrics: CandidatePreparationWorkMetrics | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Candidate grouping requires candidates.")

    mapped_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    sample_frames: set[int] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in mapped_by_id:
            raise ValueError("Candidate grouping IDs must be unique.")
        observations = _mapped_observations(candidate)
        mapped_by_id[candidate_id] = observations
        sample_frames.update(
            int(row["mapped_frame_index"])
            for row in _sample_observations(observations)
        )

    cache_result = shared_frame_cache.load_frames(
        video_path=video_path,
        video_sha256=source_video_sha256,
        frame_indices=sorted(sample_frames),
        metrics=metrics,
    )

    signatures: list[CandidateSignature] = []
    for candidate in sorted(
        candidates,
        key=lambda row: int(row["shortlist_rank"]),
    ):
        candidate_id = str(candidate["candidate_id"])
        observations = mapped_by_id[candidate_id]
        signatures.append(
            CandidateSignature(
                candidate_id=candidate_id,
                source_candidate_id=str(
                    candidate["source_candidate_id"]
                ),
                shot_id=str(candidate["shot_id"]),
                shortlist_rank=int(candidate["shortlist_rank"]),
                global_rank=int(candidate["global_rank"]),
                observations=observations,
                histogram=_signature_histogram(
                    observations=observations,
                    frames=cache_result.frames,
                    width=cache_result.width,
                    height=cache_result.height,
                ),
            )
        )

    groups: list[dict[str, Any]] = []
    signature_by_id = {
        signature.candidate_id: signature
        for signature in signatures
    }

    for signature in signatures:
        assigned = False
        for group in groups:
            member_ids = list(group["member_candidate_ids"])
            pair_decisions: list[tuple[str, PairDecision]] = []
            for member_id in member_ids:
                if metrics is not None:
                    metrics.increment("candidate_group_pair_comparisons")
                decision = _pair_decision(
                    signature,
                    signature_by_id[member_id],
                )
                pair_decisions.append((member_id, decision))
            if not pair_decisions or not all(
                decision.matched
                for _, decision in pair_decisions
            ):
                continue

            group["member_candidate_ids"].append(signature.candidate_id)
            group["member_source_candidate_ids"].append(
                signature.source_candidate_id
            )
            group["evidence"].extend(
                {
                    "left_candidate_id": member_id,
                    "right_candidate_id": signature.candidate_id,
                    "reason_codes": list(decision.reason_codes),
                    **decision.evidence,
                }
                for member_id, decision in pair_decisions
            )
            assigned = True
            if metrics is not None:
                metrics.increment("candidate_group_pair_matches")
            break

        if not assigned:
            groups.append(
                {
                    "representative_candidate_id": signature.candidate_id,
                    "representative_source_candidate_id": (
                        signature.source_candidate_id
                    ),
                    "member_candidate_ids": [signature.candidate_id],
                    "member_source_candidate_ids": [
                        signature.source_candidate_id
                    ],
                    "shot_id": signature.shot_id,
                    "representative_shortlist_rank": (
                        signature.shortlist_rank
                    ),
                    "representative_global_rank": signature.global_rank,
                    "evidence": [],
                }
            )

    normalized_groups: list[dict[str, Any]] = []
    for group in groups:
        member_ids = list(group["member_candidate_ids"])
        group_id = "cgrp_" + canonical_sha256(
            {
                "policy": CANDIDATE_GROUPING_POLICY_VERSION,
                "shortlist_patch_id": shortlist_patch_id,
                "member_candidate_ids": member_ids,
            }
        )[:20]
        reason_codes = sorted(
            {
                code
                for evidence in group["evidence"]
                for code in evidence.get("reason_codes") or []
            }
        )
        normalized_groups.append(
            {
                **group,
                "candidate_group_id": group_id,
                "member_candidate_count": len(member_ids),
                "possible_fragment_duplicate": len(member_ids) > 1,
                "grouping_reason_codes": reason_codes,
            }
        )

    source_count = len(signatures)
    display_count = len(normalized_groups)
    if metrics is not None:
        metrics.increment("candidate_group_count", display_count)
        metrics.increment(
            "grouped_candidate_suppressed_count",
            source_count - display_count,
        )
        metrics.increment(
            "candidate_group_sample_frame_count",
            len(sample_frames),
        )

    return {
        "schema_version": CANDIDATE_GROUPING_SCHEMA_VERSION,
        "policy_version": CANDIDATE_GROUPING_POLICY_VERSION,
        "immutable": True,
        "ranking_id": ranking_id,
        "shortlist_patch_id": shortlist_patch_id,
        "source_video_sha256": source_video_sha256,
        "source_candidate_count": source_count,
        "display_candidate_count": display_count,
        "suppressed_fragment_candidate_count": (
            source_count - display_count
        ),
        "groups": normalized_groups,
        "automatic_target_confirmation": False,
        "grouping_is_identity_confirmation": False,
        "full_gallery_fallback_required": True,
    }
