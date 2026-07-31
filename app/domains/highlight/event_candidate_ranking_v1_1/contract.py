from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_NAME = "target_centric_tracking_event_candidate_ranking_v1_1"
STATUS = "PROVISIONAL_SHADOW_ONLY"
UNSUPPORTED_EVENT_CLASS = "UNSUPPORTED_EVENT_CLASS"

_CANONICAL_EVENT_ALIASES = {
    "goal": "goal",
    "goals": "goal",
    "shot": "shot",
    "shots": "shot",
    "shot_on_target": "shot",
    "shot_off_target": "shot",
    "free_kick": "free_kick",
    "freekick": "free_kick",
    "free kick": "free_kick",
    "corner": "corner",
    "corner_kick": "corner",
    "corner kick": "corner",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_event_label(value: str) -> str | None:
    normalized = " ".join(value.strip().lower().replace("-", "_").split())
    return _CANONICAL_EVENT_ALIASES.get(normalized)


def validated_input_file(
    root: Path,
    relative_path: str,
    *,
    expected_sha256: str,
    allowed_suffixes: set[str],
) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("Immutable input path must be relative.")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("Immutable input path escapes discovery root.")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("Immutable input must be a regular non-symlink file.")
    if resolved.suffix.lower() not in allowed_suffixes:
        raise ValueError("Immutable input suffix is not allowed.")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError("Immutable input SHA-256 mismatch.")
    return resolved


@dataclass(frozen=True)
class CandidateObservation:
    global_frame: int
    scene_local_frame: int
    scene_local_time_sec: float
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float | None
    detection_id: str
    shot_id: str


@dataclass(frozen=True)
class CandidateSequence:
    candidate_id: str
    shot_id: str
    shot_index: int
    local_tracklet_id: str
    trackability_score: float
    observations: tuple[CandidateObservation, ...]


@dataclass(frozen=True)
class ImmutableCandidateInput:
    discovery_root: Path
    scene_candidates_path: Path
    detections_path: Path
    source_video_path: Path
    shot_boundaries_path: Path
    scene_candidates_sha256: str
    detections_sha256: str
    source_video_sha256: str
    shot_boundaries_sha256: str
    candidate_manifest_sha256: str
    video_width: int
    video_height: int
    video_fps: float
    video_frame_count: int
    candidates: tuple[CandidateSequence, ...]

    @classmethod
    def load(
        cls,
        *,
        discovery_root: Path,
        scene_candidates_relative_path: str,
        scene_candidates_sha256: str,
        detections_relative_path: str,
        detections_sha256: str,
        source_video_relative_path: str,
        source_video_sha256: str,
        shot_boundaries_relative_path: str,
        shot_boundaries_sha256: str,
        candidate_manifest_sha256: str,
        video_width: int,
        video_height: int,
        video_fps: float,
        video_frame_count: int,
    ) -> "ImmutableCandidateInput":
        candidates_path = validated_input_file(
            discovery_root,
            scene_candidates_relative_path,
            expected_sha256=scene_candidates_sha256,
            allowed_suffixes={".json"},
        )
        detections_path = validated_input_file(
            discovery_root,
            detections_relative_path,
            expected_sha256=detections_sha256,
            allowed_suffixes={".json", ".jsonl", ".csv"},
        )
        video_path = validated_input_file(
            discovery_root,
            source_video_relative_path,
            expected_sha256=source_video_sha256,
            allowed_suffixes={".mp4", ".mov", ".mkv", ".avi", ".webm"},
        )
        boundaries_path = validated_input_file(
            discovery_root,
            shot_boundaries_relative_path,
            expected_sha256=shot_boundaries_sha256,
            allowed_suffixes={".json", ".csv"},
        )
        document = json.loads(candidates_path.read_text(encoding="utf-8"))
        sequences = parse_candidate_sequences(
            document,
            fps=video_fps,
            frame_count=video_frame_count,
        )
        return cls(
            discovery_root=discovery_root.resolve(),
            scene_candidates_path=candidates_path,
            detections_path=detections_path,
            source_video_path=video_path,
            shot_boundaries_path=boundaries_path,
            scene_candidates_sha256=scene_candidates_sha256,
            detections_sha256=detections_sha256,
            source_video_sha256=source_video_sha256,
            shot_boundaries_sha256=shot_boundaries_sha256,
            candidate_manifest_sha256=candidate_manifest_sha256,
            video_width=video_width,
            video_height=video_height,
            video_fps=video_fps,
            video_frame_count=video_frame_count,
            candidates=sequences,
        )


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        value = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Candidate observation bbox_xyxy must contain four values.")
    result = tuple(float(item) for item in value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Candidate observation bbox is invalid.")
    return result


def parse_candidate_sequences(
    document: dict[str, Any],
    *,
    fps: float,
    frame_count: int,
) -> tuple[CandidateSequence, ...]:
    if document.get("candidates") is None or not isinstance(
        document["candidates"], list
    ):
        raise ValueError("scene_candidates.json candidates must be an array.")
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Video FPS and frame count must be positive.")
    parsed: list[CandidateSequence] = []
    seen_ids: set[str] = set()
    for candidate in document["candidates"]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_ids:
            raise ValueError("Candidate IDs must be present and unique.")
        seen_ids.add(candidate_id)
        shot_id = str(candidate.get("shot_id") or "")
        raw_observations = candidate.get("observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise ValueError(
                f"Candidate {candidate_id} has no observation sequence."
            )
        observations: list[CandidateObservation] = []
        for index, row in enumerate(raw_observations):
            global_frame = _first(
                row, "global_frame", "global_frame_index", "source_frame_index"
            )
            local_frame = _first(
                row, "scene_local_frame", "scene_local_frame_index", "frame_index"
            )
            if global_frame is None and local_frame is None:
                raise ValueError("Observation requires global or scene-local frame.")
            global_frame = int(
                global_frame if global_frame is not None else local_frame
            )
            local_frame = int(
                local_frame if local_frame is not None else global_frame
            )
            if global_frame < 0 or global_frame >= frame_count or local_frame < 0:
                raise ValueError("Observation frame is outside the video contract.")
            local_time = _first(
                row, "scene_local_time_sec", "local_time_sec", "time_sec"
            )
            local_time = (
                float(local_time) if local_time is not None else local_frame / fps
            )
            observations.append(
                CandidateObservation(
                    global_frame=global_frame,
                    scene_local_frame=local_frame,
                    scene_local_time_sec=local_time,
                    bbox_xyxy=_bbox(_first(row, "bbox_xyxy", "bbox")),
                    detector_confidence=(
                        float(_first(row, "detector_confidence", "confidence"))
                        if _first(row, "detector_confidence", "confidence") is not None
                        else None
                    ),
                    detection_id=str(
                        _first(row, "detection_id", "observation_id")
                        or f"{candidate_id}:{index}"
                    ),
                    shot_id=str(row.get("shot_id") or shot_id),
                )
            )
        observations.sort(key=lambda item: item.scene_local_time_sec)
        quality = candidate.get("quality") or {}
        parsed.append(
            CandidateSequence(
                candidate_id=candidate_id,
                shot_id=shot_id,
                shot_index=int(candidate.get("shot_index") or 0),
                local_tracklet_id=str(
                    candidate.get("local_tracklet_id")
                    or candidate.get("tracklet_id")
                    or candidate_id
                ),
                trackability_score=float(
                    quality.get("trackability_score")
                    if quality.get("trackability_score") is not None
                    else candidate.get("trackability_score") or 0.0
                ),
                observations=tuple(observations),
            )
        )
    return tuple(parsed)
