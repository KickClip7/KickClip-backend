from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.artifact.repository import ArtifactRepository
from app.domains.highlight.detection_cache import (
    DETECTION_CACHE_POLICY_VERSION,
    PlayerDetectionCache,
)
from app.domains.highlight.model import HighlightRevision, ScenePlayerCandidate
from app.domains.highlight.player_detector import (
    PlayerDetector,
    PlayerDetectorError,
    PlayerDetectorInferenceError,
    create_player_detector,
)
from app.domains.highlight.repository import HighlightRepository
from app.domains.media.model import MediaAsset
from app.domains.timeline.model import TimelineEvent
from app.domains.tracking.errors import (
    TrackingInputError,
    TrackingNoValidInitializationAnchorError,
)
from app.domains.tracking.input_contract import MIN_TRACKING_DURATION_SECONDS
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id


@dataclass
class CandidateTracklet:
    observations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_bbox(self) -> list[float]:
        return self.observations[-1]["bbox"]


class PlayerCandidateDiscoveryService:
    """RF-DETR-backed discovery of scene-local, unnamed player candidates."""

    CANDIDATE_ROLE_FILTER_POLICY_VERSION = "PLAYER_GOALKEEPER_ONLY_R1"
    ALLOWED_CANDIDATE_CLASS_NAMES = frozenset({"player", "goalkeeper"})

    SAMPLE_WINDOW_COUNT = 5
    SAMPLES_PER_WINDOW = 3
    STABLE_MAX_GAP_SECONDS = 0.2
    MIN_STABLE_OBSERVATIONS = 3
    MIN_BOX_HEIGHT_RATIO = 0.035
    MIN_ANCHOR_CONFIDENCE = 0.25
    MAX_ANCHOR_OVERLAP_IOU = 0.6
    BBOX_EDGE_MARGIN_PX = 1.0
    MAX_CANDIDATES_PER_SCENE = 8

    def __init__(
        self,
        db: Session,
        *,
        detector: PlayerDetector | None = None,
        settings: Settings | None = None,
    ):
        self.db = db
        self.settings = settings or get_settings()
        self.detector = detector
        self.storage = LocalStorage()
        self.repository = HighlightRepository(db)
        self.artifact_repository = ArtifactRepository(db)
        self.detection_cache = PlayerDetectionCache(
            self.storage.storage_root / "player_detection_cache_r1"
        )

    def discover(
        self,
        *,
        revision: HighlightRevision,
        source_asset: MediaAsset,
        scenes: list[TimelineEvent],
    ) -> list[ScenePlayerCandidate]:
        # Lazy imports keep API startup and non-vision tests independent of the
        # optional RF-DETR/OpenCV runtime.
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for player candidate discovery."
            ) from exc

        detector = self.detector or create_player_detector(self.settings)

        source_path = self.storage.resolve_path(source_asset.file_path)
        if not source_path.is_file():
            raise FileNotFoundError("Source MediaAsset file is missing.")
        source_video_sha256 = self._source_video_sha256(
            source_asset=source_asset,
            source_path=source_path,
        )
        cache_totals = {
            "requested_frame_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "inference_frame_count": 0,
            "cache_write_count": 0,
        }
        candidate_filter_totals = {
            "raw_detection_count": 0,
            "allowed_role_detection_count": 0,
            "eligible_candidate_detection_count": 0,
            "excluded_non_target_role_count": 0,
            "excluded_too_small_count": 0,
        }
        excluded_role_totals: dict[str, int] = {}

        revision.options = {
            **(revision.options or {}),
            "candidate_discovery": {
                "status": "RUNNING",
                "backend": detector.runtime_metadata.get("backend"),
                "runtime": detector.runtime_metadata,
                "selected_scene_count": len(scenes),
                "source_video_sha256": source_video_sha256,
                "detection_cache_policy": DETECTION_CACHE_POLICY_VERSION,
                "candidate_role_filter": {
                    "policy_version": self.CANDIDATE_ROLE_FILTER_POLICY_VERSION,
                    "allowed_class_names": sorted(
                        self.ALLOWED_CANDIDATE_CLASS_NAMES
                    ),
                },
            },
        }

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError("Unable to open source video for candidate discovery.")
        fps = float(source_asset.fps or capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            capture.release()
            raise ValueError("Source video FPS is unavailable.")

        created: list[ScenePlayerCandidate] = []
        try:
            for scene in scenes:
                scene_candidates, scene_cache = self._discover_scene(
                    cv2=cv2,
                    capture=capture,
                    detector=detector,
                    fps=fps,
                    revision=revision,
                    source_asset=source_asset,
                    source_video_sha256=source_video_sha256,
                    scene=scene,
                )
                created.extend(scene_candidates)
                cache_metrics = dict(scene_cache.get("detection_cache") or {})
                filter_metrics = dict(scene_cache.get("candidate_role_filter") or {})
                for key in cache_totals:
                    cache_totals[key] += int(cache_metrics.get(key, 0))
                for key in candidate_filter_totals:
                    candidate_filter_totals[key] += int(
                        filter_metrics.get(key, 0)
                    )
                for class_name, count in dict(
                    filter_metrics.get("excluded_role_counts") or {}
                ).items():
                    normalized = str(class_name or "unknown")
                    excluded_role_totals[normalized] = (
                        excluded_role_totals.get(normalized, 0) + int(count)
                    )
        except PlayerDetectorError:
            raise
        except TrackingInputError:
            raise
        except Exception as exc:
            raise PlayerDetectorInferenceError(
                "Player candidate discovery failed after detector "
                f"initialization: {type(exc).__name__}: {exc}",
                diagnostics=detector.runtime_metadata,
            ) from exc
        finally:
            capture.release()

        if not created:
            raise TrackingNoValidInitializationAnchorError(
                (
                    "No earliest stable player anchor leaves at least "
                    f"{MIN_TRACKING_DURATION_SECONDS:.0f}s in the selected scenes."
                ),
                diagnostics={
                    "selected_scene_ids": [
                        scene.timeline_event_id for scene in scenes
                    ],
                    "minimum_tracking_duration_sec": (
                        MIN_TRACKING_DURATION_SECONDS
                    ),
                    "selection_policy": "earliest_stable_anchor",
                    "detection_cache": {
                        "policy_version": DETECTION_CACHE_POLICY_VERSION,
                        **cache_totals,
                    },
                    "candidate_role_filter": {
                        "policy_version": (
                            self.CANDIDATE_ROLE_FILTER_POLICY_VERSION
                        ),
                        "allowed_class_names": sorted(
                            self.ALLOWED_CANDIDATE_CLASS_NAMES
                        ),
                        **candidate_filter_totals,
                        "excluded_role_counts": dict(
                            sorted(excluded_role_totals.items())
                        ),
                    },
                },
            )

        revision.options = {
            **(revision.options or {}),
            "candidate_discovery": {
                "status": "COMPLETED",
                "backend": detector.runtime_metadata.get("backend"),
                "runtime": detector.runtime_metadata,
                "selected_scene_count": len(scenes),
                "candidate_count": len(created),
                "source_video_sha256": source_video_sha256,
                "candidate_role_filter": {
                    "policy_version": self.CANDIDATE_ROLE_FILTER_POLICY_VERSION,
                    "allowed_class_names": sorted(
                        self.ALLOWED_CANDIDATE_CLASS_NAMES
                    ),
                    **candidate_filter_totals,
                    "excluded_role_counts": dict(
                        sorted(excluded_role_totals.items())
                    ),
                },
                "detection_cache": {
                    "policy_version": DETECTION_CACHE_POLICY_VERSION,
                    **cache_totals,
                    "cache_hit_ratio": (
                        round(
                            cache_totals["cache_hits"]
                            / cache_totals["requested_frame_count"],
                            6,
                        )
                        if cache_totals["requested_frame_count"] > 0
                        else 0.0
                    ),
                },
            },
        }
        return created

    def _discover_scene(
        self,
        *,
        cv2,
        capture,
        detector: PlayerDetector,
        fps: float,
        revision: HighlightRevision,
        source_asset: MediaAsset,
        source_video_sha256: str,
        scene: TimelineEvent,
    ) -> tuple[list[ScenePlayerCandidate], dict[str, Any]]:
        sample_times = self._sample_times(scene, fps=fps)
        tracklets: list[CandidateTracklet] = []
        sampled_frames: dict[float, Any] = {}
        sampled_frame_indices: dict[float, int] = {}

        for source_time in sample_times:
            frame_index = max(0, round(source_time * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            sampled_frames[source_time] = frame
            sampled_frame_indices[source_time] = frame_index

        ordered_times = list(sampled_frames)
        if not ordered_times:
            return [], {
                "detection_cache": {
                    "requested_frame_count": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "inference_frame_count": 0,
                    "cache_write_count": 0,
                },
                "candidate_role_filter": {
                    "policy_version": (
                        self.CANDIDATE_ROLE_FILTER_POLICY_VERSION
                    ),
                    "allowed_class_names": sorted(
                        self.ALLOWED_CANDIDATE_CLASS_NAMES
                    ),
                    "raw_detection_count": 0,
                    "allowed_role_detection_count": 0,
                    "eligible_candidate_detection_count": 0,
                    "excluded_non_target_role_count": 0,
                    "excluded_too_small_count": 0,
                    "excluded_role_counts": {},
                },
            }

        cache_result = self.detection_cache.detect_batch(
            source_video_sha256=source_video_sha256,
            frame_indices=[
                sampled_frame_indices[source_time]
                for source_time in ordered_times
            ],
            frames_bgr=[
                sampled_frames[source_time]
                for source_time in ordered_times
            ],
            detector=detector,
        )
        detections_by_frame = cache_result.detections_by_frame
        if len(detections_by_frame) != len(ordered_times):
            raise RuntimeError(
                "Player detector output count does not match sampled frames."
            )

        scene_filter_totals = {
            "raw_detection_count": 0,
            "allowed_role_detection_count": 0,
            "eligible_candidate_detection_count": 0,
            "excluded_non_target_role_count": 0,
            "excluded_too_small_count": 0,
        }
        scene_excluded_role_counts: dict[str, int] = {}

        for source_time, detections in zip(
            ordered_times,
            detections_by_frame,
        ):
            frame = sampled_frames[source_time]
            eligible, filter_metrics = self._filter_candidate_detections(
                detections,
                frame_height=int(frame.shape[0]),
            )
            for key in scene_filter_totals:
                scene_filter_totals[key] += int(filter_metrics.get(key, 0))
            for class_name, count in dict(
                filter_metrics.get("excluded_role_counts") or {}
            ).items():
                normalized = str(class_name or "unknown")
                scene_excluded_role_counts[normalized] = (
                    scene_excluded_role_counts.get(normalized, 0)
                    + int(count)
                )
            for detection in eligible:
                bbox = detection.bbox_xyxy
                score = detection.confidence
                max_overlap = max(
                    (
                        self._iou(bbox, other.bbox_xyxy)
                        for other in eligible
                        if other is not detection
                    ),
                    default=0.0,
                )
                observation = {
                    "source_time": source_time,
                    "source_frame_index": sampled_frame_indices[source_time],
                    "bbox": bbox,
                    "score": score,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "max_other_player_iou": max_overlap,
                }
                tracklet = self._best_tracklet(
                    tracklets,
                    bbox,
                    source_time=source_time,
                )
                if tracklet is None:
                    tracklet = CandidateTracklet()
                    tracklets.append(tracklet)
                tracklet.observations.append(observation)

        anchored_tracklets: list[tuple[CandidateTracklet, dict[str, Any]]] = []
        for tracklet in tracklets:
            anchor = self._earliest_stable_anchor(
                tracklet,
                scene_end_sec=float(scene.end_sec),
            )
            if anchor is not None:
                anchored_tracklets.append((tracklet, anchor))
        ranked = sorted(
            anchored_tracklets,
            key=lambda item: (
                -float(item[1]["source_time"]),
                len(item[0].observations),
                max(row["score"] for row in item[0].observations),
            ),
            reverse=True,
        )[: self.MAX_CANDIDATES_PER_SCENE]
        candidates: list[ScenePlayerCandidate] = []

        for index, (tracklet, anchor) in enumerate(ranked, start=1):
            source_time = float(anchor["source_time"])
            frame = sampled_frames[source_time]
            candidate_id = generate_prefixed_id("pcand")
            thumbnail_path = self._write_thumbnail(
                cv2=cv2,
                frame=frame,
                bbox=anchor["bbox"],
                revision=revision,
                scene=scene,
                candidate_id=candidate_id,
            )
            artifact = self.artifact_repository.create(
                match_id=scene.match_id,
                project_id=revision.project_id,
                analysis_job_id=None,
                artifact_type="PLAYER_CANDIDATE_THUMBNAIL",
                file_path=thumbnail_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                mime_type="image/jpeg",
                metadata_={
                    "revision_id": revision.revision_id,
                    "scene_id": scene.timeline_event_id,
                    "candidate_id": candidate_id,
                },
            )
            support = len(tracklet.observations) / max(1, len(sample_times))
            detector_score = min(1.0, max(0.0, float(anchor["score"])))
            candidate = self.repository.create_candidate(
                candidate_id=candidate_id,
                revision_id=revision.revision_id,
                scene_id=scene.timeline_event_id,
                anchor_time_sec=round(source_time - float(scene.start_sec), 6),
                anchor_source_time_sec=round(source_time, 6),
                anchor_frame_index=round(source_time * fps),
                bbox_xyxy=[round(float(value), 3) for value in anchor["bbox"]],
                thumbnail_artifact_id=artifact.artifact_id,
                track_length_frames=max(1, len(tracklet.observations)),
                trackability_score=round(
                    0.7 * support + 0.3 * detector_score,
                    4,
                ),
                status="AVAILABLE",
                metadata_={
                    "display_label": f"선수 후보 {index}",
                    "detector": detector.runtime_metadata.get("backend"),
                    "detector_provenance": detector.runtime_metadata,
                    "class_id": int(anchor["class_id"]),
                    "class_name": self._normalize_class_name(
                        anchor["class_name"]
                    ),
                    "candidate_role_filter_policy": (
                        self.CANDIDATE_ROLE_FILTER_POLICY_VERSION
                    ),
                    "confidence": round(float(anchor["score"]), 6),
                    "source_media_asset_id": source_asset.asset_id,
                    "source_video_sha256": source_video_sha256,
                    "sample_support": len(tracklet.observations),
                    "detection_cache": {
                        "policy_version": DETECTION_CACHE_POLICY_VERSION,
                        "detector_fingerprint": (
                            cache_result.detector_fingerprint
                        ),
                        "scene_cache_hits": cache_result.cache_hits,
                        "scene_cache_misses": cache_result.cache_misses,
                        "scene_inference_frame_count": (
                            cache_result.inference_frame_count
                        ),
                    },
                    "initialization_anchor": {
                        "policy": "earliest_stable_anchor",
                        "valid": True,
                        "minimum_remaining_duration_sec": (
                            MIN_TRACKING_DURATION_SECONDS
                        ),
                        "remaining_duration_sec": round(
                            float(scene.end_sec) - source_time,
                            6,
                        ),
                        "stable_observation_count": len(
                            tracklet.observations
                        ),
                        "max_other_player_iou": round(
                            float(anchor["max_other_player_iou"]),
                            6,
                        ),
                        "bbox_edge_clipped": False,
                        "observations": [
                            {
                                "source_time": round(
                                    float(observation["source_time"]),
                                    6,
                                ),
                                "source_frame_index": int(
                                    observation["source_frame_index"]
                                ),
                                "bbox_xyxy": [
                                    round(float(value), 3)
                                    for value in observation["bbox"]
                                ],
                                "confidence": round(
                                    float(observation["score"]),
                                    6,
                                ),
                            }
                            for observation in tracklet.observations
                        ],
                    },
                },
            )
            candidates.append(candidate)
        return candidates, {
            "detection_cache": cache_result.as_dict(),
            "candidate_role_filter": {
                "policy_version": self.CANDIDATE_ROLE_FILTER_POLICY_VERSION,
                "allowed_class_names": sorted(
                    self.ALLOWED_CANDIDATE_CLASS_NAMES
                ),
                **scene_filter_totals,
                "excluded_role_counts": dict(
                    sorted(scene_excluded_role_counts.items())
                ),
            },
        }

    @classmethod
    def _filter_candidate_detections(
        cls,
        detections: list[Any],
        *,
        frame_height: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        eligible: list[Any] = []
        excluded_role_counts: dict[str, int] = {}
        metrics = {
            "raw_detection_count": 0,
            "allowed_role_detection_count": 0,
            "eligible_candidate_detection_count": 0,
            "excluded_non_target_role_count": 0,
            "excluded_too_small_count": 0,
        }
        minimum_height = max(0.0, float(frame_height)) * cls.MIN_BOX_HEIGHT_RATIO

        for detection in detections:
            metrics["raw_detection_count"] += 1
            class_name = cls._normalize_class_name(
                getattr(detection, "class_name", "")
            )
            if class_name not in cls.ALLOWED_CANDIDATE_CLASS_NAMES:
                metrics["excluded_non_target_role_count"] += 1
                role_key = class_name or "unknown"
                excluded_role_counts[role_key] = (
                    excluded_role_counts.get(role_key, 0) + 1
                )
                continue

            metrics["allowed_role_detection_count"] += 1
            bbox = getattr(detection, "bbox_xyxy", None)
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or float(bbox[3]) - float(bbox[1]) < minimum_height
            ):
                metrics["excluded_too_small_count"] += 1
                continue

            eligible.append(detection)
            metrics["eligible_candidate_detection_count"] += 1

        return eligible, {
            "policy_version": cls.CANDIDATE_ROLE_FILTER_POLICY_VERSION,
            "allowed_class_names": sorted(
                cls.ALLOWED_CANDIDATE_CLASS_NAMES
            ),
            **metrics,
            "excluded_role_counts": dict(sorted(excluded_role_counts.items())),
        }

    @staticmethod
    def _normalize_class_name(value: object) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _source_video_sha256(
        *,
        source_asset: MediaAsset,
        source_path: Path,
    ) -> str:
        declared = str(source_asset.sha256 or "")
        if len(declared) == 64:
            return declared
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _best_tracklet(
        cls,
        tracklets: list[CandidateTracklet],
        bbox: list[float],
        *,
        source_time: float,
    ) -> CandidateTracklet | None:
        scored = [
            (cls._iou(tracklet.last_bbox, bbox), tracklet)
            for tracklet in tracklets
            if (
                0
                < (
                    source_time
                    - float(tracklet.observations[-1]["source_time"])
                )
                <= cls.STABLE_MAX_GAP_SECONDS
            )
        ]
        if not scored:
            return None
        score, tracklet = max(scored, key=lambda item: item[0])
        return tracklet if score >= 0.2 else None

    @classmethod
    def _earliest_stable_anchor(
        cls,
        tracklet: CandidateTracklet,
        *,
        scene_end_sec: float,
    ) -> dict[str, Any] | None:
        if len(tracklet.observations) < cls.MIN_STABLE_OBSERVATIONS:
            return None
        for observation in sorted(
            tracklet.observations,
            key=lambda row: float(row["source_time"]),
        ):
            bbox = observation["bbox"]
            width = float(observation["frame_width"])
            height = float(observation["frame_height"])
            remaining = scene_end_sec - float(observation["source_time"])
            edge_clipped = (
                float(bbox[0]) <= cls.BBOX_EDGE_MARGIN_PX
                or float(bbox[1]) <= cls.BBOX_EDGE_MARGIN_PX
                or float(bbox[2]) >= width - cls.BBOX_EDGE_MARGIN_PX
                or float(bbox[3]) >= height - cls.BBOX_EDGE_MARGIN_PX
            )
            if (
                remaining >= MIN_TRACKING_DURATION_SECONDS
                and float(observation["score"])
                >= cls.MIN_ANCHOR_CONFIDENCE
                and float(observation["max_other_player_iou"])
                <= cls.MAX_ANCHOR_OVERLAP_IOU
                and not edge_clipped
            ):
                return observation
        return None

    @staticmethod
    def _iou(first: list[float], second: list[float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0,
            first[3] - first[1],
        )
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0,
            second[3] - second[1],
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _write_thumbnail(
        self,
        *,
        cv2,
        frame,
        bbox: list[float],
        revision: HighlightRevision,
        scene: TimelineEvent,
        candidate_id: str,
    ) -> Path:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = (x2 - x1) * 0.35
        pad_y = (y2 - y1) * 0.2
        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            raise ValueError("Candidate thumbnail crop is empty.")

        output_dir = (
            self.storage.storage_root
            / "matches"
            / scene.match_id
            / "projects"
            / revision.project_id
            / "highlight"
            / revision.revision_id
            / "candidate-thumbnails"
            / scene.timeline_event_id
        ).resolve()
        if not output_dir.is_relative_to(self.storage.storage_root):
            raise ValueError("Thumbnail output path escapes storage root.")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{candidate_id}.jpg"
        if not cv2.imwrite(str(output_path), crop):
            raise RuntimeError("Failed to write candidate thumbnail.")
        return output_path

    def _sample_times(
        self,
        scene: TimelineEvent,
        *,
        fps: float,
    ) -> list[float]:
        start = float(scene.start_sec)
        end = float(scene.end_sec)
        latest_anchor = end - MIN_TRACKING_DURATION_SECONDS
        if latest_anchor < start or fps <= 0:
            return []
        margin = min(0.1, (end - start) / 10)
        usable_start = start + margin
        usable_end = max(usable_start, latest_anchor)
        if self.SAMPLE_WINDOW_COUNT == 1 or usable_end == usable_start:
            bases = [usable_start]
        else:
            step = (
                usable_end - usable_start
            ) / (self.SAMPLE_WINDOW_COUNT - 1)
            bases = [
                usable_start + step * index
                for index in range(self.SAMPLE_WINDOW_COUNT)
            ]
        frame_step = 1.0 / fps
        values = [
            round(base + frame_step * offset, 6)
            for base in bases
            for offset in range(self.SAMPLES_PER_WINDOW)
            if base + frame_step * offset < end
        ]
        return sorted(set(values))
