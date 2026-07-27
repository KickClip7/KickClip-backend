from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.artifact.repository import ArtifactRepository
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

    SAMPLE_COUNT = 5
    MIN_BOX_HEIGHT_RATIO = 0.035
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
        revision.options = {
            **(revision.options or {}),
            "candidate_discovery": {
                "status": "RUNNING",
                "backend": detector.runtime_metadata.get("backend"),
                "runtime": detector.runtime_metadata,
                "selected_scene_count": len(scenes),
            },
        }

        source_path = self.storage.resolve_path(source_asset.file_path)
        if not source_path.is_file():
            raise FileNotFoundError("Source MediaAsset file is missing.")

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
                created.extend(
                    self._discover_scene(
                        cv2=cv2,
                        capture=capture,
                        detector=detector,
                        fps=fps,
                        revision=revision,
                        source_asset=source_asset,
                        scene=scene,
                    )
                )
        except PlayerDetectorError:
            raise
        except Exception as exc:
            raise PlayerDetectorInferenceError(
                "Player candidate discovery failed after detector "
                f"initialization: {type(exc).__name__}: {exc}",
                diagnostics=detector.runtime_metadata,
            ) from exc
        finally:
            capture.release()

        revision.options = {
            **(revision.options or {}),
            "candidate_discovery": {
                "status": "COMPLETED",
                "backend": detector.runtime_metadata.get("backend"),
                "runtime": detector.runtime_metadata,
                "selected_scene_count": len(scenes),
                "candidate_count": len(created),
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
        scene: TimelineEvent,
    ) -> list[ScenePlayerCandidate]:
        sample_times = self._sample_times(scene)
        tracklets: list[CandidateTracklet] = []
        sampled_frames: dict[float, Any] = {}

        for source_time in sample_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            sampled_frames[source_time] = frame

        ordered_times = list(sampled_frames)
        detections_by_frame = detector.detect_batch(
            [sampled_frames[source_time] for source_time in ordered_times]
        )
        if len(detections_by_frame) != len(ordered_times):
            raise RuntimeError(
                "Player detector output count does not match sampled frames."
            )

        for source_time, detections in zip(
            ordered_times,
            detections_by_frame,
        ):
            frame = sampled_frames[source_time]
            for detection in detections:
                bbox = detection.bbox_xyxy
                score = detection.confidence
                if (bbox[3] - bbox[1]) < frame.shape[0] * self.MIN_BOX_HEIGHT_RATIO:
                    continue
                observation = {
                    "source_time": source_time,
                    "bbox": bbox,
                    "score": score,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                }
                tracklet = self._best_tracklet(tracklets, bbox)
                if tracklet is None:
                    tracklet = CandidateTracklet()
                    tracklets.append(tracklet)
                tracklet.observations.append(observation)

        ranked = sorted(
            tracklets,
            key=lambda item: (
                len(item.observations),
                max(row["score"] for row in item.observations),
            ),
            reverse=True,
        )[: self.MAX_CANDIDATES_PER_SCENE]
        candidates: list[ScenePlayerCandidate] = []

        for index, tracklet in enumerate(ranked, start=1):
            anchor = max(tracklet.observations, key=lambda row: row["score"])
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
                anchor_frame_index=int(round(source_time * fps)),
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
                    "class_name": str(anchor["class_name"]),
                    "confidence": round(float(anchor["score"]), 6),
                    "source_media_asset_id": source_asset.asset_id,
                    "sample_support": len(tracklet.observations),
                },
            )
            candidates.append(candidate)
        return candidates

    @classmethod
    def _best_tracklet(
        cls,
        tracklets: list[CandidateTracklet],
        bbox: list[float],
    ) -> CandidateTracklet | None:
        scored = [
            (cls._iou(tracklet.last_bbox, bbox), tracklet)
            for tracklet in tracklets
        ]
        if not scored:
            return None
        score, tracklet = max(scored, key=lambda item: item[0])
        return tracklet if score >= 0.2 else None

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

    def _sample_times(self, scene: TimelineEvent) -> list[float]:
        start = float(scene.start_sec)
        end = float(scene.end_sec)
        if end <= start:
            return [start]
        margin = min(0.1, (end - start) / 10)
        usable_start = start + margin
        usable_end = end - margin
        if self.SAMPLE_COUNT == 1:
            return [(usable_start + usable_end) / 2]
        step = (usable_end - usable_start) / (self.SAMPLE_COUNT - 1)
        values = [
            round(usable_start + step * index, 6)
            for index in range(self.SAMPLE_COUNT)
        ]
        representative = min(
            max(float(scene.timestamp_sec), usable_start),
            usable_end,
        )
        values.append(round(representative, 6))
        return sorted(set(values))
