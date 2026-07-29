from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.highlight.model import HighlightRevision, ScenePlayerCandidate
from app.domains.highlight.player_detector import (
    PlayerDetector,
    create_player_detector,
)
from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.timeline.model import TimelineEvent
from app.domains.tracking.errors import (
    TrackingClipDecodeFailedError,
    TrackingInputError,
    TrackingNoValidInitializationAnchorError,
)
from app.domains.tracking.input_contract import (
    MAX_TRACKING_DURATION_SECONDS,
    MIN_TRACKING_DURATION_SECONDS,
    validate_tracking_clip,
)
from app.storage.local_storage import LocalStorage


@dataclass(frozen=True)
class ExtractedSceneClip:
    asset: MediaAsset
    source_start_time_sec: float
    source_end_time_sec: float
    source_start_frame: int
    source_end_frame: int
    source_fps: float
    clip_fps: float
    clip_frame_count: int
    command: list[str]
    validation: dict[str, Any]
    artifact_paths: dict[str, str]


class SceneClipService:
    """Extract an anchor-rebased clip without modifying the frozen tracker."""

    MIN_STABLE_OBSERVATIONS = 3
    MIN_FRAME0_DETECTION_IOU = 0.5
    MIN_FRAME0_DETECTION_COVERAGE = 0.8

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
        self.media_repository = MediaAssetRepository(db)

    def extract_for_candidate(
        self,
        *,
        source_asset: MediaAsset,
        project_id: str,
        revision: HighlightRevision,
        scene: TimelineEvent,
        candidate: ScenePlayerCandidate,
    ) -> ExtractedSceneClip:
        source_path = self.storage.resolve_path(source_asset.file_path)
        if not source_path.is_file():
            raise FileNotFoundError("Source MediaAsset file is missing.")

        scene_start = float(scene.start_sec)
        scene_end = float(scene.end_sec)
        anchor_source_time = float(candidate.anchor_source_time_sec)
        source_start = max(scene_start, anchor_source_time)
        remaining_duration = scene_end - source_start
        anchor_diagnostics = {
            "scene_id": scene.timeline_event_id,
            "candidate_id": candidate.candidate_id,
            "scene_start_sec": scene_start,
            "scene_end_sec": scene_end,
            "candidate_anchor_source_time_sec": anchor_source_time,
            "calculated_source_start_sec": source_start,
            "remaining_duration_sec": remaining_duration,
            "minimum_tracking_duration_sec": MIN_TRACKING_DURATION_SECONDS,
        }
        anchor_contract = (candidate.metadata_ or {}).get(
            "initialization_anchor"
        )
        if isinstance(anchor_contract, dict):
            anchor_diagnostics["candidate_anchor_contract"] = anchor_contract
        if (
            not isinstance(anchor_contract, dict)
            or not anchor_contract.get("valid")
            or int(anchor_contract.get("stable_observation_count") or 0)
            < self.MIN_STABLE_OBSERVATIONS
        ):
            raise TrackingNoValidInitializationAnchorError(
                (
                    "Candidate was not produced by the earliest-stable-anchor "
                    "policy. Run player discovery again."
                ),
                diagnostics=anchor_diagnostics,
            )
        if remaining_duration < MIN_TRACKING_DURATION_SECONDS:
            raise TrackingNoValidInitializationAnchorError(
                (
                    f"Candidate anchor leaves {remaining_duration:.3f}s; "
                    f"at least {MIN_TRACKING_DURATION_SECONDS:.0f}s is required."
                ),
                diagnostics=anchor_diagnostics,
            )

        source_end = min(
            scene_end,
            source_start + MAX_TRACKING_DURATION_SECONDS,
        )
        if source_end <= source_start:
            raise TrackingNoValidInitializationAnchorError(
                "Candidate anchor leaves no scene duration to track.",
                diagnostics=anchor_diagnostics,
            )

        source_fps = float(source_asset.fps or 0.0)
        if source_fps <= 0:
            source_metadata = extract_video_metadata(source_path)
            source_fps = float(source_metadata.get("fps") or 0.0)
        if source_fps <= 0:
            raise ValueError("Source video FPS is unavailable.")

        output_dir = (
            self.storage.storage_root
            / "matches"
            / scene.match_id
            / "projects"
            / project_id
            / "highlight"
            / revision.revision_id
            / "scene-clips"
            / scene.timeline_event_id
        ).resolve()
        if not output_dir.is_relative_to(self.storage.storage_root):
            raise ValueError("Scene clip output path escapes storage root.")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{candidate.candidate_id}.mp4"
        duration = source_end - source_start
        validation_path = output_dir / "scene_clip_validation.json"
        frame0_preview_path = output_dir / "initial_bbox_frame0_preview.jpg"
        detection_match_path = output_dir / "initial_bbox_detection_match.json"
        artifact_paths = {
            "scene_clip_validation": str(validation_path),
            "initial_bbox_frame0_preview": str(frame0_preview_path),
            "initial_bbox_detection_match": str(detection_match_path),
        }
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{source_start:.6f}",
            "-i",
            source_path.as_posix(),
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            output_path.as_posix(),
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.storage.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            diagnostics = {
                **anchor_diagnostics,
                "calculated_source_end_sec": source_end,
                "requested_duration_sec": duration,
                "ffmpeg_return_code": completed.returncode,
                "ffmpeg_stderr_tail": (
                    completed.stderr[-2000:] if completed.stderr else ""
                ),
                "artifact_paths": artifact_paths,
            }
            self._write_json(
                validation_path,
                {"status": "FAILED", **diagnostics},
            )
            raise TrackingClipDecodeFailedError(
                "Scene clip extraction failed.",
                diagnostics=diagnostics,
            )

        try:
            clip_metadata = validate_tracking_clip(
                output_path,
                requested_duration_sec=duration,
            )
        except TrackingInputError as exc:
            exc.diagnostics.update(
                {
                    **anchor_diagnostics,
                    "calculated_source_end_sec": source_end,
                    "requested_duration_sec": duration,
                    "ffmpeg_return_code": completed.returncode,
                    "artifact_paths": artifact_paths,
                }
            )
            self._write_json(
                validation_path,
                {
                    "status": "FAILED",
                    "code": exc.code,
                    **exc.diagnostics,
                },
            )
            raise

        clip_fps = float(clip_metadata["fps"])
        clip_duration = float(clip_metadata["duration_sec"])
        clip_frame_count = int(clip_metadata["frame_count"])
        validation = {
            "status": "PASSED",
            "code": None,
            **anchor_diagnostics,
            "calculated_source_end_sec": source_end,
            "requested_duration_sec": duration,
            "actual_duration_sec": clip_duration,
            "requested_actual_duration_delta_sec": abs(
                clip_duration - duration
            ),
            "actual_frame_count": clip_frame_count,
            "actual_fps": clip_fps,
            "actual_width": int(clip_metadata["width"]),
            "actual_height": int(clip_metadata["height"]),
            "frame0_decodable": bool(clip_metadata["frame0_decodable"]),
            "ffmpeg_return_code": completed.returncode,
            "ffmpeg_command": command,
            "artifact_paths": artifact_paths,
        }
        self._write_json(validation_path, validation)

        try:
            match = self._validate_frame0_initialization(
                output_path=output_path,
                bbox_xyxy=[float(value) for value in candidate.bbox_xyxy],
                preview_path=frame0_preview_path,
                match_path=detection_match_path,
            )
        except TrackingInputError as exc:
            exc.diagnostics.update(
                {
                    **anchor_diagnostics,
                    "calculated_source_end_sec": source_end,
                    "requested_duration_sec": duration,
                    "actual_duration_sec": clip_duration,
                    "actual_frame_count": clip_frame_count,
                    "actual_fps": clip_fps,
                    "artifact_paths": artifact_paths,
                }
            )
            validation.update(
                {
                    "status": "FAILED",
                    "code": exc.code,
                    "failure": exc.diagnostics,
                }
            )
            self._write_json(validation_path, validation)
            raise
        validation["frame0_detection_match"] = match
        self._write_json(validation_path, validation)
        source_start_frame = round(source_start * source_fps)
        source_end_frame = max(
            source_start_frame,
            round(source_end * source_fps) - 1,
        )
        asset = self.media_repository.create(
            match_id=scene.match_id,
            asset_type="HIGHLIGHT_SCENE_CLIP",
            file_path=output_path.relative_to(self.storage.project_root).as_posix(),
            original_filename=output_path.name,
            mime_type="video/mp4",
            duration_sec=clip_duration,
            fps=clip_fps,
            width=clip_metadata.get("width"),
            height=clip_metadata.get("height"),
            size_bytes=clip_metadata.get("size_bytes"),
            sha256=self._sha256(output_path),
        )
        self.db.flush()
        return ExtractedSceneClip(
            asset=asset,
            source_start_time_sec=source_start,
            source_end_time_sec=source_end,
            source_start_frame=source_start_frame,
            source_end_frame=source_end_frame,
            source_fps=source_fps,
            clip_fps=clip_fps,
            clip_frame_count=clip_frame_count,
            command=command,
            validation=validation,
            artifact_paths=artifact_paths,
        )

    def _validate_frame0_initialization(
        self,
        *,
        output_path: Path,
        bbox_xyxy: list[float],
        preview_path: Path,
        match_path: Path,
    ) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise TrackingClipDecodeFailedError(
                "OpenCV is required to validate tracking frame 0.",
                diagnostics={"path": str(output_path)},
            ) from exc

        capture = cv2.VideoCapture(str(output_path))
        try:
            decoded, frame0 = capture.read() if capture.isOpened() else (False, None)
        finally:
            capture.release()
        if not decoded or frame0 is None:
            raise TrackingClipDecodeFailedError(
                "Tracking clip frame 0 cannot be decoded.",
                diagnostics={"path": str(output_path)},
            )

        detector = self.detector or create_player_detector(self.settings)
        detections_by_frame = detector.detect_batch([frame0])
        detections = detections_by_frame[0] if detections_by_frame else []
        matched = None
        for detection in detections:
            iou, coverage = self._iou_and_coverage(
                bbox_xyxy,
                detection.bbox_xyxy,
            )
            record = {
                "bbox_xyxy": [
                    round(float(value), 3)
                    for value in detection.bbox_xyxy
                ],
                "confidence": round(float(detection.confidence), 6),
                "class_id": int(detection.class_id),
                "class_name": str(detection.class_name),
                "iou": round(iou, 6),
                "candidate_coverage": round(coverage, 6),
            }
            if matched is None or (record["iou"], record["candidate_coverage"]) > (
                matched["iou"],
                matched["candidate_coverage"],
            ):
                matched = record

        accepted = bool(
            matched
            and (
                float(matched["iou"]) >= self.MIN_FRAME0_DETECTION_IOU
                or float(matched["candidate_coverage"])
                >= self.MIN_FRAME0_DETECTION_COVERAGE
            )
        )
        result = {
            "status": "MATCHED" if accepted else "NO_MATCH",
            "selected_bbox_xyxy": [
                round(float(value), 3) for value in bbox_xyxy
            ],
            "minimum_iou": self.MIN_FRAME0_DETECTION_IOU,
            "minimum_candidate_coverage": (
                self.MIN_FRAME0_DETECTION_COVERAGE
            ),
            "matched_detection": matched,
            "detection_count": len(detections),
            "detector_provenance": detector.runtime_metadata,
        }
        self._write_json(match_path, result)

        preview = frame0.copy()
        self._draw_bbox(
            cv2,
            preview,
            bbox_xyxy,
            (255, 128, 0),
            "selected bbox",
        )
        if matched is not None:
            self._draw_bbox(
                cv2,
                preview,
                matched["bbox_xyxy"],
                (0, 255, 0) if accepted else (0, 0, 255),
                (
                    f"RF-DETR iou={matched['iou']:.3f} "
                    f"coverage={matched['candidate_coverage']:.3f}"
                ),
            )
        if not cv2.imwrite(str(preview_path), preview):
            raise TrackingClipDecodeFailedError(
                "Failed to write frame-0 bbox validation preview.",
                diagnostics={"path": str(preview_path)},
            )

        if not accepted:
            raise TrackingNoValidInitializationAnchorError(
                (
                    "Selected bbox does not match an RF-DETR player detection "
                    "on tracking clip frame 0."
                ),
                diagnostics={
                    "path": str(output_path),
                    "artifact_paths": {
                        "initial_bbox_frame0_preview": str(preview_path),
                        "initial_bbox_detection_match": str(match_path),
                    },
                    "detection_match": result,
                },
            )
        return result

    @staticmethod
    def _draw_bbox(cv2, frame, bbox, color, label: str) -> None:
        x1, y1, x2, y2 = [round(float(value)) for value in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            label,
            (max(0, x1), max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _iou_and_coverage(
        selected: list[float],
        detected: list[float],
    ) -> tuple[float, float]:
        left = max(selected[0], detected[0])
        top = max(selected[1], detected[1])
        right = min(selected[2], detected[2])
        bottom = min(selected[3], detected[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        selected_area = max(0.0, selected[2] - selected[0]) * max(
            0.0,
            selected[3] - selected[1],
        )
        detected_area = max(0.0, detected[2] - detected[0]) * max(
            0.0,
            detected[3] - detected[1],
        )
        union = selected_area + detected_area - intersection
        return (
            intersection / union if union > 0 else 0.0,
            intersection / selected_area if selected_area > 0 else 0.0,
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
