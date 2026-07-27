from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.domains.highlight.model import HighlightRevision, ScenePlayerCandidate
from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.timeline.model import TimelineEvent
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


class SceneClipService:
    """Extract an anchor-rebased clip without modifying the frozen tracker."""

    def __init__(self, db: Session):
        self.db = db
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

        source_start = max(
            float(scene.start_sec),
            float(candidate.anchor_source_time_sec),
        )
        source_end = float(scene.end_sec)
        if source_end <= source_start:
            raise ValueError("Candidate anchor leaves no scene duration to track.")

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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Scene clip extraction failed: "
                + (completed.stderr[-2000:] if completed.stderr else "")
            )

        clip_metadata = extract_video_metadata(output_path)
        clip_fps = float(clip_metadata.get("fps") or source_fps)
        clip_duration = float(clip_metadata.get("duration_sec") or duration)
        clip_frame_count = max(1, int(round(clip_duration * clip_fps)))
        source_start_frame = int(round(source_start * source_fps))
        source_end_frame = max(
            source_start_frame,
            int(round(source_end * source_fps)) - 1,
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
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
