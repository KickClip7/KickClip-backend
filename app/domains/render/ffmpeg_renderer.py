from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.domains.clip_plan.model import ClipPlan
from app.domains.render.template_builder import RenderTemplate
from app.domains.render.tracking_transform import TrackingTransformBuilder


@dataclass(frozen=True)
class RenderedVideoResult:
    output_path: Path
    subtitle_path: Path | None
    command_log: list[str]


class FFmpegRenderer:
    """ClipPlan 기반 FFmpeg 렌더러.

    MVP 전략:
    1. ClipPlanItem별로 원본 영상에서 구간을 잘라 임시 segment mp4 생성
    2. segment들을 concat demuxer로 합치기
    3. captions_enabled=True면 렌더 영상과 같은 폴더에 SRT 파일 생성

    큰 raw output이나 중간 파일은 DB에 넣지 않고 storage 파일로만 둔다.
    """

    def render_clip_plan(
        self,
        *,
        clip_plan: ClipPlan,
        source_video_path: Path,
        output_dir: Path,
        template: RenderTemplate,
        title: str | None = None,
        captions_enabled: bool = False,
    ) -> RenderedVideoResult:
        if not source_video_path.exists() or not source_video_path.is_file():
            raise FileNotFoundError(f"Source video not found: {source_video_path}")

        if not clip_plan.items:
            raise ValueError("ClipPlan has no items to render.")

        output_dir.mkdir(parents=True, exist_ok=True)
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        command_log: list[str] = []
        segment_paths: list[Path] = []

        for index, item in enumerate(sorted(clip_plan.items, key=lambda row: row.order_index)):
            duration_sec = max(0.0, float(item.end_sec) - float(item.start_sec))
            if duration_sec <= 0:
                continue

            segment_path = segments_dir / f"segment_{index:03d}.mp4"
            command = self._build_segment_command(
                source_video_path=source_video_path,
                output_path=segment_path,
                start_sec=float(item.start_sec),
                duration_sec=duration_sec,
                template=template,
                item_metadata=item.metadata_ or {},
            )
            self._run(command)
            command_log.append(self._command_to_text(command))
            segment_paths.append(segment_path)

        if not segment_paths:
            raise ValueError("No valid ClipPlanItem duration to render.")

        concat_file = output_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file {shlex.quote(path.as_posix())}" for path in segment_paths),
            encoding="utf-8",
        )

        output_path = output_dir / "rendered_short.mp4"
        concat_command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file.as_posix(),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path.as_posix(),
        ]
        self._run(concat_command)
        command_log.append(self._command_to_text(concat_command))

        subtitle_path = None
        if captions_enabled:
            subtitle_path = output_dir / "captions.srt"
            subtitle_path.write_text(
                self._build_srt(clip_plan=clip_plan, title=title),
                encoding="utf-8",
            )

        return RenderedVideoResult(
            output_path=output_path,
            subtitle_path=subtitle_path,
            command_log=command_log,
        )

    @staticmethod
    def _build_segment_command(
        *,
        source_video_path: Path,
        output_path: Path,
        start_sec: float,
        duration_sec: float,
        template: RenderTemplate,
        item_metadata: dict | None = None,
    ) -> list[str]:
        item_metadata = item_metadata or {}
        render_strategy = str(
            item_metadata.get("render_strategy") or "FULL_FRAME"
        ).upper()
        if render_strategy == "TARGET_CENTERED":
            transform = item_metadata.get("tracking_transform")
            if not isinstance(transform, dict):
                raise ValueError(
                    "TARGET_CENTERED ClipPlanItem has no tracking transform."
                )
            video_filter = TrackingTransformBuilder.ffmpeg_filter(
                transform=transform,
                output_width=template.width,
                output_height=template.height,
            )
        elif render_strategy == "FULL_FRAME":
            # Preserve the whole source image for an explicitly full-frame
            # scene. Padding is intentional; silently cropping to another
            # player would violate the target-absence policy.
            video_filter = (
                f"scale={template.width}:{template.height}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={template.width}:{template.height}:"
                "(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            )
        else:
            raise ValueError(
                f"Unsupported render strategy in ClipPlan: {render_strategy}"
            )
        return [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            source_video_path.as_posix(),
            "-t",
            f"{duration_sec:.3f}",
            "-vf",
            video_filter,
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            output_path.as_posix(),
        ]

    @staticmethod
    def _build_srt(
        *,
        clip_plan: ClipPlan,
        title: str | None,
    ) -> str:
        lines: list[str] = []
        cursor = 0.0
        ordered_items = sorted(clip_plan.items, key=lambda row: row.order_index)

        for index, item in enumerate(ordered_items, start=1):
            duration = max(0.0, float(item.end_sec) - float(item.start_sec))
            if duration <= 0:
                continue

            start = cursor
            end = cursor + duration
            caption = title or item.reason or f"KickClip Highlight {index}"

            lines.extend(
                [
                    str(index),
                    f"{FFmpegRenderer._format_srt_time(start)} --> {FFmpegRenderer._format_srt_time(end)}",
                    caption,
                    "",
                ]
            )
            cursor = end

        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        milliseconds = int(round((seconds - int(seconds)) * 1000))
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def _run(command: list[str]) -> None:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            stderr_tail = completed.stderr[-2000:] if completed.stderr else ""
            raise RuntimeError(f"FFmpeg failed: {stderr_tail}")

    @staticmethod
    def _command_to_text(command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)
