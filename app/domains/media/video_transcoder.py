import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.paths import get_project_root


@dataclass(frozen=True)
class TranscodedPreviewClip:
    absolute_path: Path
    relative_path: str
    filename: str


def create_browser_preview_clip(
    input_path: Path,
    output_dir: Path,
    original_filename: str | None = None,
    start_sec: float = 0,
    duration_sec: float = 30,
) -> TranscodedPreviewClip | None:
    """Create short browser-friendly H.264 MP4 preview clip.

    중요:
    - 원본 90분 전체를 변환하지 않는다.
    - 기본적으로 앞 30초만 변환한다.
    - ffmpeg가 없거나 변환 실패 시 None을 반환한다.
    - 원본 RAW_VIDEO는 삭제하지 않는다.
    """
    if shutil.which("ffmpeg") is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize_stem(Path(original_filename or input_path.name).stem)
    output_filename = f"{stem}_preview_{int(duration_sec)}s_h264.mp4"
    output_path = output_dir / output_filename

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        if output_path.exists():
            output_path.unlink()
        return None

    if not output_path.exists() or output_path.stat().st_size <= 0:
        return None

    return TranscodedPreviewClip(
        absolute_path=output_path,
        relative_path=_to_project_relative_path(output_path),
        filename=output_filename,
    )


def _to_project_relative_path(path: Path) -> str:
    project_root = get_project_root()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _sanitize_stem(value: str) -> str:
    name = value.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^0-9a-zA-Z가-힣._ -]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "video"