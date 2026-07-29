import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def extract_video_metadata(file_path: str | Path) -> dict[str, Any]:
    """Extract video metadata with ffprobe if available.

    ffprobe가 없거나 실패해도 업로드 자체는 성공해야 하므로,
    실패 시 size_bytes 중심의 fallback metadata를 반환한다.
    """
    path = Path(file_path)

    fallback = {
        "duration_sec": None,
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "codec_name": None,
        "codec_tag_string": None,
        "pix_fmt": None,
        "format_name": None,
    }

    if shutil.which("ffprobe") is None:
        return fallback

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,codec_tag_string,pix_fmt,width,height,"
            "r_frame_rate,duration,nb_frames"
        ),
        "-show_entries",
        "format=format_name,duration,size",
        "-of",
        "json",
        str(path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(completed.stdout)
    except Exception:  # noqa: BLE001 - metadata probing must retain its fallback
        return fallback

    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}

    duration_sec = _safe_float(stream.get("duration"))
    if duration_sec is None:
        duration_sec = _safe_float(fmt.get("duration"))

    size_bytes = _safe_int(fmt.get("size"))
    if size_bytes is None:
        size_bytes = fallback["size_bytes"]

    return {
        "duration_sec": duration_sec,
        "fps": _parse_fps(stream.get("r_frame_rate")),
        "width": _safe_int(stream.get("width")),
        "height": _safe_int(stream.get("height")),
        "frame_count": _safe_int(stream.get("nb_frames")),
        "size_bytes": size_bytes,
        "codec_name": stream.get("codec_name"),
        "codec_tag_string": stream.get("codec_tag_string"),
        "pix_fmt": stream.get("pix_fmt"),
        "format_name": fmt.get("format_name"),
    }


def _parse_fps(value: str | None) -> float | None:
    if not value:
        return None

    if "/" not in value:
        return _safe_float(value)

    numerator, denominator = value.split("/", maxsplit=1)
    num = _safe_float(numerator)
    den = _safe_float(denominator)

    if num is None or den in (None, 0):
        return None

    return round(num / den, 3)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
