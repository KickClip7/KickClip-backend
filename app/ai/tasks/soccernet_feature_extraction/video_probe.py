from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoProbeResult:
    """Basic video metadata needed by feature extraction.

    OpenCV의 VideoCapture metadata를 사용한다. 이 값은 feature split과
    chunk 계산의 기준이 되므로 duration/fps/frame_count가 비정상이면
    조기에 예외를 발생시킨다.
    """

    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "width": self.width,
            "height": self.height,
        }


def probe_video(video_path: str | Path) -> VideoProbeResult:
    """Read video metadata with OpenCV.

    Parameters
    ----------
    video_path:
        Absolute path or project-resolved path to a local video file.

    Returns
    -------
    VideoProbeResult
        fps, frame_count, duration_sec, width, height.

    Raises
    ------
    FileNotFoundError
        If the video path does not exist.
    RuntimeError
        If OpenCV is unavailable or VideoCapture cannot open the file.
    ValueError
        If fps/frame_count/duration metadata is invalid.
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path.as_posix()}")

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "opencv-python is required for video probing. "
            "Install it before running SoccerNet feature extraction."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path.as_posix()}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        raw_frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if not _is_positive_finite(fps):
        raise ValueError(f"Invalid video fps for {path.as_posix()}: {fps}")

    if not _is_positive_finite(raw_frame_count):
        raise ValueError(
            f"Invalid video frame_count for {path.as_posix()}: {raw_frame_count}"
        )

    frame_count = int(round(raw_frame_count))
    duration_sec = frame_count / fps

    if not _is_positive_finite(duration_sec):
        raise ValueError(
            f"Invalid video duration for {path.as_posix()}: {duration_sec}"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video resolution for {path.as_posix()}: {width}x{height}"
        )

    return VideoProbeResult(
        path=path,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        width=width,
        height=height,
    )


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0
