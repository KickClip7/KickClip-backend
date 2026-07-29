from __future__ import annotations

from pathlib import Path
from typing import Any

from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.tracking.errors import (
    TrackingClipDecodeFailedError,
    TrackingClipMetadataInvalidError,
)


def probe_tracking_video(path: Path) -> dict[str, Any]:
    """Probe metadata and decode frame 0; container metadata alone is not trusted."""

    metadata = extract_video_metadata(path)
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise TrackingClipDecodeFailedError(
                "Tracking clip cannot be opened.",
                diagnostics={"path": str(path)},
            )
        try:
            width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            decoded, frame0 = capture.read()
        finally:
            capture.release()
    except (
        TrackingClipDecodeFailedError,
        TrackingClipMetadataInvalidError,
    ):
        raise
    except Exception as exc:
        raise TrackingClipDecodeFailedError(
            "Tracking clip frame 0 cannot be decoded.",
            diagnostics={"path": str(path), "exception": type(exc).__name__},
        ) from exc

    if width < 1 or height < 1 or fps <= 0 or frame_count < 1:
        raise TrackingClipMetadataInvalidError(
            "Tracking clip width, height, FPS, or frame count is invalid.",
            diagnostics={
                "path": str(path),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
            },
        )
    if not decoded or frame0 is None or getattr(frame0, "size", 0) <= 0:
        raise TrackingClipDecodeFailedError(
            "Tracking clip frame 0 cannot be decoded.",
            diagnostics={
                "path": str(path),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
            },
        )

    frame_duration = frame_count / fps
    probed_duration = float(metadata.get("duration_sec") or 0.0)
    duration = probed_duration if probed_duration > 0 else frame_duration
    return {
        **metadata,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "frame_duration_sec": frame_duration,
        "frame0_decodable": True,
    }
