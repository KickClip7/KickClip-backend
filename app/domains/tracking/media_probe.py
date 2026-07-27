from __future__ import annotations

from pathlib import Path
from typing import Any

from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.tracking.errors import TrackingValidationError


def probe_tracking_video(path: Path) -> dict[str, Any]:
    """Require readable video dimensions/fps; upload metadata alone is not trusted."""

    metadata = extract_video_metadata(path)
    if (
        metadata.get("width")
        and metadata.get("height")
        and metadata.get("fps")
        and float(metadata["fps"]) > 0
    ):
        return metadata

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise TrackingValidationError("MediaAsset video cannot be opened.")
        try:
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            capture.release()
    except TrackingValidationError:
        raise
    except Exception as exc:
        raise TrackingValidationError(
            "MediaAsset video metadata cannot be read."
        ) from exc

    if width < 1 or height < 1 or fps <= 0 or frame_count < 1:
        raise TrackingValidationError("MediaAsset video metadata is invalid.")
    return {
        **metadata,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
    }

