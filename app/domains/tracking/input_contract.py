from __future__ import annotations

from pathlib import Path
from typing import Any

from app.domains.tracking.errors import (
    TrackingClipMetadataInvalidError,
    TrackingClipTooLongError,
    TrackingClipTooShortError,
)
from app.domains.tracking.media_probe import probe_tracking_video

MIN_TRACKING_DURATION_SECONDS = 10.0
MAX_TRACKING_DURATION_SECONDS = 30.0
FRAME_COUNT_ABSOLUTE_TOLERANCE = 2
FRAME_COUNT_RELATIVE_TOLERANCE = 0.02


def validate_tracking_clip(
    path: Path,
    *,
    requested_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Validate the frozen Phase-1 input contract before a subprocess is queued."""

    metadata = probe_tracking_video(path)
    duration = float(metadata.get("duration_sec") or 0.0)
    fps = float(metadata.get("fps") or 0.0)
    frame_count = int(metadata.get("frame_count") or 0)
    diagnostics = {
        "path": str(path),
        "requested_duration_sec": requested_duration_sec,
        "actual_duration_sec": duration,
        "actual_frame_count": frame_count,
        "fps": fps,
        "width": int(metadata.get("width") or 0),
        "height": int(metadata.get("height") or 0),
        "frame0_decodable": bool(metadata.get("frame0_decodable")),
    }

    if duration < MIN_TRACKING_DURATION_SECONDS:
        raise TrackingClipTooShortError(
            (
                f"Tracking clip is {duration:.3f}s/{frame_count} frames; "
                f"minimum is {MIN_TRACKING_DURATION_SECONDS:.0f}s."
            ),
            diagnostics=diagnostics,
        )
    if duration > MAX_TRACKING_DURATION_SECONDS:
        raise TrackingClipTooLongError(
            (
                f"Tracking clip is {duration:.3f}s; "
                f"maximum is {MAX_TRACKING_DURATION_SECONDS:.0f}s."
            ),
            diagnostics=diagnostics,
        )

    expected_frames = duration * fps
    frame_tolerance = max(
        FRAME_COUNT_ABSOLUTE_TOLERANCE,
        round(expected_frames * FRAME_COUNT_RELATIVE_TOLERANCE),
    )
    frame_delta = abs(frame_count - expected_frames)
    diagnostics.update(
        {
            "expected_frame_count": expected_frames,
            "frame_count_delta": frame_delta,
            "frame_count_tolerance": frame_tolerance,
            "requested_actual_duration_delta_sec": (
                abs(duration - requested_duration_sec)
                if requested_duration_sec is not None
                else None
            ),
        }
    )
    if frame_delta > frame_tolerance:
        raise TrackingClipMetadataInvalidError(
            (
                "Tracking clip frame count does not match duration and FPS: "
                f"actual={frame_count}, expected={expected_frames:.2f}."
            ),
            diagnostics=diagnostics,
        )
    return {**metadata, "validation": diagnostics}
