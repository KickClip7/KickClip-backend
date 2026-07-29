from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.domains.highlight.model import SceneTrackingBinding


CROP_STATES = {
    "ACTIVE",
    "ACTIVE_LOW_CONFIDENCE",
    "REACQUIRED",
    "USER_CONFIRMED",
    "OCCLUDED",
}
PRESENT_STATES = CROP_STATES | {"OCCLUDED", "INITIALIZING"}
ABSENT_STATES = {"LOST", "SEARCHING", "ABSENT", "TERMINATED"}
AMBIGUITY_STATES = {"AMBIGUOUS"}


class TrackingTimelineMapper:
    """Add source-video coordinates while retaining clip-local coordinates."""

    OCCLUSION_HOLD_SEC = 0.5

    @classmethod
    def map_frames(
        cls,
        binding: SceneTrackingBinding,
        frames: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if binding.source_start_time_sec is None:
            raise ValueError("Tracking binding has no source-time mapping.")
        source_start = float(binding.source_start_time_sec)
        source_fps = float(binding.source_fps or 0.0)
        mapped: list[dict[str, Any]] = []
        for frame in frames:
            local_frame_index = int(frame.get("frame_index") or 0)
            local_time = float(frame.get("time_seconds") or 0.0)
            source_time = source_start + local_time
            source_frame = (
                int(round(source_time * source_fps))
                if source_fps > 0
                else None
            )
            mapped.append(
                {
                    **frame,
                    "local_frame_index": local_frame_index,
                    "local_time_sec": round(local_time, 6),
                    "source_frame_index": source_frame,
                    "source_time_sec": round(source_time, 6),
                }
            )
        return mapped

    @classmethod
    def summarize(
        cls,
        binding: SceneTrackingBinding,
        timeline: dict[str, Any],
    ) -> dict[str, Any]:
        frames = cls.map_frames(binding, timeline.get("frames") or [])
        crop_frames = cls._with_short_occlusion_hold(frames)
        present = cls._segments(frames, PRESENT_STATES)
        absent = cls._segments(frames, ABSENT_STATES)
        ambiguity = cls._segments(frames, AMBIGUITY_STATES)
        crop = cls._segments(crop_frames, CROP_STATES, require_bbox=True)
        confirmed = cls._segments(frames, {"USER_CONFIRMED"}, require_bbox=True)
        confidences = [
            float(frame.get("tracking_confidence") or 0.0)
            for frame in frames
            if frame.get("state") in PRESENT_STATES
        ]
        return {
            "timeline_schema_version": timeline.get("schema_version"),
            "timeline_pipeline_version": timeline.get("pipeline_version"),
            "frame_count": len(frames),
            "source_time_mapping": "clip_timestamp_plus_persisted_source_offset",
            "source_frame_mapping": (
                "fps_derived_advisory"
                if binding.source_fps
                else "unavailable"
            ),
            "target_present_segments": present,
            "target_absent_segments": absent,
            "ambiguity_segments": ambiguity,
            "user_confirmed_segments": confirmed,
            "crop_segments": crop,
            "confidence": {
                "minimum": min(confidences) if confidences else None,
                "maximum": max(confidences) if confidences else None,
                "average": (
                    round(sum(confidences) / len(confidences), 6)
                    if confidences
                    else None
                ),
            },
            "timeline_artifact": (
                f"/api/v1/tracking/jobs/{binding.tracking_job_id}/timeline"
                if binding.tracking_job_id
                else None
            ),
        }

    @classmethod
    def crop_keyframes(
        cls,
        binding: SceneTrackingBinding,
        timeline: dict[str, Any],
        *,
        source_start_sec: float,
        source_end_sec: float,
        stride: int = 5,
    ) -> list[dict[str, Any]]:
        frames = cls._with_short_occlusion_hold(
            cls.map_frames(binding, timeline.get("frames") or [])
        )
        eligible = [
            frame
            for frame in frames
            if frame.get("state") in CROP_STATES
            and frame.get("bbox_xyxy") is not None
            and source_start_sec <= frame["source_time_sec"] <= source_end_sec
        ]
        if not eligible:
            return []
        sampled = eligible[:: max(1, stride)]
        if sampled[-1] is not eligible[-1]:
            sampled.append(eligible[-1])
        return [
            {
                "source_time_sec": frame["source_time_sec"],
                "relative_time_sec": round(
                    frame["source_time_sec"] - source_start_sec,
                    6,
                ),
                "state": frame["state"],
                "bbox_xyxy": [
                    round(float(value), 3)
                    for value in frame["bbox_xyxy"]
                ],
                "tracking_confidence": frame.get("tracking_confidence"),
                "bbox_source": frame.get("bbox_source", "TRACKER"),
            }
            for frame in sampled
        ]

    @classmethod
    def _with_short_occlusion_hold(
        cls,
        frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        last_stable_bbox: list[float] | None = None
        last_stable_time: float | None = None
        stable_states = CROP_STATES.difference({"OCCLUDED"})
        for original in frames:
            frame = dict(original)
            state = frame.get("state")
            source_time = float(frame["source_time_sec"])
            bbox = frame.get("bbox_xyxy")
            if state in stable_states and bbox is not None:
                last_stable_bbox = list(bbox)
                last_stable_time = source_time
            elif (
                state == "OCCLUDED"
                and bbox is None
                and last_stable_bbox is not None
                and last_stable_time is not None
                and source_time - last_stable_time <= cls.OCCLUSION_HOLD_SEC
            ):
                frame["bbox_xyxy"] = list(last_stable_bbox)
                frame["bbox_source"] = "HELD_LAST_STABLE"
            enriched.append(frame)
        return enriched

    @staticmethod
    def _segments(
        frames: list[dict[str, Any]],
        states: set[str],
        *,
        require_bbox: bool = False,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for frame in frames:
            included = frame.get("state") in states and (
                not require_bbox or frame.get("bbox_xyxy") is not None
            )
            if included:
                if current is None:
                    current = {
                        "start_time_sec": frame["source_time_sec"],
                        "end_time_sec": frame["source_time_sec"],
                        "states": [frame.get("state")],
                    }
                else:
                    current["end_time_sec"] = frame["source_time_sec"]
                    if frame.get("state") not in current["states"]:
                        current["states"].append(frame.get("state"))
            elif current is not None:
                result.append(current)
                current = None
        if current is not None:
            result.append(current)
        return result
