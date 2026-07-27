from __future__ import annotations

import copy
import json
from pathlib import Path, PureWindowsPath
from typing import Any

from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.errors import TrackingContractError
from app.domains.tracking.model import TrackingJob


NULL_BBOX_STATES = {"LOST", "SEARCHING", "AMBIGUOUS", "ABSENT", "TERMINATED"}
TIMELINE_STATES = {
    "INITIALIZING",
    "ACTIVE",
    "ACTIVE_LOW_CONFIDENCE",
    "OCCLUDED",
    "LOST",
    "SEARCHING",
    "AMBIGUOUS",
    "REACQUIRED",
    "ABSENT",
    "USER_CONFIRMED",
    "TERMINATED",
}
REQUIRED_ROOT_FIELDS = {
    "schema_version",
    "pipeline_version",
    "test_name",
    "target_id",
    "status",
    "video",
    "shots",
    "frames",
    "ambiguities",
    "confirmations",
    "provenance",
}
REQUIRED_FRAME_FIELDS = {
    "frame_index",
    "time_seconds",
    "shot_id",
    "state",
    "bbox_xyxy",
    "tracking_confidence",
    "identity_confidence",
    "identity_source",
    "review_required",
}


class TrackingTimelineService:
    def __init__(self, artifacts: TrackingArtifactService | None = None) -> None:
        self.artifacts = artifacts or TrackingArtifactService()

    def read(
        self,
        job: TrackingJob,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict[str, Any]:
        path, _, _ = self.artifacts.resolve(job, "target_timeline_json")
        payload = self._read_object(path)
        self._validate(payload)

        frames = payload["frames"]
        requested_start = 0 if start_frame is None else start_frame
        requested_end = end_frame
        filtered = [
            copy.deepcopy(frame)
            for frame in frames
            if int(frame["frame_index"]) >= requested_start
            and (requested_end is None or int(frame["frame_index"]) <= requested_end)
        ]

        public_payload = _redact_absolute_paths(copy.deepcopy(payload))
        public_payload["frames"] = _redact_absolute_paths(filtered)
        video = public_payload.get("video")
        if isinstance(video, dict):
            video["path"] = f"media_asset:{job.media_asset_id}"
            video["media_asset_id"] = job.media_asset_id
        public_payload["range"] = {
            "start_frame": requested_start,
            "end_frame": requested_end,
            "returned_frame_count": len(filtered),
        }
        return public_payload

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackingContractError("target_timeline.json is unreadable.") from exc
        if not isinstance(payload, dict):
            raise TrackingContractError("target_timeline.json must contain an object.")
        return payload

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        missing_root = REQUIRED_ROOT_FIELDS.difference(payload)
        if missing_root:
            raise TrackingContractError(
                "Target timeline is missing required fields: "
                + ", ".join(sorted(missing_root))
            )
        if payload.get("schema_version") != "kickclip.target_centric_e2e.v1":
            raise TrackingContractError("Unexpected target timeline schema version.")
        video = payload.get("video")
        if not isinstance(video, dict):
            raise TrackingContractError("Target timeline video must be an object.")
        for key in ("width", "height", "fps", "frame_count"):
            try:
                if float(video[key]) <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError) as exc:
                raise TrackingContractError(
                    f"Target timeline video.{key} is invalid."
                ) from exc
        if not isinstance(payload.get("shots"), list):
            raise TrackingContractError("Target timeline shots must be an array.")
        if not isinstance(payload.get("ambiguities"), list) or not isinstance(
            payload.get("confirmations"),
            list,
        ):
            raise TrackingContractError(
                "Target timeline review records must be arrays."
            )
        if not isinstance(payload.get("provenance"), dict):
            raise TrackingContractError(
                "Target timeline provenance must be an object."
            )
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise TrackingContractError("Target timeline frames must be an array.")
        seen_frames: set[int] = set()
        for frame in frames:
            if not isinstance(frame, dict):
                raise TrackingContractError("Target timeline contains an invalid frame.")
            missing_frame = REQUIRED_FRAME_FIELDS.difference(frame)
            if missing_frame:
                raise TrackingContractError(
                    "Target timeline frame is missing required fields."
                )
            try:
                frame_index = int(frame["frame_index"])
                time_seconds = float(frame["time_seconds"])
            except (TypeError, ValueError) as exc:
                raise TrackingContractError(
                    "Target timeline frame index/time is invalid."
                ) from exc
            if frame_index < 0 or time_seconds < 0 or frame_index in seen_frames:
                raise TrackingContractError(
                    "Target timeline frame index/time is invalid."
                )
            seen_frames.add(frame_index)
            state = str(frame.get("state") or "")
            if state not in TIMELINE_STATES:
                raise TrackingContractError("Target timeline state is invalid.")
            for key in ("tracking_confidence", "identity_confidence"):
                try:
                    confidence = float(frame[key])
                except (TypeError, ValueError) as exc:
                    raise TrackingContractError(
                        "Target timeline confidence is invalid."
                    ) from exc
                if confidence < 0 or confidence > 1:
                    raise TrackingContractError(
                        "Target timeline confidence is invalid."
                    )
            bbox = frame.get("bbox_xyxy")
            if bbox is not None and (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not isinstance(value, (int, float)) for value in bbox)
            ):
                raise TrackingContractError("Target timeline bbox is invalid.")
            if (
                state in NULL_BBOX_STATES
                and bbox is not None
            ):
                raise TrackingContractError(
                    "Uncertain target frame contains a forbidden bbox."
                )


def _redact_absolute_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_absolute_paths(item) for item in value]
    if isinstance(value, str) and _is_absolute_path_text(value):
        return None
    return value


def _is_absolute_path_text(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
