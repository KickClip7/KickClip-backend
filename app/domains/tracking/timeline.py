from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from app.domains.tracking.artifacts import TrackingArtifactService, sha256_file
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
PHASE3C_DIRECTIONAL_SOURCE = "REAL_FROZEN_STAGE2_DIRECTIONAL_TIMELINE"
REVIEWED_SHOT_NORMALIZATION_POLICY = "IMMUTABLE_REVIEWED_SHOT_METADATA_ONLY_R1"


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
        payload = self._normalize_compatible_payload(payload)

        reviewed_shots, reviewed_source = self._reviewed_shots_for_job(
            job,
            frame_count=_video_frame_count(payload),
        )
        if reviewed_shots is not None:
            payload = self._normalize_reviewed_shot_contract(
                payload,
                reviewed_shots=reviewed_shots,
                reviewed_source=reviewed_source or {},
            )

        self._validate(payload)

        frames = payload["frames"]
        tracked_start, tracked_end = _tracked_frame_range(
            payload,
            frame_count=_video_frame_count(payload),
        )
        requested_start = tracked_start if start_frame is None else start_frame
        requested_end = tracked_end if end_frame is None else end_frame
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
            "tracked_start_frame": tracked_start,
            "tracked_end_frame": tracked_end,
        }
        return public_payload

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackingContractError(
                f"Tracking JSON artifact is unreadable: {path.name}."
            ) from exc
        if not isinstance(payload, dict):
            raise TrackingContractError(
                f"Tracking JSON artifact must contain an object: {path.name}."
            )
        return payload

    @staticmethod
    def _normalize_compatible_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Fill only the known Phase 3-C Stage-2 adapter omissions.

        No bbox, state, confidence, target identity, or review decision is
        synthesized. Only public fields absent from the frozen Stage-2 schema
        are supplied for frames explicitly marked as real Phase 3-C output.
        """

        frames = payload.get("frames")
        if not isinstance(frames, list):
            return payload

        normalized_payload = copy.deepcopy(payload)
        normalized_frames = normalized_payload.get("frames")
        if not isinstance(normalized_frames, list):
            return normalized_payload

        for frame in normalized_frames:
            if (
                not isinstance(frame, dict)
                or frame.get("phase3c_source") != PHASE3C_DIRECTIONAL_SOURCE
            ):
                continue

            identity_source = str(frame.get("identity_source") or "").strip()
            if not identity_source:
                frame["identity_source"] = _phase3c_identity_source(frame)

            if "review_required" not in frame:
                frame["review_required"] = False

        return normalized_payload

    def _reviewed_shots_for_job(
        self,
        job: TrackingJob,
        *,
        frame_count: int,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Resolve the immutable reviewed-shot artifact owned by the R1 launch.

        The DB runtime metadata is preferred. A server-owned launch manifest in
        the job root is a compatibility fallback for already-completed jobs.
        Caller-supplied paths are never accepted.
        """

        path: Path | None = None
        expected_sha: str | None = None
        source = ""

        metadata = getattr(job, "runtime_metadata", None)
        if isinstance(metadata, Mapping):
            scene = metadata.get("scene_target_selection")
            if isinstance(scene, Mapping):
                raw_path = str(scene.get("shot_boundaries_path") or "").strip()
                raw_sha = str(scene.get("shot_boundaries_sha256") or "").strip()
                if raw_path or raw_sha:
                    if not raw_path or len(raw_sha) != 64:
                        raise TrackingContractError(
                            "Reviewed shot boundary runtime metadata is incomplete."
                        )
                    path = Path(raw_path).expanduser().resolve()
                    expected_sha = raw_sha.lower()
                    source = "TRACKING_JOB_RUNTIME_METADATA"

        if path is None:
            job_root = None
            job_root_method = getattr(self.artifacts, "job_root", None)
            if callable(job_root_method):
                try:
                    job_root = Path(job_root_method(job)).resolve()
                except Exception:
                    job_root = None
            if job_root is not None:
                launch_path = (
                    job_root / "r3_inputs" / "tracking_launch_manifest.json"
                ).resolve()
                if launch_path.is_file() and launch_path.is_relative_to(job_root):
                    launch = self._read_object(launch_path)
                    record = launch.get("shot_boundaries")
                    if isinstance(record, Mapping):
                        raw_path = str(record.get("path") or "").strip()
                        raw_sha = str(record.get("sha256") or "").strip()
                        if raw_path or raw_sha:
                            if not raw_path or len(raw_sha) != 64:
                                raise TrackingContractError(
                                    "R1 launch shot boundary contract is incomplete."
                                )
                            path = Path(raw_path).expanduser().resolve()
                            expected_sha = raw_sha.lower()
                            source = "SERVER_OWNED_R1_LAUNCH_MANIFEST"

        if path is None:
            return None, None
        if not path.is_file():
            raise TrackingContractError(
                "Immutable reviewed shot boundary artifact is missing."
            )
        actual_sha = sha256_file(path)
        if expected_sha != actual_sha:
            raise TrackingContractError(
                "Immutable reviewed shot boundary SHA-256 mismatch."
            )

        document = self._read_object(path)
        reviewed_shots = _canonical_reviewed_shots(
            document,
            frame_count=frame_count,
        )
        return reviewed_shots, {
            "source": source,
            "sha256": actual_sha,
            "path": str(path),
        }

    @staticmethod
    def _normalize_reviewed_shot_contract(
        payload: dict[str, Any],
        *,
        reviewed_shots: list[dict[str, Any]],
        reviewed_source: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Repair only shot metadata using immutable reviewed boundaries.

        Tracking states, bboxes, confidence values, detections, and identity
        decisions are preserved byte-for-value. This compatibility path fixes
        only the adapter's historical local-shot/global-frame translation.
        """

        normalized = copy.deepcopy(payload)
        frames = normalized.get("frames")
        raw_shots = normalized.get("shots")
        if not isinstance(frames, list) or not isinstance(raw_shots, list):
            return normalized

        provenance = (
            dict(normalized.get("provenance") or {})
            if isinstance(normalized.get("provenance"), Mapping)
            else {}
        )
        selection_view = provenance.get("selection_anchor_view")
        offset = None
        if isinstance(selection_view, Mapping):
            raw_offset = selection_view.get("source_offset_frame")
            if raw_offset is not None:
                try:
                    offset = int(raw_offset)
                except (TypeError, ValueError) as exc:
                    raise TrackingContractError(
                        "Timeline selection anchor offset is invalid."
                    ) from exc

        runtime_by_source: dict[str, dict[str, Any]] = {}
        runtime_translation_count = 0
        for raw_shot in raw_shots:
            if not isinstance(raw_shot, Mapping):
                continue
            source_shot_id = _source_shot_for_runtime_row(
                raw_shot,
                reviewed_shots=reviewed_shots,
                source_offset_frame=offset,
            )
            if source_shot_id is None:
                continue
            if source_shot_id in runtime_by_source:
                raise TrackingContractError(
                    "Multiple runtime shots map to one reviewed source shot."
                )
            runtime_by_source[source_shot_id] = dict(raw_shot)
            runtime_translation_count += 1

        canonical_shots: list[dict[str, Any]] = []
        for reviewed in reviewed_shots:
            shot_id = str(reviewed["shot_id"])
            row = copy.deepcopy(reviewed)
            runtime = runtime_by_source.get(shot_id)
            if runtime is not None:
                reserved = {
                    "shot_index",
                    "shot_id",
                    "start_frame",
                    "end_frame",
                    "end_frame_inclusive",
                    "frame_count",
                    "cut_in_frame",
                    "cut_out_frame",
                }
                for key, value in runtime.items():
                    if key not in reserved:
                        row[key] = copy.deepcopy(value)
                row["runtime_shot_id"] = str(runtime.get("shot_id") or "")
                if "start_frame" in runtime:
                    row["runtime_local_start_frame"] = int(
                        runtime["start_frame"]
                    )
                runtime_end = runtime.get(
                    "end_frame_inclusive",
                    runtime.get("end_frame"),
                )
                if runtime_end is not None:
                    row["runtime_local_end_frame_inclusive"] = int(runtime_end)
                row["processed_start_frame"] = max(
                    int(reviewed["start_frame"]),
                    offset if offset is not None else int(reviewed["start_frame"]),
                )
            elif (
                offset is not None
                and int(reviewed["end_frame_inclusive"]) < offset
                and not row.get("status")
            ):
                row["status"] = "OUTSIDE_SELECTION_ANCHOR_VIEW"
            canonical_shots.append(row)

        corrected_frame_shot_id_count = 0
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            try:
                frame_index = int(frame.get("frame_index", -1))
            except (TypeError, ValueError):
                continue
            expected_shot_id = _shot_id_for_frame(
                reviewed_shots,
                frame_index,
            )
            if expected_shot_id is None:
                continue
            if str(frame.get("shot_id") or "") != expected_shot_id:
                frame["shot_id"] = expected_shot_id
                corrected_frame_shot_id_count += 1

        normalized["shots"] = canonical_shots
        provenance["reviewed_shot_contract_normalization"] = {
            "policy": REVIEWED_SHOT_NORMALIZATION_POLICY,
            "source": str(reviewed_source.get("source") or ""),
            "reviewed_shot_boundaries_sha256": str(
                reviewed_source.get("sha256") or ""
            ),
            "reviewed_shot_count": len(reviewed_shots),
            "runtime_shot_translation_count": runtime_translation_count,
            "corrected_frame_shot_id_count": corrected_frame_shot_id_count,
            "reviewed_shot_metadata_full_source_coverage": True,
            "timeline_frame_coverage": (
                "TRACKED_RANGE_ONLY"
                if isinstance(normalized.get("tracked_range"), Mapping)
                else "FULL_VIDEO"
            ),
            "metadata_only": True,
            "tracking_state_modified": False,
            "bbox_modified": False,
            "identity_decision_modified": False,
            "automatic_target_confirmation": False,
        }
        normalized["provenance"] = provenance
        return normalized

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
        frame_count = _video_frame_count(payload)
        tracked_start, tracked_end = _tracked_frame_range(
            payload,
            frame_count=frame_count,
        )

        shots = payload.get("shots")
        if not isinstance(shots, list):
            raise TrackingContractError("Target timeline shots must be an array.")
        canonical_shots = _validate_public_shots_for_tracked_range(
            shots,
            frame_count=frame_count,
            tracked_start=tracked_start,
            tracked_end=tracked_end,
        )

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
        expected_frame_count = tracked_end - tracked_start + 1
        if len(frames) != expected_frame_count:
            raise TrackingContractError(
                "Target timeline frame count does not match tracked_range."
            )

        seen_frames: set[int] = set()
        for frame in frames:
            if not isinstance(frame, dict):
                raise TrackingContractError("Target timeline contains an invalid frame.")
            missing_frame = REQUIRED_FRAME_FIELDS.difference(frame)
            if missing_frame:
                frame_label = frame.get("frame_index", "unknown")
                raise TrackingContractError(
                    "Target timeline frame "
                    f"{frame_label} is missing required fields: "
                    + ", ".join(sorted(missing_frame))
                )
            try:
                frame_index = int(frame["frame_index"])
                time_seconds = float(frame["time_seconds"])
            except (TypeError, ValueError) as exc:
                raise TrackingContractError(
                    "Target timeline frame index/time is invalid."
                ) from exc
            if (
                frame_index < tracked_start
                or frame_index > tracked_end
                or frame_index >= frame_count
                or time_seconds < 0
                or frame_index in seen_frames
            ):
                raise TrackingContractError(
                    "Target timeline frame index/time is invalid."
                )
            seen_frames.add(frame_index)

            expected_shot_id = _shot_id_for_frame(
                canonical_shots,
                frame_index,
            )
            if expected_shot_id is None:
                raise TrackingContractError(
                    "Target timeline frame is outside reviewed shot coverage."
                )
            if str(frame.get("shot_id") or "") != expected_shot_id:
                raise TrackingContractError(
                    "Target timeline frame shot_id conflicts with reviewed shot range."
                )

            state = str(frame.get("state") or "")
            if state not in TIMELINE_STATES:
                raise TrackingContractError("Target timeline state is invalid.")
            if not str(frame.get("identity_source") or "").strip():
                raise TrackingContractError(
                    "Target timeline identity_source is invalid."
                )
            if not isinstance(frame.get("review_required"), bool):
                raise TrackingContractError(
                    "Target timeline review_required is invalid."
                )
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
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in bbox
                )
            ):
                raise TrackingContractError("Target timeline bbox is invalid.")
            if state in NULL_BBOX_STATES and bbox is not None:
                raise TrackingContractError(
                    "Uncertain target frame contains a forbidden bbox."
                )

        if seen_frames != set(range(tracked_start, tracked_end + 1)):
            raise TrackingContractError(
                "Target timeline frame indices are not complete and contiguous "
                "within tracked_range."
            )


def _tracked_frame_range(
    payload: Mapping[str, Any],
    *,
    frame_count: int,
) -> tuple[int, int]:
    tracked = payload.get("tracked_range")
    if isinstance(tracked, Mapping):
        raw_start = tracked.get("source_start_frame")
        raw_end = tracked.get("source_end_frame_inclusive")
        if raw_start is not None or raw_end is not None:
            try:
                start = int(raw_start)
                end = int(raw_end)
            except (TypeError, ValueError) as exc:
                raise TrackingContractError(
                    "Target timeline tracked_range is invalid."
                ) from exc
            if start < 0 or end < start or end >= frame_count:
                raise TrackingContractError(
                    "Target timeline tracked_range is outside video bounds."
                )
            return start, end

    frames = payload.get("frames")
    if isinstance(frames, list) and frames:
        try:
            indices = [int(row["frame_index"]) for row in frames if isinstance(row, Mapping)]
        except (KeyError, TypeError, ValueError) as exc:
            raise TrackingContractError(
                "Target timeline frame index is invalid."
            ) from exc
        if indices:
            start = min(indices)
            end = max(indices)
            if start < 0 or end >= frame_count:
                raise TrackingContractError(
                    "Target timeline frames are outside video bounds."
                )
            return start, end

    return 0, frame_count - 1


def _video_frame_count(payload: Mapping[str, Any]) -> int:
    video = payload.get("video")
    if not isinstance(video, Mapping):
        raise TrackingContractError("Target timeline video must be an object.")
    try:
        value = float(video["frame_count"])
        frame_count = int(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackingContractError(
            "Target timeline video.frame_count is invalid."
        ) from exc
    if frame_count <= 0 or value != frame_count:
        raise TrackingContractError(
            "Target timeline video.frame_count is invalid."
        )
    return frame_count


def _canonical_reviewed_shots(
    document: Mapping[str, Any],
    *,
    frame_count: int,
) -> list[dict[str, Any]]:
    rows = (
        document.get("shots")
        or document.get("boundaries")
        or document.get("shot_boundaries")
        or []
    )
    if not isinstance(rows, list) or not rows:
        raise TrackingContractError("Reviewed shot boundaries are empty.")

    normalized: list[dict[str, Any]] = []
    for source_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TrackingContractError(
                "Reviewed shot boundary contains a non-object row."
            )
        try:
            start = int(raw.get("start_frame", -1))
            end = int(
                raw.get(
                    "end_frame_inclusive",
                    raw.get("end_frame", -1),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TrackingContractError(
                "Reviewed shot boundary frame range is invalid."
            ) from exc
        shot_id = str(raw.get("shot_id") or f"shot_{source_index:04d}").strip()
        if not shot_id or start < 0 or end < start:
            raise TrackingContractError(
                "Reviewed shot boundary frame range is invalid."
            )
        row = copy.deepcopy(dict(raw))
        row.update(
            {
                "shot_index": source_index,
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                "cut_in_frame": None if start == 0 else start,
                "cut_out_frame": None if end == frame_count - 1 else end + 1,
            }
        )
        normalized.append(row)

    normalized.sort(
        key=lambda row: (
            int(row["start_frame"]),
            int(row["end_frame_inclusive"]),
            str(row["shot_id"]),
        )
    )
    for index, row in enumerate(normalized):
        row["shot_index"] = index
    return _validate_public_shots(normalized, frame_count=frame_count)


def _validate_public_shots_for_tracked_range(
    shots: list[Any],
    *,
    frame_count: int,
    tracked_start: int,
    tracked_end: int,
) -> list[dict[str, Any]]:
    """Validate shot metadata for either full-video or anchor-forward timelines.

    Canonical scene-target tracking may start from a user-confirmed source frame.
    In that case target_timeline.json intentionally contains only the forward
    tracked range; pre-anchor frames are not fabricated as ABSENT/SEARCHING.
    Reviewed shot metadata may still cover the full source video.
    """

    if not shots:
        raise TrackingContractError("Target timeline reviewed shots are empty.")

    # Preserve the strict historical full-video contract when it applies.
    try:
        return _validate_public_shots(shots, frame_count=frame_count)
    except TrackingContractError:
        pass

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_end: int | None = None
    for index, raw in enumerate(shots):
        if not isinstance(raw, Mapping):
            raise TrackingContractError("Target timeline shot is invalid.")
        shot_id = str(raw.get("shot_id") or "").strip()
        try:
            start = int(raw.get("start_frame", -1))
            end = int(raw.get("end_frame_inclusive", raw.get("end_frame", -1)))
        except (TypeError, ValueError) as exc:
            raise TrackingContractError(
                "Target timeline shot frame range is invalid."
            ) from exc
        if (
            not shot_id
            or shot_id in seen_ids
            or start < 0
            or end < start
            or end >= frame_count
            or (previous_end is not None and start != previous_end + 1)
        ):
            raise TrackingContractError(
                "Target timeline shots do not form one unique contiguous coverage."
            )
        row = copy.deepcopy(dict(raw))
        row.update(
            {
                "shot_index": index,
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                "cut_in_frame": None if start == 0 else start,
                "cut_out_frame": None if end == frame_count - 1 else end + 1,
            }
        )
        normalized.append(row)
        seen_ids.add(shot_id)
        previous_end = end

    coverage_start = int(normalized[0]["start_frame"])
    coverage_end = int(normalized[-1]["end_frame_inclusive"])
    if coverage_start > tracked_start or coverage_end < tracked_end:
        raise TrackingContractError(
            "Target timeline shots do not cover tracked_range."
        )
    return normalized


def _validate_public_shots(
    shots: list[Any],
    *,
    frame_count: int,
) -> list[dict[str, Any]]:
    if not shots:
        raise TrackingContractError("Target timeline reviewed shots are empty.")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    expected_start = 0
    for index, raw in enumerate(shots):
        if not isinstance(raw, Mapping):
            raise TrackingContractError("Target timeline shot is invalid.")
        shot_id = str(raw.get("shot_id") or "").strip()
        try:
            start = int(raw.get("start_frame", -1))
            end = int(
                raw.get(
                    "end_frame_inclusive",
                    raw.get("end_frame", -1),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TrackingContractError(
                "Target timeline shot frame range is invalid."
            ) from exc
        if (
            not shot_id
            or shot_id in seen_ids
            or start != expected_start
            or end < start
            or end >= frame_count
        ):
            raise TrackingContractError(
                "Target timeline shots do not form one unique contiguous coverage."
            )
        observed_frame_count = raw.get("frame_count")
        if observed_frame_count is not None:
            try:
                if int(observed_frame_count) != end - start + 1:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise TrackingContractError(
                    "Target timeline shot frame_count is inconsistent."
                ) from exc
        observed_index = raw.get("shot_index")
        if observed_index is not None:
            try:
                if int(observed_index) != index:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise TrackingContractError(
                    "Target timeline shot_index is inconsistent."
                ) from exc
        row = copy.deepcopy(dict(raw))
        row.update(
            {
                "shot_index": index,
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                "cut_in_frame": None if start == 0 else start,
                "cut_out_frame": None if end == frame_count - 1 else end + 1,
            }
        )
        normalized.append(row)
        seen_ids.add(shot_id)
        expected_start = end + 1

    if expected_start != frame_count:
        raise TrackingContractError(
            "Target timeline shots do not cover the complete video."
        )
    return normalized


def _source_shot_for_runtime_row(
    raw_shot: Mapping[str, Any],
    *,
    reviewed_shots: list[dict[str, Any]],
    source_offset_frame: int | None,
) -> str | None:
    raw_id = str(raw_shot.get("shot_id") or "").strip()
    try:
        raw_start = int(raw_shot.get("start_frame", -1))
        raw_end = int(
            raw_shot.get(
                "end_frame_inclusive",
                raw_shot.get("end_frame", -1),
            )
        )
    except (TypeError, ValueError):
        return None

    # Newer adapter outputs already use source-video frame coordinates and
    # reviewed source shot IDs. Recognize that canonical contract before trying
    # the historical anchor-offset translation. Otherwise a global shot such as
    # shot_0006 (468-555) can be shifted again by the anchor offset and collide
    # with later reviewed shots, causing a false duplicate-mapping failure.
    exact = [
        shot
        for shot in reviewed_shots
        if str(shot["shot_id"]) == raw_id
        and int(shot["start_frame"]) == raw_start
        and int(shot["end_frame_inclusive"]) == raw_end
    ]
    if len(exact) == 1:
        return raw_id
    if len(exact) > 1:
        raise TrackingContractError(
            "Runtime shot range ambiguously matches reviewed shots."
        )

    # Compatibility path for historical adapter output whose shot ranges were
    # local to the selection-anchor suffix. Only rows that are not already an
    # exact reviewed source shot are translated by source_offset_frame.
    if source_offset_frame is not None and raw_start >= 0 and raw_end >= raw_start:
        source_start = source_offset_frame + raw_start
        source_end = source_offset_frame + raw_end
        matches = [
            shot
            for shot in reviewed_shots
            if int(shot["start_frame"]) <= source_start
            and source_end <= int(shot["end_frame_inclusive"])
        ]
        if len(matches) == 1:
            return str(matches[0]["shot_id"])
        if len(matches) > 1:
            raise TrackingContractError(
                "Runtime shot range ambiguously maps to reviewed shots."
            )

    return None


def _shot_id_for_frame(
    shots: list[dict[str, Any]],
    frame_index: int,
) -> str | None:
    for shot in shots:
        if (
            int(shot["start_frame"])
            <= frame_index
            <= int(shot["end_frame_inclusive"])
        ):
            return str(shot["shot_id"])
    return None


def _phase3c_identity_source(frame: Mapping[str, Any]) -> str:
    selected_detection_id = (
        frame.get("runtime_selected_detection_id")
        or frame.get("selected_detection_id")
    )
    if selected_detection_id:
        return "RFDETR_ASSOCIATED_DETECTION"

    bbox_source = str(frame.get("bbox_source") or "").strip()
    if bbox_source and bbox_source.upper() != "NONE":
        return bbox_source
    return "NONE"


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
