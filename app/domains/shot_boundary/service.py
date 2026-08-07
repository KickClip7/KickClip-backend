from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.detection_cache import PlayerDetectionCache
from app.domains.highlight.model import HighlightRevision
from app.domains.highlight.player_detector import (
    PlayerDetection,
    PlayerDetector,
    PlayerDetectorError,
    create_observation_detector,
)
from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent
from app.storage.local_storage import LocalStorage

from .model import ShotBoundaryReviewDecision, ShotBoundaryReviewSession
from .runtime_contract import (
    SceneTargetShotBoundaryContractError,
    validate_scene_target_selection_contract,
)
from .schema import (
    ShotBoundaryConfirmRequest,
    ShotBoundaryConfirmResponse,
    ShotBoundaryDraftUpdate,
    ShotBoundaryReviewResponse,
)

SCENE_VIDEO_ARTIFACT_TYPE = "SCENE_VIDEO"
DRAFT_ARTIFACT_TYPE = "SHOT_BOUNDARY_DRAFT"
REVIEWED_ARTIFACT_TYPE = "REVIEWED_SHOT_BOUNDARIES"
AUTO_ARTIFACT_TYPE = "AUTO_SHOT_BOUNDARIES"
DETECTIONS_ARTIFACT_TYPE = "SCENE_PLAYER_DETECTIONS"
OBSERVATIONS_ARTIFACT_TYPE = "SCENE_RFDETR_OBSERVATIONS"

# Initial player-candidate discovery is intentionally much cheaper than final
# target tracking. Action Spotting already tells us *when* the interesting
# event happened, and automatic shot boundaries tell us where broadcast shots
# begin/end. Candidate discovery therefore runs RF-DETR only on 1-3
# representative frames from the event-near shots instead of scanning the
# whole event clip at a fixed FPS.
#
# Downstream responsibilities remain unchanged:
#   sampled RF-DETR rows
#   -> frozen same-shot candidate grouping / quality filtering
#   -> candidate fragment grouping / duplicate suppression
#   -> Top-K recommendation
#   -> explicit user selection
#   -> canonical target-centric E2E tracking.
FAST_CANDIDATE_DETECTION_POLICY_VERSION = (
    "ACTION_SPOTTING_SHOT_LOCAL_TRIPLET_RFDETR_V5_WIDE_OBSERVATION"
)
FAST_CANDIDATE_FRAMES_PER_SHOT = 3
FAST_CANDIDATE_PRE_SHOTS_BY_LABEL: dict[str, int] = {
    "goal": 2,
    "shot": 2,
    "foul": 2,
    "card": 1,
    "freekick": 2,
    "free_kick": 2,
    "free kick": 2,
}
FAST_CANDIDATE_POST_SHOTS_BY_LABEL: dict[str, int] = {
    # Goal broadcasts commonly show scorer celebration/close-up after A, so
    # retain a slightly wider post-event shot envelope while still sampling
    # at most three frames per shot.
    "goal": 5,
    "shot": 2,
    "foul": 3,
    "card": 4,
    "freekick": 2,
    "free_kick": 2,
    "free kick": 2,
}
FAST_CANDIDATE_DEFAULT_PRE_SHOTS = 2
FAST_CANDIDATE_DEFAULT_POST_SHOTS = 3


def _event_near_shot_counts(event: TimelineEvent) -> tuple[int, int]:
    label = str(event.label or event.event_type or "").strip().lower()
    compact = label.replace("-", "_")
    before = FAST_CANDIDATE_PRE_SHOTS_BY_LABEL.get(
        label,
        FAST_CANDIDATE_PRE_SHOTS_BY_LABEL.get(
            compact,
            FAST_CANDIDATE_DEFAULT_PRE_SHOTS,
        ),
    )
    after = FAST_CANDIDATE_POST_SHOTS_BY_LABEL.get(
        label,
        FAST_CANDIDATE_POST_SHOTS_BY_LABEL.get(
            compact,
            FAST_CANDIDATE_DEFAULT_POST_SHOTS,
        ),
    )
    return max(0, int(before)), max(0, int(after))


def _representative_frames_for_shot(
    *,
    start_frame: int,
    end_frame: int,
    event_frame: int | None,
    relative_to_event: int = 0,
) -> list[int]:
    """Return a shot-local contiguous 1-3 frame representative burst.

    V4 sampled the quarter/middle/three-quarter positions of each shot.  That
    was cheap for RF-DETR, but those observations can be seconds apart.  The
    frozen scene-candidate runtime groups detections into short same-shot
    tracklets, so isolated observations were correctly rejected and could
    leave ``scene_candidates.json`` empty.

    V5 keeps the exact same RF-DETR budget (at most three frames per shot) but
    makes the three frames contiguous.  This preserves enough temporal support
    for the existing same-shot grouping without returning to dense detection.

    Anchor placement is event-conditioned:
    - event shot: centered on Action Spotting event A;
    - shot before A: near the latter part of the shot (build-up);
    - shot after A: near the early part of the shot (reaction/celebration);
    - fallback: center of the shot.
    """

    if end_frame < start_frame:
        return []
    length = end_frame - start_frame + 1
    if length <= FAST_CANDIDATE_FRAMES_PER_SHOT:
        return list(range(start_frame, end_frame + 1))

    if event_frame is not None and start_frame <= event_frame <= end_frame:
        anchor = int(event_frame)
        anchor_strategy = "ACTION_SPOTTING_EVENT"
    elif relative_to_event < 0:
        anchor = start_frame + int(round((length - 1) * 0.75))
        anchor_strategy = "PRE_EVENT_LATE_SHOT"
    elif relative_to_event > 0:
        anchor = start_frame + int(round((length - 1) * 0.25))
        anchor_strategy = "POST_EVENT_EARLY_SHOT"
    else:
        anchor = start_frame + int(round((length - 1) * 0.50))
        anchor_strategy = "SHOT_CENTER"

    # Shift the 3-frame window at the shot edges rather than shrinking it.
    # For every shot with >=3 frames this therefore returns exactly 3
    # consecutive, in-shot frame indices.
    first = anchor - 1
    last_first = end_frame - (FAST_CANDIDATE_FRAMES_PER_SHOT - 1)
    first = min(max(first, start_frame), last_first)
    frames = list(
        range(first, first + FAST_CANDIDATE_FRAMES_PER_SHOT)
    )
    # ``anchor_strategy`` is intentionally local documentation only; the
    # caller records the relation/frames in sampling provenance.
    _ = anchor_strategy
    return frames


def _sampled_candidate_frames(
    *,
    fps: float,
    frame_count: int,
    event_local_sec: float,
    shots: list[dict[str, Any]],
    pre_shot_count: int,
    post_shot_count: int,
) -> tuple[list[int], dict[str, Any]]:
    """Sample only event-near automatic shots, at 1-3 frames per shot.

    The complete Action Spotting scene clip remains the search envelope (for a
    Goal this is typically the existing ~45 s 15-before/30-after clip). We do
    not run RF-DETR over every frame in that envelope. Instead, the shot that
    contains A plus a small number of neighbouring shots are selected and each
    contributes at most three representative frames.
    """

    if fps <= 0 or frame_count <= 0:
        raise ValueError("Scene FPS and frame count must be positive.")

    duration_sec = frame_count / fps
    event_local_sec = min(max(0.0, event_local_sec), max(0.0, duration_sec))
    event_frame = min(
        frame_count - 1,
        max(0, int(round(event_local_sec * fps))),
    )

    normalized_shots: list[dict[str, Any]] = []
    for fallback_index, row in enumerate(shots):
        try:
            start = int(row["start_frame"])
            end = int(row.get("end_frame_inclusive", row.get("end_frame")))
        except (KeyError, TypeError, ValueError):
            continue
        start = min(max(0, start), frame_count - 1)
        end = min(max(start, end), frame_count - 1)
        normalized_shots.append(
            {
                "shot_index": int(row.get("shot_index", fallback_index)),
                "shot_id": str(row.get("shot_id") or f"shot_{fallback_index:04d}"),
                "start_frame": start,
                "end_frame_inclusive": end,
            }
        )

    normalized_shots.sort(
        key=lambda row: (
            int(row["start_frame"]),
            int(row["end_frame_inclusive"]),
            int(row["shot_index"]),
        )
    )
    if not normalized_shots:
        normalized_shots = [
            {
                "shot_index": 0,
                "shot_id": "shot_0000",
                "start_frame": 0,
                "end_frame_inclusive": frame_count - 1,
            }
        ]

    event_shot_position = next(
        (
            index
            for index, row in enumerate(normalized_shots)
            if int(row["start_frame"])
            <= event_frame
            <= int(row["end_frame_inclusive"])
        ),
        None,
    )
    if event_shot_position is None:
        event_shot_position = min(
            range(len(normalized_shots)),
            key=lambda index: abs(
                (
                    int(normalized_shots[index]["start_frame"])
                    + int(normalized_shots[index]["end_frame_inclusive"])
                )
                / 2.0
                - event_frame
            ),
        )

    first = max(0, event_shot_position - max(0, int(pre_shot_count)))
    last = min(
        len(normalized_shots) - 1,
        event_shot_position + max(0, int(post_shot_count)),
    )
    selected_shots = normalized_shots[first : last + 1]

    frames: set[int] = set()
    sampled_shots: list[dict[str, Any]] = []
    for absolute_position, row in enumerate(selected_shots, start=first):
        is_event_shot = absolute_position == event_shot_position
        relative_to_event = absolute_position - event_shot_position
        representative = _representative_frames_for_shot(
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame_inclusive"]),
            event_frame=(event_frame if is_event_shot else None),
            relative_to_event=relative_to_event,
        )
        frames.update(representative)
        sampled_shots.append(
            {
                **row,
                "is_event_shot": is_event_shot,
                "relative_to_event_shot": relative_to_event,
                "sampling_mode": "CONTIGUOUS_SHOT_LOCAL_TRIPLET",
                "sampled_frames": representative,
                "sampled_frame_count": len(representative),
            }
        )

    ordered = sorted(frame for frame in frames if 0 <= frame < frame_count)
    return ordered, {
        "strategy": "EVENT_NEAR_SHOTS_CONTIGUOUS_REPRESENTATIVE_TRIPLETS",
        "scene_duration_sec": duration_sec,
        "scene_frame_count": frame_count,
        "event_scene_local_sec": event_local_sec,
        "event_scene_local_frame": event_frame,
        "total_shot_count": len(normalized_shots),
        "event_shot_id": str(normalized_shots[event_shot_position]["shot_id"]),
        "event_shot_index": int(
            normalized_shots[event_shot_position]["shot_index"]
        ),
        "pre_event_shot_count": max(0, int(pre_shot_count)),
        "post_event_shot_count": max(0, int(post_shot_count)),
        "selected_shot_count": len(selected_shots),
        "frames_per_shot_max": FAST_CANDIDATE_FRAMES_PER_SHOT,
        "sampled_frame_count": len(ordered),
        "selected_shots": sampled_shots,
        "window_start_frame": int(selected_shots[0]["start_frame"]),
        "window_end_frame_inclusive": int(
            selected_shots[-1]["end_frame_inclusive"]
        ),
        "window_start_sec": int(selected_shots[0]["start_frame"]) / fps,
        "window_end_sec": (
            int(selected_shots[-1]["end_frame_inclusive"]) + 1
        )
        / fps,
    }


def _normalize_detection_to_frame(
    detection: PlayerDetection,
    *,
    frame_width: int,
    frame_height: int,
) -> PlayerDetection | None:
    if frame_width <= 0 or frame_height <= 0:
        return None
    values = [float(value) for value in detection.bbox_xyxy]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    max_x = float(frame_width - 1)
    max_y = float(frame_height - 1)
    x1 = min(max(values[0], 0.0), max_x)
    y1 = min(max(values[1], 0.0), max_y)
    x2 = min(max(values[2], 0.0), max_x)
    y2 = min(max(values[3], 0.0), max_y)
    if x2 <= x1 or y2 <= y1:
        return None
    confidence = float(detection.confidence)
    if not math.isfinite(confidence):
        return None
    return PlayerDetection(
        bbox_xyxy=[x1, y1, x2, y2],
        confidence=confidence,
        class_id=int(detection.class_id),
        class_name=str(detection.class_name),
    )


def _bbox_ratios(
    detection: PlayerDetection,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in detection.bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    height_ratio = height / max(1.0, float(frame_height))
    area_ratio = (width * height) / max(1.0, float(frame_width * frame_height))
    return height_ratio, area_ratio


def _is_closeup_frame(
    detections: list[PlayerDetection],
    *,
    frame_width: int,
    frame_height: int,
    observation_class_ids: frozenset[int],
    observation_conf_threshold: float,
    closeup_min_height_ratio: float,
    closeup_min_area_ratio: float,
) -> bool:
    # Ball detections must never turn a frame into a close-up. The broadcast
    # close-up decision is based on person-like roles only.
    person_class_ids = observation_class_ids.intersection({0, 1, 2, 3})
    for detection in detections:
        if (
            int(detection.class_id) not in person_class_ids
            or float(detection.confidence) < observation_conf_threshold
        ):
            continue
        height_ratio, area_ratio = _bbox_ratios(
            detection,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if (
            height_ratio >= closeup_min_height_ratio
            or area_ratio >= closeup_min_area_ratio
        ):
            return True
    return False


def _select_candidate_detections_for_frame(
    detections: list[PlayerDetection],
    *,
    frame_width: int,
    frame_height: int,
    is_closeup: bool,
    play_class_ids: frozenset[int],
    play_conf_threshold: float,
    observation_class_ids: frozenset[int],
    observation_conf_threshold: float,
    crop_top_k: int,
    crop_min_conf: float,
    crop_min_height_ratio: float,
    crop_min_area_ratio: float,
) -> tuple[list[PlayerDetection], list[dict[str, Any]]]:
    """Apply the exact wide-vs-observation policy to one sampled frame.

    Wide frames feed player/goalkeeper rows directly to scene-target selection
    at TRACKING_PLAY_CONF_THRESHOLD. Close-up frames first preserve all five
    RF-DETR roles, apply the stricter observation/crop gates, rank the top-K
    visible observation boxes, then expose only player/goalkeeper boxes as
    selectable candidates. Referee/staff remain explicit observation evidence
    and are never emitted as candidate detections.
    """

    observation_rows: list[dict[str, Any]] = []
    for detection in detections:
        class_id = int(detection.class_id)
        confidence = float(detection.confidence)
        height_ratio, area_ratio = _bbox_ratios(
            detection,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        observation_eligible = (
            class_id in observation_class_ids
            and confidence >= observation_conf_threshold
        )
        crop_eligible = (
            observation_eligible
            and class_id in {0, 1, 2, 3}
            and confidence >= crop_min_conf
            and height_ratio >= crop_min_height_ratio
            and area_ratio >= crop_min_area_ratio
        )
        observation_rows.append(
            {
                "detection": detection,
                "class_id": class_id,
                "confidence": confidence,
                "height_ratio": height_ratio,
                "area_ratio": area_ratio,
                "observation_eligible": observation_eligible,
                "crop_eligible": crop_eligible,
                "observation_rank": None,
                "candidate_eligible": False,
            }
        )

    if not is_closeup:
        candidates = [
            row["detection"]
            for row in observation_rows
            if row["class_id"] in play_class_ids
            and row["confidence"] >= play_conf_threshold
        ]
        candidate_keys = {id(item) for item in candidates}
        for row in observation_rows:
            row["candidate_eligible"] = id(row["detection"]) in candidate_keys
        return candidates, observation_rows

    ranked = sorted(
        (row for row in observation_rows if row["crop_eligible"]),
        key=lambda row: (
            float(row["area_ratio"]),
            float(row["height_ratio"]),
            float(row["confidence"]),
        ),
        reverse=True,
    )[: max(1, int(crop_top_k))]
    for rank, row in enumerate(ranked, start=1):
        row["observation_rank"] = rank

    # OBSERVATION_CROP_TOP_K limits persisted close-up crops only; it must
    # not silently remove valid player/goalkeeper detections from the initial
    # candidate pool. Candidate eligibility uses the observation threshold,
    # while crop rank is recorded independently for review media.
    candidates = [
        row["detection"]
        for row in observation_rows
        if row["class_id"] in play_class_ids
        and row["observation_eligible"]
        and row["confidence"] >= max(
            play_conf_threshold, observation_conf_threshold
        )
    ]
    candidate_keys = {id(item) for item in candidates}
    for row in observation_rows:
        row["candidate_eligible"] = id(row["detection"]) in candidate_keys
    return candidates, observation_rows


class ShotBoundaryWorkflowError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int = 409,
        detail: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.detail = detail or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return _sha256(path)


class ShotBoundaryReviewService:
    """Bootstrap exact-event scene inputs while preserving human review authority."""

    def __init__(
        self,
        db: Session,
        *,
        settings: Settings | None = None,
        detector_factory: Callable[[Settings], PlayerDetector] = create_observation_detector,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.media = MediaAssetRepository(db)
        self.detector_factory = detector_factory

    def _scope(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> tuple[HighlightRevision, TimelineEvent, TimelineEvent, MediaAsset]:
        revision = self.db.get(HighlightRevision, revision_id)
        event = self.db.get(TimelineEvent, event_id)
        scene = self.db.get(TimelineEvent, scene_id)
        if (
            revision is None
            or revision.project_id != project.project_id
            or event is None
            or scene is None
            or event.match_id != project.match_id
            or scene.match_id != project.match_id
            or scene_id not in (revision.selected_scene_ids or [])
            or event.source_job_id != revision.action_spotting_job_id
            or scene.source_job_id != revision.action_spotting_job_id
        ):
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_PROVENANCE_MISMATCH",
                "Project, revision, event, scene and Action Spotting source do not match.",
                http_status=422,
            )
        job = revision.action_spotting_job
        source = job.media_asset if job is not None else None
        if source is None or source.match_id != project.match_id:
            raise ShotBoundaryWorkflowError(
                "SOURCE_VIDEO_NOT_READY",
                "The exact Action Spotting source video is unavailable.",
            )
        if (
            project.owner_id
            and project.owner_id != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_SESSION_NOT_FOUND",
                "Review session not found.",
                http_status=404,
            )
        return revision, event, scene, source

    def _latest_session(
        self,
        *,
        project_id: str,
        revision_id: str,
        event_id: str,
        scene_id: str,
        user_id: str,
    ) -> ShotBoundaryReviewSession | None:
        return self.db.scalar(
            select(ShotBoundaryReviewSession)
            .where(
                ShotBoundaryReviewSession.owner_user_id == user_id,
                ShotBoundaryReviewSession.project_id == project_id,
                ShotBoundaryReviewSession.revision_id == revision_id,
                ShotBoundaryReviewSession.event_id == event_id,
                ShotBoundaryReviewSession.scene_id == scene_id,
            )
            .order_by(ShotBoundaryReviewSession.session_revision.desc())
        )

    def status(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> ShotBoundaryReviewResponse:
        self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        session = self._latest_session(
            project_id=project.project_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            user_id=user.user_id,
        )
        if session is None:
            return ShotBoundaryReviewResponse(
                status="NOT_PREPARED",
                project_id=project.project_id,
                revision_id=revision_id,
                event_id=event_id,
                scene_id=scene_id,
            )
        return self._response(session)

    def prepare(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        new_review_revision: bool = False,
    ) -> ShotBoundaryReviewResponse:
        revision, event, scene, source = self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        source_path = self.storage.resolve_path(source.file_path)
        if not source_path.is_file():
            raise ShotBoundaryWorkflowError(
                "SOURCE_VIDEO_NOT_READY", "Source video file is missing."
            )
        source_sha = _sha256(source_path)
        if not source.sha256 or source.sha256 != source_sha:
            raise ShotBoundaryWorkflowError(
                "SOURCE_VIDEO_SHA256_MISMATCH",
                "Source video SHA-256 no longer matches its MediaAsset.",
            )
        existing = self._latest_session(
            project_id=project.project_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            user_id=user.user_id,
        )
        if existing is not None and not new_review_revision:
            metadata = existing.scene_video_artifact.metadata_ or {}
            if (
                metadata.get("source_video_asset_id") == source.asset_id
                and metadata.get("source_video_sha256") == source_sha
                and abs(
                    float(metadata.get("source_start_seconds", -1))
                    - float(scene.start_sec)
                )
                <= 1e-6
                and abs(
                    float(metadata.get("source_end_seconds", -1)) - float(scene.end_sec)
                )
                <= 1e-6
            ):
                return self._response(existing)

        scene_asset, scene_artifact = self._materialize_scene(
            project=project,
            revision=revision,
            event=event,
            scene=scene,
            source=source,
            source_path=source_path,
        )
        scene_path = self.storage.resolve_path(scene_asset.file_path)
        automatic = self._analyze_cuts(
            scene_path=scene_path,
            output_root=scene_path.parent,
            scene_id=scene_id,
            scene_sha=scene_asset.sha256 or _sha256(scene_path),
            fps=float(scene_asset.fps or 0),
            frame_count=int((scene_artifact.metadata_ or {}).get("frame_count") or 0),
        )
        draft_path = scene_path.parent / "shot_boundary_draft.v1.json"
        draft_sha = _atomic_json(draft_path, automatic)
        self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=DRAFT_ARTIFACT_TYPE,
            file_path=draft_path.relative_to(self.storage.project_root).as_posix(),
            mime_type="application/json",
            metadata_={
                "status": "WAITING_REVIEW",
                "revision_id": revision_id,
                "event_id": event_id,
                "scene_id": scene_id,
                "scene_video_sha256": scene_asset.sha256,
                "sha256": draft_sha,
                "automatic_confirmation": False,
            },
        )
        session_revision = (existing.session_revision + 1) if existing else 1
        session = ShotBoundaryReviewSession(
            owner_user_id=user.user_id,
            project_id=project.project_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            source_video_asset_id=source.asset_id,
            scene_video_asset_id=scene_asset.asset_id,
            scene_video_artifact_id=scene_artifact.artifact_id,
            status="WAITING_REVIEW",
            session_revision=session_revision,
            draft_revision=1,
            automatic_draft_json=automatic,
            draft_json=automatic,
            error_json={},
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return self._response(session)

    def prepare_for_candidate_discovery(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        new_review_revision: bool = False,
    ) -> ShotBoundaryReviewResponse:
        """Prepare automatic cuts as the default authoritative segmentation.

        Normal product flow does not pause for camera-cut approval.  The
        automatic detector is followed by structural/provenance validation and
        candidate-detection materialization.  Human boundary editing remains an
        exceptional fallback only when that structural gate rejects the result.

        Target identity is never confirmed here.
        """

        self.prepare(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            new_review_revision=new_review_revision,
        )
        self.prepare_candidate_discovery_inputs(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        return self.status(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )

    def _current_candidate_detection_policy(self) -> dict[str, Any]:
        play_class_ids = frozenset(self.settings.tracking_play_class_ids)
        observation_class_ids = frozenset(self.settings.observation_class_ids)
        return {
            "policy_version": FAST_CANDIDATE_DETECTION_POLICY_VERSION,
            "class_mapping": {
                "0": "player",
                "1": "goalkeeper",
                "2": "referee",
                "3": "staff",
                "4": "ball",
            },
            "rfdetr_base_conf_threshold": float(
                self.settings.RFDETR_BASE_CONF_THRESHOLD
            ),
            "tracking_play_classes": sorted(play_class_ids),
            "tracking_play_conf_threshold": float(
                self.settings.TRACKING_PLAY_CONF_THRESHOLD
            ),
            "observation_classes": sorted(observation_class_ids),
            "observation_conf_threshold": float(
                self.settings.OBSERVATION_CONF_THRESHOLD
            ),
            "observation_crop_top_k": int(
                self.settings.OBSERVATION_CROP_TOP_K
            ),
            "closeup_min_box_height_ratio": float(
                self.settings.CLOSEUP_MIN_BOX_HEIGHT_RATIO
            ),
            "closeup_min_box_area_ratio": float(
                self.settings.CLOSEUP_MIN_BOX_AREA_RATIO
            ),
            "closeup_crop_min_conf": float(
                self.settings.CLOSEUP_CROP_MIN_CONF
            ),
            "closeup_crop_min_height_ratio": float(
                self.settings.CLOSEUP_CROP_MIN_HEIGHT_RATIO
            ),
            "closeup_crop_min_area_ratio": float(
                self.settings.CLOSEUP_CROP_MIN_AREA_RATIO
            ),
            "tracking_iou_threshold": float(
                self.settings.TRACKING_IOU_THRESHOLD
            ),
            "tracking_tracker": str(self.settings.TRACKING_TRACKER),
            "tracking_tracker_applied_to_initial_candidate_discovery": False,
            "bbox_bounds_policy": "inclusive_width_minus_1_height_minus_1_v1",
        }

    def _detection_artifact_is_usable(
        self,
        artifact: Artifact,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> bool:
        metadata = artifact.metadata_ or {}
        path = self.storage.resolve_path(artifact.file_path)
        return (
            artifact.project_id == project.project_id
            and artifact.match_id == project.match_id
            and artifact.artifact_type == DETECTIONS_ARTIFACT_TYPE
            and metadata.get("revision_id") == revision_id
            and metadata.get("event_id") == event_id
            and metadata.get("scene_id") == scene_id
            and metadata.get("candidate_detection_policy")
            == self._current_candidate_detection_policy()
            and path.is_file()
            and _sha256(path) == metadata.get("sha256")
        )

    def prepare_candidate_discovery_inputs(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> dict[str, str]:
        """Materialize structurally safe automatic candidate-discovery inputs.

        This path deliberately creates neither a human decision nor a reviewed
        artifact. Target identity remains an explicit user choice downstream.
        """

        revision, _event, scene, source = self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        mapping = (revision.options or {}).get("candidate_pipeline_inputs") or {}
        existing = self.db.get(
            Artifact,
            str(
                mapping.get("shot_boundaries_artifact_id")
                or mapping.get("reviewed_shots_artifact_id")
                or ""
            ),
        )
        existing_detections = self.db.get(
            Artifact, str(mapping.get("detections_artifact_id") or "")
        )
        if (
            existing is not None
            and existing.artifact_type == AUTO_ARTIFACT_TYPE
            and existing_detections is not None
            and self._detection_artifact_is_usable(
                existing_detections,
                project=project,
                revision_id=revision_id,
                event_id=event_id,
                scene_id=scene_id,
            )
            and self._automatic_artifact_is_usable(
                existing,
                project=project,
                revision_id=revision_id,
                event_id=event_id,
                scene_id=scene_id,
            )
        ):
            return self._candidate_discovery_contract(
                existing,
                existing_detections,
                mapping=mapping,
            )
        if existing is not None and existing.artifact_type == REVIEWED_ARTIFACT_TYPE:
            reviewed = self.validate_candidate_contract(
                project=project,
                user=user,
                revision_id=revision_id,
                event_id=event_id,
                scene_id=scene_id,
            )
            reviewed_detections = self.db.get(
                Artifact, str(reviewed["detections_artifact_id"])
            )
            if (
                reviewed_detections is None
                or not self._detection_artifact_is_usable(
                    reviewed_detections,
                    project=project,
                    revision_id=revision_id,
                    event_id=event_id,
                    scene_id=scene_id,
                )
            ):
                # Keep the immutable human-reviewed boundary artifact, but
                # regenerate candidate detections under the current RF-DETR
                # wide/close-up policy instead of reusing stale 0.25-only
                # artifacts from earlier revisions.
                review_session = self._required_session(
                    project.project_id,
                    revision_id,
                    event_id,
                    scene_id,
                    user.user_id,
                )
                reviewed_detections = self._materialize_detections(
                    session=review_session,
                    project=project,
                    revision=revision,
                )
                mapping = {
                    **mapping,
                    "detections_artifact_id": reviewed_detections.artifact_id,
                }
                revision.options = {
                    **(revision.options or {}),
                    "candidate_pipeline_inputs": mapping,
                }
                self.db.commit()
            return {
                "shot_boundaries_artifact_id": reviewed[
                    "reviewed_shots_artifact_id"
                ],
                "shot_boundaries_sha256": reviewed[
                    "reviewed_shot_boundaries_sha256"
                ],
                "detections_artifact_id": reviewed_detections.artifact_id,
                "boundary_origin": "HUMAN_REVIEWED",
            }

        prepared = self.prepare(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        session = self._required_session(
            project.project_id, revision_id, event_id, scene_id, user.user_id
        )
        scene_metadata = session.scene_video_artifact.metadata_ or {}
        frame_count = int(scene_metadata.get("frame_count") or 0)
        fps = float(scene_metadata.get("fps") or 0)
        scene_sha = str(scene_metadata.get("scene_video_sha256") or "")
        automatic = dict(session.automatic_draft_json or {})
        shots = self.canonicalize_shots(
            list(automatic.get("shots") or []), frame_count=frame_count
        )
        errors = self.validate_shots(
            shots, frame_count=frame_count, require_shot_indexes=True
        )
        if automatic.get("scene_video_sha256") != scene_sha:
            errors.append("automatic boundary scene SHA does not match scene video")
        duration_seconds = frame_count / fps if fps > 0 else 0
        maximum_cut_count = min(100, max(4, math.ceil(duration_seconds * 4)))
        if len(shots) - 1 > maximum_cut_count:
            errors.append("automatic cut count exceeds the structural safety limit")
        if any(int(row["frame_count"]) <= 0 for row in shots):
            errors.append("every automatic shot must have a positive duration")
        if errors:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_REVIEW_REQUIRED",
                "Automatic shot boundaries require human correction before candidate discovery.",
                detail={
                    "reason": "AUTOMATIC_BOUNDARY_STRUCTURAL_GATE_FAILED",
                    "validation_errors": errors,
                    "review_status": prepared.status,
                },
            )

        automatic_shots = []
        for row in shots:
            automatic_shots.append(
                {
                    "shot_index": int(row["shot_index"]),
                    "shot_id": str(row["shot_id"]),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame_inclusive"]),
                    "end_frame_inclusive": int(row["end_frame_inclusive"]),
                    "frame_count": int(row["frame_count"]),
                    "cut_in_frame": row["cut_in_frame"],
                    "cut_out_frame": row["cut_out_frame"],
                    "start_time_sec": int(row["start_frame"]) / fps,
                    "end_time_sec": (int(row["end_frame_inclusive"]) + 1) / fps,
                    "boundary_state": "AUTO_DETECTED",
                }
            )
        document = {
            "schema_version": "kickclip.auto_shot_boundaries.v1",
            "artifact_type": AUTO_ARTIFACT_TYPE,
            "boundary_origin": "AUTO_DETECTED",
            "human_reviewed": False,
            "automatic_target_confirmation": False,
            "automatic_confirmation": False,
            "project_id": project.project_id,
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "source_video_asset_id": source.asset_id,
            "source_video_sha256": source.sha256,
            "scene_video_artifact_id": session.scene_video_artifact_id,
            "scene_video_sha256": scene_sha,
            "video": {
                "sha256": scene_sha,
                "fps": fps,
                "frame_count": frame_count,
            },
            "fps": fps,
            "frame_count": frame_count,
            "shots": automatic_shots,
            "structural_validation": {
                "status": "PASS",
                "complete_event_window_coverage": True,
                "gap_count": 0,
                "overlap_count": 0,
                "maximum_cut_count": maximum_cut_count,
                "observed_cut_count": len(shots) - 1,
            },
        }
        document["content_sha256"] = _canonical_sha256(document)
        auto_root = (
            self.storage.resolve_path(session.scene_video_asset.file_path).parent
            / "automatic"
            / document["content_sha256"][:24]
        )
        auto_path = auto_root / "auto_shot_boundaries.json"
        artifact_sha = _atomic_json(auto_path, document)
        configured_runtime_root = (
            self.settings.SCENE_DISCOVERY_PROJECT_ROOT
            or self.settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
            or self.settings.TRACKING_PROJECT_ROOT
        )
        runtime_root = (
            Path(configured_runtime_root).expanduser().resolve()
            if configured_runtime_root
            else (self.storage.project_root / ".tracking-runtime").resolve()
        )
        try:
            validate_scene_target_selection_contract(
                package_root=(
                    runtime_root
                    / "target_centric_tracking_scene_target_selection_v1"
                ),
                artifact_path=auto_path,
                video_sha256=scene_sha,
                frame_count=frame_count,
            )
        except SceneTargetShotBoundaryContractError as exc:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_REVIEW_REQUIRED",
                "Automatic shot boundaries are incompatible with candidate discovery.",
                detail={
                    "reason": exc.reason,
                    "runtime_message": exc.runtime_message,
                },
            ) from exc
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=AUTO_ARTIFACT_TYPE,
            file_path=auto_path.relative_to(self.storage.project_root).as_posix(),
            mime_type="application/json",
            metadata_={
                "status": "STRUCTURALLY_VALID",
                "revision_id": revision_id,
                "event_id": event_id,
                "scene_id": scene_id,
                "scene_video_sha256": scene_sha,
                "source_video_asset_id": source.asset_id,
                "source_video_sha256": source.sha256,
                "boundary_origin": "AUTO_DETECTED",
                "human_reviewed": False,
                "automatic_target_confirmation": False,
                "sha256": artifact_sha,
                "structural_validation": document["structural_validation"],
            },
        )
        detections = self._materialize_detections(
            session=session,
            project=project,
            revision=revision,
        )
        mapping = {
            "status": "MATERIALIZED",
            "scene_id": scene_id,
            "scene_video_asset_id": session.scene_video_asset_id,
            "scene_video_artifact_id": session.scene_video_artifact_id,
            "shot_boundaries_artifact_id": artifact.artifact_id,
            "shot_boundaries_sha256": artifact_sha,
            "boundary_origin": "AUTO_DETECTED",
            "human_reviewed": False,
            "detections_artifact_id": detections.artifact_id,
            "source_video_sha256": scene_sha,
            "action_spotting_source_video_sha256": source.sha256,
            "event_source_start_sec": float(scene.start_sec),
            "event_source_end_sec": float(scene.end_sec),
            "source_start_frame": round(
                float(scene.start_sec) * float(source.fps or fps)
            ),
            "source_frame_mapping_basis": "timeline_event_start_sec_x_source_asset_fps",
            "candidate_source_to_video_frame_offset": 0,
            "candidate_video_frame_mapping_basis": "scene_candidate_frame_index_is_scene_video_frame_index",
            "automatic_target_confirmation": False,
        }
        revision.options = {
            **(revision.options or {}),
            "candidate_pipeline_inputs": mapping,
        }
        self.db.commit()
        return self._candidate_discovery_contract(
            artifact, detections, mapping=mapping
        )

    def _automatic_artifact_is_usable(
        self,
        artifact: Artifact,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> bool:
        metadata = artifact.metadata_ or {}
        path = self.storage.resolve_path(artifact.file_path)
        return (
            artifact.project_id == project.project_id
            and artifact.match_id == project.match_id
            and metadata.get("revision_id") == revision_id
            and metadata.get("event_id") == event_id
            and metadata.get("scene_id") == scene_id
            and metadata.get("status") == "STRUCTURALLY_VALID"
            and metadata.get("boundary_origin") == "AUTO_DETECTED"
            and metadata.get("human_reviewed") is False
            and metadata.get("automatic_target_confirmation") is False
            and path.is_file()
            and _sha256(path) == metadata.get("sha256")
        )

    @staticmethod
    def _candidate_discovery_contract(
        boundaries: Artifact,
        detections: Artifact,
        *,
        mapping: dict[str, Any],
    ) -> dict[str, str]:
        return {
            "shot_boundaries_artifact_id": boundaries.artifact_id,
            "shot_boundaries_sha256": str(
                (boundaries.metadata_ or {}).get("sha256") or ""
            ),
            "detections_artifact_id": detections.artifact_id,
            "boundary_origin": str(mapping.get("boundary_origin") or ""),
        }

    def update_draft(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        payload: ShotBoundaryDraftUpdate,
    ) -> ShotBoundaryReviewResponse:
        self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        session = self._required_session(
            project.project_id, revision_id, event_id, scene_id, user.user_id
        )
        if session.status == "CONFIRMED":
            raise ShotBoundaryWorkflowError(
                "REVIEWED_SHOT_BOUNDARIES_IMMUTABLE",
                "Confirmed boundaries are immutable; start a new review revision.",
            )
        if payload.draft_revision != session.draft_revision:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_DRAFT_CONFLICT",
                "The draft changed on the server. Reload before saving.",
                detail={"current_draft_revision": session.draft_revision},
            )
        scene_sha = str(
            (session.scene_video_artifact.metadata_ or {}).get("scene_video_sha256")
            or ""
        )
        if payload.scene_video_sha256 and payload.scene_video_sha256 != scene_sha:
            raise ShotBoundaryWorkflowError(
                "SCENE_VIDEO_SHA256_MISMATCH",
                "Draft scene SHA-256 does not match the review session.",
                http_status=422,
            )
        frame_count = int(
            (session.scene_video_artifact.metadata_ or {}).get("frame_count") or 0
        )
        fps = float((session.scene_video_artifact.metadata_ or {}).get("fps") or 0)
        shots = self.canonicalize_shots(
            [
                item.model_dump(exclude={"start_seconds", "end_seconds_inclusive"})
                for item in payload.shots
            ],
            frame_count=frame_count,
        )
        errors = self.validate_shots(
            shots,
            frame_count=frame_count,
            require_shot_indexes=True,
        )
        if errors:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_DRAFT_INVALID",
                "Shot boundaries do not cover the complete scene.",
                http_status=422,
                detail={"validation_errors": errors},
            )
        for shot in shots:
            shot["start_seconds"] = shot["start_frame"] / fps
            shot["end_seconds_inclusive"] = shot["end_frame_inclusive"] / fps
            shot["review_status"] = "PENDING"
        session.draft_json = {
            **dict(session.draft_json or {}),
            "shots": shots,
            "note": payload.note,
            "automatic_confirmation": False,
        }
        session.draft_revision += 1
        self.db.commit()
        self.db.refresh(session)
        return self._response(session)

    def reset_draft(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> ShotBoundaryReviewResponse:
        session = self._required_session(
            project.project_id, revision_id, event_id, scene_id, user.user_id
        )
        self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        if session.status == "CONFIRMED":
            raise ShotBoundaryWorkflowError(
                "REVIEWED_SHOT_BOUNDARIES_IMMUTABLE",
                "Confirmed boundaries are immutable.",
            )
        session.draft_json = dict(session.automatic_draft_json)
        session.draft_revision += 1
        self.db.commit()
        return self._response(session)

    def confirm(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        payload: ShotBoundaryConfirmRequest,
    ) -> ShotBoundaryConfirmResponse:
        revision, _event, scene, source = self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        session = self._required_session(
            project.project_id, revision_id, event_id, scene_id, user.user_id
        )
        prior = self.db.scalar(
            select(ShotBoundaryReviewDecision).where(
                ShotBoundaryReviewDecision.session_id == session.session_id,
                ShotBoundaryReviewDecision.idempotency_key == payload.idempotency_key,
            )
        )
        if prior is not None and prior.artifact_id:
            artifact = self.db.get(Artifact, prior.artifact_id)
            return self._confirm_response(session, artifact)
        if session.status == "CONFIRMED":
            raise ShotBoundaryWorkflowError(
                "REVIEWED_SHOT_BOUNDARIES_IMMUTABLE",
                "This review session is already confirmed.",
            )
        if payload.draft_revision != session.draft_revision:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_DRAFT_CONFLICT",
                "Confirm used a stale draft revision.",
                detail={"current_draft_revision": session.draft_revision},
            )
        scene_metadata = session.scene_video_artifact.metadata_ or {}
        scene_sha = str(scene_metadata.get("scene_video_sha256") or "")
        if payload.scene_video_sha256 and payload.scene_video_sha256 != scene_sha:
            raise ShotBoundaryWorkflowError(
                "SCENE_VIDEO_SHA256_MISMATCH",
                "Confirm scene SHA-256 does not match.",
                http_status=422,
            )
        source_path = self.storage.resolve_path(source.file_path)
        scene_path = self.storage.resolve_path(session.scene_video_asset.file_path)
        if (
            source.asset_id != session.source_video_asset_id
            or not source_path.is_file()
            or not scene_path.is_file()
            or _sha256(source_path) != scene_metadata.get("source_video_sha256")
            or _sha256(scene_path) != scene_sha
            or session.scene_video_asset.sha256 != scene_sha
            or scene.source_job_id != revision.action_spotting_job_id
            or scene_metadata.get("source_video_asset_id") != source.asset_id
            or scene_metadata.get("revision_id") != revision_id
            or scene_metadata.get("event_id") != event_id
        ):
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_PROVENANCE_MISMATCH",
                "Review provenance changed before confirmation.",
                http_status=422,
            )
        shots = self.canonicalize_shots(
            list((session.draft_json or {}).get("shots") or []),
            frame_count=int(scene_metadata.get("frame_count") or 0),
        )
        frame_count = int(scene_metadata.get("frame_count") or 0)
        errors = self.validate_shots(
            shots,
            frame_count=frame_count,
            require_shot_indexes=True,
        )
        if errors:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_CONFIRM_VALIDATION_FAILED",
                "Shot boundary confirmation failed validation.",
                http_status=422,
                detail={"validation_errors": errors},
            )
        reviewed_at = datetime.now(timezone.utc)
        canonical_shots = shots
        reviewed_shots: list[dict[str, Any]] = []
        for shot_index, row in enumerate(canonical_shots):
            start_frame = int(row["start_frame"])
            end_frame = int(
                row.get("end_frame_inclusive", row.get("end_frame"))
            )
            reviewed_shots.append(
                {
                    "shot_index": shot_index,
                    "shot_id": str(row["shot_id"]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "end_frame_inclusive": end_frame,
                    "frame_count": end_frame - start_frame + 1,
                    "start_time_sec": start_frame / float(scene_metadata["fps"]),
                    "end_time_sec": (end_frame + 1)
                    / float(scene_metadata["fps"]),
                    "cut_in_frame": None if shot_index == 0 else start_frame,
                    "cut_out_frame": (
                        None
                        if shot_index == len(canonical_shots) - 1
                        else end_frame + 1
                    ),
                    "review_state": "REVIEWED_PASS",
                    "review_status": "REVIEWED_PASS",
                    "status": "REVIEWED_PASS",
                }
            )
        document = {
            "schema_version": "kickclip.reviewed_shot_boundaries.v1",
            "project_id": project.project_id,
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "source_video_asset_id": source.asset_id,
            "source_video_sha256": source.sha256,
            "scene_video_artifact_id": session.scene_video_artifact_id,
            "scene_video_sha256": scene_sha,
            "video": {
                "sha256": scene_sha,
                "fps": scene_metadata["fps"],
                "frame_count": frame_count,
            },
            "fps": scene_metadata["fps"],
            "frame_count": frame_count,
            "review_status": "CONFIRMED",
            "reviewer_user_id": user.user_id,
            "reviewed_at": reviewed_at.isoformat(),
            "reviewer_note": payload.reviewer_note,
            "automatic_confirmation": False,
            "diagnostics": {
                "pending_cut_frames": [],
                "review_required": False,
                "retrieval_authorized": True,
            },
            "review_contract": {
                "pending_cut_frames": [],
                "review_required": False,
                "retrieval_authorized": True,
                "automatic_confirmation": False,
            },
            "shots": reviewed_shots,
        }
        document["content_sha256"] = _canonical_sha256(document)
        root = (
            self.storage.resolve_path(session.scene_video_asset.file_path).parent
            / "reviewed"
            / f"r{session.session_revision:04d}"
        )
        reviewed_path = root / "reviewed_shot_boundaries.json"
        artifact_sha = _atomic_json(reviewed_path, document)
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=REVIEWED_ARTIFACT_TYPE,
            file_path=reviewed_path.relative_to(self.storage.project_root).as_posix(),
            mime_type="application/json",
            metadata_={
                "status": "REVIEWED_PASS",
                "review_state": "REVIEWED_PASS",
                "revision_id": revision_id,
                "event_id": event_id,
                "scene_id": scene_id,
                "source_video_asset_id": source.asset_id,
                "source_video_sha256": scene_sha,
                "action_spotting_source_video_sha256": source.sha256,
                "scene_video_artifact_id": session.scene_video_artifact_id,
                "scene_video_sha256": scene_sha,
                "event_source_start_sec": float(scene.start_sec),
                "event_source_end_sec": float(scene.end_sec),
                "sha256": artifact_sha,
                "review_session_id": session.session_id,
                "reviewer_user_id": user.user_id,
                "automatic_confirmation": False,
                "artifact_revision": session.session_revision,
            },
        )
        decision = ShotBoundaryReviewDecision(
            session_id=session.session_id,
            reviewer_user_id=user.user_id,
            draft_revision=session.draft_revision,
            decision="CONFIRMED",
            note=payload.reviewer_note,
            idempotency_key=payload.idempotency_key,
            automatic_confirmation=False,
            artifact_id=artifact.artifact_id,
        )
        self.db.add(decision)
        session.confirmed_artifact_id = artifact.artifact_id
        session.status = "CONFIRMED"
        session.confirmed_at = reviewed_at
        # Commit the human decision before running optional model inference.
        # A detector crash must never erase a reviewed boundary artifact.
        self.db.commit()
        artifact_id = artifact.artifact_id
        session_id = session.session_id

        try:
            session = self.db.get(ShotBoundaryReviewSession, session_id)
            revision = self.db.get(HighlightRevision, revision_id)
            detections = self._materialize_detections(
                session=session, project=project, revision=revision
            )
            session.detections_artifact_id = detections.artifact_id
            session.error_json = {}
            detection_status = "READY"
        except Exception as exc:  # noqa: BLE001 - reviewed artifact survives detector failures
            self.db.rollback()
            session = self.db.get(ShotBoundaryReviewSession, session_id)
            session.error_json = {
                "code": getattr(exc, "code", "SCENE_PLAYER_DETECTION_FAILED"),
                "message": self._safe_detection_error(exc),
            }
            detections = None
            detection_status = "FAILED_RETRYABLE"

        revision = self.db.get(HighlightRevision, revision_id)
        mapping = {
            "status": "MATERIALIZED" if detections else "WAITING_DETECTIONS",
            "scene_id": scene_id,
            "scene_video_asset_id": session.scene_video_asset_id,
            "scene_video_artifact_id": session.scene_video_artifact_id,
            "reviewed_shots_artifact_id": artifact_id,
            "shot_boundaries_artifact_id": artifact_id,
            "shot_boundaries_sha256": artifact_sha,
            "boundary_origin": "HUMAN_REVIEWED",
            "human_reviewed": True,
            "detections_artifact_id": detections.artifact_id if detections else None,
            "source_video_sha256": scene_sha,
            "action_spotting_source_video_sha256": source.sha256,
            "event_source_start_sec": float(scene.start_sec),
            "event_source_end_sec": float(scene.end_sec),
            "source_start_frame": round(
                float(scene.start_sec) * float(source.fps or scene_metadata["fps"])
            ),
            "source_frame_mapping_basis": "timeline_event_start_sec_x_source_asset_fps",
            "candidate_source_to_video_frame_offset": 0,
            "candidate_video_frame_mapping_basis": "scene_candidate_frame_index_is_scene_video_frame_index",
            "automatic_review_approval": False,
            "review_session_id": session.session_id,
        }
        revision.options = {
            **(revision.options or {}),
            "candidate_pipeline_inputs": mapping,
        }
        self.db.commit()
        return ShotBoundaryConfirmResponse(
            status="CONFIRMED",
            artifact_id=artifact_id,
            artifact_sha256=artifact_sha,
            reviewed_shot_count=len(reviewed_shots),
            scene_video_sha256=scene_sha,
            detections_status=detection_status,
            detections_artifact_id=detections.artifact_id if detections else None,
        )

    def retry_detections(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> ShotBoundaryReviewResponse:
        revision, _event, scene, source = self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        session = self._required_session(
            project.project_id, revision_id, event_id, scene_id, user.user_id
        )
        if session.status != "CONFIRMED" or not session.confirmed_artifact_id:
            raise ShotBoundaryWorkflowError(
                "REVIEWED_SHOT_BOUNDARIES_NOT_READY",
                "Human confirmation is required before detections.",
            )
        if session.detections_artifact_id:
            return self._response(session)
        try:
            detections = self._materialize_detections(
                session=session, project=project, revision=revision
            )
            session.detections_artifact_id = detections.artifact_id
            session.error_json = {}
            scene_metadata = session.scene_video_artifact.metadata_ or {}
            mapping = {
                **((revision.options or {}).get("candidate_pipeline_inputs") or {}),
                "status": "MATERIALIZED",
                "scene_id": scene_id,
                "scene_video_asset_id": session.scene_video_asset_id,
                "scene_video_artifact_id": session.scene_video_artifact_id,
                "reviewed_shots_artifact_id": session.confirmed_artifact_id,
                "shot_boundaries_artifact_id": session.confirmed_artifact_id,
                "shot_boundaries_sha256": str(
                    (session.confirmed_artifact.metadata_ or {}).get("sha256") or ""
                ),
                "boundary_origin": "HUMAN_REVIEWED",
                "human_reviewed": True,
                "detections_artifact_id": detections.artifact_id,
                "source_video_sha256": scene_metadata.get("scene_video_sha256"),
                "action_spotting_source_video_sha256": source.sha256,
                "event_source_start_sec": float(scene.start_sec),
                "event_source_end_sec": float(scene.end_sec),
                "source_start_frame": round(
                    float(scene.start_sec) * float(source.fps or scene_metadata["fps"])
                ),
                "source_frame_mapping_basis": "timeline_event_start_sec_x_source_asset_fps",
                "candidate_source_to_video_frame_offset": 0,
                "candidate_video_frame_mapping_basis": "scene_candidate_frame_index_is_scene_video_frame_index",
                "automatic_review_approval": False,
                "review_session_id": session.session_id,
            }
            revision.options = {
                **(revision.options or {}),
                "candidate_pipeline_inputs": mapping,
            }
            self.db.commit()
            return self._response(session)
        except Exception as exc:
            self.db.rollback()
            session = self.db.get(ShotBoundaryReviewSession, session.session_id)
            session.error_json = {
                "code": getattr(exc, "code", "SCENE_PLAYER_DETECTION_FAILED"),
                "message": self._safe_detection_error(exc),
            }
            self.db.commit()
            raise ShotBoundaryWorkflowError(
                session.error_json["code"],
                session.error_json["message"],
                detail={"reviewed_artifact_preserved": True},
            ) from exc

    def validate_candidate_contract(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> dict[str, str]:
        """Validate the exact immutable inputs consumed by candidate discovery.

        R15 fresh-machine revisions use a ShotBoundaryReviewSession. Older
        R12-R14 revisions may only carry immutable artifact IDs in
        HighlightRevision.options. Both paths are accepted, but every file,
        hash, scope and frozen scene-target-selection shot contract is checked.
        """

        revision, _event, _scene, _source = self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        mapping = (revision.options or {}).get("candidate_pipeline_inputs") or {}
        session = self._latest_session(
            project_id=project.project_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            user_id=user.user_id,
        )

        def invalid(reason: str, message: str, **detail: Any) -> None:
            raise ShotBoundaryWorkflowError(
                "REVIEWED_SHOT_BOUNDARY_CONTRACT_INVALID",
                message,
                http_status=409,
                detail={"reason": reason, **detail},
            )

        session_confirmed = bool(
            session is not None
            and session.status == "CONFIRMED"
            and session.confirmed_artifact_id
        )
        reviewed_artifact_id = str(
            (session.confirmed_artifact_id if session_confirmed else None)
            or mapping.get("reviewed_shots_artifact_id")
            or ""
        )
        detections_artifact_id = str(
            (session.detections_artifact_id if session_confirmed else None)
            or mapping.get("detections_artifact_id")
            or ""
        )
        review_session_id = str(
            (session.session_id if session_confirmed else None)
            or mapping.get("review_session_id")
            or ""
        )
        if not reviewed_artifact_id:
            invalid(
                "REVIEWED_ARTIFACT_ID_MISSING",
                "Confirmed reviewed shot-boundary artifact is missing.",
            )
        if not detections_artifact_id:
            invalid(
                "DETECTIONS_ARTIFACT_ID_MISSING",
                "Scene player detections artifact is missing.",
            )

        reviewed = self.db.get(Artifact, reviewed_artifact_id)
        detections = self.db.get(Artifact, detections_artifact_id)
        if reviewed is None:
            invalid(
                "REVIEWED_ARTIFACT_NOT_FOUND",
                "Confirmed reviewed shot-boundary artifact does not exist.",
                artifact_id=reviewed_artifact_id,
            )
        if detections is None:
            invalid(
                "DETECTIONS_ARTIFACT_NOT_FOUND",
                "Scene player detections artifact does not exist.",
                artifact_id=detections_artifact_id,
            )

        for artifact, expected_type, label in (
            (reviewed, REVIEWED_ARTIFACT_TYPE, "reviewed shot boundaries"),
            (detections, DETECTIONS_ARTIFACT_TYPE, "scene player detections"),
        ):
            if artifact.artifact_type != expected_type:
                invalid(
                    "ARTIFACT_TYPE_MISMATCH",
                    f"The {label} artifact type is invalid.",
                    artifact_id=artifact.artifact_id,
                    expected_type=expected_type,
                    actual_type=artifact.artifact_type,
                )
            if artifact.match_id != project.match_id or artifact.project_id not in {
                None,
                project.project_id,
            }:
                invalid(
                    "ARTIFACT_SCOPE_MISMATCH",
                    f"The {label} artifact belongs to a different project or match.",
                    artifact_id=artifact.artifact_id,
                )

        reviewed_path = self.storage.resolve_path(reviewed.file_path)
        detections_path = self.storage.resolve_path(detections.file_path)
        if not reviewed_path.is_file():
            invalid(
                "REVIEWED_ARTIFACT_FILE_MISSING",
                "Reviewed shot-boundary artifact file is missing.",
                artifact_id=reviewed.artifact_id,
            )
        if not detections_path.is_file():
            invalid(
                "DETECTIONS_ARTIFACT_FILE_MISSING",
                "Scene player detections artifact file is missing.",
                artifact_id=detections.artifact_id,
            )

        reviewed_sha256 = _sha256(reviewed_path)
        detections_sha256 = _sha256(detections_path)
        declared_reviewed_sha256 = str((reviewed.metadata_ or {}).get("sha256") or "")
        declared_detections_sha256 = str((detections.metadata_ or {}).get("sha256") or "")
        if len(declared_reviewed_sha256) != 64 or reviewed_sha256 != declared_reviewed_sha256:
            invalid(
                "REVIEWED_ARTIFACT_SHA256_MISMATCH",
                "Reviewed shot-boundary artifact SHA-256 is invalid.",
                artifact_id=reviewed.artifact_id,
            )
        if len(declared_detections_sha256) != 64 or detections_sha256 != declared_detections_sha256:
            invalid(
                "DETECTIONS_ARTIFACT_SHA256_MISMATCH",
                "Scene player detections artifact SHA-256 is invalid.",
                artifact_id=detections.artifact_id,
            )

        try:
            document = json.loads(reviewed_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            invalid(
                "REVIEWED_ARTIFACT_JSON_INVALID",
                "Reviewed shot-boundary artifact is not valid JSON.",
                error_type=type(exc).__name__,
            )
            raise AssertionError("unreachable") from exc
        if not isinstance(document, dict):
            invalid(
                "REVIEWED_ARTIFACT_JSON_INVALID",
                "Reviewed shot-boundary artifact root must be an object.",
            )

        metadata = reviewed.metadata_ or {}
        exact_values = {
            "project_id": project.project_id,
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
        }
        for key, expected in exact_values.items():
            declared = document.get(key)
            metadata_value = metadata.get(key)
            if declared not in {None, "", expected}:
                invalid(
                    "REVIEWED_ARTIFACT_PROVENANCE_MISMATCH",
                    f"Reviewed shot-boundary {key} does not match the request.",
                    expected=expected,
                    actual=declared,
                )
            if metadata_value not in {None, "", expected}:
                invalid(
                    "REVIEWED_ARTIFACT_PROVENANCE_MISMATCH",
                    f"Reviewed shot-boundary metadata {key} does not match the request.",
                    expected=expected,
                    actual=metadata_value,
                )

        if document.get("automatic_confirmation") is not False:
            invalid(
                "AUTOMATIC_CONFIRMATION_FORBIDDEN",
                "Reviewed shot boundaries were not explicitly human-confirmed.",
            )
        if metadata.get("automatic_confirmation") is True:
            invalid(
                "AUTOMATIC_CONFIRMATION_FORBIDDEN",
                "Reviewed shot-boundary metadata declares automatic confirmation.",
            )

        scene_artifact_id = str(
            document.get("scene_video_artifact_id")
            or metadata.get("scene_video_artifact_id")
            or mapping.get("scene_video_artifact_id")
            or ""
        )
        scene_artifact = self.db.get(Artifact, scene_artifact_id) if scene_artifact_id else None
        if scene_artifact is None:
            invalid(
                "SCENE_VIDEO_ARTIFACT_NOT_FOUND",
                "Reviewed boundaries do not reference a valid scene video artifact.",
            )
        scene_path = self.storage.resolve_path(scene_artifact.file_path)
        if not scene_path.is_file():
            invalid(
                "SCENE_VIDEO_FILE_MISSING",
                "The immutable scene video file is missing.",
                artifact_id=scene_artifact.artifact_id,
            )
        scene_metadata = scene_artifact.metadata_ or {}
        scene_sha256 = str(
            scene_metadata.get("scene_video_sha256")
            or scene_metadata.get("sha256")
            or ""
        )
        actual_scene_sha256 = _sha256(scene_path)
        if len(scene_sha256) != 64 or actual_scene_sha256 != scene_sha256:
            invalid(
                "SCENE_VIDEO_SHA256_MISMATCH",
                "The immutable scene video SHA-256 is invalid.",
                artifact_id=scene_artifact.artifact_id,
            )

        video = document.get("video") or {}
        if not isinstance(video, dict):
            invalid(
                "VIDEO_CONTRACT_INVALID",
                "Reviewed shot-boundary video contract must be an object.",
            )
        if str(video.get("sha256") or "") != scene_sha256:
            invalid(
                "VIDEO_SHA256_MISMATCH",
                "Reviewed shot-boundary video SHA-256 does not match the scene video.",
            )
        frame_count = int(scene_metadata.get("frame_count") or 0)
        try:
            declared_frame_count = int(video.get("frame_count") or -1)
        except (TypeError, ValueError):
            declared_frame_count = -1
        if frame_count <= 0 or declared_frame_count != frame_count:
            invalid(
                "VIDEO_FRAME_COUNT_MISMATCH",
                "Reviewed shot-boundary frame count does not match the scene video.",
                expected=frame_count,
                actual=declared_frame_count,
            )

        diagnostics = document.get("diagnostics") or {}
        review_contract = document.get("review_contract") or {}
        pending = list(
            (diagnostics.get("pending_cut_frames") if isinstance(diagnostics, dict) else None)
            or (review_contract.get("pending_cut_frames") if isinstance(review_contract, dict) else None)
            or []
        )
        if pending:
            invalid(
                "PENDING_CUT_FRAMES",
                "All shot-cut candidates must be resolved before candidate discovery.",
                pending_cut_frames=pending,
            )
        if isinstance(diagnostics, dict) and diagnostics.get("review_required") is True:
            invalid(
                "SHOT_BOUNDARY_REVIEW_REQUIRED",
                "Shot-boundary review is still required.",
            )
        if isinstance(diagnostics, dict) and diagnostics.get("retrieval_authorized") is False:
            invalid(
                "SHOT_BOUNDARY_RETRIEVAL_NOT_AUTHORIZED",
                "Reviewed shot boundaries are not authorized for retrieval.",
            )

        shots = document.get("shots")
        if not isinstance(shots, list) or not shots:
            invalid(
                "SHOTS_MISSING",
                "Reviewed shot-boundary artifact has no shots.",
            )
        expected_start = 0
        for index, shot in enumerate(shots):
            if not isinstance(shot, dict):
                invalid(
                    "SHOT_ROW_INVALID",
                    "Every reviewed shot row must be an object.",
                    shot_index=index,
                )
            try:
                shot_index = int(shot.get("shot_index", -1))
                start_frame = int(shot.get("start_frame", -1))
                end_frame = int(shot.get("end_frame_inclusive", -1))
            except (TypeError, ValueError):
                invalid(
                    "SHOT_FRAME_VALUES_INVALID",
                    "Reviewed shot frame values must be integers.",
                    shot_index=index,
                )
                raise AssertionError("unreachable")
            if shot_index != index:
                invalid(
                    "SHOT_INDEXES_NOT_CONTIGUOUS",
                    "Shot indexes are not contiguous.",
                    expected=index,
                    actual=shot_index,
                )
            if start_frame != expected_start or end_frame < start_frame:
                invalid(
                    "SHOT_FRAME_COVERAGE_NOT_CONTIGUOUS",
                    "Shot frame coverage is not contiguous.",
                    shot_index=index,
                    expected_start=expected_start,
                    actual_start=start_frame,
                    end_frame_inclusive=end_frame,
                )
            review_state = str(shot.get("review_state") or "")
            if review_state not in {"REVIEWED_PASS", "PASS"}:
                invalid(
                    "SHOT_NOT_REVIEWED",
                    "Every shot boundary must be reviewed.",
                    shot_index=index,
                    review_state=review_state,
                )
            expected_start = end_frame + 1
        if expected_start != frame_count:
            invalid(
                "SHOT_FRAME_COVERAGE_INCOMPLETE",
                "Shot boundaries do not cover every scene frame exactly once.",
                expected_frame_count=frame_count,
                covered_until_exclusive=expected_start,
            )

        detection_metadata = detections.metadata_ or {}
        detection_scene_sha = str(
            detection_metadata.get("scene_video_sha256")
            or detection_metadata.get("source_video_sha256")
            or ""
        )
        if detection_scene_sha != scene_sha256:
            invalid(
                "DETECTIONS_SCENE_SHA256_MISMATCH",
                "Scene player detections were generated from a different scene video.",
                expected=scene_sha256,
                actual=detection_scene_sha,
            )
        for key, expected in (("revision_id", revision_id), ("event_id", event_id), ("scene_id", scene_id)):
            actual = detection_metadata.get(key)
            if actual not in {None, "", expected}:
                invalid(
                    "DETECTIONS_PROVENANCE_MISMATCH",
                    f"Scene player detections {key} does not match the request.",
                    expected=expected,
                    actual=actual,
                )

        return {
            "reviewed_shots_artifact_id": reviewed.artifact_id,
            "reviewed_shot_boundaries_sha256": reviewed_sha256,
            "detections_artifact_id": detections.artifact_id,
            "detections_sha256": detections_sha256,
            "scene_video_artifact_id": scene_artifact.artifact_id,
            "scene_video_sha256": scene_sha256,
            "review_session_id": review_session_id,
        }

    @staticmethod
    def canonicalize_shots(
        shots: list[dict[str, Any]], *, frame_count: int
    ) -> list[dict[str, Any]]:
        canonical = sorted(
            (dict(row) for row in shots),
            key=lambda row: (
                int(row["start_frame"]),
                int(row.get("end_frame_inclusive", row.get("end_frame"))),
                str(row["shot_id"]),
            ),
        )
        for shot_index, row in enumerate(canonical):
            start = int(row["start_frame"])
            end = int(row.get("end_frame_inclusive", row.get("end_frame")))
            row.update(
                {
                    "shot_index": shot_index,
                    "start_frame": start,
                    "end_frame": end,
                    "end_frame_inclusive": end,
                    "frame_count": end - start + 1,
                    "cut_in_frame": None if start == 0 else start,
                    "cut_out_frame": None if end == frame_count - 1 else end + 1,
                }
            )
        return canonical

    @staticmethod
    def validate_shots(
        shots: list[dict[str, Any]],
        *,
        frame_count: int,
        require_shot_indexes: bool = False,
    ) -> list[str]:
        errors: list[str] = []
        if frame_count <= 0:
            return ["frame_count must be positive"]
        if not shots:
            return ["at least one shot is required"]
        ids = [str(row.get("shot_id") or "") for row in shots]
        if any(not value for value in ids):
            errors.append("shot_id is required")
        if len(ids) != len(set(ids)):
            errors.append("shot IDs must be unique")
        normalized: list[tuple[int, int]] = []
        shot_indexes: list[int] = []
        for index, row in enumerate(shots):
            try:
                start = int(row["start_frame"])
                end = int(row.get("end_frame_inclusive", row.get("end_frame")))
            except (KeyError, TypeError, ValueError):
                errors.append(f"shot {index} has invalid frame values")
                continue
            if start > end:
                errors.append(
                    f"shot {ids[index] or index} start_frame exceeds end_frame"
                )
            if start < 0 or end >= frame_count:
                errors.append(f"shot {ids[index] or index} is outside scene frames")
            normalized.append((start, end))
            if require_shot_indexes:
                value = row.get("shot_index")
                if isinstance(value, bool) or not isinstance(value, int):
                    errors.append(f"shot {ids[index] or index} shot_index must be int")
                else:
                    shot_indexes.append(value)
        if len(normalized) != len(shots):
            return errors
        if normalized[0][0] != 0:
            errors.append("first shot must start at frame 0")
        if normalized[-1][1] != frame_count - 1:
            errors.append(f"last shot must end at frame {frame_count - 1}")
        for index, ((_, previous_end), (start, _)) in enumerate(
            pairwise(normalized), start=1
        ):
            if start > previous_end + 1:
                errors.append(f"gap before shot {ids[index]}")
            if start <= previous_end:
                errors.append(f"overlap before shot {ids[index]}")
        if require_shot_indexes:
            expected_indexes = list(range(len(shots)))
            if len(shot_indexes) != len(set(shot_indexes)):
                errors.append("shot indexes must be unique")
            if shot_indexes != expected_indexes:
                errors.append("shot indexes must be contiguous and match frame order")
        return errors

    def _materialize_scene(
        self,
        *,
        project: Project,
        revision: HighlightRevision,
        event: TimelineEvent,
        scene: TimelineEvent,
        source: MediaAsset,
        source_path: Path,
    ) -> tuple[MediaAsset, Artifact]:
        root = (
            self.storage.storage_root
            / "shot_boundary_reviews"
            / project.project_id
            / revision.revision_id
            / scene.timeline_event_id
            / str(source.sha256)[:16]
        ).resolve()
        if not root.is_relative_to(self.storage.storage_root):
            raise ShotBoundaryWorkflowError(
                "STORAGE_PATH_INVALID", "Scene output path escapes storage root."
            )
        root.mkdir(parents=True, exist_ok=True)
        output = root / "scene.mp4"
        duration = float(scene.end_sec) - float(scene.start_sec)
        if duration <= 0:
            raise ShotBoundaryWorkflowError(
                "SCENE_RANGE_INVALID",
                "Scene end must be after scene start.",
                http_status=422,
            )
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{float(scene.start_sec):.6f}",
            "-i",
            source_path.as_posix(),
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            output.as_posix(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output.is_file():
            raise ShotBoundaryWorkflowError(
                "SCENE_VIDEO_MATERIALIZATION_FAILED",
                "FFmpeg could not materialize the exact event scene clip.",
                detail={"ffmpeg_stderr_tail": (completed.stderr or "")[-2000:]},
            )
        metadata = extract_video_metadata(output)
        fps = float(metadata.get("fps") or 0)
        frame_count = int(
            metadata.get("frame_count")
            or round(float(metadata.get("duration_sec") or duration) * fps)
        )
        if (
            fps <= 0
            or frame_count <= 0
            or not metadata.get("width")
            or not metadata.get("height")
        ):
            raise ShotBoundaryWorkflowError(
                "SCENE_VIDEO_METADATA_FAILED",
                "Scene FPS, frame count or dimensions are unavailable.",
            )
        scene_sha = _sha256(output)
        asset = self.media.create(
            match_id=project.match_id,
            asset_type="HIGHLIGHT_SCENE_CLIP",
            file_path=output.relative_to(self.storage.project_root).as_posix(),
            original_filename=output.name,
            mime_type="video/mp4",
            duration_sec=float(metadata.get("duration_sec") or duration),
            fps=fps,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            size_bytes=output.stat().st_size,
            sha256=scene_sha,
        )
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=SCENE_VIDEO_ARTIFACT_TYPE,
            file_path=output.relative_to(self.storage.project_root).as_posix(),
            mime_type="video/mp4",
            metadata_={
                "status": "READY",
                "project_id": project.project_id,
                "revision_id": revision.revision_id,
                "event_id": event.timeline_event_id,
                "scene_id": scene.timeline_event_id,
                "source_video_asset_id": source.asset_id,
                "source_video_sha256": source.sha256,
                "scene_video_sha256": scene_sha,
                "source_start_seconds": float(scene.start_sec),
                "source_end_seconds": float(scene.end_sec),
                "fps": fps,
                "frame_count": frame_count,
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
                "storage_relative_path": output.relative_to(
                    self.storage.storage_root
                ).as_posix(),
                "sha256": scene_sha,
            },
        )
        self.db.flush()
        return asset, artifact

    def _analyze_cuts(
        self,
        *,
        scene_path: Path,
        output_root: Path,
        scene_id: str,
        scene_sha: str,
        fps: float,
        frame_count: int,
    ) -> dict[str, Any]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_ANALYZER_UNAVAILABLE",
                "OpenCV is required for shot-cut analysis.",
            ) from exc
        capture = cv2.VideoCapture(str(scene_path))
        scores: list[tuple[int, float]] = []
        frames: dict[int, Any] = {}
        previous = None
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            small = cv2.resize(frame, (160, 90))
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            if previous is not None:
                scores.append(
                    (index, float(np.mean(cv2.absdiff(lab, previous))) / 255.0)
                )
            previous = lab
            index += 1
        capture.release()
        if index <= 0:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_ANALYSIS_FAILED", "Scene video has no decodable frames."
            )
        frame_count = index
        values = np.array([score for _, score in scores], dtype=float)
        if len(values):
            median = float(np.median(values))
            median_absolute_deviation = float(np.median(np.abs(values - median)))
            threshold = max(0.10, median + 8.0 * median_absolute_deviation)
        else:
            threshold = 1.0
        candidates: list[tuple[int, float]] = []
        min_distance = max(4, round(fps * 0.25))
        for frame_index, score in sorted(scores, key=lambda row: row[1], reverse=True):
            if score < threshold or any(
                abs(frame_index - accepted) < min_distance for accepted, _ in candidates
            ):
                continue
            candidates.append((frame_index, min(1.0, score / max(threshold, 1e-9))))
            if len(candidates) >= 100:
                break
        candidates.sort()
        thumbs = output_root / "cut-thumbnails"
        thumbs.mkdir(parents=True, exist_ok=True)
        contact_frames = sorted(
            set(
                [0, frame_count - 1]
                + [value for cut, _ in candidates for value in (max(0, cut - 1), cut)]
            )
        )
        capture = cv2.VideoCapture(str(scene_path))
        wanted = set(contact_frames)
        frame_index = 0
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                frames[frame_index] = frame
                wanted.remove(frame_index)
            frame_index += 1
        capture.release()
        cut_rows = []
        for cut, score in candidates:
            before = thumbs / f"cut_{cut:08d}_before.jpg"
            after = thumbs / f"cut_{cut:08d}_after.jpg"
            if cut - 1 in frames:
                cv2.imwrite(str(before), frames[cut - 1])
            if cut in frames:
                cv2.imwrite(str(after), frames[cut])
            cut_rows.append(
                {
                    "cut_frame": cut,
                    "score": round(float(score), 6),
                    "hard_cut_candidate": True,
                    "before_thumbnail_name": before.name if before.is_file() else None,
                    "after_thumbnail_name": after.name if after.is_file() else None,
                }
            )
        contact_path = output_root / "contact_sheet.jpg"
        tiles = [
            cv2.resize(frames[key], (320, 180))
            for key in contact_frames[:20]
            if key in frames
        ]
        if tiles:
            columns = 4
            blank = np.zeros_like(tiles[0])
            rows = []
            for offset in range(0, len(tiles), columns):
                row = tiles[offset : offset + columns]
                row.extend([blank] * (columns - len(row)))
                rows.append(np.hstack(row))
            cv2.imwrite(str(contact_path), np.vstack(rows))
        boundaries = [0] + [cut for cut, _ in candidates] + [frame_count]
        shots = [
            {
                "shot_id": f"shot_{idx:04d}",
                "start_frame": start,
                "end_frame_inclusive": end - 1,
                "start_seconds": start / fps,
                "end_seconds_inclusive": (end - 1) / fps,
                "review_status": "PENDING",
            }
            for idx, (start, end) in enumerate(pairwise(boundaries))
        ]
        return {
            "schema_version": "kickclip.shot_boundary_draft.v1",
            "scene_id": scene_id,
            "scene_video_sha256": scene_sha,
            "fps": fps,
            "frame_count": frame_count,
            "cuts": cut_rows,
            "shots": shots,
            "draft_shots": shots,
            "automatic_confirmation": False,
            "analysis": {
                "method": "opencv_lab_frame_difference_v1",
                "threshold": threshold,
            },
        }

    def _materialize_detections(
        self,
        *,
        session: ShotBoundaryReviewSession,
        project: Project,
        revision: HighlightRevision,
    ) -> Artifact:
        scene_path = self.storage.resolve_path(session.scene_video_asset.file_path)
        scene_sha = session.scene_video_asset.sha256 or _sha256(scene_path)
        event = self.db.get(TimelineEvent, session.event_id)
        scene = self.db.get(TimelineEvent, session.scene_id)
        if event is None or scene is None:
            raise ShotBoundaryWorkflowError(
                "ACTION_SPOTTING_EVENT_NOT_READY",
                "The exact Action Spotting event and scene are unavailable.",
            )

        scene_metadata = session.scene_video_artifact.metadata_ or {}
        scene_fps = float(
            scene_metadata.get("fps")
            or session.scene_video_asset.fps
            or 0
        )
        declared_frame_count = int(scene_metadata.get("frame_count") or 0)
        if scene_fps <= 0 or declared_frame_count <= 0:
            raise ShotBoundaryWorkflowError(
                "SCENE_PLAYER_DETECTION_FAILED",
                "Scene FPS or frame count is unavailable for candidate detection.",
            )

        event_local_sec = float(event.timestamp_sec) - float(scene.start_sec)
        boundary_document = (
            session.draft_json
            if session.status == "CONFIRMED" and session.draft_json
            else session.automatic_draft_json
        ) or {}
        candidate_shots = self.canonicalize_shots(
            list(boundary_document.get("shots") or []),
            frame_count=declared_frame_count,
        )
        pre_shot_count, post_shot_count = _event_near_shot_counts(event)
        sampled_frames, sampling = _sampled_candidate_frames(
            fps=scene_fps,
            frame_count=declared_frame_count,
            event_local_sec=event_local_sec,
            shots=candidate_shots,
            pre_shot_count=pre_shot_count,
            post_shot_count=post_shot_count,
        )
        if not sampled_frames:
            raise ShotBoundaryWorkflowError(
                "SCENE_PLAYER_DETECTION_FAILED",
                "Event-near shot sampling produced no candidate frames.",
            )

        # Candidate discovery intentionally runs RF-DETR in observation mode:
        # inference starts from the 0.15 base threshold and retains all five
        # checkpoint classes. Wide/close-up policy is applied *after* inference
        # so referee/staff evidence is never discarded before role filtering.
        detector = self.detector_factory(self.settings)
        runtime = dict(detector.runtime_metadata)
        portable_runtime = {
            key: value
            for key, value in runtime.items()
            if key not in {"checkpoint_path", "project_root", "output_root"}
        }
        model_sha = str(
            runtime.get("checkpoint_sha256")
            or _canonical_sha256(runtime)
        )

        play_class_ids = frozenset(self.settings.tracking_play_class_ids)
        observation_class_ids = frozenset(self.settings.observation_class_ids)
        play_conf_threshold = float(self.settings.TRACKING_PLAY_CONF_THRESHOLD)
        observation_conf_threshold = float(
            self.settings.OBSERVATION_CONF_THRESHOLD
        )
        detector_policy = self._current_candidate_detection_policy()
        sampling_contract = {
            "policy_version": FAST_CANDIDATE_DETECTION_POLICY_VERSION,
            "event_id": event.timeline_event_id,
            "event_label": event.label,
            "event_type": event.event_type,
            "event_timestamp_sec": float(event.timestamp_sec),
            "scene_start_sec": float(scene.start_sec),
            "scene_end_sec": float(scene.end_sec),
            "event_window_source": "ACTION_SPOTTING_SCENE_CLIP",
            "sampled_frame_count": len(sampled_frames),
            **sampling,
        }
        cache_key = _canonical_sha256(
            {
                "scene_video_sha256": scene_sha,
                "model_sha256": model_sha,
                "candidate_detection_sampling": sampling_contract,
                "detector_policy": detector_policy,
            }
        )
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == DETECTIONS_ARTIFACT_TYPE,
            )
        ).all()
        for artifact in rows:
            metadata = artifact.metadata_ or {}
            if (
                metadata.get("cache_key") == cache_key
                and metadata.get("candidate_detection_policy") == detector_policy
            ):
                path = self.storage.resolve_path(artifact.file_path)
                if path.is_file() and _sha256(path) == metadata.get("sha256"):
                    return artifact

        try:
            import cv2
        except ImportError as exc:
            raise ShotBoundaryWorkflowError(
                "SCENE_PLAYER_DETECTION_FAILED",
                "OpenCV is required for detection materialization.",
            ) from exc

        cache = PlayerDetectionCache(
            self.storage.storage_root / "player_detection_cache_r1"
        )
        root = scene_path.parent / "detections" / cache_key
        root.mkdir(parents=True, exist_ok=True)
        closeup_crop_root = root / "closeup_crops"
        capture = cv2.VideoCapture(str(scene_path))
        if not capture.isOpened():
            raise ShotBoundaryWorkflowError(
                "SCENE_PLAYER_DETECTION_FAILED",
                "The scene video cannot be opened for candidate detection.",
            )

        candidate_rows: list[dict[str, Any]] = []
        observation_rows: list[dict[str, Any]] = []
        batch_size = max(1, int(self.settings.PLAYER_DETECTOR_BATCH_SIZE))
        batch_indices: list[int] = []
        batch_frames: list[Any] = []
        # We seek directly to the sparse keyframes. This avoids decoding every
        # frame in the Action Spotting clip merely to run inference on a few of
        # them. Keep the legacy metric name for API compatibility; under V4 it
        # now means successfully decoded representative frames.
        decoded_window_frame_count = 0
        wide_frame_count = 0
        closeup_frame_count = 0
        raw_base_detection_count = 0
        observation_eligible_count = 0
        closeup_crop_eligible_count = 0
        saved_closeup_crop_count = 0
        excluded_role_counts: dict[str, int] = {}

        def flush_batch() -> None:
            nonlocal wide_frame_count
            nonlocal closeup_frame_count
            nonlocal raw_base_detection_count
            nonlocal observation_eligible_count
            nonlocal closeup_crop_eligible_count
            nonlocal saved_closeup_crop_count
            if not batch_frames:
                return
            detected = cache.detect_batch(
                source_video_sha256=scene_sha,
                frame_indices=list(batch_indices),
                frames_bgr=list(batch_frames),
                detector=detector,
            )
            for index, frame, detections in zip(
                batch_indices,
                batch_frames,
                detected.detections_by_frame,
            ):
                frame_height, frame_width = frame.shape[:2]
                normalized: list[PlayerDetection] = []
                for item in detections:
                    clean = _normalize_detection_to_frame(
                        item,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    if clean is not None:
                        normalized.append(clean)
                raw_base_detection_count += len(normalized)

                is_closeup = _is_closeup_frame(
                    normalized,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    observation_class_ids=observation_class_ids,
                    observation_conf_threshold=observation_conf_threshold,
                    closeup_min_height_ratio=float(
                        self.settings.CLOSEUP_MIN_BOX_HEIGHT_RATIO
                    ),
                    closeup_min_area_ratio=float(
                        self.settings.CLOSEUP_MIN_BOX_AREA_RATIO
                    ),
                )
                if is_closeup:
                    closeup_frame_count += 1
                else:
                    wide_frame_count += 1

                candidates, frame_observations = (
                    _select_candidate_detections_for_frame(
                        normalized,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        is_closeup=is_closeup,
                        play_class_ids=play_class_ids,
                        play_conf_threshold=play_conf_threshold,
                        observation_class_ids=observation_class_ids,
                        observation_conf_threshold=observation_conf_threshold,
                        crop_top_k=int(self.settings.OBSERVATION_CROP_TOP_K),
                        crop_min_conf=float(self.settings.CLOSEUP_CROP_MIN_CONF),
                        crop_min_height_ratio=float(
                            self.settings.CLOSEUP_CROP_MIN_HEIGHT_RATIO
                        ),
                        crop_min_area_ratio=float(
                            self.settings.CLOSEUP_CROP_MIN_AREA_RATIO
                        ),
                    )
                )

                for raw_index, row in enumerate(frame_observations):
                    item = row["detection"]
                    if row["observation_eligible"]:
                        observation_eligible_count += 1
                    if is_closeup and row["crop_eligible"]:
                        closeup_crop_eligible_count += 1
                    if (
                        row["observation_eligible"]
                        and int(item.class_id) not in play_class_ids
                    ):
                        role = str(item.class_name or "unknown").strip().lower()
                        excluded_role_counts[role] = (
                            excluded_role_counts.get(role, 0) + 1
                        )
                    x1, y1, x2, y2 = item.bbox_xyxy
                    crop_relative_path = None
                    if is_closeup and row["observation_rank"] is not None:
                        ix1 = max(0, min(frame_width - 1, int(math.floor(x1))))
                        iy1 = max(0, min(frame_height - 1, int(math.floor(y1))))
                        ix2 = max(0, min(frame_width - 1, int(math.ceil(x2))))
                        iy2 = max(0, min(frame_height - 1, int(math.ceil(y2))))
                        if ix2 > ix1 and iy2 > iy1:
                            crop = frame[iy1 : iy2 + 1, ix1 : ix2 + 1]
                            if crop.size > 0:
                                frame_crop_root = (
                                    closeup_crop_root / f"frame_{index:08d}"
                                )
                                frame_crop_root.mkdir(
                                    parents=True, exist_ok=True
                                )
                                crop_path = frame_crop_root / (
                                    f"rank_{int(row['observation_rank']):02d}_"
                                    f"class_{int(item.class_id)}_"
                                    f"obs_{raw_index:04d}.jpg"
                                )
                                if cv2.imwrite(str(crop_path), crop):
                                    saved_closeup_crop_count += 1
                                    crop_relative_path = crop_path.relative_to(
                                        self.storage.project_root
                                    ).as_posix()
                    observation_rows.append(
                        {
                            "frame": index,
                            "frame_index": index,
                            "time_ms": int(
                                round(index * 1000.0 / scene_fps)
                            ),
                            "detection_index": raw_index,
                            "detection_id": (
                                f"obs_{index:08d}_{raw_index:04d}"
                            ),
                            "class_id": int(item.class_id),
                            "class_name": str(item.class_name),
                            "confidence": float(item.confidence),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "box_height_ratio": round(
                                float(row["height_ratio"]), 8
                            ),
                            "box_area_ratio": round(
                                float(row["area_ratio"]), 8
                            ),
                            "is_closeup_frame": bool(is_closeup),
                            "observation_eligible": bool(
                                row["observation_eligible"]
                            ),
                            "crop_eligible": bool(row["crop_eligible"]),
                            "observation_rank": row["observation_rank"],
                            "candidate_eligible": bool(
                                row["candidate_eligible"]
                            ),
                            "crop_path": crop_relative_path,
                            "source_model": runtime.get("model_class")
                            or runtime.get("backend"),
                            "model_sha256": model_sha,
                            "scene_video_sha256": scene_sha,
                        }
                    )

                # Frozen Scene Target Selection consumes only the candidate
                # CSV. It must never receive referee/staff/ball rows.
                for detection_index, item in enumerate(candidates):
                    x1, y1, x2, y2 = item.bbox_xyxy
                    candidate_rows.append(
                        {
                            "frame": index,
                            "frame_index": index,
                            "time_ms": int(
                                round(index * 1000.0 / scene_fps)
                            ),
                            "detection_index": detection_index,
                            "detection_id": (
                                f"det_{index:08d}_{detection_index:04d}"
                            ),
                            "class_id": int(item.class_id),
                            "class_name": str(item.class_name),
                            "confidence": float(item.confidence),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "source_model": runtime.get("model_class")
                            or runtime.get("backend"),
                            "model_sha256": model_sha,
                            "scene_video_sha256": scene_sha,
                        }
                    )
            batch_indices.clear()
            batch_frames.clear()

        try:
            for frame_index in sampled_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                decoded_window_frame_count += 1
                batch_indices.append(int(frame_index))
                batch_frames.append(frame)
                if len(batch_frames) >= batch_size:
                    flush_batch()
            flush_batch()
        finally:
            capture.release()

        output = root / "detections.csv"
        observations_output = root / "observations.csv"
        candidate_fields = [
            "frame",
            "frame_index",
            "time_ms",
            "detection_index",
            "detection_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "source_model",
            "model_sha256",
            "scene_video_sha256",
        ]
        observation_fields = [
            *candidate_fields[:12],
            "box_height_ratio",
            "box_area_ratio",
            "is_closeup_frame",
            "observation_eligible",
            "crop_eligible",
            "observation_rank",
            "candidate_eligible",
            "crop_path",
            *candidate_fields[12:],
        ]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=candidate_fields)
            writer.writeheader()
            writer.writerows(candidate_rows)
        with observations_output.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=observation_fields)
            writer.writeheader()
            writer.writerows(observation_rows)

        digest = _sha256(output)
        observations_digest = _sha256(observations_output)
        observation_artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=OBSERVATIONS_ARTIFACT_TYPE,
            file_path=observations_output.relative_to(
                self.storage.project_root
            ).as_posix(),
            mime_type="text/csv",
            metadata_={
                "status": "READY",
                "revision_id": revision.revision_id,
                "event_id": session.event_id,
                "scene_id": session.scene_id,
                "source_video_sha256": scene_sha,
                "scene_video_sha256": scene_sha,
                "model_sha256": model_sha,
                "cache_key": cache_key,
                "observations_sha256": observations_digest,
                "sha256": observations_digest,
                "observation_count": len(observation_rows),
                "observation_eligible_count": observation_eligible_count,
                "closeup_crop_eligible_count": closeup_crop_eligible_count,
                "saved_closeup_crop_count": saved_closeup_crop_count,
                "wide_frame_count": wide_frame_count,
                "closeup_frame_count": closeup_frame_count,
                "excluded_role_counts": dict(
                    sorted(excluded_role_counts.items())
                ),
                "candidate_detection_policy": detector_policy,
                "sampling": sampling_contract,
                "runtime": portable_runtime,
                "automatic_target_confirmation": False,
            },
        )
        return self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=DETECTIONS_ARTIFACT_TYPE,
            file_path=output.relative_to(self.storage.project_root).as_posix(),
            mime_type="text/csv",
            metadata_={
                "status": "READY",
                "revision_id": revision.revision_id,
                "event_id": session.event_id,
                "scene_id": session.scene_id,
                "source_video_sha256": scene_sha,
                "scene_video_sha256": scene_sha,
                "model_sha256": model_sha,
                "cache_key": cache_key,
                "detections_sha256": digest,
                "sha256": digest,
                "frame_count": declared_frame_count,
                "sampled_frame_count": len(sampled_frames),
                "decoded_window_frame_count": decoded_window_frame_count,
                "decoded_representative_frame_count": decoded_window_frame_count,
                "raw_base_detection_count": raw_base_detection_count,
                "observation_eligible_count": observation_eligible_count,
                "candidate_detection_count": len(candidate_rows),
                "detection_count": len(candidate_rows),
                "wide_frame_count": wide_frame_count,
                "closeup_frame_count": closeup_frame_count,
                "closeup_crop_eligible_count": closeup_crop_eligible_count,
                "saved_closeup_crop_count": saved_closeup_crop_count,
                "excluded_role_counts": dict(
                    sorted(excluded_role_counts.items())
                ),
                "observations_artifact_id": observation_artifact.artifact_id,
                "observations_sha256": observations_digest,
                "candidate_detection_policy": detector_policy,
                "sampling": sampling_contract,
                "runtime": portable_runtime,
                "automatic_target_confirmation": False,
            },
        )

    @staticmethod
    def _safe_detection_error(exc: Exception) -> str:
        if isinstance(exc, PlayerDetectorError):
            return "Scene player detection failed; check server-side RF-DETR model configuration."
        return str(exc)

    def _required_session(
        self,
        project_id: str,
        revision_id: str,
        event_id: str,
        scene_id: str,
        user_id: str,
    ) -> ShotBoundaryReviewSession:
        session = self._latest_session(
            project_id=project_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            user_id=user_id,
        )
        if session is None:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_REVIEW_NOT_PREPARED",
                "Prepare the review first.",
                http_status=404,
            )
        return session

    def _automatic_ready_mapping(
        self,
        session: ShotBoundaryReviewSession,
    ) -> tuple[dict[str, Any], Artifact, Artifact] | None:
        """Return the verified automatic boundary/detection contract if ready."""

        revision = self.db.get(HighlightRevision, session.revision_id)
        project = self.db.get(Project, session.project_id)
        if revision is None or project is None:
            return None
        mapping = (revision.options or {}).get("candidate_pipeline_inputs") or {}
        if (
            mapping.get("scene_id") != session.scene_id
            or mapping.get("status") != "MATERIALIZED"
            or mapping.get("boundary_origin") != "AUTO_DETECTED"
            or mapping.get("human_reviewed") is not False
            or mapping.get("automatic_target_confirmation") is not False
        ):
            return None
        boundaries = self.db.get(
            Artifact, str(mapping.get("shot_boundaries_artifact_id") or "")
        )
        detections = self.db.get(
            Artifact, str(mapping.get("detections_artifact_id") or "")
        )
        if boundaries is None or detections is None:
            return None
        if not self._automatic_artifact_is_usable(
            boundaries,
            project=project,
            revision_id=session.revision_id,
            event_id=session.event_id,
            scene_id=session.scene_id,
        ):
            return None
        if not self._detection_artifact_is_usable(
            detections,
            project=project,
            revision_id=session.revision_id,
            event_id=session.event_id,
            scene_id=session.scene_id,
        ):
            return None
        return dict(mapping), boundaries, detections

    def _response(
        self, session: ShotBoundaryReviewSession
    ) -> ShotBoundaryReviewResponse:
        metadata = session.scene_video_artifact.metadata_ or {}
        automatic_cuts = []
        for row in (session.automatic_draft_json or {}).get("cuts") or []:
            cut = dict(row)
            base = (
                f"/api/v1/projects/{session.project_id}/highlight/revisions/{session.revision_id}"
                f"/events/{session.event_id}/shot-boundaries/media"
            )
            if cut.pop("before_thumbnail_name", None):
                cut["before_thumbnail_url"] = (
                    f"{base}/cut/{cut['cut_frame']}/before?scene_id={session.scene_id}"
                )
            if cut.pop("after_thumbnail_name", None):
                cut["after_thumbnail_url"] = (
                    f"{base}/cut/{cut['cut_frame']}/after?scene_id={session.scene_id}"
                )
            automatic_cuts.append(cut)

        effective_status = str(session.status)
        authoritative_artifact_id = session.confirmed_artifact_id
        boundary_origin: str | None = None
        human_reviewed = False
        review_required = session.status not in {"CONFIRMED"}
        detections_status = (
            "READY"
            if session.detections_artifact_id
            else ("FAILED_RETRYABLE" if session.error_json else None)
        )

        automatic_ready = self._automatic_ready_mapping(session)
        if automatic_ready is not None:
            _mapping, boundaries, _detections = automatic_ready
            # The review session remains available as an optional correction
            # workspace, but normal product flow does not pause here.
            effective_status = "AUTO_READY"
            authoritative_artifact_id = boundaries.artifact_id
            boundary_origin = "AUTO_DETECTED"
            human_reviewed = False
            review_required = False
            detections_status = "READY"
        elif session.status == "CONFIRMED" and session.confirmed_artifact_id:
            effective_status = "CONFIRMED"
            authoritative_artifact_id = session.confirmed_artifact_id
            boundary_origin = "HUMAN_REVIEWED"
            human_reviewed = True
            review_required = False

        response_shots = [
            dict(row) for row in ((session.draft_json or {}).get("shots") or [])
        ]
        if effective_status == "AUTO_READY":
            for row in response_shots:
                row["review_status"] = "AUTO_ACCEPTED"

        return ShotBoundaryReviewResponse(
            status=effective_status,
            project_id=session.project_id,
            revision_id=session.revision_id,
            event_id=session.event_id,
            scene_id=session.scene_id,
            review_session_id=session.session_id,
            scene_video_url=f"/api/v1/media/{session.scene_video_asset_id}/stream",
            scene_video_sha256=metadata.get("scene_video_sha256"),
            source_video_asset_id=session.source_video_asset_id,
            source_video_sha256=metadata.get("source_video_sha256"),
            source_start_seconds=metadata.get("source_start_seconds"),
            source_end_seconds=metadata.get("source_end_seconds"),
            fps=metadata.get("fps"),
            frame_count=metadata.get("frame_count"),
            width=metadata.get("width"),
            height=metadata.get("height"),
            draft_revision=session.draft_revision,
            shots=response_shots,
            automatic_cuts=automatic_cuts,
            contact_sheet_url=(
                f"/api/v1/projects/{session.project_id}/highlight/revisions/{session.revision_id}"
                f"/events/{session.event_id}/shot-boundaries/media/contact-sheet?scene_id={session.scene_id}"
            ),
            confirmed_artifact_id=session.confirmed_artifact_id,
            authoritative_artifact_id=authoritative_artifact_id,
            boundary_origin=boundary_origin,
            human_reviewed=human_reviewed,
            review_required=review_required,
            detections_status=detections_status,
            error=session.error_json or None,
            automatic_confirmation=False,
        )

    def media_path(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        media_kind: str,
        cut_frame: int | None = None,
        side: str | None = None,
    ) -> Path:
        self._scope(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        session = self._required_session(
            project.project_id, revision_id, event_id, scene_id, user.user_id
        )
        root = self.storage.resolve_path(session.scene_video_asset.file_path).parent
        if media_kind == "contact-sheet":
            path = root / "contact_sheet.jpg"
        elif (
            media_kind == "cut"
            and cut_frame is not None
            and side in {"before", "after"}
        ):
            path = root / "cut-thumbnails" / f"cut_{cut_frame:08d}_{side}.jpg"
            if not path.is_file():
                frame_count = int(
                    (session.scene_video_artifact.metadata_ or {}).get("frame_count")
                    or 0
                )
                target = cut_frame - 1 if side == "before" else cut_frame
                if target < 0 or target >= frame_count:
                    raise ShotBoundaryWorkflowError(
                        "SHOT_BOUNDARY_MEDIA_NOT_FOUND",
                        "Review media not found.",
                        http_status=404,
                    )
                try:
                    import cv2
                except ImportError as exc:
                    raise ShotBoundaryWorkflowError(
                        "SHOT_BOUNDARY_MEDIA_NOT_FOUND",
                        "Review media not found.",
                        http_status=404,
                    ) from exc
                capture = cv2.VideoCapture(
                    str(self.storage.resolve_path(session.scene_video_asset.file_path))
                )
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = capture.read()
                capture.release()
                if ok:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), frame)
        else:
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_MEDIA_NOT_FOUND",
                "Review media not found.",
                http_status=404,
            )
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ShotBoundaryWorkflowError(
                "SHOT_BOUNDARY_MEDIA_NOT_FOUND",
                "Review media not found.",
                http_status=404,
            )
        return path

    def _confirm_response(
        self, session: ShotBoundaryReviewSession, artifact: Artifact | None
    ) -> ShotBoundaryConfirmResponse:
        if artifact is None:
            raise ShotBoundaryWorkflowError(
                "REVIEWED_SHOT_BOUNDARIES_NOT_READY", "Confirmed artifact is missing."
            )
        return ShotBoundaryConfirmResponse(
            artifact_id=artifact.artifact_id,
            artifact_sha256=str((artifact.metadata_ or {}).get("sha256") or ""),
            reviewed_shot_count=len((session.draft_json or {}).get("shots") or []),
            scene_video_sha256=str(
                (session.scene_video_artifact.metadata_ or {}).get("scene_video_sha256")
                or ""
            ),
            detections_status="READY"
            if session.detections_artifact_id
            else "FAILED_RETRYABLE",
            detections_artifact_id=session.detections_artifact_id,
        )
