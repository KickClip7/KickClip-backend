from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.auth.model import User
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.errors import (
    TrackingConflictError,
    TrackingContractError,
    TrackingUnavailableError,
    TrackingValidationError,
)
from app.domains.tracking.media_probe import probe_tracking_video
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.repository import TrackingJobRepository
from app.domains.tracking.schema import (
    TrackingAmbiguityConfirmationRequest,
    TrackingArtifactRead,
    TrackingArtifactsResponse,
    TrackingErrorRead,
    TrackingJobCreateRequest,
    TrackingJobCreateResponse,
    TrackingJobResponse,
    TrackingPendingActionResponse,
    TrackingReviewRequest,
)
from app.domains.tracking.state_mapper import read_pipeline_state
from app.domains.tracking.status import WAITING_STATUSES, TrackingBackendStatus
from app.domains.tracking.validation import (
    build_action_key,
    generate_tracking_test_name,
    pending_review_candidate_ids,
    validate_bbox_xyxy,
)
from app.domains.tracking.verifier import (
    TrackingInstallationVerifier,
    configured_absolute_path,
    get_tracking_verifier,
)
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts"}


class TrackingJobService:
    def __init__(
        self,
        db: Session,
        *,
        settings: Settings | None = None,
        verifier: TrackingInstallationVerifier | None = None,
        artifacts: TrackingArtifactService | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repository = TrackingJobRepository(db)
        self.verifier = verifier or get_tracking_verifier()
        self.artifacts = artifacts or TrackingArtifactService(self.settings)
        self.storage = LocalStorage()

    def create_job(
        self,
        *,
        user: User,
        asset: MediaAsset,
        payload: TrackingJobCreateRequest,
        project: Project | None,
    ) -> TrackingJobCreateResponse:
        installation = self.verifier.check()
        if not installation.available:
            raise TrackingUnavailableError(installation.message)
        self._validate_video_asset(asset)
        if payload.match_id is not None and payload.match_id != asset.match_id:
            raise TrackingValidationError(
                "match_id does not match the selected MediaAsset."
            )
        if project is not None and project.match_id != asset.match_id:
            raise TrackingValidationError(
                "project_id does not belong to the selected MediaAsset's Match."
            )

        video_path = self.storage.resolve_path(asset.file_path)
        if not video_path.is_file():
            raise TrackingValidationError("MediaAsset video file is missing.")
        metadata = probe_tracking_video(video_path)
        width = int(metadata["width"])
        height = int(metadata["height"])
        bbox = validate_bbox_xyxy(
            payload.initial_bbox_xyxy,
            width=width,
            height=height,
        )

        job_id = generate_prefixed_id("trk")
        test_name = generate_tracking_test_name(job_id)
        output_root = configured_absolute_path(
            self.settings.TRACKING_OUTPUT_ROOT,
            "TRACKING_OUTPUT_ROOT",
        )
        output_directory = (output_root / test_name).resolve()
        if not output_directory.is_relative_to(output_root):
            raise TrackingValidationError("Generated tracking output path is invalid.")

        try:
            job = self.repository.create(
                tracking_job_id=job_id,
                owner_id=user.user_id,
                match_id=asset.match_id,
                project_id=project.project_id if project is not None else None,
                media_asset_id=asset.asset_id,
                test_name=test_name,
                initial_bbox=bbox,
                bbox_format=payload.bbox_format,
                device=self.settings.TRACKING_DEVICE,
                reacquisition_mode=self.settings.TRACKING_REACQUISITION_MODE,
                status=TrackingBackendStatus.QUEUED.value,
                output_directory=str(output_directory),
                pipeline_state_path=str(output_directory / "pipeline_state.json"),
                queued_action={"kind": "new"},
                runtime_metadata={
                    "source_video": {
                        "width": width,
                        "height": height,
                        "fps": float(metadata["fps"]),
                        "frame_count": metadata.get("frame_count"),
                    }
                },
            )
            self.db.commit()
            self.db.refresh(job)
        except Exception:
            self.db.rollback()
            raise

        return TrackingJobCreateResponse(
            job_id=job.tracking_job_id,
            status=TrackingBackendStatus(job.status),
            status_url=f"/api/v1/tracking/jobs/{job.tracking_job_id}",
        )

    def queue_review(
        self,
        *,
        tracking_job_id: str,
        user: User,
        payload: TrackingReviewRequest,
    ) -> TrackingJob:
        job = self.repository.get_for_update(tracking_job_id)
        if job is None:
            raise TrackingConflictError("Tracking job no longer exists.")
        key = build_action_key("review", payload.stage.value, payload.decision.value)
        if job.owner_id != user.user_id and not user.developer_mode_enabled:
            raise TrackingConflictError()
        if job.last_action_key == key:
            self.db.commit()
            return job

        state = self._load_current_state(job)
        pending = state.get("pending_action")
        if not isinstance(pending, Mapping):
            raise TrackingConflictError()
        pending_type = str(pending.get("type") or "")
        expected_stage = str(pending.get("review_stage") or "")
        if pending_type == "MEMORY_REVIEW":
            expected_stage = "MEMORY"
            expected_status = TrackingBackendStatus.WAITING_MEMORY_REVIEW.value
        elif pending_type in {"SEGMENT_VISUAL_REVIEW", "PHASE1_INTERNAL_REVIEW"}:
            expected_status = TrackingBackendStatus.WAITING_SEGMENT_REVIEW.value
        else:
            raise TrackingConflictError()
        if job.status != expected_status or payload.stage.value != expected_stage:
            raise TrackingConflictError(
                f"Pipeline is waiting for review stage {expected_stage or 'unknown'}."
            )

        job.queued_action = {
            "kind": "review",
            "stage": payload.stage.value,
            "decision": payload.decision.value,
            "reviewer": user.user_id,
            "note": payload.note,
        }
        self._mark_queued(job, key)
        self.db.commit()
        self.db.refresh(job)
        return job

    def queue_ambiguity_confirmation(
        self,
        *,
        tracking_job_id: str,
        ambiguity_id: str,
        user: User,
        payload: TrackingAmbiguityConfirmationRequest,
    ) -> TrackingJob:
        candidate = payload.candidate_id or ""
        key = build_action_key(
            "ambiguity",
            ambiguity_id,
            payload.decision.value,
            candidate,
        )
        job = self.repository.get_for_update(tracking_job_id)
        if job is None:
            raise TrackingConflictError("Tracking job no longer exists.")
        if job.owner_id != user.user_id and not user.developer_mode_enabled:
            raise TrackingConflictError()
        if job.last_action_key == key:
            self.db.commit()
            return job
        if (
            job.status
            != TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
            or job.pending_ambiguity_id != ambiguity_id
        ):
            raise TrackingConflictError()

        state = self._load_current_state(job)
        allowed = pending_review_candidate_ids(state, ambiguity_id)
        if payload.decision.value == "candidate":
            if not candidate:
                raise TrackingValidationError(
                    "candidate_id is required for candidate confirmation."
                )
            if candidate not in allowed:
                raise TrackingValidationError(
                    "candidate_id is not in the current review candidate list."
                )
        elif payload.candidate_id is not None:
            raise TrackingValidationError(
                "candidate_id must be omitted when confirming target absence."
            )

        job.queued_action = {
            "kind": "ambiguity",
            "ambiguity_id": ambiguity_id,
            "decision": payload.decision.value,
            "candidate_id": candidate or None,
            "reviewer": user.user_id,
            "note": payload.note,
        }
        self._mark_queued(job, key)
        self.db.commit()
        self.db.refresh(job)
        return job

    def to_response(self, job: TrackingJob) -> TrackingJobResponse:
        pending = self._public_pending_action(job)
        timeline = (job.artifact_index or {}).get("target_timeline_json") or {}
        has_timeline = bool(timeline.get("exists"))
        error = (
            TrackingErrorRead(type=job.error_type, message=job.error_message)
            if job.error_type or job.error_message
            else None
        )
        return TrackingJobResponse(
            job_id=job.tracking_job_id,
            owner_user_id=job.owner_id,
            match_id=job.match_id,
            project_id=job.project_id,
            media_asset_id=job.media_asset_id,
            status=TrackingBackendStatus(job.status),
            pipeline_status=job.pipeline_status,
            pipeline_decision=job.pipeline_decision,
            current_stage=job.current_stage,
            initial_bbox_xyxy=list(job.initial_bbox),
            bbox_format=job.bbox_format,
            device=job.device,
            reacquisition_mode=job.reacquisition_mode,
            pending_action=pending,
            artifacts_url=(
                f"/api/v1/tracking/jobs/{job.tracking_job_id}/artifacts"
            ),
            timeline_url=(
                f"/api/v1/tracking/jobs/{job.tracking_job_id}/timeline"
                if has_timeline
                else None
            ),
            schema_version=job.schema_version,
            pipeline_version=job.pipeline_version,
            video_sha256=job.video_sha256,
            frozen_manifest_present=job.frozen_manifest_present,
            process_return_code=job.process_return_code,
            error=error,
            created_at=job.created_at,
            started_at=job.started_at,
            updated_at=job.updated_at,
            finished_at=job.finished_at,
        )

    def artifacts_response(self, job: TrackingJob) -> TrackingArtifactsResponse:
        rows: list[TrackingArtifactRead] = []
        for key, record in sorted((job.artifact_index or {}).items()):
            if not isinstance(record, Mapping):
                continue
            exists = bool(record.get("exists"))
            rows.append(
                TrackingArtifactRead(
                    key=key,
                    kind=str(record.get("kind") or "artifact"),
                    exists=exists,
                    mime_type=(
                        str(record["mime_type"])
                        if record.get("mime_type")
                        else None
                    ),
                    size_bytes=(
                        int(record["size_bytes"])
                        if record.get("size_bytes") is not None
                        else None
                    ),
                    updated_at=record.get("updated_at"),
                    sha256=(
                        str(record["sha256"]) if record.get("sha256") else None
                    ),
                    url=(
                        f"/api/v1/tracking/jobs/{job.tracking_job_id}"
                        f"/artifacts/{key}"
                        if exists
                        else None
                    ),
                )
            )
        return TrackingArtifactsResponse(
            job_id=job.tracking_job_id,
            artifacts=rows,
        )

    def _public_pending_action(
        self,
        job: TrackingJob,
    ) -> TrackingPendingActionResponse | None:
        if job.status not in WAITING_STATUSES:
            return None
        try:
            state = self._load_current_state(job)
        except TrackingContractError:
            return None
        pending = state.get("pending_action")
        if not isinstance(pending, Mapping):
            return None
        action_type = str(pending.get("type") or "")
        candidate_ids = [
            str(value)
            for value in pending.get("candidate_ids", [])
            if isinstance(value, str)
        ]
        if action_type == "MEMORY_REVIEW":
            required = "Approve or reject the MEMORY review after inspecting artifacts."
        elif action_type == "CROSS_SHOT_CONFIRMATION":
            required = "Select a listed candidate or confirm that the target is absent."
        else:
            required = (
                "Approve or reject the exact visual review stage after inspection."
            )
        artifact_keys = [
            key
            for key, record in (job.artifact_index or {}).items()
            if isinstance(record, Mapping)
            and bool(record.get("exists"))
            and record.get("kind")
            in {"review_contact_sheet", "review_preview", "ambiguity_contact_sheet"}
        ]
        return TrackingPendingActionResponse(
            type=action_type,
            review_stage=(
                str(pending["review_stage"])
                if pending.get("review_stage")
                else None
            ),
            ambiguity_id=(
                str(pending["ambiguity_id"])
                if pending.get("ambiguity_id")
                else None
            ),
            shot_id=str(pending["shot_id"]) if pending.get("shot_id") else None,
            candidate_ids=candidate_ids,
            recommended_candidate=(
                str(pending["recommended_candidate"])
                if pending.get("recommended_candidate")
                else None
            ),
            artifact_keys=sorted(artifact_keys),
            required_action=required,
        )

    def _load_current_state(self, job: TrackingJob) -> dict[str, Any]:
        state = read_pipeline_state(Path(job.pipeline_state_path))
        if state is None:
            raise TrackingContractError("Tracking pipeline state is missing.")
        if state.get("test_name") != job.test_name:
            raise TrackingContractError("Tracking pipeline state job mismatch.")
        return state

    @staticmethod
    def _mark_queued(job: TrackingJob, action_key: str) -> None:
        job.status = TrackingBackendStatus.QUEUED.value
        job.last_action_key = action_key
        job.error_type = None
        job.error_message = None
        job.process_return_code = None
        job.finished_at = None

    @staticmethod
    def _validate_video_asset(asset: MediaAsset) -> None:
        mime_is_video = bool(
            asset.mime_type and asset.mime_type.lower().startswith("video/")
        )
        type_is_video = "VIDEO" in asset.asset_type.upper()
        suffix_is_video = Path(asset.file_path).suffix.lower() in VIDEO_SUFFIXES
        if not (mime_is_video or type_is_video or suffix_is_video):
            raise TrackingValidationError(
                "Selected MediaAsset is not a supported video."
            )

