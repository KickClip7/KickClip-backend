from __future__ import annotations

import base64
import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.auth.model import User
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.project.model import Project
from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.errors import (
    TrackingConflictError,
    TrackingContractError,
    TrackingUnavailableError,
    TrackingValidationError,
)
from app.domains.tracking.input_contract import validate_tracking_clip
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.execution import LEGACY_EXECUTION_KIND
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateMemoryRevisionR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
)
from app.domains.tracking.repository import TrackingJobRepository
from app.domains.tracking.schema import (
    TrackingAmbiguityConfirmationRequest,
    TrackingArtifactRead,
    TrackingArtifactsResponse,
    TrackingCandidateRead,
    TrackingErrorRead,
    TrackingJobCreateRequest,
    TrackingJobCreateResponse,
    TrackingJobResponse,
    TrackingPendingActionResponse,
    TrackingReviewRequest,
)
from app.domains.tracking.state_mapper import read_pipeline_state
from app.domains.tracking.status import (
    WAITING_STATUSES,
    TrackingBackendStatus,
    tracking_outcome,
    tracking_progress,
    tracking_retryable,
)
from app.domains.tracking.validation import (
    build_action_key,
    generate_tracking_test_name,
    pending_review_candidate_ids,
    validate_bbox_xyxy,
)
from app.domains.tracking.verifier import (
    SceneTargetTrackingInstallationVerifier,
    TrackingInstallationVerifier,
    configured_absolute_path,
    get_scene_target_tracking_verifier,
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
        scene_target_verifier: (
            SceneTargetTrackingInstallationVerifier | None
        ) = None,
        artifacts: TrackingArtifactService | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repository = TrackingJobRepository(db)
        self.media_repository = MediaAssetRepository(db)
        self.verifier = verifier or get_tracking_verifier()
        self.scene_target_verifier = (
            scene_target_verifier or get_scene_target_tracking_verifier()
        )
        self.artifacts = artifacts or TrackingArtifactService(self.settings)
        self.storage = LocalStorage()

    def create_job(
        self,
        *,
        user: User,
        asset: MediaAsset,
        payload: TrackingJobCreateRequest,
        project: Project | None,
        input_validation: Mapping[str, Any] | None = None,
        input_artifacts: Mapping[str, str] | None = None,
        cache_discriminator: str | None = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> TrackingJobCreateResponse:
        self._validate_video_asset(asset)
        if payload.match_id is not None and payload.match_id != asset.match_id:
            raise TrackingValidationError(
                "match_id does not match the selected MediaAsset."
            )
        if project is not None and project.match_id != asset.match_id:
            raise TrackingValidationError(
                "project_id does not belong to the selected MediaAsset's Match."
            )
        locked_asset = self.media_repository.get_for_update(asset.asset_id)
        if locked_asset is None:
            raise TrackingValidationError("MediaAsset no longer exists.")
        asset = locked_asset
        has_scene_target_context = isinstance(
            (runtime_context or {}).get("scene_target_selection"),
            Mapping,
        )
        if has_scene_target_context:
            installation = self.scene_target_verifier.check()
            if not installation.available:
                raise TrackingUnavailableError(installation.message)
        reusable = self.repository.find_equivalent_reusable(
            owner_id=user.user_id,
            media_asset_id=asset.asset_id,
            project_id=project.project_id if project is not None else None,
            initial_bbox=payload.initial_bbox_xyxy,
            bbox_format=payload.bbox_format,
            reacquisition_mode=self.settings.TRACKING_REACQUISITION_MODE,
            cache_discriminator=cache_discriminator,
        )
        if reusable is not None:
            return self._create_response(reusable, reused=True)

        if not has_scene_target_context:
            installation = self.verifier.check()
            if not installation.available:
                raise TrackingUnavailableError(installation.message)

        video_path = self.storage.resolve_path(asset.file_path)
        if not video_path.is_file():
            raise TrackingValidationError("MediaAsset video file is missing.")
        metadata = validate_tracking_clip(
            video_path,
            requested_duration_sec=(
                float(asset.duration_sec)
                if asset.duration_sec is not None
                else None
            ),
        )
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
                execution_kind=LEGACY_EXECUTION_KIND,
                output_directory=str(output_directory),
                pipeline_state_path=str(output_directory / "pipeline_state.json"),
                queued_action={"kind": "new"},
                runtime_metadata={
                    "source_video": {
                        "width": width,
                        "height": height,
                        "fps": float(metadata["fps"]),
                        "frame_count": metadata.get("frame_count"),
                        "duration_sec": metadata.get("duration_sec"),
                        "sha256": asset.sha256,
                    },
                    "input_validation": dict(
                        input_validation or metadata.get("validation") or {}
                    ),
                    "input_artifacts": dict(input_artifacts or {}),
                    **dict(runtime_context or {}),
                },
            )
            self.db.commit()
            self.db.refresh(job)
        except Exception:
            self.db.rollback()
            raise

        return self._create_response(job, reused=False)

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
        if job.last_action_key == key and job.status not in WAITING_STATUSES:
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
                "candidate_id must be omitted for absent/none-of-these/non-player-role decisions."
            )

        action_kind = {
            "none_of_these": "none_of_these",
            "non_player_role": "non_player_role",
        }.get(payload.decision.value, "ambiguity")
        job.queued_action = {
            "kind": action_kind,
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
        latest = (
            self.db.get(EventCandidateReviewDecisionR1, job.latest_decision_id)
            if job.latest_decision_id
            else None
        )
        memory = (
            self.db.get(EventCandidateMemoryRevisionR1, job.current_memory_revision_id)
            if job.current_memory_revision_id
            else None
        )
        ambiguity = None
        pipeline = self.db.scalar(
            select(EventCandidatePipelineR1).where(
                EventCandidatePipelineR1.tracking_job_id == job.tracking_job_id
            )
        )
        if job.pending_ambiguity_id:
            ambiguity = self.db.scalar(
                select(EventCandidateAmbiguityR1).where(
                    EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id,
                    EventCandidateAmbiguityR1.ambiguity_id == job.pending_ambiguity_id,
                )
            )
        active_candidates: list[dict] = []
        if ambiguity is not None:
            active_candidates = [
                dict(item)
                for item in (ambiguity.candidates or [])
                if str(item.get("status") or "").upper() == "PENDING"
            ]
            active_candidate_ids = [
                str(item.get("candidate_id") or "")
                for item in active_candidates
                if item.get("candidate_id")
            ]
            if list(ambiguity.candidate_ids or []) != active_candidate_ids:
                raise TrackingContractError(
                    "Pending ambiguity candidate_ids do not match PENDING candidates."
                )
        if job.status == TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value:
            if (
                pipeline is None
                or pipeline.pending_ambiguity_id != job.pending_ambiguity_id
                or ambiguity is None
                or ambiguity.status != "WAITING"
                or not active_candidates
            ):
                raise TrackingContractError(
                    "WAITING_CROSS_SHOT_CONFIRMATION requires one active WAITING ambiguity."
                )
        if (
            pending is not None
            and ambiguity is not None
            and pending.ambiguity_id == ambiguity.ambiguity_id
        ):
            pending = pending.model_copy(
                update={
                    "candidate_ids": list(ambiguity.candidate_ids or []),
                    "candidates": [
                        TrackingCandidateRead.model_validate(item)
                        for item in active_candidates
                    ],
                }
            )
        artifact_readiness = {
            "pipeline_state": Path(job.pipeline_state_path).is_file(),
            "timeline": bool(job.timeline_path and Path(job.timeline_path).is_file()),
            "pending_ambiguity": bool(ambiguity and active_candidates),
            "target_memory": bool(memory and memory.reference_frame_ids and memory.references),
        }
        preview_readiness = {
            "full_frame": bool(job.tracking_preview_path and Path(job.tracking_preview_path).is_file()),
            "target_centered": bool(job.target_centered_preview_path and Path(job.target_centered_preview_path).is_file()),
        }
        return TrackingJobResponse(
            job_id=job.tracking_job_id,
            owner_user_id=job.owner_id,
            match_id=job.match_id,
            project_id=job.project_id,
            media_asset_id=job.media_asset_id,
            status=TrackingBackendStatus(job.status),
            execution_kind=job.execution_kind,
            pipeline_stage=job.pipeline_stage,
            processing_status=job.processing_status,
            outcome=tracking_outcome(job.status, job.error_type),
            progress=tracking_progress(job.status, job.current_stage),
            retryable=tracking_retryable(job.status, job.error_type),
            status_url=f"/api/v1/tracking/jobs/{job.tracking_job_id}",
            pipeline_status=job.pipeline_status,
            pipeline_decision=job.pipeline_decision,
            current_stage=job.current_stage,
            pending_ambiguity_id=job.pending_ambiguity_id,
            pending_candidates=list(pending.candidates) if pending else [],
            latest_decision=(
                {
                    "decision_id": latest.decision_id,
                    "ambiguity_id": latest.ambiguity_id,
                    "candidate_id": latest.candidate_id,
                    "state": latest.decision_state,
                    "artifact_sha256": latest.decision_artifact_sha256,
                }
                if latest else None
            ),
            current_memory_revision=(
                {
                    "memory_revision_id": memory.memory_revision_id,
                    "previous_memory_revision_id": memory.previous_memory_revision_id,
                    "previous_memory_sha256": memory.previous_memory_sha256,
                    "new_memory_sha256": memory.new_memory_sha256,
                    "source_candidate_id": memory.source_candidate_id,
                    "reference_frame_ids": list(memory.reference_frame_ids),
                    "references": list(memory.references),
                    "crop_sha256": (
                        memory.references[0].get("crop_sha256")
                        if memory.references else None
                    ),
                    "scale_banks": dict(memory.scale_banks),
                    "artifact_sha256": memory.artifact_sha256,
                }
                if memory else None
            ),
            next_ambiguity=(
                {
                    "ambiguity_id": ambiguity.ambiguity_id,
                    "shot_id": ambiguity.shot_id,
                    "status": ambiguity.status,
                    "candidate_ids": list(ambiguity.candidate_ids),
                    "candidates": active_candidates,
                    "artifact_sha256": ambiguity.artifact_sha256,
                    "full_frame_context_sha256": ambiguity.full_frame_context_sha256,
                    "shot_clip_sha256": ambiguity.shot_clip_sha256,
                    "generation": ambiguity.generation,
                }
                if ambiguity else None
            ),
            review_progress={
                "reviewed_candidate_count": int(
                    ((pipeline.summary or {}) if pipeline else {}).get(
                        "reviewed_candidate_count", 0
                    )
                ),
                "remaining_candidate_count": len(active_candidates)
                if ambiguity
                else 0,
                "candidate_batch_generation": int(ambiguity.generation)
                if ambiguity
                else int(
                    ((pipeline.summary or {}) if pipeline else {}).get(
                        "candidate_batch_generation", 0
                    )
                ),
                "current_shot": job.current_shot_id,
                "preparing_next_candidates": bool(
                    job.status in {
                        TrackingBackendStatus.QUEUED.value,
                        TrackingBackendStatus.RUNNING.value,
                    }
                    and not job.pending_ambiguity_id
                ),
            },
            current_shot=job.current_shot_id,
            next_shot=job.next_shot_id,
            completed=job.pipeline_stage == "COMPLETED",
            completed_at=job.completed_at,
            failure_code=job.failure_code,
            artifact_readiness=artifact_readiness,
            preview_readiness=preview_readiness,
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

    @staticmethod
    def _create_response(
        job: TrackingJob,
        *,
        reused: bool,
    ) -> TrackingJobCreateResponse:
        return TrackingJobCreateResponse(
            job_id=job.tracking_job_id,
            status=TrackingBackendStatus(job.status),
            outcome=tracking_outcome(job.status, job.error_type),
            progress=tracking_progress(job.status, job.current_stage),
            retryable=tracking_retryable(job.status, job.error_type),
            reused=reused,
            status_url=f"/api/v1/tracking/jobs/{job.tracking_job_id}",
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
            required = "Select a listed candidate, choose none of these, or confirm that the target is absent."
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
        candidates = self._public_ambiguity_candidates(
            job=job,
            state=state,
            pending=pending,
            candidate_ids=candidate_ids,
        )
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
            candidates=candidates,
            recommended_candidate=(
                str(pending["recommended_candidate"])
                if pending.get("recommended_candidate")
                else None
            ),
            artifact_keys=sorted(artifact_keys),
            required_action=required,
        )

    def _public_ambiguity_candidates(
        self,
        *,
        job: TrackingJob,
        state: Mapping[str, Any],
        pending: Mapping[str, Any],
        candidate_ids: list[str],
    ) -> list[TrackingCandidateRead]:
        if str(pending.get("type") or "") != "CROSS_SHOT_CONFIRMATION":
            return []

        ambiguity_id = str(pending.get("ambiguity_id") or "")
        ambiguity = next(
            (
                value
                for value in state.get("ambiguities", [])
                if isinstance(value, Mapping)
                and str(value.get("ambiguity_id") or "") == ambiguity_id
            ),
            {},
        )
        candidate_metadata = {
            str(value.get("candidate_id")): value
            for value in ambiguity.get("review_candidates", [])
            if isinstance(value, Mapping) and value.get("candidate_id")
        }
        shot_id = str(pending.get("shot_id") or ambiguity.get("shot_id") or "")
        safe_shot_id = self.artifacts._safe_component(shot_id)
        root = self.artifacts.job_root(job)

        rows: list[TrackingCandidateRead] = []
        for fallback_rank, candidate_id in enumerate(candidate_ids, start=1):
            safe_candidate_id = self.artifacts._safe_component(candidate_id)
            metadata = candidate_metadata.get(candidate_id, {})
            frame_urls: list[str] = []
            if safe_shot_id and safe_candidate_id:
                strip = (
                    root
                    / "work"
                    / "shots"
                    / safe_shot_id
                    / "candidate_strips"
                    / f"{safe_candidate_id}.jpg"
                ).resolve()
                if strip.is_relative_to(root) and strip.is_file():
                    mime_type = (
                        mimetypes.guess_type(strip.name)[0] or "image/jpeg"
                    )
                    encoded = base64.b64encode(strip.read_bytes()).decode("ascii")
                    frame_urls.append(f"data:{mime_type};base64,{encoded}")

            rank_value = metadata.get("retrieval_rank")
            score_value = metadata.get("retrieval_score")
            rows.append(
                TrackingCandidateRead(
                    candidate_id=candidate_id,
                    rank=(
                        int(rank_value)
                        if rank_value is not None
                        else fallback_rank
                    ),
                    score=(
                        float(score_value)
                        if score_value is not None
                        else None
                    ),
                    frame_image_urls=frame_urls,
                    shot_id=str(metadata.get("shot_id") or "") or None,
                    tracklet_id=str(metadata.get("tracklet_id") or "") or None,
                    reviewability=str(metadata.get("reviewability") or "") or None,
                    best_frame=(int(metadata["best_frame"]) if metadata.get("best_frame") is not None else None),
                    review_bundle={
                        key: value
                        for key, value in metadata.items()
                        if key.endswith("_path") or key.endswith("_sha256")
                    },
                )
            )
        return rows

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
