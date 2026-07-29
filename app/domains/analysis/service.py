from sqlalchemy.orm import Session

from app.domains.analysis.job_status import QUEUED
from app.domains.analysis.job_steps import (
    get_default_steps_for_job_type,
    get_first_pending_step_key,
)
from app.domains.analysis.job_types import normalize_job_type
from app.domains.analysis.model import AnalysisJob
from app.domains.analysis.repository import (
    AnalysisJobRepository,
    AnalysisJobStepRepository,
)
from app.domains.analysis.schema import (
    AnalysisJobCreate,
    AnalysisJobCreateRequest,
    AnalysisJobCreateResponse,
    AnalysisJobStatusResponse,
    AnalysisJobStepCompactRead,
)
from app.domains.match.repository import MatchRepository
from app.domains.action_spotting.status import (
    action_spotting_public_error,
    action_spotting_workflow_status,
)


class AnalysisJobService:
    def __init__(self, db: Session):
        self.db = db
        self.job_repository = AnalysisJobRepository(db)
        self.step_repository = AnalysisJobStepRepository(db)
        self.match_repository = MatchRepository(db)

    def create_analysis_job(self, data: AnalysisJobCreate) -> AnalysisJob:
        job_type = normalize_job_type(data.job_type)

        match = self.match_repository.get_by_id(data.match_id)
        if match is None:
            raise ValueError("Match not found")

        default_steps = get_default_steps_for_job_type(job_type)
        current_step = get_first_pending_step_key(default_steps)

        job = self.job_repository.create_job(
            match_id=data.match_id,
            job_type=job_type,
            status=QUEUED,
            progress=0,
            current_step=current_step,
            options=data.options,
            media_asset_id=data.media_asset_id,
            cache_key=data.cache_key,
            video_sha256=data.video_sha256,
            model_version=data.model_version,
            policy_version=data.policy_version,
        )

        for step_data in default_steps:
            self.step_repository.create_step(
                analysis_job_id=job.analysis_job_id,
                **step_data,
            )

        self.db.commit()

        loaded_job = self.job_repository.get_by_id(job.analysis_job_id)
        if loaded_job is None:
            raise RuntimeError("Created analysis job could not be loaded.")

        return loaded_job

    def create_analysis_job_for_match(
        self,
        match_id: str,
        data: AnalysisJobCreateRequest,
    ) -> AnalysisJobCreateResponse:
        job_type = normalize_job_type(data.job_type)
        # Serialize creation per Match on databases that support row locks.
        # This closes the common double-click/concurrent-request race without a
        # schema change and keeps the existing POST contract intact.
        match = self.match_repository.get_for_update(match_id)
        if match is None:
            raise ValueError("Match not found")
        for existing in self.job_repository.list_active_by_match_and_type(
            match_id=match_id,
            job_type=job_type,
        ):
            if (existing.options or {}) == (data.options or {}):
                response = self.to_create_response(existing)
                return response.model_copy(update={"reused": True})
        job = self.create_analysis_job(
            AnalysisJobCreate(
                match_id=match_id,
                job_type=job_type,
                options=data.options,
            )
        )
        return self.to_create_response(job)

    def get_analysis_job(self, analysis_job_id: str) -> AnalysisJob | None:
        return self.job_repository.get_by_id(analysis_job_id)

    def get_analysis_job_status(
        self,
        analysis_job_id: str,
    ) -> AnalysisJobStatusResponse | None:
        job = self.job_repository.get_by_id(analysis_job_id)
        if job is None:
            return None
        return self.to_status_response(job)

    def list_match_analysis_jobs(self, match_id: str) -> list[AnalysisJob]:
        return self.job_repository.list_by_match(match_id)

    def to_create_response(self, job: AnalysisJob) -> AnalysisJobCreateResponse:
        return AnalysisJobCreateResponse(
            analysis_job_id=job.analysis_job_id,
            status=job.status.lower(),
            progress=job.progress,
            current_step=job.current_step,
            steps=[
                AnalysisJobStepCompactRead(
                    key=step.step_key,
                    label=step.label,
                    status=step.status.lower(),
                    progress=step.progress,
                )
                for step in job.steps
            ],
        )

    def to_status_response(self, job: AnalysisJob) -> AnalysisJobStatusResponse:
        is_action_spotting = job.job_type == "HIGHLIGHT_SPOTTING"
        return AnalysisJobStatusResponse(
            analysis_job_id=job.analysis_job_id,
            match_id=job.match_id,
            job_type=job.job_type,
            status=job.status.lower(),
            progress=job.progress,
            current_step=job.current_step,
            options=job.options,
            media_asset_id=job.media_asset_id,
            video_sha256=job.video_sha256,
            model_version=job.model_version,
            policy_version=job.policy_version,
            steps=[
                AnalysisJobStepCompactRead(
                    key=step.step_key,
                    label=step.label,
                    status=step.status.lower(),
                    progress=step.progress,
                )
                for step in job.steps
            ],
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            retryable=job.status == "FAILED",
            workflow_status=(
                action_spotting_workflow_status(
                    status=job.status,
                    options=job.options,
                )
                if is_action_spotting
                else None
            ),
            workflow_state_history=(
                list(
                    (job.options or {}).get(
                        "action_spotting_state_history"
                    )
                    or []
                )
                if is_action_spotting
                else []
            ),
            error=(
                action_spotting_public_error(job.options)
                if is_action_spotting
                else None
            ),
        )
