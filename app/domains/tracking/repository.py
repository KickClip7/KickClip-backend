from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domains.tracking.model import TrackingJob
from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.execution import LEGACY_EXECUTION_KIND, R1_EXECUTION_KIND


class TrackingJobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> TrackingJob:
        job = TrackingJob(**kwargs)
        self.db.add(job)
        self.db.flush()
        return job

    def get_by_id(self, tracking_job_id: str) -> TrackingJob | None:
        return self.db.scalar(
            select(TrackingJob).where(
                TrackingJob.tracking_job_id == tracking_job_id
            )
        )

    def get_for_update(self, tracking_job_id: str) -> TrackingJob | None:
        return self.db.scalar(
            select(TrackingJob)
            .where(TrackingJob.tracking_job_id == tracking_job_id)
            .with_for_update()
        )

    def claim_queued(
        self,
        tracking_job_id: str,
        *,
        execution_kind: str = LEGACY_EXECUTION_KIND,
    ) -> bool:
        result = self.db.execute(
            update(TrackingJob)
            .where(
                TrackingJob.tracking_job_id == tracking_job_id,
                TrackingJob.status == TrackingBackendStatus.QUEUED.value,
                TrackingJob.execution_kind == execution_kind,
            )
            .values(
                status=TrackingBackendStatus.RUNNING.value,
                run_attempt=TrackingJob.run_attempt + 1,
            )
        )
        return bool(result.rowcount)

    def list_recoverable(
        self,
        *,
        execution_kind: str = LEGACY_EXECUTION_KIND,
    ) -> list[TrackingJob]:
        return list(
            self.db.scalars(
                select(TrackingJob)
                .where(
                    TrackingJob.status.in_(
                        [
                            TrackingBackendStatus.QUEUED.value,
                            TrackingBackendStatus.RUNNING.value,
                        ]
                    ),
                    TrackingJob.execution_kind == execution_kind,
                )
                .order_by(TrackingJob.created_at.asc())
            ).all()
        )

    def quarantine_unroutable(self) -> int:
        result = self.db.execute(
            update(TrackingJob)
            .where(
                TrackingJob.execution_kind.notin_([LEGACY_EXECUTION_KIND, R1_EXECUTION_KIND]),
                TrackingJob.status.in_([TrackingBackendStatus.QUEUED.value, TrackingBackendStatus.RUNNING.value]),
            )
            .values(
                status=TrackingBackendStatus.FAILED.value,
                processing_status="FAILED",
                failure_code="UNROUTABLE_TRACKING_JOB",
                error_type="UNROUTABLE_TRACKING_JOB",
                error_message="Tracking execution kind is missing or unsupported.",
            )
        )
        return int(result.rowcount or 0)

    def find_equivalent_reusable(
        self,
        *,
        owner_id: str,
        media_asset_id: str,
        project_id: str | None,
        initial_bbox: list[float],
        bbox_format: str,
        reacquisition_mode: str,
        cache_discriminator: str | None = None,
    ) -> TrackingJob | None:
        stmt = (
            select(TrackingJob)
            .where(
                TrackingJob.owner_id == owner_id,
                TrackingJob.media_asset_id == media_asset_id,
                TrackingJob.project_id == project_id,
                TrackingJob.bbox_format == bbox_format,
                TrackingJob.reacquisition_mode == reacquisition_mode,
                TrackingJob.execution_kind == LEGACY_EXECUTION_KIND,
                TrackingJob.status.notin_(
                    [
                        TrackingBackendStatus.FAILED.value,
                        TrackingBackendStatus.CANCELLED.value,
                    ]
                ),
            )
            .order_by(TrackingJob.created_at.desc())
        )
        for job in self.db.scalars(stmt):
            scene_context = (job.runtime_metadata or {}).get(
                "scene_target_selection"
            )
            if cache_discriminator is None:
                if scene_context:
                    continue
            elif not isinstance(scene_context, dict) or (
                scene_context.get("tracking_cache_key")
                != cache_discriminator
            ):
                continue
            if [float(value) for value in job.initial_bbox] == [
                float(value) for value in initial_bbox
            ]:
                return job
        return None
