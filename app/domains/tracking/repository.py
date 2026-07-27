from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domains.tracking.model import TrackingJob
from app.domains.tracking.status import TrackingBackendStatus


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

    def claim_queued(self, tracking_job_id: str) -> bool:
        result = self.db.execute(
            update(TrackingJob)
            .where(
                TrackingJob.tracking_job_id == tracking_job_id,
                TrackingJob.status == TrackingBackendStatus.QUEUED.value,
            )
            .values(
                status=TrackingBackendStatus.RUNNING.value,
                run_attempt=TrackingJob.run_attempt + 1,
            )
        )
        return bool(result.rowcount)

    def list_recoverable(self) -> list[TrackingJob]:
        return list(
            self.db.scalars(
                select(TrackingJob)
                .where(
                    TrackingJob.status.in_(
                        [
                            TrackingBackendStatus.QUEUED.value,
                            TrackingBackendStatus.RUNNING.value,
                        ]
                    )
                )
                .order_by(TrackingJob.created_at.asc())
            ).all()
        )

