from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.domains.analysis.model import AnalysisJob, AnalysisJobStep


class AnalysisJobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, **kwargs) -> AnalysisJob:
        job = AnalysisJob(**kwargs)
        self.db.add(job)
        self.db.flush()
        return job

    def get_by_id(self, analysis_job_id: str) -> AnalysisJob | None:
        stmt = (
            select(AnalysisJob)
            .options(selectinload(AnalysisJob.steps))
            .where(AnalysisJob.analysis_job_id == analysis_job_id)
        )
        return self.db.scalar(stmt)

    def list_by_match(self, match_id: str) -> list[AnalysisJob]:
        stmt = (
            select(AnalysisJob)
            .options(selectinload(AnalysisJob.steps))
            .where(AnalysisJob.match_id == match_id)
            .order_by(AnalysisJob.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def latest_by_match_ids(
        self,
        match_ids: list[str],
    ) -> dict[str, AnalysisJob]:
        """Load one latest durable job per Match in a single query."""
        if not match_ids:
            return {}
        ranked = (
            select(
                AnalysisJob.analysis_job_id.label("analysis_job_id"),
                func.row_number()
                .over(
                    partition_by=AnalysisJob.match_id,
                    order_by=(
                        AnalysisJob.created_at.desc(),
                        AnalysisJob.analysis_job_id.desc(),
                    ),
                )
                .label("row_number"),
            )
            .where(AnalysisJob.match_id.in_(match_ids))
            .subquery()
        )
        stmt = (
            select(AnalysisJob)
            .join(
                ranked,
                ranked.c.analysis_job_id == AnalysisJob.analysis_job_id,
            )
            .where(ranked.c.row_number == 1)
        )
        jobs = list(self.db.scalars(stmt).all())
        return {job.match_id: job for job in jobs}

    def list_active_by_match_and_type(
        self,
        *,
        match_id: str,
        job_type: str,
    ) -> list[AnalysisJob]:
        stmt = (
            select(AnalysisJob)
            .options(selectinload(AnalysisJob.steps))
            .where(
                AnalysisJob.match_id == match_id,
                AnalysisJob.job_type == job_type,
                AnalysisJob.status.in_(["QUEUED", "RUNNING"]),
            )
            .order_by(AnalysisJob.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def get_by_cache_key(self, cache_key: str) -> AnalysisJob | None:
        stmt = (
            select(AnalysisJob)
            .options(selectinload(AnalysisJob.steps))
            .where(AnalysisJob.cache_key == cache_key)
        )
        return self.db.scalar(stmt)

    def claim_queued(self, analysis_job_id: str) -> bool:
        stmt = (
            update(AnalysisJob)
            .where(
                AnalysisJob.analysis_job_id == analysis_job_id,
                AnalysisJob.status == "QUEUED",
            )
            .values(status="RUNNING")
        )
        result = self.db.execute(stmt)
        return int(result.rowcount or 0) == 1


class AnalysisJobStepRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_step(self, **kwargs) -> AnalysisJobStep:
        step = AnalysisJobStep(**kwargs)
        self.db.add(step)
        self.db.flush()
        return step

    def list_by_job(self, analysis_job_id: str) -> list[AnalysisJobStep]:
        stmt = (
            select(AnalysisJobStep)
            .where(AnalysisJobStep.analysis_job_id == analysis_job_id)
            .order_by(AnalysisJobStep.created_at.asc())
        )
        return list(self.db.scalars(stmt).all())
