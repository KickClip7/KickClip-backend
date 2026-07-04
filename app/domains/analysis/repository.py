from sqlalchemy import select
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