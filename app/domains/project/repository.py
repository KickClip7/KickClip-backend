from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.domains.clip_plan.model import ClipPlan
from app.domains.match.model import Match
from app.domains.project.model import Project
from app.domains.render.model import RenderJob


class ProjectRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Project:
        project = Project(**kwargs)
        self.db.add(project)
        self.db.flush()
        return project

    def get_by_id(self, project_id: str) -> Project | None:
        stmt = (
            select(Project)
            .execution_options(populate_existing=True)
            .options(*self._summary_load_options())
            .where(Project.project_id == project_id)
        )
        return self.db.scalar(stmt)

    def list_by_match(self, match_id: str) -> list[Project]:
        stmt = (
            select(Project)
            .execution_options(populate_existing=True)
            .options(*self._summary_load_options())
            .where(Project.match_id == match_id)
            .order_by(Project.updated_at.desc(), Project.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def list_recent(
        self,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> list[Project]:
        stmt = (
            select(Project)
            .execution_options(populate_existing=True)
            .options(*self._summary_load_options())
        )
        if owner_id is not None:
            stmt = stmt.where(Project.match.has(Match.owner_id == owner_id))
        stmt = stmt.order_by(
            func.coalesce(Project.last_opened_at, Project.updated_at).desc(),
            Project.updated_at.desc(),
        ).limit(limit)
        return list(self.db.scalars(stmt).all())

    @staticmethod
    def _summary_load_options():
        return (
            selectinload(Project.match),
            selectinload(Project.artifacts),
            selectinload(Project.clip_plans).selectinload(ClipPlan.items),
            selectinload(Project.clip_plans)
            .selectinload(ClipPlan.render_jobs)
            .selectinload(RenderJob.output_artifact),
        )
