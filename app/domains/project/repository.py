from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.domains.match.model import Match
from app.domains.project.model import Project


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
            .options(selectinload(Project.match))
            .where(Project.project_id == project_id)
        )
        return self.db.scalar(stmt)

    def list_by_match(self, match_id: str) -> list[Project]:
        stmt = (
            select(Project)
            .where(Project.match_id == match_id)
            .order_by(Project.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def list_recent(
        self,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> list[Project]:
        stmt = (
            select(Project)
            .options(selectinload(Project.match))
        )
        if owner_id is not None:
            stmt = stmt.where(Project.match.has(Match.owner_id == owner_id))
        stmt = stmt.order_by(Project.updated_at.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())
