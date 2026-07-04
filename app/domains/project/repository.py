from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

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
        stmt = select(Project).where(Project.project_id == project_id)
        return self.db.scalar(stmt)

    def list_recent(self, limit: int = 10) -> list[Project]:
        stmt = (
            select(Project)
            .options(selectinload(Project.matches))
            .order_by(Project.updated_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt).all())