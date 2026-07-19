from sqlalchemy.orm import Session

from app.domains.project.model import Project
from app.domains.project.repository import ProjectRepository
from app.domains.project.schema import (
    ProjectCreate,
    ProjectRecentItem,
    ProjectUpdate,
)


class ProjectService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = ProjectRepository(db)

    def create_project(
        self,
        data: ProjectCreate,
        *,
        owner_id: str | None = None,
    ) -> Project:
        project = self.repository.create(
            **data.model_dump(),
            owner_id=owner_id,
        )
        self.db.commit()
        self.db.refresh(project)
        return project

    def get_project(self, project_id: str) -> Project | None:
        return self.repository.get_by_id(project_id)

    def list_recent_projects(
        self,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> list[Project]:
        return self.repository.list_recent(limit=limit, owner_id=owner_id)

    def list_recent_project_cards(
        self,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> list[ProjectRecentItem]:
        projects = self.repository.list_recent(limit=limit, owner_id=owner_id)

        result: list[ProjectRecentItem] = []

        for project in projects:
            first_match = project.matches[0] if project.matches else None

            thumbnail_url = None
            if project.thumbnail_artifact_id:
                thumbnail_url = (
                    f"/api/v1/artifacts/{project.thumbnail_artifact_id}/download"
                )

            result.append(
                ProjectRecentItem(
                    project_id=project.project_id,
                    title=project.title,
                    status=project.status,
                    thumbnail_url=thumbnail_url,
                    duration_sec=first_match.duration_sec if first_match else None,
                    created_at=project.created_at,
                )
            )

        return result

    def update_project(self, project_id: str, data: ProjectUpdate) -> Project | None:
        project = self.repository.get_by_id(project_id)
        if project is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(project, key, value)

        self.db.commit()
        self.db.refresh(project)
        return project
