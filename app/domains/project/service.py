from sqlalchemy.orm import Session

from app.domains.project.model import Project
from app.domains.project.repository import ProjectRepository
from app.domains.project.schema import (
    ProjectCreate,
    ProjectRecentItem,
    ProjectUpdate,
)
from app.domains.match.repository import MatchRepository
from app.domains.artifact.repository import ArtifactRepository


class ProjectService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = ProjectRepository(db)
        self.match_repository = MatchRepository(db)
        self.artifact_repository = ArtifactRepository(db)

    def create_project(
        self,
        data: ProjectCreate,
        *,
        match_id: str,
        owner_id: str,
    ) -> Project:
        match = self.match_repository.get_by_id(match_id)
        if match is None:
            raise ValueError("Match not found")
        if match.owner_id != owner_id:
            raise ValueError("Match owner does not match project owner")
        project = self.repository.create(
            **data.model_dump(),
            match_id=match_id,
            owner_id=owner_id,
        )
        self.db.commit()
        self.db.refresh(project)
        return project

    def list_match_projects(self, match_id: str) -> list[Project]:
        return self.repository.list_by_match(match_id)

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
            thumbnail_url = None
            if project.thumbnail_artifact_id:
                thumbnail_url = (
                    f"/api/v1/artifacts/{project.thumbnail_artifact_id}/download"
                )

            result.append(
                ProjectRecentItem(
                    project_id=project.project_id,
                    match_id=project.match_id,
                    title=project.title,
                    status=project.status,
                    thumbnail_url=thumbnail_url,
                    duration_sec=project.match.duration_sec,
                    created_at=project.created_at,
                )
            )

        return result

    def update_project(self, project_id: str, data: ProjectUpdate) -> Project | None:
        project = self.repository.get_by_id(project_id)
        if project is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        thumbnail_artifact_id = update_data.get("thumbnail_artifact_id")
        if thumbnail_artifact_id is not None:
            artifact = self.artifact_repository.get_by_id(thumbnail_artifact_id)
            if (
                artifact is None
                or artifact.match_id != project.match_id
                or artifact.project_id != project.project_id
            ):
                raise ValueError(
                    "Thumbnail artifact does not belong to this project"
                )
        for key, value in update_data.items():
            setattr(project, key, value)

        self.db.commit()
        self.db.refresh(project)
        return project
