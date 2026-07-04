from sqlalchemy.orm import Session

from app.domains.match.model import Match
from app.domains.match.repository import MatchRepository
from app.domains.match.schema import MatchCreate, MatchCreateRequest, MatchUpdate
from app.domains.project.repository import ProjectRepository


class MatchService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = MatchRepository(db)
        self.project_repository = ProjectRepository(db)

    def create_match(self, data: MatchCreate) -> Match:
        payload = data.model_dump()
        payload["metadata_"] = payload.pop("metadata", {})
        match = self.repository.create(**payload)

        self.db.commit()
        self.db.refresh(match)
        return match

    def create_match_for_project(
        self,
        project_id: str,
        data: MatchCreateRequest,
    ) -> Match:
        project = self.project_repository.get_by_id(project_id)
        if project is None:
            raise ValueError("Project not found")

        payload = data.model_dump()
        payload["project_id"] = project_id
        payload["metadata_"] = payload.pop("metadata", {})

        match = self.repository.create(**payload)

        self.db.commit()
        self.db.refresh(match)
        return match

    def get_match(self, match_id: str) -> Match | None:
        return self.repository.get_by_id(match_id)

    def list_project_matches(self, project_id: str) -> list[Match]:
        return self.repository.list_by_project(project_id)

    def update_match(self, match_id: str, data: MatchUpdate) -> Match | None:
        match = self.repository.get_by_id(match_id)
        if match is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        if "metadata" in update_data:
            update_data["metadata_"] = update_data.pop("metadata")

        for key, value in update_data.items():
            setattr(match, key, value)

        self.db.commit()
        self.db.refresh(match)
        return match