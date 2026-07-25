from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.artifact.schema import ArtifactCreate
from app.domains.project.repository import ProjectRepository


class ArtifactService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = ArtifactRepository(db)
        self.project_repository = ProjectRepository(db)

    def create_artifact(self, data: ArtifactCreate) -> Artifact:
        if data.project_id is not None:
            project = self.project_repository.get_by_id(data.project_id)
            if project is None or project.match_id != data.match_id:
                raise ValueError("Artifact project does not belong to match")
        if data.analysis_job_id is not None and data.project_id is not None:
            raise ValueError("Analysis artifacts must be match-scoped")
        payload = data.model_dump()
        payload["metadata_"] = payload.pop("metadata", {})

        artifact = self.repository.create(**payload)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        return self.repository.get_by_id(artifact_id)

    def list_match_artifacts(self, match_id: str) -> list[Artifact]:
        return self.repository.list_by_match(match_id)

    def list_analysis_job_artifacts(self, analysis_job_id: str) -> list[Artifact]:
        return self.repository.list_by_analysis_job(analysis_job_id)
