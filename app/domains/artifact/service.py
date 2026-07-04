from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.artifact.schema import ArtifactCreate


class ArtifactService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = ArtifactRepository(db)

    def create_artifact(self, data: ArtifactCreate) -> Artifact:
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