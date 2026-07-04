from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact


class ArtifactRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Artifact:
        artifact = Artifact(**kwargs)
        self.db.add(artifact)
        self.db.flush()
        return artifact

    def get_by_id(self, artifact_id: str) -> Artifact | None:
        stmt = select(Artifact).where(Artifact.artifact_id == artifact_id)
        return self.db.scalar(stmt)

    def list_by_match(self, match_id: str) -> list[Artifact]:
        stmt = (
            select(Artifact)
            .where(Artifact.match_id == match_id)
            .order_by(Artifact.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def list_by_analysis_job(self, analysis_job_id: str) -> list[Artifact]:
        stmt = (
            select(Artifact)
            .where(Artifact.analysis_job_id == analysis_job_id)
            .order_by(Artifact.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())