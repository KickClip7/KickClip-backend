from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_artifact_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.storage.local_storage import LocalStorage


router = APIRouter()


@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    artifact = require_artifact_access(db, artifact_id, current_user)
    try:
        path = LocalStorage().resolve_path(artifact.file_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found",
        ) from exc
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found",
        )
    return FileResponse(
        path=path,
        media_type=artifact.mime_type or "application/octet-stream",
        filename=Path(path).name,
    )
