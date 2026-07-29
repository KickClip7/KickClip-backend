from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_artifact_access
from app.core.config import get_settings
from app.domains.auth.dependencies import get_optional_current_user
from app.domains.auth.model import User
from app.domains.auth.repository import UserRepository
from app.domains.auth.security import InvalidTokenError, decode_artifact_token
from app.storage.local_storage import LocalStorage


router = APIRouter()


@router.get("/{artifact_id}/download")
def download_artifact(
    artifact_id: str,
    token: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
) -> FileResponse:
    artifact = _authorize_artifact(
        db=db,
        artifact_id=artifact_id,
        current_user=current_user,
        token=token,
    )
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


def _authorize_artifact(
    *,
    db: Session,
    artifact_id: str,
    current_user: User | None,
    token: str | None,
):
    if current_user is not None:
        return require_artifact_access(db, artifact_id, current_user)
    if token:
        try:
            payload = decode_artifact_token(
                token,
                artifact_id=artifact_id,
                secret_key=get_settings().AUTH_SECRET_KEY,
            )
        except InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired artifact token",
            ) from exc
        user = UserRepository(db).get_by_id(str(payload.get("sub") or ""))
        if user is not None and user.is_active:
            return require_artifact_access(db, artifact_id, user)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
