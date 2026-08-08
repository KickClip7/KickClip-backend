from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.domains.auth.model import User
from app.domains.auth.repository import UserRepository
from app.domains.auth.security import InvalidTokenError, decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)


def authenticate_current_user(
    credentials: HTTPAuthorizationCredentials | None,
    db: Session,
) -> User:
    """Authenticate one bearer token using an already-scoped DB session.

    Media endpoints use this helper with an explicit short-lived SessionLocal
    context so the SQLAlchemy connection can be returned before FileResponse
    starts streaming a potentially large image/video body.
    """

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized
    try:
        payload = decode_access_token(
            credentials.credentials,
            get_settings().AUTH_SECRET_KEY,
        )
    except InvalidTokenError as exc:
        raise unauthorized from exc
    user = UserRepository(db).get_by_id(str(payload["sub"]))
    if user is None or not user.is_active:
        raise unauthorized
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    return authenticate_current_user(credentials, db)


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    if credentials is None:
        return None
    return get_current_user(credentials=credentials, db=db)
