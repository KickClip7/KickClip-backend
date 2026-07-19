from __future__ import annotations

import secrets
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.auth.model import User
from app.domains.auth.repository import RefreshTokenRepository, UserRepository
from app.domains.auth.schema import LoginRequest, SignUpRequest, TokenResponse, UserRead
from app.domains.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_expiry,
    verify_password,
)


class AuthenticationError(ValueError):
    pass


class AuthService:
    def __init__(self, db: Session, settings: Settings | None = None):
        self.db = db
        self.settings = settings or get_settings()
        self.users = UserRepository(db)
        self.refresh_tokens = RefreshTokenRepository(db)

    def signup(self, data: SignUpRequest) -> TokenResponse:
        if self.users.get_by_email(data.email) is not None:
            raise AuthenticationError("Email is already registered")
        try:
            user = self.users.create(
                email=data.email,
                password_hash=hash_password(data.password),
                display_name=data.display_name.strip(),
                role="USER",
                is_active=True,
                developer_mode_enabled=False,
            )
            response = self._issue_token_pair(user)
            self.db.commit()
            self.db.refresh(user)
            response.user = UserRead.model_validate(user)
            return response
        except IntegrityError as exc:
            self.db.rollback()
            raise AuthenticationError("Email is already registered") from exc

    def login(self, data: LoginRequest) -> TokenResponse:
        user = self.users.get_by_email(data.email)
        if user is None or not verify_password(data.password, user.password_hash):
            raise AuthenticationError("Invalid email or password")
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        user.last_login_at = datetime.now(timezone.utc)
        response = self._issue_token_pair(user)
        self.db.commit()
        self.db.refresh(user)
        response.user = UserRead.model_validate(user)
        return response

    def refresh(self, plain_token: str) -> TokenResponse:
        stored = self.refresh_tokens.get_by_hash(
            hash_refresh_token(plain_token),
            for_update=True,
        )
        now = datetime.now(timezone.utc)
        if (
            stored is None
            or stored.revoked_at is not None
            or _as_utc(stored.expires_at) <= now
        ):
            raise AuthenticationError("Invalid or expired refresh token")
        user = self.users.get_by_id(stored.user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("User account is unavailable")

        stored.revoked_at = now
        response, replacement_id = self._issue_token_pair(user, include_id=True)
        stored.replaced_by_token_id = replacement_id
        self.db.commit()
        self.db.refresh(user)
        response.user = UserRead.model_validate(user)
        return response

    def logout(self, plain_token: str) -> None:
        stored = self.refresh_tokens.get_by_hash(
            hash_refresh_token(plain_token),
            for_update=True,
        )
        if stored is not None and stored.revoked_at is None:
            stored.revoked_at = datetime.now(timezone.utc)
            self.db.commit()

    def set_developer_mode(
        self,
        *,
        user: User,
        enabled: bool,
        access_key: str | None,
    ) -> User:
        if enabled:
            configured_key = self.settings.DEVELOPER_ACCESS_KEY
            if not self.settings.DEVELOPER_MODE_ENABLED or not configured_key:
                raise AuthenticationError("Developer mode is disabled on this server")
            if access_key is None or not secrets.compare_digest(access_key, configured_key):
                raise AuthenticationError("Invalid developer access key")
            user.role = "DEVELOPER"
        user.developer_mode_enabled = enabled
        self.db.commit()
        self.db.refresh(user)
        return user

    def _issue_token_pair(
        self,
        user: User,
        *,
        include_id: bool = False,
    ) -> TokenResponse | tuple[TokenResponse, str]:
        self._validate_secret()
        access_token, expires_in = create_access_token(
            user_id=user.user_id,
            secret_key=self.settings.AUTH_SECRET_KEY,
            expires_minutes=self.settings.AUTH_ACCESS_TOKEN_MINUTES,
        )
        plain_refresh = create_refresh_token()
        stored = self.refresh_tokens.create(
            user_id=user.user_id,
            token_hash=hash_refresh_token(plain_refresh),
            expires_at=refresh_expiry(self.settings.AUTH_REFRESH_TOKEN_DAYS),
        )
        response = TokenResponse(
            access_token=access_token,
            refresh_token=plain_refresh,
            expires_in=expires_in,
            user=UserRead.model_validate(user),
        )
        return (response, stored.refresh_token_id) if include_id else response

    def _validate_secret(self) -> None:
        secret = self.settings.AUTH_SECRET_KEY
        if len(secret) < 32:
            raise RuntimeError("AUTH_SECRET_KEY must contain at least 32 characters")
        if self.settings.ENV.lower() in {"prod", "production"} and secret.startswith("local-"):
            raise RuntimeError("Set a production AUTH_SECRET_KEY before starting the service")


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
