from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.auth.model import RefreshToken, User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> User:
        user = User(**kwargs)
        self.db.add(user)
        self.db.flush()
        return user

    def get_by_id(self, user_id: str) -> User | None:
        return self.db.scalar(select(User).where(User.user_id == user_id))

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(User.email == email))


class RefreshTokenRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> RefreshToken:
        token = RefreshToken(**kwargs)
        self.db.add(token)
        self.db.flush()
        return token

    def get_by_hash(
        self,
        token_hash: str,
        *,
        for_update: bool = False,
    ) -> RefreshToken | None:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        if for_update:
            stmt = stmt.with_for_update()
        return self.db.scalar(stmt)
