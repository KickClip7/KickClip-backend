from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.domains.match.model import Match


class MatchRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Match:
        match = Match(**kwargs)
        self.db.add(match)
        self.db.flush()
        return match

    def get_by_id(self, match_id: str) -> Match | None:
        stmt = (
            select(Match)
            .options(
                selectinload(Match.media_assets),
                selectinload(Match.artifacts),
            )
            .where(Match.match_id == match_id)
        )
        return self.db.scalar(stmt)

    def get_for_update(self, match_id: str) -> Match | None:
        return self.db.scalar(
            select(Match)
            .where(Match.match_id == match_id)
            .with_for_update()
        )

    def list_by_owner(self, owner_id: str) -> list[Match]:
        stmt = (
            select(Match)
            .options(
                selectinload(Match.media_assets),
                selectinload(Match.artifacts),
            )
            .where(Match.owner_id == owner_id)
            .order_by(Match.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def list_all(self) -> list[Match]:
        stmt = (
            select(Match)
            .options(
                selectinload(Match.media_assets),
                selectinload(Match.artifacts),
            )
            .order_by(Match.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())
