from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.media.model import MediaAsset


class MediaAssetRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> MediaAsset:
        asset = MediaAsset(**kwargs)
        self.db.add(asset)
        self.db.flush()
        return asset

    def get_by_id(self, asset_id: str) -> MediaAsset | None:
        stmt = select(MediaAsset).where(MediaAsset.asset_id == asset_id)
        return self.db.scalar(stmt)

    def get_for_update(self, asset_id: str) -> MediaAsset | None:
        return self.db.scalar(
            select(MediaAsset)
            .where(MediaAsset.asset_id == asset_id)
            .with_for_update()
        )

    def list_by_match(self, match_id: str) -> list[MediaAsset]:
        stmt = (
            select(MediaAsset)
            .where(MediaAsset.match_id == match_id)
            .order_by(MediaAsset.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())
