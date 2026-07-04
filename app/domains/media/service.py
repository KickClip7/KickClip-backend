from sqlalchemy.orm import Session

from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.media.schema import MediaAssetCreate


class MediaAssetService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = MediaAssetRepository(db)

    def create_media_asset(self, data: MediaAssetCreate) -> MediaAsset:
        asset = self.repository.create(**data.model_dump())
        self.db.commit()
        self.db.refresh(asset)
        return asset

    def get_media_asset(self, asset_id: str) -> MediaAsset | None:
        return self.repository.get_by_id(asset_id)

    def list_match_media_assets(self, match_id: str) -> list[MediaAsset]:
        return self.repository.list_by_match(match_id)