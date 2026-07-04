from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage


RAW_VIDEO_ASSET_TYPE = "RAW_VIDEO"


@dataclass(frozen=True)
class RawVideoAssetRef:
    """Resolved RAW_VIDEO MediaAsset for SoccerNet feature extraction."""

    asset: MediaAsset
    resolved_path: Path

    @property
    def asset_id(self) -> str:
        return self.asset.asset_id

    @property
    def match_id(self) -> str:
        return self.asset.match_id

    @property
    def file_path(self) -> str:
        return self.asset.file_path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset.asset_id,
            "asset_type": self.asset.asset_type,
            "match_id": self.asset.match_id,
            "file_path": self.asset.file_path,
            "resolved_path": self.resolved_path.as_posix(),
            "original_filename": self.asset.original_filename,
            "mime_type": self.asset.mime_type,
            "duration_sec": self.asset.duration_sec,
            "fps": self.asset.fps,
            "width": self.asset.width,
            "height": self.asset.height,
            "size_bytes": self.asset.size_bytes,
        }


class RawVideoResolveError(RuntimeError):
    """Raised when a RAW_VIDEO asset cannot be resolved to a local file."""


class RawVideoResolver:
    """Resolve the uploaded RAW_VIDEO asset for a match.

    Feature extraction must start from the persisted MediaAsset, not directly from
    an upload request. This keeps the analysis job reusable and debuggable.
    """

    def __init__(self, db: Session, storage: LocalStorage | None = None) -> None:
        self.db = db
        self.storage = storage or LocalStorage()
        self.media_repository = MediaAssetRepository(db)

    def resolve_for_match(self, match_id: str) -> RawVideoAssetRef:
        asset = self.find_raw_video_asset(match_id)
        if asset is None:
            raise RawVideoResolveError(
                f"No {RAW_VIDEO_ASSET_TYPE} MediaAsset found for match_id={match_id}."
            )

        resolved_path = self.storage.resolve_path(asset.file_path)
        if not resolved_path.exists():
            raise RawVideoResolveError(
                "RAW_VIDEO file does not exist on disk: "
                f"asset_id={asset.asset_id}, file_path={asset.file_path}, "
                f"resolved_path={resolved_path.as_posix()}"
            )

        if not resolved_path.is_file():
            raise RawVideoResolveError(
                "RAW_VIDEO path is not a file: "
                f"asset_id={asset.asset_id}, resolved_path={resolved_path.as_posix()}"
            )

        return RawVideoAssetRef(asset=asset, resolved_path=resolved_path)

    def find_raw_video_asset(self, match_id: str) -> MediaAsset | None:
        """Return the latest RAW_VIDEO asset for the match.

        MediaAssetRepository.list_by_match already orders by created_at desc, so
        this naturally selects the most recent uploaded raw video when multiple
        rows exist.
        """

        assets = self.media_repository.list_by_match(match_id)
        return next(
            (asset for asset in assets if asset.asset_type == RAW_VIDEO_ASSET_TYPE),
            None,
        )
