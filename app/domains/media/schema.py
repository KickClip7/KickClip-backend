from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MediaAssetCreate(BaseModel):
    match_id: str
    asset_type: str
    file_path: str
    original_filename: str | None = None
    mime_type: str | None = None
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None


class MediaAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    match_id: str
    asset_type: str
    file_path: str
    original_filename: str | None
    mime_type: str | None
    duration_sec: float | None
    fps: float | None
    width: int | None
    height: int | None
    size_bytes: int | None
    created_at: datetime
    updated_at: datetime


class MediaAssetResponse(MediaAssetRead):
    stream_url: str
    download_url: str


class SignedMediaUrlResponse(BaseModel):
    asset_id: str
    url: str
    expires_in: int
