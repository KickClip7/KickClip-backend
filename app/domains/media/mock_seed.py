from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy.orm import Session

from app.core.paths import get_project_root, get_storage_root
from app.domains.match.mock import MOCK_MATCH_METADATA_KEY, is_mock_match
from app.domains.match.repository import MatchRepository
from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.media.model import MediaAsset
from app.storage.workspace import get_match_raw_video_subdir


MockVideoLinkMode = Literal["auto", "hardlink", "copy"]


@dataclass(frozen=True)
class MockVideoSeedResult:
    asset_id: str
    source_path: Path
    stored_path: Path
    materialization: str
    duration_sec: float
    fps: float | None
    width: int | None
    height: int | None


def seed_mock_video_asset(
    db: Session,
    *,
    match_id: str,
    source_path: str | Path,
    required_duration_sec: float | None = None,
    link_mode: MockVideoLinkMode = "auto",
) -> MockVideoSeedResult:
    """Register a teammate-local video as the mock match RAW_VIDEO.

    The original file is never committed or referenced from outside STORAGE_ROOT.
    A deterministic hard link (preferred) or copy is created under the match
    workspace and a deterministic MediaAsset row is inserted or refreshed.
    """

    match = MatchRepository(db).get_by_id(match_id)
    if match is None:
        raise ValueError(
            f"목업 Match가 DB에 없습니다: {match_id}. timeline 시드를 먼저 실행하세요."
        )
    if not is_mock_match(match):
        raise ValueError(
            "실제 Match에는 목업 영상을 연결할 수 없습니다: "
            f"match_id={match_id}"
        )

    source = _resolve_source_path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"목업 원본 영상이 없습니다: {source}")

    target_dir = (
        get_storage_root().resolve() / get_match_raw_video_subdir(match_id)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"mock_{source.name}"
    materialization = _materialize_video(
        source=source,
        target=target,
        link_mode=link_mode,
    )

    metadata = extract_video_metadata(target)
    duration_sec = metadata.get("duration_sec")
    if duration_sec is None or float(duration_sec) <= 0:
        raise RuntimeError(
            "영상 길이를 확인할 수 없습니다. ffprobe가 PATH에 있는지 확인하세요: "
            f"{target}"
        )
    duration_sec = float(duration_sec)

    if (
        required_duration_sec is not None
        and duration_sec + 0.5 < float(required_duration_sec)
    ):
        raise ValueError(
            "목업 영상이 timeline event 범위를 모두 포함하지 않습니다. "
            f"video_duration={duration_sec:.3f}, "
            f"required_event_end={float(required_duration_sec):.3f}"
        )

    asset_id = _mock_video_asset_id(match_id)
    asset = db.get(MediaAsset, asset_id)
    if asset is not None and asset.match_id != match_id:
        raise ValueError(
            "동일한 deterministic asset_id가 다른 Match에 사용 중입니다: "
            f"{asset_id}"
        )

    file_path = _to_storage_file_path(target)
    mime_type = mimetypes.guess_type(source.name)[0] or "video/mp4"

    try:
        if asset is None:
            asset = MediaAsset(
                asset_id=asset_id,
                match_id=match_id,
                asset_type="RAW_VIDEO",
                file_path=file_path,
                original_filename=source.name,
                mime_type=mime_type,
            )
            db.add(asset)

        asset.asset_type = "RAW_VIDEO"
        asset.file_path = file_path
        asset.original_filename = source.name
        asset.mime_type = mime_type
        asset.duration_sec = duration_sec
        asset.fps = metadata.get("fps")
        asset.width = metadata.get("width")
        asset.height = metadata.get("height")
        asset.size_bytes = metadata.get("size_bytes") or target.stat().st_size

        match.duration_sec = duration_sec
        marker = dict((match.metadata_ or {}).get(MOCK_MATCH_METADATA_KEY) or {})
        marker.update(
            {
                "video_source_path": str(source),
                "video_asset_id": asset_id,
                "video_duration_sec": duration_sec,
            }
        )
        match.metadata_ = {
            **(match.metadata_ or {}),
            MOCK_MATCH_METADATA_KEY: marker,
        }

        db.commit()
        db.refresh(asset)
        db.refresh(match)
    except Exception:
        db.rollback()
        raise

    return MockVideoSeedResult(
        asset_id=asset.asset_id,
        source_path=source,
        stored_path=target,
        materialization=materialization,
        duration_sec=duration_sec,
        fps=asset.fps,
        width=asset.width,
        height=asset.height,
    )


def _resolve_source_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = get_project_root() / path
    return path.resolve()


def _materialize_video(
    *,
    source: Path,
    target: Path,
    link_mode: MockVideoLinkMode,
) -> str:
    if link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"지원하지 않는 MOCK_VIDEO_LINK_MODE입니다: {link_mode}")

    if source == target.resolve(strict=False):
        return "source_in_storage"

    if target.exists():
        try:
            if os.path.samefile(source, target):
                return "existing_hardlink"
        except OSError:
            pass
        target.unlink()

    if link_mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise

    shutil.copy2(source, target)
    return "copy"


def _to_storage_file_path(target: Path) -> str:
    project_root = get_project_root().resolve()
    resolved = target.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        # LocalStorage accepts absolute paths only when they remain inside
        # the configured STORAGE_ROOT.
        return str(resolved)


def _mock_video_asset_id(match_id: str) -> str:
    digest = hashlib.sha256(f"mock-video:{match_id}".encode("utf-8")).hexdigest()[:24]
    return f"asset_mock_{digest}"
