from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sqlalchemy.orm import Session

from app.ai.tasks.soccernet_feature_extraction.splitter import (
    FeatureSplitResult,
    split_merged_feature,
)
from app.domains.auth.model import User
from app.domains.match.model import Match
from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage
from app.storage.workspace import (
    get_match_raw_video_subdir,
    get_match_soccernet_features_subdir,
)


PRELOADED_METADATA_KEY = "kickclip_preloaded_source"
FEATURE_DIM = 512
FEATURE_FPS = 2.0
FEATURE_MIME_TYPE = "application/x-npy"
MaterializationMode = Literal["auto", "hardlink", "copy"]


@dataclass(frozen=True)
class PreloadedMatchSpec:
    key: str
    video_path: Path
    feature_path: Path
    home_team: str | None = None
    away_team: str | None = None
    competition: str | None = None
    season: str | None = None
    match_id: str | None = None


@dataclass(frozen=True)
class PreloadedMatchResult:
    match_id: str
    raw_video_asset_id: str
    feature_asset_ids: dict[str, str]
    video_path: Path
    merged_feature_path: Path
    half1_feature_path: Path
    half2_feature_path: Path
    video_materialization: str
    feature_materialization: str
    video_duration_sec: float
    feature_shape: tuple[int, int]
    half1_shape: tuple[int, int]
    half2_shape: tuple[int, int]
    feature_fps: float
    owner_id: str


class PreloadedMatchImporter:
    """Register a local MP4/NPY pair as an already-extracted uploaded match."""

    def __init__(self, db: Session, storage: LocalStorage | None = None) -> None:
        self.db = db
        self.storage = storage or LocalStorage()
        self.media_repository = MediaAssetRepository(db)

    def import_match(
        self,
        spec: PreloadedMatchSpec,
        *,
        owner_id: str,
        link_mode: MaterializationMode = "auto",
        compute_video_sha256: bool = True,
    ) -> PreloadedMatchResult:
        owner = self.db.get(User, owner_id)
        if owner is None:
            raise ValueError(f"Owner user does not exist: {owner_id}")
        if not owner.is_active:
            raise ValueError(f"Owner user is inactive: {owner_id}")

        video_source = spec.video_path.resolve()
        feature_source = spec.feature_path.resolve()
        self._validate_sources(video_source, feature_source)
        feature_shape, feature_dtype = self._inspect_feature(feature_source)

        match_id = spec.match_id or self._match_id(spec.key)
        match = self.db.get(Match, match_id)
        if match is not None and not (match.metadata_ or {}).get(
            PRELOADED_METADATA_KEY
        ):
            raise ValueError(
                "Refusing to overwrite a non-preloaded Match with the same ID: "
                f"{match_id}"
            )

        video_target = (
            self.storage.storage_root
            / get_match_raw_video_subdir(match_id)
            / video_source.name
        )
        feature_dir = (
            self.storage.storage_root
            / get_match_soccernet_features_subdir(match_id)
        )
        merged_target = feature_dir / "merged_feature.npy"
        half1_target = feature_dir / "half1_feature.npy"
        half2_target = feature_dir / "half2_feature.npy"

        video_materialization = self._materialize(
            source=video_source,
            target=video_target,
            mode=link_mode,
        )
        feature_materialization = self._materialize(
            source=feature_source,
            target=merged_target,
            mode=link_mode,
        )

        # The Champion contract is fixed at 2 fps. Using the feature-derived
        # duration makes a midpoint split land on the exact middle row even
        # when a container reports a few padding frames differently.
        feature_duration_sec = feature_shape[0] / FEATURE_FPS
        split_result = split_merged_feature(
            merged_path=merged_target,
            half1_path=half1_target,
            half2_path=half2_target,
            duration_sec=feature_duration_sec,
            split_strategy="midpoint",
            expected_dim=FEATURE_DIM,
            overwrite=True,
        )

        video_metadata = self._probe_video(video_target)
        video_duration_sec = float(
            video_metadata.get("duration_sec") or feature_duration_sec
        )
        if abs(video_duration_sec - feature_duration_sec) > 30.0:
            raise ValueError(
                "Video duration and 2-fps feature duration differ by more than "
                f"30 seconds: video={video_duration_sec:.3f}, "
                f"feature={feature_duration_sec:.3f}"
            )

        video_sha256 = (
            self._sha256(video_target) if compute_video_sha256 else None
        )
        marker = {
            "key": spec.key,
            "source_video_path": str(video_source),
            "source_feature_path": str(feature_source),
            "feature_shape": list(feature_shape),
            "feature_dtype": feature_dtype,
            "feature_fps": FEATURE_FPS,
            "split_strategy": "midpoint",
            "split_index": split_result.split_index,
            "video_materialization": video_materialization,
            "feature_materialization": feature_materialization,
        }

        try:
            if match is None:
                match = Match(
                    match_id=match_id,
                    owner_id=owner_id,
                    home_team=spec.home_team,
                    away_team=spec.away_team,
                    competition=spec.competition,
                    season=spec.season,
                    duration_sec=video_duration_sec,
                    metadata_={PRELOADED_METADATA_KEY: marker},
                )
                self.db.add(match)
                self.db.flush()
            else:
                match.owner_id = owner_id
                match.home_team = spec.home_team
                match.away_team = spec.away_team
                match.competition = spec.competition
                match.season = spec.season
                match.duration_sec = video_duration_sec
                match.metadata_ = {
                    **(match.metadata_ or {}),
                    PRELOADED_METADATA_KEY: marker,
                }

            raw_asset = self._upsert_asset(
                match_id=match_id,
                asset_type="RAW_VIDEO",
                file_path=video_target,
                original_filename=video_source.name,
                mime_type=mimetypes.guess_type(video_source.name)[0] or "video/mp4",
                duration_sec=video_duration_sec,
                fps=video_metadata.get("fps"),
                width=video_metadata.get("width"),
                height=video_metadata.get("height"),
                size_bytes=video_target.stat().st_size,
                sha256=video_sha256,
            )
            merged_asset = self._upsert_feature_asset(
                match_id=match_id,
                asset_type="SOCCERNET_FEATURE",
                path=merged_target,
                original_filename=feature_source.name,
                duration_sec=feature_duration_sec,
            )
            half1_asset = self._upsert_feature_asset(
                match_id=match_id,
                asset_type="SOCCERNET_FEATURE_HALF1",
                path=half1_target,
                original_filename=f"{feature_source.stem}_half1.npy",
                duration_sec=split_result.half1_shape[0] / FEATURE_FPS,
            )
            half2_asset = self._upsert_feature_asset(
                match_id=match_id,
                asset_type="SOCCERNET_FEATURE_HALF2",
                path=half2_target,
                original_filename=f"{feature_source.stem}_half2.npy",
                duration_sec=split_result.half2_shape[0] / FEATURE_FPS,
            )

            self.db.commit()
            self.db.refresh(match)
            self.db.refresh(raw_asset)
        except Exception:
            self.db.rollback()
            raise

        return PreloadedMatchResult(
            match_id=match.match_id,
            raw_video_asset_id=raw_asset.asset_id,
            feature_asset_ids={
                "SOCCERNET_FEATURE": merged_asset.asset_id,
                "SOCCERNET_FEATURE_HALF1": half1_asset.asset_id,
                "SOCCERNET_FEATURE_HALF2": half2_asset.asset_id,
            },
            video_path=video_target,
            merged_feature_path=merged_target,
            half1_feature_path=half1_target,
            half2_feature_path=half2_target,
            video_materialization=video_materialization,
            feature_materialization=feature_materialization,
            video_duration_sec=video_duration_sec,
            feature_shape=feature_shape,
            half1_shape=split_result.half1_shape,
            half2_shape=split_result.half2_shape,
            feature_fps=FEATURE_FPS,
            owner_id=owner_id,
        )

    def _upsert_feature_asset(
        self,
        *,
        match_id: str,
        asset_type: str,
        path: Path,
        original_filename: str,
        duration_sec: float,
    ) -> MediaAsset:
        return self._upsert_asset(
            match_id=match_id,
            asset_type=asset_type,
            file_path=path,
            original_filename=original_filename,
            mime_type=FEATURE_MIME_TYPE,
            duration_sec=duration_sec,
            fps=FEATURE_FPS,
            width=None,
            height=None,
            size_bytes=path.stat().st_size,
            sha256=None,
        )

    def _upsert_asset(
        self,
        *,
        match_id: str,
        asset_type: str,
        file_path: Path,
        original_filename: str,
        mime_type: str,
        duration_sec: float,
        fps: float | None,
        width: int | None,
        height: int | None,
        size_bytes: int,
        sha256: str | None,
    ) -> MediaAsset:
        existing = next(
            (
                asset
                for asset in self.media_repository.list_by_match(match_id)
                if asset.asset_type == asset_type
            ),
            None,
        )
        relative_path = self._to_project_relative(file_path)
        if existing is None:
            existing = MediaAsset(
                asset_id=self._asset_id(match_id, asset_type),
                match_id=match_id,
                asset_type=asset_type,
                file_path=relative_path,
            )
            self.db.add(existing)

        existing.file_path = relative_path
        existing.original_filename = original_filename
        existing.mime_type = mime_type
        existing.duration_sec = duration_sec
        existing.fps = fps
        existing.width = width
        existing.height = height
        existing.size_bytes = size_bytes
        existing.sha256 = sha256
        self.db.flush()
        return existing

    @staticmethod
    def _validate_sources(video_path: Path, feature_path: Path) -> None:
        if not video_path.is_file():
            raise FileNotFoundError(f"Preloaded video does not exist: {video_path}")
        if video_path.suffix.lower() != ".mp4":
            raise ValueError(f"Preloaded video must be an MP4 file: {video_path}")
        if not feature_path.is_file():
            raise FileNotFoundError(f"Preloaded feature does not exist: {feature_path}")
        if feature_path.suffix.lower() != ".npy":
            raise ValueError(f"Preloaded feature must be an NPY file: {feature_path}")

    @staticmethod
    def _inspect_feature(path: Path) -> tuple[tuple[int, int], str]:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not read preloaded feature: {path}") from exc
        shape = tuple(int(value) for value in array.shape)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Preloaded feature must have shape [T, {FEATURE_DIM}], got {shape}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Preloaded feature must be numeric, got {array.dtype}")
        return (shape[0], shape[1]), str(array.dtype)

    @staticmethod
    def _probe_video(path: Path) -> dict:
        metadata = extract_video_metadata(path)
        if metadata.get("duration_sec"):
            return metadata
        try:
            import cv2  # type: ignore
        except Exception:
            return metadata

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return metadata
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            return {
                **metadata,
                "duration_sec": frames / fps if fps > 0 and frames > 0 else None,
                "fps": fps or None,
                "width": width or None,
                "height": height or None,
            }
        finally:
            capture.release()

    @staticmethod
    def _materialize(
        *,
        source: Path,
        target: Path,
        mode: MaterializationMode,
    ) -> str:
        if mode not in {"auto", "hardlink", "copy"}:
            raise ValueError(f"Unsupported materialization mode: {mode}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source == target.resolve(strict=False):
            return "source_in_storage"
        if target.exists():
            try:
                if os.path.samefile(source, target):
                    return "existing_hardlink"
            except OSError:
                pass
            target.unlink()
        if mode in {"auto", "hardlink"}:
            try:
                os.link(source, target)
                return "hardlink"
            except OSError:
                if mode == "hardlink":
                    raise
        shutil.copy2(source, target)
        return "copy"

    def _to_project_relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.storage.storage_root):
            raise ValueError(f"Preloaded target escapes STORAGE_ROOT: {resolved}")
        return resolved.relative_to(self.storage.project_root).as_posix()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _match_id(key: str) -> str:
        safe_key = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in key.lower()
        ).strip("_")
        candidate = f"match_preloaded_{safe_key}"
        if len(candidate) <= 64:
            return candidate
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"match_preloaded_{digest}"

    @staticmethod
    def _asset_id(match_id: str, asset_type: str) -> str:
        digest = hashlib.sha256(
            f"preloaded:{match_id}:{asset_type}".encode("utf-8")
        ).hexdigest()[:24]
        return f"asset_preloaded_{digest}"
