from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage


COMBINED_FEATURE_ASSET_TYPE = "SOCCERNET_FEATURE"
DEFAULT_HALF_FEATURE_ASSET_TYPES = [
    "SOCCERNET_FEATURE_HALF1",
    "SOCCERNET_FEATURE_HALF2",
]


@dataclass(frozen=True)
class HighlightFeatureArrayInfo:
    """Lightweight metadata read from a .npy feature file.

    The feature array itself is not loaded into memory when mmap_mode='r' is
    used. This is intended for diagnostics/readiness checks before real model
    inference is implemented.
    """

    asset_id: str
    asset_type: str
    path: Path
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int | None = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def num_rows(self) -> int | None:
        return int(self.shape[0]) if self.ndim >= 1 else None

    @property
    def feature_dim(self) -> int | None:
        return int(self.shape[1]) if self.ndim >= 2 else None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "path": self.path.as_posix(),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "ndim": self.ndim,
            "num_rows": self.num_rows,
            "feature_dim": self.feature_dim,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class HighlightFeatureBundle:
    """Feature assets required by a SoccerNet-feature-based highlight model."""

    assets: list[MediaAsset]
    resolved_paths: list[Path]
    missing_asset_types: list[str] = field(default_factory=list)
    feature_infos: list[HighlightFeatureArrayInfo] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    layout: str = "missing"  # combined | halves | partial | missing

    @property
    def available(self) -> bool:
        return (
            bool(self.assets)
            and not self.missing_asset_types
            and not self.validation_errors
            and len(self.assets) == len(self.feature_infos)
        )

    @property
    def asset_types(self) -> list[str]:
        return [asset.asset_type for asset in self.assets]

    @property
    def feature_shapes(self) -> list[list[int]]:
        return [list(info.shape) for info in self.feature_infos]

    @property
    def feature_dtypes(self) -> list[str]:
        return [info.dtype for info in self.feature_infos]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "layout": self.layout,
            "asset_ids": [asset.asset_id for asset in self.assets],
            "asset_types": self.asset_types,
            "paths": [path.as_posix() for path in self.resolved_paths],
            "missing_asset_types": self.missing_asset_types,
            "validation_errors": self.validation_errors,
            "feature_shapes": self.feature_shapes,
            "feature_dtypes": self.feature_dtypes,
            "feature_infos": [info.to_metadata() for info in self.feature_infos],
        }


class HighlightFeatureLoader:
    """Find and validate SoccerNet feature MediaAsset rows for highlight inference.

    Current champion experiments are SoccerNet-feature based, so the real
    predictor needs .npy features rather than RAW_VIDEO. This loader now checks
    both DB rows and actual .npy readability.
    """

    def __init__(self, db: Session):
        self.db = db
        self.storage = LocalStorage()
        self.media_repository = MediaAssetRepository(db)

    def load_for_match(
        self,
        *,
        match_id: str,
        expected_asset_types: list[str],
        validate: bool = True,
        expected_feature_dim: int = 512,
    ) -> HighlightFeatureBundle:
        assets = self.media_repository.list_by_match(match_id)
        assets_by_type = self._latest_assets_by_type(assets)

        combined_errors: list[str] = []

        # Prefer a single combined feature when it is present and valid.
        combined = assets_by_type.get(COMBINED_FEATURE_ASSET_TYPE)
        if combined is not None:
            combined_bundle = self._bundle(
                [combined],
                missing_asset_types=[],
                layout="combined",
                validate=validate,
                expected_feature_dim=expected_feature_dim,
            )
            if combined_bundle.available:
                return combined_bundle
            combined_errors.extend(combined_bundle.validation_errors)

        # Fall back to half1/half2 assets. This is important when an old or
        # corrupted combined asset remains in DB but freshly generated half
        # assets are valid.
        required_half_types = self._required_half_asset_types(expected_asset_types)
        selected_assets: list[MediaAsset] = []
        missing_asset_types: list[str] = []

        for asset_type in required_half_types:
            asset = assets_by_type.get(asset_type)
            if asset is None:
                missing_asset_types.append(asset_type)
            else:
                selected_assets.append(asset)

        layout = "halves" if len(selected_assets) == len(required_half_types) else "partial"
        if not selected_assets:
            layout = "missing"

        half_bundle = self._bundle(
            selected_assets,
            missing_asset_types=missing_asset_types,
            layout=layout,
            validate=validate,
            expected_feature_dim=expected_feature_dim,
            extra_validation_errors=combined_errors,
        )
        return half_bundle

    def _bundle(
        self,
        assets: list[MediaAsset],
        missing_asset_types: list[str],
        *,
        layout: str,
        validate: bool,
        expected_feature_dim: int,
        extra_validation_errors: list[str] | None = None,
    ) -> HighlightFeatureBundle:
        resolved_paths = [self.storage.resolve_path(asset.file_path) for asset in assets]
        feature_infos: list[HighlightFeatureArrayInfo] = []
        validation_errors: list[str] = list(extra_validation_errors or [])

        if validate:
            for asset, path in zip(assets, resolved_paths):
                info, error = self._inspect_feature_array(
                    asset=asset,
                    path=path,
                    expected_feature_dim=expected_feature_dim,
                )
                if info is not None:
                    feature_infos.append(info)
                if error is not None:
                    validation_errors.append(error)

        return HighlightFeatureBundle(
            assets=assets,
            resolved_paths=resolved_paths,
            missing_asset_types=missing_asset_types,
            feature_infos=feature_infos,
            validation_errors=validation_errors,
            layout=layout,
        )

    def _inspect_feature_array(
        self,
        *,
        asset: MediaAsset,
        path: Path,
        expected_feature_dim: int,
    ) -> tuple[HighlightFeatureArrayInfo | None, str | None]:
        if not path.exists():
            return None, (
                f"feature file does not exist: asset_type={asset.asset_type}, "
                f"asset_id={asset.asset_id}, path={path.as_posix()}"
            )

        if not path.is_file():
            return None, (
                f"feature path is not a file: asset_type={asset.asset_type}, "
                f"asset_id={asset.asset_id}, path={path.as_posix()}"
            )

        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            return None, f"numpy is not importable while validating feature assets: {exc}"

        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            return None, (
                f"np.load failed for feature asset: asset_type={asset.asset_type}, "
                f"asset_id={asset.asset_id}, path={path.as_posix()}, error={exc}"
            )

        shape = tuple(int(value) for value in array.shape)
        dtype = str(array.dtype)
        info = HighlightFeatureArrayInfo(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            path=path,
            shape=shape,
            dtype=dtype,
            size_bytes=path.stat().st_size if path.exists() else None,
        )

        if len(shape) != 2:
            return info, (
                f"feature array must be 2-D: asset_type={asset.asset_type}, "
                f"asset_id={asset.asset_id}, shape={shape}"
            )

        if shape[0] <= 0:
            return info, (
                f"feature array has no rows: asset_type={asset.asset_type}, "
                f"asset_id={asset.asset_id}, shape={shape}"
            )

        if shape[1] != expected_feature_dim:
            return info, (
                f"feature dim mismatch: asset_type={asset.asset_type}, "
                f"asset_id={asset.asset_id}, expected_dim={expected_feature_dim}, "
                f"actual_dim={shape[1]}, shape={shape}"
            )

        return info, None

    @staticmethod
    def _latest_assets_by_type(assets: list[MediaAsset]) -> dict[str, MediaAsset]:
        # MediaAssetRepository.list_by_match returns created_at desc, so the
        # first item per type is the latest one.
        result: dict[str, MediaAsset] = {}
        for asset in assets:
            result.setdefault(asset.asset_type, asset)
        return result

    @staticmethod
    def _required_half_asset_types(expected_asset_types: list[str]) -> list[str]:
        half_types = [
            asset_type
            for asset_type in expected_asset_types
            if asset_type != COMBINED_FEATURE_ASSET_TYPE
        ]
        return half_types or DEFAULT_HALF_FEATURE_ASSET_TYPES.copy()
