"""사전 추출 피처를 업로드된 Match에 등록하는 로컬 데모 부트스트랩.

UI 업로드 경로는 RAW_VIDEO 자산만 만들기 때문에, SoccerNet 피처 추출 러너가
준비되지 않은 환경에서는 Champion Action Spotting이 half 피처 부재로 실패한다.
이 모듈은 `PRELOADED_FEATURE_NPY_PATH`가 가리키는 사전 추출 [T, 512] NPY를
Match의 merged/half1/half2 피처 자산으로 등록해 그 간극을 메운다.

- 설정이 비어 있으면 아무 동작도 하지 않는다 (실제 추출 파이프라인 경로 유지).
- 영상 길이와 2fps 피처 길이가 30초 넘게 어긋나면 다른 경기의 피처로 보고 거부한다.
- 결과는 flush까지만 수행하며 커밋은 호출자가 담당한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.ai.tasks.soccernet_feature_extraction.splitter import split_merged_feature
from app.core.config import get_settings
from app.domains.match.model import Match
from app.domains.media.preloaded import (
    FEATURE_DIM,
    FEATURE_FPS,
    PreloadedMatchImporter,
)
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage
from app.storage.workspace import get_match_soccernet_features_subdir


FEATURE_DURATION_TOLERANCE_SEC = 30.0
HALF_ASSET_TYPES = ("SOCCERNET_FEATURE_HALF1", "SOCCERNET_FEATURE_HALF2")


def ensure_preloaded_half_features(
    db: Session,
    match: Match,
) -> dict[str, Any]:
    """Match에 half 피처가 없으면 사전 추출 NPY를 분할 등록한다."""

    settings = get_settings()
    configured = settings.PRELOADED_FEATURE_NPY_PATH.strip()
    if settings.is_production or not configured:
        return {"status": "disabled"}

    storage = LocalStorage()
    repository = MediaAssetRepository(db)
    assets_by_type: dict[str, Any] = {}
    for asset in repository.list_by_match(match.match_id):
        assets_by_type.setdefault(asset.asset_type, asset)

    existing_halves = [assets_by_type.get(kind) for kind in HALF_ASSET_TYPES]
    if all(
        asset is not None and storage.resolve_path(asset.file_path).is_file()
        for asset in existing_halves
    ):
        return {"status": "already_available"}

    source = Path(configured).expanduser().resolve()
    if not source.is_file():
        return {"status": "feature_file_missing", "path": str(source)}

    try:
        array = np.load(source, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - 손상 파일은 등록만 건너뛴다
        return {"status": "feature_file_invalid", "error": str(exc)}
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] != FEATURE_DIM:
        return {"status": "feature_shape_invalid", "shape": list(shape)}

    feature_duration_sec = shape[0] / FEATURE_FPS
    if match.duration_sec is None:
        return {"status": "match_duration_unknown"}
    if abs(float(match.duration_sec) - feature_duration_sec) > (
        FEATURE_DURATION_TOLERANCE_SEC
    ):
        return {
            "status": "duration_mismatch",
            "video_duration_sec": float(match.duration_sec),
            "feature_duration_sec": feature_duration_sec,
        }

    feature_dir = storage.storage_root / get_match_soccernet_features_subdir(
        match.match_id
    )
    merged_target = feature_dir / "merged_feature.npy"
    half1_target = feature_dir / "half1_feature.npy"
    half2_target = feature_dir / "half2_feature.npy"

    importer = PreloadedMatchImporter(db, storage=storage)
    materialization = importer._materialize(  # noqa: SLF001 - 동일 도메인 헬퍼 재사용
        source=source,
        target=merged_target,
        mode="auto",
    )
    split_result = split_merged_feature(
        merged_path=merged_target,
        half1_path=half1_target,
        half2_path=half2_target,
        duration_sec=feature_duration_sec,
        split_strategy="midpoint",
        expected_dim=FEATURE_DIM,
        overwrite=True,
    )
    importer._upsert_feature_asset(  # noqa: SLF001
        match_id=match.match_id,
        asset_type="SOCCERNET_FEATURE",
        path=merged_target,
        original_filename=source.name,
        duration_sec=feature_duration_sec,
    )
    importer._upsert_feature_asset(  # noqa: SLF001
        match_id=match.match_id,
        asset_type="SOCCERNET_FEATURE_HALF1",
        path=half1_target,
        original_filename=f"{source.stem}_half1.npy",
        duration_sec=split_result.half1_shape[0] / FEATURE_FPS,
    )
    importer._upsert_feature_asset(  # noqa: SLF001
        match_id=match.match_id,
        asset_type="SOCCERNET_FEATURE_HALF2",
        path=half2_target,
        original_filename=f"{source.stem}_half2.npy",
        duration_sec=split_result.half2_shape[0] / FEATURE_FPS,
    )
    db.flush()
    return {
        "status": "registered",
        "materialization": materialization,
        "feature_shape": list(shape),
        "half1_shape": list(split_result.half1_shape),
        "half2_shape": list(split_result.half2_shape),
    }
