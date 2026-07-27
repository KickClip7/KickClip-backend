from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.soccernet_feature_extraction.config import (
    SoccerNetFeatureExtractionConfig,
    build_soccernet_feature_extraction_config,
)
from app.ai.tasks.soccernet_feature_extraction.extractor import (
    SoccerNetFeatureExtractionResult,
    SoccerNetFeatureExtractor,
)
from app.ai.tasks.soccernet_feature_extraction.raw_video_resolver import (
    RawVideoAssetRef,
    RawVideoResolver,
)
from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage


FEATURE_EXTRACTION_STEP_KEY = "feature_extraction"

SOCCERNET_FEATURE_ASSET_TYPE = "SOCCERNET_FEATURE"
SOCCERNET_FEATURE_HALF1_ASSET_TYPE = "SOCCERNET_FEATURE_HALF1"
SOCCERNET_FEATURE_HALF2_ASSET_TYPE = "SOCCERNET_FEATURE_HALF2"
SOCCERNET_FEATURE_METADATA_ARTIFACT_TYPE = "SOCCERNET_FEATURE_METADATA"
FEATURE_MIME_TYPE = "application/x-npy"
METADATA_MIME_TYPE = "application/json"

FEATURE_ASSET_TYPES = [
    SOCCERNET_FEATURE_ASSET_TYPE,
    SOCCERNET_FEATURE_HALF1_ASSET_TYPE,
    SOCCERNET_FEATURE_HALF2_ASSET_TYPE,
]


@dataclass(frozen=True)
class ExistingFeatureAssets:
    assets_by_type: dict[str, MediaAsset]

    @property
    def available(self) -> bool:
        return all(
            asset_type in self.assets_by_type
            for asset_type in (
                SOCCERNET_FEATURE_HALF1_ASSET_TYPE,
                SOCCERNET_FEATURE_HALF2_ASSET_TYPE,
            )
        )

    @property
    def asset_types(self) -> list[str]:
        return list(self.assets_by_type.keys())

    def to_metadata(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "asset_types": self.asset_types,
            "assets": {
                asset_type: {
                    "asset_id": asset.asset_id,
                    "file_path": asset.file_path,
                    "mime_type": asset.mime_type,
                    "duration_sec": asset.duration_sec,
                    "fps": asset.fps,
                    "size_bytes": asset.size_bytes,
                }
                for asset_type, asset in self.assets_by_type.items()
            },
        }


class SoccerNetFeatureExtractionTask(BaseAITask):
    """Create SoccerNet-style feature assets from a match RAW_VIDEO asset.

    Responsibility boundary:
    - This task converts RAW_VIDEO -> SOCCERNET_FEATURE* MediaAssets.
    - HighlightSpottingTask remains responsible for SOCCERNET_FEATURE -> events.
    """

    task_type = "SOCCERNET_FEATURE_EXTRACTION"

    def __init__(self) -> None:
        self.storage = LocalStorage()
        self.config: SoccerNetFeatureExtractionConfig | None = None
        self.raw_video: RawVideoAssetRef | None = None
        self.extraction_result: SoccerNetFeatureExtractionResult | None = None

    def prepare(self, context: TaskContext) -> None:
        self.config = build_soccernet_feature_extraction_config(
            context.job.options or {}
        )

    def run(self, context: TaskContext) -> dict[str, Any]:
        if self.config is None:
            raise RuntimeError("SoccerNetFeatureExtractionTask.prepare() was not called.")

        context.mark_step_running(FEATURE_EXTRACTION_STEP_KEY)

        if self.config.should_skip:
            context.mark_step_completed(FEATURE_EXTRACTION_STEP_KEY)
            return {
                "task_type": self.task_type,
                "status": "skipped",
                "reason": "feature extraction is disabled or skipped by config",
                "config": self.config.to_metadata(),
            }

        existing = self._find_existing_feature_assets(context)
        if (
            self.config.mode == "auto"
            and self.config.reuse_existing_feature
            and existing.available
        ):
            context.mark_step_completed(FEATURE_EXTRACTION_STEP_KEY)
            return {
                "task_type": self.task_type,
                "status": "reused",
                "reason": "existing SOCCERNET_FEATURE asset was found",
                "config": self.config.to_metadata(),
                "existing_feature_assets": existing.to_metadata(),
            }

        try:
            resolver = RawVideoResolver(context.db, storage=self.storage)
            self.raw_video = resolver.resolve_for_match(context.job.match_id)

            extractor = SoccerNetFeatureExtractor(self.config, storage=self.storage)
            self.extraction_result = extractor.extract(
                video_path=self.raw_video.resolved_path,
                match_id=context.job.match_id,
                source_asset_id=self.raw_video.asset_id,
                source_video_path=self.raw_video.file_path,
            )

        except Exception as exc:
            if self._should_fail_job(context, exc):
                raise

            # auto/fallback 개발 모드에서는 E2E를 유지한다.
            # feature asset이 없으면 HighlightSpottingTask가 fallback으로 이어진다.
            context.mark_step_completed(FEATURE_EXTRACTION_STEP_KEY)
            return {
                "task_type": self.task_type,
                "status": "failed_but_continued",
                "reason": str(exc),
                "config": self.config.to_metadata(),
            }

        context.mark_step_completed(FEATURE_EXTRACTION_STEP_KEY)
        return {
            "task_type": self.task_type,
            "status": "completed",
            "config": self.config.to_metadata(),
            "raw_video": self.raw_video.to_metadata() if self.raw_video else None,
            "extraction": self.extraction_result.to_metadata()
            if self.extraction_result
            else None,
            "_extraction_result": self.extraction_result,
        }

    def save_artifacts(self, context: TaskContext, result: dict[str, Any]) -> None:
        if result.get("status") != "completed":
            return

        extraction_result = result.get("_extraction_result")
        if not isinstance(extraction_result, SoccerNetFeatureExtractionResult):
            raise RuntimeError("Missing SoccerNetFeatureExtractionResult in task result.")

        media_repo = MediaAssetRepository(context.db)
        artifact_repo = ArtifactRepository(context.db)

        created_or_updated_assets = {
            SOCCERNET_FEATURE_ASSET_TYPE: self._upsert_feature_asset(
                context=context,
                media_repo=media_repo,
                asset_type=SOCCERNET_FEATURE_ASSET_TYPE,
                file_path=extraction_result.merged_path,
                original_filename="merged_feature.npy",
                duration_sec=extraction_result.video_probe.duration_sec,
                feature_fps=extraction_result.split_result.feature_fps,
            ),
            SOCCERNET_FEATURE_HALF1_ASSET_TYPE: self._upsert_feature_asset(
                context=context,
                media_repo=media_repo,
                asset_type=SOCCERNET_FEATURE_HALF1_ASSET_TYPE,
                file_path=extraction_result.half1_path,
                original_filename="half1_feature.npy",
                duration_sec=extraction_result.split_result.split_sec,
                feature_fps=extraction_result.split_result.feature_fps,
            ),
            SOCCERNET_FEATURE_HALF2_ASSET_TYPE: self._upsert_feature_asset(
                context=context,
                media_repo=media_repo,
                asset_type=SOCCERNET_FEATURE_HALF2_ASSET_TYPE,
                file_path=extraction_result.half2_path,
                original_filename="half2_feature.npy",
                duration_sec=max(
                    extraction_result.video_probe.duration_sec
                    - extraction_result.split_result.split_sec,
                    0.0,
                ),
                feature_fps=extraction_result.split_result.feature_fps,
            ),
        }

        metadata_artifact = self._upsert_metadata_artifact(
            context=context,
            artifact_repo=artifact_repo,
            file_path=extraction_result.metadata_path,
            metadata={
                "task_type": self.task_type,
                "status": result.get("status"),
                "source_asset_id": self.raw_video.asset_id if self.raw_video else None,
                "feature_asset_ids": {
                    asset_type: asset.asset_id
                    for asset_type, asset in created_or_updated_assets.items()
                },
                "feature_metadata": extraction_result.metadata_file.metadata,
            },
        )

        context.db.flush()
        context.db.commit()

        result["saved_assets"] = {
            asset_type: {
                "asset_id": asset.asset_id,
                "file_path": asset.file_path,
            }
            for asset_type, asset in created_or_updated_assets.items()
        }
        result["saved_artifact"] = {
            "artifact_id": metadata_artifact.artifact_id,
            "file_path": metadata_artifact.file_path,
        }

    def _find_existing_feature_assets(self, context: TaskContext) -> ExistingFeatureAssets:
        repo = MediaAssetRepository(context.db)
        assets = repo.list_by_match(context.job.match_id)
        assets_by_type: dict[str, MediaAsset] = {}

        for asset in assets:
            if asset.asset_type not in FEATURE_ASSET_TYPES:
                continue
            if asset.asset_type in assets_by_type:
                continue
            if not self._asset_file_exists(asset):
                continue
            assets_by_type[asset.asset_type] = asset

        return ExistingFeatureAssets(assets_by_type=assets_by_type)

    def _asset_file_exists(self, asset: MediaAsset) -> bool:
        try:
            return self.storage.resolve_path(asset.file_path).is_file()
        except Exception:
            return False

    def _upsert_feature_asset(
        self,
        *,
        context: TaskContext,
        media_repo: MediaAssetRepository,
        asset_type: str,
        file_path: Path,
        original_filename: str,
        duration_sec: float | None,
        feature_fps: float | None,
    ) -> MediaAsset:
        existing = self._find_latest_media_asset_by_type(context, asset_type)
        relative_path = self._to_project_relative_path(file_path)
        size_bytes = file_path.stat().st_size if file_path.exists() else None

        if existing is None:
            return media_repo.create(
                match_id=context.job.match_id,
                asset_type=asset_type,
                file_path=relative_path,
                original_filename=original_filename,
                mime_type=FEATURE_MIME_TYPE,
                duration_sec=duration_sec,
                fps=feature_fps,
                width=None,
                height=None,
                size_bytes=size_bytes,
            )

        existing.file_path = relative_path
        existing.original_filename = original_filename
        existing.mime_type = FEATURE_MIME_TYPE
        existing.duration_sec = duration_sec
        existing.fps = feature_fps
        existing.width = None
        existing.height = None
        existing.size_bytes = size_bytes
        context.db.add(existing)
        context.db.flush()
        return existing

    def _upsert_metadata_artifact(
        self,
        *,
        context: TaskContext,
        artifact_repo: ArtifactRepository,
        file_path: Path,
        metadata: dict[str, Any],
    ) -> Artifact:
        existing = self._find_latest_metadata_artifact_for_job(context)
        relative_path = self._to_project_relative_path(file_path)

        if existing is None:
            return artifact_repo.create(
                match_id=context.job.match_id,
                analysis_job_id=context.job.analysis_job_id,
                artifact_type=SOCCERNET_FEATURE_METADATA_ARTIFACT_TYPE,
                file_path=relative_path,
                mime_type=METADATA_MIME_TYPE,
                metadata_=metadata,
            )

        existing.file_path = relative_path
        existing.mime_type = METADATA_MIME_TYPE
        existing.metadata_ = metadata
        context.db.add(existing)
        context.db.flush()
        return existing

    def _find_latest_media_asset_by_type(
        self,
        context: TaskContext,
        asset_type: str,
    ) -> MediaAsset | None:
        repo = MediaAssetRepository(context.db)
        for asset in repo.list_by_match(context.job.match_id):
            if asset.asset_type == asset_type:
                return asset
        return None

    def _find_latest_metadata_artifact_for_job(
        self,
        context: TaskContext,
    ) -> Artifact | None:
        repo = ArtifactRepository(context.db)
        artifacts = repo.list_by_analysis_job(context.job.analysis_job_id)
        for artifact in artifacts:
            if artifact.artifact_type == SOCCERNET_FEATURE_METADATA_ARTIFACT_TYPE:
                return artifact
        return None

    def _should_fail_job(self, context: TaskContext, exc: Exception) -> bool:
        if self.config is None:
            return True

        options = context.job.options or {}
        predictor_mode = str(options.get("highlight_predictor_mode") or "auto").lower()
        strict_real_model = bool(options.get("strict_real_model", False))
        allow_fallback_when_missing = bool(options.get("allow_fallback_when_missing", True))

        if not self.config.allow_analysis_to_continue_on_failure:
            return True

        if strict_real_model:
            return True

        if predictor_mode == "real" and not allow_fallback_when_missing:
            return True

        return False

    def _to_project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.storage.project_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
