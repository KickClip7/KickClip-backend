from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.ai.registry.model_registry import ModelRegistry
from app.core.paths import get_project_root
from app.domains.analysis.model import AnalysisJob
from app.domains.analysis.repository import AnalysisJobRepository
from app.domains.analysis.schema import AnalysisJobCreate
from app.domains.analysis.service import AnalysisJobService
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage


class ActionSpottingCacheService:
    """Resolve one durable Action Spotting job per source/model/policy tuple."""

    def __init__(self, db: Session):
        self.db = db
        self.analysis_repository = AnalysisJobRepository(db)
        self.media_repository = MediaAssetRepository(db)
        self.storage = LocalStorage()

    def source_asset(self, match_id: str) -> MediaAsset:
        assets = self.media_repository.list_by_match(match_id)
        for asset_type in ("RAW_VIDEO", "RAW_VIDEO_HALF1", "WEB_PREVIEW_VIDEO"):
            asset = next(
                (row for row in assets if row.asset_type == asset_type),
                None,
            )
            if asset is not None:
                return asset
        raise ValueError("No source MediaAsset found for Action Spotting.")

    def get_or_create(
        self,
        *,
        match_id: str,
        request_options: dict,
    ) -> tuple[AnalysisJob, bool]:
        asset = self.source_asset(match_id)
        video_sha256 = asset.sha256 or self._hash_asset(asset)
        if asset.sha256 != video_sha256:
            asset.sha256 = video_sha256
            self.db.flush()

        model_card = ModelRegistry().get_model_card(
            task_type="highlight_spotting",
            alias="champion",
        )
        model_version = str(model_card.model_version or model_card.id)
        checkpoint_sha256 = str(
            model_card.extra.get("expected_checkpoint_sha256") or ""
        )
        policy_version = self._policy_version()
        cache_key = hashlib.sha256(
            "|".join(
                (
                    asset.asset_id,
                    video_sha256,
                    model_version,
                    checkpoint_sha256,
                    policy_version,
                )
            ).encode("utf-8")
        ).hexdigest()
        existing = self.analysis_repository.get_by_cache_key(cache_key)
        if existing is not None:
            if existing.status == "FAILED":
                error_contract = (existing.options or {}).get(
                    "action_spotting_error"
                ) or {}
                if not bool(error_contract.get("retryable")):
                    return existing, True
                existing.status = "QUEUED"
                existing.progress = 0
                existing.error_message = None
                existing.started_at = None
                existing.completed_at = None
                for step in existing.steps:
                    step.status = "QUEUED"
                    step.progress = 0
                    step.started_at = None
                    step.completed_at = None
                    step.error_message = None
                existing.current_step = (
                    existing.steps[0].step_key if existing.steps else None
                )
                options = dict(existing.options or {})
                options.pop("action_spotting_error", None)
                options["action_spotting_state"] = "ACTION_SPOTTING_QUEUED"
                history = list(
                    options.get("action_spotting_state_history") or []
                )
                history.append("ACTION_SPOTTING_QUEUED")
                options["action_spotting_state_history"] = history
                existing.options = options
            self.db.commit()
            return existing, True

        try:
            job = AnalysisJobService(self.db).create_analysis_job(
                AnalysisJobCreate(
                    match_id=match_id,
                    job_type="HIGHLIGHT_SPOTTING",
                    media_asset_id=asset.asset_id,
                    cache_key=cache_key,
                    video_sha256=video_sha256,
                    model_version=model_version,
                    policy_version=policy_version,
                    options={
                        **request_options,
                        "action_spotting_state": "ACTION_SPOTTING_QUEUED",
                        "action_spotting_state_history": [
                            "ACTION_SPOTTING_QUEUED"
                        ],
                        "action_spotting_cache": {
                            "media_asset_id": asset.asset_id,
                            "video_sha256": video_sha256,
                            "model_version": model_version,
                            "checkpoint_sha256": checkpoint_sha256,
                            "policy_version": policy_version,
                        },
                    },
                )
            )
        except IntegrityError:
            self.db.rollback()
            existing = self.analysis_repository.get_by_cache_key(cache_key)
            if existing is None:
                raise
            return existing, True
        return job, False

    def _hash_asset(self, asset: MediaAsset) -> str:
        path = self.storage.resolve_path(asset.file_path)
        if not path.is_file():
            raise ValueError("Source MediaAsset file is missing.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _policy_version() -> str:
        root = get_project_root()
        paths = [
            root / "configs" / "highlight_spotting.yaml",
            root
            / "configs"
            / "models"
            / "action_spotting"
            / "soccer_spotter_v9"
            / "inference_policy.json",
        ]
        if not all(path.is_file() for path in paths):
            return "highlight_scene_policy_missing"
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.read_bytes())
        return f"v9_inference_policy_{digest.hexdigest()[:16]}"
