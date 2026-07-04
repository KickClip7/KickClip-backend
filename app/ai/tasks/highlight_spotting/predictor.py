from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ai.registry.model_card import ModelCard
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.highlight_spotting.adapters.base import HighlightRawPrediction
from app.ai.tasks.highlight_spotting.adapters.soccer_highlight_former import (
    SoccerHighlightFormerAdapter,
)
from app.ai.tasks.highlight_spotting.config import HighlightSpottingRuntimeConfig
from app.ai.tasks.highlight_spotting.feature_loader import (
    HighlightFeatureBundle,
    HighlightFeatureLoader,
)
from app.ai.tasks.highlight_spotting.model_loader import (
    HighlightModelLoader,
    HighlightModelReadiness,
)
from app.domains.media.repository import MediaAssetRepository


class HighlightPredictionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class HighlightPredictionDiagnostics:
    requested_mode: str
    selected_mode: str
    fallback_reason: str | None
    model_readiness: dict[str, Any]
    feature_bundle: dict[str, Any]
    real_adapter: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "fallback_reason": self.fallback_reason,
            "model_readiness": self.model_readiness,
            "feature_bundle": self.feature_bundle,
            "real_adapter": self.real_adapter,
        }


class HighlightSpottingPredictor:
    """Highlight spotting predictor adapter.

    Round 25:
    - fallback mode remains available for E2E stability.
    - auto/real mode can now use the SoccerHighlightFormer torch adapter when
      champion artifacts and SoccerNet feature assets are ready.
    """

    def __init__(
        self,
        model_card: ModelCard,
        runtime_config: HighlightSpottingRuntimeConfig,
    ):
        self.model_card = model_card
        self.runtime_config = runtime_config
        self.model_loader = HighlightModelLoader()
        self.diagnostics: HighlightPredictionDiagnostics | None = None

    def predict(self, context: TaskContext) -> list[dict[str, Any]]:
        readiness = self._inspect_model_readiness()
        feature_bundle = HighlightFeatureLoader(context.db).load_for_match(
            match_id=context.job.match_id,
            expected_asset_types=self.runtime_config.expected_feature_asset_types,
            validate=True,
            expected_feature_dim=512,
        )

        if self.runtime_config.force_fallback:
            reason = "fallback mode forced by dummy_mode or config"
            self._set_diagnostics("fallback", reason, readiness, feature_bundle)
            return self._predict_with_fallback(context, reason)

        if self.runtime_config.request_real_model:
            if not readiness.ready or not feature_bundle.available:
                reason = self._build_unavailable_reason(readiness, feature_bundle)
                if self.runtime_config.allow_fallback_when_missing:
                    self._set_diagnostics("fallback", reason, readiness, feature_bundle)
                    return self._predict_with_fallback(context, reason)
                self._set_diagnostics("unavailable", reason, readiness, feature_bundle)
                raise HighlightPredictionUnavailable(reason)

            try:
                self._set_diagnostics("real", None, readiness, feature_bundle)
                return self._predict_with_champion_model(
                    context=context,
                    readiness=readiness,
                    feature_bundle=feature_bundle,
                )
            except Exception as exc:
                reason = f"real SoccerHighlightFormer adapter failed: {exc}"
                if self.runtime_config.allow_fallback_when_missing:
                    self._set_diagnostics("fallback", reason, readiness, feature_bundle)
                    return self._predict_with_fallback(context, reason)
                self._set_diagnostics("unavailable", reason, readiness, feature_bundle)
                raise HighlightPredictionUnavailable(reason) from exc

        # auto mode: ready이면 real 시도, 실패하면 fallback.
        if readiness.ready and feature_bundle.available:
            try:
                self._set_diagnostics("real", None, readiness, feature_bundle)
                return self._predict_with_champion_model(
                    context=context,
                    readiness=readiness,
                    feature_bundle=feature_bundle,
                )
            except Exception as exc:
                reason = f"real SoccerHighlightFormer adapter failed: {exc}"
                self._set_diagnostics("fallback", reason, readiness, feature_bundle)
                return self._predict_with_fallback(context, reason)

        reason = self._build_unavailable_reason(readiness, feature_bundle)
        self._set_diagnostics("fallback", reason, readiness, feature_bundle)
        return self._predict_with_fallback(context, reason)

    def _inspect_model_readiness(self) -> HighlightModelReadiness:
        return self.model_loader.inspect(
            checkpoint_path=self.runtime_config.checkpoint_path,
            model_config_path=self.runtime_config.model_config_path,
            label_map_path=self.runtime_config.label_map_path,
            model_enabled=self.model_card.enabled,
            adapter_name=self.runtime_config.model_adapter,
        )

    def _predict_with_champion_model(
        self,
        *,
        context: TaskContext,
        readiness: HighlightModelReadiness,
        feature_bundle: HighlightFeatureBundle,
    ) -> list[dict[str, Any]]:
        if self.runtime_config.checkpoint_path is None:
            raise HighlightPredictionUnavailable("checkpoint_path is not configured")

        adapter = SoccerHighlightFormerAdapter.from_model_dir(
            self.runtime_config.checkpoint_path.parent,
        )
        preflight = adapter.preflight()
        if not preflight.ready_for_real_adapter:
            raise HighlightPredictionUnavailable(
                "SoccerHighlightFormer adapter preflight failed: "
                + "; ".join(preflight.reasons)
            )

        raw_predictions = adapter.predict(
            feature_bundle=feature_bundle,
            match_duration_sec=context.job.match.duration_sec,
            device=self.runtime_config.device,
            batch_size=self.runtime_config.real_adapter_batch_size,
            max_candidates=self.runtime_config.real_adapter_max_candidates,
            debug_min_threshold=self.runtime_config.debug_min_threshold,
        )

        real_adapter_metadata = {
            "preflight": preflight.to_metadata(),
            "num_raw_predictions": len(raw_predictions),
            "device": self.runtime_config.device,
            "batch_size": self.runtime_config.real_adapter_batch_size,
            "max_candidates": self.runtime_config.real_adapter_max_candidates,
            "debug_min_threshold": self.runtime_config.debug_min_threshold,
        }

        if not raw_predictions:
            # No event above threshold is a valid model result, especially for
            # short clips or calm match segments. It should produce an empty
            # EVENT_CANDIDATES artifact, not fail the AnalysisJob.
            real_adapter_metadata["status"] = "completed_no_predictions"
            real_adapter_metadata["reason"] = (
                "SoccerHighlightFormer produced no raw predictions above configured thresholds"
            )
            self._set_diagnostics(
                "real",
                None,
                readiness,
                feature_bundle,
                real_adapter=real_adapter_metadata,
            )
            return []

        real_adapter_metadata["status"] = "completed"
        self._set_diagnostics(
            "real",
            None,
            readiness,
            feature_bundle,
            real_adapter=real_adapter_metadata,
        )
        return self._raw_predictions_to_dicts(raw_predictions, feature_bundle)

    def _raw_predictions_to_dicts(
        self,
        raw_predictions: list[HighlightRawPrediction],
        feature_bundle: HighlightFeatureBundle,
    ) -> list[dict[str, Any]]:
        feature_bundle_metadata = feature_bundle.to_metadata()
        candidates: list[dict[str, Any]] = []
        for prediction in raw_predictions:
            item = prediction.to_postprocessor_input()
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("predictor_mode", "real")
            metadata.setdefault("feature_bundle", feature_bundle_metadata)
            item["metadata"] = metadata
            item["source"] = "soccer_highlight_former"
            candidates.append(item)
        return candidates

    def _predict_with_fallback(
        self,
        context: TaskContext,
        fallback_reason: str,
    ) -> list[dict[str, Any]]:
        match = context.job.match
        duration = match.duration_sec or 5400.0

        media_repo = MediaAssetRepository(context.db)
        raw_assets = [
            asset
            for asset in media_repo.list_by_match(context.job.match_id)
            if asset.asset_type == "RAW_VIDEO"
        ]

        source_asset_id = raw_assets[0].asset_id if raw_assets else None
        points = self._build_demo_points(duration)
        feature_bundle_metadata = (
            self.diagnostics.feature_bundle if self.diagnostics is not None else {}
        )

        candidates: list[dict[str, Any]] = []
        for item in points:
            timestamp_sec = min(float(item["timestamp_sec"]), max(duration - 1, 1))

            candidates.append(
                {
                    "label": item["label"],
                    "half": 1 if timestamp_sec < duration / 2 else 2,
                    "timestamp_sec": timestamp_sec,
                    "confidence": item["confidence"],
                    "highlight_score": item["highlight_score"],
                    "team_name": match.home_team,
                    "player_ids": item.get("player_ids", []),
                    "metadata": {
                        "predictor_mode": "fallback",
                        "reason": fallback_reason,
                        "source_asset_id": source_asset_id,
                        "real_model_requested_mode": self.runtime_config.predictor_mode,
                        "feature_bundle": feature_bundle_metadata,
                    },
                    "source": "fallback_highlight_spotting",
                }
            )

        return candidates

    def _set_diagnostics(
        self,
        selected_mode: str,
        fallback_reason: str | None,
        readiness: HighlightModelReadiness,
        feature_bundle: HighlightFeatureBundle,
        real_adapter: dict[str, Any] | None = None,
    ) -> None:
        self.diagnostics = HighlightPredictionDiagnostics(
            requested_mode=self.runtime_config.predictor_mode,
            selected_mode=selected_mode,
            fallback_reason=fallback_reason,
            model_readiness=readiness.to_metadata(),
            feature_bundle=feature_bundle.to_metadata(),
            real_adapter=real_adapter,
        )

    @staticmethod
    def _build_unavailable_reason(readiness: HighlightModelReadiness, feature_bundle) -> str:
        reasons: list[str] = []
        if not readiness.ready:
            reasons.extend(readiness.reasons)
        if not feature_bundle.available:
            if feature_bundle.missing_asset_types:
                reasons.append(
                    "required SoccerNet feature assets are missing: "
                    + ", ".join(feature_bundle.missing_asset_types)
                )
            if feature_bundle.validation_errors:
                reasons.append(
                    "required SoccerNet feature assets failed validation: "
                    + " | ".join(feature_bundle.validation_errors)
                )
            if not feature_bundle.assets:
                reasons.append("no SoccerNet feature assets were found for this match")
        return "; ".join(reasons) or "real predictor is unavailable"

    @staticmethod
    def _build_demo_points(duration_sec: float) -> list[dict[str, Any]]:
        # 너무 짧은 영상에서도 후보가 생성되도록 duration에 비례해서 timestamp를 만든다.
        def at(ratio: float) -> float:
            return max(min(duration_sec * ratio, duration_sec - 1), 1.0)

        return [
            {
                "label": "goal",
                "timestamp_sec": at(0.12),
                "confidence": 0.92,
                "highlight_score": 9.5,
                "player_ids": ["player_007", "player_010"],
            },
            {
                "label": "shot",
                "timestamp_sec": at(0.28),
                "confidence": 0.81,
                "highlight_score": 8.1,
                "player_ids": ["player_009"],
            },
            {
                "label": "foul",
                "timestamp_sec": at(0.45),
                "confidence": 0.72,
                "highlight_score": 6.8,
                "player_ids": [],
            },
            {
                "label": "corner",
                "timestamp_sec": at(0.62),
                "confidence": 0.67,
                "highlight_score": 6.5,
                "player_ids": [],
            },
            {
                "label": "shot",
                "timestamp_sec": at(0.78),
                "confidence": 0.86,
                "highlight_score": 8.4,
                "player_ids": ["player_007"],
            },
        ]
