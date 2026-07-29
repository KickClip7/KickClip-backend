from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ai.registry.model_card import ModelCard
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.highlight_spotting.adapters.base import HighlightRawPrediction
from app.ai.tasks.highlight_spotting.adapters.soccer_spotter_v9 import (
    CHAMPION_IDENTIFIER,
    SoccerSpotterV9Adapter,
)
from app.ai.tasks.highlight_spotting.config import HighlightSpottingRuntimeConfig
from app.ai.tasks.highlight_spotting.feature_loader import (
    HighlightFeatureBundle,
    HighlightFeatureLoader,
)
from app.domains.action_spotting.errors import (
    ActionSpottingError,
    checkpoint_mismatch,
    feature_mismatch,
    runtime_unavailable,
)


HighlightPredictionUnavailable = ActionSpottingError


@dataclass(frozen=True)
class HighlightPredictionDiagnostics:
    requested_mode: str
    selected_mode: str
    model_readiness: dict[str, Any]
    feature_bundle: dict[str, Any]
    real_adapter: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "model_readiness": self.model_readiness,
            "feature_bundle": self.feature_bundle,
            "real_adapter": self.real_adapter,
        }


class HighlightSpottingPredictor:
    """Champion-only Action Spotting predictor with no synthetic fallback."""

    def __init__(
        self,
        model_card: ModelCard,
        runtime_config: HighlightSpottingRuntimeConfig,
    ) -> None:
        self.model_card = model_card
        self.runtime_config = runtime_config
        self.diagnostics: HighlightPredictionDiagnostics | None = None

    def predict(self, context: TaskContext) -> list[dict[str, Any]]:
        if not self.model_card.enabled:
            raise runtime_unavailable("Champion Action Spotting is disabled.")
        if self.runtime_config.predictor_mode != "real":
            raise runtime_unavailable(
                "Champion Action Spotting does not support fallback or mock inference.",
                requested_mode=self.runtime_config.predictor_mode,
            )

        feature_bundle = HighlightFeatureLoader(context.db).load_for_match(
            match_id=context.job.match_id,
            expected_asset_types=self.runtime_config.expected_feature_asset_types,
            validate=True,
            expected_feature_dim=512,
        )
        if not feature_bundle.available:
            raise feature_mismatch(
                "SoccerNet PCA512 half features are missing or invalid.",
                feature_bundle=feature_bundle.to_metadata(include_paths=True),
            )

        model_dir = (
            self.runtime_config.checkpoint_path.parent
            if self.runtime_config.checkpoint_path is not None
            else None
        )
        adapter = SoccerSpotterV9Adapter.from_model_dir(model_dir)
        if self.model_card.id != CHAMPION_IDENTIFIER:
            raise checkpoint_mismatch(
                "Configured Action Spotting model is not the Champion.",
                expected=CHAMPION_IDENTIFIER,
                actual=self.model_card.id,
            )
        if (
            self.runtime_config.checkpoint_path is None
            or self.runtime_config.checkpoint_path.resolve()
            != adapter.paths.checkpoint_path.resolve()
            or self.runtime_config.model_config_path is None
            or self.runtime_config.model_config_path.resolve()
            != adapter.paths.model_config_path.resolve()
        ):
            raise checkpoint_mismatch(
                "Model registry paths do not match the Champion runtime bundle."
            )
        preflight = adapter.preflight()
        self.diagnostics = HighlightPredictionDiagnostics(
            requested_mode="real",
            selected_mode="real",
            model_readiness=preflight.to_metadata(),
            feature_bundle=feature_bundle.to_metadata(include_paths=False),
        )
        if not preflight.ready_for_real_adapter:
            # Re-run typed validators so a stable specific error code reaches
            # the job failure contract.
            adapter._validate_contract()
            adapter._load_model(device="cpu")
            raise runtime_unavailable(
                "Champion adapter preflight failed.",
                reasons=preflight.reasons,
            )

        raw_predictions = adapter.predict(
            feature_bundle=feature_bundle,
            match_duration_sec=context.job.match.duration_sec,
            device=self.runtime_config.device,
            batch_size=self.runtime_config.real_adapter_batch_size,
            max_candidates=self.runtime_config.real_adapter_max_candidates,
            debug_min_threshold=self.runtime_config.debug_min_threshold,
        )
        self.diagnostics = HighlightPredictionDiagnostics(
            requested_mode="real",
            selected_mode="real",
            model_readiness=preflight.to_metadata(),
            feature_bundle=feature_bundle.to_metadata(include_paths=False),
            real_adapter={
                "status": (
                    "completed" if raw_predictions else "completed_no_predictions"
                ),
                "num_raw_predictions": len(raw_predictions),
                "selected_device": str(adapter._device),
            },
        )
        return self._raw_predictions_to_dicts(raw_predictions, feature_bundle)

    @staticmethod
    def _raw_predictions_to_dicts(
        raw_predictions: list[HighlightRawPrediction],
        feature_bundle: HighlightFeatureBundle,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for prediction in raw_predictions:
            item = prediction.to_postprocessor_input()
            metadata = dict(item.get("metadata") or {})
            metadata["predictor_mode"] = "real"
            metadata["feature_contract"] = {
                "layout": feature_bundle.layout,
                "asset_types": feature_bundle.asset_types,
                "feature_shapes": feature_bundle.feature_shapes,
            }
            item["metadata"] = metadata
            item["source"] = "champion_action_spotting"
            item["player_ids"] = []
            candidates.append(item)
        return candidates
