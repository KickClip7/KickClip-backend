from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.highlight_spotting.config import (
    HighlightSpottingRuntimeConfig,
    build_highlight_runtime_config,
    load_highlight_postprocess_config,
)
from app.ai.tasks.highlight_spotting.postprocessor import HighlightPostprocessor
from app.ai.tasks.highlight_spotting.predictor import (
    HighlightPredictionUnavailable,
    HighlightSpottingPredictor,
)
from app.domains.artifact.repository import ArtifactRepository
from app.domains.timeline.schema import TimelineEventCreate
from app.domains.timeline.service import TimelineEventService
from app.storage.local_storage import LocalStorage
from app.storage.workspace import get_match_event_candidates_subdir


class HighlightSpottingTask(BaseAITask):
    task_type = "HIGHLIGHT_SPOTTING"

    def __init__(self) -> None:
        self.storage = LocalStorage()
        self.model_card = None
        self.runtime_config: HighlightSpottingRuntimeConfig | None = None
        self.postprocessor: HighlightPostprocessor | None = None
        self.predictor_diagnostics: dict[str, Any] = {}

    def prepare(self, context: TaskContext) -> None:
        self.model_card = context.model_registry.get_model_card(
            task_type="highlight_spotting",
            alias="champion",
        )
        self.runtime_config = build_highlight_runtime_config(
            model_card=self.model_card,
            job_options=context.job.options or {},
        )
        config = load_highlight_postprocess_config(self.model_card)
        self.postprocessor = HighlightPostprocessor(**config)

    def run(self, context: TaskContext) -> dict[str, Any]:
        if self.model_card is None or self.postprocessor is None or self.runtime_config is None:
            raise RuntimeError("HighlightSpottingTask.prepare() was not called.")

        self._complete_step_if_exists(context, "scene_detection")

        context.mark_step_running("event_classification")

        predictor = HighlightSpottingPredictor(
            model_card=self.model_card,
            runtime_config=self.runtime_config,
        )

        try:
            raw_predictions = predictor.predict(context)
        except HighlightPredictionUnavailable:
            # JobRunner will mark the whole job as FAILED with the detailed exception.
            raise

        self.predictor_diagnostics = (
            predictor.diagnostics.to_metadata()
            if predictor.diagnostics is not None
            else {}
        )

        match_duration_sec = context.job.match.duration_sec
        candidates = self.postprocessor.process(
            raw_predictions=raw_predictions,
            match_duration_sec=match_duration_sec,
        )

        context.mark_step_completed("event_classification")

        # HIGHLIGHT_SPOTTING 단독 job에는 scoring step이 존재한다.
        # FULL_MATCH_ANALYSIS에서는 PlayerTrackingTask 이후 scoring을 처리한다.
        if context.get_step("player_tracking") is None:
            self._complete_step_if_exists(context, "scoring")

        return {
            "task_type": self.task_type,
            "analysis_job_id": context.job.analysis_job_id,
            "match_id": context.job.match_id,
            "model": self.model_card.model_dump(),
            "runtime_config": self._runtime_config_metadata(),
            "predictor_diagnostics": self.predictor_diagnostics,
            "num_candidates": len(candidates),
            "candidates": candidates,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_artifacts(self, context: TaskContext, result: dict[str, Any]) -> None:
        artifact_path = self._save_event_candidates_json(context, result)

        artifact_repo = ArtifactRepository(context.db)
        artifact = artifact_repo.create(
            match_id=context.job.match_id,
            analysis_job_id=context.job.analysis_job_id,
            artifact_type="EVENT_CANDIDATES",
            file_path=artifact_path,
            mime_type="application/json",
            metadata_={
                "task_type": self.task_type,
                "num_candidates": result["num_candidates"],
                "model_id": result["model"].get("id"),
                "model_name": result["model"].get("model_name"),
                "predictor_mode": self._detect_predictor_mode(result["candidates"]),
                "requested_predictor_mode": result.get("runtime_config", {}).get("predictor_mode"),
                "predictor_diagnostics": result.get("predictor_diagnostics") or {},
            },
        )
        context.db.flush()

        timeline_service = TimelineEventService(context.db)
        rows = [
            self._candidate_to_timeline_create(
                context=context,
                source_artifact_id=artifact.artifact_id,
                candidate=candidate,
            )
            for candidate in result["candidates"]
        ]

        timeline_service.replace_events_for_job(
            source_job_id=context.job.analysis_job_id,
            rows=rows,
            commit=False,
        )

        context.db.commit()

    def _save_event_candidates_json(
        self,
        context: TaskContext,
        result: dict[str, Any],
    ) -> str:
        output_dir = (
            self.storage.storage_root
            / get_match_event_candidates_subdir(context.job.match_id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"event_candidates_{context.job.analysis_job_id}.json"
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        return self._to_project_relative_path(output_path)

    def _candidate_to_timeline_create(
        self,
        context: TaskContext,
        source_artifact_id: str,
        candidate: dict[str, Any],
    ) -> TimelineEventCreate:
        return TimelineEventCreate(
            match_id=context.job.match_id,
            source_artifact_id=source_artifact_id,
            source_job_id=context.job.analysis_job_id,
            event_type=candidate["event_type"],
            label=candidate["label"],
            half=candidate.get("half"),
            timestamp_sec=candidate["timestamp_sec"],
            start_sec=candidate["start_sec"],
            end_sec=candidate["end_sec"],
            duration_sec=candidate["duration_sec"],
            confidence=candidate.get("confidence"),
            highlight_score=candidate.get("highlight_score"),
            title=candidate.get("title"),
            description=candidate.get("description"),
            team_name=candidate.get("team_name"),
            player_ids=candidate.get("player_ids") or [],
            metadata=candidate.get("metadata") or {},
        )

    def _runtime_config_metadata(self) -> dict[str, Any]:
        if self.runtime_config is None:
            return {}

        return {
            "predictor_mode": self.runtime_config.predictor_mode,
            "allow_fallback_when_missing": self.runtime_config.allow_fallback_when_missing,
            "strict_real_model": self.runtime_config.strict_real_model,
            "model_adapter": self.runtime_config.model_adapter,
            "device": self.runtime_config.device,
            "checkpoint_configured": self.runtime_config.checkpoint_path is not None,
            "model_configured": self.runtime_config.model_config_path is not None,
            "expected_feature_asset_types": self.runtime_config.expected_feature_asset_types,
        }

    def _complete_step_if_exists(
        self,
        context: TaskContext,
        step_key: str,
    ) -> None:
        step = context.get_step(step_key)
        if step is None:
            return

        if step.status == "COMPLETED":
            return

        context.mark_step_running(step_key)
        context.mark_step_completed(step_key)

    def _to_project_relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.storage.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _detect_predictor_mode(candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "empty"

        metadata = candidates[0].get("metadata") or {}
        return metadata.get("predictor_mode", "unknown")
