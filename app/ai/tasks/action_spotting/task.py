from __future__ import annotations

from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.highlight_spotting.task import HighlightSpottingTask
from app.ai.tasks.soccernet_feature_extraction.task import (
    SoccerNetFeatureExtractionTask,
)


class ActionSpottingPipelineTask(BaseAITask):
    """RAW_VIDEO -> SoccerNet PCA512 -> four-class action events."""

    task_type = "HIGHLIGHT_SPOTTING"

    def run(self, context: TaskContext) -> dict[str, Any]:
        options = context.job.options or {}
        feature_result: dict[str, Any] | None = None

        if options.get("run_feature_extraction", True):
            feature_result = SoccerNetFeatureExtractionTask().execute(context)

        spotting_result = HighlightSpottingTask().execute(context)
        return {
            "task_type": self.task_type,
            "analysis_job_id": context.job.analysis_job_id,
            "feature_extraction": {
                "status": feature_result.get("status"),
                "reason": feature_result.get("reason"),
            }
            if feature_result is not None
            else {"status": "skipped", "reason": "disabled by job option"},
            "num_candidates": spotting_result.get("num_candidates", 0),
        }

    def save_artifacts(self, context: TaskContext, result: dict[str, Any]) -> None:
        # Each child task persists its own MediaAsset, Artifact, and TimelineEvent rows.
        return None
