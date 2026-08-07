from __future__ import annotations

from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.highlight_spotting.task import HighlightSpottingTask
from app.ai.tasks.soccernet_feature_extraction.task import (
    SoccerNetFeatureExtractionTask,
)
from app.domains.action_spotting.errors import (
    ActionSpottingError,
    feature_extraction_failed,
    inference_failed,
)


class ActionSpottingPipelineTask(BaseAITask):
    """RAW_VIDEO -> SoccerNet PCA512 halves -> six-class Champion events."""

    task_type = "HIGHLIGHT_SPOTTING"

    def run(self, context: TaskContext) -> dict[str, Any]:
        options = context.job.options or {}
        feature_result: dict[str, Any] | None = None

        if options.get("run_feature_extraction", True):
            try:
                feature_result = SoccerNetFeatureExtractionTask().execute(context)
            except ActionSpottingError:
                raise
            except Exception as exc:
                raise feature_extraction_failed(
                    "SoccerNet PCA512 feature extraction failed before Action Spotting.",
                    exception_type=type(exc).__name__,
                    detail=str(exc),
                ) from exc
            if feature_result.get("status") in {"completed", "reused"}:
                self._set_workflow_state(context, "FEATURE_READY")

        self._set_workflow_state(context, "ACTION_SPOTTING_RUNNING")
        try:
            spotting_result = HighlightSpottingTask().execute(context)
        except ActionSpottingError:
            raise
        except Exception as exc:
            raise inference_failed(
                "Champion Action Spotting inference failed.",
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc
        self._set_workflow_state(context, "ACTION_SPOTTING_COMPLETED")
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

    @staticmethod
    def _set_workflow_state(context: TaskContext, state: str) -> None:
        options = dict(context.job.options or {})
        options["action_spotting_state"] = state
        history = list(options.get("action_spotting_state_history") or [])
        if not history or history[-1] != state:
            history.append(state)
        options["action_spotting_state_history"] = history
        context.job.options = options
        context.commit()
