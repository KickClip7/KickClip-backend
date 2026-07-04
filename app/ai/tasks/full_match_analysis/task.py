from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.highlight_spotting.task import HighlightSpottingTask
from app.ai.tasks.player_tracking.task import PlayerTrackingTask
from app.ai.tasks.soccernet_feature_extraction.task import SoccerNetFeatureExtractionTask


class FullMatchAnalysisTask(BaseAITask):
    task_type = "FULL_MATCH_ANALYSIS"

    def run(self, context: TaskContext) -> dict[str, Any]:
        options = context.job.options or {}

        result: dict[str, Any] = {
            "task_type": self.task_type,
            "analysis_job_id": context.job.analysis_job_id,
            "subtasks": [],
        }

        if options.get("run_feature_extraction", True):
            feature_result = SoccerNetFeatureExtractionTask().execute(context)
            result["subtasks"].append(
                {
                    "task_type": "SOCCERNET_FEATURE_EXTRACTION",
                    "status": feature_result.get("status"),
                    "reason": feature_result.get("reason"),
                }
            )

        if options.get("run_highlight_spotting", True):
            highlight_result = HighlightSpottingTask().execute(context)
            result["subtasks"].append(
                {
                    "task_type": "HIGHLIGHT_SPOTTING",
                    "num_candidates": highlight_result.get("num_candidates"),
                }
            )

        if options.get("run_player_tracking", True):
            player_result = PlayerTrackingTask().execute(context)
            result["subtasks"].append(
                {
                    "task_type": "PLAYER_TRACKING",
                    "num_players": player_result.get("num_players"),
                    "num_tracks": player_result.get("num_tracks"),
                }
            )

        return result

    def save_artifacts(self, context: TaskContext, result: dict[str, Any]) -> None:
        # 각각의 subtask가 자체 artifact를 저장한다.
        return None
