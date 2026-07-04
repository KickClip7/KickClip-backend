from abc import ABC
from typing import Any

from app.ai.runtime.task_context import TaskContext


class BaseAITask(ABC):
    """Common interface for all KickClip AI tasks.

    모든 AI task는 다음 흐름을 따른다.

    prepare()
    → run()
    → save_artifacts()

    7회차 HighlightSpottingTask, 8회차 PlayerTrackingTask,
    9회차 TimelineFusionTask도 이 인터페이스를 따르게 만든다.
    """

    task_type: str = "BASE_AI_TASK"

    def prepare(self, context: TaskContext) -> None:
        """Prepare task inputs, configs, model handles, etc."""
        return None

    def run(self, context: TaskContext) -> Any:
        """Run task main logic."""
        raise NotImplementedError

    def save_artifacts(self, context: TaskContext, result: Any) -> None:
        """Save artifacts after task execution.

        6회차 dummy task는 artifact를 만들지 않는다.
        7회차부터 event_candidates.json 같은 artifact를 저장한다.
        """
        return None

    def execute(self, context: TaskContext) -> Any:
        self.prepare(context)
        result = self.run(context)
        self.save_artifacts(context, result)
        return result