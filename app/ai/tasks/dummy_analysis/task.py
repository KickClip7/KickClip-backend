import time
from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.domains.analysis.job_status import COMPLETED


class DummyAnalysisTask(BaseAITask):
    """Dummy task for validating AnalysisJob runtime flow.

    실제 AI 모델 없이 다음 흐름만 검증한다.

    QUEUED
    → RUNNING
    → 각 step RUNNING/COMPLETED
    → COMPLETED
    """

    task_type = "DUMMY_ANALYSIS"

    def __init__(self, step_sleep_sec: float = 0.8):
        self.step_sleep_sec = step_sleep_sec

    def prepare(self, context: TaskContext) -> None:
        # 6회차에서는 실제 input/model 준비가 없다.
        return None

    def run(self, context: TaskContext) -> dict[str, Any]:
        executed_steps: list[str] = []

        for step in context.ordered_steps():
            if step.status == COMPLETED:
                continue

            context.mark_step_running(step.step_key)
            time.sleep(self.step_sleep_sec)

            context.mark_step_completed(step.step_key)
            executed_steps.append(step.step_key)

        return {
            "task_type": self.task_type,
            "analysis_job_id": context.job.analysis_job_id,
            "executed_steps": executed_steps,
        }

    def save_artifacts(self, context: TaskContext, result: Any) -> None:
        # 6회차에서는 artifact를 만들지 않는다.
        # 7회차 HighlightSpottingTask에서 EVENT_CANDIDATES artifact 저장을 구현한다.
        return None