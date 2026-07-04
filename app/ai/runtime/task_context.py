from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.ai.registry.model_registry import ModelRegistry
from app.core.config import Settings, get_settings
from app.domains.analysis.job_status import COMPLETED, FAILED, RUNNING
from app.domains.analysis.model import AnalysisJob, AnalysisJobStep


@dataclass
class TaskContext:
    """Runtime context passed to AI tasks.

    task가 DB session, job, model registry, settings에 접근할 수 있게 한다.
    단, 실제 모델 로딩/추론 로직은 각 task 내부에서 구현한다.
    """

    db: Session
    job: AnalysisJob
    model_registry: ModelRegistry
    settings: Settings

    @classmethod
    def create(
        cls,
        db: Session,
        job: AnalysisJob,
        model_registry: ModelRegistry | None = None,
    ) -> "TaskContext":
        return cls(
            db=db,
            job=job,
            model_registry=model_registry or ModelRegistry(),
            settings=get_settings(),
        )

    def ordered_steps(self) -> list[AnalysisJobStep]:
        return sorted(self.job.steps, key=lambda step: step.created_at)

    def get_step(self, step_key: str) -> AnalysisJobStep | None:
        for step in self.job.steps:
            if step.step_key == step_key:
                return step
        return None

    def mark_job_running(self) -> None:
        now = datetime.now(timezone.utc)

        self.job.status = RUNNING
        self.job.started_at = self.job.started_at or now
        self.job.error_message = None

        next_step = self._find_next_not_completed_step()
        self.job.current_step = next_step.step_key if next_step else None

        self._recalculate_job_progress()
        self.commit()

    def mark_step_running(self, step_key: str) -> None:
        step = self.get_step(step_key)
        if step is None:
            return

        now = datetime.now(timezone.utc)

        step.status = RUNNING
        step.started_at = step.started_at or now
        step.error_message = None

        self.job.status = RUNNING
        self.job.current_step = step.step_key
        self.job.started_at = self.job.started_at or now

        self._recalculate_job_progress()
        self.commit()

    def mark_step_completed(self, step_key: str) -> None:
        step = self.get_step(step_key)
        if step is None:
            return

        now = datetime.now(timezone.utc)

        step.status = COMPLETED
        step.progress = 100
        step.completed_at = now
        step.error_message = None

        next_step = self._find_next_not_completed_step()
        self.job.current_step = next_step.step_key if next_step else None

        self._recalculate_job_progress()
        self.commit()

    def mark_job_completed(self) -> None:
        now = datetime.now(timezone.utc)

        for step in self.job.steps:
            if step.status != COMPLETED:
                step.status = COMPLETED
                step.progress = 100
                step.completed_at = step.completed_at or now

        self.job.status = COMPLETED
        self.job.progress = 100
        self.job.current_step = None
        self.job.completed_at = now
        self.job.error_message = None

        self.commit()

    def mark_job_failed(self, error_message: str) -> None:
        now = datetime.now(timezone.utc)

        self.job.status = FAILED
        self.job.completed_at = now
        self.job.error_message = error_message

        current_step = self.get_step(self.job.current_step) if self.job.current_step else None
        if current_step is not None:
            current_step.status = FAILED
            current_step.error_message = error_message
            current_step.completed_at = now

        self.commit()

    def commit(self) -> None:
        self.db.add(self.job)
        self.db.commit()
        self.db.refresh(self.job)

    def _find_next_not_completed_step(self) -> AnalysisJobStep | None:
        for step in self.ordered_steps():
            if step.status != COMPLETED:
                return step
        return None

    def _recalculate_job_progress(self) -> None:
        steps = self.ordered_steps()
        if not steps:
            return

        total = sum(step.progress for step in steps)
        self.job.progress = int(total / len(steps))