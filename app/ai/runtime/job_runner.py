from app.ai.registry.model_registry import ModelRegistry
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.action_spotting.task import ActionSpottingPipelineTask
from app.ai.tasks.dummy_analysis.task import DummyAnalysisTask
from app.ai.tasks.full_match_analysis.task import FullMatchAnalysisTask
from app.ai.tasks.player_tracking.task import PlayerTrackingTask
from app.db.session import SessionLocal
from app.domains.analysis.job_types import (
    BALL_TRACKING,
    FULL_MATCH_ANALYSIS,
    HIGHLIGHT_SPOTTING,
    PLAYER_TRACKING,
    TIMELINE_FUSION,
)
from app.domains.analysis.repository import AnalysisJobRepository


class JobRunner:
    """Run AnalysisJob by job_type.

    현재 실행 구성:
    - FULL_MATCH_ANALYSIS: feature/highlight/player 파이프라인
    - HIGHLIGHT_SPOTTING: ActionSpottingPipelineTask
    - PLAYER_TRACKING: PlayerTrackingTask
    - BALL_TRACKING / TIMELINE_FUSION: 아직 dummy
    """

    def __init__(self, model_registry: ModelRegistry | None = None):
        self.model_registry = model_registry or ModelRegistry()

    def run(self, analysis_job_id: str) -> None:
        db = SessionLocal()

        try:
            repository = AnalysisJobRepository(db)
            job = repository.get_by_id(analysis_job_id)

            if job is None:
                return

            context = TaskContext.create(
                db=db,
                job=job,
                model_registry=self.model_registry,
            )

            context.mark_job_running()

            task = self._build_task(job.job_type)
            task.execute(context)

            context.mark_job_completed()

        except Exception as exc:
            db.rollback()
            self._mark_failed_safely(
                db=db,
                analysis_job_id=analysis_job_id,
                error_message=str(exc),
            )
        finally:
            db.close()

    def _build_task(self, job_type: str):
        if job_type == FULL_MATCH_ANALYSIS:
            return FullMatchAnalysisTask()

        if job_type == HIGHLIGHT_SPOTTING:
            return ActionSpottingPipelineTask()

        if job_type == PLAYER_TRACKING:
            return PlayerTrackingTask()

        if job_type == BALL_TRACKING:
            return DummyAnalysisTask()

        if job_type == TIMELINE_FUSION:
            return DummyAnalysisTask()

        raise ValueError(f"Unsupported job_type for JobRunner: {job_type}")

    def _mark_failed_safely(
        self,
        db,
        analysis_job_id: str,
        error_message: str,
    ) -> None:
        try:
            repository = AnalysisJobRepository(db)
            job = repository.get_by_id(analysis_job_id)

            if job is None:
                return

            context = TaskContext.create(
                db=db,
                job=job,
                model_registry=self.model_registry,
            )
            context.mark_job_failed(error_message)

        except Exception:
            db.rollback()


def run_analysis_job_background(analysis_job_id: str) -> None:
    runner = JobRunner()
    runner.run(analysis_job_id)
