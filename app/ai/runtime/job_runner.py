from app.ai.registry.model_registry import ModelRegistry
from app.ai.runtime.task_context import TaskContext
from app.ai.runtime.gpu_coordinator import claim_gpu_slot
from app.ai.tasks.action_spotting.task import ActionSpottingPipelineTask
from app.ai.tasks.dummy_analysis.task import DummyAnalysisTask
from app.ai.tasks.full_match_analysis.task import FullMatchAnalysisTask
from app.ai.tasks.player_tracking.task import PlayerTrackingTask
from app.db.session import BackgroundSessionLocal
from app.domains.analysis.job_types import (
    BALL_TRACKING,
    FULL_MATCH_ANALYSIS,
    HIGHLIGHT_SPOTTING,
    PLAYER_TRACKING,
    TIMELINE_FUSION,
)
from app.domains.analysis.repository import AnalysisJobRepository
from app.domains.action_spotting.errors import ActionSpottingError
from app.domains.action_spotting.failure import attach_action_spotting_failure


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
        db = BackgroundSessionLocal()

        try:
            repository = AnalysisJobRepository(db)
            if not repository.claim_queued(analysis_job_id):
                db.rollback()
                return
            db.commit()
            job = repository.get_by_id(analysis_job_id)
            if job is None:
                return

            context = TaskContext.create(
                db=db,
                job=job,
                model_registry=self.model_registry,
            )

            if job.job_type == HIGHLIGHT_SPOTTING:
                options = dict(job.options or {})
                options["action_spotting_state"] = "ACTION_SPOTTING_RUNNING"
                history = list(
                    options.get("action_spotting_state_history") or []
                )
                if not history or history[-1] != "ACTION_SPOTTING_RUNNING":
                    history.append("ACTION_SPOTTING_RUNNING")
                options["action_spotting_state_history"] = history
                job.options = options
            context.mark_job_running()

            task = self._build_task(job.job_type)
            if job.job_type in {
                FULL_MATCH_ANALYSIS,
                HIGHLIGHT_SPOTTING,
                PLAYER_TRACKING,
            }:
                with claim_gpu_slot():
                    task.execute(context)
            else:
                task.execute(context)

            context.mark_job_completed()

        except Exception as exc:
            db.rollback()
            self._mark_failed_safely(
                db=db,
                analysis_job_id=analysis_job_id,
                error=exc,
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
        error: Exception,
    ) -> None:
        try:
            repository = AnalysisJobRepository(db)
            job = repository.get_by_id(analysis_job_id)

            if job is None:
                return

            if isinstance(error, ActionSpottingError):
                attach_action_spotting_failure(db, job, error)
                error_message = error.public_message
            else:
                error_message = str(error)

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
