from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.render.model import RenderJob


class RenderJobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        clip_plan_id: str,
        ratio: str,
        resolution: str | None,
        quality: str,
        captions_enabled: bool,
        music_asset_id: str | None,
        options: dict,
    ) -> RenderJob:
        render_job = RenderJob(
            clip_plan_id=clip_plan_id,
            status="queued",
            progress=0,
            ratio=ratio,
            resolution=resolution,
            quality=quality,
            captions_enabled=captions_enabled,
            music_asset_id=music_asset_id,
            options=options,
        )
        self.db.add(render_job)
        self.db.flush()
        return render_job

    def get_by_id(self, render_job_id: str) -> RenderJob | None:
        stmt = select(RenderJob).where(RenderJob.render_job_id == render_job_id)
        return self.db.scalar(stmt)

    def list_by_clip_plan(self, clip_plan_id: str) -> list[RenderJob]:
        stmt = (
            select(RenderJob)
            .where(RenderJob.clip_plan_id == clip_plan_id)
            .order_by(RenderJob.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def mark_running(self, render_job: RenderJob) -> RenderJob:
        render_job.status = "running"
        render_job.progress = max(render_job.progress, 5)
        render_job.started_at = datetime.now(timezone.utc)
        render_job.error_message = None
        self.db.flush()
        return render_job

    def update_progress(self, render_job: RenderJob, progress: int) -> RenderJob:
        render_job.progress = max(0, min(100, progress))
        self.db.flush()
        return render_job

    def mark_completed(
        self,
        *,
        render_job: RenderJob,
        output_artifact_id: str,
        subtitle_artifact_id: str | None = None,
        runtime_sec: float | None = None,
    ) -> RenderJob:
        render_job.status = "completed"
        render_job.progress = 100
        render_job.output_artifact_id = output_artifact_id
        render_job.subtitle_artifact_id = subtitle_artifact_id
        render_job.runtime_sec = runtime_sec
        render_job.completed_at = datetime.now(timezone.utc)
        render_job.error_message = None
        self.db.flush()
        return render_job

    def mark_failed(
        self,
        *,
        render_job: RenderJob,
        error_message: str,
        runtime_sec: float | None = None,
    ) -> RenderJob:
        render_job.status = "failed"
        render_job.progress = min(render_job.progress, 99)
        render_job.error_message = error_message[:4000]
        render_job.runtime_sec = runtime_sec
        render_job.completed_at = datetime.now(timezone.utc)
        self.db.flush()
        return render_job
