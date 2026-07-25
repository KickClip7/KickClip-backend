from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.domains.artifact.repository import ArtifactRepository
from app.domains.clip_plan.model import ClipPlan
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.render.ffmpeg_renderer import FFmpegRenderer
from app.domains.render.model import RenderJob
from app.domains.render.repository import RenderJobRepository
from app.domains.render.schema import RenderCreateRequest
from app.domains.render.template_builder import RenderTemplateBuilder
from app.storage.local_storage import LocalStorage


class RenderJobService:
    def __init__(self, db: Session):
        self.db = db
        self.render_job_repository = RenderJobRepository(db)
        self.clip_plan_repository = ClipPlanRepository(db)
        self.media_asset_repository = MediaAssetRepository(db)
        self.artifact_repository = ArtifactRepository(db)
        self.storage = LocalStorage()
        self.template_builder = RenderTemplateBuilder()
        self.renderer = FFmpegRenderer()

    def create_render_job(self, data: RenderCreateRequest) -> RenderJob:
        clip_plan = self.clip_plan_repository.get_by_id(data.clip_plan_id)
        if clip_plan is None:
            raise ValueError("ClipPlan not found")

        merged_options = self._merge_options(
            clip_plan_options=clip_plan.options or {},
            request_options=data.options.model_dump(exclude_unset=True),
        )

        ratio = merged_options.get("ratio") or "9:16"
        quality = merged_options.get("quality") or "1080p"
        template = self.template_builder.build(ratio=ratio, quality=quality)
        captions_enabled = bool(merged_options.get("captions_enabled") or False)
        music_asset_id = merged_options.get("music_asset_id")

        try:
            render_job = self.render_job_repository.create(
                clip_plan_id=clip_plan.clip_plan_id,
                ratio=template.ratio,
                resolution=template.resolution_label,
                quality=template.quality,
                captions_enabled=captions_enabled,
                music_asset_id=music_asset_id,
                options={
                    **merged_options,
                    "template": {
                        "ratio": template.ratio,
                        "quality": template.quality,
                        "width": template.width,
                        "height": template.height,
                        "resolution": template.resolution_label,
                    },
                },
            )
            self.db.commit()
            self.db.refresh(render_job)
            return render_job
        except Exception:
            self.db.rollback()
            raise

    def get_render_job(self, render_job_id: str) -> RenderJob | None:
        return self.render_job_repository.get_by_id(render_job_id)

    def run_render_job(self, render_job_id: str) -> None:
        started = time.perf_counter()
        render_job = self.render_job_repository.get_by_id(render_job_id)
        if render_job is None:
            return

        try:
            self.render_job_repository.mark_running(render_job)
            self.db.commit()

            clip_plan = self._get_clip_plan_or_raise(render_job.clip_plan_id)
            match_id = clip_plan.project.match_id
            project_id = clip_plan.project_id
            source_asset = self._select_source_video_asset(match_id)
            source_video_path = self.storage.resolve_path(source_asset.file_path)

            template = self.template_builder.build(
                ratio=render_job.ratio,
                quality=render_job.quality,
            )
            output_dir = self._build_output_dir(
                match_id=match_id,
                project_id=project_id,
                render_job_id=render_job.render_job_id,
            )

            self.render_job_repository.update_progress(render_job, 20)
            self.db.commit()

            result = self.renderer.render_clip_plan(
                clip_plan=clip_plan,
                source_video_path=source_video_path,
                output_dir=output_dir,
                template=template,
                title=(render_job.options or {}).get("title") or clip_plan.summary,
                captions_enabled=render_job.captions_enabled,
            )

            self.render_job_repository.update_progress(render_job, 85)
            self.db.commit()

            output_artifact = self.artifact_repository.create(
                match_id=match_id,
                project_id=project_id,
                analysis_job_id=None,
                artifact_type="RENDERED_VIDEO",
                file_path=self._to_project_relative_path(result.output_path),
                mime_type="video/mp4",
                metadata_={
                    "render_job_id": render_job.render_job_id,
                    "clip_plan_id": clip_plan.clip_plan_id,
                    "ratio": render_job.ratio,
                    "quality": render_job.quality,
                    "resolution": render_job.resolution,
                    "command_log": result.command_log,
                },
            )

            subtitle_artifact_id = None
            if result.subtitle_path is not None:
                subtitle_artifact = self.artifact_repository.create(
                    match_id=match_id,
                    project_id=project_id,
                    analysis_job_id=None,
                    artifact_type="SUBTITLE_FILE",
                    file_path=self._to_project_relative_path(result.subtitle_path),
                    mime_type="application/x-subrip",
                    metadata_={
                        "render_job_id": render_job.render_job_id,
                        "clip_plan_id": clip_plan.clip_plan_id,
                    },
                )
                subtitle_artifact_id = subtitle_artifact.artifact_id

            runtime_sec = round(time.perf_counter() - started, 3)
            self.render_job_repository.mark_completed(
                render_job=render_job,
                output_artifact_id=output_artifact.artifact_id,
                subtitle_artifact_id=subtitle_artifact_id,
                runtime_sec=runtime_sec,
            )
            self.db.commit()

        except Exception as exc:
            self.db.rollback()
            runtime_sec = round(time.perf_counter() - started, 3)
            failed_job = self.render_job_repository.get_by_id(render_job_id)
            if failed_job is not None:
                self.render_job_repository.mark_failed(
                    render_job=failed_job,
                    error_message=str(exc),
                    runtime_sec=runtime_sec,
                )
                self.db.commit()

    def get_output_artifact_path(self, render_job_id: str) -> tuple[RenderJob, Path] | None:
        render_job = self.render_job_repository.get_by_id(render_job_id)
        if render_job is None or render_job.output_artifact_id is None:
            return None

        artifact = self.artifact_repository.get_by_id(render_job.output_artifact_id)
        if artifact is None:
            return None

        file_path = self.storage.resolve_path(artifact.file_path)
        return render_job, file_path

    @staticmethod
    def _merge_options(
        *,
        clip_plan_options: dict,
        request_options: dict,
    ) -> dict:
        export_options = dict(clip_plan_options.get("export_options") or {})
        merged = {
            **clip_plan_options,
            **export_options,
            **request_options,
        }
        # 빈 문자열이 들어온 경우 기존 옵션을 망가뜨리지 않도록 제거한다.
        return {key: value for key, value in merged.items() if value is not None}

    def _get_clip_plan_or_raise(self, clip_plan_id: str) -> ClipPlan:
        clip_plan = self.clip_plan_repository.get_by_id(clip_plan_id)
        if clip_plan is None:
            raise ValueError("ClipPlan not found")
        return clip_plan

    def _select_source_video_asset(self, match_id: str) -> MediaAsset:
        assets = self.media_asset_repository.list_by_match(match_id)

        for asset_type in ["RAW_VIDEO", "RAW_VIDEO_HALF1", "WEB_PREVIEW_VIDEO"]:
            asset = next((row for row in assets if row.asset_type == asset_type), None)
            if asset is not None:
                return asset

        raise ValueError("No source video asset found for render. RAW_VIDEO is required.")

    def _build_output_dir(
        self,
        *,
        match_id: str,
        project_id: str,
        render_job_id: str,
    ) -> Path:
        return (
            self.storage.storage_root
            / "matches"
            / match_id
            / "projects"
            / project_id
            / "renders"
            / render_job_id
        )

    def _to_project_relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.storage.project_root).as_posix()
        except ValueError:
            return path.as_posix()
