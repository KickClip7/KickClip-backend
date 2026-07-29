from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.signed_url import build_signed_artifact_url
from app.domains.analysis.model import AnalysisJob
from app.domains.analysis.repository import AnalysisJobRepository
from app.domains.clip_plan.model import ClipPlan
from app.domains.highlight.model import HighlightRevision
from app.domains.highlight.repository import HighlightRepository
from app.domains.project.model import Project
from app.domains.project.repository import ProjectRepository
from app.domains.project.schema import (
    ProjectCreate,
    ProjectRead,
    ProjectRecentItem,
    ProjectUpdate,
)
from app.domains.match.repository import MatchRepository
from app.domains.artifact.repository import ArtifactRepository
from app.domains.render.model import RenderJob


PROJECT_THUMBNAIL_TYPES = {
    "PROJECT_THUMBNAIL",
    "HIGHLIGHT_THUMBNAIL",
    "THUMBNAIL",
    "AI_RECOMMENDED_THUMBNAIL",
}


class ProjectService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = ProjectRepository(db)
        self.match_repository = MatchRepository(db)
        self.artifact_repository = ArtifactRepository(db)
        self.highlight_repository = HighlightRepository(db)
        self.analysis_repository = AnalysisJobRepository(db)

    def create_project(
        self,
        data: ProjectCreate,
        *,
        match_id: str,
        owner_id: str,
    ) -> Project:
        match = self.match_repository.get_by_id(match_id)
        if match is None:
            raise ValueError("Match not found")
        if match.owner_id != owner_id:
            raise ValueError("Match owner does not match project owner")
        project = self.repository.create(
            **data.model_dump(),
            match_id=match_id,
            owner_id=owner_id,
        )
        self.db.commit()
        self.db.refresh(project)
        return project

    def list_match_projects(
        self,
        match_id: str,
        *,
        user_id: str | None = None,
    ) -> list[ProjectRead]:
        projects = self.repository.list_by_match(match_id)
        revisions = self.highlight_repository.latest_by_project_ids(
            [project.project_id for project in projects]
        )
        analysis_jobs = self.analysis_repository.latest_by_match_ids(
            list({project.match_id for project in projects})
        )
        return [
            self._project_read(
                project,
                user_id=user_id,
                revision=revisions.get(project.project_id),
                analysis_job=analysis_jobs.get(project.match_id),
            )
            for project in projects
        ]

    def get_project(self, project_id: str) -> Project | None:
        return self.repository.get_by_id(project_id)

    def list_recent_projects(
        self,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> list[Project]:
        return self.repository.list_recent(limit=limit, owner_id=owner_id)

    def list_recent_project_cards(
        self,
        limit: int = 10,
        owner_id: str | None = None,
        user_id: str | None = None,
    ) -> list[ProjectRecentItem]:
        projects = self.repository.list_recent(limit=limit, owner_id=owner_id)
        revisions = self.highlight_repository.latest_by_project_ids(
            [project.project_id for project in projects]
        )
        analysis_jobs = self.analysis_repository.latest_by_match_ids(
            list({project.match_id for project in projects})
        )
        return [
            self._recent_item(
                project,
                user_id=user_id,
                revision=revisions.get(project.project_id),
                analysis_job=analysis_jobs.get(project.match_id),
            )
            for project in projects
        ]

    def get_project_read(
        self,
        project_id: str,
        *,
        user_id: str | None = None,
        touch: bool = False,
    ) -> ProjectRead | None:
        project = self.repository.get_by_id(project_id)
        if project is None:
            return None
        if touch:
            project.last_opened_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(project)
        revision = self.highlight_repository.latest_by_project_ids(
            [project.project_id]
        ).get(project.project_id)
        analysis_job = self.analysis_repository.latest_by_match_ids(
            [project.match_id]
        ).get(project.match_id)
        return self._project_read(
            project,
            user_id=user_id,
            revision=revision,
            analysis_job=analysis_job,
        )

    def touch_last_opened(self, project_id: str) -> None:
        project = self.repository.get_by_id(project_id)
        if project is None:
            return
        project.last_opened_at = datetime.now(timezone.utc)
        self.db.commit()

    def update_project(self, project_id: str, data: ProjectUpdate) -> Project | None:
        project = self.repository.get_by_id(project_id)
        if project is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        thumbnail_artifact_id = update_data.get("thumbnail_artifact_id")
        if thumbnail_artifact_id is not None:
            artifact = self.artifact_repository.get_by_id(thumbnail_artifact_id)
            if (
                artifact is None
                or artifact.match_id != project.match_id
                or artifact.project_id != project.project_id
                or not (artifact.mime_type or "").lower().startswith("image/")
            ):
                raise ValueError(
                    "Thumbnail artifact does not belong to this project"
                )
        for key, value in update_data.items():
            setattr(project, key, value)

        self.db.commit()
        self.db.refresh(project)
        return project

    def _project_read(
        self,
        project: Project,
        *,
        user_id: str | None,
        revision: HighlightRevision | None,
        analysis_job: AnalysisJob | None,
    ) -> ProjectRead:
        aggregate = self._aggregate(
            project,
            user_id=user_id,
            revision=revision,
            analysis_job=analysis_job,
        )
        base = ProjectRead.model_validate(project)
        return base.model_copy(update=aggregate)

    def _recent_item(
        self,
        project: Project,
        *,
        user_id: str | None,
        revision: HighlightRevision | None,
        analysis_job: AnalysisJob | None,
    ) -> ProjectRecentItem:
        aggregate = self._aggregate(
            project,
            user_id=user_id,
            revision=revision,
            analysis_job=analysis_job,
        )
        return ProjectRecentItem(
            project_id=project.project_id,
            match_id=project.match_id,
            title=project.title,
            status=aggregate["status"],
            thumbnail_url=aggregate["thumbnail_url"],
            duration_sec=aggregate["duration_sec"],
            clip_count=aggregate["clip_count"],
            edit_mode=aggregate["edit_mode"],
            progress=aggregate["progress"],
            ratio=aggregate["ratio"],
            last_opened_at=project.last_opened_at,
            created_at=project.created_at,
        )

    def _aggregate(
        self,
        project: Project,
        *,
        user_id: str | None,
        revision: HighlightRevision | None,
        analysis_job: AnalysisJob | None,
    ) -> dict:
        valid_plans = [plan for plan in project.clip_plans if plan.items]
        latest_plan = self._latest(valid_plans)
        render_jobs = [
            render
            for plan in project.clip_plans
            for render in plan.render_jobs
        ]
        latest_render = self._latest(render_jobs)
        thumbnail = self._select_thumbnail(project)
        thumbnail_url = None
        if thumbnail is not None:
            thumbnail_url = (
                build_signed_artifact_url(thumbnail.artifact_id, user_id)[0]
                if user_id is not None
                else f"/api/v1/artifacts/{thumbnail.artifact_id}/download"
            )
        ratio = self._ratio(latest_plan, latest_render)
        duration = None
        clip_count = 0
        edit_mode = None
        if latest_plan is not None:
            clip_count = len(latest_plan.items)
            duration = latest_plan.actual_duration_sec
            if duration is None:
                duration = round(
                    sum(float(item.duration_sec) for item in latest_plan.items),
                    3,
                )
            edit_mode = latest_plan.mode
        status, progress = self._status_and_progress(
            project=project,
            latest_plan=latest_plan,
            latest_render=latest_render,
            revision=revision,
            analysis_job=analysis_job,
        )
        return {
            "status": status,
            "thumbnail_url": thumbnail_url,
            "duration_sec": duration,
            "clip_count": clip_count,
            "edit_mode": edit_mode,
            "progress": progress,
            "ratio": ratio,
        }

    @staticmethod
    def _latest(rows):
        if not rows:
            return None
        return max(
            rows,
            key=lambda row: (
                row.updated_at or row.created_at,
                row.created_at,
            ),
        )

    @staticmethod
    def _ratio(
        clip_plan: ClipPlan | None,
        render_job: RenderJob | None,
    ) -> str | None:
        if clip_plan is not None:
            options = clip_plan.options or {}
            export_options = options.get("export_options") or {}
            value = export_options.get("ratio") or options.get("ratio")
            if value:
                return str(value)
        return render_job.ratio if render_job is not None else None

    @staticmethod
    def _select_thumbnail(project: Project) -> Artifact | None:
        images = [
            artifact
            for artifact in project.artifacts
            if (artifact.mime_type or "").lower().startswith("image/")
        ]
        if project.thumbnail_artifact_id:
            selected = next(
                (
                    artifact
                    for artifact in images
                    if artifact.artifact_id == project.thumbnail_artifact_id
                ),
                None,
            )
            if selected is not None:
                return selected
        eligible = [
            artifact
            for artifact in images
            if artifact.artifact_type in PROJECT_THUMBNAIL_TYPES
        ]
        return ProjectService._latest(eligible)

    @staticmethod
    def _status_and_progress(
        *,
        project: Project,
        latest_plan: ClipPlan | None,
        latest_render: RenderJob | None,
        revision: HighlightRevision | None,
        analysis_job: AnalysisJob | None,
    ) -> tuple[str, int]:
        if revision is not None:
            status = revision.status
            render = revision.render_job
            if render is not None:
                if render.status == "completed":
                    return "COMPLETED", 100
                if render.status == "failed":
                    return "FAILED", min(
                        99,
                        85 + round(render.progress * 0.14),
                    )
                if render.status in {"queued", "running"}:
                    return "RENDERING", min(
                        99,
                        85 + round(render.progress * 0.14),
                    )
            if status == "FAILED":
                return "FAILED", 0
            if status == "COMPLETED":
                return "COMPLETED", 100
            if status == "CLIP_PLAN_READY":
                return status, 85
            if status in {
                "TRACKING_RUNNING",
                "TRACKING_CONFIRMATION_REQUIRED",
                "NO_TARGET_SCENES",
            }:
                return status, 65
            if status in {
                "PLAYER_DISCOVERY_RUNNING",
                "PLAYER_SELECTION_REQUIRED",
            }:
                return status, 50
            if status in {"SCENES_SELECTED", "SCENES_READY"}:
                return status, 35
            if status == "ACTION_SPOTTING_RUNNING":
                action_progress = (
                    revision.action_spotting_job.progress
                    if revision.action_spotting_job is not None
                    else 0
                )
                return status, min(34, round(action_progress * 0.34))
            return status, 10
        if latest_render is not None:
            if latest_render.status == "completed":
                return "COMPLETED", 100
            if latest_render.status == "failed":
                return "FAILED", min(99, 85 + round(latest_render.progress * 0.14))
            if latest_render.status in {"queued", "running"}:
                return "RENDERING", min(
                    99,
                    85 + round(latest_render.progress * 0.14),
                )
        if latest_plan is not None:
            return "CLIP_PLAN_READY", 85
        if analysis_job is not None:
            if analysis_job.status == "FAILED":
                return "FAILED", analysis_job.progress
            if analysis_job.status in {"QUEUED", "RUNNING"}:
                return "ANALYZING", analysis_job.progress
            if analysis_job.status == "COMPLETED":
                return project.status, 30
        return project.status, 0
