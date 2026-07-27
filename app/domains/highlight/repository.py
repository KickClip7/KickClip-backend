from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.domains.highlight.model import (
    HighlightRevision,
    PlayerFocusSubject,
    ScenePlayerCandidate,
    SceneTrackingBinding,
)


class HighlightRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_revision(self, **kwargs) -> HighlightRevision:
        revision = HighlightRevision(**kwargs)
        self.db.add(revision)
        self.db.flush()
        return revision

    def get_revision(self, revision_id: str) -> HighlightRevision | None:
        stmt = (
            select(HighlightRevision)
            .options(
                selectinload(HighlightRevision.focus_subject),
                selectinload(HighlightRevision.action_spotting_job),
                selectinload(HighlightRevision.render_job),
            )
            .where(HighlightRevision.revision_id == revision_id)
        )
        return self.db.scalar(stmt)

    def get_current_revision(self, project_id: str) -> HighlightRevision | None:
        stmt = (
            select(HighlightRevision)
            .options(
                selectinload(HighlightRevision.focus_subject),
                selectinload(HighlightRevision.action_spotting_job),
                selectinload(HighlightRevision.render_job),
            )
            .where(HighlightRevision.project_id == project_id)
            .order_by(HighlightRevision.revision_number.desc())
            .limit(1)
        )
        return self.db.scalar(stmt)

    def list_revisions(self, project_id: str) -> list[HighlightRevision]:
        stmt = (
            select(HighlightRevision)
            .where(HighlightRevision.project_id == project_id)
            .order_by(HighlightRevision.revision_number.asc())
        )
        return list(self.db.scalars(stmt).all())

    def next_revision_number(self, project_id: str) -> int:
        stmt = select(func.max(HighlightRevision.revision_number)).where(
            HighlightRevision.project_id == project_id
        )
        current = self.db.scalar(stmt)
        return int(current or 0) + 1

    def create_focus_subject(self, **kwargs) -> PlayerFocusSubject:
        subject = PlayerFocusSubject(**kwargs)
        self.db.add(subject)
        self.db.flush()
        return subject

    def get_focus_subject(self, focus_subject_id: str) -> PlayerFocusSubject | None:
        return self.db.get(PlayerFocusSubject, focus_subject_id)

    def create_candidate(self, **kwargs) -> ScenePlayerCandidate:
        candidate = ScenePlayerCandidate(**kwargs)
        self.db.add(candidate)
        self.db.flush()
        return candidate

    def list_candidates(
        self,
        revision_id: str,
        scene_id: str | None = None,
    ) -> list[ScenePlayerCandidate]:
        stmt = select(ScenePlayerCandidate).where(
            ScenePlayerCandidate.revision_id == revision_id
        )
        if scene_id is not None:
            stmt = stmt.where(ScenePlayerCandidate.scene_id == scene_id)
        stmt = stmt.order_by(
            ScenePlayerCandidate.scene_id.asc(),
            ScenePlayerCandidate.trackability_score.desc(),
        )
        return list(self.db.scalars(stmt).all())

    def get_candidate(
        self,
        *,
        revision_id: str,
        candidate_id: str,
    ) -> ScenePlayerCandidate | None:
        stmt = select(ScenePlayerCandidate).where(
            ScenePlayerCandidate.revision_id == revision_id,
            ScenePlayerCandidate.candidate_id == candidate_id,
        )
        return self.db.scalar(stmt)

    def create_binding(self, **kwargs) -> SceneTrackingBinding:
        binding = SceneTrackingBinding(**kwargs)
        self.db.add(binding)
        self.db.flush()
        return binding

    def get_binding(
        self,
        *,
        revision_id: str,
        scene_id: str,
    ) -> SceneTrackingBinding | None:
        stmt = select(SceneTrackingBinding).where(
            SceneTrackingBinding.revision_id == revision_id,
            SceneTrackingBinding.scene_id == scene_id,
        )
        return self.db.scalar(stmt)

    def get_binding_by_tracking_job(
        self,
        tracking_job_id: str,
    ) -> SceneTrackingBinding | None:
        stmt = select(SceneTrackingBinding).where(
            SceneTrackingBinding.tracking_job_id == tracking_job_id
        )
        return self.db.scalar(stmt)

    def list_bindings(self, revision_id: str) -> list[SceneTrackingBinding]:
        stmt = (
            select(SceneTrackingBinding)
            .options(selectinload(SceneTrackingBinding.tracking_job))
            .where(SceneTrackingBinding.revision_id == revision_id)
            .order_by(SceneTrackingBinding.created_at.asc())
        )
        return list(self.db.scalars(stmt).all())
