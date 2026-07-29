from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.domains.highlight.model import (
    EarlierAnchorProposal,
    HighlightDraft,
    HighlightRevision,
    PlayerFocusSubject,
    ScenePlayerCandidate,
    SceneTargetSelection,
    SceneTargetSelectionReference,
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

    def get_revision_for_update(
        self,
        revision_id: str,
    ) -> HighlightRevision | None:
        stmt = (
            select(HighlightRevision)
            .options(
                selectinload(HighlightRevision.focus_subject),
                selectinload(HighlightRevision.action_spotting_job),
                selectinload(HighlightRevision.render_job),
            )
            .where(HighlightRevision.revision_id == revision_id)
            .with_for_update()
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

    def latest_by_project_ids(
        self,
        project_ids: list[str],
    ) -> dict[str, HighlightRevision]:
        if not project_ids:
            return {}
        ranked = (
            select(
                HighlightRevision.revision_id.label("revision_id"),
                func.row_number()
                .over(
                    partition_by=HighlightRevision.project_id,
                    order_by=(
                        HighlightRevision.revision_number.desc(),
                        HighlightRevision.created_at.desc(),
                    ),
                )
                .label("row_number"),
            )
            .where(HighlightRevision.project_id.in_(project_ids))
            .subquery()
        )
        stmt = (
            select(HighlightRevision)
            .options(
                selectinload(HighlightRevision.action_spotting_job),
                selectinload(HighlightRevision.render_job),
            )
            .join(ranked, ranked.c.revision_id == HighlightRevision.revision_id)
            .where(ranked.c.row_number == 1)
        )
        rows = list(self.db.scalars(stmt).all())
        return {row.project_id: row for row in rows}

    def next_revision_number(self, project_id: str) -> int:
        stmt = select(func.max(HighlightRevision.revision_number)).where(
            HighlightRevision.project_id == project_id
        )
        current = self.db.scalar(stmt)
        return int(current or 0) + 1

    def get_draft(self, project_id: str) -> HighlightDraft | None:
        return self.db.scalar(
            select(HighlightDraft).where(
                HighlightDraft.project_id == project_id
            )
        )

    def get_draft_for_update(self, project_id: str) -> HighlightDraft | None:
        return self.db.scalar(
            select(HighlightDraft)
            .where(HighlightDraft.project_id == project_id)
            .with_for_update()
        )

    def create_draft(self, **kwargs) -> HighlightDraft:
        draft = HighlightDraft(**kwargs)
        self.db.add(draft)
        self.db.flush()
        return draft

    def delete_draft(self, draft: HighlightDraft) -> None:
        self.db.delete(draft)
        self.db.flush()

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

    def get_binding_for_update(
        self,
        binding_id: str,
    ) -> SceneTrackingBinding | None:
        return self.db.scalar(
            select(SceneTrackingBinding)
            .where(SceneTrackingBinding.binding_id == binding_id)
            .with_for_update()
        )

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

    def next_target_selection_revision(
        self,
        *,
        revision_id: str,
        scene_id: str,
    ) -> int:
        current = self.db.scalar(
            select(func.max(SceneTargetSelection.selection_revision)).where(
                SceneTargetSelection.revision_id == revision_id,
                SceneTargetSelection.scene_id == scene_id,
            )
        )
        return int(current or 0) + 1

    def create_target_selection(self, **kwargs) -> SceneTargetSelection:
        selection = SceneTargetSelection(**kwargs)
        self.db.add(selection)
        self.db.flush()
        return selection

    def get_target_selection(
        self,
        selection_id: str,
    ) -> SceneTargetSelection | None:
        return self.db.get(SceneTargetSelection, selection_id)

    def get_target_selection_for_update(
        self,
        selection_id: str,
    ) -> SceneTargetSelection | None:
        return self.db.scalar(
            select(SceneTargetSelection)
            .where(SceneTargetSelection.selection_id == selection_id)
            .with_for_update()
        )

    def list_target_selections(
        self,
        *,
        revision_id: str,
        scene_id: str | None = None,
    ) -> list[SceneTargetSelection]:
        stmt = select(SceneTargetSelection).where(
            SceneTargetSelection.revision_id == revision_id
        )
        if scene_id is not None:
            stmt = stmt.where(SceneTargetSelection.scene_id == scene_id)
        return list(
            self.db.scalars(
                stmt.order_by(SceneTargetSelection.selection_revision.asc())
            ).all()
        )

    def create_target_reference(
        self,
        **kwargs,
    ) -> SceneTargetSelectionReference:
        reference = SceneTargetSelectionReference(**kwargs)
        self.db.add(reference)
        self.db.flush()
        return reference

    def create_earlier_anchor_proposal(
        self,
        **kwargs,
    ) -> EarlierAnchorProposal:
        proposal = EarlierAnchorProposal(**kwargs)
        self.db.add(proposal)
        self.db.flush()
        return proposal

    def list_earlier_anchor_proposals(
        self,
        selection_id: str,
    ) -> list[EarlierAnchorProposal]:
        return list(
            self.db.scalars(
                select(EarlierAnchorProposal)
                .where(EarlierAnchorProposal.selection_id == selection_id)
                .order_by(EarlierAnchorProposal.retrieval_rank.asc())
            ).all()
        )
