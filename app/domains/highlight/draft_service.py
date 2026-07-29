from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.clip_plan.model import ClipPlan
from app.domains.highlight.model import HighlightDraft, HighlightRevision
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.schema import (
    HighlightDraftRead,
    HighlightDraftSaveRequest,
    HighlightDraftState,
)
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository


class HighlightDraftVersionConflict(ValueError):
    def __init__(self, *, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Highlight draft version conflict: expected {expected}, actual {actual}."
        )


class HighlightDraftService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = HighlightRepository(db)
        self.timeline_repository = TimelineEventRepository(db)

    def get(self, project_id: str) -> HighlightDraft | None:
        return self.repository.get_draft(project_id)

    def save(
        self,
        *,
        project: Project,
        user_id: str,
        data: HighlightDraftSaveRequest,
    ) -> HighlightDraft:
        # Serialize concurrent first-save requests as well as updates.
        self.db.scalar(
            select(Project)
            .where(Project.project_id == project.project_id)
            .with_for_update()
        )
        revision = self._validate_revision(project.project_id, data.revision_id)
        self._validate_clip_plan(project.project_id, data.clip_plan_id)
        self._validate_state(
            project=project,
            revision=revision,
            state=data,
        )

        draft = self.repository.get_draft_for_update(project.project_id)
        if draft is None:
            if data.expected_version not in {None, 0}:
                raise HighlightDraftVersionConflict(
                    expected=data.expected_version,
                    actual=0,
                )
            draft = self.repository.create_draft(
                project_id=project.project_id,
                revision_id=data.revision_id,
                clip_plan_id=data.clip_plan_id,
                saved_by_user_id=user_id,
                version=1,
                state=self._state_payload(data),
            )
        else:
            if (
                data.expected_version is not None
                and data.expected_version != draft.version
            ):
                raise HighlightDraftVersionConflict(
                    expected=data.expected_version,
                    actual=draft.version,
                )
            draft.revision_id = data.revision_id
            draft.clip_plan_id = data.clip_plan_id
            draft.saved_by_user_id = user_id
            draft.version += 1
            draft.state = self._state_payload(data)

        self.db.commit()
        self.db.refresh(draft)
        return draft

    def delete(self, project_id: str) -> bool:
        draft = self.repository.get_draft_for_update(project_id)
        if draft is None:
            return False
        self.repository.delete_draft(draft)
        self.db.commit()
        return True

    @staticmethod
    def read(draft: HighlightDraft) -> HighlightDraftRead:
        state = HighlightDraftState.model_validate(draft.state or {})
        return HighlightDraftRead(
            draft_id=draft.draft_id,
            project_id=draft.project_id,
            revision_id=draft.revision_id,
            clip_plan_id=draft.clip_plan_id,
            saved_by_user_id=draft.saved_by_user_id,
            version=draft.version,
            created_at=draft.created_at,
            updated_at=draft.updated_at,
            **state.model_dump(),
        )

    def _validate_revision(
        self,
        project_id: str,
        revision_id: str | None,
    ) -> HighlightRevision | None:
        if revision_id is None:
            return None
        revision = self.repository.get_revision(revision_id)
        if revision is None or revision.project_id != project_id:
            raise ValueError("Highlight revision does not belong to this project.")
        return revision

    def _validate_clip_plan(
        self,
        project_id: str,
        clip_plan_id: str | None,
    ) -> None:
        if clip_plan_id is None:
            return
        clip_plan = self.db.get(ClipPlan, clip_plan_id)
        if clip_plan is None or clip_plan.project_id != project_id:
            raise ValueError("Clip plan does not belong to this project.")

    def _validate_state(
        self,
        *,
        project: Project,
        revision: HighlightRevision | None,
        state: HighlightDraftState,
    ) -> None:
        referenced_ids = set(state.selected_scene_ids)
        referenced_ids.update(item.scene_id for item in state.timeline_items)
        events: dict[str, TimelineEvent] = {}
        for scene_id in referenced_ids:
            event = self.timeline_repository.get_by_id(scene_id)
            if event is None or event.match_id != project.match_id:
                raise ValueError(
                    "Draft scene does not belong to this project's match."
                )
            if (
                revision is not None
                and revision.action_spotting_job_id is not None
                and event.source_job_id != revision.action_spotting_job_id
            ):
                raise ValueError(
                    "Draft scene does not belong to the selected revision."
                )
            events[scene_id] = event

        for item in state.timeline_items:
            event = events[item.scene_id]
            if (
                item.start_sec < event.start_sec - 0.001
                or item.end_sec > event.end_sec + 0.001
            ):
                raise ValueError(
                    "Draft timeline item must stay within its scene bounds."
                )

    @staticmethod
    def _state_payload(data: HighlightDraftSaveRequest) -> dict:
        return HighlightDraftState(
            selected_scene_ids=data.selected_scene_ids,
            timeline_items=data.timeline_items,
            export_options=data.export_options,
            editor_state=data.editor_state,
            metadata=data.metadata,
        ).model_dump(mode="json")
