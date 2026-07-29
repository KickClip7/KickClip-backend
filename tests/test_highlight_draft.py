from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import models  # noqa: F401
from app.db.base import Base
from app.domains.analysis.model import AnalysisJob
from app.domains.auth.model import User
from app.domains.highlight.draft_service import (
    HighlightDraftService,
    HighlightDraftVersionConflict,
)
from app.domains.highlight.model import HighlightRevision
from app.domains.highlight.schema import HighlightDraftSaveRequest
from app.domains.match.model import Match
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent


def _request(
    *,
    expected_version: int | None = None,
    start_sec: float = 11.0,
    editor_state: dict | None = None,
) -> HighlightDraftSaveRequest:
    return HighlightDraftSaveRequest(
        revision_id="hrev_draft",
        expected_version=expected_version,
        selected_scene_ids=["evt_draft"],
        timeline_items=[
            {
                "scene_id": "evt_draft",
                "start_sec": start_sec,
                "end_sec": 19.0,
                "order_index": 0,
                "render_strategy": "FULL_FRAME",
            }
        ],
        export_options={
            "ratio": "9:16",
            "captions_enabled": True,
        },
        editor_state=editor_state or {"playhead_sec": 12.5},
    )


def test_highlight_draft_can_be_saved_restored_versioned_and_deleted() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)

    with Session(engine, expire_on_commit=False) as db:
        user = User(
            user_id="usr_draft",
            email="draft@example.com",
            password_hash="!",
            display_name="Draft Test",
        )
        match = Match(
            match_id="match_draft",
            owner_id=user.user_id,
            duration_sec=100,
        )
        project = Project(
            project_id="proj_draft",
            match_id=match.match_id,
            owner_id=user.user_id,
            title="Draft Project",
        )
        job = AnalysisJob(
            analysis_job_id="job_draft",
            match_id=match.match_id,
            job_type="HIGHLIGHT_SPOTTING",
            status="COMPLETED",
            completed_at=now,
        )
        event = TimelineEvent(
            timeline_event_id="evt_draft",
            match_id=match.match_id,
            source_job_id=job.analysis_job_id,
            event_type="action_spotting",
            label="goal",
            timestamp_sec=15,
            start_sec=10,
            end_sec=20,
            duration_sec=10,
        )
        revision = HighlightRevision(
            revision_id="hrev_draft",
            project_id=project.project_id,
            revision_number=1,
            action_spotting_job_id=job.analysis_job_id,
            user_request="골 장면",
            structured_request={},
            selected_scene_ids=[event.timeline_event_id],
            scene_selection=[],
            status="SCENES_SELECTED",
            options={},
        )
        db.add_all([user, match, project, job, event, revision])
        db.commit()

        service = HighlightDraftService(db)
        created = service.save(
            project=project,
            user_id=user.user_id,
            data=_request(),
        )
        assert created.version == 1
        restored = service.read(service.get(project.project_id))
        assert restored.revision_id == revision.revision_id
        assert restored.selected_scene_ids == [event.timeline_event_id]
        assert restored.editor_state == {"playhead_sec": 12.5}

        updated = service.save(
            project=project,
            user_id=user.user_id,
            data=_request(
                expected_version=1,
                editor_state={"playhead_sec": 18.0, "panel": "captions"},
            ),
        )
        assert updated.draft_id == created.draft_id
        assert updated.version == 2
        assert service.read(updated).editor_state["panel"] == "captions"

        with pytest.raises(HighlightDraftVersionConflict) as conflict:
            service.save(
                project=project,
                user_id=user.user_id,
                data=_request(expected_version=1),
            )
        assert conflict.value.actual == 2
        db.rollback()

        with pytest.raises(ValueError, match="scene bounds"):
            service.save(
                project=project,
                user_id=user.user_id,
                data=_request(expected_version=2, start_sec=9.0),
            )
        db.rollback()

        assert service.delete(project.project_id) is True
        assert service.get(project.project_id) is None
        assert service.delete(project.project_id) is False

    engine.dispose()
