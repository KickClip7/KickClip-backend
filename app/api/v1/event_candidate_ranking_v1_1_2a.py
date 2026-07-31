from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_project_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1_2a.backend_adapter import (
    EventCandidateRankingV112aBackendAdapter,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.schema import (
    EventCandidateRankingV112aRequest,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a_integration import (
    EVENT_RANKING_V1_1_2A,
)
from app.domains.highlight.scene_ai_task import (
    SceneAITaskService,
    get_scene_ai_task_executor,
)
from app.domains.highlight.schema import SceneAITaskRead


router = APIRouter()


@router.post(
    "/projects/{project_id}/highlight/revisions/{revision_id}"
    "/event-candidate-rankings/v1.1.2a",
    response_model=SceneAITaskRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run Event Candidate Ranking V1.1.2a compatibility hotfix",
)
def create_event_candidate_ranking_v1_1_2a(
    project_id: str,
    revision_id: str,
    payload: EventCandidateRankingV112aRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SceneAITaskRead:
    project = require_project_access(db, project_id, current_user)
    adapter = EventCandidateRankingV112aBackendAdapter(db)
    try:
        resolved, freeze_material = adapter.prepare(
            project=project,
            revision_id=revision_id,
            event_id=payload.event_id,
            scene_id=payload.scene_id,
            shortlist_size=payload.shortlist_size,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    task_service = SceneAITaskService(db)
    task, _ = task_service.enqueue(
        user=current_user,
        project=project,
        task_type=EVENT_RANKING_V1_1_2A,
        payload={
            "revision_id": revision_id,
            "shortlist_size": payload.shortlist_size,
            "resolved_event": resolved.to_dict(),
            "freeze_material": freeze_material,
        },
    )
    if task.status == "QUEUED":
        get_scene_ai_task_executor().submit(task.task_id)
    return task_service.read(task)

