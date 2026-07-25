from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.domains.auth.access import require_clip_plan_access, require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.agent.schema import (
    AgentClipPlanRequest,
    AgentClipPlanResponse,
    ExportAssistantRequest,
    ExportAssistantResponse,
    ExportMetadataRecommendRequest,
    ExportMetadataRecommendResponse,
    PlayerProfileResolveRequest,
    PlayerProfileResolveResponse,
)
from app.domains.agent.export_workflow_agent import (
    ExportAgentConfigurationError,
    LangGraphExportAgent,
)
from app.domains.agent.qwen_export_assistant import QwenExportAssistant
from app.domains.agent.service import AgentService
from app.domains.timeline.dev_context import resolve_agent_match_id


router = APIRouter()


@router.post(
    "/export-assistant/invoke",
    response_model=ExportAssistantResponse,
    summary="GPT-4o-mini LangGraph 내보내기 Agent 실행",
)
def invoke_export_assistant(
    data: ExportAssistantRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExportAssistantResponse:
    if data.clip_plan_id:
        require_clip_plan_access(db, data.clip_plan_id, current_user)
    elif data.match_id:
        require_match_access(db, data.match_id, current_user)

    try:
        return LangGraphExportAgent(db).invoke(data)
    except ExportAgentConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"내보내기 Agent 실행에 실패했습니다: {exc}",
        ) from exc


@router.post(
    "/export-metadata/recommend",
    response_model=ExportMetadataRecommendResponse,
    summary="Qwen 기반 제목, 해시태그, 썸네일 타임스탬프 추천",
)
def recommend_export_metadata(
    data: ExportMetadataRecommendRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExportMetadataRecommendResponse:
    if data.clip_plan_id:
        require_clip_plan_access(db, data.clip_plan_id, current_user)
    elif data.match_id:
        require_match_access(db, data.match_id, current_user)

    try:
        return QwenExportAssistant(db).recommend(
            clip_plan_id=data.clip_plan_id,
            match_id=data.match_id,
            language=data.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qwen 추천 생성에 실패했습니다: {exc}",
        ) from exc


@router.post(
    "/clip-plan",
    response_model=AgentClipPlanResponse,
    summary="Agent 프롬프트 기반 ClipPlan 생성",
)
def create_agent_clip_plan(
    data: AgentClipPlanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AgentClipPlanResponse:
    match_id = resolve_agent_match_id(data.match_id, get_settings())
    require_match_access(db, match_id, current_user)
    service = AgentService(db)

    try:
        return service.create_clip_plan(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/player-profile/resolve",
    response_model=PlayerProfileResolveResponse,
    summary="선수 프로필 연결",
)
def resolve_player_profile(
    data: PlayerProfileResolveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PlayerProfileResolveResponse:
    require_match_access(db, data.match_id, current_user)
    service = AgentService(db)

    try:
        return service.resolve_player_profile(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
