from __future__ import annotations

import uuid

from langgraph.types import Command
from sqlalchemy.orm import Session

from app.ai.agents.clip_tools import get_current_state
from app.ai.agents.edit_workflow_agent import build_edit_workflow_graph
from app.core.config import get_settings
from app.domains.clip_plan.schema import (
    ClipPlanItemCreate,
    ClipPlanUpdateRequest,
    ManualClipPlanCreateRequest,
)
from app.domains.clip_plan.service import ClipPlanService
from app.domains.media.repository import MediaAssetRepository
from app.domains.media.signed_url import build_signed_media_url
from app.domains.timeline.dev_context import (
    resolve_agent_match_id,
    select_dev_timeline_events,
)
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.timeline.schema import TimelineEventRead

_VIDEO_ASSET_TYPE_PRIORITY = ("RAW_VIDEO", "RAW_VIDEO_HALF1", "WEB_PREVIEW_VIDEO")

# session_id -> match_id. 프로세스 인메모리 저장이라 서버 재시작 시 세션이 소실된다
# (로컬 데모/개발 범위에서만 사용, DB/Redis 등 영속 저장소는 쓰지 않는다).
_SESSION_MATCH_IDS: dict[str, str] = {}
_SESSION_PROJECT_IDS: dict[str, str] = {}
_SESSION_CLIP_PLAN_IDS: dict[str, str] = {}
_SESSION_USER_IDS: dict[str, str] = {}


class SessionNotFoundError(LookupError):
    pass


class SessionService:
    def __init__(self, db: Session):
        self.db = db
        self.media_assets = MediaAssetRepository(db)
        self.timeline_events = TimelineEventRepository(db)
        self.clip_plans = ClipPlanService(db)
        self.graph = build_edit_workflow_graph()

    def start(
        self,
        match_id: str,
        user_id: str,
        *,
        project_id: str | None = None,
    ) -> dict:
        settings = get_settings()
        effective_match_id = resolve_agent_match_id(match_id, settings)
        rows = select_dev_timeline_events(
            self.timeline_events.list_current_by_match(effective_match_id),
            match_id=effective_match_id,
            settings=settings,
        )
        if not rows:
            raise FileNotFoundError(
                "DB에 에이전트용 timeline event가 없습니다: "
                f"match_id={effective_match_id}. "
                "scripts/seed_mock_timeline_events.py를 먼저 실행하세요."
            )
        events = [
            TimelineEventRead.model_validate(row).model_dump(
                mode="json",
                by_alias=True,
            )
            for row in rows
        ]

        session_id = uuid.uuid4().hex
        _SESSION_MATCH_IDS[session_id] = effective_match_id
        _SESSION_USER_IDS[session_id] = user_id

        current_clips: list[dict] = []
        clip_plan_id = None
        if project_id is not None:
            _SESSION_PROJECT_IDS[session_id] = project_id
            plans = self.clip_plans.repository.list_by_project(project_id)
            latest = next((plan for plan in plans if plan.items), None)
            if latest is not None:
                clip_plan_id = latest.clip_plan_id
                _SESSION_CLIP_PLAN_IDS[session_id] = latest.clip_plan_id
                events_by_id = {
                    event["timeline_event_id"]: event
                    for event in events
                }
                for item in sorted(latest.items, key=lambda row: row.order_index):
                    event = events_by_id.get(item.timeline_event_id)
                    if event is None:
                        continue
                    current_clips.append(
                        {
                            **event,
                            "start_sec": item.start_sec,
                            "end_sec": item.end_sec,
                            "duration_sec": item.duration_sec,
                            "metadata": {
                                **(event.get("metadata") or {}),
                                **(item.metadata_ or {}),
                            },
                        }
                    )

        config = self._config(session_id)
        self.graph.update_state(
            config,
            # current_clips는 "하이라이트 구성본"이므로 챗봇과 대화하기 전엔 빈 상태로 시작해야 한다.
            # "원본 전체 경기" 탭에서 보여줄 전체 후보 목록은 all_events가 따로 담당한다.
            {
                "match_id": effective_match_id,
                "all_events": events,
                "current_clips": current_clips,
            },
        )

        return {
            "session_id": session_id,
            "total_events": len(events),
            "video_url": self._resolve_video_url(effective_match_id, user_id),
            "project_id": project_id,
            "clip_plan_id": clip_plan_id,
        }

    def chat(
        self,
        session_id: str,
        message: str,
        *,
        user_id: str | None = None,
    ) -> dict:
        self._require_session_owner(session_id, user_id)
        config = self._config(session_id)
        snapshot = self.graph.get_state(config)

        if snapshot.next and "ask_user" in snapshot.next:
            result = self.graph.invoke(Command(resume=message), config)
        else:
            result = self.graph.invoke({"user_message": message}, config)

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            return {
                "needs_clarification": True,
                "current_clips": snapshot.values.get("current_clips") or [],
                "reply_text": None,
                "clarification_question": payload.get("clarification_question"),
                "clarification_options": payload.get("clarification_options"),
            }

        response = {
            "needs_clarification": False,
            "current_clips": result.get("current_clips") or [],
            "reply_text": result.get("final_response"),
            "clarification_question": None,
            "clarification_options": None,
        }
        response["clip_plan_id"] = self._sync_clip_plan(
            session_id,
            response["current_clips"],
        )
        return response

    def get_state(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict:
        self._require_session_owner(session_id, user_id)
        config = self._config(session_id)
        snapshot = self.graph.get_state(config)
        summary = get_current_state(snapshot.values)
        return {
            "current_clips": summary["current_clips"],
            "all_events": snapshot.values.get("all_events") or [],
            "project_id": _SESSION_PROJECT_IDS.get(session_id),
            "clip_plan_id": _SESSION_CLIP_PLAN_IDS.get(session_id),
        }

    def _sync_clip_plan(
        self,
        session_id: str,
        clips: list[dict],
    ) -> str | None:
        project_id = _SESSION_PROJECT_IDS.get(session_id)
        if project_id is None:
            return None
        items = [
            ClipPlanItemCreate(
                timeline_event_id=str(clip["timeline_event_id"]),
                start_sec=float(clip["start_sec"]),
                end_sec=float(clip["end_sec"]),
                order_index=index,
                reason=clip.get("description"),
                metadata={
                    **(clip.get("metadata") or {}),
                    "source": "chat_edit_session",
                    "session_id": session_id,
                },
            )
            for index, clip in enumerate(clips)
        ]
        clip_plan_id = _SESSION_CLIP_PLAN_IDS.get(session_id)
        if not items and clip_plan_id is None:
            return None
        if clip_plan_id is not None:
            updated = self.clip_plans.update_clip_plan(
                clip_plan_id,
                ClipPlanUpdateRequest(
                    mode="MANUAL",
                    summary="대화형 편집 세션에서 구성한 하이라이트",
                    items=items,
                ),
            )
            if updated is not None:
                return updated.clip_plan_id
        created = self.clip_plans.create_manual_clip_plan(
            ManualClipPlanCreateRequest(
                project_id=project_id,
                mode="MANUAL",
                summary="대화형 편집 세션에서 구성한 하이라이트",
                items=items,
            ),
            created_by=_SESSION_USER_IDS[session_id],
        )
        _SESSION_CLIP_PLAN_IDS[session_id] = created.clip_plan_id
        return created.clip_plan_id

    @staticmethod
    def _require_session_owner(
        session_id: str,
        user_id: str | None,
    ) -> None:
        if user_id is None:
            return
        if _SESSION_USER_IDS.get(session_id) != user_id:
            raise SessionNotFoundError(f"세션을 찾을 수 없습니다: {session_id}")

    def _resolve_video_url(self, match_id: str, user_id: str) -> str | None:
        assets = self.media_assets.list_by_match(match_id)
        for asset_type in _VIDEO_ASSET_TYPE_PRIORITY:
            asset = next((row for row in assets if row.asset_type == asset_type), None)
            if asset is not None:
                url, _ = build_signed_media_url(asset.asset_id, user_id)
                return url
        return None

    @staticmethod
    def _config(session_id: str) -> dict:
        if session_id not in _SESSION_MATCH_IDS:
            raise SessionNotFoundError(f"세션을 찾을 수 없습니다: {session_id}")
        return {"configurable": {"thread_id": session_id}}
