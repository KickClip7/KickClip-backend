from __future__ import annotations

import uuid

from langgraph.types import Command
from sqlalchemy.orm import Session

from app.ai.agents.clip_tools import get_current_state
from app.ai.agents.edit_workflow_agent import build_edit_workflow_graph
from app.domains.media.repository import MediaAssetRepository
from app.domains.media.signed_url import build_signed_media_url
from app.domains.timeline.service import get_timeline_events

_VIDEO_ASSET_TYPE_PRIORITY = ("RAW_VIDEO", "RAW_VIDEO_HALF1", "WEB_PREVIEW_VIDEO")

# session_id -> match_id. 프로세스 인메모리 저장이라 서버 재시작 시 세션이 소실된다
# (로컬 데모/개발 범위에서만 사용, DB/Redis 등 영속 저장소는 쓰지 않는다).
_SESSION_MATCH_IDS: dict[str, str] = {}


class SessionNotFoundError(LookupError):
    pass


class SessionService:
    def __init__(self, db: Session):
        self.media_assets = MediaAssetRepository(db)
        self.graph = build_edit_workflow_graph()

    def start(self, match_id: str, user_id: str) -> dict:
        data = get_timeline_events(match_id)
        events = data["events"]

        session_id = uuid.uuid4().hex
        _SESSION_MATCH_IDS[session_id] = match_id

        config = self._config(session_id)
        self.graph.update_state(
            config,
            {"match_id": match_id, "all_events": events, "current_clips": events},
        )

        return {
            "session_id": session_id,
            "total_events": len(events),
            "video_url": self._resolve_video_url(match_id, user_id),
        }

    def chat(self, session_id: str, message: str) -> dict:
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

        return {
            "needs_clarification": False,
            "current_clips": result.get("current_clips") or [],
            "reply_text": result.get("final_response"),
            "clarification_question": None,
            "clarification_options": None,
        }

    def get_state(self, session_id: str) -> dict:
        config = self._config(session_id)
        snapshot = self.graph.get_state(config)
        summary = get_current_state(snapshot.values)
        return {
            "current_clips": summary["current_clips"],
            "all_events": snapshot.values.get("all_events") or [],
        }

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
