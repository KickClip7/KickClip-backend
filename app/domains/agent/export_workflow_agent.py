from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.agent.qwen_export_assistant import QwenExportAssistant
from app.domains.agent.schema import (
    ExportAssistantRecommendation,
    ExportAssistantRequest,
    ExportAssistantResponse,
    ExportAssistantSettings,
)


class ExportAgentConfigurationError(RuntimeError):
    """Raised when the LLM agent cannot be configured."""


class ExportAgentState(MessagesState):
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: int


class LangGraphExportAgent:
    """GPT-4o-mini tool-calling agent for the export workspace."""

    def __init__(self, db: Session, *, model: Any | None = None, qwen: QwenExportAssistant | None = None):
        self.db = db
        self.settings = get_settings()
        self.qwen = qwen or QwenExportAssistant(db)
        self._provided_model = model

    def invoke(self, data: ExportAssistantRequest) -> ExportAssistantResponse:
        tools = self._build_tools(data)
        model = self._provided_model or self._create_model()
        tool_model = model.bind_tools(tools)
        tools_by_name = {item.name: item for item in tools}
        system_message = SystemMessage(content=self._system_prompt(data))

        def call_model(state: ExportAgentState) -> dict[str, Any]:
            response = tool_model.invoke([system_message, *state["messages"]])
            return {
                "messages": [response],
                "llm_calls": int(state.get("llm_calls", 0)) + 1,
            }

        def call_tools(state: ExportAgentState) -> dict[str, Any]:
            last_message = state["messages"][-1]
            tool_messages: list[ToolMessage] = []
            tool_results: list[dict[str, Any]] = []

            for tool_call in getattr(last_message, "tool_calls", []):
                tool_name = str(tool_call.get("name") or "")
                tool_id = str(tool_call.get("id") or tool_name)
                selected_tool = tools_by_name.get(tool_name)
                if selected_tool is None:
                    result = {
                        "kind": "tool_error",
                        "tool": tool_name,
                        "message": "지원하지 않는 도구입니다.",
                    }
                else:
                    try:
                        result = selected_tool.invoke(tool_call.get("args") or {})
                    except Exception as exc:
                        result = {
                            "kind": "tool_error",
                            "tool": tool_name,
                            "message": str(exc),
                        }

                tool_results.append(result)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(self._compact_tool_result(result), ensure_ascii=False),
                        tool_call_id=tool_id,
                    )
                )

            return {"messages": tool_messages, "tool_results": tool_results}

        def route_after_model(state: ExportAgentState) -> str:
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return END

        graph_builder = StateGraph(ExportAgentState)
        graph_builder.add_node("model", call_model)
        graph_builder.add_node("tools", call_tools)
        graph_builder.add_edge(START, "model")
        graph_builder.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
        graph_builder.add_edge("tools", "model")
        graph = graph_builder.compile()

        result = graph.invoke(
            {
                "messages": self._conversation_messages(data),
                "tool_results": [],
                "llm_calls": 0,
            },
            {"recursion_limit": 8},
        )
        return self._response_from_state(result)

    def _create_model(self) -> ChatOpenAI:
        if not self.settings.OPENAI_API_KEY.strip():
            raise ExportAgentConfigurationError(
                "OPENAI_API_KEY가 설정되지 않아 내보내기 Agent를 실행할 수 없습니다."
            )
        return ChatOpenAI(
            model=self.settings.OPENAI_AGENT_MODEL,
            api_key=self.settings.OPENAI_API_KEY,
            temperature=0,
            timeout=self.settings.OPENAI_AGENT_TIMEOUT_SECONDS,
            max_retries=1,
        )

    def _build_tools(self, data: ExportAssistantRequest) -> list[Any]:
        @tool
        def analyze_video_with_qwen(
            targets: list[Literal["title", "hashtags", "thumbnail"]],
        ) -> dict[str, Any]:
            """Analyze the football video with Qwen for only the requested targets. Targets must be selected from title, hashtags, and thumbnail."""
            recommendation = self.qwen.recommend_selected(
                targets=targets,
                clip_plan_id=data.clip_plan_id,
                match_id=data.match_id,
                language=data.language,
            )
            return {
                "kind": "qwen_recommendation",
                "recommendation": recommendation.model_dump(mode="json"),
            }

        @tool
        def apply_export_settings(
            ratio: str | None = None,
            captions_enabled: bool | None = None,
            music: str | None = None,
            quality: str | None = None,
            title: str | None = None,
        ) -> dict[str, Any]:
            """Apply explicit export settings. Supported ratios are 9:16, 1:1, 16:9 and qualities are 720p, 1080p, 4K."""
            patch: dict[str, Any] = {}
            if ratio is not None:
                if ratio not in {"9:16", "1:1", "16:9"}:
                    raise ValueError("지원하는 비율은 9:16, 1:1, 16:9입니다.")
                patch["ratio"] = ratio
            if captions_enabled is not None:
                patch["captions_enabled"] = captions_enabled
            if music is not None:
                patch["music"] = music.strip() or "없음"
            if quality is not None:
                normalized_quality = "4K" if quality.lower() == "4k" else quality.lower()
                if normalized_quality not in {"720p", "1080p", "4K"}:
                    raise ValueError("지원하는 화질은 720p, 1080p, 4K입니다.")
                patch["quality"] = normalized_quality
            if title is not None:
                patch["title"] = title.strip()[:60]
            if not patch:
                raise ValueError("변경할 내보내기 설정이 없습니다.")
            return {"kind": "settings_patch", "settings_patch": patch}

        return [analyze_video_with_qwen, apply_export_settings]

    def _system_prompt(self, data: ExportAssistantRequest) -> str:
        current_settings = data.current_settings.model_dump(exclude_none=True)
        return (
            "당신은 KickClip의 내보내기 편집 Agent입니다. 사용자의 한국어 자연어 요청을 이해하고 필요한 도구를 선택하세요. "
            "영상 내용을 근거로 제목·해시태그·썸네일 추천이 필요한 요청에는 analyze_video_with_qwen 도구를 호출하세요. "
            "이때 targets에는 사용자가 실제로 요청한 항목만 넣으세요: 제목은 title, 해시태그는 hashtags, 썸네일은 thumbnail입니다. "
            "예를 들어 썸네일만 요청하면 targets=[\"thumbnail\"]만 사용하며 제목이나 해시태그를 함께 생성하지 않습니다. "
            "사용자가 모두, 게시 정보 전체, 세 가지를 요청한 경우에만 세 항목을 모두 넣으세요. Qwen 결과를 임의로 만들어내지 마세요. "
            "비율·자막·배경음악·화질·사용자가 명시한 제목 변경에는 apply_export_settings 도구를 호출하세요. "
            "요청한 설정이 현재 설정과 이미 같으면 도구를 다시 호출하지 말고 이미 적용되어 있다고 정확히 안내하세요. "
            "한 요청에 두 종류의 작업이 있으면 두 도구를 모두 호출할 수 있습니다. "
            "도구 실행 후에는 실제 결과만 짧고 자연스러운 한국어로 설명하세요. 지원 범위를 벗어난 요청에는 가능한 기능을 안내하세요. "
            f"현재 내보내기 설정: {json.dumps(current_settings, ensure_ascii=False)}"
        )

    @staticmethod
    def _conversation_messages(data: ExportAssistantRequest) -> list[Any]:
        messages: list[Any] = []
        for item in data.conversation[-12:]:
            if item.role == "assistant":
                messages.append(AIMessage(content=item.content))
            else:
                messages.append(HumanMessage(content=item.content))
        messages.append(HumanMessage(content=data.message))
        return messages

    @staticmethod
    def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("kind") != "qwen_recommendation":
            return result
        recommendation = dict(result.get("recommendation") or {})
        if recommendation.get("thumbnail"):
            thumbnail = dict(recommendation["thumbnail"])
            thumbnail.pop("image_data_url", None)
            recommendation["thumbnail"] = thumbnail
        return {"kind": result["kind"], "recommendation": recommendation}

    def _response_from_state(self, state: ExportAgentState) -> ExportAssistantResponse:
        settings_patch: dict[str, Any] = {}
        recommendation: ExportAssistantRecommendation | None = None
        tools_used: list[str] = []

        for result in state.get("tool_results", []):
            kind = result.get("kind")
            if kind == "settings_patch":
                settings_patch.update(result.get("settings_patch") or {})
                tools_used.append("apply_export_settings")
            elif kind == "qwen_recommendation":
                current = ExportAssistantRecommendation.model_validate(result["recommendation"])
                if recommendation is None:
                    recommendation = current
                else:
                    merged = recommendation.model_dump()
                    merged.update(current.model_dump(exclude_none=True))
                    merged["requested_fields"] = list(dict.fromkeys([
                        *recommendation.requested_fields,
                        *current.requested_fields,
                    ]))
                    recommendation = ExportAssistantRecommendation.model_validate(merged)
                tools_used.append("analyze_video_with_qwen")

        final_message = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
            ),
            "요청 처리를 완료했습니다.",
        )
        if not isinstance(final_message, str):
            final_message = str(final_message)

        return ExportAssistantResponse(
            message=final_message.strip() or "요청 처리를 완료했습니다.",
            settings_patch=ExportAssistantSettings(**settings_patch),
            recommendation=recommendation,
            tools_used=list(dict.fromkeys(tools_used)),
            model_id=self.settings.OPENAI_AGENT_MODEL,
        )
