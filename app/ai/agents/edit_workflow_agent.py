from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.ai.agents.clip_tools import (
    adjust_clip_duration,
    compute_total_duration,
    get_clips_by_half,
    get_clips_by_label,
    rank_by_importance,
    remove_clip,
    select_clip_combination,
)
from app.ai.agents.edit_intent_schema import EditIntent
from app.core.config import get_settings
from app.domains.timeline.service import get_timeline_events

MAX_RETRIES = 5


class EditWorkflowState(TypedDict, total=False):
    match_id: str
    all_events: list[dict]
    current_clips: list[dict]
    user_message: str
    conversation: list[dict]
    intent: dict
    plan_result: dict
    validation_ok: bool
    validation_reason: str | None
    retry_count: int
    clarification_question: str | None
    clarification_options: list[str] | None
    final_response: str | None
    status: str


def load_session(state: EditWorkflowState) -> dict:
    """순수 함수. all_events가 이미 로드돼 있으면(예: 세션 API가 미리 seed) 아무것도 안 한다."""
    if state.get("all_events"):
        return {}

    data = get_timeline_events(state["match_id"])
    events = data["events"]
    return {"all_events": events, "current_clips": state.get("current_clips") or events}


def understand_request(state: EditWorkflowState) -> dict:
    settings = get_settings()
    model = ChatOpenAI(
        model=settings.OPENAI_AGENT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        timeout=settings.OPENAI_AGENT_TIMEOUT_SECONDS,
        max_retries=1,
    )
    structured_model = model.with_structured_output(EditIntent)

    pool = state.get("current_clips") or state.get("all_events") or []
    available_labels = sorted({str(clip.get("label")) for clip in pool if clip.get("label")})
    available_halves = sorted({clip.get("half") for clip in pool if clip.get("half") is not None})
    full_label_taxonomy = ["Goal", "Shot", "Card", "Corner", "Substitution", "Penalty"]

    system_prompt = (
        "당신은 축구 하이라이트 편집 에이전트의 의도 분석기입니다. "
        "사용자의 한국어 요청을 분석해 EditIntent 구조로 응답하세요. "
        f"label 필드에 쓸 수 있는 값은 {full_label_taxonomy} 중 하나뿐입니다. "
        "사용자가 언급한 장면 종류를 이 목록 중 하나로 매핑해서 채우세요(예: '득점'/'골'→Goal, '페널티'→Penalty). "
        f"단, 현재 실제 데이터에 존재하는 라벨은 {available_labels}뿐이고 전/후반은 {available_halves}입니다 — "
        "요청한 라벨이 이 목록에 없다는 이유만으로는(즉 실제 데이터에 없는 장면이라는 이유만으로는) "
        "needs_clarification을 true로 설정하지 마세요. label/half로 명확히 특정 가능하면 "
        "그 라벨이 실제 데이터에 있든 없든 항상 needs_clarification=false로 두고 intent_type과 관련 필드만 채우세요. "
        "needs_clarification=true는 오직 '임팩트 있는', '멋진', '재밌는'처럼 위 라벨 목록 중 무엇에도 "
        "매핑할 수 없는 주관적 표현일 때만 사용하세요. 이 경우 clarification_question에 되물을 질문을, "
        f"clarification_options에는 실제 데이터에 존재하는 라벨 목록({available_labels}) 기반의 선택지를 채우세요. "
        "intent_type 의미: filter=조건에 맞는 장면만 보여달라, build=목표 길이/개수로 하이라이트를 구성해달라, "
        "remove=현재 구성에서 특정 순번 클립을 빼달라, adjust=특정 순번 클립 길이를 늘리거나 줄여달라, "
        "confirm=지금 구성을 그대로 확정해달라."
    )

    # ask_user에서 재개된 경우 conversation 마지막 항목이 이미 이번 user_message이므로 중복 추가하지 않는다.
    conversation = list(state.get("conversation") or [])
    last_turn = conversation[-1] if conversation else None
    if not last_turn or last_turn.get("role") != "user" or last_turn.get("content") != state["user_message"]:
        conversation.append({"role": "user", "content": state["user_message"]})

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for turn in conversation[-9:]:
        content = str(turn.get("content", ""))
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    intent: EditIntent = structured_model.invoke(messages)

    return {
        "intent": intent.model_dump(),
        "conversation": conversation,
        "retry_count": 0,
        "clarification_question": intent.clarification_question,
        "clarification_options": intent.clarification_options,
    }


def is_ambiguous(state: EditWorkflowState) -> str:
    return "ask_user" if (state.get("intent") or {}).get("needs_clarification") else "plan"


def ask_user(state: EditWorkflowState) -> dict:
    intent = state.get("intent") or {}
    payload = {
        "clarification_question": intent.get("clarification_question") or "어떤 걸 원하시나요?",
        "clarification_options": intent.get("clarification_options") or [],
    }

    answer = interrupt(payload)

    conversation = list(state.get("conversation") or [])
    conversation.append({"role": "assistant", "content": payload["clarification_question"]})
    conversation.append({"role": "user", "content": str(answer)})

    return {"conversation": conversation, "user_message": str(answer)}


def plan_or_revise_edit(state: EditWorkflowState) -> dict:
    intent = state.get("intent") or {}
    intent_type = intent.get("intent_type")
    all_events = state.get("all_events") or []
    current_clips = state.get("current_clips") or all_events

    plan_clips: list[dict] = []
    action_summary = "요청을 처리했어요."

    if intent_type == "filter":
        pool = all_events
        if intent.get("label"):
            pool = get_clips_by_label(intent["label"], pool)
        if intent.get("half") is not None:
            pool = get_clips_by_half(intent["half"], pool)
        plan_clips = rank_by_importance(pool)
        action_summary = f"'{intent.get('label') or '전체'}' 조건으로 {len(plan_clips)}개 장면을 찾았어요."

    elif intent_type == "build":
        pool = all_events
        if intent.get("label"):
            pool = get_clips_by_label(intent["label"], pool)
        if intent.get("half") is not None:
            pool = get_clips_by_half(intent["half"], pool)
        ranked = rank_by_importance(pool)
        target_duration = intent.get("target_duration")
        if target_duration:
            plan_clips = select_clip_combination(ranked, float(target_duration))
        else:
            count = intent.get("target_clip_count") or 5
            plan_clips = ranked[:count]
        action_summary = f"{len(plan_clips)}개 장면으로 하이라이트를 구성했어요."

    elif intent_type == "remove":
        try:
            plan_clips = remove_clip(current_clips, intent.get("target_index"))
            action_summary = f"{intent.get('target_index')}번째 클립을 제거했어요."
        except ValueError:
            plan_clips = []
            action_summary = "제거할 클립 순번을 찾지 못했어요."

    elif intent_type == "adjust":
        index = intent.get("target_index")
        if index and 1 <= index <= len(current_clips):
            position = index - 1
            updated = adjust_clip_duration(current_clips[position], float(intent.get("adjust_delta_sec") or 0))
            plan_clips = [*current_clips[:position], updated, *current_clips[position + 1 :]]
            action_summary = f"{index}번째 클립 길이를 조정했어요."
        else:
            plan_clips = []
            action_summary = "조정할 클립 순번을 찾지 못했어요."

    elif intent_type == "confirm":
        plan_clips = current_clips
        action_summary = "현재 구성을 그대로 확정했어요."

    return {
        "plan_result": {"clips": plan_clips, "reply_text": action_summary},
        "retry_count": state.get("retry_count", 0),
    }


def validate_plan(state: EditWorkflowState) -> dict:
    plan = state.get("plan_result") or {}
    clips = plan.get("clips") or []
    intent = state.get("intent") or {}

    if not clips and intent.get("intent_type") != "confirm":
        return {
            "validation_ok": False,
            "validation_reason": "empty_result",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    target_duration = intent.get("target_duration")
    if intent.get("intent_type") == "build" and target_duration:
        total = compute_total_duration(clips)
        tolerance = max(10.0, float(target_duration) * 0.3)
        if abs(total - float(target_duration)) > tolerance:
            return {
                "validation_ok": False,
                "validation_reason": "duration_off_target",
                "retry_count": state.get("retry_count", 0) + 1,
            }

    return {
        "validation_ok": True,
        "validation_reason": None,
        "current_clips": clips,
    }


def route_after_validate(state: EditWorkflowState) -> str:
    if state.get("validation_ok"):
        return "save"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "fail"
    return "revise"


def save_and_respond(state: EditWorkflowState) -> dict:
    plan = state.get("plan_result") or {}
    return {
        "final_response": plan.get("reply_text") or "편집을 반영했어요.",
        "status": "success",
    }


def fail_respond(state: EditWorkflowState) -> dict:
    intent = state.get("intent") or {}
    available_labels = sorted({str(clip.get("label")) for clip in (state.get("all_events") or []) if clip.get("label")})
    label = intent.get("label")

    if label:
        message = (
            f"'{label}' 라벨의 장면을 찾지 못했어요. "
            f"사용 가능한 라벨은 {', '.join(available_labels)}입니다. 다시 요청해주시겠어요?"
        )
    else:
        message = (
            f"요청하신 조건에 맞는 편집안을 만들지 못했어요. "
            f"사용 가능한 라벨은 {', '.join(available_labels)}입니다."
        )

    return {"final_response": message, "status": "failed"}


def _build_graph():
    graph_builder = StateGraph(EditWorkflowState)
    graph_builder.add_node("load_session", load_session)
    graph_builder.add_node("understand_request", understand_request)
    graph_builder.add_node("ask_user", ask_user)
    graph_builder.add_node("plan_or_revise_edit", plan_or_revise_edit)
    graph_builder.add_node("validate_plan", validate_plan)
    graph_builder.add_node("save_and_respond", save_and_respond)
    graph_builder.add_node("fail_respond", fail_respond)

    graph_builder.add_edge(START, "load_session")
    graph_builder.add_edge("load_session", "understand_request")
    graph_builder.add_conditional_edges(
        "understand_request",
        is_ambiguous,
        {"ask_user": "ask_user", "plan": "plan_or_revise_edit"},
    )
    graph_builder.add_edge("ask_user", "understand_request")
    graph_builder.add_edge("plan_or_revise_edit", "validate_plan")
    graph_builder.add_conditional_edges(
        "validate_plan",
        route_after_validate,
        {"save": "save_and_respond", "revise": "plan_or_revise_edit", "fail": "fail_respond"},
    )
    graph_builder.add_edge("save_and_respond", END)
    graph_builder.add_edge("fail_respond", END)

    return graph_builder.compile(checkpointer=InMemorySaver())


_GRAPH_SINGLETON: Any | None = None


def build_edit_workflow_graph():
    global _GRAPH_SINGLETON
    if _GRAPH_SINGLETON is None:
        _GRAPH_SINGLETON = _build_graph()
    return _GRAPH_SINGLETON
