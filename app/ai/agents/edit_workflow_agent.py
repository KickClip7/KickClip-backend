from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.ai.agents.clip_tools import (
    adjust_clip_duration,
    compute_total_duration,
    get_clips_by_half,
    get_clips_by_labels,
    rank_by_importance,
    remove_clip,
    remove_clips_by_labels,
    remove_nth_clip_of_labels,
    select_clip_combination,
    select_counts_per_label,
    select_one_per_label,
    sort_chronologically,
)
from app.ai.agents.edit_intent_schema import EditIntent
from app.ai.agents.label_taxonomy import (
    SUPPORTED_LABEL_DISPLAY,
    SUPPORTED_LABELS,
    analyze_scene_terms,
    fuzzy_match_supported_label,
    normalize_label,
    supported_labels_description,
)
from app.core.config import get_settings
from app.domains.timeline.service import get_timeline_events

MAX_RETRIES = 5


class EditWorkflowState(TypedDict, total=False):
    match_id: str
    all_events: list[dict]
    current_clips: list[dict]
    user_message: str
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
    match_stats = build_match_stats_summary(
        state.get("all_events") or [],
        state.get("current_clips") or [],
    )
    full_label_taxonomy = list(SUPPORTED_LABELS)
    taxonomy_desc = ", ".join(
        f"{label}({SUPPORTED_LABEL_DISPLAY[label]})" for label in SUPPORTED_LABELS
    )

    system_prompt = (
        "당신은 축구 하이라이트 편집 에이전트의 의도 분석기입니다. "
        "사용자의 한국어 요청을 분석해 EditIntent 구조로 응답하세요. "
        f"labels 필드는 문자열 리스트이며 각 값은 {full_label_taxonomy} 중 하나여야 합니다 "
        f"(각 라벨의 한국어 의미: {taxonomy_desc}). "
        "사용자가 언급한 장면 종류를 이 목록 중 하나로 매핑해서 채우세요(예: '득점'/'골'→goal, '페널티'→penalty). "
        "표기 변형·띄어쓰기·오타·별칭이라도 장면의 종류가 같으면 반드시 해당 라벨로 매핑하세요 "
        "(예: '패널티킥'/'페널티 킥'/'PK'→penalty, '콘너킥'/'코너'→corner, '슛'/'슛팅'→shot). "
        "이 목록에 해당하지 않는 '다른 종류의' 장면(예: 프리킥, 파울, 교체, 오프사이드, 골키퍼 선방)을 요청받으면 "
        "절대로 비슷해 보이는 다른 라벨로 대체하지 마세요. 예를 들어 '프리킥'을 corner로 바꾸면 안 됩니다. "
        "이 경우에만 intent_type='unsupported'로 두고 unsupported_terms에 사용자가 말한 표현을 그대로 담으세요. "
        "unsupported는 표기가 낯설어서가 아니라 장면의 종류 자체가 목록에 없을 때만 사용하는 값입니다. "
        "'골, 슈팅 하나씩'처럼 여러 종류를 언급하면 labels에 여러 값을 모두 담으세요. "
        "'슈팅 2개와 코너킥 2개'처럼 라벨마다 개수를 따로 말하면 label_counts에 "
        "{label, count} 항목으로 각각 담으세요(이때 labels에도 같은 라벨들을 담습니다). "
        "라벨별 개수를 말하지 않았다면 label_counts는 비워두세요. "
        "부정·수식 문맥에서 스쳐 지나간 장면 종류는 요청이 아니므로 labels와 label_counts에 넣지 마세요. "
        "예를 들어 '득점을 시도했지만 아쉬운 슈팅 2개'는 골을 원한다는 뜻이 아니라 "
        "골로 이어지지 않은 슈팅 2개를 원한다는 뜻이므로 goal이 아니라 shot만 담아야 합니다. "
        f"단, 현재 실제 데이터에 존재하는 라벨은 {available_labels}뿐이고 전/후반은 {available_halves}입니다 — "
        "요청한 라벨이 이 목록에 없다는 이유만으로는(즉 실제 데이터에 없는 장면이라는 이유만으로는) "
        "needs_clarification을 true로 설정하지 마세요. labels/half로 명확히 특정 가능하면 "
        "그 라벨이 실제 데이터에 있든 없든 항상 needs_clarification=false로 두고 intent_type과 관련 필드만 채우세요. "
        "'개쩌는', '멋진', '임팩트 있는', '조회수 잘 나올 것 같은' 같은 주관적 품질 표현은 "
        "장면 종류를 되물을 이유가 아니라 '에이전트가 알아서 좋은 장면을 골라달라'는 뜻입니다. "
        "이 경우 labels는 null로 두고 intent_type과 목표 길이/개수만 채우세요(중요도 기반 선별은 시스템이 합니다). "
        "'알아서', '아무거나', '너가 골라줘', '추천해줘'라고 하면 절대 되묻지 말고 labels=null인 build로 처리하세요. "
        "needs_clarification=true는 마지막 수단입니다: 목표 길이도 개수도 장면 종류도 없고 "
        "무엇을 원하는지 전혀 알 수 없을 때만 사용하세요. 이 경우 clarification_question에 되물을 질문을, "
        f"clarification_options에는 실제 데이터에 존재하는 라벨 목록({available_labels}) 기반의 선택지를 채우세요. "
        "되묻기는 '어떤 장면 종류를 원하는지'를 묻는 용도로만 쓰세요 — 전반/후반처럼 사용자가 말하지 않은 "
        "다른 조건은 되묻지 말고 해당 필드를 null로 두세요. "
        "지원하지 않는 장면(unsupported)일 때는 되묻지 말고 needs_clarification=false로 두세요. "
        "intent_type 의미: filter=조건에 맞는 장면만 보여달라, "
        "build=지금까지의 구성은 무시하고 목표 길이/개수로 하이라이트를 처음부터 새로 구성해달라, "
        "add=지금 구성에 있는 클립들은 그대로 유지한 채 새로운 조건의 장면을 추가로 더 넣어달라"
        "(예: '~도 추가해줘', '~더 넣어줘', '그대로 두고 ~ 추가해줘'), "
        "remove=현재 구성에서 클립을 빼달라. '두번째 클립 빼줘'처럼 순번이면 target_index만, "
        "'골 장면은 빼줘'처럼 장면 종류면 labels만 채우세요(이때 target_index는 null). "
        "'두번째 골 빼줘'처럼 둘 다 말하면 labels와 target_index를 함께 채우세요. "
        "라벨을 빼달라는 요청은 조건이 명확한 것이므로 절대 needs_clarification을 켜지 마세요. "
        "adjust=특정 순번 클립 길이를 늘리거나 줄여달라, "
        "confirm=지금 구성을 그대로 확정해달라, "
        "unsupported=위 라벨 목록으로 표현할 수 없는 장면 종류를 요청함(예: '프리킥만 모아줘'), "
        "chitchat=편집 명령이 아닌 일반 질문/인사/잡담(예: '너는 뭘 할 수 있어?', '안녕'). "
        "'골 몇 개 있어?', '지금 구성 총 길이는?'처럼 경기 데이터나 현재 구성에 대해 묻는 질문도 "
        "편집 명령이 아니므로 chitchat입니다. "
        f"현재 경기의 실제 데이터 통계: {match_stats} "
        "데이터 질문에는 이 통계를 근거로 response_text에 정확한 숫자로 답하고, 통계에 없는 내용은 "
        "지어내지 마세요. '경기 정보를 제공해 주시면'처럼 이미 아는 것을 되묻는 답변은 금지입니다. "
        "chitchat인 경우 needs_clarification은 항상 false로 두고, response_text에 친절한 한국어 답변을 채우세요 "
        "(에이전트가 할 수 있는 것을 소개할 땐 라벨 필터/전후반 필터/목표 길이로 하이라이트 구성/클립 제거/클립 길이 조정을 예시로 드세요). "
        "add일 때 target_duration이 있으면 이는 새로 추가되는 클립만의 목표 길이가 아니라 "
        "'기존 클립 + 새로 추가되는 클립' 전체 합계의 목표 길이입니다(예: 이미 40초짜리 구성이 있는데 "
        "'60초 이내로 슈팅 추가해줘'라고 하면 남는 20초 예산 안에서만 슈팅을 채워 넣으라는 뜻). "
        "이번 요청은 이전 대화와 독립된 새 명령입니다 — 이전에 언급됐던 라벨/전후반 조건을 자동으로 이어붙이지 말고, "
        "이번 메시지에 실제로 언급된 조건만 채우세요(언급 안 된 필드는 null). "
        "메시지에 '(선택한 조건: X)'가 포함돼 있으면, 이는 직전에 되물어서 사용자가 이미 명확히 답변한 것입니다. "
        "이 경우 메시지 앞부분에 남아있는 주관적 표현(예: '임팩트 있는')은 무시하고, 절대 needs_clarification을 "
        "다시 true로 설정하지 말고 X를 labels 필드에 그대로 사용하세요. "
        "X가 '알아서 골라줘' 같은 위임 표현이면 labels=null인 build로 처리하세요. "
        "되묻기로 재개된 경우에도 원래 메시지에 있던 목표 길이/개수(예: '1분 30초짜리')는 "
        "그대로 target_duration/target_clip_count에 유지해야 합니다."
    )

    # 매 턴을 독립된 새 요청으로 취급한다(이전 대화를 전부 보내면 LLM이 조건을 계속 이어붙이는 버그가 있었음).
    # AskUser에서 재개된 경우에는 ask_user 노드가 원래 요청과 답변을 이미 하나의 user_message로 합쳐서 넘겨준다.
    messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=state["user_message"])]

    intent: EditIntent = structured_model.invoke(messages)
    payload = _guard_unsupported_labels(intent.model_dump(), state["user_message"])
    payload = _normalize_intent_labels(payload)
    payload = _apply_actionability_guard(payload, state["user_message"])

    return {
        "intent": payload,
        "retry_count": 0,
        "clarification_question": payload.get("clarification_question"),
        "clarification_options": payload.get("clarification_options"),
    }


_DURATION_PATTERN = re.compile(r"(\d+)\s*분(?:\s*(\d+)\s*초)?")
_SECONDS_ONLY_PATTERN = re.compile(r"(\d+)\s*초")
_DELEGATION_KEYWORDS = ("알아서", "아무거나", "너가", "네가", "추천", "골라줘", "맡길")
_BUILD_KEYWORDS = ("만들", "구성", "편집")


def _extract_target_duration_sec(message: str) -> int | None:
    """원문에서 '1분 30초', '90초' 같은 목표 길이를 읽는다."""
    match = _DURATION_PATTERN.search(str(message))
    if match:
        return int(match.group(1)) * 60 + (int(match.group(2)) if match.group(2) else 0)
    match = _SECONDS_ONLY_PATTERN.search(str(message))
    if match:
        return int(match.group(1))
    return None


def _apply_actionability_guard(payload: dict, user_message: str) -> dict:
    """되묻기 남용을 막고, LLM이 흘린 목표 길이를 원문에서 복원하는 결정적 방어선.

    '1분 30초짜리 개쩌는 하이라이트 만들어줘'는 주관 표현 때문에 되물을 요청이 아니라
    이미 실행 가능한 요청이다. 목표 길이·제작 동사·위임 표현('알아서', '너가 골라줘')이
    있으면 되묻기를 걷어내고 에이전트가 알아서 고르는 build로 진행한다.
    """
    message = str(user_message)
    duration = _extract_target_duration_sec(message)

    if payload.get("needs_clarification"):
        actionable = (
            duration is not None
            or any(keyword in message for keyword in _BUILD_KEYWORDS)
            or any(keyword in message for keyword in _DELEGATION_KEYWORDS)
        )
        if actionable:
            payload["needs_clarification"] = False
            payload["clarification_question"] = None
            payload["clarification_options"] = None
            if not payload.get("labels") and not payload.get("label_counts"):
                payload["intent_type"] = "build"

    # 되묻기 왕복 뒤 LLM이 원래 메시지의 목표 길이를 잃어버리는 사고를 원문에서 복원한다.
    if (
        duration is not None
        and payload.get("intent_type") in ("build", "add")
        and not payload.get("target_duration")
        and not payload.get("target_clip_count")
        and not payload.get("label_counts")
    ):
        payload["target_duration"] = duration

    return payload


def _default_clarification_options(state: EditWorkflowState) -> list[str]:
    """되묻기 선택지가 비었을 때 실제 데이터의 라벨 + '알아서 골라줘'로 채운다."""
    options: list[str] = []
    for clip in state.get("all_events") or []:
        normalized = normalize_label(str(clip.get("label") or ""))
        display = SUPPORTED_LABEL_DISPLAY.get(normalized) if normalized else None
        if display and display not in options:
            options.append(display)
    return [*options, "알아서 골라줘"]


def _format_match_time(seconds: float) -> str:
    """UI 표기와 같은 mm:ss (경기 98분 골은 98:04처럼 분이 60을 넘을 수 있다)."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _label_timestamps(all_events: list[dict], label: str, limit: int = 8) -> str:
    stamps = [
        _format_match_time(clip.get("timestamp_sec") or clip.get("start_sec") or 0.0)
        for clip in all_events
        if str(clip.get("label") or "").strip().lower() == label
    ]
    if not stamps:
        return ""
    shown = ", ".join(stamps[:limit])
    return shown if len(stamps) <= limit else f"{shown} 외 {len(stamps) - limit}개"


def build_match_stats_summary(all_events: list[dict], current_clips: list[dict]) -> str:
    """의도 분석 LLM에 주입하는 실제 데이터 통계.

    '골 몇 개 있어?' 같은 데이터 질문에 LLM이 지어내지 않고 답하려면 라벨 이름만이
    아니라 개수와 시각이 프롬프트에 있어야 한다.
    """
    counts: dict[str, list[float]] = {}
    for clip in all_events:
        label = str(clip.get("label") or "").strip().lower()
        if label:
            counts.setdefault(label, []).append(
                float(clip.get("timestamp_sec") or clip.get("start_sec") or 0.0)
            )

    ordered = [label for label in SUPPORTED_LABELS if label in counts] + [
        label for label in counts if label not in SUPPORTED_LABELS
    ]
    parts = []
    for label in ordered:
        display = SUPPORTED_LABEL_DISPLAY.get(label, label)
        stamps = _label_timestamps(all_events, label)
        parts.append(f"{display} {len(counts[label])}개({stamps})" if stamps else f"{display} {len(counts[label])}개")

    summary = f"이 경기에서 찾은 장면 총 {len(all_events)}개: {', '.join(parts) or '없음'}."
    if current_clips:
        summary += (
            f" 현재 하이라이트 구성본: {len(current_clips)}개, "
            f"총 {round(compute_total_duration(current_clips))}초 ({describe_label_mix(current_clips)})."
        )
    else:
        summary += " 현재 하이라이트 구성본은 비어 있습니다."
    return summary


def _answer_data_question(message: str, all_events: list[dict], current_clips: list[dict]) -> str | None:
    """'골 몇 개 있어?' 같은 개수 질문에 데이터에서 직접 답한다.

    LLM 프롬프트에 통계를 주입해도 LLM이 흘려버릴 수 있어서, 가장 흔한 질문 형태는
    결정적으로 답해 정확성을 보장한다. 해당 없으면 None을 돌려 LLM 답변을 쓴다.
    """
    text = str(message)
    if "몇" not in text and "개수" not in text:
        return None

    _, mentioned_labels = analyze_scene_terms(text)
    if mentioned_labels:
        answers = []
        for label in mentioned_labels:
            display = SUPPORTED_LABEL_DISPLAY.get(label, label)
            stamps = _label_timestamps(all_events, label)
            count = sum(
                1 for clip in all_events
                if str(clip.get("label") or "").strip().lower() == label
            )
            if count:
                answers.append(f"이 경기에서 {display} 장면은 {count}개예요 ({stamps} 지점).")
            else:
                answers.append(f"이 경기에서 {display} 장면은 찾지 못했어요.")
        follow_up = f"'{SUPPORTED_LABEL_DISPLAY.get(mentioned_labels[0], mentioned_labels[0])} 장면만 보여줘'라고 하면 타임라인에 바로 띄워드려요."
        return " ".join([*answers, follow_up])

    if "장면" in text or "클립" in text or "이벤트" in text or "개" in text:
        return build_match_stats_summary(all_events, current_clips)
    return None


def _normalize_intent_labels(payload: dict) -> dict:
    """LLM이 라벨 값을 한국어나 오타로 돌려줘도('패널티킥') 표준 라벨로 정규화한다."""
    labels = payload.get("labels")
    if labels:
        normalized: list[str] = []
        for label in labels:
            resolved = (
                normalize_label(str(label))
                or fuzzy_match_supported_label(str(label))
                or str(label)
            )
            if resolved not in normalized:
                normalized.append(resolved)
        payload["labels"] = normalized
    return payload


def _infer_action_from_message(message: str) -> str:
    """intent_type이 'unsupported'로 소모된 뒤 라벨이 복구됐을 때, 원문 동사로 동작을 추정한다."""
    text = str(message)
    if any(keyword in text for keyword in ("추가", "더 넣", "더넣", "넣어")):
        return "add"
    if any(keyword in text for keyword in ("빼", "삭제", "제거", "지워")):
        return "remove"
    if any(keyword in text for keyword in ("만들", "구성")):
        return "build"
    return "filter"


def _guard_unsupported_labels(payload: dict, user_message: str) -> dict:
    """LLM의 '지원 여부' 오판을 원문 기준으로 양방향 교정하는 결정적 방어선.

    방향 1 — 대체 되돌리기: '프리킥'(진짜 미지원)을 corner 같은 비슷한 라벨로
    바꿔치기한 경우, 원문 스캔으로 잡아내 unsupported로 되돌린다.
    방향 2 — 오판 복구: '패널티킥'(페널티의 표기 변형)을 미지원으로 거절한 경우,
    자모 퍼지 매칭으로 지원 라벨임을 확인해 원래 편집 동작으로 복구한다.
    """
    scan_unsupported, mentioned_labels = analyze_scene_terms(user_message)

    # LLM이 미지원이라고 보고한 표현을 재심사한다: 표기 변형이면 라벨로 복구.
    recovered_labels: list[str] = []
    llm_unsupported: list[str] = []
    for term in payload.get("unsupported_terms") or []:
        label = fuzzy_match_supported_label(term)
        if label:
            if label not in recovered_labels:
                recovered_labels.append(label)
        elif term not in llm_unsupported:
            llm_unsupported.append(term)

    unsupported_terms = scan_unsupported + [
        term for term in llm_unsupported if term not in scan_unsupported
    ]
    confirmed_labels = mentioned_labels + [
        label for label in recovered_labels if label not in mentioned_labels
    ]

    if not unsupported_terms and not recovered_labels:
        return payload

    # 지원 여부는 되물어서 해결되는 문제가 아니라 그대로 알려줘야 한다.
    payload["needs_clarification"] = False
    payload["clarification_question"] = None
    payload["clarification_options"] = None

    if unsupported_terms:
        payload["unsupported_terms"] = unsupported_terms
        if confirmed_labels:
            # "프리킥이랑 골 보여줘"처럼 지원 라벨도 같이 말한 경우: 지원되는 것만
            # 처리하고 나머지는 제외했다고 알린다. LLM이 만들어낸 대체 라벨은 버린다.
            payload["labels"] = confirmed_labels
            if payload.get("intent_type") == "unsupported":
                payload["intent_type"] = _infer_action_from_message(user_message)
        else:
            payload["intent_type"] = "unsupported"
            payload["labels"] = None
    else:
        # 전부 표기 변형 오판이었다: 미지원 취급을 걷어내고 원래 편집 동작으로 복구한다.
        payload["unsupported_terms"] = None
        existing = [
            normalized
            for label in (payload.get("labels") or [])
            if (normalized := normalize_label(str(label))) is not None
        ]
        payload["labels"] = existing + [
            label for label in confirmed_labels if label not in existing
        ]
        if payload.get("intent_type") == "unsupported":
            payload["intent_type"] = _infer_action_from_message(user_message)

    return payload


def is_ambiguous(state: EditWorkflowState) -> str:
    return "ask_user" if (state.get("intent") or {}).get("needs_clarification") else "plan"


def ask_user(state: EditWorkflowState) -> dict:
    intent = state.get("intent") or {}
    options = intent.get("clarification_options") or []
    if not options:
        # 선택지 없는 되묻기는 사용자를 막막하게 한다. 실제 데이터 라벨로 채워 버튼을 띄운다.
        options = _default_clarification_options(state)
    elif "알아서 골라줘" not in options:
        options = [*options, "알아서 골라줘"]
    payload = {
        "clarification_question": intent.get("clarification_question") or "어떤 걸 원하시나요?",
        "clarification_options": options,
    }

    answer = interrupt(payload)

    # 원래 요청 + 되묻기 답변을 하나의 메시지로 합쳐서 다음 UnderstandRequest가 전체 맥락을 갖게 한다.
    merged_message = f"{state['user_message']} (선택한 조건: {answer})"

    return {"user_message": merged_message}


def resolve_label_counts(intent: dict) -> dict[str, int]:
    """intent.label_counts를 {표준 라벨: 개수}로 정규화한다.

    알 수 없는 라벨과 1 미만의 개수는 버린다. 같은 라벨이 두 번 나오면 합친다.
    """
    resolved: dict[str, int] = {}
    for entry in intent.get("label_counts") or []:
        if not isinstance(entry, dict):
            continue
        raw_label = entry.get("label") or ""
        label = normalize_label(raw_label) or fuzzy_match_supported_label(raw_label)
        count = entry.get("count")
        if label is None or not isinstance(count, int) or count < 1:
            continue
        resolved[label] = resolved.get(label, 0) + count
    return resolved


def describe_labels(labels: list[str]) -> str:
    """라벨 목록을 사용자에게 보여줄 한국어 문구로 만든다 (예: '슈팅/코너킥')."""
    display = []
    for label in labels:
        normalized = normalize_label(label)
        display.append(SUPPORTED_LABEL_DISPLAY.get(normalized, str(label)) if normalized else str(label))
    return "/".join(display)


def describe_label_mix(clips: list[dict]) -> str:
    """실제로 고른 클립의 라벨 구성을 알려준다 (예: '슈팅 2개, 코너킥 2개').

    "3개 장면으로 구성했어요"처럼 총 개수만 알려주면 요청과 다른 라벨이 섞여도
    사용자가 알아챌 수 없어서, 무엇을 몇 개 넣었는지 항상 밝힌다.
    """
    counts: dict[str, int] = {}
    for clip in clips:
        raw = str(clip.get("label") or "")
        normalized = normalize_label(raw)
        display = SUPPORTED_LABEL_DISPLAY.get(normalized, raw) if normalized else raw
        counts[display] = counts.get(display, 0) + 1
    return ", ".join(f"{label} {count}개" for label, count in counts.items())


def describe_label_shortfall(label_counts: dict[str, int], clips: list[dict]) -> str:
    """요청한 개수를 못 채운 라벨을 알려준다. 모자란 자리를 조용히 넘기지 않기 위함."""
    if not label_counts:
        return ""
    actual: dict[str, int] = {}
    for clip in clips:
        normalized = normalize_label(str(clip.get("label") or ""))
        if normalized:
            actual[normalized] = actual.get(normalized, 0) + 1
    missing = [
        f"{SUPPORTED_LABEL_DISPLAY.get(label, label)} {requested - actual.get(label, 0)}개"
        for label, requested in label_counts.items()
        if actual.get(label, 0) < requested
    ]
    if not missing:
        return ""
    return f"({', '.join(missing)}는 경기에 남은 장면이 없어서 채우지 못했어요.)"


def plan_or_revise_edit(state: EditWorkflowState) -> dict:
    intent = state.get("intent") or {}
    intent_type = intent.get("intent_type")
    all_events = state.get("all_events") or []
    current_clips = state.get("current_clips") or all_events

    plan_clips: list[dict] = []
    action_summary = "요청을 처리했어요."
    unsupported_terms = intent.get("unsupported_terms") or []
    duration_shortfall = False

    # LLM이 동작(intent)은 분류했지만 장면 조건이 전부 미지원 표현뿐인 경우:
    # 조건 없는 filter/build/add로 흘러가 엉뚱한 클립을 집어오지 않도록 막는다.
    if (
        intent_type in ("filter", "build", "add")
        and unsupported_terms
        and not (intent.get("labels") or intent.get("label_counts"))
    ):
        intent_type = "unsupported"

    if intent_type == "unsupported":
        # 현재 구성은 건드리지 않는다. 지원하지 않는 요청 때문에 편집본이 날아가면 안 된다.
        plan_clips = current_clips
        terms_desc = ", ".join(unsupported_terms) if unsupported_terms else "요청하신 장면"
        action_summary = (
            f"'{terms_desc}' 장면은 현재 분석 모델이 찾아내지 못해서 편집에 사용할 수 없어요. "
            f"지금 사용할 수 있는 장면은 {supported_labels_description()}입니다."
        )

    elif intent_type == "filter":
        pool = all_events
        labels = intent.get("labels") or []
        if labels:
            pool = get_clips_by_labels(labels, pool)
        if intent.get("half") is not None:
            pool = get_clips_by_half(intent["half"], pool)
        # 필터는 "조건에 맞는 장면 전체"를 보여주는 것이므로 중요도가 아니라
        # 경기 시간 순서로 배치해야 한다 (2:0 골이 1:0 골보다 먼저 나오면 안 됨).
        plan_clips = sort_chronologically(pool)
        if plan_clips:
            action_summary = f"{describe_label_mix(plan_clips)}를 찾았어요."
        else:
            action_summary = "조건에 맞는 장면을 찾지 못했어요."

    elif intent_type == "build":
        labels = intent.get("labels") or []
        half_filtered = get_clips_by_half(intent["half"], all_events) if intent.get("half") is not None else all_events
        target_duration = intent.get("target_duration")
        label_counts = resolve_label_counts(intent)

        if label_counts:
            # "슈팅 2개와 코너킥 2개"처럼 라벨마다 개수를 지정한 요청.
            plan_clips = select_counts_per_label(label_counts, half_filtered)
        elif len(labels) > 1:
            # "골, 슈팅 하나씩" 처럼 개수 없이 여러 라벨만 말하면 라벨별 최고점 클립을 하나씩 고른다.
            plan_clips = select_one_per_label(labels, half_filtered)
        else:
            pool = get_clips_by_labels(labels, half_filtered) if labels else half_filtered
            ranked = rank_by_importance(pool)
            if target_duration:
                # 재료가 목표 길이보다 적으면 실패시키지 말고 전부 담은 뒤 사실대로 알린다.
                if compute_total_duration(ranked) <= float(target_duration):
                    plan_clips = sort_chronologically(ranked)
                    duration_shortfall = bool(plan_clips)
                else:
                    plan_clips = select_clip_combination(ranked, float(target_duration))
            else:
                count = intent.get("target_clip_count") or 5
                # 선택은 중요도 상위 N개로 하되, 배치는 경기 시간 순서를 따른다.
                plan_clips = sort_chronologically(ranked[:count])

        if not plan_clips:
            action_summary = "요청하신 조건에 맞는 장면을 찾지 못했어요."
        elif duration_shortfall:
            pool_desc = describe_labels(labels) if labels else "조건에 맞는"
            action_summary = (
                f"{pool_desc} 장면이 총 {round(compute_total_duration(plan_clips))}초뿐이라 "
                f"목표 {target_duration}초를 다 채우진 못했어요. {describe_label_mix(plan_clips)}를 모두 담았어요."
            )
        else:
            # 장면 종류를 지정하지 않은 요청('개쩌는 하이라이트')은 에이전트가 골랐음을 밝힌다.
            chosen_note = "" if labels or label_counts else "중요도가 높은 순으로 골라 "
            action_summary = f"{chosen_note}{describe_label_mix(plan_clips)}로 하이라이트를 구성했어요."
        shortfall = describe_label_shortfall(label_counts, plan_clips)
        if shortfall:
            action_summary = f"{action_summary} {shortfall}"

    elif intent_type == "add":
        labels = intent.get("labels") or []
        half_filtered = get_clips_by_half(intent["half"], all_events) if intent.get("half") is not None else all_events
        existing_ids = {clip.get("timeline_event_id") for clip in current_clips}
        candidate_pool = get_clips_by_labels(labels, half_filtered) if labels else half_filtered
        pool = [clip for clip in candidate_pool if clip.get("timeline_event_id") not in existing_ids]
        ranked = rank_by_importance(pool)
        target_duration = intent.get("target_duration")
        label_counts = resolve_label_counts(intent)
        label_desc = describe_labels(labels) if labels else "새"

        if label_counts:
            # "슈팅 2개 더 넣어줘"처럼 라벨별 개수를 지정한 추가 요청.
            new_clips = select_counts_per_label(label_counts, pool)
        elif target_duration:
            # target_duration은 "기존 + 새로 추가" 전체 합계 목표다. 이미 채운 만큼을 뺀 남는 예산만큼만 채운다.
            current_total = compute_total_duration(current_clips)
            remaining_budget = float(target_duration) - current_total
            new_clips = select_clip_combination(ranked, remaining_budget) if remaining_budget > 0 else []
        else:
            count = intent.get("target_clip_count") or 1
            new_clips = ranked[:count]

        plan_clips = sort_chronologically([*current_clips, *new_clips])

        if target_duration and not new_clips and (float(target_duration) - compute_total_duration(current_clips)) <= 0:
            action_summary = f"이미 {round(compute_total_duration(current_clips))}초로 목표 {target_duration}초를 채워서 더 추가하지 못했어요."
        elif not new_clips:
            action_summary = f"'{label_desc}' 조건에 맞는 새 장면을 더 찾지 못해서 추가하지 못했어요."
        elif target_duration:
            action_summary = f"기존 {len(current_clips)}개 클립은 그대로 두고, 전체 {target_duration}초 이내로 맞추려고 {describe_label_mix(new_clips)}를 추가했어요."
        else:
            action_summary = f"기존 {len(current_clips)}개 클립은 그대로 두고, {describe_label_mix(new_clips)}를 추가했어요."
        shortfall = describe_label_shortfall(label_counts, new_clips)
        if shortfall:
            action_summary = f"{action_summary} {shortfall}"

    elif intent_type == "chitchat":
        plan_clips = current_clips
        # '골 몇 개 있어?' 같은 개수 질문은 LLM 답변 대신 데이터에서 직접 답해 정확성을 보장한다.
        # current_clips 변수는 비었을 때 all_events로 폴백하므로, 통계에는 실제 구성본을 쓴다.
        data_answer = _answer_data_question(
            state.get("user_message") or "", all_events, state.get("current_clips") or []
        )
        action_summary = data_answer or intent.get("response_text") or (
            "저는 축구 하이라이트 편집을 도와드려요. "
            "'골 장면만 보여줘', '전반전만 보여줘', '그중에 두번째 빼줘' 같은 요청을 해보세요."
        )

    elif intent_type == "remove":
        remove_labels = [
            normalize_label(str(label))
            or fuzzy_match_supported_label(str(label))
            or str(label).strip().lower()
            for label in (intent.get("labels") or [])
        ]
        target_index = intent.get("target_index")

        if remove_labels and target_index:
            # "두번째 골 빼줘" — 해당 라벨 클립 중 N번째 하나만 제거.
            try:
                plan_clips = remove_nth_clip_of_labels(current_clips, remove_labels, target_index)
                action_summary = f"{target_index}번째 {describe_labels(remove_labels)} 클립을 뺐어요."
            except ValueError:
                plan_clips = current_clips
                matched = len(get_clips_by_labels(remove_labels, current_clips))
                action_summary = (
                    f"현재 구성에 {describe_labels(remove_labels)} 장면이 {matched}개뿐이라 "
                    f"{target_index}번째를 찾지 못했어요."
                )
        elif remove_labels:
            # "골 장면은 빼줘" — 라벨로 지목된 클립을 전부 제거.
            remaining = remove_clips_by_labels(current_clips, remove_labels)
            removed_count = len(current_clips) - len(remaining)
            plan_clips = remaining
            if removed_count == 0:
                plan_clips = current_clips
                action_summary = (
                    f"현재 구성에는 {describe_labels(remove_labels)} 장면이 없어서 뺄 게 없어요."
                )
            elif remaining:
                action_summary = (
                    f"{describe_labels(remove_labels)} 장면 {removed_count}개를 뺐어요. "
                    f"이제 {describe_label_mix(remaining)}가 남았어요."
                )
            else:
                action_summary = (
                    f"{describe_labels(remove_labels)} 장면 {removed_count}개를 뺐더니 구성이 비었어요. "
                    "새로 구성하려면 '골 3개로 만들어줘'처럼 요청해주세요."
                )
        else:
            try:
                plan_clips = remove_clip(current_clips, target_index)
                action_summary = f"{target_index}번째 클립을 제거했어요."
            except ValueError:
                plan_clips = current_clips
                action_summary = (
                    f"몇 번째 클립을 뺄지 찾지 못했어요. 지금 구성은 {len(current_clips)}개이고, "
                    "'두번째 클립 빼줘'나 '골 장면은 빼줘'처럼 말씀해주세요."
                )

    elif intent_type == "adjust":
        index = intent.get("target_index")
        if index and 1 <= index <= len(current_clips):
            position = index - 1
            updated = adjust_clip_duration(current_clips[position], float(intent.get("adjust_delta_sec") or 0))
            plan_clips = [*current_clips[:position], updated, *current_clips[position + 1 :]]
            action_summary = f"{index}번째 클립 길이를 조정했어요."
        else:
            # 순번을 못 찾았다고 구성본을 비우면 안 된다. 그대로 두고 되물을 말만 남긴다.
            plan_clips = current_clips
            action_summary = (
                f"조정할 클립 순번을 찾지 못했어요. 지금 구성은 {len(current_clips)}개이고, "
                "'두번째 클립 5초 늘려줘'처럼 말씀해주세요."
            )

    elif intent_type == "confirm":
        plan_clips = current_clips
        action_summary = "현재 구성을 그대로 확정했어요."

    if unsupported_terms and intent_type != "unsupported":
        # 지원 라벨과 섞여 들어온 경우: 무엇이 빠졌는지 밝히고 나머지 결과를 준다.
        action_summary = (
            f"'{', '.join(unsupported_terms)}' 장면은 현재 분석 모델이 찾지 못해서 제외했어요. "
            f"{action_summary}"
        )

    return {
        "plan_result": {
            "clips": plan_clips,
            "reply_text": action_summary,
            "duration_shortfall": duration_shortfall,
        },
        "retry_count": state.get("retry_count", 0),
    }


def validate_plan(state: EditWorkflowState) -> dict:
    plan = state.get("plan_result") or {}
    clips = plan.get("clips") or []
    intent = state.get("intent") or {}
    intent_type = intent.get("intent_type")
    labels = intent.get("labels") or []

    # unsupported는 재계획해도 결과가 달라지지 않는 정상 종료다. remove/adjust도 현재
    # 구성본에 대한 결정적 연산이라 재시도가 무의미하고, 전부 빼서 구성이 비는 것도
    # 정당한 결과이므로 재시도 루프에 넣지 않는다.
    if not clips and intent_type not in ("confirm", "chitchat", "unsupported", "remove", "adjust"):
        return {
            "validation_ok": False,
            "validation_reason": "empty_result",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    target_duration = intent.get("target_duration")
    # 라벨별 개수/하나씩 고르는 build는 개수가 목표이지 길이가 목표가 아니므로 길이 검증을 생략한다.
    # 재료 부족(duration_shortfall)은 계획 단계가 이미 사실대로 안내했으므로 실패로 만들지 않는다.
    label_counts = resolve_label_counts(intent)
    if (
        intent_type == "build"
        and target_duration
        and len(labels) <= 1
        and not label_counts
        and not plan.get("duration_shortfall")
    ):
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
    labels = intent.get("labels") or []

    if intent.get("intent_type") in ("remove", "adjust"):
        # remove/adjust 실패를 "장면을 찾지 못했다"는 검색 실패 문구로 답하면 오답이다.
        message = (
            "요청하신 편집을 적용하지 못했어요. '두번째 클립 빼줘'처럼 순번으로, "
            "또는 '골 장면은 빼줘'처럼 장면 종류로 다시 알려주시겠어요?"
        )
    elif labels:
        message = (
            f"'{', '.join(labels)}' 라벨의 장면을 찾지 못했어요. "
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
