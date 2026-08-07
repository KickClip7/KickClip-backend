"""EditWorkflow 에이전트가 쓰는 장면 라벨 taxonomy와 한국어 표현 해석기.

현재 Action Spotting 챔피언 모델(soccer_spotter_v9)이 실제로 만들어내는 라벨은
configs/models/action_spotting/soccer_spotter_v9/inference_policy.json 의
class_order 5종뿐이다. app/ai/tasks/highlight_spotting/label_map.py 에는 foul,
free_kick 이 남아 있지만 그건 은퇴한 6-class 모델의 잔재라서 현재 데이터에는
절대 나타나지 않는다.

프리킥처럼 모델이 만들지 않는 장면을 요청받았을 때 코너킥 같은 "비슷한" 라벨로
바꿔치기하면 사용자는 잘못된 결과를 옳은 결과로 오해한다. 그래서 지원하지 않는
장면은 조용히 대체하지 말고 지원하지 않는다고 알려야 한다.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

SUPPORTED_LABELS: tuple[str, ...] = ("goal", "shot", "penalty", "card", "corner")

SUPPORTED_LABEL_DISPLAY: dict[str, str] = {
    "goal": "골",
    "shot": "슈팅",
    "penalty": "페널티",
    "card": "카드",
    "corner": "코너킥",
}

# 사용자 표현 -> 표준 라벨. 사용자 메시지 부분 문자열 검사와 LLM이 돌려준 labels
# 정규화에 함께 쓴다.
LABEL_ALIASES: dict[str, str] = {
    "goal": "goal",
    "골": "goal",
    "득점": "goal",
    "shot": "shot",
    "슈팅": "shot",
    "슛": "shot",
    "penalty": "penalty",
    "페널티킥": "penalty",
    "페널티": "penalty",
    "pk": "penalty",
    "card": "card",
    "카드": "card",
    "경고": "card",
    "퇴장": "card",
    "corner_kick": "corner",
    "cornerkick": "corner",
    "corner": "corner",
    "코너킥": "corner",
    "코너": "corner",
}

# 자주 요청받지만 현재 모델이 만들어내지 않는 장면들. 여기 걸린 표현은 LLM이
# 무엇으로 매핑했든 "지원 불가"로 되돌린다.
UNSUPPORTED_TERM_ALIASES: dict[str, str] = {
    "free_kick": "프리킥",
    "free kick": "프리킥",
    "freekick": "프리킥",
    "프리킥": "프리킥",
    "foul": "파울",
    "파울": "파울",
    "반칙": "파울",
    "substitution": "교체",
    "교체": "교체",
    "offside": "오프사이드",
    "오프사이드": "오프사이드",
    "save": "선방",
    "세이브": "선방",
    "선방": "선방",
    "goalkeeper": "골키퍼",
    "골키퍼": "골키퍼",
    "키퍼": "골키퍼",
    "goal_kick": "골킥",
    "골킥": "골킥",
    "throw_in": "스로인",
    "스로인": "스로인",
    "assist": "어시스트",
    "어시스트": "어시스트",
    "드리블": "드리블",
    "태클": "태클",
    "헤딩": "헤딩",
    "크로스": "크로스",
}

# 부분 문자열 검사는 항상 긴 표현부터 해야 한다. "골킥"/"골키퍼"를 먼저 잡아내지
# 않으면 그 안의 "골"이 goal 로 잡힌다.
_UNSUPPORTED_SCAN_ORDER: tuple[str, ...] = tuple(
    sorted(UNSUPPORTED_TERM_ALIASES, key=len, reverse=True)
)
_LABEL_SCAN_ORDER: tuple[str, ...] = tuple(sorted(LABEL_ALIASES, key=len, reverse=True))


def supported_labels_description() -> str:
    """사용자에게 보여줄 '지금 쓸 수 있는 장면' 문구."""
    return ", ".join(SUPPORTED_LABEL_DISPLAY[label] for label in SUPPORTED_LABELS)


def normalize_label(value: str) -> str | None:
    """LLM이 돌려준 라벨 문자열을 표준 라벨로 정규화한다. 모르는 값이면 None."""
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in SUPPORTED_LABELS:
        return key
    return LABEL_ALIASES.get(key) or LABEL_ALIASES.get(key.replace("_", " "))


# ---- 자모 기반 퍼지 매칭 -------------------------------------------------------
# '패널티킥'(페널티의 흔한 오기), '슛팅' 같은 표기 변형·오타는 단어 사전에 일일이
# 추가하는 방식(룰 베이스)으로는 끝이 없다. 한글을 자모로 분해해 지원 라벨 별칭과의
# 유사도를 재면 처음 보는 오타에도 일반화된다.

_CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONGSEONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]

# 자모가 이보다 짧은 별칭('골', '슛', 'pk')은 퍼지 대상에서 제외한다.
# 짧은 문자열은 우연히 높은 유사도가 나와 오탐('고을'→골)을 만들기 때문에 정확 일치로만 잡는다.
_FUZZY_MIN_JAMO = 4
_FUZZY_THRESHOLD = 0.75

# 편집 대화에 늘 등장하는 문맥 단어는 퍼지 매칭에서 제외한다.
# 예: '경기'가 카드의 별칭 '경고'와 자모 유사도 0.8로 잘못 매칭되는 오탐 방지.
_FUZZY_STOPWORDS = frozenset(
    {
        "경기",
        "영상",
        "장면",
        "클립",
        "하이라이트",
        "편집",
        "구성",
        "시간",
        "전반",
        "후반",
        "전반전",
        "후반전",
        "해당",
        "이번",
        "지금",
    }
)

_PARTICLES = ("이랑", "으로", "을", "를", "이", "가", "은", "는", "도", "만", "과", "와", "랑", "의", "에", "로")


def _to_jamo(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            index = code - 0xAC00
            out.append(_CHOSEONG[index // 588])
            out.append(_JUNGSEONG[(index % 588) // 28])
            final = _JONGSEONG[index % 28]
            if final:
                out.append(final)
        else:
            out.append(ch.lower())
    return "".join(out)


def fuzzy_match_supported_label(term: str, threshold: float = _FUZZY_THRESHOLD) -> str | None:
    """표기 변형·오타를 지원 라벨로 해석한다 (예: '패널티킥'→penalty, '슛팅'→shot).

    미지원 표현(프리킥 등)과 그 오타가 지원 라벨로 새면 안 되므로, 미지원 별칭과의
    유사도가 지원 별칭과의 유사도 이상이면 매칭하지 않는다.
    """
    cleaned = str(term).strip().lower().replace(" ", "").replace("-", "_")
    if not cleaned:
        return None
    exact = normalize_label(cleaned)
    if exact:
        return exact
    if cleaned in UNSUPPORTED_TERM_ALIASES or cleaned in _FUZZY_STOPWORDS:
        return None

    term_jamo = _to_jamo(cleaned)
    if len(term_jamo) < _FUZZY_MIN_JAMO:
        return None

    best_supported: tuple[float, str | None] = (0.0, None)
    for alias, label in LABEL_ALIASES.items():
        alias_jamo = _to_jamo(alias)
        if len(alias_jamo) < _FUZZY_MIN_JAMO:
            continue
        ratio = SequenceMatcher(None, term_jamo, alias_jamo).ratio()
        if ratio > best_supported[0]:
            best_supported = (ratio, label)

    best_unsupported = 0.0
    for alias in UNSUPPORTED_TERM_ALIASES:
        alias_jamo = _to_jamo(alias)
        if len(alias_jamo) < _FUZZY_MIN_JAMO:
            continue
        ratio = SequenceMatcher(None, term_jamo, alias_jamo).ratio()
        best_unsupported = max(best_unsupported, ratio)

    if best_supported[0] >= threshold and best_supported[0] > best_unsupported:
        return best_supported[1]
    return None


def _scene_term_tokens(message: str) -> set[str]:
    """메시지에서 장면 표현 후보 토큰을 뽑는다. 조사('~을', '~이랑')를 뗀 변형도 포함한다."""
    tokens = re.findall(r"[가-힣]+|[a-zA-Z_]+", str(message))
    variants: set[str] = set()
    for token in tokens:
        variants.add(token)
        for particle in _PARTICLES:
            if token.endswith(particle) and len(token) > len(particle):
                variants.add(token[: -len(particle)])
                break
    return variants


def analyze_scene_terms(message: str) -> tuple[list[str], list[str]]:
    """사용자 원문에서 (지원하지 않는 장면 표현, 실제로 언급된 표준 라벨)을 뽑는다.

    LLM 응답이 아니라 원문을 직접 본다. 프롬프트만으로는 대체 매핑을 막지 못해서
    이 함수가 결정적 방어선 역할을 한다. 지원하지 않는 표현을 먼저 걸러내고 그
    자리를 지운 뒤 라벨을 찾기 때문에 "골킥"의 "골"이 goal 로 새지 않는다.
    """
    text = str(message).lower()

    unsupported: list[str] = []
    for term in _UNSUPPORTED_SCAN_ORDER:
        if term in text:
            display = UNSUPPORTED_TERM_ALIASES[term]
            if display not in unsupported:
                unsupported.append(display)
            text = text.replace(term, " ")

    labels: list[str] = []
    for term in _LABEL_SCAN_ORDER:
        if term in text:
            label = LABEL_ALIASES[term]
            if label not in labels:
                labels.append(label)
            text = text.replace(term, " ")

    # 남은 텍스트를 토큰 단위로 퍼지 매칭한다. '패널티킥'처럼 사전에 없는 표기
    # 변형·오타를 여기서 회수한다 (정확 일치로 소비된 부분은 이미 지워진 상태).
    for token in _scene_term_tokens(text):
        label = fuzzy_match_supported_label(token)
        if label and label not in labels:
            labels.append(label)

    return unsupported, labels
