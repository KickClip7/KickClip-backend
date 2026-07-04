SUPPORTED_EVENT_LABELS = [
    "goal",
    "shot",
    "foul",
    "card",
    "free_kick",
    "corner",
]


LABEL_TO_DISPLAY = {
    "goal": {
        "tag": "GOAL",
        "title": "GOAL",
        "description": "득점으로 이어진 핵심 장면입니다.",
    },
    "shot": {
        "tag": "SHOT",
        "title": "SHOT",
        "description": "슈팅이 발생한 공격 장면입니다.",
    },
    "foul": {
        "tag": "FOUL",
        "title": "FOUL",
        "description": "파울이 발생한 경기 흐름 전환 장면입니다.",
    },
    "card": {
        "tag": "CARD",
        "title": "CARD",
        "description": "카드가 나온 주요 판정 장면입니다.",
    },
    "free_kick": {
        "tag": "FREE KICK",
        "title": "FREE KICK",
        "description": "프리킥으로 이어진 세트피스 장면입니다.",
    },
    "corner": {
        "tag": "CORNER",
        "title": "CORNER",
        "description": "코너킥으로 이어진 세트피스 장면입니다.",
    },
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")

    if normalized == "freekick":
        normalized = "free_kick"

    if normalized not in SUPPORTED_EVENT_LABELS:
        return "shot"

    return normalized


def get_display_info(label: str) -> dict:
    normalized = normalize_label(label)
    return LABEL_TO_DISPLAY.get(normalized, LABEL_TO_DISPLAY["shot"])