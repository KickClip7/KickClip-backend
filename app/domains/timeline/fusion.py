from __future__ import annotations

from app.domains.player.model import Player
from app.domains.auth.event_weights import calculate_importance_score
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.schema import FrontendTimelineEvent
from app.domains.player.schema import FrontendPlayer


FRONTEND_CATEGORY_BY_BACKEND_LABEL = {
    "goal": "goal",
    "shot": "shot",
    "penalty": "penalty",
    "foul": "foul",
    "card": "card",
    "substitution": "substitution",
    "free_kick": "freekick",
    "freekick": "freekick",
    "corner": "corner",
}

BACKEND_LABEL_BY_FRONTEND_CATEGORY = {
    "all": None,
    "goal": "goal",
    "shot": "shot",
    "penalty": "penalty",
    "foul": "foul",
    "card": "card",
    "substitution": "substitution",
    "free_kick": "free_kick",
    "freekick": "free_kick",
    "corner": "corner",
}

FRONTEND_EVENT_TYPES = [
    "goal",
    "shot",
    "penalty",
    "foul",
    "card",
    "substitution",
    "freekick",
    "corner",
]

BACKEND_EVENT_LABELS = [
    "goal",
    "shot",
    "penalty",
    "foul",
    "card",
    "substitution",
    "free_kick",
    "corner",
]

DEFAULT_TAG_BY_CATEGORY = {
    "goal": "GOAL",
    "shot": "SHOT",
    "penalty": "PENALTY",
    "foul": "FOUL",
    "card": "CARD",
    "substitution": "SUBSTITUTION",
    "freekick": "FREEKICK",
    "corner": "CORNER",
}

DEFAULT_TITLE_BY_CATEGORY = {
    "goal": "득점 장면",
    "shot": "슈팅 장면",
    "penalty": "페널티 장면",
    "foul": "파울 장면",
    "card": "카드 장면",
    "substitution": "선수 교체 장면",
    "freekick": "프리킥 장면",
    "corner": "코너킥 장면",
}

DEFAULT_DESCRIPTION_BY_CATEGORY = {
    "goal": "AI가 하이라이트 후보로 감지한 득점 장면입니다.",
    "shot": "AI가 하이라이트 후보로 감지한 슈팅 장면입니다.",
    "penalty": "AI가 하이라이트 후보로 감지한 페널티 장면입니다.",
    "foul": "AI가 하이라이트 후보로 감지한 파울 장면입니다.",
    "card": "AI가 하이라이트 후보로 감지한 카드 장면입니다.",
    "substitution": "AI가 하이라이트 후보로 감지한 선수 교체 장면입니다.",
    "freekick": "AI가 하이라이트 후보로 감지한 프리킥 장면입니다.",
    "corner": "AI가 하이라이트 후보로 감지한 코너킥 장면입니다.",
}


def normalize_timeline_category_filter(category: str | None) -> str | None:
    """Map frontend category query values to backend labels.

    DB에는 free_kick으로 저장하지만 프론트 더미/필터는 freekick을 사용한다.
    따라서 API 경계에서만 변환한다.
    """
    if category is None:
        return None

    normalized = category.strip().lower()
    if not normalized:
        return None

    if normalized not in BACKEND_LABEL_BY_FRONTEND_CATEGORY:
        return normalized

    return BACKEND_LABEL_BY_FRONTEND_CATEGORY[normalized]


def build_frontend_event(
    event: TimelineEvent,
    *,
    match_duration_sec: float | None = None,
    event_weights: dict[str, float] | None = None,
    context_bonus: float = 0.0,
) -> FrontendTimelineEvent:
    metadata = event.metadata_ or {}
    category = FRONTEND_CATEGORY_BY_BACKEND_LABEL.get(event.label, event.label)
    duration_sec = max(0.0, float(event.duration_sec or 0.0))
    min_start_sec = max(0.0, float(event.start_sec or 0.0))
    natural_end_sec = max(min_start_sec, float(event.end_sec or min_start_sec))

    return FrontendTimelineEvent(
        id=event.timeline_event_id,
        category=category,
        tag=str(metadata.get("tag") or DEFAULT_TAG_BY_CATEGORY.get(category, event.label.upper())),
        title=event.title or DEFAULT_TITLE_BY_CATEGORY.get(category, "하이라이트 장면"),
        time=_format_match_time(event.timestamp_sec),
        minute=_safe_int(event.timestamp_sec // 60),
        duration=max(1, _safe_int(round(event.duration_sec))),
        start=_format_clip_timecode(event.start_sec),
        end=_format_clip_timecode(event.end_sec),
        duration_sec=round(duration_sec, 3),
        min_start_sec=round(min_start_sec, 3),
        natural_end_sec=round(natural_end_sec, 3),
        half=_build_half(event.timestamp_sec, match_duration_sec, event.half),
        importance_score=_build_importance_score(
            event.label,
            event.confidence,
            event_weights,
            context_bonus,
        ),
        score=_build_score(event.confidence, event.highlight_score),
        highlightScore=_build_highlight_score(event.highlight_score, event.confidence),
        description=(
            event.description
            or DEFAULT_DESCRIPTION_BY_CATEGORY.get(category, "AI가 감지한 하이라이트 후보입니다.")
        ),
        playerIds=list(event.player_ids or []),
        team=event.team_name or "",
    )


def build_frontend_player(player: Player) -> FrontendPlayer:
    return FrontendPlayer(
        id=player.player_id,
        name=player.display_name,
        number=player.number,
        team=player.team_name or "",
        role=player.role or "",
        identityStatus=player.identity_status,
    )


def _build_score(
    confidence: float | None,
    highlight_score: float | None,
) -> float:
    """Build a frontend-friendly 0~100 score.

    confidence는 정확도가 아니라 모델 내부 점수이므로 이름을 score로만 변환한다.
    """
    if confidence is not None:
        raw = confidence * 100 if confidence <= 1 else confidence
    elif highlight_score is not None:
        raw = highlight_score * 10 if highlight_score <= 10 else highlight_score
    else:
        raw = 0

    return round(_clamp(raw, 0, 100), 1)


def _build_highlight_score(
    highlight_score: float | None,
    confidence: float | None,
) -> int:
    """Build frontend highlightScore in 1~10 range."""
    if highlight_score is not None:
        raw = highlight_score if highlight_score <= 10 else highlight_score / 10
    elif confidence is not None:
        raw = confidence * 10 if confidence <= 1 else confidence / 10
    else:
        raw = 1

    return int(round(_clamp(raw, 1, 10)))


def _build_importance_score(
    event_label: str,
    confidence: float | None,
    event_weights: dict[str, float] | None,
    context_bonus: float,
) -> float:
    """Calculate the current user's event-weighted importance score."""
    return calculate_importance_score(
        event_label=event_label,
        confidence_score=confidence,
        event_weights=event_weights,
        context_bonus=context_bonus,
    )


def _build_half(
    timestamp_sec: float | None,
    match_duration_sec: float | None,
    stored_half: int | None,
) -> int | None:
    """Split the video at its midpoint; use the stored value if duration is unknown."""
    if match_duration_sec is not None and match_duration_sec > 0:
        return 1 if float(timestamp_sec or 0.0) < match_duration_sec / 2 else 2
    return stored_half


def _format_match_time(seconds: float | None) -> str:
    total_seconds = max(0, int(round(seconds or 0)))
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def _format_clip_timecode(seconds: float | None) -> str:
    value = max(0.0, float(seconds or 0.0))
    minutes = int(value // 60)
    remaining_seconds = value - (minutes * 60)
    return f"{minutes:02d}:{remaining_seconds:04.1f}"


def _safe_int(value: float | int | None) -> int:
    if value is None:
        return 0
    return int(value)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
