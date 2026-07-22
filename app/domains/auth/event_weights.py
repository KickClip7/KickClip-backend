from __future__ import annotations

import math


DEFAULT_EVENT_WEIGHTS: dict[str, float] = {
    "goal": 0.30,
    "penalty": 0.24,
    "shot": 0.18,
    "card": 0.13,
    "corner": 0.09,
    "substitution": 0.06,
}

EVENT_LABEL_ALIASES = {
    "corner_kick": "corner",
    "cornerkick": "corner",
}


def default_event_weights() -> dict[str, float]:
    return DEFAULT_EVENT_WEIGHTS.copy()


def normalize_event_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    return EVENT_LABEL_ALIASES.get(normalized, normalized)


def normalize_event_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("event_weights must contain at least one event")
    if len(weights) > 100:
        raise ValueError("event_weights cannot contain more than 100 events")

    normalized_weights: dict[str, float] = {}
    for raw_label, raw_weight in weights.items():
        label = normalize_event_label(str(raw_label))
        if not label:
            raise ValueError("event weight labels cannot be empty")
        if len(label) > 64:
            raise ValueError("event weight labels cannot exceed 64 characters")

        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("event weights must be finite non-negative numbers")
        normalized_weights[label] = normalized_weights.get(label, 0.0) + weight

    total = sum(normalized_weights.values())
    if total <= 0:
        raise ValueError("at least one event weight must be greater than zero")

    return {
        label: round(weight / total, 8)
        for label, weight in normalized_weights.items()
    }


def resolve_event_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        return default_event_weights()
    try:
        return normalize_event_weights(weights)
    except (TypeError, ValueError):
        return default_event_weights()


def calculate_importance_score(
    *,
    event_label: str,
    confidence_score: float | None,
    event_weights: dict[str, float] | None,
    context_bonus: float = 0.0,
) -> float:
    weights = resolve_event_weights(event_weights)
    label = normalize_event_label(event_label)
    fallback_weight = min(
        [*DEFAULT_EVENT_WEIGHTS.values(), *weights.values()],
        default=0.0,
    )
    event_weight = weights.get(
        label,
        DEFAULT_EVENT_WEIGHTS.get(label, fallback_weight),
    )
    confidence = _clamp(float(confidence_score or 0.0), 0.0, 1.0)
    bonus = float(context_bonus)
    if not math.isfinite(bonus):
        bonus = 0.0

    raw_score = event_weight * (0.5 + 0.5 * confidence) + bonus
    return round(_clamp(raw_score, 0.0, 1.0) * 100, 2)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
