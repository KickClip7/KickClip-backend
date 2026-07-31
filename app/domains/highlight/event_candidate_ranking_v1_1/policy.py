from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_policy(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("policy_id")
        != "target_centric_tracking_event_candidate_ranking_v1_1"
        or document.get("status") != "PROVISIONAL_SHADOW_ONLY"
        or document.get("automatic_target_confirmation") is not False
    ):
        raise ValueError("V1.1 policy freeze contract is invalid.")
    return document


def _get(row: dict[str, Any], dotted_path: str) -> float | None:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if isinstance(value, (int, float)) else None


def _normalize(value: float, spec: dict[str, Any]) -> float:
    transform = spec.get("transform", "identity")
    if transform == "inverse_decay":
        scale = float(spec["scale"])
        return math.exp(-max(0.0, value) / scale)
    if transform == "clip":
        low = float(spec.get("min", 0.0))
        high = float(spec.get("max", 1.0))
        if high <= low:
            raise ValueError("Policy clip range is invalid.")
        return max(0.0, min(1.0, (value - low) / (high - low)))
    if transform == "inverse_clip":
        low = float(spec.get("min", 0.0))
        high = float(spec.get("max", 1.0))
        if high <= low:
            raise ValueError("Policy clip range is invalid.")
        return 1.0 - max(0.0, min(1.0, (value - low) / (high - low)))
    return max(0.0, min(1.0, value))


def score_candidate(
    row: dict[str, Any],
    *,
    canonical_label: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    group_name = policy["event_policy_map"][canonical_label]
    group = policy["event_policies"][group_name]
    feature_specs = group["features"]
    weighted = 0.0
    available_weight = 0.0
    contributions: dict[str, float | None] = {}
    for dotted_path, spec in feature_specs.items():
        value = _get(row["raw_features"], dotted_path)
        if value is None:
            contributions[dotted_path] = None
            continue
        contribution = _normalize(value, spec)
        weight = float(spec["weight"])
        contributions[dotted_path] = contribution
        weighted += contribution * weight
        available_weight += weight
    event_relevance = weighted / available_weight if available_weight else 0.0
    trackability = float(row["trackability_score"])
    recommendation = (
        event_relevance * float(group["recommendation_mix"]["event_relevance"])
        + trackability * float(group["recommendation_mix"]["trackability"])
    )
    reason_codes = [
        item["code"]
        for item in group["reason_rules"]
        if (
            (value := _get(row["raw_features"], item["feature"])) is not None
            and (
                ("gte" in item and value >= float(item["gte"]))
                or ("lte" in item and value <= float(item["lte"]))
            )
        )
    ]
    risk_codes = list(row.get("risk_codes") or [])
    risk_codes.extend(
        item["code"]
        for item in group["risk_rules"]
        if (
            (value := _get(row["raw_features"], item["feature"])) is not None
            and (
                ("gte" in item and value >= float(item["gte"]))
                or ("lte" in item and value <= float(item["lte"]))
            )
        )
    )
    return {
        **row,
        "event_policy_group": group_name,
        "event_relevance_score": event_relevance,
        "recommendation_score": recommendation,
        "reason_codes": sorted(set(reason_codes)),
        "risk_codes": sorted(set(risk_codes)),
        "score_contributions": contributions,
    }
