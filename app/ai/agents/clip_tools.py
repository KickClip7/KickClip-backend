"""EditWorkflow 그래프의 PlanOrReviseEdit 노드가 EditIntent를 보고 직접 호출하는 순수 함수들.

@tool 데코레이터를 쓰지 않는다 (LLM이 직접 도구를 호출하는 게 아니라,
노드 코드가 EditIntent.intent_type을 보고 판단해서 호출하는 방식).
모든 clip은 TimelineEventRead 스키마를 따르는 dict(label, start_sec, end_sec,
duration_sec, highlight_score, timestamp_sec, half, metadata 등)를 가정한다.
"""

from __future__ import annotations

import itertools
from typing import Any


def get_clips_by_label(label: str, candidates: list[dict]) -> list[dict]:
    normalized = str(label).strip().lower()
    return [clip for clip in candidates if str(clip.get("label", "")).strip().lower() == normalized]


def get_clips_by_half(half: int, candidates: list[dict]) -> list[dict]:
    return [clip for clip in candidates if clip.get("half") == half]


def rank_by_importance(clips: list[dict]) -> list[dict]:
    return sorted(clips, key=lambda clip: float(clip.get("highlight_score") or 0.0), reverse=True)


def select_clip_combination(
    candidates: list[dict],
    target_duration: float,
    max_pool: int = 8,
) -> list[dict]:
    """highlight_score 상위 max_pool개 중 n=2/3/4 조합을 비교해 target_duration에 가장 가까운 조합을 고른다."""
    if not candidates:
        return []

    pool = rank_by_importance(candidates)[:max_pool]

    best_combo: tuple[dict, ...] | None = None
    best_diff: float | None = None
    for size in (2, 3, 4):
        if size > len(pool):
            continue
        for combo in itertools.combinations(pool, size):
            total = sum(float(clip.get("duration_sec") or 0.0) for clip in combo)
            diff = abs(total - target_duration)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_combo = combo

    if best_combo is None:
        return pool

    return sorted(best_combo, key=lambda clip: float(clip.get("timestamp_sec") or 0.0))


def remove_clip(clips: list[dict], target_index: int | None) -> list[dict]:
    """target_index는 1-based ("두번째"=2)."""
    if not target_index or target_index < 1 or target_index > len(clips):
        raise ValueError(f"유효하지 않은 target_index입니다: {target_index}")
    return [clip for position, clip in enumerate(clips, start=1) if position != target_index]


def adjust_clip_duration(clip: dict, delta_sec: float) -> dict:
    """end_sec을 delta_sec만큼 늘리거나 줄인다. metadata.max_end_sec을 넘지 않고,
    start_sec+1초보다 짧아지지 않도록 clamp한다."""
    metadata = clip.get("metadata") or {}
    start_sec = float(clip["start_sec"])
    max_end_sec = float(metadata.get("max_end_sec", clip["end_sec"]))

    new_end_sec = float(clip["end_sec"]) + float(delta_sec)
    new_end_sec = min(max(new_end_sec, start_sec + 1.0), max_end_sec)

    updated = dict(clip)
    updated["end_sec"] = new_end_sec
    updated["duration_sec"] = new_end_sec - start_sec
    return updated


def compute_total_duration(clips: list[dict]) -> float:
    return sum(float(clip.get("duration_sec") or 0.0) for clip in clips)


def get_current_state(state: dict) -> dict:
    """세션 State에서 현재 편집 상태 요약을 반환하는 순수 함수 (API/테스트 스크립트 조회용)."""
    current_clips = state.get("current_clips") or []
    return {
        "current_clips": current_clips,
        "total_duration_sec": compute_total_duration(current_clips),
        "status": state.get("status"),
        "retry_count": state.get("retry_count", 0),
    }
