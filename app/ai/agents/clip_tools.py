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


def get_clips_by_labels(labels: list[str], candidates: list[dict]) -> list[dict]:
    """labels 중 하나라도 일치하는 클립을 원래 순서 그대로 반환한다(합집합)."""
    normalized = {str(label).strip().lower() for label in labels}
    return [clip for clip in candidates if str(clip.get("label", "")).strip().lower() in normalized]


def select_one_per_label(labels: list[str], candidates: list[dict]) -> list[dict]:
    """각 라벨에서 highlight_score가 가장 높은 클립을 하나씩 골라 시간순으로 반환한다."""
    picks: list[dict] = []
    seen_ids: set[str] = set()
    for label in labels:
        ranked = rank_by_importance(get_clips_by_label(label, candidates))
        for clip in ranked:
            clip_id = clip.get("timeline_event_id")
            if clip_id not in seen_ids:
                picks.append(clip)
                seen_ids.add(clip_id)
                break
    return sort_chronologically(picks)


def select_counts_per_label(label_counts: dict[str, int], candidates: list[dict]) -> list[dict]:
    """라벨별로 요청받은 개수만큼 highlight_score 상위 클립을 골라 시간순으로 반환한다.

    요청 개수보다 실제 장면이 적으면 있는 만큼만 담는다. 모자란 자리를 다른 라벨로
    채우면 사용자가 요청하지 않은 장면이 섞이므로 절대 채우지 않는다.
    """
    picks: list[dict] = []
    seen_ids: set[str] = set()
    for label, count in label_counts.items():
        taken = 0
        for clip in rank_by_importance(get_clips_by_label(label, candidates)):
            if taken >= count:
                break
            clip_id = clip.get("timeline_event_id")
            if clip_id in seen_ids:
                continue
            picks.append(clip)
            seen_ids.add(clip_id)
            taken += 1
    return sort_chronologically(picks)


def get_clips_by_half(half: int, candidates: list[dict]) -> list[dict]:
    return [clip for clip in candidates if clip.get("half") == half]


def rank_by_importance(clips: list[dict]) -> list[dict]:
    return sorted(clips, key=lambda clip: float(clip.get("highlight_score") or 0.0), reverse=True)


def sort_chronologically(clips: list[dict]) -> list[dict]:
    """편집 결과로 사용자에게 보여줄 클립은 항상 경기 시간 순서를 따른다."""
    return sorted(clips, key=lambda clip: float(clip.get("timestamp_sec") or 0.0))


def select_clip_combination(
    candidates: list[dict],
    target_duration: float,
    max_pool: int = 12,
) -> list[dict]:
    """highlight_score 상위 max_pool개 중 1~8개 조합을 비교해 target_duration에 가장 가까운 조합을 고른다.

    조합 크기를 4개로 묶으면 '3분짜리'처럼 긴 목표를 절대 못 채운다. 12개 풀에서
    8개까지의 조합은 수천 개 수준이라 완전탐색해도 비용이 무시할 만하다.
    """
    if not candidates:
        return []

    pool = rank_by_importance(candidates)[:max_pool]

    best_combo: tuple[dict, ...] | None = None
    best_diff: float | None = None
    for size in range(1, min(len(pool), 8) + 1):
        for combo in itertools.combinations(pool, size):
            total = sum(float(clip.get("duration_sec") or 0.0) for clip in combo)
            diff = abs(total - target_duration)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_combo = combo

    if best_combo is None:
        return pool

    return sort_chronologically(list(best_combo))


def remove_clip(clips: list[dict], target_index: int | None) -> list[dict]:
    """target_index는 1-based ("두번째"=2)."""
    if not target_index or target_index < 1 or target_index > len(clips):
        raise ValueError(f"유효하지 않은 target_index입니다: {target_index}")
    return [clip for position, clip in enumerate(clips, start=1) if position != target_index]


def remove_clips_by_labels(clips: list[dict], labels: list[str]) -> list[dict]:
    """'골 장면은 빼줘'처럼 라벨로 지목된 클립을 모두 제외한 나머지를 순서대로 반환한다."""
    normalized = {str(label).strip().lower() for label in labels}
    return [clip for clip in clips if str(clip.get("label", "")).strip().lower() not in normalized]


def remove_nth_clip_of_labels(clips: list[dict], labels: list[str], target_index: int | None) -> list[dict]:
    """'두번째 골 빼줘'처럼 해당 라벨 클립 중 target_index(1-based)번째 하나만 제거한다."""
    normalized = {str(label).strip().lower() for label in labels}
    positions = [
        position
        for position, clip in enumerate(clips)
        if str(clip.get("label", "")).strip().lower() in normalized
    ]
    if not target_index or target_index < 1 or target_index > len(positions):
        raise ValueError(f"유효하지 않은 target_index입니다: {target_index}")
    drop = positions[target_index - 1]
    return [clip for position, clip in enumerate(clips) if position != drop]


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
