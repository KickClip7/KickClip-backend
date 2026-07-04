from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.domains.player.model import Player
from app.domains.timeline.model import TimelineEvent


FRONTEND_TO_BACKEND_LABEL = {
    "goal": "goal",
    "shot": "shot",
    "foul": "foul",
    "card": "card",
    "freekick": "free_kick",
    "free_kick": "free_kick",
    "corner": "corner",
}

LABEL_PRIORITY = {
    "goal": 100,
    "shot": 80,
    "free_kick": 70,
    "corner": 60,
    "foul": 50,
    "card": 40,
}


@dataclass(frozen=True)
class PlannedClipItem:
    timeline_event_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    reason: str


@dataclass(frozen=True)
class PlannedClipPlan:
    summary: str
    target_duration_sec: float | None
    total_duration_sec: float
    items: list[PlannedClipItem] = field(default_factory=list)
    parsed_labels: list[str] = field(default_factory=list)
    parsed_half: int | None = None
    selected_player_id: str | None = None


class RuleBasedClipPlanner:
    """초기 Agent용 rule-based clip planner.

    10회차에서는 LLM 없이 프롬프트 키워드와 기존 TimelineEvent 점수를 기반으로
    ClipPlan 후보를 만든다. 이후 LLM Agent로 바꿔도 API 응답 구조는 유지한다.
    """

    def build_plan(
        self,
        *,
        prompt: str,
        events: list[TimelineEvent],
        players: list[Player],
        target_duration_sec: int | float | None = 30,
        selected_player_id: str | None = None,
        options: dict | None = None,
    ) -> PlannedClipPlan:
        options = options or {}
        normalized_prompt = self._normalize(prompt)

        target_duration = self._parse_target_duration(
            normalized_prompt=normalized_prompt,
            fallback=target_duration_sec,
        )
        parsed_labels = self._parse_labels(normalized_prompt, options)
        parsed_half = self._parse_half(normalized_prompt)
        inferred_player_id = selected_player_id or self._infer_player_id_from_prompt(
            normalized_prompt,
            players,
        )

        filtered_events = self._filter_events(
            events=events,
            labels=parsed_labels,
            half=parsed_half,
            selected_player_id=inferred_player_id,
        )

        ranked_events = sorted(
            filtered_events,
            key=lambda event: self._ranking_key(event, inferred_player_id),
            reverse=True,
        )

        max_items = int(options.get("max_items") or 12)
        planned_items = self._pick_until_target(
            events=ranked_events,
            target_duration_sec=target_duration,
            max_items=max_items,
            selected_player_id=inferred_player_id,
        )

        total_duration = round(sum(item.duration_sec for item in planned_items), 2)
        summary = self._build_summary(
            prompt=prompt,
            labels=parsed_labels,
            half=parsed_half,
            selected_player_id=inferred_player_id,
            total_duration=total_duration,
            item_count=len(planned_items),
        )

        return PlannedClipPlan(
            summary=summary,
            target_duration_sec=target_duration,
            total_duration_sec=total_duration,
            items=planned_items,
            parsed_labels=parsed_labels,
            parsed_half=parsed_half,
            selected_player_id=inferred_player_id,
        )

    def _filter_events(
        self,
        *,
        events: list[TimelineEvent],
        labels: list[str],
        half: int | None,
        selected_player_id: str | None,
    ) -> list[TimelineEvent]:
        filtered: list[TimelineEvent] = []

        for event in events:
            if labels and event.label not in labels:
                continue

            if half is not None and event.half != half:
                continue

            if selected_player_id and selected_player_id not in (event.player_ids or []):
                continue

            filtered.append(event)

        # 선수 필터를 걸었는데 결과가 0개라면, 프론트 UX를 위해 같은 조건의 선수 필터만 완화한다.
        if selected_player_id and not filtered:
            return self._filter_events(
                events=events,
                labels=labels,
                half=half,
                selected_player_id=None,
            )

        return filtered

    def _pick_until_target(
        self,
        *,
        events: list[TimelineEvent],
        target_duration_sec: float | None,
        max_items: int,
        selected_player_id: str | None,
    ) -> list[PlannedClipItem]:
        selected: list[PlannedClipItem] = []
        total = 0.0

        for event in events:
            duration = max(0.0, float(event.end_sec) - float(event.start_sec))
            if duration <= 0:
                duration = max(1.0, float(event.duration_sec or 0.0))

            if selected and target_duration_sec is not None:
                if total + duration > target_duration_sec * 1.15:
                    continue

            reason = self._build_reason(event, selected_player_id)
            selected.append(
                PlannedClipItem(
                    timeline_event_id=event.timeline_event_id,
                    start_sec=float(event.start_sec),
                    end_sec=float(event.end_sec),
                    duration_sec=round(duration, 2),
                    reason=reason,
                )
            )
            total += duration

            if len(selected) >= max_items:
                break
            if target_duration_sec is not None and total >= target_duration_sec:
                break

        return selected

    @staticmethod
    def _ranking_key(
        event: TimelineEvent,
        selected_player_id: str | None,
    ) -> tuple[float, float, float, float]:
        player_bonus = 20.0 if selected_player_id in (event.player_ids or []) else 0.0
        highlight_score = float(event.highlight_score or 0.0)
        confidence = float(event.confidence or 0.0)
        label_priority = float(LABEL_PRIORITY.get(event.label, 0))
        # 최신순이 아니라 점수 우선, 같은 점수에서는 앞 시간 이벤트가 먼저 오도록 timestamp는 음수 처리한다.
        time_score = -float(event.timestamp_sec or 0.0) / 100000.0
        return (
            highlight_score + player_bonus,
            confidence,
            label_priority,
            time_score,
        )

    @staticmethod
    def _build_reason(
        event: TimelineEvent,
        selected_player_id: str | None,
    ) -> str:
        base = f"{event.label} event with high highlight score"
        if selected_player_id and selected_player_id in (event.player_ids or []):
            return f"{base}; selected player is involved"
        return base

    def _parse_labels(self, normalized_prompt: str, options: dict) -> list[str]:
        labels: list[str] = []

        keyword_rules = [
            ("goal", ["골", "득점", "goal"]),
            ("shot", ["슈팅", "슛", "유효슈팅", "shot"]),
            ("foul", ["파울", "반칙", "foul"]),
            ("card", ["카드", "옐로", "레드", "card"]),
            ("free_kick", ["프리킥", "free kick", "free_kick", "freekick"]),
            ("corner", ["코너", "코너킥", "corner"]),
        ]

        for label, keywords in keyword_rules:
            if any(keyword in normalized_prompt for keyword in keywords):
                labels.append(label)

        if labels:
            return labels

        option_labels = options.get("allow_event_types") or []
        normalized_option_labels = [
            FRONTEND_TO_BACKEND_LABEL.get(str(label).lower(), str(label).lower())
            for label in option_labels
        ]
        return [label for label in normalized_option_labels if label != "all"]

    @staticmethod
    def _parse_half(normalized_prompt: str) -> int | None:
        if "전반" in normalized_prompt or "first half" in normalized_prompt:
            return 1
        if "후반" in normalized_prompt or "second half" in normalized_prompt:
            return 2
        return None

    @staticmethod
    def _parse_target_duration(
        *,
        normalized_prompt: str,
        fallback: int | float | None,
    ) -> float | None:
        match = re.search(r"(\d+)\s*초", normalized_prompt)
        if match:
            return float(match.group(1))

        if "짧게" in normalized_prompt or "빠르게" in normalized_prompt:
            return 20.0

        if "쇼츠" in normalized_prompt or "shorts" in normalized_prompt:
            return float(fallback or 30)

        return float(fallback) if fallback is not None else None

    @staticmethod
    def _infer_player_id_from_prompt(
        normalized_prompt: str,
        players: list[Player],
    ) -> str | None:
        for player in players:
            candidate_names = [
                player.display_name,
                f"{player.number}번" if player.number is not None else None,
                str(player.number) if player.number is not None else None,
            ]
            metadata = player.metadata_ or {}
            for alias in metadata.get("aliases", []) or []:
                candidate_names.append(str(alias))

            for candidate in candidate_names:
                if candidate and candidate.lower() in normalized_prompt:
                    return player.player_id

        return None

    @staticmethod
    def _build_summary(
        *,
        prompt: str,
        labels: list[str],
        half: int | None,
        selected_player_id: str | None,
        total_duration: float,
        item_count: int,
    ) -> str:
        if item_count == 0:
            return "조건에 맞는 장면을 찾지 못했습니다."

        parts: list[str] = []
        if labels:
            parts.append("/".join(labels))
        else:
            parts.append("전체 하이라이트")

        if half == 1:
            parts.append("전반")
        elif half == 2:
            parts.append("후반")

        if selected_player_id:
            parts.append("선택 선수 중심")

        return f"{', '.join(parts)} 기준으로 {item_count}개 장면, 약 {total_duration:g}초 ClipPlan을 생성했습니다."

    @staticmethod
    def _normalize(text: str) -> str:
        return (text or "").strip().lower()