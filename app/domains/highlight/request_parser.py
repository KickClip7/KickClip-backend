from __future__ import annotations

import re
from typing import Any

from app.domains.highlight.schema import (
    HighlightFocusMode,
    HighlightRequest,
    HighlightScope,
    HighlightScopeType,
)


EVENT_KEYWORDS = {
    "goal": ("골", "득점", "goal"),
    "shot": ("슛", "슈팅", "shot"),
    "penalty": ("페널티", "패널티", "penalty", "pk"),
    "foul": ("파울", "반칙", "foul"),
    "free_kick": ("프리킥", "free kick", "free_kick"),
    "corner": ("코너", "코너킥", "corner"),
    "card": ("카드", "옐로", "레드", "card"),
}


class HighlightRequestParser:
    """Conservative rule parser for edit intent.

    Period-relative requests remain relative until real Match metadata supplies
    the boundary. In particular, this parser never converts "first half" into
    an assumed 0..2700 source-video interval.
    """

    def parse(
        self,
        text: str,
        *,
        match_metadata: dict[str, Any] | None = None,
        current_revision_exists: bool = False,
    ) -> HighlightRequest:
        normalized = (text or "").strip().lower()
        metadata = match_metadata or {}
        scope = self._parse_scope(
            normalized,
            metadata,
            current_revision_exists=current_revision_exists,
        )
        labels = [
            label
            for label, keywords in EVENT_KEYWORDS.items()
            if any(keyword in normalized for keyword in keywords)
        ]
        duration = self._parse_duration(normalized)
        player_focus = any(
            keyword in normalized
            for keyword in ("선수", "줌", "따라가", "포커스", "player", "zoom")
        )
        ratio = "9:16"
        if "1:1" in normalized or "정사각" in normalized:
            ratio = "1:1"
        elif "16:9" in normalized or "가로" in normalized:
            ratio = "16:9"

        return HighlightRequest(
            scope=scope,
            event_labels=labels,
            desired_duration_sec=duration,
            focus_mode=(
                HighlightFocusMode.PLAYER
                if player_focus
                else HighlightFocusMode.NONE
            ),
            aspect_ratio=ratio,
            parser_provenance={
                "parser": "conservative_rule_parser_v1",
                "period_boundary_assumed": False,
                "references_current_scene_set": scope.type
                == HighlightScopeType.SELECTED_SCENES,
            },
        )

    def _parse_scope(
        self,
        text: str,
        metadata: dict[str, Any],
        *,
        current_revision_exists: bool,
    ) -> HighlightScope:
        if current_revision_exists and any(
            phrase in text
            for phrase in ("이 장면", "방금 고른", "선택한 장면", "these scenes")
        ):
            return HighlightScope(type=HighlightScopeType.SELECTED_SCENES)

        custom = self._parse_explicit_source_range(text)
        if custom is not None:
            return custom

        if "전반" in text or "first half" in text:
            return self._period_scope(
                HighlightScopeType.FIRST_HALF,
                metadata,
                relative_start=self._parse_relative_minute(text, "전반"),
            )
        if "후반" in text or "second half" in text:
            return self._period_scope(
                HighlightScopeType.SECOND_HALF,
                metadata,
                relative_start=self._parse_relative_minute(text, "후반"),
            )
        return HighlightScope(type=HighlightScopeType.FULL_MATCH)

    @staticmethod
    def _parse_explicit_source_range(text: str) -> HighlightScope | None:
        match = re.search(
            r"(?:영상\s*)?(\d+(?:\.\d+)?)\s*초\s*(?:부터|~|-)\s*"
            r"(\d+(?:\.\d+)?)\s*초",
            text,
        )
        if not match:
            return None
        return HighlightScope(
            type=HighlightScopeType.CUSTOM_RANGE,
            start_time_sec=float(match.group(1)),
            end_time_sec=float(match.group(2)),
            boundary_source="USER_EXPLICIT",
        )

    def _period_scope(
        self,
        scope_type: HighlightScopeType,
        metadata: dict[str, Any],
        *,
        relative_start: float | None,
    ) -> HighlightScope:
        boundaries = self._period_boundaries(metadata)
        key = "first_half" if scope_type == HighlightScopeType.FIRST_HALF else "second_half"
        boundary = boundaries.get(key)
        if boundary is None:
            return HighlightScope(
                type=scope_type,
                relative_start_sec=relative_start,
                requires_period_boundary=True,
            )

        start, end = boundary
        if relative_start is not None:
            start = min(start + relative_start, end)
        return HighlightScope(
            type=scope_type,
            start_time_sec=start,
            end_time_sec=end,
            relative_start_sec=relative_start,
            boundary_source="MATCH_METADATA",
            requires_period_boundary=False,
        )

    @staticmethod
    def _period_boundaries(
        metadata: dict[str, Any],
    ) -> dict[str, tuple[float, float]]:
        raw = metadata.get("period_boundaries") or metadata.get("periods") or {}
        result: dict[str, tuple[float, float]] = {}
        aliases = {
            "first_half": ("first_half", "FIRST_HALF", "1"),
            "second_half": ("second_half", "SECOND_HALF", "2"),
        }
        for target, keys in aliases.items():
            value = next((raw.get(key) for key in keys if raw.get(key) is not None), None)
            if isinstance(value, dict):
                start = value.get("start_time_sec", value.get("start_sec"))
                end = value.get("end_time_sec", value.get("end_sec"))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                start, end = value
            else:
                continue
            try:
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue
            if 0 <= start_value < end_value:
                result[target] = (start_value, end_value)
        return result

    @staticmethod
    def _parse_relative_minute(text: str, period_word: str) -> float | None:
        pattern = rf"{period_word}\s*(\d+)\s*분(?:부터|이후)"
        match = re.search(pattern, text)
        return float(match.group(1)) * 60.0 if match else None

    @staticmethod
    def _parse_duration(text: str) -> float:
        match = re.search(r"(\d+)\s*초", text)
        if match:
            return min(max(float(match.group(1)), 1.0), 600.0)
        if "짧게" in text:
            return 30.0
        return 60.0
