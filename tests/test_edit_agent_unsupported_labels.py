"""지원하지 않는 장면(프리킥 등)이 비슷한 라벨로 대체되지 않는지 검증한다.

'프리킥 영상만 모아줘' -> "'corner' 조건으로 5개 장면을 찾았어요" 회귀 방지.
LLM 호출 없이 결정적 방어선(_guard_unsupported_labels)과 그래프 노드만 검사한다.
"""

import pytest

from app.ai.agents.edit_workflow_agent import (
    _guard_unsupported_labels,
    plan_or_revise_edit,
    validate_plan,
)
from app.ai.agents.label_taxonomy import analyze_scene_terms, normalize_label


def _clip(event_id: str, label: str, timestamp: float) -> dict:
    return {
        "timeline_event_id": event_id,
        "label": label,
        "half": 1,
        "timestamp_sec": timestamp,
        "start_sec": timestamp,
        "end_sec": timestamp + 20.0,
        "duration_sec": 20.0,
        "highlight_score": 0.8,
        "metadata": {},
    }


ALL_EVENTS = [
    _clip("evt_goal", "goal", 100.0),
    _clip("evt_shot", "shot", 200.0),
    _clip("evt_corner", "corner", 300.0),
]


class TestSceneTermAnalysis:
    def test_free_kick_is_unsupported(self):
        unsupported, labels = analyze_scene_terms("프리킥 영상만 모아줘")
        assert unsupported == ["프리킥"]
        assert labels == []

    @pytest.mark.parametrize("message", ["골킥 장면 보여줘", "골키퍼 선방 모아줘"])
    def test_goal_lookalikes_do_not_leak_into_goal(self, message):
        """'골킥'/'골키퍼' 안의 '골'이 goal 라벨로 새면 안 된다."""
        unsupported, labels = analyze_scene_terms(message)
        assert unsupported
        assert "goal" not in labels

    def test_corner_kick_still_resolves(self):
        unsupported, labels = analyze_scene_terms("코너킥만 보여줘")
        assert unsupported == []
        assert labels == ["corner"]

    def test_mixed_request_keeps_supported_label(self):
        unsupported, labels = analyze_scene_terms("프리킥이랑 골 장면 보여줘")
        assert unsupported == ["프리킥"]
        assert labels == ["goal"]

    def test_normalize_label_rejects_retired_labels(self):
        assert normalize_label("Corner") == "corner"
        assert normalize_label("득점") == "goal"
        assert normalize_label("free_kick") is None


class TestUnsupportedGuard:
    def test_llm_substitution_is_reverted(self):
        """LLM이 프리킥을 corner로 바꿔치기해도 원문 기준으로 되돌린다."""
        payload = _guard_unsupported_labels(
            {"intent_type": "filter", "labels": ["corner"], "needs_clarification": False},
            "프리킥 영상만 모아줘",
        )
        assert payload["intent_type"] == "unsupported"
        assert payload["labels"] is None
        assert payload["unsupported_terms"] == ["프리킥"]

    def test_unsupported_never_asks_a_clarifying_question(self):
        payload = _guard_unsupported_labels(
            {
                "intent_type": "filter",
                "labels": ["corner"],
                "needs_clarification": True,
                "clarification_question": "전반과 후반 중 어느 쪽에서 원하시나요?",
                "clarification_options": ["전반", "후반"],
            },
            "프리킥 영상만 모아줘",
        )
        assert payload["needs_clarification"] is False
        assert payload["clarification_question"] is None

    def test_mixed_request_drops_only_unsupported_part(self):
        payload = _guard_unsupported_labels(
            {"intent_type": "filter", "labels": ["corner"], "needs_clarification": False},
            "프리킥이랑 골 장면 보여줘",
        )
        assert payload["intent_type"] == "filter"
        assert payload["labels"] == ["goal"]
        assert payload["unsupported_terms"] == ["프리킥"]

    def test_supported_request_is_left_untouched(self):
        original = {"intent_type": "filter", "labels": ["goal"], "needs_clarification": False}
        assert _guard_unsupported_labels(dict(original), "골 장면만 보여줘") == original


class TestUnsupportedPlanning:
    def test_plan_preserves_current_clips_and_explains(self):
        current = [ALL_EVENTS[0]]
        result = plan_or_revise_edit(
            {
                "intent": {"intent_type": "unsupported", "unsupported_terms": ["프리킥"]},
                "all_events": ALL_EVENTS,
                "current_clips": current,
            }
        )
        assert result["plan_result"]["clips"] == current
        reply = result["plan_result"]["reply_text"]
        assert "프리킥" in reply
        assert "코너킥" in reply and "골" in reply
        assert "corner" not in reply

    def test_mixed_request_reports_dropped_term(self):
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "filter",
                    "labels": ["goal"],
                    "unsupported_terms": ["프리킥"],
                },
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        assert result["plan_result"]["clips"] == [ALL_EVENTS[0]]
        assert "제외했어요" in result["plan_result"]["reply_text"]

    def test_unsupported_does_not_enter_retry_loop(self):
        """구성이 비어 있어도 재계획 5회를 돌지 않고 바로 응답해야 한다."""
        result = validate_plan(
            {
                "plan_result": {"clips": [], "reply_text": "..."},
                "intent": {"intent_type": "unsupported", "unsupported_terms": ["프리킥"]},
                "retry_count": 0,
            }
        )
        assert result["validation_ok"] is True

    def test_genuinely_empty_filter_still_retries(self):
        result = validate_plan(
            {
                "plan_result": {"clips": [], "reply_text": "..."},
                "intent": {"intent_type": "filter", "labels": ["penalty"]},
                "retry_count": 0,
            }
        )
        assert result["validation_ok"] is False
        assert result["retry_count"] == 1
