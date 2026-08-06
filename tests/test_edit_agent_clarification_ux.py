"""되묻기 남용·목표 길이 소실·재료 부족 실패를 막는 UX 방어선을 검증한다.

회귀 대상 (화면 시나리오):
1. '1분 30초짜리 개쩌는 하이라이트 만들어줘' -> "어떤 장면 종류를 원하는지..." 되묻기
2. 되묻기에 'penalty' 답변 -> 1분30초가 소실되고 '페널티 1개로 구성'
3. '너가 생각했을때 조회수 잘나올거 같은거' 답변 -> 또 되묻기 (무한 루프)
LLM 호출 없이 결정적 계층만 검사한다.
"""

from app.ai.agents.clip_tools import compute_total_duration, select_clip_combination
from app.ai.agents.edit_workflow_agent import (
    _apply_actionability_guard,
    _default_clarification_options,
    _extract_target_duration_sec,
    plan_or_revise_edit,
    validate_plan,
)


def _clip(event_id: str, label: str, timestamp: float, duration: float, score: float) -> dict:
    return {
        "timeline_event_id": event_id,
        "label": label,
        "half": 1,
        "timestamp_sec": timestamp,
        "start_sec": timestamp,
        "end_sec": timestamp + duration,
        "duration_sec": duration,
        "highlight_score": score,
        "metadata": {},
    }


# 화면 데이터 축소판: 페널티는 1개(45초)뿐
ALL_EVENTS = [
    _clip("g1", "goal", 100.0, 45.0, 0.95),
    _clip("g2", "goal", 900.0, 40.0, 0.90),
    _clip("s1", "shot", 200.0, 22.0, 0.85),
    _clip("s2", "shot", 400.0, 22.0, 0.80),
    _clip("s3", "shot", 600.0, 22.0, 0.75),
    _clip("p1", "penalty", 300.0, 45.0, 0.70),
    _clip("c1", "corner", 500.0, 28.0, 0.65),
    _clip("c2", "corner", 700.0, 28.0, 0.60),
]


class TestDurationExtraction:
    def test_minute_second_forms(self):
        assert _extract_target_duration_sec("1분 30초짜리 하이라이트") == 90
        assert _extract_target_duration_sec("1분30초짜리") == 90
        assert _extract_target_duration_sec("2분짜리로 만들어줘") == 120
        assert _extract_target_duration_sec("90초로 만들어줘") == 90

    def test_no_duration(self):
        assert _extract_target_duration_sec("골 장면만 보여줘") is None


class TestClarificationCancellation:
    def test_subjective_build_request_is_not_interrogated(self):
        """화면 문제 1: 목표 길이 있는 주관적 요청은 되묻지 않고 바로 만든다."""
        payload = _apply_actionability_guard(
            {
                "intent_type": "build",
                "labels": None,
                "needs_clarification": True,
                "clarification_question": "어떤 장면 종류를 원하는지 말씀해 주실 수 있나요?",
            },
            "1분 30초짜리 개쩌는 하이라이트 영상 만들어줘",
        )
        assert payload["needs_clarification"] is False
        assert payload["intent_type"] == "build"
        assert payload["target_duration"] == 90

    def test_delegation_answer_never_reasks(self):
        """화면 문제 3: '너가 골라줘'는 위임이지 모호함이 아니다."""
        payload = _apply_actionability_guard(
            {"intent_type": "filter", "labels": None, "needs_clarification": True},
            "1분 30초짜리 개쩌는 하이라이트 영상 만들어줘 (선택한 조건: 너가 생각했을때 조회수 잘나올거 같은거)",
        )
        assert payload["needs_clarification"] is False
        assert payload["intent_type"] == "build"
        assert payload["target_duration"] == 90

    def test_truly_unactionable_request_still_clarifies(self):
        """제작 동사도 길이도 위임도 없는 요청은 되묻기를 유지한다."""
        payload = _apply_actionability_guard(
            {"intent_type": "filter", "labels": None, "needs_clarification": True},
            "임팩트 있는 장면만 보여줘",
        )
        assert payload["needs_clarification"] is True

    def test_labels_from_llm_are_kept_when_cancelling(self):
        payload = _apply_actionability_guard(
            {"intent_type": "filter", "labels": ["goal"], "needs_clarification": True},
            "멋진 골 장면으로 만들어줘",
        )
        assert payload["needs_clarification"] is False
        assert payload["intent_type"] == "filter"
        assert payload["labels"] == ["goal"]


class TestDurationRestoration:
    def test_resumed_answer_keeps_original_duration(self):
        """화면 문제 2: 되묻기 왕복 후에도 1분30초가 유지된다."""
        payload = _apply_actionability_guard(
            {"intent_type": "build", "labels": ["penalty"], "needs_clarification": False},
            "1분 30초짜리 하이라이트 영상 만들어줘. 조회수 많이 나오는 영상으로 만들어줘 (선택한 조건: penalty)",
        )
        assert payload["target_duration"] == 90

    def test_restoration_respects_existing_values(self):
        payload = _apply_actionability_guard(
            {"intent_type": "build", "labels": None, "target_duration": 60},
            "1분 30초짜리로",
        )
        assert payload["target_duration"] == 60

    def test_restoration_skips_count_based_requests(self):
        payload = _apply_actionability_guard(
            {
                "intent_type": "build",
                "labels": ["shot"],
                "label_counts": [{"label": "shot", "count": 2}],
            },
            "슈팅 2개 30초 정도로",
        )
        assert payload.get("target_duration") is None

    def test_adjust_is_untouched(self):
        """'2번째 클립 5초 늘려줘'의 5초가 목표 길이로 오인되면 안 된다."""
        payload = _apply_actionability_guard(
            {"intent_type": "adjust", "target_index": 2, "adjust_delta_sec": 5},
            "2번째 클립 5초 늘려줘",
        )
        assert payload.get("target_duration") is None


class TestDurationShortfall:
    def test_insufficient_pool_delivers_everything_with_explanation(self):
        """페널티 45초뿐인데 90초 요청: 실패 대신 전부 담고 사실대로 알린다."""
        intent = {"intent_type": "build", "labels": ["penalty"], "target_duration": 90}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": ALL_EVENTS, "current_clips": []}
        )
        plan = result["plan_result"]
        assert len(plan["clips"]) == 1
        assert "45초뿐이라" in plan["reply_text"]
        assert "90초를 다 채우진 못했어요" in plan["reply_text"]
        verdict = validate_plan({"plan_result": plan, "intent": intent, "retry_count": 0})
        assert verdict["validation_ok"] is True

    def test_sufficient_pool_still_validates_duration(self):
        intent = {"intent_type": "build", "labels": ["shot"], "target_duration": 44}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": ALL_EVENTS, "current_clips": []}
        )
        plan = result["plan_result"]
        assert plan["duration_shortfall"] is False
        assert compute_total_duration(plan["clips"]) == 44.0


class TestBestJudgmentBuild:
    def test_no_label_build_uses_importance_and_says_so(self):
        """'개쩌는 하이라이트 90초' -> 전 라벨 중요도 순으로 채우고 그렇게 골랐다고 밝힌다."""
        intent = {"intent_type": "build", "labels": None, "target_duration": 90}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": ALL_EVENTS, "current_clips": []}
        )
        plan = result["plan_result"]
        total = compute_total_duration(plan["clips"])
        assert abs(total - 90) <= max(10.0, 90 * 0.3)
        verdict = validate_plan({"plan_result": plan, "intent": intent, "retry_count": 0})
        assert verdict["validation_ok"] is True

    def test_no_label_count_build_mentions_agent_choice(self):
        intent = {"intent_type": "build", "labels": None}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": ALL_EVENTS, "current_clips": []}
        )
        assert result["plan_result"]["reply_text"].startswith("중요도가 높은 순으로 골라")


class TestLongTargets:
    def test_combination_search_reaches_beyond_four_clips(self):
        """예전 4개 조합 한계(~2분)를 넘어 3분짜리도 채울 수 있다."""
        many = [
            _clip(f"s{i}", "shot", 100.0 * i, 25.0, 0.9 - i * 0.01) for i in range(10)
        ]
        picked = select_clip_combination(many, 180.0)
        assert len(picked) > 4
        assert abs(compute_total_duration(picked) - 180.0) <= 10.0


class TestClarificationOptions:
    def test_default_options_from_real_data_plus_delegate(self):
        options = _default_clarification_options({"all_events": ALL_EVENTS})
        assert options == ["골", "슈팅", "페널티", "코너킥", "알아서 골라줘"]
