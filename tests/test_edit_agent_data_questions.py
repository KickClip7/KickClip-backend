"""'해당 경기에 골 몇 개 있어?' 같은 데이터 질문에 실제 데이터로 답하는지 검증한다.

회귀 대상: 데이터 질문이 chitchat으로 분류되지만 LLM이 개수 정보를 받지 못해
"경기 정보를 제공해 주시면 확인해 드릴 수 있습니다" 같은 무의미한 답이 나오던 문제.
LLM 호출 없이 통계 헬퍼와 계획 노드만 검사한다.
"""

from app.ai.agents.edit_workflow_agent import (
    _answer_data_question,
    build_match_stats_summary,
    plan_or_revise_edit,
)


def _clip(event_id: str, label: str, timestamp: float) -> dict:
    return {
        "timeline_event_id": event_id,
        "label": label,
        "half": 1,
        "timestamp_sec": timestamp,
        "start_sec": timestamp,
        "end_sec": timestamp + 22.0,
        "duration_sec": 22.0,
        "highlight_score": 0.8,
        "metadata": {},
    }


# 화면 데이터 축소판: 골 3개 (43:23, 67:13, 98:04 지점)
ALL_EVENTS = [
    _clip("g1", "goal", 2603.0),
    _clip("g2", "goal", 4033.0),
    _clip("g3", "goal", 5884.0),
    _clip("s1", "shot", 200.0),
    _clip("s2", "shot", 400.0),
    _clip("c1", "corner", 500.0),
]


class TestDataQuestionAnswers:
    def test_goal_count_question_is_answered_from_data(self):
        """화면 시나리오: '해당 경기에 골 몇개 있어?' -> 3개 + 시각까지."""
        answer = _answer_data_question("해당 경기에 골 몇개 있어?", ALL_EVENTS, [])
        assert answer is not None
        assert "골 장면은 3개예요" in answer
        assert "43:23" in answer and "67:13" in answer and "98:04" in answer

    def test_answer_suggests_next_action(self):
        answer = _answer_data_question("골 몇 개 있어?", ALL_EVENTS, [])
        assert "골 장면만 보여줘" in answer

    def test_variant_spelling_still_matches(self):
        """퍼지 계층 재사용: '슛팅 몇개?'도 shot으로 해석된다."""
        answer = _answer_data_question("슛팅 몇개 있어?", ALL_EVENTS, [])
        assert "슈팅 장면은 2개예요" in answer

    def test_zero_count_label_is_honest(self):
        answer = _answer_data_question("페널티 몇 개 있어?", ALL_EVENTS, [])
        assert "페널티 장면은 찾지 못했어요" in answer

    def test_overall_count_question(self):
        answer = _answer_data_question("장면이 총 몇개야?", ALL_EVENTS, [])
        assert "총 6개" in answer
        assert "골 3개" in answer

    def test_non_question_returns_none(self):
        assert _answer_data_question("안녕", ALL_EVENTS, []) is None
        assert _answer_data_question("골 장면만 보여줘", ALL_EVENTS, []) is None

    def test_context_words_do_not_leak_into_labels(self):
        """'해당 경기에'의 '경기'가 카드 별칭 '경고'로 퍼지 오탐되면 안 된다."""
        answer = _answer_data_question("해당 경기에 골 몇개 있어?", ALL_EVENTS, [])
        assert "골 장면은 3개예요" in answer
        assert "카드" not in answer


class TestStatsSummary:
    def test_summary_contains_counts_and_timestamps(self):
        summary = build_match_stats_summary(ALL_EVENTS, [])
        assert "총 6개" in summary
        assert "골 3개(43:23, 67:13, 98:04)" in summary
        assert "구성본은 비어 있습니다" in summary

    def test_summary_reports_current_composition(self):
        summary = build_match_stats_summary(ALL_EVENTS, [ALL_EVENTS[0], ALL_EVENTS[3]])
        assert "현재 하이라이트 구성본: 2개" in summary
        assert "총 44초" in summary


class TestChitchatIntegration:
    def test_count_question_overrides_llm_response(self):
        """LLM이 무의미한 답을 만들어도 데이터 기반 답변이 우선한다."""
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "chitchat",
                    "response_text": "경기 정보를 제공해 주시면 확인해 드릴 수 있습니다!",
                },
                "all_events": ALL_EVENTS,
                "current_clips": [],
                "user_message": "해당 경기에 골 몇개 있어?",
            }
        )
        reply = result["plan_result"]["reply_text"]
        assert "골 장면은 3개예요" in reply
        assert "경기 정보를 제공해" not in reply

    def test_composition_is_untouched_by_questions(self):
        current = [ALL_EVENTS[0], ALL_EVENTS[3]]
        result = plan_or_revise_edit(
            {
                "intent": {"intent_type": "chitchat", "response_text": "..."},
                "all_events": ALL_EVENTS,
                "current_clips": current,
                "user_message": "골 몇개 있어?",
            }
        )
        assert result["plan_result"]["clips"] == current

    def test_plain_chitchat_still_uses_llm_response(self):
        result = plan_or_revise_edit(
            {
                "intent": {"intent_type": "chitchat", "response_text": "안녕하세요!"},
                "all_events": ALL_EVENTS,
                "current_clips": [],
                "user_message": "안녕",
            }
        )
        assert result["plan_result"]["reply_text"] == "안녕하세요!"
