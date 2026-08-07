"""라벨 기반 제거("골 장면은 빼줘")가 동작하는지 검증한다.

회귀 대상: remove가 순번(target_index)만 지원해서 "골 장면은 빼줘"가
ValueError -> 빈 결과 -> 5회 헛 재시도 -> "'goal' 라벨의 장면을 찾지 못했어요"라는
검색 실패용 오답 메시지로 끝나던 문제. LLM 호출 없이 노드와 순수 함수만 검사한다.
"""

import pytest

from app.ai.agents.clip_tools import remove_clips_by_labels, remove_nth_clip_of_labels
from app.ai.agents.edit_workflow_agent import (
    fail_respond,
    plan_or_revise_edit,
    validate_plan,
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


# 화면의 실제 구성본: 코너킥/골/슈팅/골/슈팅/골/코너킥 (7개)
CURRENT = [
    _clip("c1", "corner", 2547.0),
    _clip("g1", "goal", 2603.0),
    _clip("s1", "shot", 2757.0),
    _clip("g2", "goal", 4033.0),
    _clip("s2", "shot", 4577.0),
    _clip("g3", "goal", 5884.0),
    _clip("c2", "corner", 6186.0),
]


def _run_remove(intent_fields: dict, current: list[dict] | None = None) -> dict:
    result = plan_or_revise_edit(
        {
            "intent": {"intent_type": "remove", **intent_fields},
            "all_events": CURRENT,
            "current_clips": CURRENT if current is None else current,
        }
    )
    return result["plan_result"]


class TestRemoveTools:
    def test_remove_clips_by_labels(self):
        remaining = remove_clips_by_labels(CURRENT, ["goal"])
        assert [clip["timeline_event_id"] for clip in remaining] == ["c1", "s1", "s2", "c2"]

    def test_remove_nth_clip_of_labels(self):
        remaining = remove_nth_clip_of_labels(CURRENT, ["goal"], 2)
        ids = [clip["timeline_event_id"] for clip in remaining]
        assert "g2" not in ids and len(ids) == 6

    def test_remove_nth_out_of_range_raises(self):
        with pytest.raises(ValueError):
            remove_nth_clip_of_labels(CURRENT, ["goal"], 4)


class TestRemoveByLabel:
    def test_goal_scenes_are_removed(self):
        """'골 장면은 빼줘' -> 골 3개가 빠지고 4개가 남는다."""
        plan = _run_remove({"labels": ["goal"], "target_index": None})
        labels = [clip["label"] for clip in plan["clips"]]
        assert labels == ["corner", "shot", "shot", "corner"]
        assert "골 장면 3개를 뺐어요" in plan["reply_text"]
        assert "슈팅 2개" in plan["reply_text"] and "코너킥 2개" in plan["reply_text"]

    def test_korean_label_from_llm_is_normalized(self):
        plan = _run_remove({"labels": ["골"]})
        assert len(plan["clips"]) == 4

    def test_multiple_labels_removed_at_once(self):
        plan = _run_remove({"labels": ["goal", "corner"]})
        assert [clip["label"] for clip in plan["clips"]] == ["shot", "shot"]

    def test_label_not_in_composition_keeps_clips(self):
        """구성에 없는 라벨을 빼달라고 하면 구성을 건드리지 않고 알려만 준다."""
        plan = _run_remove({"labels": ["penalty"]})
        assert plan["clips"] == CURRENT
        assert "없어서 뺄 게 없어요" in plan["reply_text"]

    def test_removing_everything_is_a_valid_result(self):
        """전부 빼서 구성이 비어도 재시도 루프에 빠지지 않는다."""
        intent = {"intent_type": "remove", "labels": ["goal", "shot", "corner"]}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": CURRENT, "current_clips": CURRENT}
        )
        assert result["plan_result"]["clips"] == []
        assert "구성이 비었어요" in result["plan_result"]["reply_text"]
        verdict = validate_plan(
            {"plan_result": result["plan_result"], "intent": intent, "retry_count": 0}
        )
        assert verdict["validation_ok"] is True

    def test_nth_of_label_removal(self):
        """'두번째 골 빼줘' -> 골 중 2번째(g2)만 제거."""
        plan = _run_remove({"labels": ["goal"], "target_index": 2})
        ids = [clip["timeline_event_id"] for clip in plan["clips"]]
        assert "g2" not in ids and len(ids) == 6
        assert "2번째 골 클립을 뺐어요" in plan["reply_text"]

    def test_nth_of_label_out_of_range_keeps_clips(self):
        plan = _run_remove({"labels": ["goal"], "target_index": 5})
        assert plan["clips"] == CURRENT
        assert "3개뿐이라" in plan["reply_text"]


class TestRemoveByIndexStillWorks:
    def test_index_removal_unchanged(self):
        plan = _run_remove({"target_index": 1})
        assert [clip["timeline_event_id"] for clip in plan["clips"]][0] == "g1"
        assert "1번째 클립을 제거했어요" in plan["reply_text"]

    def test_missing_index_keeps_clips_and_does_not_retry(self):
        """순번도 라벨도 없으면 구성을 비우지 않고 되묻는다. 재시도 루프 금지."""
        intent = {"intent_type": "remove", "target_index": None}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": CURRENT, "current_clips": CURRENT}
        )
        assert result["plan_result"]["clips"] == CURRENT
        assert "몇 번째 클립을 뺄지 찾지 못했어요" in result["plan_result"]["reply_text"]
        verdict = validate_plan(
            {"plan_result": result["plan_result"], "intent": intent, "retry_count": 0}
        )
        assert verdict["validation_ok"] is True


class TestAdjustFailureNoLongerDestructive:
    def test_bad_index_keeps_clips_and_passes_validation(self):
        intent = {"intent_type": "adjust", "target_index": 99, "adjust_delta_sec": 5}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": CURRENT, "current_clips": CURRENT}
        )
        assert result["plan_result"]["clips"] == CURRENT
        verdict = validate_plan(
            {"plan_result": result["plan_result"], "intent": intent, "retry_count": 0}
        )
        assert verdict["validation_ok"] is True


class TestFailRespondMessage:
    def test_remove_failure_is_not_worded_as_search_failure(self):
        """만에 하나 fail_respond까지 가도 '장면을 찾지 못했다'는 오답 문구를 쓰지 않는다."""
        result = fail_respond(
            {
                "intent": {"intent_type": "remove", "labels": ["goal"]},
                "all_events": CURRENT,
            }
        )
        assert "라벨의 장면을 찾지 못했어요" not in result["final_response"]
        assert "다시 알려주시겠어요" in result["final_response"]
