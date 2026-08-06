"""라벨별 개수 요청("슈팅 2개와 코너킥 2개")이 그대로 반영되는지 검증한다.

회귀 대상: 다중 라벨 요청이 개수와 무관하게 항상 '라벨당 1개'로 처리되어
"슈팅 2개와 코너킥 2개" -> 슈팅 1 + 골 1 + 코너킥 1 (3개)로 나오던 문제.
LLM 호출 없이 계획/검증 노드와 순수 함수만 검사한다.
"""

import pytest

from app.ai.agents.clip_tools import select_counts_per_label
from app.ai.agents.edit_intent_schema import EditIntent
from app.ai.agents.edit_workflow_agent import (
    describe_label_mix,
    plan_or_revise_edit,
    resolve_label_counts,
    validate_plan,
)


def _clip(event_id: str, label: str, timestamp: float, score: float) -> dict:
    return {
        "timeline_event_id": event_id,
        "label": label,
        "half": 1,
        "timestamp_sec": timestamp,
        "start_sec": timestamp,
        "end_sec": timestamp + 22.0,
        "duration_sec": 22.0,
        "highlight_score": score,
        "metadata": {},
    }


# 화면의 실제 분포와 같은 비율: 골 3, 슈팅 14, 코너킥 8
ALL_EVENTS = (
    [_clip(f"g{i}", "goal", 1000 + i * 100, 0.90 - i * 0.01) for i in range(3)]
    + [_clip(f"s{i}", "shot", 2000 + i * 100, 0.80 - i * 0.01) for i in range(14)]
    + [_clip(f"c{i}", "corner", 3000 + i * 100, 0.70 - i * 0.01) for i in range(8)]
)


def _label_histogram(clips: list[dict]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for clip in clips:
        histogram[clip["label"]] = histogram.get(clip["label"], 0) + 1
    return histogram


class TestLabelCountResolution:
    def test_schema_accepts_per_label_counts(self):
        intent = EditIntent(
            intent_type="build",
            labels=["shot", "corner"],
            label_counts=[{"label": "슈팅", "count": 2}, {"label": "코너킥", "count": 2}],
        )
        assert resolve_label_counts(intent.model_dump()) == {"shot": 2, "corner": 2}

    def test_unknown_labels_and_bad_counts_are_dropped(self):
        intent = {
            "label_counts": [
                {"label": "프리킥", "count": 2},
                {"label": "shot", "count": 0},
                {"label": "corner", "count": 3},
            ]
        }
        assert resolve_label_counts(intent) == {"corner": 3}

    def test_absent_label_counts_is_empty(self):
        assert resolve_label_counts({"intent_type": "build", "labels": ["shot"]}) == {}


class TestSelectCountsPerLabel:
    def test_takes_requested_count_per_label(self):
        picked = select_counts_per_label({"shot": 2, "corner": 2}, ALL_EVENTS)
        assert _label_histogram(picked) == {"shot": 2, "corner": 2}

    def test_picks_highest_scoring_clips(self):
        picked = select_counts_per_label({"shot": 2}, ALL_EVENTS)
        assert [clip["timeline_event_id"] for clip in picked] == ["s0", "s1"]

    def test_result_is_chronological(self):
        picked = select_counts_per_label({"corner": 2, "shot": 2}, ALL_EVENTS)
        stamps = [clip["timestamp_sec"] for clip in picked]
        assert stamps == sorted(stamps)

    def test_shortage_is_not_padded_with_other_labels(self):
        """골이 3개뿐인데 5개를 요청하면 3개만 준다. 다른 라벨로 채우면 안 된다."""
        picked = select_counts_per_label({"goal": 5}, ALL_EVENTS)
        assert _label_histogram(picked) == {"goal": 3}


class TestBuildWithLabelCounts:
    def test_requested_counts_are_honored(self):
        """'슈팅 2개와 코너킥 2개' -> 정확히 4개, 골은 섞이지 않는다."""
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "build",
                    "labels": ["shot", "corner"],
                    "label_counts": [
                        {"label": "shot", "count": 2},
                        {"label": "corner", "count": 2},
                    ],
                },
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        clips = result["plan_result"]["clips"]
        assert len(clips) == 4
        assert _label_histogram(clips) == {"shot": 2, "corner": 2}

    def test_label_counts_win_over_stray_label(self):
        """'득점을 시도했지만'으로 goal이 labels에 섞여 들어와도 개수 요청이 우선한다."""
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "build",
                    "labels": ["goal", "shot", "corner"],
                    "label_counts": [
                        {"label": "shot", "count": 2},
                        {"label": "corner", "count": 2},
                    ],
                },
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        assert _label_histogram(result["plan_result"]["clips"]) == {"shot": 2, "corner": 2}

    def test_reply_reports_the_label_breakdown(self):
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "build",
                    "labels": ["shot", "corner"],
                    "label_counts": [
                        {"label": "shot", "count": 2},
                        {"label": "corner", "count": 2},
                    ],
                },
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        reply = result["plan_result"]["reply_text"]
        assert "슈팅 2개" in reply and "코너킥 2개" in reply

    def test_shortfall_is_reported(self):
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "build",
                    "labels": ["goal"],
                    "label_counts": [{"label": "goal", "count": 5}],
                },
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        assert len(result["plan_result"]["clips"]) == 3
        assert "채우지 못했어요" in result["plan_result"]["reply_text"]

    def test_count_based_build_skips_duration_validation(self):
        intent = {
            "intent_type": "build",
            "labels": ["shot", "corner"],
            "target_duration": 300,
            "label_counts": [{"label": "shot", "count": 2}, {"label": "corner", "count": 2}],
        }
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": ALL_EVENTS, "current_clips": []}
        )
        verdict = validate_plan(
            {"plan_result": result["plan_result"], "intent": intent, "retry_count": 0}
        )
        assert verdict["validation_ok"] is True


class TestExistingBehaviourPreserved:
    def test_one_per_label_still_applies_without_counts(self):
        """'골, 슈팅 하나씩'처럼 개수를 말하지 않으면 기존대로 라벨당 1개."""
        result = plan_or_revise_edit(
            {
                "intent": {"intent_type": "build", "labels": ["goal", "shot"]},
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        assert _label_histogram(result["plan_result"]["clips"]) == {"goal": 1, "shot": 1}

    def test_single_label_count_still_uses_target_clip_count(self):
        result = plan_or_revise_edit(
            {
                "intent": {"intent_type": "build", "labels": ["shot"], "target_clip_count": 3},
                "all_events": ALL_EVENTS,
                "current_clips": [],
            }
        )
        assert _label_histogram(result["plan_result"]["clips"]) == {"shot": 3}

    def test_duration_build_is_unchanged(self):
        intent = {"intent_type": "build", "labels": ["shot"], "target_duration": 44}
        result = plan_or_revise_edit(
            {"intent": intent, "all_events": ALL_EVENTS, "current_clips": []}
        )
        clips = result["plan_result"]["clips"]
        assert sum(clip["duration_sec"] for clip in clips) == pytest.approx(44.0)


class TestAddWithLabelCounts:
    def test_add_honors_per_label_counts(self):
        current = [ALL_EVENTS[0]]
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "add",
                    "labels": ["shot"],
                    "label_counts": [{"label": "shot", "count": 3}],
                },
                "all_events": ALL_EVENTS,
                "current_clips": current,
            }
        )
        clips = result["plan_result"]["clips"]
        assert _label_histogram(clips) == {"goal": 1, "shot": 3}
        assert "슈팅 3개" in result["plan_result"]["reply_text"]

    def test_add_never_duplicates_existing_clips(self):
        current = [clip for clip in ALL_EVENTS if clip["label"] == "shot"][:2]
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "add",
                    "labels": ["shot"],
                    "label_counts": [{"label": "shot", "count": 2}],
                },
                "all_events": ALL_EVENTS,
                "current_clips": current,
            }
        )
        ids = [clip["timeline_event_id"] for clip in result["plan_result"]["clips"]]
        assert len(ids) == len(set(ids)) == 4


class TestLabelMixDescription:
    def test_uses_korean_display_names(self):
        clips = [ALL_EVENTS[0], ALL_EVENTS[3], ALL_EVENTS[4]]
        assert describe_label_mix(clips) == "골 1개, 슈팅 2개"

    def test_empty_clip_list(self):
        assert describe_label_mix([]) == ""
