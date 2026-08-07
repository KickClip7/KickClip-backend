"""표기 변형·오타('패널티킥', '슛팅')가 지원 라벨로 해석되는지 검증한다.

회귀 대상: '패널티킥을 추가해줘'를 LLM이 unsupported로 오판해
"'패널티킥' 장면은 현재 분석 모델이 찾아내지 못해서..."로 거절하던 문제.
자모 퍼지 매칭은 사전 추가 없이 처음 보는 오타에도 일반화되어야 한다.
LLM 호출 없이 결정적 계층만 검사한다.
"""

from app.ai.agents.edit_workflow_agent import (
    _guard_unsupported_labels,
    _normalize_intent_labels,
    plan_or_revise_edit,
    resolve_label_counts,
)
from app.ai.agents.label_taxonomy import (
    analyze_scene_terms,
    fuzzy_match_supported_label,
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


ALL_EVENTS = [
    _clip("g1", "goal", 100.0),
    _clip("s1", "shot", 200.0),
    _clip("p1", "penalty", 300.0),
    _clip("c1", "corner", 400.0),
]


class TestFuzzyMatching:
    def test_spelling_variants_resolve(self):
        """'패널티킥'(페널티 오기+킥 접미) 같은 변형이 사전 등록 없이 매핑된다."""
        assert fuzzy_match_supported_label("패널티킥") == "penalty"
        assert fuzzy_match_supported_label("패널티") == "penalty"
        assert fuzzy_match_supported_label("슛팅") == "shot"
        assert fuzzy_match_supported_label("콜너킥") == "corner"

    def test_exact_aliases_still_resolve(self):
        assert fuzzy_match_supported_label("페널티킥") == "penalty"
        assert fuzzy_match_supported_label("PK") == "penalty"

    def test_genuinely_unsupported_terms_never_match(self):
        """프리킥·파울은 비슷한 라벨로 새면 안 된다 — 이게 지난번 수정의 핵심 계약."""
        assert fuzzy_match_supported_label("프리킥") is None
        assert fuzzy_match_supported_label("파울") is None
        assert fuzzy_match_supported_label("골킥") is None
        assert fuzzy_match_supported_label("오버헤드킥") is None

    def test_short_tokens_require_exact_match(self):
        """짧은 별칭('골')은 퍼지 오탐이 쉬워 정확 일치로만 잡는다."""
        assert fuzzy_match_supported_label("골") == "goal"
        assert fuzzy_match_supported_label("고을") is None

    def test_message_scan_recovers_variant_with_particle(self):
        unsupported, labels = analyze_scene_terms("패널티킥을 추가해줘")
        assert unsupported == []
        assert labels == ["penalty"]


class TestGuardRecovery:
    def test_llm_unsupported_misjudgment_is_recovered(self):
        """화면 시나리오: LLM이 '패널티킥'을 unsupported로 오판 → add(penalty)로 복구."""
        payload = _guard_unsupported_labels(
            {
                "intent_type": "unsupported",
                "labels": None,
                "unsupported_terms": ["패널티킥"],
                "needs_clarification": False,
            },
            "패널티킥을 추가해줘",
        )
        assert payload["intent_type"] == "add"
        assert payload["labels"] == ["penalty"]
        assert payload["unsupported_terms"] is None

    def test_action_verb_inference(self):
        for message, expected in [
            ("패널티킥을 추가해줘", "add"),
            ("패널티킥 장면 빼줘", "remove"),
            ("패널티킥으로 하이라이트 만들어줘", "build"),
            ("패널티킥 장면만 보여줘", "filter"),
        ]:
            payload = _guard_unsupported_labels(
                {
                    "intent_type": "unsupported",
                    "labels": None,
                    "unsupported_terms": ["패널티킥"],
                    "needs_clarification": False,
                },
                message,
            )
            assert payload["intent_type"] == expected, message

    def test_llm_action_with_variant_term_gets_labels(self):
        """LLM이 동작은 맞게 분류하고 표현만 미지원 처리한 경우: 라벨만 복구된다."""
        payload = _guard_unsupported_labels(
            {
                "intent_type": "add",
                "labels": None,
                "unsupported_terms": ["패널티킥"],
                "needs_clarification": False,
            },
            "패널티킥을 추가해줘",
        )
        assert payload["intent_type"] == "add"
        assert payload["labels"] == ["penalty"]

    def test_genuine_unsupported_still_blocked(self):
        payload = _guard_unsupported_labels(
            {
                "intent_type": "filter",
                "labels": ["corner"],
                "needs_clarification": False,
            },
            "프리킥 영상만 모아줘",
        )
        assert payload["intent_type"] == "unsupported"
        assert payload["unsupported_terms"] == ["프리킥"]

    def test_mixed_unsupported_and_variant(self):
        """'프리킥이랑 패널티킥 보여줘' → 프리킥은 제외, 패널티킥은 penalty로 처리."""
        payload = _guard_unsupported_labels(
            {
                "intent_type": "unsupported",
                "labels": None,
                "unsupported_terms": ["프리킥", "패널티킥"],
                "needs_clarification": False,
            },
            "프리킥이랑 패널티킥 보여줘",
        )
        assert payload["intent_type"] == "filter"
        assert payload["labels"] == ["penalty"]
        assert payload["unsupported_terms"] == ["프리킥"]


class TestLabelNormalization:
    def test_llm_labels_in_korean_or_typo_are_normalized(self):
        payload = _normalize_intent_labels(
            {"intent_type": "filter", "labels": ["패널티킥", "슈팅"]}
        )
        assert payload["labels"] == ["penalty", "shot"]

    def test_label_counts_accept_variants(self):
        resolved = resolve_label_counts(
            {"label_counts": [{"label": "패널티킥", "count": 1}, {"label": "슛팅", "count": 2}]}
        )
        assert resolved == {"penalty": 1, "shot": 2}


class TestPlannerProtection:
    def test_add_with_only_unsupported_terms_does_not_grab_random_clips(self):
        """LLM이 add + 미지원 표현만 돌려준 경우, 조건 없는 add로 새지 않는다."""
        current = [ALL_EVENTS[0]]
        result = plan_or_revise_edit(
            {
                "intent": {
                    "intent_type": "add",
                    "labels": None,
                    "unsupported_terms": ["오버헤드킥"],
                },
                "all_events": ALL_EVENTS,
                "current_clips": current,
            }
        )
        assert result["plan_result"]["clips"] == current
        assert "오버헤드킥" in result["plan_result"]["reply_text"]

    def test_end_to_end_recovered_add(self):
        """복구된 의도로 계획까지: 기존 구성 유지 + 페널티 1개 추가."""
        payload = _guard_unsupported_labels(
            {
                "intent_type": "unsupported",
                "labels": None,
                "unsupported_terms": ["패널티킥"],
                "needs_clarification": False,
            },
            "패널티킥을 추가해줘",
        )
        current = [ALL_EVENTS[0], ALL_EVENTS[1]]
        result = plan_or_revise_edit(
            {"intent": payload, "all_events": ALL_EVENTS, "current_clips": current}
        )
        labels = [clip["label"] for clip in result["plan_result"]["clips"]]
        assert labels == ["goal", "shot", "penalty"]

    def test_remove_with_variant_label(self):
        result = plan_or_revise_edit(
            {
                "intent": {"intent_type": "remove", "labels": ["패널티킥"]},
                "all_events": ALL_EVENTS,
                "current_clips": ALL_EVENTS,
            }
        )
        labels = [clip["label"] for clip in result["plan_result"]["clips"]]
        assert labels == ["goal", "shot", "corner"]
