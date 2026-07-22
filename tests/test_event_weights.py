import unittest

from app.domains.auth.event_weights import (
    calculate_importance_score,
    default_event_weights,
    normalize_event_weights,
)
from app.ai.tasks.highlight_spotting.label_map import get_display_info, normalize_label


class EventWeightsTest(unittest.TestCase):
    def test_default_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(default_event_weights().values()), 1.0)

    def test_weights_are_normalized_and_aliases_are_merged(self) -> None:
        result = normalize_event_weights(
            {
                "Goal": 3,
                "corner": 1,
                "corner_kick": 1,
            }
        )

        self.assertEqual(result, {"goal": 0.6, "corner": 0.4})

    def test_importance_score_uses_event_weight_and_confidence(self) -> None:
        score = calculate_importance_score(
            event_label="goal",
            confidence_score=0.9,
            event_weights=default_event_weights(),
        )

        self.assertEqual(score, 28.5)

    def test_context_bonus_is_added_before_scaling(self) -> None:
        score = calculate_importance_score(
            event_label="goal",
            confidence_score=0.9,
            event_weights=default_event_weights(),
            context_bonus=0.02,
        )

        self.assertEqual(score, 30.5)

    def test_same_event_scores_differ_by_user_preferences(self) -> None:
        goal_focused = normalize_event_weights({"goal": 0.8, "card": 0.2})
        card_focused = normalize_event_weights({"goal": 0.2, "card": 0.8})

        goal_focused_score = calculate_importance_score(
            event_label="card",
            confidence_score=0.8,
            event_weights=goal_focused,
        )
        card_focused_score = calculate_importance_score(
            event_label="card",
            confidence_score=0.8,
            event_weights=card_focused,
        )

        self.assertEqual(goal_focused_score, 18.0)
        self.assertEqual(card_focused_score, 72.0)

    def test_all_zero_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_event_weights({"goal": 0, "shot": 0})

    def test_future_model_label_is_preserved(self) -> None:
        self.assertEqual(normalize_label("Own Goal"), "own_goal")
        self.assertEqual(get_display_info("Own Goal")["tag"], "OWN GOAL")

    def test_missing_user_weight_uses_model_default(self) -> None:
        score = calculate_importance_score(
            event_label="shot",
            confidence_score=1.0,
            event_weights={"goal": 1.0},
        )

        self.assertEqual(score, 18.0)


if __name__ == "__main__":
    unittest.main()
