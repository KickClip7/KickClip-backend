from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NOT_RUN = "NOT_RUN"
ACTOR_NOT_IN_CANDIDATE_SET = "ACTOR_NOT_IN_CANDIDATE_SET"


@dataclass(frozen=True)
class ApprovedAnnotation:
    annotation_status: str
    approval_mode: str
    primary_actor_visible: bool
    primary_actor_in_candidate_set: bool
    primary_actor_candidate_ids: tuple[str, ...]
    directly_related_candidate_ids: tuple[str, ...]
    actor_missing_reason: str | None
    reviewer: str
    reviewed_at: str

    def validate(self, candidate_ids: set[str]) -> None:
        if self.annotation_status != "COMPLETE":
            return
        if self.approval_mode not in {"FINAL_APPROVED", "EXPLICIT_CONSENSUS"}:
            raise ValueError("Complete annotation requires final approval or consensus.")
        if self.primary_actor_in_candidate_set:
            if not self.primary_actor_candidate_ids:
                raise ValueError("Primary actor candidate IDs are required.")
            if not set(self.primary_actor_candidate_ids).issubset(candidate_ids):
                raise ValueError("Primary actor annotation references unknown candidates.")
        elif self.primary_actor_visible:
            if self.actor_missing_reason != ACTOR_NOT_IN_CANDIDATE_SET:
                raise ValueError("Visible missing actor requires ACTOR_NOT_IN_CANDIDATE_SET.")
        if not set(self.directly_related_candidate_ids).issubset(candidate_ids):
            raise ValueError("Related annotation references unknown candidates.")


def finalize_reviews(
    reviews: list[dict[str, Any]],
    *,
    approval_mode: str,
    approved_reviewer: str | None = None,
) -> ApprovedAnnotation:
    if not reviews:
        raise ValueError("At least one annotation review is required.")
    if approval_mode == "FINAL_APPROVED":
        selected = [
            review
            for review in reviews
            if review.get("reviewer") == approved_reviewer
        ]
        if len(selected) != 1:
            raise ValueError("Final approval must select exactly one reviewer.")
        source = selected[0]
    elif approval_mode == "EXPLICIT_CONSENSUS":
        if len(reviews) < 2:
            raise ValueError("Explicit consensus requires multiple reviewers.")
        ground_truth_fields = (
            "primary_actor_visible",
            "primary_actor_in_candidate_set",
            "primary_actor_candidate_ids",
            "directly_related_candidate_ids",
            "actor_missing_reason",
        )
        signatures = {
            tuple(
                tuple(review.get(field) or ())
                if field.endswith("_ids")
                else review.get(field)
                for field in ground_truth_fields
            )
            for review in reviews
        }
        if len(signatures) != 1:
            raise ValueError(
                "Reviewer labels disagree; union is forbidden and consensus is absent."
            )
        source = reviews[0]
    else:
        raise ValueError("Unsupported annotation approval mode.")
    annotation = ApprovedAnnotation(
        annotation_status="COMPLETE",
        approval_mode=approval_mode,
        primary_actor_visible=bool(source["primary_actor_visible"]),
        primary_actor_in_candidate_set=bool(
            source["primary_actor_in_candidate_set"]
        ),
        primary_actor_candidate_ids=tuple(
            source.get("primary_actor_candidate_ids") or ()
        ),
        directly_related_candidate_ids=tuple(
            source.get("directly_related_candidate_ids") or ()
        ),
        actor_missing_reason=source.get("actor_missing_reason"),
        reviewer=(
            approved_reviewer
            if approval_mode == "FINAL_APPROVED"
            else "CONSENSUS:" + ",".join(
                sorted(str(review["reviewer"]) for review in reviews)
            )
        ),
        reviewed_at=str(
            max(str(review["reviewed_at"]) for review in reviews)
        ),
    )
    return annotation


def evaluate_event(
    *,
    ranked_candidate_ids: list[str],
    shortlist_candidate_ids: list[str],
    annotation: ApprovedAnnotation,
    candidate_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    candidate_ids = set(ranked_candidate_ids)
    annotation.validate(candidate_ids)
    if annotation.annotation_status != "COMPLETE":
        return {
            "scope": "SINGLE_EVENT",
            "metric_status": NOT_RUN,
            "reason": "ANNOTATION_INCOMPLETE",
        }
    coverage = (
        1.0
        if annotation.primary_actor_visible
        and annotation.primary_actor_in_candidate_set
        else 0.0
        if annotation.primary_actor_visible
        else None
    )
    result: dict[str, Any] = {
        "scope": "SINGLE_EVENT",
        "metric_status": "COMPLETE",
        "candidate_generation_actor_coverage": coverage,
        "no_reliable_shortlist": float(not shortlist_candidate_ids),
    }
    if not annotation.primary_actor_in_candidate_set:
        result.update(
            {
                "primary_actor_recall_at_1": NOT_RUN,
                "primary_actor_recall_at_3": NOT_RUN,
                "primary_actor_recall_at_5": NOT_RUN,
                "mrr": NOT_RUN,
            }
        )
        return result
    actors = set(annotation.primary_actor_candidate_ids)
    actor_ranks = [
        index
        for index, candidate_id in enumerate(ranked_candidate_ids, start=1)
        if candidate_id in actors
    ]
    best_rank = min(actor_ranks)
    result.update(
        {
            "primary_actor_recall_at_1": float(best_rank <= 1),
            "primary_actor_recall_at_3": float(best_rank <= 3),
            "primary_actor_recall_at_5": float(best_rank <= 5),
            "mrr": 1.0 / best_rank,
        }
    )
    if candidate_roles and ranked_candidate_ids:
        top_role = candidate_roles.get(ranked_candidate_ids[0])
        result["broadcast_closeup_non_actor_top_1"] = float(
            top_role == "BROADCAST_CLOSEUP_NON_ACTOR"
        )
        result["unrelated_player_top_1"] = float(
            top_role == "UNRELATED_PLAYER"
        )
    else:
        result["broadcast_closeup_non_actor_top_1"] = NOT_RUN
        result["unrelated_player_top_1"] = NOT_RUN
    return result


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [
        event
        for event in events
        if event.get("metric_status") == "COMPLETE"
    ]
    metric_names = (
        "candidate_generation_actor_coverage",
        "primary_actor_recall_at_1",
        "primary_actor_recall_at_3",
        "primary_actor_recall_at_5",
        "mrr",
        "broadcast_closeup_non_actor_top_1",
        "unrelated_player_top_1",
        "no_reliable_shortlist",
    )
    aggregates: dict[str, Any] = {}
    for name in metric_names:
        values = [
            float(event[name])
            for event in complete
            if isinstance(event.get(name), (int, float))
        ]
        aggregates[name] = sum(values) / len(values) if values else NOT_RUN
    return {
        "scope": "DATASET_AGGREGATE",
        "event_count": len(events),
        "evaluated_event_count": len(complete),
        "metrics": aggregates,
    }
