from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import PACKAGE_NAME, SCHEMA_VERSION, STATUS
from .contract import (
    ReviewedShot,
    canonical_sha256,
    global_ranking_fingerprint,
    load_reviewed_shots,
    sha256_file,
    validate_v112a_ranking,
)


# Product shortlist policy only. This does not change the frozen V1.1.2a
# recommendation score or global rank. It only changes which already-eligible
# candidates are surfaced first for user target selection.
PRODUCT_SHORTLIST_POLICY_VERSION = "FIELD_CONTEXT_PRIORITY_R1"
PRODUCT_WIDE_BBOX_AREA_RATIO_MAX = 0.08


class EventCandidateRankingV12ShortlistPatch:
    """Rebuild the product shortlist while preserving V1.1.2a scores/ranks.

    V1.2 is a product surfacing layer, not a scorer. The global V1.1.2a rank
    remains immutable; only shortlist inclusion/display order is made field-first
    so low-resolution wide-shot players are not displaced by large role-unverified
    close-ups.
    """

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.policy_path = self.package_root / "shortlist_policy.json"
        self.input_schema_path = self.package_root / "input_schema.json"
        self.output_schema_path = self.package_root / "output_schema.json"
        self.evidence_schema_path = (
            self.package_root / "ranking_evidence_artifact_schema.json"
        )
        self.manifest_path = self.package_root / "manifest.json"
        self.policy = json.loads(self.policy_path.read_text(encoding="utf-8"))

    @staticmethod
    def _event_shot(
        shots: tuple[ReviewedShot, ...],
        event_local_sec: float,
    ) -> ReviewedShot:
        for index, shot in enumerate(shots):
            is_last = index == len(shots) - 1
            if (
                shot.start_time_sec
                <= event_local_sec
                < shot.end_time_sec
            ) or (is_last and event_local_sec == shot.end_time_sec):
                return shot
        raise ValueError("Event timestamp is outside approved shots.")

    @staticmethod
    def _eligible(
        row: dict[str, Any],
        *,
        policy: dict[str, Any],
    ) -> bool:
        return str(row.get("reliability_state") or "") in set(
            policy["eligible_reliability_states"]
        )

    @staticmethod
    def _nested_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @classmethod
    def _product_context_priority(
        cls,
        row: dict[str, Any],
        *,
        phase: str,
        adjacent_shot_id: str | None,
    ) -> tuple[int, str]:
        """Return a deterministic product-only shortlist priority.

        The frozen V1.1.2a score/rank is never recomputed. The product UI needs
        actual on-field player candidates to remain reviewable even when they are
        LOW_RESOLUTION, while large event/post close-ups that RF-DETR persistently
        labels as ``player`` must not automatically outrank them just because the
        crop is large and sharp.

        Evidence is intentionally conservative:
        - the manually/structurally action-adjacent previous shot is strongest;
        - small/wide candidates or candidates with usable ball evidence are
          field-supported;
        - pre-event candidates are preferred to event/post context-neutral rows;
        - large close-up-only rows remain visible, but are role-unverified and
          therefore sorted last.
        """

        if adjacent_shot_id is not None and row.get("shot_id") == adjacent_shot_id:
            return 0, "PRODUCT_ACTION_ADJACENT_PRIORITY"

        raw_features = cls._nested_dict(row.get("raw_features"))
        visual = cls._nested_dict(raw_features.get("visual"))
        ball = cls._nested_dict(raw_features.get("ball"))

        area_value = visual.get("bbox_area_ratio")
        bbox_area_ratio: float | None = None
        if isinstance(area_value, (int, float)):
            candidate_area = float(area_value)
            if candidate_area >= 0:
                bbox_area_ratio = candidate_area

        selected_ball_observation_count = ball.get(
            "selected_ball_observation_count",
            0,
        )
        if not isinstance(selected_ball_observation_count, (int, float)):
            selected_ball_observation_count = 0
        ball_supported = bool(
            str(ball.get("state") or "").upper() == "AVAILABLE"
            and float(selected_ball_observation_count) > 0
        )
        wide_supported = bool(
            bbox_area_ratio is not None
            and bbox_area_ratio <= PRODUCT_WIDE_BBOX_AREA_RATIO_MAX
        )
        if wide_supported or ball_supported:
            return 1, "PRODUCT_FIELD_SUPPORTED_PRIORITY"

        closeup_only = bool(
            bbox_area_ratio is not None
            and bbox_area_ratio > PRODUCT_WIDE_BBOX_AREA_RATIO_MAX
        )
        if closeup_only:
            return 4, "PRODUCT_CLOSEUP_ROLE_UNVERIFIED"

        if phase == "PRE_EVENT_ACTION":
            return 2, "PRODUCT_PRE_EVENT_CONTEXT_PRIORITY"
        return 3, "PRODUCT_EVENT_POST_CONTEXT_PRIORITY"

    def run(
        self,
        *,
        source_ranking: dict[str, Any],
        reviewed_shots: dict[str, Any],
        source_ranking_artifact_id: str,
        source_ranking_sha256: str,
        shortlist_size: int = 5,
    ) -> dict[str, Any]:
        if shortlist_size != int(self.policy["shortlist_size"]):
            raise ValueError("V1.2 patch supports the frozen shortlist size only.")
        global_rows = validate_v112a_ranking(source_ranking)
        shots = load_reviewed_shots(reviewed_shots)
        event_context = source_ranking.get("event_context") or {}
        event_local_sec = float(
            event_context["event_scene_local_time_sec"]
        )
        event_shot = self._event_shot(shots, event_local_sec)
        previous_shot = (
            shots[event_shot.shot_index - 1]
            if event_shot.shot_index > 0
            else None
        )
        adjacent_active = bool(
            previous_shot is not None
            and event_local_sec - event_shot.start_time_sec
            <= float(self.policy["action_adjacent_start_delta_sec"])
        )
        adjacent_shot_id = (
            previous_shot.shot_id if adjacent_active and previous_shot else None
        )
        eligible = [
            row
            for row in global_rows
            if self._eligible(row, policy=self.policy)
        ]
        phase_by_id = {
            row["candidate_id"]: (
                "PRE_EVENT_ACTION"
                if next(
                    shot.shot_index
                    for shot in shots
                    if shot.shot_id == row["shot_id"]
                )
                < event_shot.shot_index
                else "EVENT_CONTAINING_POST"
            )
            for row in eligible
        }

        product_context_by_id = {
            row["candidate_id"]: self._product_context_priority(
                row,
                phase=phase_by_id[row["candidate_id"]],
                adjacent_shot_id=adjacent_shot_id,
            )
            for row in eligible
        }
        eligible_product_order = sorted(
            eligible,
            key=lambda row: (
                product_context_by_id[row["candidate_id"]][0],
                int(row["rank"]),
                str(row["candidate_id"]),
            ),
        )

        selected: dict[str, dict[str, Any]] = {}
        inclusion_reasons: dict[str, list[str]] = {}
        shot_counts: Counter[str] = Counter()
        phase_counts: Counter[str] = Counter()
        max_per_shot = int(self.policy["max_candidates_per_shot"])
        max_event_post = int(
            self.policy["phase_diversity"][
                "max_event_containing_post"
            ]
        )

        def can_include(row: dict[str, Any]) -> bool:
            if row["candidate_id"] in selected:
                return False
            if shot_counts[row["shot_id"]] >= max_per_shot:
                return False
            phase = phase_by_id[row["candidate_id"]]
            if (
                phase == "EVENT_CONTAINING_POST"
                and phase_counts[phase] >= max_event_post
            ):
                return False
            return len(selected) < shortlist_size

        def include(row: dict[str, Any], reason: str) -> bool:
            if not can_include(row):
                return False
            candidate_id = row["candidate_id"]
            selected[candidate_id] = row
            reasons = inclusion_reasons.setdefault(candidate_id, [])
            reasons.append(reason)
            product_reason = product_context_by_id[candidate_id][1]
            if product_reason not in reasons:
                reasons.append(product_reason)
            shot_counts[row["shot_id"]] += 1
            phase_counts[phase_by_id[candidate_id]] += 1
            return True

        phase_fallbacks: list[dict[str, Any]] = []
        if adjacent_shot_id is not None:
            adjacent = next(
                (
                    row
                    for row in eligible_product_order
                    if row["shot_id"] == adjacent_shot_id
                ),
                None,
            )
            if adjacent is None:
                phase_fallbacks.append(
                    {
                        "slot": "ACTION_ADJACENT_PREVIOUS_SHOT",
                        "reason": "NO_ELIGIBLE_ACTION_ADJACENT_CANDIDATE",
                    }
                )
            else:
                include(adjacent, "CUT_ADJACENT_ACTION_CANDIDATE")

        preferred_pre = int(
            self.policy["phase_diversity"]["preferred_pre_event_action"]
        )
        minimum_pre = int(
            self.policy["phase_diversity"]["min_pre_event_action"]
        )
        # Product-first selection: preserve the frozen phase quotas, but satisfy
        # the pre-event action slots before spending capacity on event/post
        # close-ups. This keeps LOW_RESOLUTION on-field players eligible.
        for row in eligible_product_order:
            if phase_counts["PRE_EVENT_ACTION"] >= preferred_pre:
                break
            if phase_by_id[row["candidate_id"]] == "PRE_EVENT_ACTION":
                include(row, "PRE_EVENT_ACTION_PHASE_SLOT")

        for row in eligible_product_order:
            if phase_counts["EVENT_CONTAINING_POST"] >= max_event_post:
                break
            if phase_by_id[row["candidate_id"]] == "EVENT_CONTAINING_POST":
                include(row, "EVENT_CONTAINING_POST_PHASE_SLOT")
        if phase_counts["PRE_EVENT_ACTION"] < minimum_pre:
            phase_fallbacks.append(
                {
                    "slot": "PRE_EVENT_ACTION_PHASE",
                    "reason": "NO_ELIGIBLE_PRE_EVENT_ACTION_CANDIDATE",
                    "minimum": minimum_pre,
                    "actual": phase_counts["PRE_EVENT_ACTION"],
                }
            )
        elif phase_counts["PRE_EVENT_ACTION"] < preferred_pre:
            phase_fallbacks.append(
                {
                    "slot": "PRE_EVENT_ACTION_PHASE",
                    "reason": "PREFERRED_PRE_EVENT_ACTION_COUNT_UNAVAILABLE",
                    "preferred": preferred_pre,
                    "actual": phase_counts["PRE_EVENT_ACTION"],
                }
            )

        for row in eligible_product_order:
            if len(selected) >= shortlist_size:
                break
            include(row, "GLOBAL_SCORE_FALLBACK")
        if len(selected) < shortlist_size:
            phase_fallbacks.append(
                {
                    "slot": "GLOBAL_SCORE_FALLBACK",
                    "reason": "NO_ADDITIONAL_ELIGIBLE_DIVERSE_CANDIDATE",
                    "requested": shortlist_size,
                    "actual": len(selected),
                }
            )

        selected_rows = sorted(
            selected.values(),
            key=lambda row: (
                product_context_by_id[row["candidate_id"]][0],
                int(row["rank"]),
                str(row["candidate_id"]),
            ),
        )
        shortlist = []
        for shortlist_rank, row in enumerate(selected_rows, start=1):
            copied = copy.deepcopy(row)
            copied["original_global_rank"] = int(row["rank"])
            copied["shortlist_rank"] = shortlist_rank
            copied["shortlist_patch_reason_codes"] = inclusion_reasons[
                row["candidate_id"]
            ]
            copied["shortlist_phase"] = phase_by_id[row["candidate_id"]]
            shortlist.append(copied)

        decisions = []
        for row in global_rows:
            candidate_id = row["candidate_id"]
            if candidate_id in selected:
                decisions.append(
                    {
                        "candidate_id": candidate_id,
                        "original_global_rank": int(row["rank"]),
                        "decision": "INCLUDED",
                        "reason_codes": inclusion_reasons[candidate_id],
                        "phase": phase_by_id[candidate_id],
                    }
                )
                continue
            phase = phase_by_id.get(candidate_id)
            if not self._eligible(row, policy=self.policy):
                reason = "INELIGIBLE_RELIABILITY_STATE"
            elif shot_counts[row["shot_id"]] >= max_per_shot:
                reason = "SHOT_CANDIDATE_CAP"
            elif (
                phase == "EVENT_CONTAINING_POST"
                and phase_counts[phase] >= max_event_post
            ):
                reason = "EVENT_CONTAINING_POST_PHASE_CAP"
            else:
                reason = "SHORTLIST_CAPACITY_REACHED"
            decisions.append(
                {
                    "candidate_id": candidate_id,
                    "original_global_rank": int(row["rank"]),
                    "decision": "EXCLUDED",
                    "reason_codes": [reason],
                    "phase": phase,
                }
            )

        source_fingerprint = global_ranking_fingerprint(global_rows)
        output_global_rows = copy.deepcopy(list(global_rows))
        output_fingerprint = global_ranking_fingerprint(
            tuple(output_global_rows)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "package": PACKAGE_NAME,
            "status": STATUS,
            "ranking_status": STATUS,
            "ranking_weights_changed": False,
            "scores_recomputed": False,
            "human_labels_used_for_shortlist": False,
            "automatic_target_confirmation": False,
            "production_recommendation_ui": "BLOCKED",
            "source_ranking": {
                "artifact_id": source_ranking_artifact_id,
                "sha256": source_ranking_sha256,
                "package": source_ranking["package"],
                "schema_version": source_ranking["schema_version"],
                "global_ranking_fingerprint": source_fingerprint,
            },
            "event_context": copy.deepcopy(event_context),
            "shot_context": {
                "reviewed_shot_count": len(shots),
                "event_shot_id": event_shot.shot_id,
                "event_shot_index": event_shot.shot_index,
                "event_shot_start_time_sec": event_shot.start_time_sec,
                "event_time_from_shot_start_sec": (
                    event_local_sec - event_shot.start_time_sec
                ),
                "action_adjacent_previous_shot_id": adjacent_shot_id,
                "action_adjacent_slot_active": adjacent_active,
            },
            "shortlist_policy": copy.deepcopy(self.policy),
            "shortlist": shortlist,
            "shortlist_decisions": decisions,
            "phase_counts": dict(phase_counts),
            "shot_counts": dict(shot_counts),
            "phase_fallbacks": phase_fallbacks,
            "all_candidates": output_global_rows,
            "global_ranking_invariant": {
                "source_fingerprint": source_fingerprint,
                "output_fingerprint": output_fingerprint,
                "original_global_ranks_unchanged": (
                    source_fingerprint == output_fingerprint
                ),
                "original_scores_unchanged": (
                    source_fingerprint == output_fingerprint
                ),
            },
            "full_gallery_fallback": copy.deepcopy(
                source_ranking["full_gallery_fallback"]
            ),
            "freeze": {
                "source_v1_1_2a_manifest_sha256": (
                    source_ranking["freeze"]["source_manifest_sha256"]
                ),
                "source_ranking_policy_sha256": (
                    source_ranking["freeze"]["ranking_policy_sha256"]
                ),
                "shortlist_policy_sha256": sha256_file(self.policy_path),
                "input_schema_sha256": sha256_file(self.input_schema_path),
                "output_schema_sha256": sha256_file(self.output_schema_path),
                "ranking_evidence_artifact_schema_sha256": sha256_file(
                    self.evidence_schema_path
                ),
                "source_manifest_sha256": sha256_file(self.manifest_path),
                "shortlist_selection_fingerprint": canonical_sha256(
                    [
                        {
                            "candidate_id": row["candidate_id"],
                            "original_global_rank": row[
                                "original_global_rank"
                            ],
                            "shortlist_rank": row["shortlist_rank"],
                        }
                        for row in shortlist
                    ]
                ),
            },
            "diagnostic_evidence_used_for_scoring": False,
            "diagnostic_evidence_used_for_shortlist": False,
        }

