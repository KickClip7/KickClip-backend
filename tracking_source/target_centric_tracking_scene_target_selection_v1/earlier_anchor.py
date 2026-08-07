from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from common import atomic_json, now_iso, read_json


def propose_earlier_candidates(
    *,
    output_root: Path,
    selection: dict[str, Any],
    score_candidates: Callable[
        [dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]
    ],
    maximum_candidates: int,
) -> dict[str, Any]:
    candidates = read_json(output_root / "scene_candidates.json")
    selected = next(
        row
        for row in candidates["candidates"]
        if row["candidate_id"] == selection["selected_candidate_id"]
    )
    earlier = [
        row
        for row in candidates["candidates"]
        if int(row["shot_index"]) < int(selected["shot_index"])
    ]
    ranked = score_candidates(selected, earlier) if earlier else []
    proposals = []
    for rank, row in enumerate(ranked[:maximum_candidates], start=1):
        candidate = row["candidate"]
        proposals.append(
            {
                "candidate_id": candidate["candidate_id"],
                "shot_id": candidate["shot_id"],
                "shot_index": candidate["shot_index"],
                "retrieval_rank": rank,
                "retrieval_score": row.get("retrieval_score"),
                "prototype_similarity": row.get("prototype_similarity"),
                "ranking_metrics": row.get("frozen_metrics") or {},
                "representative_observations": [
                    candidate["representative_observation"]
                ],
                "representative_observation": candidate[
                    "representative_observation"
                ],
                "tracking_initialization_observation": candidate[
                    "tracking_initialization_observation"
                ],
                "artifacts": candidate["artifacts"],
                "automatic_confirmation_allowed": False,
                "provenance": row.get("provenance") or {},
            }
        )
    state = (
        "WAITING_EARLIER_ANCHOR_CONFIRMATION"
        if proposals
        else "NO_SAFE_EARLIER_CANDIDATE"
    )
    result = {
        "schema_version": "kickclip.earlier_anchor_proposals.v1",
        "selection_id": selection["selection_id"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "search_scope": {
            "maximum_shot_index_exclusive": selected["shot_index"],
            "candidate_count": len(earlier),
        },
        "discovery_state": (
            "EARLIER_CANDIDATES_FOUND"
            if proposals
            else "NO_SAFE_EARLIER_CANDIDATE"
        ),
        "state": state,
        "proposals": proposals,
        "automatic_confirmation_allowed": False,
        "evaluation_target_ids_used": False,
    }
    atomic_json(output_root / "earlier_candidate_proposals.json", result)
    return result


def confirm_earlier_anchor(
    *,
    output_root: Path,
    selection: dict[str, Any],
    candidate_id: str | None,
    reviewer: str,
    reject_all: bool,
) -> dict[str, Any]:
    proposals = read_json(output_root / "earlier_candidate_proposals.json")
    allowed = {
        str(row["candidate_id"]): row for row in proposals.get("proposals", [])
    }
    if reject_all:
        state = "USER_REJECTED_ALL_EARLIER_CANDIDATES"
        decision = {
            "schema_version": "kickclip.earlier_anchor_decision.v1",
            "selection_id": selection["selection_id"],
            "state": state,
            "confirmed_candidate_id": None,
            "anchor_shot_id": selection["selected_shot_id"],
            "anchor_frame_index": None,
            "anchor_bbox_xyxy": None,
            "start_mode": "START_FROM_SELECTED_SHOT",
            "confirmed_by": reviewer,
            "confirmed_at": now_iso(),
        }
    else:
        if not candidate_id or candidate_id not in allowed:
            raise ValueError(
                "Earlier anchor candidate must be in the current proposal list."
            )
        proposal = allowed[candidate_id]
        initialization = proposal["tracking_initialization_observation"]
        if initialization["validation_state"] != "VALID":
            raise ValueError("Earlier candidate has no valid tracking anchor.")
        decision = {
            "schema_version": "kickclip.earlier_anchor_decision.v1",
            "selection_id": selection["selection_id"],
            "state": "USER_CONFIRMED_EARLIER_ANCHOR",
            "confirmed_candidate_id": candidate_id,
            "anchor_shot_id": proposal["shot_id"],
            "anchor_frame_index": initialization["frame_index"],
            "anchor_bbox_xyxy": initialization["bbox_xyxy"],
            "start_mode": "USER_CONFIRMED_EARLIER_ANCHOR",
            "confirmed_by": reviewer,
            "confirmed_at": now_iso(),
        }
    atomic_json(output_root / "earlier_anchor_decision.json", decision)
    selection = dict(selection)
    selection["earlier_anchor_resolution"] = decision
    atomic_json(output_root / "target_selection.json", selection)
    return decision


def selected_shot_fallback(
    *,
    output_root: Path,
    selection: dict[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    candidates = read_json(output_root / "scene_candidates.json")
    selected = next(
        row
        for row in candidates["candidates"]
        if row["candidate_id"] == selection["selected_candidate_id"]
    )
    initialization = selected["tracking_initialization_observation"]
    if initialization["validation_state"] != "VALID":
        raise ValueError("Selected candidate has no valid tracking anchor.")
    decision = {
        "schema_version": "kickclip.earlier_anchor_decision.v1",
        "selection_id": selection["selection_id"],
        "state": "START_FROM_SELECTED_SHOT",
        "confirmed_candidate_id": selected["candidate_id"],
        "anchor_shot_id": selected["shot_id"],
        "anchor_frame_index": initialization["frame_index"],
        "anchor_bbox_xyxy": initialization["bbox_xyxy"],
        "start_mode": "START_FROM_SELECTED_SHOT",
        "confirmed_by": reviewer,
        "confirmed_at": now_iso(),
    }
    atomic_json(output_root / "earlier_anchor_decision.json", decision)
    return decision
