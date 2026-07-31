from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from app.domains.highlight.event_candidate_ranking_v1_2.service import (
    EventCandidateRankingV12ShortlistPatch,
)
from app.domains.highlight.event_candidate_ranking_v1_2.verifier import (
    EventCandidateRankingV12Verifier,
)
from scripts.generate_v1_2_full_class_diagnostic import (
    iou,
    top5_class_audit,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_2"
)
V112A_MANIFEST = (
    ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_1_2a/manifest.json"
)
V112A_MANIFEST_SHA256 = (
    "d1d9bc507f268b7aecf319fe25c5f967bd2d58179939459f752b26b8044ddafd"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(
    rank: int,
    *,
    shot: int,
    score: float,
    reliability: str = "PARTIAL_FEATURES",
) -> dict:
    return {
        "candidate_id": (
            "scene_candidate_regression_"
            f"shot_{shot:04d}_track_{rank:04d}"
        ),
        "rank": rank,
        "recommendation_score": score,
        "shot_id": f"shot_{shot:04d}",
        "reliability_state": reliability,
    }


def source_ranking(rows: list[dict]) -> dict:
    return {
        "package": (
            "target_centric_tracking_event_candidate_ranking_v1_1_2a"
        ),
        "schema_version": "kickclip.event_candidate_ranking.v1_1_2a",
        "automatic_target_confirmation": False,
        "event_context": {
            "event_id": "event_goal",
            "event_scene_local_time_sec": 15.0,
        },
        "all_candidates": rows,
        "full_gallery_fallback": [
            row["candidate_id"] for row in rows
        ],
        "freeze": {
            "source_manifest_sha256": V112A_MANIFEST_SHA256,
            "ranking_policy_sha256": (
                "7359adf60fba7128080bcd59895074f883a5c18f456ad16"
                "f1510d92b8af21e44"
            ),
        },
    }


def reviewed_shots() -> dict:
    boundaries = [0, 43, 102, 303, 348, 399, 468]
    times = [0.0, 1.72, 4.08, 12.12, 13.92, 15.96, 18.72]
    return {
        "video": {"frame_count": 468},
        "shots": [
            {
                "shot_id": f"shot_{index:04d}",
                "shot_index": index,
                "start_frame": start,
                "end_frame_inclusive": end - 1,
                "start_time_sec": times[index],
                "end_time_sec": times[index + 1],
                "review_state": "REVIEWED_PASS",
            }
            for index, (start, end) in enumerate(
                zip(boundaries, boundaries[1:])
            )
        ],
    }


def representative_rows() -> list[dict]:
    layout = [
        (4, 0.8643768775669207),
        (4, 0.8515696805526234),
        (4, 0.8223776065541373),
        (4, 0.8181435362052659),
        (4, 0.8154003832203746),
        (3, 0.8114574153107094),
        (4, 0.808635),
        (4, 0.770582),
        (4, 0.740000),
        (3, 0.7134070587419118),
        (4, 0.700000),
        (4, 0.690000),
        (4, 0.680000),
        (4, 0.675000),
        (4, 0.670000),
        (4, 0.668000),
        (3, 0.665010),
        (4, 0.650000),
        (4, 0.640000),
        (4, 0.630000),
        (4, 0.620000),
        (4, 0.610000),
        (2, 0.593496705798441),
    ]
    return [
        candidate(rank, shot=shot, score=score)
        for rank, (shot, score) in enumerate(layout, start=1)
    ]


def run_patch(rows: list[dict]) -> dict:
    return EventCandidateRankingV12ShortlistPatch(PACKAGE_ROOT).run(
        source_ranking=source_ranking(rows),
        reviewed_shots=reviewed_shots(),
        source_ranking_artifact_id="artifact_v112a",
        source_ranking_sha256="a" * 64,
        shortlist_size=5,
    )


def test_representative_counterfactual_shortlist_only() -> None:
    rows = representative_rows()
    before = [
        (
            row["candidate_id"],
            row["rank"],
            row["recommendation_score"],
        )
        for row in rows
    ]
    output = run_patch(rows)
    selected_global_ranks = [
        row["original_global_rank"] for row in output["shortlist"]
    ]
    assert selected_global_ranks == [1, 2, 6, 10, 23]
    assert output["shot_counts"] == {
        "shot_0003": 2,
        "shot_0004": 2,
        "shot_0002": 1,
    }
    assert output["phase_counts"] == {
        "PRE_EVENT_ACTION": 3,
        "EVENT_CONTAINING_POST": 2,
    }
    assert output["shortlist"][2]["shortlist_patch_reason_codes"] == [
        "CUT_ADJACENT_ACTION_CANDIDATE"
    ]
    assert any(
        decision["reason_codes"] == ["SHOT_CANDIDATE_CAP"]
        for decision in output["shortlist_decisions"]
    )
    after = [
        (
            row["candidate_id"],
            row["rank"],
            row["recommendation_score"],
        )
        for row in output["all_candidates"]
    ]
    assert before == after
    assert output["ranking_weights_changed"] is False
    assert output["scores_recomputed"] is False
    assert output["human_labels_used_for_shortlist"] is False
    assert output["automatic_target_confirmation"] is False

    # Human ground truth enters only after shortlist generation.
    primary_actor_ids = {
        rows[5]["candidate_id"],
        rows[9]["candidate_id"],
        rows[16]["candidate_id"],
    }
    shortlisted = {
        row["candidate_id"] for row in output["shortlist"]
    }
    assert primary_actor_ids & shortlisted


def test_shot_cap_is_two_and_one_shot_cannot_own_top_five() -> None:
    output = run_patch(representative_rows())
    assert max(output["shot_counts"].values()) == 2
    assert len(output["shot_counts"]) >= 2


def test_action_adjacent_slot_requires_event_shot_start_within_two_seconds() -> None:
    output = run_patch(representative_rows())
    assert output["shot_context"]["event_shot_id"] == "shot_0004"
    assert output["shot_context"]["event_time_from_shot_start_sec"] == 1.08
    assert output["shot_context"]["action_adjacent_previous_shot_id"] == (
        "shot_0003"
    )
    assert output["shot_context"]["action_adjacent_slot_active"] is True


def test_phase_shortage_does_not_force_unreliable_candidate() -> None:
    rows = [
        candidate(rank, shot=4, score=1.0 - rank / 100)
        for rank in range(1, 7)
    ]
    rows.extend(
        [
            candidate(
                rank,
                shot=3,
                score=1.0 - rank / 100,
                reliability="NO_RELIABLE_SHORTLIST",
            )
            for rank in range(7, 10)
        ]
    )
    output = run_patch(rows)
    assert all(
        row["reliability_state"] != "NO_RELIABLE_SHORTLIST"
        for row in output["shortlist"]
    )
    assert any(
        row["reason"] == "NO_ELIGIBLE_ACTION_ADJACENT_CANDIDATE"
        for row in output["phase_fallbacks"]
    )
    assert any(
        row["reason"] == "NO_ELIGIBLE_PRE_EVENT_ACTION_CANDIDATE"
        for row in output["phase_fallbacks"]
    )


def test_output_schema_and_verifier() -> None:
    output = run_patch(representative_rows())
    Draft202012Validator(
        json.loads(
            (PACKAGE_ROOT / "output_schema.json").read_text(encoding="utf-8")
        )
    ).validate(output)
    result = EventCandidateRankingV12Verifier(PACKAGE_ROOT).check()
    assert result.verified is True
    assert result.ranking_weights_changed is False
    assert result.automatic_target_confirmation is False


def test_frozen_v112a_manifest_is_unchanged() -> None:
    assert sha256(V112A_MANIFEST) == V112A_MANIFEST_SHA256


def test_full_class_diagnostic_audits_boxes_but_is_not_ranking_input() -> None:
    candidate_id = "scene_candidate_shot_0004_track_0001"
    scene_candidates = {
        "candidates": [
            {
                "candidate_id": candidate_id,
                "observations": [
                    {
                        "frame_index": 303,
                        "bbox_xyxy": [10.0, 10.0, 30.0, 50.0],
                    }
                ],
            }
        ]
    }
    ranking = {
        "shortlist": [
            {
                "candidate_id": candidate_id,
                "rank": 1,
            }
        ]
    }
    detections = [
        {
            "frame_index": 303,
            "detection_id": "det_staff",
            "class_id": 3,
            "class_name": "staff",
            "confidence": 0.9,
            "bbox_xyxy": [10.0, 10.0, 30.0, 50.0],
        }
    ]
    assert iou(
        [10.0, 10.0, 30.0, 50.0],
        [10.0, 10.0, 30.0, 50.0],
    ) == 1.0
    audit = top5_class_audit(
        scene_candidates=scene_candidates,
        ranking=ranking,
        detections=detections,
        frame_start=303,
        frame_end=398,
        minimum_iou=0.5,
    )
    assert audit[0]["majority_matched_class"] == "staff"
    assert audit[0]["staff_classified"] is True


def test_ranking_evidence_contract_keeps_full_class_data_disconnected() -> None:
    evidence = {
        "schema_version": "kickclip.ranking_evidence.v1",
        "player_discovery_detections": {
            "artifact_id": "artifact_player_discovery",
            "sha256": "a" * 64,
            "classes": ["player", "goalkeeper"],
            "immutable": True,
        },
        "ranking_object_detections": {
            "artifact_id": "artifact_full_class_diagnostic",
            "sha256": "b" * 64,
            "classes": [
                "player",
                "goalkeeper",
                "referee",
                "staff",
                "ball",
            ],
            "frame_range": {
                "start_frame": 303,
                "end_frame_inclusive": 398,
            },
            "diagnostic_only": True,
        },
        "connected_to_scoring": False,
    }
    Draft202012Validator(
        json.loads(
            (PACKAGE_ROOT / "ranking_evidence_artifact_schema.json").read_text(
                encoding="utf-8"
            )
        )
    ).validate(evidence)
