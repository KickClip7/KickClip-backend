from __future__ import annotations

from pathlib import Path
from typing import Any

from common import atomic_json, canonical_sha256, read_json, sha256_file
from contracts import tracking_cache_key


R2_MANIFEST_SHA256 = (
    "751338f51c4f7c08bb24576e5afea7d82b9cbbdefbb651ff1a3c1d5c45ffcc86"
)


def create_tracking_launch_manifest(
    *,
    output_root: Path,
    scene_video: Path,
    shot_boundaries_path: Path,
    selection_path: Path,
    reference_set_path: Path,
    earlier_decision_path: Path,
    candidate_cache_key_value: str,
    r3_manifest_sha256: str | None,
    cross_shot_mode: str = "assisted",
) -> dict[str, Any]:
    selection = read_json(selection_path)
    references = read_json(reference_set_path)
    decision = read_json(earlier_decision_path)
    candidates = read_json(output_root / "scene_candidates.json")
    if sha256_file(shot_boundaries_path) != str(
        candidates["shot_boundary"]["sha256"]
    ):
        raise ValueError(
            "Reviewed shot boundaries changed after candidate discovery."
        )
    candidate_by_id = {
        row["candidate_id"]: row for row in candidates["candidates"]
    }
    if decision["state"] == "USER_CONFIRMED_EARLIER_ANCHOR":
        anchor_candidate_id = decision["confirmed_candidate_id"]
    else:
        anchor_candidate_id = selection["selected_candidate_id"]
    if decision.get("selection_id") != selection.get("selection_id"):
        raise ValueError("Earlier anchor decision belongs to another selection.")
    if references.get("selection_id") != selection.get("selection_id"):
        raise ValueError("Target reference set belongs to another selection.")
    if anchor_candidate_id not in candidate_by_id:
        raise ValueError("Tracking anchor does not belong to this scene.")
    anchor_candidate = candidate_by_id[anchor_candidate_id]
    initialization = anchor_candidate["tracking_initialization_observation"]
    if initialization["validation_state"] != "VALID":
        raise ValueError("Tracking anchor initialization is invalid.")
    key = tracking_cache_key(
        candidate_cache_key_value=candidate_cache_key_value,
        target_selection_revision=int(
            selection["target_selection_revision"]
        ),
        selected_candidate_id=str(selection["selected_candidate_id"]),
        earlier_anchor_decision=decision,
        target_reference_set_path=reference_set_path,
        r2_production_manifest_sha256=R2_MANIFEST_SHA256,
        cross_shot_mode=cross_shot_mode,
    )
    result = {
        "schema_version": "kickclip.scene_tracking_launch_manifest.v1",
        "selection_id": selection["selection_id"],
        "target_selection_revision": selection[
            "target_selection_revision"
        ],
        "selected_candidate_id": selection["selected_candidate_id"],
        "anchor_candidate_id": anchor_candidate_id,
        "anchor_shot_id": anchor_candidate["shot_id"],
        "anchor_frame_index": initialization["frame_index"],
        "anchor_bbox_xyxy": initialization["bbox_xyxy"],
        "anchor_source": decision["state"],
        "cross_shot_mode": cross_shot_mode,
        "tracking_cache_key": key,
        "inputs": {
            "video_sha256": sha256_file(scene_video),
            "shot_boundaries_sha256": sha256_file(shot_boundaries_path),
            "target_selection_sha256": sha256_file(selection_path),
            "target_reference_set_sha256": sha256_file(reference_set_path),
            "earlier_anchor_decision_sha256": sha256_file(
                earlier_decision_path
            ),
        },
        "runtime": {
            "wrapper": "target_centric_tracking_v2_production_r3",
            "r3_manifest_sha256": r3_manifest_sha256,
            "r2_production_manifest_sha256": R2_MANIFEST_SHA256,
            "r2_algorithm_copied_or_modified": False,
        },
        "pre_anchor_contract": {
            "bbox_count": 0,
            "tracking_scope": "BEFORE_SELECTED_ANCHOR_NOT_EVALUATED",
            "absence_claimed": False,
        },
        "reference_set": {
            "reference_count": len(references.get("references", [])),
            "passed_as_provenance": True,
            "frozen_v2_memory_modified": False,
        },
        "identity_auto_confirmed": False,
        "manifest_id": "tracking_launch_" + canonical_sha256(
            {
                "selection": selection["selection_id"],
                "cache_key": key,
            }
        )[:20],
    }
    atomic_json(output_root / "tracking_launch_manifest.json", result)
    return result
