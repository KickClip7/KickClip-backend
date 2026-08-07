from __future__ import annotations

from pathlib import Path
from typing import Any

from common import canonical_sha256, read_json, sha256_file


class ShotBoundaryContractError(ValueError):
    pass


def validate_reviewed_shot_boundaries(
    *,
    path: Path,
    video_sha256: str,
    frame_count: int,
) -> dict[str, Any]:
    artifact = read_json(path)
    video = artifact.get("video") or {}
    diagnostics = artifact.get("diagnostics") or {}
    review = artifact.get("review_contract") or {}
    if str(video.get("sha256")) != video_sha256:
        raise ShotBoundaryContractError(
            "Reviewed shot boundary video SHA-256 does not match the scene video."
        )
    if int(video.get("frame_count") or -1) != frame_count:
        raise ShotBoundaryContractError(
            "Reviewed shot boundary frame count does not match the scene video."
        )
    pending = list(
        diagnostics.get("pending_cut_frames")
        or review.get("pending_cut_frames")
        or []
    )
    if pending or bool(diagnostics.get("review_required")):
        raise ShotBoundaryContractError(
            "WAITING_SHOT_BOUNDARY_REVIEW: all cut candidates must be resolved."
        )
    if diagnostics.get("retrieval_authorized") is False:
        raise ShotBoundaryContractError(
            "WAITING_SHOT_BOUNDARY_REVIEW: retrieval is not authorized."
        )
    shots = artifact.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ShotBoundaryContractError("Shot boundary artifact has no shots.")
    expected_start = 0
    for index, shot in enumerate(shots):
        if int(shot.get("shot_index", -1)) != index:
            raise ShotBoundaryContractError("Shot indexes are not contiguous.")
        start = int(shot.get("start_frame", -1))
        end = int(shot.get("end_frame_inclusive", -1))
        if start != expected_start or end < start:
            raise ShotBoundaryContractError("Shot frame coverage is not contiguous.")
        if str(shot.get("review_state")) not in {"REVIEWED_PASS", "PASS"}:
            raise ShotBoundaryContractError("Every shot boundary must be reviewed.")
        expected_start = end + 1
    if expected_start != frame_count:
        raise ShotBoundaryContractError(
            "Shot boundaries do not cover every scene frame exactly once."
        )
    return artifact


def candidate_cache_key(
    *,
    video_sha256: str,
    scene_start_sec: float,
    scene_end_sec: float,
    shot_boundary_sha256: str,
    rfdetr_checkpoint_sha256: str,
    policy_sha256: str,
    package_manifest_sha256: str,
    schema_version: str,
) -> str:
    return canonical_sha256(
        {
            "video_sha256": video_sha256,
            "scene_start_sec": scene_start_sec,
            "scene_end_sec": scene_end_sec,
            "shot_boundary_sha256": shot_boundary_sha256,
            "rfdetr_checkpoint_sha256": rfdetr_checkpoint_sha256,
            "policy_sha256": policy_sha256,
            "package_manifest_sha256": package_manifest_sha256,
            "schema_version": schema_version,
        }
    )


def tracking_cache_key(
    *,
    candidate_cache_key_value: str,
    target_selection_revision: int,
    selected_candidate_id: str,
    earlier_anchor_decision: dict[str, Any],
    target_reference_set_path: Path,
    r2_production_manifest_sha256: str,
    cross_shot_mode: str,
) -> str:
    return canonical_sha256(
        {
            "candidate_cache_key": candidate_cache_key_value,
            "target_selection_revision": target_selection_revision,
            "selected_candidate_id": selected_candidate_id,
            "earlier_anchor_decision": earlier_anchor_decision,
            "target_reference_set_sha256": sha256_file(
                target_reference_set_path
            ),
            "r2_production_manifest_sha256": r2_production_manifest_sha256,
            "cross_shot_mode": cross_shot_mode,
        }
    )

