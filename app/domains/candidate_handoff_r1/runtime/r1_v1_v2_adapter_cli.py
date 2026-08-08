#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "kickclip.r1_v1_v2_adapter_state.v1"
PIPELINE_VERSION = "EVENT_CANDIDATE_HANDOFF_R1_REAL_V1_V2_ADAPTER"
SELECTION_VIEW_SCHEMA = "kickclip.r1_selection_bidirectional_view.v2"
BIDIRECTIONAL_INPUT_POLICY = "SELECTED_SHOT_ANCHOR_BIDIRECTIONAL_INPUTS_R1"
BIDIRECTIONAL_PHASE1_POLICY = "FROZEN_STAGE1_STAGE2_BOTH_DIRECTIONS_R1"
PHASE3B_REPORT_SCHEMA = "kickclip.phase3b_bidirectional_phase1_report.v1"
PHASE3C_MERGE_POLICY = "SOURCE_FRAME_REMAP_FORWARD_ANCHOR_PRIMARY_R1"
PHASE3C_TIMELINE_SCHEMA = "kickclip.phase3c_selected_shot_timeline.v1"
PHASE3C_REPORT_SCHEMA = "kickclip.phase3c_bidirectional_timeline_merge_report.v1"
REVIEWED_SHOT_CONTRACT_SCHEMA = "kickclip.reviewed_shot_contract.v1"
PHASE4A_MEMORY_SCHEMA = "kickclip.initial_target_memory_revision.r1_1"
PHASE4A_REPORT_SCHEMA = "kickclip.phase4a_initial_target_memory_report.v1"
PHASE4A_POLICY = "IMMUTABLE_SELECTION_REFERENCES_PLUS_REAL_PHASE3C_ACTIVE_R1"
PHASE4B_REPORT_SCHEMA = "kickclip.phase4b_multi_shot_search_report.v1"
PHASE4B_ATTEMPT_SCHEMA = "kickclip.phase4b_shot_search_attempt.v1"
PHASE4B_POLICY = "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R12_GROUP_CONFIDENCE_REVIEW_CATALOG"
PHASE4B_CANDIDATE_ROLE_FILTER_POLICY = "PLAYER_GOALKEEPER_ONLY_R2_WITH_ROLE_NEGATIVES"
PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY = "SAME_SHOT_STAFF_REFEREE_NEGATIVE_GALLERY_R1"
PHASE4B_CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "staff",
    4: "ball",
}
PHASE4B_ALLOWED_CANDIDATE_CLASS_IDS = frozenset({0, 1})
PHASE4B_ALLOWED_CANDIDATE_CLASS_NAMES = frozenset({"player", "goalkeeper"})
PHASE4B_NEGATIVE_ROLE_CLASS_IDS = frozenset({2, 3})
PHASE4B_NEGATIVE_ROLE_CLASS_NAMES = frozenset({"referee", "staff"})
PHASE4B_ROLE_CONFUSION_TEMPORAL_WINDOW = 2
PHASE4B_ROLE_CONFUSION_MIN_IOU = 0.50
PHASE4B_ROLE_CONFUSION_MIN_OBSERVATIONS = 2
PHASE4B_ROLE_CONFUSION_MIN_RATIO = 0.10
PHASE4B_NEGATIVE_ROLE_MAX_CROPS_PER_CLASS = 12
PHASE4B_NEGATIVE_ROLE_MIN_GALLERY_SIZE = 3
PHASE4B_REVIEW_MIN_CROP_MARGIN_MEDIAN = 0.05
PHASE4B_REVIEW_MIN_POSITIVE_MARGIN_SUPPORT = 0.67
PHASE4B_REVIEW_MIN_PROTOTYPE_NEGATIVE_MARGIN = 0.03
# Detector-labeled staff/referee evidence is noisier than user-confirmed
# negatives, so it must not hard-reject on a single weak signal.  P1 only
# vetoes when at least two independent identity comparisons say the candidate
# is closer to detector-role negatives than to the selected target.
PHASE4B_DETECTOR_ROLE_HARD_MAX_CROP_MARGIN_MEDIAN = 0.0
PHASE4B_DETECTOR_ROLE_HARD_MAX_POSITIVE_MARGIN_SUPPORT = 0.50
PHASE4B_DETECTOR_ROLE_HARD_MAX_PROTOTYPE_MARGIN = 0.0
PHASE4B_DETECTOR_ROLE_HARD_MIN_FAILED_SIGNALS = 2
PHASE4B_DUPLICATE_MIN_COMMON_FRAMES = 3
PHASE4B_DUPLICATE_MIN_MEAN_IOU = 0.45
PHASE4B_DUPLICATE_MIN_TEMPORAL_OVERLAP = 0.40
PHASE4B_APPROVED_CONTINUE = "PHASE4B_APPROVED_CONTINUE"
PHASE4B_SEARCH_EXHAUSTED_STATUS = "SEARCH_EXHAUSTED_NO_REVIEWABLE_CANDIDATE"
PHASE4B_CONTINUE_DECISION = "CONTINUE_PHASE4B_NEXT_REVIEWED_SHOT"
PHASE4B_ALL_EXHAUSTED_DECISION = "SAFE_BLOCK_PHASE4B_ALL_REMAINING_SHOTS_EXHAUSTED"
PHASE4B_NONE_OF_THESE_SHOT_STATUS = "SEARCH_EXHAUSTED_NONE_OF_THESE"
PHASE4B_IDENTITY_NEGATIVE_MEMORY_SCHEMA = "kickclip.phase4b_identity_negative_memory.v1"
PHASE4B_IDENTITY_NEGATIVE_MEMORY_POLICY = "USER_REJECTED_CROSS_SHOT_CANDIDATES_R1"
PHASE4B_COMBINED_NEGATIVE_POLICY = "PROVENANCE_ROLE_USER_HARD_DETECTOR_STRONG_VETO_IDENTITY_ROBUST_R5"
PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY = "SCALE_COMPATIBLE_CLUSTER_ROBUST_IDENTITY_NEGATIVE_R1"
PHASE4B_ROLE_NEGATIVE_SCORING_POLICY = "USER_CONFIRMED_ROLE_HARD_DETECTOR_ROLE_STRONG_VETO_R3"
PHASE4B_RESCUE_REVIEW_POLICY = "GROUP_CONFIDENCE_ASSISTED_REVIEW_R5"
PHASE4B_PREVIOUS_POLICY_R7 = "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R7_IDENTITY_OBSERVABILITY"
PHASE4B_PREVIOUS_POLICY_R8 = "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R8_BANKED_NEGATIVE_RESCUE"
PHASE4B_PREVIOUS_POLICY_R9 = "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R9_PROVENANCE_ROLE_TRAJECTORY_REVIEW"
PHASE4B_PREVIOUS_POLICY_R10 = "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R10_IDENTITY_PURITY_SEGMENTS"
PHASE4B_PREVIOUS_POLICY_R11 = "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_R11_TARGET_CONTEXT_CORROBORATION"
PHASE4B_RESCORABLE_PREVIOUS_POLICIES = frozenset({
    PHASE4B_PREVIOUS_POLICY_R7,
    PHASE4B_PREVIOUS_POLICY_R8,
    PHASE4B_PREVIOUS_POLICY_R9,
    PHASE4B_PREVIOUS_POLICY_R10,
    PHASE4B_PREVIOUS_POLICY_R11,
})
PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY = (
    "USER_CONFIRMED_NON_PLAYER_SCALE_COMPATIBLE_CLUSTER_ROBUST_HARD_GATE_R1"
)
PHASE4B_DETECTOR_ROLE_SCORING_POLICY = (
    "DETECTOR_LABELED_STAFF_REFEREE_STRONG_MULTI_EVIDENCE_VETO_R2"
)
PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY = (
    "CORROBORATION_GROUP_CONFIDENCE_REVIEW_CATALOG_R4"
)
PHASE4B_REVIEW_CATALOG_POLICY = "ALL_SAFE_PLAUSIBLE_EVIDENCE_GROUPS_R1"
PHASE4B_REVIEW_PROMOTION_POLICY = "USER_SELECTED_SAFE_CANDIDATE_PROMOTION_R1"
PHASE4B_GROUP_REPRESENTATIVE_POLICY = "IDENTITY_PURITY_CLEAN_RATIO_REPRESENTATIVE_R1"
PHASE4B_GROUP_CONFIDENCE_POLICY = "INDEPENDENT_TRACKLET_GROUP_AGREEMENT_R1"
PHASE4B_IDENTITY_NEGATIVE_TOP_K = 3
PHASE4B_IDENTITY_NEGATIVE_MIN_CROP_MARGIN_MEDIAN = 0.0
PHASE4B_IDENTITY_NEGATIVE_MIN_POSITIVE_MARGIN_SUPPORT = 0.50
PHASE4B_IDENTITY_NEGATIVE_MIN_PROTOTYPE_MARGIN = 0.0
PHASE4B_RESCUE_REVIEW_MAX_CANDIDATES = 3
PHASE4B_IDENTITY_NEGATIVE_DUPLICATE_COSINE = 0.995
PHASE4B_USER_ROLE_NEGATIVE_TOP_K = 3
PHASE4B_USER_ROLE_MIN_CROP_MARGIN_MEDIAN = 0.0
PHASE4B_USER_ROLE_MIN_POSITIVE_MARGIN_SUPPORT = 0.50
PHASE4B_USER_ROLE_MIN_PROTOTYPE_MARGIN = 0.0
PHASE4B_CONTINUATION_MAX_GAP_FRAMES = 3
PHASE4B_CONTINUATION_MIN_PROTOTYPE_COSINE = 0.88
PHASE4B_CONTINUATION_MIN_ENDPOINT_IOU = 0.10
PHASE4B_CONTINUATION_MAX_CENTER_DISTANCE_BY_HEIGHT = 1.50
PHASE4B_CONTINUATION_MAX_HEIGHT_RATIO = 2.50
PHASE4B_STABLE_EVIDENCE_CLEAN_FRAME_CAP = 24
PHASE4B_STABLE_EVIDENCE_DURATION_CAP = 48


def _register_sports_osnet_numpy_safe_globals(torch: Any, np: Any) -> dict[str, Any]:
    """Product compatibility for the frozen Sports-OSNet checkpoint.

    The caller verifies the checkpoint SHA-256 first.  This function only
    allowlists the NumPy globals required for ``torch.load(weights_only=True)``
    on PyTorch 2.6+/NumPy 2.x; unsafe pickle loading remains forbidden.
    """
    serialization = getattr(torch, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        raise RuntimeError(
            "torch.serialization.add_safe_globals is required for Sports-OSNet"
        )

    numpy_core = getattr(np, "_core", None)
    multiarray = getattr(numpy_core, "multiarray", None)
    numpy_scalar = getattr(multiarray, "scalar", None)
    if numpy_scalar is None:
        legacy_core = getattr(np, "core", None)
        legacy_multiarray = getattr(legacy_core, "multiarray", None)
        numpy_scalar = getattr(legacy_multiarray, "scalar", None)
    if numpy_scalar is None:
        raise RuntimeError("NumPy multiarray.scalar is unavailable")

    entries: list[Any] = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (numpy_scalar, "numpy._core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
    ]
    names = [
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar",
        "numpy.dtype",
    ]
    numpy_dtypes = getattr(np, "dtypes", None)
    for dtype_name in ("Float64DType", "Float32DType"):
        dtype_class = getattr(numpy_dtypes, dtype_name, None) if numpy_dtypes is not None else None
        if dtype_class is not None:
            qualified_name = f"numpy.dtypes.{dtype_name}"
            entries.append((dtype_class, qualified_name))
            names.append(qualified_name)
    add_safe_globals(entries)
    return {
        "status": "REGISTERED",
        "policy_version": "SPORTS_OSNET_NUMPY2_WEIGHTS_ONLY_COMPAT_R2",
        "registered_safe_globals": names,
        "weights_only_false_allowed": False,
    }
PHASE4B_NON_PLAYER_ROLE_SHOT_STATUS = "SEARCH_EXHAUSTED_NON_PLAYER_ROLE"
PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_SCHEMA = "kickclip.phase4b_persistent_role_negative_memory.v1"
PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY = "USER_CONFIRMED_NON_PLAYER_PLUS_DETECTOR_ROLE_NEGATIVES_R1"
PHASE4B_PERSISTENT_ROLE_NEGATIVE_DUPLICATE_COSINE = 0.995
PHASE4B_IDENTITY_OBSERVABILITY_POLICY = "SINGLE_PERSON_IDENTITY_OBSERVABILITY_R1"
PHASE4B_UNREVIEWABLE_GROUP_OCCLUSION_STATUS = "SEARCH_EXHAUSTED_UNREVIEWABLE_GROUP_OCCLUSION"
PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_RATIO = 0.20
PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_FRAMES = 3
PHASE4B_OBSERVABILITY_MAXIMUM_REQUIRED_CLEAN_FRAMES = 6
PHASE4B_OBSERVABILITY_MAX_CANDIDATE_OVERLAP_RATIO = 0.35
PHASE4B_OBSERVABILITY_DUPLICATE_DETECTION_IOU = 0.78
PHASE4B_OBSERVABILITY_MAX_WIDTH_HEIGHT_RATIO = 1.20
PHASE4B_OBSERVABILITY_MIN_HEIGHT_RATIO = 0.08
PHASE4B_OBSERVABILITY_MIN_HEIGHT_PIXELS = 64.0
PHASE4B_OBSERVABILITY_EDGE_MARGIN_RATIO = 0.02
PHASE4B_OBSERVABILITY_MAX_CROWDED_RATIO = 0.80
PHASE4B_OBSERVABILITY_MAX_UNSTABLE_RATIO = 0.60
PHASE4B_OBSERVABILITY_MAX_AREA_JUMP_RATIO = 2.75
PHASE4B_OBSERVABILITY_MAX_ASPECT_JUMP_RATIO = 2.00
PHASE4B_OBSERVABILITY_MAX_CENTER_JUMP_NORMALIZED = 1.25
PHASE4B_OBSERVABILITY_MIN_DETECTION_CONFIDENCE = 0.25
PHASE4B_IDENTITY_PURITY_POLICY = "TARGET_AGNOSTIC_CONFIRMED_CHANGE_POINT_SEGMENTATION_R2"
PHASE4B_IDENTITY_PURITY_MAX_PROBES = 48
PHASE4B_IDENTITY_PURITY_MAX_GAP_SECONDS = 0.20
PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_OBSERVATIONS = 3
PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_CLEAN_FRAMES = 3
PHASE4B_IDENTITY_PURITY_LOCAL_WINDOW = 3
PHASE4B_IDENTITY_PURITY_MIN_LOCAL_COHERENCE = 0.65
PHASE4B_IDENTITY_PURITY_MAX_CROSS_COSINE = 0.60
PHASE4B_IDENTITY_PURITY_MAX_ADJACENT_COSINE = 0.55
PHASE4B_IDENTITY_PURITY_MIN_CONTRAST = 0.10
PHASE4B_IDENTITY_PURITY_GEOMETRY_JUMP_THRESHOLD = 0.75
PHASE4B_IDENTITY_PURITY_MIN_INTERNAL_MEDIAN_COSINE = 0.50
PHASE4B_TARGET_CONTEXT_POLICY = "APPROVED_REFERENCE_TORSO_COLOR_SOFT_EVIDENCE_R1"
PHASE4B_TARGET_CONTEXT_H_BINS = 12
PHASE4B_TARGET_CONTEXT_S_BINS = 4
PHASE4B_TARGET_CONTEXT_V_BINS = 4
PHASE4B_TARGET_CONTEXT_GRAY_BINS = 16
PHASE4B_TARGET_CONTEXT_MIN_SIMILARITY = 0.0
PHASE4B_CORROBORATION_POLICY = "INDEPENDENT_TRACKLET_SPATIOTEMPORAL_APPEARANCE_CORROBORATION_R1"
PHASE4B_CORROBORATION_MIN_COMMON_FRAMES = 2
PHASE4B_CORROBORATION_MIN_PROTOTYPE_COSINE = 0.75
PHASE4B_CORROBORATION_MIN_MEAN_IOU = 0.15
PHASE4B_CORROBORATION_MAX_CENTER_DISTANCE_BY_HEIGHT = 0.85
PHASE4B_CORROBORATION_MAX_HEIGHT_RATIO = 2.25
PHASE4B_CORROBORATION_MAX_CONTEXT_DISTANCE = 0.30


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate immutable R1 selection artifacts and invoke the supplied "
            "target-centric V1/V2 research runtime without inventing production_r3."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--initial-bbox", nargs=4, type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--reacquisition-mode", default="assisted")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tracking-launch-manifest", type=Path, required=True)
    parser.add_argument("--shot-boundaries", type=Path, required=True)
    parser.add_argument("--target-selection", type=Path, required=True)
    parser.add_argument("--target-reference-set", type=Path, required=True)
    parser.add_argument("--earlier-anchor-decision", type=Path, required=True)
    parser.add_argument("--target-memory-revision", type=Path, default=None)
    parser.add_argument("--target-memory-sha256", default=None)
    parser.add_argument("--candidate-scoring-generation", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ambiguity-id", default=None)
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--confirmed-candidate", default=None)
    decision.add_argument("--confirm-absent", action="store_true")
    decision.add_argument("--rejected-candidate", default=None)
    decision.add_argument("--unreviewable-candidate", default=None)
    decision.add_argument(
        "--promote-review-candidate",
        default=None,
        help=(
            "Promote any safe plausible candidate from the current review catalog "
            "into the pending assisted-review set without confirming identity or "
            "changing positive/negative memory."
        ),
    )
    decision.add_argument(
        "--reject-all-candidates",
        action="store_true",
        help=(
            "Record an explicit NONE_OF_THESE decision for the pending ambiguity, "
            "materialize a persistent user-confirmed identity-negative memory, and "
            "continue searching the next reviewed shot."
        ),
    )
    decision.add_argument(
        "--reject-all-candidates-as-non-player-role",
        action="store_true",
        help=(
            "Record that every pending candidate is a coach/staff/referee or other "
            "non-player role, materialize persistent role-negative memory, and "
            "continue searching the next reviewed shot."
        ),
    )
    review = parser.add_mutually_exclusive_group()
    review.add_argument(
        "--approve-review",
        choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"),
        default=None,
    )
    review.add_argument(
        "--reject-review",
        choices=("MEMORY", "SEGMENT", "STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"),
        default=None,
    )
    parser.add_argument("--reviewer", default="USER")
    parser.add_argument("--review-note", default="")
    parser.add_argument("--review-decision-artifact", type=Path, default=None)
    parser.add_argument("--review-decision-sha256", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--materialize-selection-view-only",
        action="store_true",
        help=(
            "Validate immutable selection inputs and materialize the forward/backward "
            "selected-shot video views without running Phase-1 or the provided E2E runtime."
        ),
    )
    parser.add_argument(
        "--execute-bidirectional-phase1-only",
        action="store_true",
        help=(
            "Materialize the selected-shot views and execute the frozen Phase-1 "
            "Stage 1/2 contract in both forward and backward directions without "
            "running the provided cross-shot E2E runtime."
        ),
    )
    parser.add_argument(
        "--merge-bidirectional-timelines-only",
        action="store_true",
        help=(
            "Merge existing Phase 3-B forward/backward Stage-2 timelines into "
            "source-frame coordinates without rerunning inference or the E2E runtime."
        ),
    )
    parser.add_argument(
        "--build-phase4a-initial-memory-only",
        action="store_true",
        help=(
            "Build and validate the immutable initial target-memory revision from "
            "the completed Phase 3-C selected-shot timeline and immutable selection "
            "references without rerunning detector, tracker, or cross-shot scoring."
        ),
    )
    parser.add_argument(
        "--run-phase4b-first-cross-shot-only",
        action="store_true",
        help=(
            "Use an approved immutable target-memory revision to run the frozen "
            "target-agnostic detector/tracklet/ReID ranking contract across reviewed "
            "shots in order. Shots with no reviewable candidate are marked exhausted; "
            "execution stops at the first assisted-confirmation candidate or after all "
            "remaining reviewed shots are safely exhausted."
        ),
    )
    parser.add_argument(
        "--contract-test-skip-phase1-compatibility",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def required_research_files(root: Path) -> dict[str, Path]:
    return {
        "e2e_runner": root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        "e2e_verifier": root / "target_centric_tracking_e2e_v1" / "verify_e2e_installation.py",
        "phase1_runner": root / "target_centric_tracking_v1" / "run_phase1_frozen_pipeline.py",
        "phase1_manifest": root / "target_centric_tracking_v1" / "phase1_frozen_manifest.json",
        "phase1_stage0": root / "target_centric_tracking_v1" / "stage0_audit_inputs.py",
        "same_shot_stage1": root / "target_centric_tracking_v1" / "stage1_generate_rfdetr_detections.py",
        "same_shot_stage2": root / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
        "short_clip_compatibility": root / "target_centric_tracking_v2" / "stage3c0_run_short_clip_phase1_compatibility.py",
        "cross_shot_b0": root / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        "cross_shot_b1": root / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        "cross_shot_b2": root / "target_centric_tracking_v2" / "stage3b2_make_safe_cross_shot_decision.py",
        "cross_shot_b3": root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        "v6_reid_helper": root / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    }


def validate_input_file(path: Path, expected_sha: str | None, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return digest


def dependency_report(root: Path) -> dict[str, Any]:
    files = required_research_files(root)
    phase1_manifest_path = files["phase1_manifest"]
    models: list[dict[str, Any]] = []
    if phase1_manifest_path.is_file():
        manifest = read_object(phase1_manifest_path)
        for logical, record in (manifest.get("models") or {}).items():
            if not isinstance(record, Mapping) or not record.get("verified"):
                continue
            model_path = root / str(record.get("path") or "").replace("\\", "/")
            models.append(
                {
                    "logical": logical,
                    "path": str(model_path),
                    "expected_sha256": record.get("sha256"),
                    "exists": model_path.is_file(),
                    "sha256_matches": (
                        model_path.is_file()
                        and sha256_file(model_path) == str(record.get("sha256") or "")
                    ),
                }
            )
    missing = [str(path) for path in files.values() if not path.is_file()]
    missing.extend(
        row["path"]
        for row in models
        if not row["exists"] or not row["sha256_matches"]
    )
    return {
        "files": {name: str(path) for name, path in files.items()},
        "models": models,
        "missing": sorted(set(missing)),
        "global_ID_tracking_upgrade_v7_is_not_aliased_to_v6": True,
    }


def write_blocked_state(
    output_dir: Path,
    *,
    decision: str,
    failure_code: str,
    message: str,
    launch: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    extra_runtime: Mapping[str, Any] | None = None,
) -> None:
    state = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "updated_at": now_iso(),
        "status": "COMPLETE_WITH_SAFE_BLOCK",
        "decision": decision,
        "failure_code": failure_code,
        "message": message,
        "execution_kind": "EVENT_CANDIDATE_HANDOFF_R1",
        "pending_action": None,
        "shots": [],
        "ambiguities": [],
        "confirmations": [],
        "runtime": {
            "integration_path": "B.ADD_THIN_BACKEND_ADAPTER_TO_V1_V2_STAGES",
            "provided_e2e_runner_used": False,
            "selection_aware_video_adapter_used": False,
            "synthetic_tracking_used": False,
            "observation_copy_used_as_success": False,
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
            "launch_manifest_sha256": sha256_file(Path(str(launch["path"]))),
            "dependency_report": dependencies,
            **dict(extra_runtime or {}),
        },
    }
    atomic_json(output_dir / "pipeline_state.json", state)
    atomic_json(
        output_dir / "pipeline_summary.json",
        {
            "status": state["status"],
            "decision": decision,
            "failure_code": failure_code,
            "active_or_reacquired_bbox_frames": 0,
            "unresolved_shot_count": None,
            "preview_generated": False,
            "full_event_recommendation_e2e": "NOT_RUN",
        },
    )


def invoke(command: Sequence[str], cwd: Path) -> int:
    completed = subprocess.run(list(command), cwd=str(cwd), shell=False, check=False)
    return int(completed.returncode)


def _safe_name(value: object) -> str:
    text = str(value or "unknown")
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in text)


def _video_metadata(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {path}")
    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    if (
        metadata["width"] < 1
        or metadata["height"] < 1
        or metadata["fps"] <= 0
        or metadata["frame_count"] < 1
    ):
        raise RuntimeError(f"Invalid source video metadata: {path}")
    metadata["duration_seconds"] = metadata["frame_count"] / metadata["fps"]
    return metadata


def _find_ffmpeg() -> Path:
    candidates: list[Path] = []
    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(Path(discovered))
    for name in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        configured = os.environ.get(name)
        if configured:
            candidates.append(Path(configured))
    if os.name == "nt":
        candidates.extend(
            [
                Path("D:/ffmpeg/bin/ffmpeg.exe"),
                Path("C:/ffmpeg/bin/ffmpeg.exe"),
                Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
            ]
        )
    for candidate in candidates:
        executable = candidate.expanduser().resolve()
        if not executable.is_file():
            continue
        checked = subprocess.run(
            [str(executable), "-version"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if checked.returncode == 0:
            return executable
    raise RuntimeError(
        "FFmpeg is required to materialize an exact reversed selected-shot source."
    )


def _write_reversed_video_range(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    end_frame_inclusive: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if start_frame < 0 or end_frame_inclusive < start_frame:
        raise ValueError("Invalid reversed video frame range.")

    ffmpeg = _find_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    temporary.unlink(missing_ok=True)
    fps = float(metadata["fps"])
    expected = end_frame_inclusive - start_frame + 1
    filter_chain = (
        f"trim=start_frame={start_frame}:end_frame={end_frame_inclusive + 1},"
        "setpts=PTS-STARTPTS,reverse,"
        f"setpts=N/({fps:.12f}*TB),fps={fps:.12f}"
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        filter_chain,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.12f}",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(
        command,
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "FFmpeg failed to materialize reversed selected-shot source: "
            + (completed.stderr.strip() or subprocess.list2cmdline(command))
        )
    observed = _video_metadata(temporary)
    if int(observed["frame_count"]) != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Reversed selected-shot frame mismatch: "
            f"{observed['frame_count']}/{expected}"
        )
    if (
        int(observed["width"]) != int(metadata["width"])
        or int(observed["height"]) != int(metadata["height"])
        or abs(float(observed["fps"]) - fps) > 1e-3
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Reversed selected-shot metadata mismatch.")
    temporary.replace(output)
    return {
        "ffmpeg_path": str(ffmpeg),
        "ffmpeg_sha256": sha256_file(ffmpeg),
        "codec": "H264_LIBX264_CRF18",
        "frame_count": expected,
    }


def _write_video_range(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    end_frame_inclusive: int,
    metadata: Mapping[str, Any],
) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(metadata["fps"]),
        (int(metadata["width"]), int(metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video writer: {temporary}")
    count = 0
    try:
        for _ in range(start_frame, end_frame_inclusive + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended while materializing: {output}")
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    expected = end_frame_inclusive - start_frame + 1
    if count != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Materialized video frame mismatch: {count}/{expected}")
    temporary.replace(output)


def _read_shots(
    boundaries: Mapping[str, Any],
    *,
    frame_count: int,
) -> list[dict[str, Any]]:
    rows = (
        boundaries.get("shots")
        or boundaries.get("boundaries")
        or boundaries.get("shot_boundaries")
        or []
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reviewed shot boundaries are empty.")

    shots: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("Reviewed shot boundary contains a non-object row.")
        start = int(raw.get("start_frame", -1))
        end = int(raw.get("end_frame_inclusive", raw.get("end_frame", -1)))
        shot_id = str(raw.get("shot_id") or f"shot_{index:04d}").strip()
        if not shot_id or start < 0 or end < start:
            raise ValueError("Reviewed shot boundary is invalid.")
        shots.append(
            {
                **dict(raw),
                "shot_index": index,
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                "cut_in_frame": None if start == 0 else start,
                "cut_out_frame": (
                    None if end == frame_count - 1 else end + 1
                ),
            }
        )

    shots.sort(
        key=lambda row: (
            int(row["start_frame"]),
            int(row["end_frame_inclusive"]),
            str(row["shot_id"]),
        )
    )
    seen_ids: set[str] = set()
    expected_start = 0
    for index, row in enumerate(shots):
        shot_id = str(row["shot_id"])
        start = int(row["start_frame"])
        end = int(row["end_frame_inclusive"])
        if (
            shot_id in seen_ids
            or start != expected_start
            or end >= frame_count
        ):
            raise ValueError(
                "Reviewed shots must form one unique contiguous full-video coverage."
            )
        row["shot_index"] = index
        row["frame_count"] = end - start + 1
        row["cut_in_frame"] = None if start == 0 else start
        row["cut_out_frame"] = None if end == frame_count - 1 else end + 1
        seen_ids.add(shot_id)
        expected_start = end + 1
    if expected_start != frame_count:
        raise ValueError("Reviewed shots do not cover the complete source video.")
    return shots


def materialize_selection_view(
    *,
    output_dir: Path,
    launch: Mapping[str, Any],
    target_selection: Mapping[str, Any],
    boundaries: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    source_video = Path(
        str((launch.get("source_video") or {}).get("path") or "")
    ).resolve()
    validate_input_file(
        source_video,
        str((launch.get("source_video") or {}).get("sha256") or ""),
        "source video",
    )
    metadata = _video_metadata(source_video)
    anchor_frame = int(target_selection["best_anchor_frame"])
    selected_shot_id = str(target_selection["shot_id"])
    shots = _read_shots(boundaries, frame_count=int(metadata["frame_count"]))
    selected_index = next(
        (
            index
            for index, row in enumerate(shots)
            if row["shot_id"] == selected_shot_id
        ),
        None,
    )
    if selected_index is None:
        raise ValueError("Selected shot is absent from reviewed boundaries.")
    selected = shots[selected_index]
    shot_start = int(selected["start_frame"])
    shot_end = int(selected["end_frame_inclusive"])
    if not shot_start <= anchor_frame <= shot_end:
        raise ValueError("Selected anchor is outside its reviewed shot.")

    workspace = output_dir / "_selection_anchor_view"
    workspace.mkdir(parents=True, exist_ok=True)
    suffix_video = workspace / "selection_anchor_suffix.mp4"
    forward_video = workspace / "selected_shot_forward.mp4"
    backward_video = workspace / "selected_shot_backward.mp4"
    view_path = workspace / "selection_view.json"

    forward_frame_count = shot_end - anchor_frame + 1
    backward_frame_count = anchor_frame - shot_start + 1
    expected_view = {
        "schema_version": SELECTION_VIEW_SCHEMA,
        "bidirectional_input_policy": BIDIRECTIONAL_INPUT_POLICY,
        "source_video_sha256": metadata["sha256"],
        "source_offset_frame": anchor_frame,
        "source_frame_count": metadata["frame_count"],
        "selected_shot_id": selected_shot_id,
        "selected_shot_start_frame": shot_start,
        "selected_shot_end_frame_inclusive": shot_end,
        "forward_frame_count": forward_frame_count,
        "backward_frame_count": backward_frame_count,
        "reviewed_shot_contract_schema": REVIEWED_SHOT_CONTRACT_SCHEMA,
        "reviewed_shot_count": len(shots),
    }

    reusable = False
    observed: dict[str, Any] = {}
    if (
        view_path.is_file()
        and suffix_video.is_file()
        and forward_video.is_file()
        and backward_video.is_file()
        and not overwrite
    ):
        observed = read_object(view_path)
        reusable = all(
            observed.get(key) == value
            for key, value in expected_view.items()
        )
        if reusable:
            recorded_forward = observed.get("selected_shot_forward_source") or {}
            recorded_backward = observed.get("selected_shot_backward_source") or {}
            recorded_suffix = observed.get("selection_view_video") or {}
            reusable = (
                recorded_forward.get("sha256") == sha256_file(forward_video)
                and recorded_backward.get("sha256") == sha256_file(backward_video)
                and recorded_suffix.get("sha256") == sha256_file(suffix_video)
                and int(recorded_forward.get("frame_count") or 0)
                == forward_frame_count
                and int(recorded_backward.get("frame_count") or 0)
                == backward_frame_count
            )

    reverse_materialization: dict[str, Any]
    if not reusable:
        _write_video_range(
            source_video,
            suffix_video,
            start_frame=anchor_frame,
            end_frame_inclusive=int(metadata["frame_count"]) - 1,
            metadata=metadata,
        )
        _write_video_range(
            source_video,
            forward_video,
            start_frame=anchor_frame,
            end_frame_inclusive=shot_end,
            metadata=metadata,
        )
        reverse_materialization = _write_reversed_video_range(
            source_video,
            backward_video,
            start_frame=shot_start,
            end_frame_inclusive=anchor_frame,
            metadata=metadata,
        )
    else:
        reverse_materialization = dict(
            observed.get("reverse_materialization") or {}
        )

    suffix_metadata = _video_metadata(suffix_video)
    forward_metadata = _video_metadata(forward_video)
    backward_metadata = _video_metadata(backward_video)
    if int(forward_metadata["frame_count"]) != forward_frame_count:
        raise RuntimeError("Forward selected-shot frame count mismatch.")
    if int(backward_metadata["frame_count"]) != backward_frame_count:
        raise RuntimeError("Backward selected-shot frame count mismatch.")

    future = shots[selected_index:]
    shot_map: list[dict[str, Any]] = []
    for raw_index, shot in enumerate(future):
        local_start = (
            0
            if raw_index == 0
            else int(shot["start_frame"]) - anchor_frame
        )
        local_end = int(shot["end_frame_inclusive"]) - anchor_frame
        shot_map.append(
            {
                "raw_shot_id": f"shot_{raw_index:04d}",
                "raw_shot_index": raw_index,
                "source_shot_id": shot["shot_id"],
                "source_start_frame": int(shot["start_frame"]),
                "source_end_frame_inclusive": int(shot["end_frame_inclusive"]),
                "processed_source_start_frame": max(
                    anchor_frame, int(shot["start_frame"])
                ),
                "local_start_frame": local_start,
                "local_end_frame_inclusive": local_end,
            }
        )
    translated_cuts = [
        int(row["local_start_frame"])
        for row in shot_map[1:]
        if int(row["local_start_frame"]) > 0
    ]

    forward_source = {
        "path": str(forward_video),
        "sha256": sha256_file(forward_video),
        "direction": "FORWARD",
        "source_anchor_frame": anchor_frame,
        "source_terminal_frame": shot_end,
        "source_frame_formula": "source_frame=anchor_frame+local_frame",
        **{
            key: value
            for key, value in forward_metadata.items()
            if key not in {"path", "sha256"}
        },
    }
    backward_source = {
        "path": str(backward_video),
        "sha256": sha256_file(backward_video),
        "direction": "BACKWARD_REVERSED_VIDEO",
        "source_anchor_frame": anchor_frame,
        "source_terminal_frame": shot_start,
        "source_frame_formula": "source_frame=anchor_frame-local_frame",
        **{
            key: value
            for key, value in backward_metadata.items()
            if key not in {"path", "sha256"}
        },
    }
    view = {
        "schema_version": SELECTION_VIEW_SCHEMA,
        "created_at": now_iso(),
        "bidirectional_input_policy": BIDIRECTIONAL_INPUT_POLICY,
        "source_video": metadata,
        "source_video_sha256": metadata["sha256"],
        "source_offset_frame": anchor_frame,
        "source_frame_count": metadata["frame_count"],
        "selected_shot_id": selected_shot_id,
        "selected_shot_start_frame": shot_start,
        "selected_shot_end_frame_inclusive": shot_end,
        "forward_frame_count": forward_frame_count,
        "backward_frame_count": backward_frame_count,
        "reviewed_shot_contract_schema": REVIEWED_SHOT_CONTRACT_SCHEMA,
        "reviewed_shot_count": len(shots),
        "reviewed_shot_boundaries_sha256": str(
            (launch.get("shot_boundaries") or {}).get("sha256") or ""
        ),
        "reviewed_shots": [dict(row) for row in shots],
        "unresolved_prefix_frame_count": anchor_frame - shot_start,
        "selection_view_video": {
            "path": str(suffix_video),
            "sha256": sha256_file(suffix_video),
            **{
                key: value
                for key, value in suffix_metadata.items()
                if key not in {"path", "sha256"}
            },
        },
        "same_shot_pre_cut_source": forward_source,
        "selected_shot_forward_source": forward_source,
        "selected_shot_backward_source": backward_source,
        "bidirectional_merge_contract": {
            "anchor_is_present_in_both_sources": True,
            "anchor_conflict_policy": "FORWARD_ANCHOR_PRIMARY",
            "forward_local_to_source": "anchor_frame+local_frame",
            "backward_local_to_source": "anchor_frame-local_frame",
            "prefix_interpolation_forbidden": True,
            "observation_copy_as_tracking_success_forbidden": True,
            "reverse_phase1_execution_complete": False,
            "forward_phase1_execution_complete": False,
            "timeline_merge_complete": False,
        },
        "cross_shot_search_source": {
            "path": str(suffix_video),
            "sha256": sha256_file(suffix_video),
            "source_start_frame": shot_end + 1,
            "reviewed_shot_count": max(0, len(future) - 1),
        },
        "translated_cut_frames": translated_cuts,
        "shot_map": shot_map,
        "reverse_materialization": reverse_materialization,
        "frame_zero_fallback_used": False,
        "actual_anchor_materialized_as_view_frame_zero": True,
        "selected_shot_bidirectional_inputs_materialized": True,
    }
    atomic_json(view_path, view)
    return view



def _phase1_stage2_timeline_summary(
    *,
    timeline_path: Path,
    expected_video_sha256: str,
    expected_frame_count: int,
    direction: str,
) -> dict[str, Any]:
    timeline_path = timeline_path.resolve()
    timeline_sha256 = validate_input_file(
        timeline_path,
        None,
        f"{direction.lower()} Phase-1 Stage-2 timeline",
    )
    timeline = read_object(timeline_path)
    frames = timeline.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(
            f"{direction} Phase-1 Stage-2 timeline has no frame list."
        )
    if len(frames) != int(expected_frame_count):
        raise RuntimeError(
            f"{direction} Phase-1 Stage-2 frame mismatch: "
            f"{len(frames)}/{expected_frame_count}"
        )
    observed_indices = [
        int(row.get("frame_index", -1))
        for row in frames
        if isinstance(row, Mapping)
    ]
    if observed_indices != list(range(expected_frame_count)):
        raise RuntimeError(
            f"{direction} Phase-1 Stage-2 frame indices are not contiguous."
        )
    video = timeline.get("video") if isinstance(timeline.get("video"), Mapping) else {}
    if str(video.get("sha256") or "") != expected_video_sha256:
        raise RuntimeError(
            f"{direction} Phase-1 Stage-2 source video SHA-256 mismatch."
        )
    state_counts: dict[str, int] = {}
    bbox_frame_count = 0
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                f"{direction} Phase-1 Stage-2 timeline contains a non-object frame."
            )
        state = str(raw.get("state") or "")
        state_counts[state] = state_counts.get(state, 0) + 1
        if raw.get("bbox_xyxy") is not None:
            bbox_frame_count += 1
    return {
        "path": str(timeline_path),
        "sha256": timeline_sha256,
        "schema_version": timeline.get("schema_version"),
        "stage": timeline.get("stage"),
        "frame_count": len(frames),
        "bbox_frame_count": bbox_frame_count,
        "state_counts": dict(sorted(state_counts.items())),
        "video_sha256": expected_video_sha256,
    }


def execute_bidirectional_selected_shot_phase1(
    *,
    project_root: Path,
    output_dir: Path,
    test_name: str,
    view: dict[str, Any],
    initial_bbox: Sequence[float],
    device: str,
    overwrite: bool,
    compatibility_module: Any,
) -> dict[str, Any]:
    """Execute real frozen Stage 1/2 on both anchor directions.

    Review-required return states are preserved but do not block this Phase 3-B
    gate because Phase 3-B authorizes only the conservative Stage-2 timelines.
    No visual-review decision is inferred or supplied here.
    """

    workspace = output_dir / "_bidirectional_phase1"
    workspace.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(test_name)
    runtime_device = "cpu" if device == "mps" else device

    directions = {
        "forward": {
            "source": dict(view["selected_shot_forward_source"]),
            "expected_frame_count": int(view["forward_frame_count"]),
            "test_name": f"e2e_{safe}__selected_shot_forward_phase1",
        },
        "backward": {
            "source": dict(view["selected_shot_backward_source"]),
            "expected_frame_count": int(view["backward_frame_count"]),
            "test_name": f"e2e_{safe}__selected_shot_backward_phase1",
        },
    }

    results: dict[str, Any] = {}
    pending_reviews: list[dict[str, Any]] = []
    for direction, contract in directions.items():
        source = Path(str(contract["source"]["path"])).resolve()
        validate_input_file(
            source,
            str(contract["source"]["sha256"]),
            f"selected-shot {direction} source",
        )
        phase1_test_name = str(contract["test_name"])
        expected_frame_count = int(contract["expected_frame_count"])
        if expected_frame_count == 1:
            results[direction] = {
                "direction": direction.upper(),
                "phase1_test_name": None,
                "source": contract["source"],
                "execution_required": False,
                "execution_status": "ANCHOR_ONLY_NO_DIRECTIONAL_RANGE",
                "review_required": False,
                "review_stage": None,
                "automatic_review_decision_used": False,
                "real_rfdetr_inference_performed": False,
                "real_same_shot_stage2_performed": False,
                "stage2_timeline": None,
                "anchor_frame_is_merged_from_forward_primary": True,
            }
            continue
        gate_path = workspace / f"{direction}_compatibility_gate.json"
        gate = compatibility_module.execute_phase1_compatibility(
            project_root=project_root,
            video=source,
            initial_bbox=initial_bbox,
            original_stage0_test_name=(
                f"{phase1_test_name}__stage0_original"
            ),
            phase1_test_name=phase1_test_name,
            compatibility_test_name=(
                f"{phase1_test_name}__short_clip_compatibility"
            ),
            device=runtime_device,
            gate_path=gate_path,
            review_decisions=None,
            reviewer="KICKCLIP_PHASE3B_BIDIRECTIONAL_GATE",
            review_note=(
                "Phase 3-B executes only the frozen Stage 1/2 contract. "
                "No automatic visual-review decision is supplied."
            ),
            allow_review_required=True,
            require_final_outputs=False,
            overwrite_phase1=overwrite,
        )
        if gate.get("stage1_authorized") is not True:
            raise RuntimeError(
                f"{direction} Phase-1 Stage 1 was not authorized."
            )
        if gate.get("real_same_shot_stage2_performed") is not True:
            raise RuntimeError(
                f"{direction} real same-shot Stage 2 evidence is missing."
            )
        phase1_dir = (
            project_root
            / "runs"
            / "target_centric_tracking_v1"
            / phase1_test_name
        )
        timeline_summary = _phase1_stage2_timeline_summary(
            timeline_path=phase1_dir / "target_timeline.json",
            expected_video_sha256=str(contract["source"]["sha256"]),
            expected_frame_count=int(contract["expected_frame_count"]),
            direction=direction.upper(),
        )
        result = {
            "direction": direction.upper(),
            "phase1_test_name": phase1_test_name,
            "execution_required": True,
            "source": contract["source"],
            "compatibility_gate_path": str(gate_path),
            "compatibility_gate_sha256": sha256_file(gate_path),
            "execution_status": gate.get("execution_status"),
            "review_required": bool(gate.get("review_required")),
            "review_stage": gate.get("review_stage"),
            "automatic_review_decision_used": bool(
                gate.get("automatic_review_decision_used")
            ),
            "real_rfdetr_inference_performed": bool(
                gate.get("real_rfdetr_inference_performed")
            ),
            "real_same_shot_stage2_performed": bool(
                gate.get("real_same_shot_stage2_performed")
            ),
            "stage2_timeline": timeline_summary,
        }
        results[direction] = result
        if result["review_required"]:
            pending_reviews.append(
                {
                    "direction": direction.upper(),
                    "stage": result["review_stage"],
                    "compatibility_gate_path": str(gate_path),
                }
            )

    if any(
        result["automatic_review_decision_used"]
        for result in results.values()
    ):
        raise RuntimeError(
            "Automatic review decisions are forbidden in Phase 3-B."
        )

    report = {
        "schema_version": PHASE3B_REPORT_SCHEMA,
        "created_at": now_iso(),
        "status": "PASS",
        "decision": "AUTHORIZE_PHASE3C_BIDIRECTIONAL_TIMELINE_MERGE",
        "policy": BIDIRECTIONAL_PHASE1_POLICY,
        "selected_shot_id": view["selected_shot_id"],
        "selected_shot_start_frame": int(
            view["selected_shot_start_frame"]
        ),
        "anchor_frame": int(view["source_offset_frame"]),
        "selected_shot_end_frame_inclusive": int(
            view["selected_shot_end_frame_inclusive"]
        ),
        "anchor_bbox_xyxy": [float(value) for value in initial_bbox],
        "forward": results["forward"],
        "backward": results["backward"],
        "pending_visual_reviews": pending_reviews,
        "pending_visual_review_count": len(pending_reviews),
        "automatic_review_decision_used": False,
        "reverse_phase1_execution_complete": True,
        "forward_phase1_execution_complete": True,
        "timeline_merge_complete": False,
        "tracking_observations_copied_as_success": False,
        "synthetic_tracking_used": False,
    }
    report_path = output_dir / "phase3b_bidirectional_phase1_report.json"
    atomic_json(report_path, report)
    report_sha256 = sha256_file(report_path)

    merge_contract = dict(view.get("bidirectional_merge_contract") or {})
    merge_contract.update(
        {
            "reverse_phase1_execution_complete": True,
            "forward_phase1_execution_complete": True,
            "timeline_merge_complete": False,
            "phase3b_report_path": str(report_path),
            "phase3b_report_sha256": report_sha256,
            "forward_phase1_execution_required": bool(
                results["forward"]["execution_required"]
            ),
            "backward_phase1_execution_required": bool(
                results["backward"]["execution_required"]
            ),
        }
    )
    for direction in ("forward", "backward"):
        timeline = results[direction].get("stage2_timeline")
        if isinstance(timeline, Mapping):
            merge_contract[f"{direction}_stage2_timeline_path"] = timeline[
                "path"
            ]
            merge_contract[f"{direction}_stage2_timeline_sha256"] = timeline[
                "sha256"
            ]
        else:
            merge_contract[f"{direction}_stage2_timeline_path"] = None
            merge_contract[f"{direction}_stage2_timeline_sha256"] = None
    view["bidirectional_merge_contract"] = merge_contract
    view["selected_shot_bidirectional_phase1_executed"] = True
    view_path = output_dir / "_selection_anchor_view" / "selection_view.json"
    atomic_json(view_path, view)
    return report


def _phase3b_runtime_contract(
    *,
    output_dir: Path,
) -> dict[str, Any]:
    path = output_dir / "phase3b_bidirectional_phase1_report.json"
    if not path.is_file():
        return {
            "complete": False,
            "report_path": None,
            "report_sha256": None,
        }
    report = read_object(path)
    complete = (
        report.get("status") == "PASS"
        and report.get("reverse_phase1_execution_complete") is True
        and report.get("forward_phase1_execution_complete") is True
        and report.get("automatic_review_decision_used") is False
    )
    return {
        "complete": complete,
        "report_path": str(path.resolve()),
        "report_sha256": sha256_file(path),
        "policy": report.get("policy"),
        "pending_visual_review_count": int(
            report.get("pending_visual_review_count") or 0
        ),
    }



def _phase3c_runtime_contract(
    *,
    output_dir: Path,
) -> dict[str, Any]:
    report_path = output_dir / "phase3c_bidirectional_timeline_merge_report.json"
    timeline_path = output_dir / "phase3c_selected_shot_timeline.json"
    if not report_path.is_file() or not timeline_path.is_file():
        return {
            "complete": False,
            "report_path": None,
            "report_sha256": None,
            "timeline_path": None,
            "timeline_sha256": None,
            "resolved_prefix_frame_count": 0,
        }
    report = read_object(report_path)
    expected_timeline_sha = str(report.get("merged_timeline_sha256") or "")
    actual_timeline_sha = sha256_file(timeline_path)
    complete = (
        report.get("status") == "PASS"
        and report.get("decision") == "AUTHORIZE_PHASE3D_SERVICE_INTEGRATION_VALIDATION"
        and report.get("timeline_merge_complete") is True
        and report.get("automatic_target_confirmation") is False
        and report.get("synthetic_tracking_used") is False
        and len(expected_timeline_sha) == 64
        and expected_timeline_sha == actual_timeline_sha
    )
    return {
        "complete": complete,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "timeline_path": str(timeline_path.resolve()),
        "timeline_sha256": actual_timeline_sha,
        "policy": report.get("policy"),
        "resolved_prefix_frame_count": int(
            report.get("resolved_prefix_frame_count") or 0
        ),
        "merged_bbox_frame_count": int(
            report.get("merged_bbox_frame_count") or 0
        ),
    }


def _validated_directional_stage2_frames(
    *,
    direction: str,
    result: Mapping[str, Any],
    expected_frame_count: int,
) -> list[dict[str, Any]]:
    timeline_record = result.get("stage2_timeline")
    execution_required = bool(result.get("execution_required"))
    if not execution_required:
        if expected_frame_count != 1 or timeline_record is not None:
            raise RuntimeError(
                f"{direction} anchor-only Phase 3-B contract is invalid."
            )
        return []
    if not isinstance(timeline_record, Mapping):
        raise RuntimeError(f"{direction} Stage-2 timeline record is missing.")
    path = Path(str(timeline_record.get("path") or "")).resolve()
    expected_sha = str(timeline_record.get("sha256") or "")
    validate_input_file(path, expected_sha, f"{direction} Stage-2 timeline")
    document = read_object(path)
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frame_count:
        raise RuntimeError(
            f"{direction} Stage-2 frame count mismatch: "
            f"{len(frames) if isinstance(frames, list) else 'invalid'}"
            f"/{expected_frame_count}"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{direction} Stage-2 frame is not an object.")
        local = int(raw.get("frame_index", -1))
        if local < 0 or local >= expected_frame_count or local in seen:
            raise RuntimeError(
                f"{direction} Stage-2 local frame indices are invalid."
            )
        seen.add(local)
        bbox = raw.get("bbox_xyxy")
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
            raise RuntimeError(f"{direction} Stage-2 bbox is invalid at {local}.")
        normalized.append(dict(raw))
    if seen != set(range(expected_frame_count)):
        raise RuntimeError(f"{direction} Stage-2 timeline is not contiguous.")
    normalized.sort(key=lambda row: int(row["frame_index"]))
    return normalized


def _source_mapped_directional_frame(
    *,
    raw: Mapping[str, Any],
    direction: str,
    source_frame: int,
    local_frame: int,
    fps: float,
    shot_id: str,
) -> dict[str, Any]:
    row = dict(raw)
    original_detection_id = row.get("selected_detection_id")

    identity_source = str(row.get("identity_source") or "").strip()
    if not identity_source:
        if original_detection_id:
            row["identity_source"] = "RFDETR_ASSOCIATED_DETECTION"
        else:
            bbox_source = str(row.get("bbox_source") or "").strip()
            row["identity_source"] = (
                bbox_source
                if bbox_source and bbox_source.upper() != "NONE"
                else "NONE"
            )
    if "review_required" not in row:
        # Phase 3-B owns visual-review accounting. Missing Stage-2 frame
        # metadata is not converted into an approval or synthetic review.
        row["review_required"] = False

    row.update(
        {
            "frame_index": source_frame,
            "time_ms": int(round(source_frame * 1000.0 / fps)),
            "time_seconds": source_frame / fps,
            "shot_id": shot_id,
            "runtime_direction": direction,
            "runtime_local_frame_index": local_frame,
            "runtime_selected_detection_id": original_detection_id,
            "runtime_frame_mapping": (
                "source_frame=anchor_frame+local_frame"
                if direction == "FORWARD"
                else "source_frame=anchor_frame-local_frame"
            ),
            "phase3c_source": "REAL_FROZEN_STAGE2_DIRECTIONAL_TIMELINE",
        }
    )
    return row


def _segments_from_source_frames(
    frames: list[dict[str, Any]],
    *,
    fps: float,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(frames) + 1):
        boundary = index == len(frames)
        if not boundary:
            boundary = (
                str(frames[index].get("state") or "")
                != str(frames[start].get("state") or "")
                or str(frames[index].get("runtime_direction") or "")
                != str(frames[start].get("runtime_direction") or "")
            )
        if not boundary:
            continue
        rows = frames[start:index]
        first = rows[0]
        last = rows[-1]
        segments.append(
            {
                "segment_index": len(segments),
                "state": str(first.get("state") or ""),
                "runtime_direction": str(
                    first.get("runtime_direction") or ""
                ),
                "start_frame": int(first["frame_index"]),
                "end_frame_inclusive": int(last["frame_index"]),
                "start_ms": int(round(int(first["frame_index"]) * 1000.0 / fps)),
                "end_ms_exclusive": int(
                    round((int(last["frame_index"]) + 1) * 1000.0 / fps)
                ),
                "frame_count": len(rows),
                "bbox_frame_count": sum(
                    row.get("bbox_xyxy") is not None for row in rows
                ),
            }
        )
        start = index
    return segments


def merge_bidirectional_selected_shot_timelines(
    *,
    output_dir: Path,
    view: dict[str, Any],
    initial_bbox: Sequence[float],
) -> dict[str, Any]:
    """Remap both real Stage-2 timelines to source frames and merge them.

    The backward video is only a temporal reversal; bbox coordinates are not
    transformed. The anchor exists in both directional videos and the forward
    anchor is authoritative. No interpolation, observation copying, or identity
    confirmation is introduced by this merge.
    """

    phase3b_path = output_dir / "phase3b_bidirectional_phase1_report.json"
    if not phase3b_path.is_file():
        raise FileNotFoundError(phase3b_path)
    phase3b = read_object(phase3b_path)
    if (
        phase3b.get("status") != "PASS"
        or phase3b.get("decision")
        != "AUTHORIZE_PHASE3C_BIDIRECTIONAL_TIMELINE_MERGE"
        or phase3b.get("automatic_review_decision_used") is not False
    ):
        raise RuntimeError("Phase 3-B did not authorize timeline merging.")

    shot_start = int(view["selected_shot_start_frame"])
    anchor = int(view["source_offset_frame"])
    shot_end = int(view["selected_shot_end_frame_inclusive"])
    forward_count = int(view["forward_frame_count"])
    backward_count = int(view["backward_frame_count"])
    expected_shot_count = shot_end - shot_start + 1
    if forward_count != shot_end - anchor + 1:
        raise RuntimeError("Forward selected-shot mapping contract mismatch.")
    if backward_count != anchor - shot_start + 1:
        raise RuntimeError("Backward selected-shot mapping contract mismatch.")

    forward_result = phase3b.get("forward")
    backward_result = phase3b.get("backward")
    if not isinstance(forward_result, Mapping) or not isinstance(
        backward_result, Mapping
    ):
        raise RuntimeError("Phase 3-B directional results are missing.")
    forward_frames = _validated_directional_stage2_frames(
        direction="FORWARD",
        result=forward_result,
        expected_frame_count=forward_count,
    )
    backward_frames = _validated_directional_stage2_frames(
        direction="BACKWARD",
        result=backward_result,
        expected_frame_count=backward_count,
    )

    source_video = view.get("source_video")
    if not isinstance(source_video, Mapping):
        raise RuntimeError("Selection view source-video contract is missing.")
    fps = float(source_video.get("fps") or 0.0)
    if fps <= 0:
        raise RuntimeError("Selection view FPS is invalid.")
    shot_id = str(view["selected_shot_id"])
    mapped: dict[int, dict[str, Any]] = {}

    # Map the backward branch first. Forward then replaces the duplicate anchor.
    for raw in backward_frames:
        local = int(raw["frame_index"])
        source_frame = anchor - local
        if not shot_start <= source_frame <= anchor:
            raise RuntimeError("Backward source-frame mapping escaped selected shot.")
        mapped[source_frame] = _source_mapped_directional_frame(
            raw=raw,
            direction="BACKWARD",
            source_frame=source_frame,
            local_frame=local,
            fps=fps,
            shot_id=shot_id,
        )

    for raw in forward_frames:
        local = int(raw["frame_index"])
        source_frame = anchor + local
        if not anchor <= source_frame <= shot_end:
            raise RuntimeError("Forward source-frame mapping escaped selected shot.")
        mapped[source_frame] = _source_mapped_directional_frame(
            raw=raw,
            direction="FORWARD",
            source_frame=source_frame,
            local_frame=local,
            fps=fps,
            shot_id=shot_id,
        )

    anchor_source = "FORWARD_STAGE2"
    if anchor not in mapped and backward_frames:
        raw = backward_frames[0]
        mapped[anchor] = _source_mapped_directional_frame(
            raw=raw,
            direction="BACKWARD",
            source_frame=anchor,
            local_frame=0,
            fps=fps,
            shot_id=shot_id,
        )
        anchor_source = "BACKWARD_STAGE2_FALLBACK"
    if anchor not in mapped:
        if len(initial_bbox) != 4:
            raise RuntimeError("Selected anchor bbox is invalid.")
        mapped[anchor] = {
            "frame_index": anchor,
            "time_ms": int(round(anchor * 1000.0 / fps)),
            "time_seconds": anchor / fps,
            "shot_id": shot_id,
            "state": "INITIALIZING",
            "bbox_xyxy": [float(value) for value in initial_bbox],
            "bbox_source": "USER_SELECTED_IMMUTABLE_ANCHOR",
            "tracking_confidence": 1.0,
            "identity_confidence": 1.0,
            "selected_detection_id": None,
            "decision_reason": "SINGLE_FRAME_SELECTED_SHOT_ANCHOR",
            "review_required": False,
            "runtime_direction": "ANCHOR_ONLY",
            "runtime_local_frame_index": 0,
            "runtime_frame_mapping": "source_frame=anchor_frame",
            "phase3c_source": "IMMUTABLE_USER_SELECTED_ANCHOR",
        }
        anchor_source = "IMMUTABLE_USER_SELECTED_ANCHOR"

    expected_indices = set(range(shot_start, shot_end + 1))
    observed_indices = set(mapped)
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        extra = sorted(observed_indices - expected_indices)
        raise RuntimeError(
            "Bidirectional selected-shot merge is incomplete: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    frames = [mapped[index] for index in range(shot_start, shot_end + 1)]
    state_counts: dict[str, int] = {}
    for row in frames:
        state = str(row.get("state") or "")
        state_counts[state] = state_counts.get(state, 0) + 1
    bbox_count = sum(row.get("bbox_xyxy") is not None for row in frames)
    prefix_rows = [row for row in frames if int(row["frame_index"]) < anchor]
    prefix_bbox_count = sum(
        row.get("bbox_xyxy") is not None for row in prefix_rows
    )

    timeline = {
        "schema_version": PHASE3C_TIMELINE_SCHEMA,
        "generated_at": now_iso(),
        "status": "BIDIRECTIONAL_SELECTED_SHOT_MERGED",
        "policy": PHASE3C_MERGE_POLICY,
        "selected_shot_id": shot_id,
        "video": dict(source_video),
        "selected_shot_start_frame": shot_start,
        "anchor_frame": anchor,
        "selected_shot_end_frame_inclusive": shot_end,
        "frame_count": len(frames),
        "bbox_frame_count": bbox_count,
        "state_counts": dict(sorted(state_counts.items())),
        "segments": _segments_from_source_frames(frames, fps=fps),
        "frames": frames,
        "provenance": {
            "phase3b_report_path": str(phase3b_path.resolve()),
            "phase3b_report_sha256": sha256_file(phase3b_path),
            "forward_stage2_timeline_sha256": (
                (forward_result.get("stage2_timeline") or {}).get("sha256")
                if isinstance(forward_result.get("stage2_timeline"), Mapping)
                else None
            ),
            "backward_stage2_timeline_sha256": (
                (backward_result.get("stage2_timeline") or {}).get("sha256")
                if isinstance(backward_result.get("stage2_timeline"), Mapping)
                else None
            ),
            "anchor_conflict_policy": "FORWARD_ANCHOR_PRIMARY",
            "anchor_source": anchor_source,
            "prefix_interpolation_used": False,
            "observation_copy_used_as_tracking_success": False,
            "synthetic_tracking_used": False,
            "automatic_target_confirmation": False,
        },
    }
    timeline_path = output_dir / "phase3c_selected_shot_timeline.json"
    atomic_json(timeline_path, timeline)
    timeline_sha = sha256_file(timeline_path)

    report = {
        "schema_version": PHASE3C_REPORT_SCHEMA,
        "created_at": now_iso(),
        "status": "PASS",
        "decision": "AUTHORIZE_PHASE3D_SERVICE_INTEGRATION_VALIDATION",
        "policy": PHASE3C_MERGE_POLICY,
        "selected_shot_id": shot_id,
        "selected_shot_start_frame": shot_start,
        "anchor_frame": anchor,
        "selected_shot_end_frame_inclusive": shot_end,
        "expected_selected_shot_frame_count": expected_shot_count,
        "merged_frame_count": len(frames),
        "merged_bbox_frame_count": bbox_count,
        "state_counts": dict(sorted(state_counts.items())),
        "resolved_prefix_frame_count": len(prefix_rows),
        "resolved_prefix_bbox_frame_count": prefix_bbox_count,
        "source_frame_coverage_complete": True,
        "duplicate_source_frame_count": 0,
        "anchor_conflict_policy": "FORWARD_ANCHOR_PRIMARY",
        "anchor_source": anchor_source,
        "merged_timeline_path": str(timeline_path.resolve()),
        "merged_timeline_sha256": timeline_sha,
        "timeline_merge_complete": True,
        "prefix_interpolation_used": False,
        "observation_copy_used_as_tracking_success": False,
        "synthetic_tracking_used": False,
        "automatic_target_confirmation": False,
    }
    report_path = output_dir / "phase3c_bidirectional_timeline_merge_report.json"
    atomic_json(report_path, report)
    report_sha = sha256_file(report_path)

    merge_contract = dict(view.get("bidirectional_merge_contract") or {})
    merge_contract.update(
        {
            "timeline_merge_complete": True,
            "merge_policy": PHASE3C_MERGE_POLICY,
            "phase3c_report_path": str(report_path.resolve()),
            "phase3c_report_sha256": report_sha,
            "phase3c_selected_shot_timeline_path": str(timeline_path.resolve()),
            "phase3c_selected_shot_timeline_sha256": timeline_sha,
            "resolved_prefix_frame_count": len(prefix_rows),
            "resolved_prefix_bbox_frame_count": prefix_bbox_count,
            "prefix_interpolation_used": False,
        }
    )
    view["bidirectional_merge_contract"] = merge_contract
    view["unresolved_prefix_frame_count_before_merge"] = int(
        view.get("unresolved_prefix_frame_count") or 0
    )
    view["unresolved_prefix_frame_count"] = 0
    view["selected_shot_bidirectional_timeline_merged"] = True
    view_path = output_dir / "_selection_anchor_view" / "selection_view.json"
    atomic_json(view_path, view)
    return report


def _overlay_phase3c_selected_shot_timeline(
    *,
    full_timeline_path: Path,
    selected_shot_timeline_path: Path,
    view: Mapping[str, Any],
) -> None:
    full = read_object(full_timeline_path)
    selected = read_object(selected_shot_timeline_path)
    full_frames = full.get("frames")
    selected_frames = selected.get("frames")
    if not isinstance(full_frames, list) or not isinstance(selected_frames, list):
        raise RuntimeError("Phase 3-C overlay timeline has no frame list.")
    by_index = {
        int(row.get("frame_index", -1)): dict(row)
        for row in full_frames
        if isinstance(row, Mapping)
    }
    shot_start = int(view["selected_shot_start_frame"])
    shot_end = int(view["selected_shot_end_frame_inclusive"])
    expected = set(range(shot_start, shot_end + 1))
    observed: set[int] = set()
    for raw in selected_frames:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Phase 3-C selected-shot frame is invalid.")
        index = int(raw.get("frame_index", -1))
        if index not in expected or index in observed:
            raise RuntimeError("Phase 3-C selected-shot frame coverage is invalid.")
        observed.add(index)
        by_index[index] = dict(raw)
    if observed != expected:
        raise RuntimeError("Phase 3-C selected-shot overlay is incomplete.")
    source_frame_count = int((view.get("source_video") or {}).get("frame_count") or 0)
    if set(by_index) != set(range(source_frame_count)):
        raise RuntimeError("Normalized full timeline frame coverage changed during overlay.")
    full["frames"] = [by_index[index] for index in range(source_frame_count)]
    provenance = (
        dict(full.get("provenance") or {})
        if isinstance(full.get("provenance"), Mapping)
        else {}
    )
    provenance["selected_shot_bidirectional_merge"] = {
        "policy": PHASE3C_MERGE_POLICY,
        "timeline_path": str(selected_shot_timeline_path.resolve()),
        "timeline_sha256": sha256_file(selected_shot_timeline_path),
        "selected_shot_start_frame": shot_start,
        "anchor_frame": int(view["source_offset_frame"]),
        "selected_shot_end_frame_inclusive": shot_end,
        "prefix_interpolation_used": False,
        "observation_copy_used_as_tracking_success": False,
        "synthetic_tracking_used": False,
    }
    selection_anchor_view = dict(provenance.get("selection_anchor_view") or {})
    selection_anchor_view["unresolved_prefix_frame_count"] = 0
    selection_anchor_view["timeline_merge_complete"] = True
    provenance["selection_anchor_view"] = selection_anchor_view
    full["provenance"] = provenance
    atomic_json(full_timeline_path, full)


def _shot_map_by_raw(view: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["raw_shot_id"]): dict(row)
        for row in view.get("shot_map") or []
        if isinstance(row, Mapping)
    }


def _reviewed_shots_from_view(
    view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = view.get("reviewed_shots")
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _shot_for_global_frame(view: Mapping[str, Any], frame_index: int) -> str:
    reviewed = _reviewed_shots_from_view(view)
    for row in reviewed:
        if (
            int(row["start_frame"])
            <= frame_index
            <= int(row["end_frame_inclusive"])
        ):
            return str(row["shot_id"])

    # Compatibility fallback for selection views produced before the complete
    # reviewed-shot contract was embedded. New runs never use this path.
    for row in view.get("shot_map") or []:
        if not isinstance(row, Mapping):
            continue
        if (
            int(row["source_start_frame"])
            <= frame_index
            <= int(row["source_end_frame_inclusive"])
        ):
            return str(row["source_shot_id"])
    return str(view.get("selected_shot_id") or "")


def _normalize_source_shots(
    *,
    raw_shots: Sequence[Any],
    view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviewed = _reviewed_shots_from_view(view)
    if not reviewed:
        # Old selection views are retained only for explicit compatibility.
        return [dict(row) for row in raw_shots if isinstance(row, Mapping)]

    mapping = _shot_map_by_raw(view)
    raw_by_id = {
        str(row.get("shot_id") or ""): dict(row)
        for row in raw_shots
        if isinstance(row, Mapping)
    }
    mapping_by_source = {
        str(row["source_shot_id"]): row
        for row in mapping.values()
    }
    offset = int(view["source_offset_frame"])
    result: list[dict[str, Any]] = []
    reserved = {
        "shot_index",
        "shot_id",
        "start_frame",
        "end_frame",
        "end_frame_inclusive",
        "frame_count",
        "cut_in_frame",
        "cut_out_frame",
    }
    for reviewed_row in reviewed:
        row = dict(reviewed_row)
        source_shot_id = str(row["shot_id"])
        map_row = mapping_by_source.get(source_shot_id)
        runtime = (
            raw_by_id.get(str(map_row["raw_shot_id"]))
            if map_row is not None
            else None
        )
        if runtime is not None:
            for key, value in runtime.items():
                if key not in reserved:
                    row[key] = value
            row["runtime_shot_id"] = str(runtime.get("shot_id") or "")
            row["runtime_local_start_frame"] = int(
                runtime.get("start_frame", map_row["local_start_frame"])
            )
            row["runtime_local_end_frame_inclusive"] = int(
                runtime.get(
                    "end_frame_inclusive",
                    runtime.get("end_frame", map_row["local_end_frame_inclusive"]),
                )
            )
            row["processed_start_frame"] = int(
                map_row["processed_source_start_frame"]
            )
        elif int(row["end_frame_inclusive"]) < offset and not row.get("status"):
            row["status"] = "OUTSIDE_SELECTION_ANCHOR_VIEW"
        result.append(row)
    return result


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered or [
            "candidate_id",
            "frame_index",
            "analysis_local_frame_index",
            "detection_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_full_frame(source: Path, frame_index: int, output: Path) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index}: {source}")
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Cannot write full-frame evidence: {output}")


def _write_bbox_crop(
    source: Path, frame_index: int, bbox: Sequence[float], output: Path
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read ACTIVE frame {frame_index}: {source}")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid ACTIVE crop bbox at frame {frame_index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame[y1:y2, x1:x2]):
        raise RuntimeError(f"Cannot write ACTIVE crop: {output}")


def build_initial_memory_revision(
    *,
    root: Path,
    output_dir: Path,
    raw_state: Mapping[str, Any],
    launch: Mapping[str, Any],
    view: Mapping[str, Any],
) -> tuple[Path, str] | None:
    pending = raw_state.get("pending_action")
    if not isinstance(pending, Mapping) or pending.get("type") != "MEMORY_REVIEW":
        return None
    memory = raw_state.get("memory")
    timeline_value = memory.get("source_timeline") if isinstance(memory, Mapping) else None
    if not timeline_value:
        raise RuntimeError("MEMORY_REVIEW is missing the same-shot source timeline.")
    timeline_path = Path(str(timeline_value)).resolve()
    timeline_sha = validate_input_file(timeline_path, None, "same-shot timeline")
    timeline = read_object(timeline_path)
    active = [
        dict(row)
        for row in timeline.get("frames") or []
        if isinstance(row, Mapping)
        and str(row.get("state") or "").upper() in {"ACTIVE", "REACQUIRED"}
        and isinstance(row.get("bbox_xyxy"), list)
        and len(row["bbox_xyxy"]) == 4
    ]
    if not active:
        raise RuntimeError("Same-shot Phase-1 produced no real ACTIVE/REACQUIRED frames.")
    selected_rows = [active[index] for index in sorted({0, len(active) // 2, len(active) - 1})]
    source = Path(str(view["source_video"]["path"])).resolve()
    offset = int(view["source_offset_frame"])
    active_references: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        local_frame = int(row["frame_index"])
        source_frame = local_frame + offset
        crop = output_dir / "initial_target_memory" / "active_crops" / (
            f"active_{index:02d}_frame_{source_frame:06d}.jpg"
        )
        _write_bbox_crop(source, source_frame, row["bbox_xyxy"], crop)
        active_references.append(
            {
                "kind": "SAME_SHOT_ACTIVE_CROP",
                "frame_id": source_frame,
                "runtime_local_frame_id": local_frame,
                "path": str(crop),
                "sha256": sha256_file(crop),
                "scale_class": "same-shot-active",
                "state": str(row["state"]),
                "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
            }
        )

    target_reference_path = Path(str(launch["target_reference_set"]["path"])).resolve()
    target_reference = read_object(target_reference_path)
    native_references: list[dict[str, Any]] = []
    for raw in target_reference.get("references") or []:
        if not isinstance(raw, Mapping):
            continue
        path = Path(str(raw.get("path") or "")).resolve()
        digest = validate_input_file(path, str(raw.get("sha256") or ""), "native reference")
        native_references.append(
            {
                "kind": "IMMUTABLE_REVIEW_BUNDLE_REFERENCE",
                "frame_id": int(raw["frame_id"]),
                "path": str(path),
                "sha256": digest,
                "scale_class": str(raw.get("scale") or "unknown"),
            }
        )
    if not native_references:
        raise RuntimeError("Immutable target reference set is empty.")

    work = raw_state.get("work") if isinstance(raw_state.get("work"), Mapping) else {}
    v2_name = str(work.get("v2_memory_test_name") or "")
    runtime_memory_path = root / "runs" / "target_centric_tracking_v2" / v2_name / "stage3a2_target_memory.json"
    runtime_memory = read_object(runtime_memory_path)
    scoring_inputs: dict[str, dict[str, Any]] = {}
    embeddings = runtime_memory.get("embeddings") if isinstance(runtime_memory.get("embeddings"), Mapping) else {}
    for logical, key in (("target_embeddings", "target_path"), ("negative_embeddings", "negative_path")):
        path = Path(str(embeddings.get(key) or "")).resolve()
        scoring_inputs[logical] = {"path": str(path), "sha256": validate_input_file(path, None, logical)}

    target_selection = read_object(Path(str(launch["target_selection"]["path"])).resolve())
    selection_id = str(target_selection["selection_id"])
    candidate_id = str(target_selection["selected_candidate_id"])
    seed = hashlib.sha256(
        f"{selection_id}:{candidate_id}:{timeline_sha}".encode("utf-8")
    ).hexdigest()[:16]
    revision_id = f"ecmem_initial_{seed}"
    document = {
        "schema_version": "kickclip.initial_target_memory_revision.r1",
        "immutable": True,
        "memory_revision_id": revision_id,
        "selection_id": selection_id,
        "candidate_id": candidate_id,
        "source_shot_id": str(target_selection["shot_id"]),
        "source_tracklet_id": str(target_selection["tracklet_id"]),
        "same_shot_timeline_path": str(timeline_path),
        "same_shot_timeline_sha256": timeline_sha,
        "native_references": native_references,
        "active_references": active_references,
        "references": native_references + active_references,
        "reference_count": len(native_references) + len(active_references),
        "active_or_reacquired_frame_count": len(active),
        "runtime_scoring_inputs": scoring_inputs,
        "runtime_memory_source_path": str(runtime_memory_path),
        "runtime_memory_source_sha256": sha256_file(runtime_memory_path),
        "automatic_target_confirmation": False,
    }
    path = output_dir / "initial_target_memory" / f"{revision_id}.json"
    atomic_json(path, document)
    return path, sha256_file(path)


def _phase4a_select_diverse_active_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_count: int = 5,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: int(row["frame_index"]),
    )
    if not ordered:
        return []
    count = min(maximum_count, len(ordered))
    if count == 1:
        return [ordered[0]]
    positions = {
        int(round(index * (len(ordered) - 1) / (count - 1)))
        for index in range(count)
    }
    return [ordered[index] for index in sorted(positions)]


def _phase4a_write_contact_sheet(
    *,
    references: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    images: list[tuple[np.ndarray, str]] = []
    for reference in references:
        path = Path(str(reference.get("path") or "")).resolve()
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read Phase 4-A memory reference: {path}")
        label = (
            f"frame={int(reference.get('frame_id', -1))} "
            f"source={str(reference.get('kind') or 'UNKNOWN')}"
        )
        images.append((image, label))
    if not images:
        raise RuntimeError("Phase 4-A contact sheet has no references.")

    tile_width = 360
    tile_height = 300
    header_height = 42
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    canvas = np.zeros(
        (rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )
    for index, (image, label) in enumerate(images):
        row_index = index // columns
        column_index = index % columns
        available_height = tile_height - header_height
        source_height, source_width = image.shape[:2]
        scale = min(tile_width / source_width, available_height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        x0 = column_index * tile_width
        y0 = row_index * tile_height
        offset_x = x0 + (tile_width - resized_width) // 2
        offset_y = y0 + header_height + (available_height - resized_height) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        cv2.putText(
            canvas,
            label[:58],
            (x0 + 8, y0 + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Cannot write Phase 4-A contact sheet: {output_path}")


def build_phase4a_initial_memory_from_phase3c(
    *,
    output_dir: Path,
    launch: Mapping[str, Any],
    view: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an immutable initial memory candidate from source-backed evidence.

    This function does not run ReID, rank a cross-shot candidate, approve the
    memory, or confirm a target. It combines the immutable user-selected
    candidate references with a small, diverse set of real Phase 3-C
    ACTIVE/REACQUIRED RF-DETR-associated observations.
    """

    report_path = output_dir / "phase3c_bidirectional_timeline_merge_report.json"
    timeline_path = output_dir / "phase3c_selected_shot_timeline.json"
    report = read_object(report_path)
    if (
        report.get("status") != "PASS"
        or report.get("decision")
        != "AUTHORIZE_PHASE3D_SERVICE_INTEGRATION_VALIDATION"
        or report.get("timeline_merge_complete") is not True
        or report.get("source_frame_coverage_complete") is not True
        or int(report.get("duplicate_source_frame_count") or 0) != 0
        or report.get("synthetic_tracking_used") is not False
        or report.get("automatic_target_confirmation") is not False
    ):
        raise RuntimeError("Phase 3-C did not authorize Phase 4-A memory construction.")
    timeline_sha = validate_input_file(
        timeline_path,
        str(report.get("merged_timeline_sha256") or ""),
        "Phase 3-C selected-shot timeline",
    )
    timeline = read_object(timeline_path)
    if str(timeline.get("selected_shot_id") or "") != str(view.get("selected_shot_id") or ""):
        raise RuntimeError("Phase 3-C selected-shot identity mismatch.")

    active_rows = [
        dict(row)
        for row in timeline.get("frames") or []
        if isinstance(row, Mapping)
        and str(row.get("state") or "").upper() in {"ACTIVE", "REACQUIRED"}
        and isinstance(row.get("bbox_xyxy"), list)
        and len(row["bbox_xyxy"]) == 4
        and str(row.get("bbox_source") or "").upper()
        == "RFDETR_ASSOCIATED_DETECTION"
        and bool(row.get("selected_detection_id"))
    ]
    if not active_rows:
        raise RuntimeError(
            "Phase 4-A requires at least one real ACTIVE/REACQUIRED "
            "RF-DETR-associated selected-shot observation."
        )
    selected_active_rows = _phase4a_select_diverse_active_rows(active_rows)

    target_reference_path = Path(str(launch["target_reference_set"]["path"])).resolve()
    target_reference_sha = validate_input_file(
        target_reference_path,
        str(launch["target_reference_set"].get("sha256") or ""),
        "immutable target reference set",
    )
    target_reference = read_object(target_reference_path)
    native_references: list[dict[str, Any]] = []
    for raw in target_reference.get("references") or []:
        if not isinstance(raw, Mapping):
            continue
        path = Path(str(raw.get("path") or "")).resolve()
        digest = validate_input_file(
            path,
            str(raw.get("sha256") or ""),
            "immutable target reference crop",
        )
        native_references.append(
            {
                "kind": "IMMUTABLE_USER_SELECTED_CANDIDATE_REFERENCE",
                "frame_id": int(raw["frame_id"]),
                "path": str(path),
                "sha256": digest,
                "crop_sha256": digest,
                "scale_class": str(raw.get("scale") or "unknown"),
                "reference_quality": "USER_SELECTED_IDENTITY_PURE_BUNDLE",
                "scoring_eligible": True,
            }
        )
    if len(native_references) < 3:
        raise RuntimeError("Phase 4-A requires at least three immutable target references.")

    source = Path(str(view["source_video"]["path"])).resolve()
    validate_input_file(
        source,
        str(view.get("source_video_sha256") or view["source_video"].get("sha256") or ""),
        "Phase 4-A source video",
    )
    active_references: list[dict[str, Any]] = []
    for index, row in enumerate(selected_active_rows, start=1):
        source_frame = int(row["frame_index"])
        crop = output_dir / "initial_target_memory" / "phase3c_active_crops" / (
            f"phase3c_active_{index:02d}_frame_{source_frame:06d}.jpg"
        )
        _write_bbox_crop(source, source_frame, row["bbox_xyxy"], crop)
        digest = sha256_file(crop)
        active_references.append(
            {
                "kind": "REAL_PHASE3C_ACTIVE_OBSERVATION",
                "frame_id": source_frame,
                "path": str(crop),
                "sha256": digest,
                "crop_sha256": digest,
                "scale_class": "phase3c-active",
                "reference_quality": "SOURCE_BACKED_TRACKING_EVIDENCE_PENDING_MEMORY_REVIEW",
                "state": str(row["state"]).upper(),
                "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
                "selected_detection_id": str(row["selected_detection_id"]),
                "tracking_confidence": float(row.get("tracking_confidence") or 0.0),
                "identity_confidence": float(row.get("identity_confidence") or 0.0),
                "scoring_eligible": False,
            }
        )

    target_selection_path = Path(str(launch["target_selection"]["path"])).resolve()
    target_selection = read_object(target_selection_path)
    selection_id = str(target_selection["selection_id"])
    candidate_id = str(target_selection["selected_candidate_id"])
    seed = hashlib.sha256(
        (
            f"{selection_id}:{candidate_id}:{timeline_sha}:"
            f"{target_reference_sha}:{PHASE4A_POLICY}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    revision_id = f"ecmem_initial_{seed}"
    references = native_references + active_references
    memory_path = output_dir / "initial_target_memory" / f"{revision_id}.json"
    memory_document = {
        "schema_version": PHASE4A_MEMORY_SCHEMA,
        "immutable": True,
        "memory_revision_id": revision_id,
        "selection_id": selection_id,
        "candidate_id": candidate_id,
        "source_shot_id": str(target_selection["shot_id"]),
        "source_tracklet_id": str(target_selection["tracklet_id"]),
        "policy": PHASE4A_POLICY,
        "phase3c_selected_shot_timeline_path": str(timeline_path),
        "phase3c_selected_shot_timeline_sha256": timeline_sha,
        "target_reference_set_path": str(target_reference_path),
        "target_reference_set_sha256": target_reference_sha,
        "native_references": native_references,
        "active_references": active_references,
        "references": references,
        "reference_count": len(references),
        "scoring_reference_count": len(native_references),
        "pending_review_reference_count": len(active_references),
        "active_or_reacquired_frame_count": len(active_rows),
        "scale_banks_used": sorted(
            {
                str(reference.get("scale_class") or "unknown")
                for reference in references
            }
        ),
        "phase3c_active_evidence_is_automatic_identity_confirmation": False,
        "cross_shot_scoring_performed": False,
        "automatic_target_confirmation": False,
    }
    atomic_json(memory_path, memory_document)
    memory_sha = sha256_file(memory_path)

    contact_sheet_path = output_dir / "initial_target_memory" / "phase4a_memory_contact_sheet.jpg"
    _phase4a_write_contact_sheet(
        references=references,
        output_path=contact_sheet_path,
    )
    phase4a_report = {
        "schema_version": PHASE4A_REPORT_SCHEMA,
        "created_at": now_iso(),
        "status": "PASS",
        "decision": "AUTHORIZE_PHASE4A_INITIAL_MEMORY_REVIEW",
        "policy": PHASE4A_POLICY,
        "memory_revision_id": revision_id,
        "memory_revision_path": str(memory_path),
        "memory_revision_sha256": memory_sha,
        "memory_contact_sheet_path": str(contact_sheet_path),
        "memory_contact_sheet_sha256": sha256_file(contact_sheet_path),
        "selected_shot_id": str(view["selected_shot_id"]),
        "phase3c_active_or_reacquired_frame_count": len(active_rows),
        "selected_phase3c_active_reference_count": len(active_references),
        "immutable_native_reference_count": len(native_references),
        "total_reference_count": len(references),
        "cross_shot_scoring_performed": False,
        "synthetic_tracking_used": False,
        "observation_copy_used_as_tracking_success": False,
        "automatic_target_confirmation": False,
    }
    phase4a_report_path = output_dir / "phase4a_initial_target_memory_report.json"
    atomic_json(phase4a_report_path, phase4a_report)
    phase4a_report["report_path"] = str(phase4a_report_path)
    phase4a_report["report_sha256"] = sha256_file(phase4a_report_path)
    return phase4a_report


def _activate_phase4a_memory_review_state(
    *,
    state: dict[str, Any],
    output_dir: Path,
    phase4a: Mapping[str, Any],
    view: Mapping[str, Any],
) -> None:
    state["status"] = "NEEDS_CONFIRMATION"
    state["decision"] = "PAUSE_FOR_PHASE4A_INITIAL_TARGET_MEMORY_REVIEW"
    state["pending_action"] = {
        "type": "MEMORY_REVIEW",
        "review_stage": "MEMORY",
        "memory_revision_id": str(phase4a["memory_revision_id"]),
        "contact_sheet": str(phase4a["memory_contact_sheet_path"]),
        "preview": str(output_dir / "full_frame_tracking_preview.mp4"),
        "automatic_target_confirmation": False,
    }
    selected_shot_id = str(view.get("selected_shot_id") or "")
    for shot in state.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        if str(shot.get("shot_id") or "") == selected_shot_id:
            shot["status"] = "INITIAL_TARGET_MEMORY_REVIEW_REQUIRED"
        elif str(shot.get("status") or "").upper() == "SEARCHING_NO_MEMORY":
            shot["status"] = "SEARCHING_MEMORY_REVIEW_PENDING"
    runtime = dict(state.get("runtime") or {})
    runtime.update(
        {
            "phase4a_initial_memory_build_complete": True,
            "phase4a_initial_memory_review_status": "PENDING",
            "phase4a_initial_memory_report_path": str(phase4a["report_path"]),
            "phase4a_initial_memory_report_sha256": str(phase4a["report_sha256"]),
            "memory_revision_path": str(phase4a["memory_revision_path"]),
            "memory_revision_sha256": str(phase4a["memory_revision_sha256"]),
            "reference_count": int(phase4a["total_reference_count"]),
            "backend_memory_used_by_provided_e2e_scoring": False,
            "cross_shot_scoring_performed": False,
            "automatic_target_confirmation": False,
        }
    )
    state["runtime"] = runtime


def _handle_phase4a_memory_review_resume(
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> int | str | None:
    if not args.resume or not (args.approve_review or args.reject_review):
        return None
    state_path = output_dir / "pipeline_state.json"
    if not state_path.is_file():
        return None
    state = read_object(state_path)
    pending = state.get("pending_action")
    runtime = state.get("runtime")
    if (
        not isinstance(pending, Mapping)
        or pending.get("type") != "MEMORY_REVIEW"
        or not isinstance(runtime, Mapping)
        or runtime.get("phase4a_initial_memory_build_complete") is not True
    ):
        return None
    memory_path = Path(str(runtime.get("memory_revision_path") or "")).resolve()
    memory_sha = str(runtime.get("memory_revision_sha256") or "")
    validate_input_file(memory_path, memory_sha, "Phase 4-A memory revision")
    if args.target_memory_revision is not None:
        supplied = args.target_memory_revision.resolve()
        supplied_sha = validate_input_file(
            supplied,
            args.target_memory_sha256,
            "backend target memory revision",
        )
        if supplied != memory_path or supplied_sha != memory_sha:
            raise RuntimeError("Phase 4-A reviewed memory does not match backend memory.")

    runtime = dict(runtime)
    state["pending_action"] = None
    if args.approve_review == "MEMORY":
        runtime["phase4a_initial_memory_review_status"] = "PASS"
        runtime["phase4a_initial_memory_reviewed_at"] = now_iso()
        runtime["phase4b_cross_shot_scoring_authorized"] = True
        state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
        state["decision"] = "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING"
        for shot in state.get("shots") or []:
            if isinstance(shot, dict) and str(shot.get("status") or "") == "SEARCHING_MEMORY_REVIEW_PENDING":
                shot["status"] = "SEARCHING_MEMORY_READY"
    elif args.reject_review == "MEMORY":
        runtime["phase4a_initial_memory_review_status"] = "FAIL"
        runtime["phase4b_cross_shot_scoring_authorized"] = False
        state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
        state["decision"] = "BLOCK_PHASE4A_INITIAL_MEMORY_REVIEW_FAIL"
    else:
        return None
    runtime["backend_memory_used_by_provided_e2e_scoring"] = False
    runtime["cross_shot_scoring_performed"] = False
    runtime["automatic_target_confirmation"] = False
    state["runtime"] = runtime
    state["updated_at"] = now_iso()
    atomic_json(state_path, state)
    summary_path = output_dir / "pipeline_summary.json"
    summary = read_object(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "status": state["status"],
            "decision": state["decision"],
            "phase4a_initial_memory_review_status": runtime[
                "phase4a_initial_memory_review_status"
            ],
            "phase4b_cross_shot_scoring_authorized": bool(
                runtime.get("phase4b_cross_shot_scoring_authorized")
            ),
            "cross_shot_scoring_performed": False,
        }
    )
    atomic_json(summary_path, summary)
    if args.approve_review == "MEMORY":
        return PHASE4B_APPROVED_CONTINUE
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def _phase4b_run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    accepted_return_codes: set[int] | frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUTF8", "1")
    # Required for the already-verified legacy checkpoints on recent PyTorch.
    environment.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        shell=False,
        check=False,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode not in accepted_return_codes:
        raise RuntimeError(
            f"Phase 4-B command failed ({completed.returncode}): "
            + subprocess.list2cmdline(list(command))
        )
    return completed


def _phase4b_ready_shots(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = {
        "SEARCHING_MEMORY_READY",
        "SEARCHING_NO_MEMORY",
        "SAFE_REJECTED_SEARCHING",
    }
    rows = [
        dict(row)
        for row in state.get("shots") or []
        if isinstance(row, Mapping)
        and str(row.get("status") or "").upper() in allowed
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("shot_index") or 0),
            int(row.get("start_frame") or 0),
            str(row.get("shot_id") or ""),
        )
    )
    return rows


def _phase4b_first_ready_shot(state: Mapping[str, Any]) -> dict[str, Any]:
    shots = _phase4b_ready_shots(state)
    if not shots:
        raise RuntimeError("No reviewed shot is ready for Phase 4-B scoring.")
    return shots[0]


def _phase4b_migrate_prior_no_candidate_results(state: dict[str, Any]) -> bool:
    """Convert earlier terminal no-candidate results into resumable shot exhaustion.

    R3 treated one no-candidate shot as a terminal job safe block. R4 preserves that
    evidence, marks only that shot exhausted, and continues with the next reviewed
    shot using the same approved target-memory revision.
    """

    changed = False
    search_results = state.get("shot_search_results")
    if not isinstance(search_results, Mapping):
        search_results = {}
    exhausted_ids: list[str] = []
    for shot_id, payload in search_results.items():
        if not isinstance(payload, Mapping):
            continue
        candidate_count = int(payload.get("candidate_count") or 0)
        operational_state = str(payload.get("status") or "").upper()
        if candidate_count != 0 or operational_state not in {
            "SAFE_REJECTED_SEARCHING",
            "SAFE_REJECTED_NO_CANDIDATES",
        }:
            continue
        exhausted_ids.append(str(shot_id))

    for shot in state.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("shot_id") or "")
        status = str(shot.get("status") or "").upper()
        if shot_id in exhausted_ids and status in {
            "SAFE_REJECTED_SEARCHING",
            "SEARCHING_MEMORY_READY",
            "SEARCHING_NO_MEMORY",
        }:
            shot["status"] = PHASE4B_SEARCH_EXHAUSTED_STATUS
            shot["phase4b_exhausted_without_reviewable_candidate"] = True
            changed = True

    if changed and state.get("pending_action") is None:
        remaining = _phase4b_ready_shots(state)
        if remaining:
            state["status"] = "READY_FOR_PHASE4B_CONTINUATION"
            state["decision"] = PHASE4B_CONTINUE_DECISION
        else:
            state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
            state["decision"] = PHASE4B_ALL_EXHAUSTED_DECISION
        runtime = dict(state.get("runtime") or {})
        runtime.update(
            {
                "phase4b_policy": PHASE4B_POLICY,
                "phase4b_multi_shot_continuation_enabled": True,
                "phase4b_migrated_exhausted_shot_ids": sorted(exhausted_ids),
                "automatic_target_confirmation": False,
            }
        )
        state["runtime"] = runtime
        state["updated_at"] = now_iso()
    return changed


def _phase4b_scoring_references(
    memory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = [
        dict(row)
        for row in memory.get("references") or []
        if isinstance(row, Mapping)
    ]
    scoring = [row for row in all_rows if row.get("scoring_eligible") is True]
    if len(scoring) < 3:
        raise RuntimeError("Phase 4-B requires at least three approved scoring references.")
    for row in scoring:
        path = Path(str(row.get("path") or "")).resolve()
        expected = str(row.get("sha256") or row.get("crop_sha256") or "")
        validate_input_file(path, expected, "Phase 4-B scoring reference")
    return scoring, all_rows


def _phase4b_gate_candidate_rows(
    ranked: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    negative_memory_available: bool,
) -> dict[str, Any]:
    if not ranked:
        return {
            "operational_state": "SAFE_REJECTED_SEARCHING",
            "decision": "RETAIN_SEARCHING_NO_CANDIDATES",
            "automatically_proposed_candidate": None,
            "all_auto_gates_passed": False,
            "plausible_candidate_exists": False,
            "negative_memory_available": negative_memory_available,
            "gates": {},
        }
    top1 = dict(ranked[0])
    top2 = dict(ranked[1]) if len(ranked) > 1 else None
    top1_score = float(top1["retrieval_score"])
    top1_proto = float(top1["prototype_target_similarity"])
    top2_score = float(top2["retrieval_score"]) if top2 else top1_score
    gap = top1_score - top2_score if top2 else 0.0
    gates = {
        "retrieval_score": {
            "value": top1_score,
            "required": f">={float(policy['minimum_retrieval_score'])}",
            "passed": top1_score >= float(policy["minimum_retrieval_score"]),
        },
        "prototype_similarity": {
            "value": top1_proto,
            "required": f">={float(policy['minimum_prototype_similarity'])}",
            "passed": top1_proto >= float(policy["minimum_prototype_similarity"]),
        },
        "top1_top2_score_gap": {
            "value": gap,
            "required": f">={float(policy['minimum_top1_top2_gap'])}",
            "passed": bool(top2) and gap >= float(policy["minimum_top1_top2_gap"]),
        },
        "negative_memory_available": {
            "value": negative_memory_available,
            "required": True,
            "passed": negative_memory_available,
        },
    }
    plausible = (
        top1_score >= float(policy["minimum_plausible_score"])
        and top1_proto >= float(policy["minimum_plausible_prototype"])
    )
    # Phase 4-B is assisted-only. Even a gate-complete top candidate is never
    # silently linked. Missing negative memory additionally forbids auto-safe.
    all_pass = all(bool(row["passed"]) for row in gates.values())
    return {
        "operational_state": "AMBIGUOUS" if plausible else "LOW_CONFIDENCE_REVIEWABLE",
        "decision": "AUTHORIZE_USER_CONFIRMATION_FALLBACK",
        "automatically_proposed_candidate": None,
        "all_auto_gates_passed": all_pass,
        "plausible_candidate_exists": plausible,
        "negative_memory_available": negative_memory_available,
        "top_candidate": top1,
        "second_candidate": top2,
        "gates": gates,
        "automatic_target_confirmation": False,
    }



def _phase4b_filter_candidate_detections(
    by_frame: Mapping[int, Sequence[Any]],
    *,
    end_frame_inclusive: int | None = None,
) -> tuple[dict[int, list[Any]], dict[int, list[Any]], dict[str, Any]]:
    """Split detector rows into candidate roles and same-shot role negatives.

    Stage 1 records all frozen RF-DETR classes. Phase 4-B candidate tracklets
    may use only player/goalkeeper rows, while referee/staff rows are retained
    as explicit negative evidence for assisted cross-shot review.
    """

    filtered: dict[int, list[Any]] = {}
    negative_roles: dict[int, list[Any]] = {}
    raw_detection_count = 0
    eligible_detection_count = 0
    negative_role_detection_count = 0
    excluded_role_counts: dict[str, int] = {}
    allowed_role_counts: dict[str, int] = {}
    negative_role_counts: dict[str, int] = {}

    for frame_index in sorted(by_frame):
        if end_frame_inclusive is not None and int(frame_index) > int(end_frame_inclusive):
            continue
        kept: list[Any] = []
        negatives: list[Any] = []
        for detection in by_frame[frame_index]:
            raw_detection_count += 1
            class_id = int(detection.class_id)
            class_name = str(detection.class_name or "").strip().lower()
            expected_name = PHASE4B_CLASS_NAMES.get(class_id)
            if expected_name is None or class_name != expected_name:
                raise RuntimeError(
                    "Phase 4-B detector class mapping mismatch: "
                    f"class_id={class_id}, class_name={class_name!r}, "
                    f"expected={expected_name!r}"
                )
            if (
                class_id in PHASE4B_ALLOWED_CANDIDATE_CLASS_IDS
                and class_name in PHASE4B_ALLOWED_CANDIDATE_CLASS_NAMES
            ):
                kept.append(detection)
                eligible_detection_count += 1
                allowed_role_counts[class_name] = allowed_role_counts.get(class_name, 0) + 1
            elif (
                class_id in PHASE4B_NEGATIVE_ROLE_CLASS_IDS
                and class_name in PHASE4B_NEGATIVE_ROLE_CLASS_NAMES
            ):
                negatives.append(detection)
                negative_role_detection_count += 1
                negative_role_counts[class_name] = negative_role_counts.get(class_name, 0) + 1
                excluded_role_counts[class_name] = excluded_role_counts.get(class_name, 0) + 1
            else:
                excluded_role_counts[class_name] = excluded_role_counts.get(class_name, 0) + 1
        if kept:
            filtered[int(frame_index)] = kept
        if negatives:
            negative_roles[int(frame_index)] = negatives

    return filtered, negative_roles, {
        "policy_version": PHASE4B_CANDIDATE_ROLE_FILTER_POLICY,
        "negative_role_memory_policy": PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
        "allowed_class_ids": sorted(PHASE4B_ALLOWED_CANDIDATE_CLASS_IDS),
        "allowed_class_names": sorted(PHASE4B_ALLOWED_CANDIDATE_CLASS_NAMES),
        "negative_role_class_ids": sorted(PHASE4B_NEGATIVE_ROLE_CLASS_IDS),
        "negative_role_class_names": sorted(PHASE4B_NEGATIVE_ROLE_CLASS_NAMES),
        "reviewed_shot_local_end_frame_inclusive": end_frame_inclusive,
        "raw_detection_count": raw_detection_count,
        "eligible_candidate_detection_count": eligible_detection_count,
        "negative_role_detection_count": negative_role_detection_count,
        "excluded_non_target_role_count": raw_detection_count - eligible_detection_count,
        "allowed_role_counts": dict(sorted(allowed_role_counts.items())),
        "negative_role_counts": dict(sorted(negative_role_counts.items())),
        "excluded_role_counts": dict(sorted(excluded_role_counts.items())),
        "class_mapping_validated": True,
        "analysis_tail_excluded_from_role_counts": end_frame_inclusive is not None,
    }


def _phase4b_bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0



def _phase4b_bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _phase4b_bbox_intersection_area(
    left: Sequence[float], right: Sequence[float]
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _phase4b_bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


def _phase4b_point_inside_bbox(
    point: tuple[float, float], bbox: Sequence[float]
) -> bool:
    return (
        float(bbox[0]) <= point[0] <= float(bbox[2])
        and float(bbox[1]) <= point[1] <= float(bbox[3])
    )


def _phase4b_identity_observability(
    candidate_detections: Sequence[Any],
    all_person_by_frame: Mapping[int, Sequence[Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    """Measure whether a tracklet contains identity-usable single-person crops.

    This gate is deliberately appearance-agnostic. It does not inspect team color,
    jersey number, celebration semantics, or a specific shot. It only asks whether
    enough observations show one sufficiently large, geometrically stable person
    without strong contamination from other detected people.
    """

    ordered = sorted(
        candidate_detections,
        key=lambda detection: (
            int(detection.frame),
            str(detection.detection_id),
        ),
    )
    observation_count = len(ordered)
    minimum_clean_frames = min(
        PHASE4B_OBSERVABILITY_MAXIMUM_REQUIRED_CLEAN_FRAMES,
        max(
            PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_FRAMES,
            int(math.ceil(observation_count * PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_RATIO)),
        ),
    )
    if not ordered:
        return {
            "policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "passed": False,
            "classification": "UNREVIEWABLE_LOW_IDENTITY_OBSERVABILITY",
            "observation_count": 0,
            "clean_frame_count": 0,
            "clean_frame_ratio": 0.0,
            "minimum_clean_frame_count": minimum_clean_frames,
            "clean_detection_ids": [],
            "per_observation": [],
            "rejection_reasons": ["EMPTY_TRACKLET"],
        }

    unstable_detection_ids: set[str] = set()
    for previous, current in zip(ordered, ordered[1:]):
        previous_bbox = [float(value) for value in previous.bbox]
        current_bbox = [float(value) for value in current.bbox]
        previous_area = max(_phase4b_bbox_area(previous_bbox), 1.0)
        current_area = max(_phase4b_bbox_area(current_bbox), 1.0)
        area_jump = max(previous_area / current_area, current_area / previous_area)
        previous_width = max(1.0, previous_bbox[2] - previous_bbox[0])
        previous_height = max(1.0, previous_bbox[3] - previous_bbox[1])
        current_width = max(1.0, current_bbox[2] - current_bbox[0])
        current_height = max(1.0, current_bbox[3] - current_bbox[1])
        previous_aspect = previous_width / previous_height
        current_aspect = current_width / current_height
        aspect_jump = max(
            previous_aspect / max(current_aspect, 1e-6),
            current_aspect / max(previous_aspect, 1e-6),
        )
        previous_center = _phase4b_bbox_center(previous_bbox)
        current_center = _phase4b_bbox_center(current_bbox)
        center_distance = math.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        center_jump_normalized = center_distance / max(
            math.sqrt(previous_area), math.sqrt(current_area), 1.0
        )
        if (
            area_jump > PHASE4B_OBSERVABILITY_MAX_AREA_JUMP_RATIO
            or aspect_jump > PHASE4B_OBSERVABILITY_MAX_ASPECT_JUMP_RATIO
            or center_jump_normalized > PHASE4B_OBSERVABILITY_MAX_CENTER_JUMP_NORMALIZED
        ):
            unstable_detection_ids.add(str(current.detection_id))

    edge_margin_x = max(1.0, frame_width * PHASE4B_OBSERVABILITY_EDGE_MARGIN_RATIO)
    edge_margin_y = max(1.0, frame_height * PHASE4B_OBSERVABILITY_EDGE_MARGIN_RATIO)
    minimum_height = max(
        PHASE4B_OBSERVABILITY_MIN_HEIGHT_PIXELS,
        frame_height * PHASE4B_OBSERVABILITY_MIN_HEIGHT_RATIO,
    )
    per_observation: list[dict[str, Any]] = []
    clean_detection_ids: list[str] = []
    crowded_count = 0
    multi_person_count = 0
    truncated_count = 0
    unstable_count = 0
    horizontal_count = 0
    too_small_count = 0
    maximum_overlap_ratio = 0.0

    for detection in ordered:
        detection_id = str(detection.detection_id)
        frame = int(detection.frame)
        bbox = [float(value) for value in detection.bbox]
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        area = max(_phase4b_bbox_area(bbox), 1.0)
        aspect_ratio = width / max(height, 1.0)
        candidate_center = _phase4b_bbox_center(bbox)

        edge_contacts = sum(
            (
                bbox[0] <= edge_margin_x,
                bbox[1] <= edge_margin_y,
                bbox[2] >= frame_width - edge_margin_x,
                bbox[3] >= frame_height - edge_margin_y,
            )
        )
        severe_truncation = edge_contacts >= 2
        horizontal_or_partial = aspect_ratio > PHASE4B_OBSERVABILITY_MAX_WIDTH_HEIGHT_RATIO
        too_small = height < minimum_height
        low_confidence = float(detection.confidence) < PHASE4B_OBSERVABILITY_MIN_DETECTION_CONFIDENCE
        unstable_geometry = detection_id in unstable_detection_ids

        max_overlap_ratio = 0.0
        max_iou = 0.0
        distinct_overlap_count = 0
        other_center_inside_count = 0
        candidate_center_inside_other_count = 0
        duplicate_like_detection_count = 0
        for other in all_person_by_frame.get(frame, ()):
            other_id = str(other.detection_id)
            if other_id == detection_id:
                continue
            other_bbox = [float(value) for value in other.bbox]
            iou = _phase4b_bbox_iou(bbox, other_bbox)
            if iou >= PHASE4B_OBSERVABILITY_DUPLICATE_DETECTION_IOU:
                duplicate_like_detection_count += 1
                continue
            intersection = _phase4b_bbox_intersection_area(bbox, other_bbox)
            overlap_ratio = intersection / area if area > 0.0 else 0.0
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)
            max_iou = max(max_iou, iou)
            other_center = _phase4b_bbox_center(other_bbox)
            other_center_inside = _phase4b_point_inside_bbox(other_center, bbox)
            candidate_center_inside = _phase4b_point_inside_bbox(
                candidate_center, other_bbox
            )
            if overlap_ratio >= 0.10 or iou >= 0.10:
                distinct_overlap_count += 1
            if other_center_inside and overlap_ratio >= 0.10:
                other_center_inside_count += 1
            if candidate_center_inside and overlap_ratio >= 0.20:
                candidate_center_inside_other_count += 1

        maximum_overlap_ratio = max(maximum_overlap_ratio, max_overlap_ratio)
        crowded = (
            max_overlap_ratio > PHASE4B_OBSERVABILITY_MAX_CANDIDATE_OVERLAP_RATIO
            or distinct_overlap_count >= 2
            or other_center_inside_count >= 1
            or candidate_center_inside_other_count >= 1
        )
        multi_person_contaminated = (
            distinct_overlap_count >= 2
            or other_center_inside_count >= 1
            or candidate_center_inside_other_count >= 1
        )
        reasons: list[str] = []
        if crowded:
            reasons.append("CROWDED_PERSON_OVERLAP")
        if multi_person_contaminated:
            reasons.append("MULTI_PERSON_CONTAMINATED_CROP")
        if severe_truncation:
            reasons.append("SEVERE_FRAME_EDGE_TRUNCATION")
        if horizontal_or_partial:
            reasons.append("NON_UPRIGHT_OR_PARTIAL_BODY_GEOMETRY")
        if too_small:
            reasons.append("INSUFFICIENT_PERSON_PIXEL_HEIGHT")
        if low_confidence:
            reasons.append("LOW_DETECTOR_CONFIDENCE")
        if unstable_geometry:
            reasons.append("IDENTITY_TRACKLET_UNSTABLE")
        clean = not reasons
        if clean:
            clean_detection_ids.append(detection_id)
        crowded_count += int(crowded)
        multi_person_count += int(multi_person_contaminated)
        truncated_count += int(severe_truncation)
        unstable_count += int(unstable_geometry)
        horizontal_count += int(horizontal_or_partial)
        too_small_count += int(too_small)
        per_observation.append(
            {
                "detection_id": detection_id,
                "frame": frame,
                "confidence": float(detection.confidence),
                "bbox_xyxy": bbox,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_width_height_ratio": aspect_ratio,
                "edge_contact_count": edge_contacts,
                "severe_frame_edge_truncation": severe_truncation,
                "maximum_other_person_iou": max_iou,
                "maximum_candidate_overlap_ratio": max_overlap_ratio,
                "distinct_overlapping_person_count": distinct_overlap_count,
                "other_person_center_inside_count": other_center_inside_count,
                "candidate_center_inside_other_count": candidate_center_inside_other_count,
                "duplicate_like_detection_count": duplicate_like_detection_count,
                "crowded": crowded,
                "multi_person_contaminated": multi_person_contaminated,
                "unstable_geometry": unstable_geometry,
                "clean_for_reid": clean,
                "rejection_reasons": reasons,
            }
        )

    clean_count = len(clean_detection_ids)
    clean_ratio = clean_count / observation_count
    crowded_ratio = crowded_count / observation_count
    multi_person_ratio = multi_person_count / observation_count
    truncated_ratio = truncated_count / observation_count
    unstable_ratio = unstable_count / observation_count
    horizontal_ratio = horizontal_count / observation_count
    too_small_ratio = too_small_count / observation_count
    passed = (
        clean_count >= minimum_clean_frames
        and clean_ratio >= PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_RATIO
        and crowded_ratio <= PHASE4B_OBSERVABILITY_MAX_CROWDED_RATIO
        and unstable_ratio <= PHASE4B_OBSERVABILITY_MAX_UNSTABLE_RATIO
    )
    rejection_reasons: list[str] = []
    if clean_count < minimum_clean_frames:
        rejection_reasons.append("INSUFFICIENT_CLEAN_IDENTITY_FRAMES")
    if clean_ratio < PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_RATIO:
        rejection_reasons.append("INSUFFICIENT_CLEAN_IDENTITY_RATIO")
    if crowded_ratio > PHASE4B_OBSERVABILITY_MAX_CROWDED_RATIO:
        rejection_reasons.append("EXCESSIVE_GROUP_OCCLUSION")
    if multi_person_ratio >= 0.50:
        rejection_reasons.append("MULTI_PERSON_CONTAMINATION_DOMINANT")
    if unstable_ratio > PHASE4B_OBSERVABILITY_MAX_UNSTABLE_RATIO:
        rejection_reasons.append("TRACKLET_GEOMETRY_UNSTABLE")
    if horizontal_ratio >= 0.50:
        rejection_reasons.append("NON_UPRIGHT_OR_PARTIAL_BODY_DOMINANT")
    if too_small_ratio >= 0.80:
        rejection_reasons.append("PERSON_TOO_SMALL_FOR_IDENTITY")

    group_occlusion = (
        crowded_ratio >= 0.50
        or multi_person_ratio >= 0.50
        or horizontal_ratio >= 0.50
    )
    return {
        "policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
        "passed": passed,
        "classification": (
            "REVIEWABLE_SINGLE_PERSON_IDENTITY"
            if passed
            else (
                "UNREVIEWABLE_GROUP_OCCLUSION"
                if group_occlusion
                else "UNREVIEWABLE_LOW_IDENTITY_OBSERVABILITY"
            )
        ),
        "observation_count": observation_count,
        "clean_frame_count": clean_count,
        "clean_frame_ratio": clean_ratio,
        "minimum_clean_frame_count": minimum_clean_frames,
        "minimum_clean_frame_ratio": PHASE4B_OBSERVABILITY_MINIMUM_CLEAN_RATIO,
        "crowded_frame_count": crowded_count,
        "crowded_frame_ratio": crowded_ratio,
        "multi_person_contaminated_frame_count": multi_person_count,
        "multi_person_contamination_ratio": multi_person_ratio,
        "bbox_truncated_frame_count": truncated_count,
        "bbox_truncated_frame_ratio": truncated_ratio,
        "unstable_geometry_frame_count": unstable_count,
        "unstable_geometry_frame_ratio": unstable_ratio,
        "horizontal_or_partial_frame_count": horizontal_count,
        "horizontal_or_partial_frame_ratio": horizontal_ratio,
        "too_small_frame_count": too_small_count,
        "too_small_frame_ratio": too_small_ratio,
        "maximum_person_overlap_ratio": maximum_overlap_ratio,
        "clean_detection_ids": clean_detection_ids,
        "per_observation": per_observation,
        "rejection_reasons": rejection_reasons,
        "automatic_target_confirmation": False,
    }


def _phase4b_evenly_sample_detections(
    detections: Sequence[Any],
    maximum_count: int,
) -> list[Any]:
    ordered = sorted(
        detections,
        key=lambda detection: (
            int(detection.frame),
            str(detection.detection_id),
        ),
    )
    if maximum_count < 1:
        raise ValueError("maximum_count must be positive")
    if len(ordered) <= maximum_count:
        return ordered
    if maximum_count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        int(round(position * (len(ordered) - 1) / (maximum_count - 1)))
        for position in range(maximum_count)
    }
    return [ordered[index] for index in sorted(indices)]


def _phase4b_cosine(left: Any, right: Any, np: Any) -> float:
    left_value = np.asarray(left, dtype=np.float32).reshape(-1)
    right_value = np.asarray(right, dtype=np.float32).reshape(-1)
    left_norm = float(np.linalg.norm(left_value))
    right_norm = float(np.linalg.norm(right_value))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return -1.0
    return float(np.dot(left_value, right_value) / (left_norm * right_norm))


def _phase4b_target_context_descriptor(
    image: Any,
    *,
    cv2: Any,
    np: Any,
) -> Any:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[0] < 4 or value.shape[1] < 4:
        raise ValueError("Target-context descriptor requires a non-empty BGR image.")
    height, width = int(value.shape[0]), int(value.shape[1])
    x1 = max(0, min(width - 1, int(round(width * 0.18))))
    x2 = max(x1 + 1, min(width, int(round(width * 0.82))))
    y1 = max(0, min(height - 1, int(round(height * 0.08))))
    y2 = max(y1 + 1, min(height, int(round(height * 0.64))))
    torso = value[y1:y2, x1:x2]
    if torso.size == 0:
        torso = value
    torso = cv2.resize(torso, (64, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    hsv_hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [
            PHASE4B_TARGET_CONTEXT_H_BINS,
            PHASE4B_TARGET_CONTEXT_S_BINS,
            PHASE4B_TARGET_CONTEXT_V_BINS,
        ],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32).reshape(-1)
    gray_hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [PHASE4B_TARGET_CONTEXT_GRAY_BINS],
        [0, 256],
    ).astype(np.float32).reshape(-1)
    hsv_norm = float(np.linalg.norm(hsv_hist))
    gray_norm = float(np.linalg.norm(gray_hist))
    if hsv_norm > 0.0:
        hsv_hist /= hsv_norm
    if gray_norm > 0.0:
        gray_hist /= gray_norm
    descriptor = np.concatenate(
        [0.78 * hsv_hist, 0.22 * gray_hist],
        axis=0,
    ).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 0.0:
        raise ValueError("Target-context descriptor has zero norm.")
    return descriptor / norm


def _phase4b_aggregate_target_context_descriptor(
    images: Sequence[Any],
    *,
    cv2: Any,
    np: Any,
) -> Any:
    descriptors = [
        _phase4b_target_context_descriptor(image, cv2=cv2, np=np)
        for image in images
    ]
    if not descriptors:
        raise ValueError("At least one approved target reference is required.")
    matrix = np.stack(descriptors).astype(np.float32)
    aggregate = np.median(matrix, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(aggregate))
    if norm <= 0.0:
        raise ValueError("Aggregated target-context descriptor has zero norm.")
    return aggregate / norm


def _phase4b_target_context_similarity(
    candidate_descriptor: Any,
    target_descriptor: Any,
    *,
    np: Any,
) -> float:
    cosine = _phase4b_cosine(candidate_descriptor, target_descriptor, np)
    return float(max(0.0, min(1.0, cosine)))


def _phase4b_pairwise_cosine_median(matrix: Any, np: Any) -> float:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 2:
        return 1.0
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = value / norms
    similarities = normalized @ normalized.T
    upper = similarities[np.triu_indices(value.shape[0], 1)]
    return float(np.median(upper)) if upper.size else 1.0


def _phase4b_identity_purity_geometry_jump(
    left: Any,
    right: Any,
) -> float:
    left_bbox = [float(value) for value in left.bbox]
    right_bbox = [float(value) for value in right.bbox]
    left_center = _phase4b_bbox_center(left_bbox)
    right_center = _phase4b_bbox_center(right_bbox)
    left_height = max(1.0, left_bbox[3] - left_bbox[1])
    right_height = max(1.0, right_bbox[3] - right_bbox[1])
    center_distance = math.hypot(
        right_center[0] - left_center[0],
        right_center[1] - left_center[1],
    ) / max(left_height, right_height, 1.0)
    height_ratio = max(
        left_height / right_height,
        right_height / left_height,
    )
    height_component = min(2.0, abs(math.log(max(height_ratio, 1e-6))))
    return float(center_distance + 0.35 * height_component)


def _phase4b_identity_purity_change_points(
    probes: Sequence[Any],
    embeddings_by_detection_id: Mapping[str, Any],
    *,
    fps: float,
    np: Any,
) -> list[dict[str, Any]]:
    ordered = [
        detection
        for detection in sorted(
            probes,
            key=lambda detection: (
                int(detection.frame),
                str(detection.detection_id),
            ),
        )
        if str(detection.detection_id) in embeddings_by_detection_id
    ]
    if len(ordered) < 2:
        return []

    maximum_gap_frames = max(
        2,
        int(round(max(fps, 1.0) * PHASE4B_IDENTITY_PURITY_MAX_GAP_SECONDS)),
    )
    window = PHASE4B_IDENTITY_PURITY_LOCAL_WINDOW
    change_points: list[dict[str, Any]] = []
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        previous_embedding = embeddings_by_detection_id[str(previous.detection_id)]
        current_embedding = embeddings_by_detection_id[str(current.detection_id)]
        adjacent_cosine = _phase4b_cosine(previous_embedding, current_embedding, np)
        frame_gap = int(current.frame) - int(previous.frame)
        geometry_jump = _phase4b_identity_purity_geometry_jump(previous, current)

        left = ordered[max(0, index - window) : index]
        right = ordered[index : min(len(ordered), index + window)]
        left_matrix = np.stack(
            [embeddings_by_detection_id[str(row.detection_id)] for row in left]
        ).astype(np.float32)
        right_matrix = np.stack(
            [embeddings_by_detection_id[str(row.detection_id)] for row in right]
        ).astype(np.float32)
        left_coherence = _phase4b_pairwise_cosine_median(left_matrix, np)
        right_coherence = _phase4b_pairwise_cosine_median(right_matrix, np)
        left_norm = left_matrix / np.maximum(
            np.linalg.norm(left_matrix, axis=1, keepdims=True), 1e-12
        )
        right_norm = right_matrix / np.maximum(
            np.linalg.norm(right_matrix, axis=1, keepdims=True), 1e-12
        )
        cross_cosine = float(np.median(left_norm @ right_norm.T))
        local_coherence = min(left_coherence, right_coherence)
        contrast = local_coherence - cross_cosine

        reasons: list[str] = []
        if frame_gap > maximum_gap_frames:
            reasons.append("TEMPORAL_GAP")
        enough_context = len(left) >= 2 and len(right) >= 2
        if (
            enough_context
            and local_coherence >= PHASE4B_IDENTITY_PURITY_MIN_LOCAL_COHERENCE
            and cross_cosine <= PHASE4B_IDENTITY_PURITY_MAX_CROSS_COSINE
            and adjacent_cosine <= PHASE4B_IDENTITY_PURITY_MAX_ADJACENT_COSINE
            and contrast >= PHASE4B_IDENTITY_PURITY_MIN_CONTRAST
        ):
            reasons.append("APPEARANCE_DISCONTINUITY")
        elif (
            enough_context
            and adjacent_cosine <= PHASE4B_IDENTITY_PURITY_MAX_ADJACENT_COSINE
            and geometry_jump >= PHASE4B_IDENTITY_PURITY_GEOMETRY_JUMP_THRESHOLD
            and contrast >= PHASE4B_IDENTITY_PURITY_MIN_CONTRAST
        ):
            reasons.append("APPEARANCE_GEOMETRY_DISCONTINUITY")

        if reasons:
            change_points.append(
                {
                    "split_before_frame": int(current.frame),
                    "previous_detection_id": str(previous.detection_id),
                    "current_detection_id": str(current.detection_id),
                    "frame_gap": frame_gap,
                    "adjacent_cosine": adjacent_cosine,
                    "left_coherence_median": left_coherence,
                    "right_coherence_median": right_coherence,
                    "cross_cosine_median": cross_cosine,
                    "coherence_contrast": contrast,
                    "geometry_jump": geometry_jump,
                    "reasons": reasons,
                }
            )
    return change_points


def _phase4b_plan_identity_purity_segments(
    detections: Sequence[Any],
    probes: Sequence[Any],
    embeddings_by_detection_id: Mapping[str, Any],
    *,
    fps: float,
    np: Any,
) -> dict[str, Any]:
    ordered = sorted(
        detections,
        key=lambda detection: (
            int(detection.frame),
            str(detection.detection_id),
        ),
    )
    if not ordered:
        return {
            "policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "segments": [],
            "change_points": [],
            "split_parent": False,
            "split_reasons": ["EMPTY_TRACKLET"],
        }

    change_points = _phase4b_identity_purity_change_points(
        probes,
        embeddings_by_detection_id,
        fps=fps,
        np=np,
    )
    change_by_frame: dict[int, list[str]] = {}
    for change in change_points:
        change_by_frame.setdefault(int(change["split_before_frame"]), []).extend(
            str(reason) for reason in change.get("reasons") or []
        )

    maximum_gap_frames = max(
        2,
        int(round(max(fps, 1.0) * PHASE4B_IDENTITY_PURITY_MAX_GAP_SECONDS)),
    )
    segment_specs: list[dict[str, Any]] = []
    start_index = 0
    boundary_before: list[str] = []
    for index in range(1, len(ordered)):
        current = ordered[index]
        previous = ordered[index - 1]
        reasons = list(change_by_frame.get(int(current.frame), []))
        if int(current.frame) - int(previous.frame) > maximum_gap_frames:
            reasons.append("TEMPORAL_GAP")
        reasons = sorted(set(reasons))
        if not reasons:
            continue
        if index - start_index < PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_OBSERVATIONS:
            continue
        segment_specs.append(
            {
                "start_index": start_index,
                "end_index_exclusive": index,
                "boundary_before": boundary_before,
                "boundary_after": reasons,
            }
        )
        start_index = index
        boundary_before = reasons

    segment_specs.append(
        {
            "start_index": start_index,
            "end_index_exclusive": len(ordered),
            "boundary_before": boundary_before,
            "boundary_after": [],
        }
    )

    if (
        len(segment_specs) >= 2
        and segment_specs[-1]["end_index_exclusive"]
        - segment_specs[-1]["start_index"]
        < PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_OBSERVATIONS
    ):
        tail = segment_specs.pop()
        segment_specs[-1]["end_index_exclusive"] = tail["end_index_exclusive"]
        segment_specs[-1]["boundary_after"] = []
        segment_specs[-1]["merged_short_tail"] = True

    segments: list[dict[str, Any]] = []
    for index, spec in enumerate(segment_specs, start=1):
        rows = ordered[spec["start_index"] : spec["end_index_exclusive"]]
        row_ids = {str(row.detection_id) for row in rows}
        segment_embeddings = [
            embeddings_by_detection_id[str(probe.detection_id)]
            for probe in probes
            if str(probe.detection_id) in row_ids
            and str(probe.detection_id) in embeddings_by_detection_id
        ]
        internal_median = (
            _phase4b_pairwise_cosine_median(
                np.stack(segment_embeddings).astype(np.float32), np
            )
            if len(segment_embeddings) >= 2
            else 1.0
        )
        purity_passed = (
            len(rows) >= PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_OBSERVATIONS
            and internal_median >= PHASE4B_IDENTITY_PURITY_MIN_INTERNAL_MEDIAN_COSINE
        )
        segments.append(
            {
                "segment_index": index,
                "start_frame": int(rows[0].frame),
                "end_frame_inclusive": int(rows[-1].frame),
                "observation_count": len(rows),
                "detection_ids": [str(row.detection_id) for row in rows],
                "boundary_before": list(spec.get("boundary_before") or []),
                "boundary_after": list(spec.get("boundary_after") or []),
                "merged_short_tail": bool(spec.get("merged_short_tail")),
                "probe_count": len(segment_embeddings),
                "internal_pairwise_cosine_median": internal_median,
                "passed": purity_passed,
                "classification": (
                    "TEMPORALLY_COHERENT_IDENTITY_SEGMENT"
                    if purity_passed
                    else "UNREVIEWABLE_MIXED_IDENTITY_SEGMENT"
                ),
                "rejection_reasons": (
                    []
                    if purity_passed
                    else ["LOW_INTERNAL_APPEARANCE_COHERENCE"]
                ),
            }
        )

    split_reasons = sorted(
        {
            reason
            for segment in segments
            for reason in (
                list(segment.get("boundary_before") or [])
                + list(segment.get("boundary_after") or [])
            )
        }
    )
    return {
        "policy": PHASE4B_IDENTITY_PURITY_POLICY,
        "segments": segments,
        "change_points": change_points,
        "split_parent": len(segments) > 1,
        "split_reasons": split_reasons,
        "maximum_review_window_frames": 0,
        "forced_time_window_split": False,
        "maximum_gap_frames": maximum_gap_frames,
        "probe_count": len(probes),
        "automatic_target_confirmation": False,
    }


def _phase4b_make_working_track(b0: Any, detections: Sequence[Any], internal_id: int) -> Any:
    track = b0.WorkingTrack(internal_id=internal_id)
    for detection in sorted(
        detections,
        key=lambda row: (int(row.frame), str(row.detection_id)),
    ):
        track.append(
            b0.Observation(
                frame_index=int(detection.frame),
                detection_id=str(detection.detection_id),
                bbox_xyxy=[float(value) for value in detection.bbox],
                confidence=float(detection.confidence),
            )
        )
    return track


def _phase4b_segment_candidates_by_identity_purity(
    *,
    state: Mapping[str, Any],
    shot: Mapping[str, Any],
    runtime: Mapping[str, Any],
    base_candidates: Sequence[Mapping[str, Any]],
    detections_by_candidate: Mapping[str, Sequence[Any]],
    clean_detections_by_candidate: Mapping[str, Sequence[Any]],
    strip_dir: Path,
) -> dict[str, Any]:
    import numpy as np

    if not base_candidates:
        return {
            "assignments": [],
            "candidates": [],
            "detections_by_candidate": {},
            "clean_detections_by_candidate": {},
            "parent_tracklet_summaries": [],
            "rejected_identity_purity_segments": [],
            "split_parent_tracklet_count": 0,
            "identity_segment_candidate_count": 0,
            "purity_probe_embedding_count": 0,
        }

    stage2b = runtime["stage2b"]
    probe_by_parent: dict[str, list[Any]] = {}
    all_probes: dict[str, Any] = {}
    for candidate in base_candidates:
        candidate_id = str(candidate["candidate_id"])
        probes = _phase4b_evenly_sample_detections(
            clean_detections_by_candidate[candidate_id],
            PHASE4B_IDENTITY_PURITY_MAX_PROBES,
        )
        probe_by_parent[candidate_id] = probes
        for probe in probes:
            all_probes[str(probe.detection_id)] = probe

    crops = stage2b.collect_crops(
        runtime["clip"],
        list(all_probes.values()),
        int(runtime["clip_meta"]["frame_count"]),
        int(runtime["clip_meta"]["width"]),
        int(runtime["clip_meta"]["height"]),
    )
    raw_embeddings = stage2b.embed_crops(
        crops,
        runtime["model"],
        runtime["transform"],
        runtime["torch"],
        runtime["device"],
        32,
    )
    embeddings_by_detection_id: dict[str, Any] = {}
    for detection_id, embedding in raw_embeddings.items():
        value = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if norm <= 0.0 or not np.isfinite(value).all():
            raise RuntimeError(
                "Phase 4-B identity-purity probe embedding is invalid: "
                f"{detection_id}"
            )
        embeddings_by_detection_id[str(detection_id)] = value / norm

    strip_dir.mkdir(parents=True, exist_ok=True)
    for candidate in base_candidates:
        (strip_dir / f"{str(candidate['candidate_id'])}.jpg").unlink(missing_ok=True)
    source_start = int(shot["start_frame"])
    frame_width = int(state["video"]["width"])
    frame_height = int(state["video"]["height"])
    fps = float(state["video"].get("fps") or runtime["clip_meta"].get("fps") or 25.0)
    final_assignments: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    final_detections: dict[str, list[Any]] = {}
    final_clean_detections: dict[str, list[Any]] = {}
    parent_summaries: list[dict[str, Any]] = []
    rejected_segments: list[dict[str, Any]] = []

    for parent_number, base in enumerate(base_candidates, start=1):
        parent_id = str(base["candidate_id"])
        parent_rows = sorted(
            detections_by_candidate[parent_id],
            key=lambda row: (int(row.frame), str(row.detection_id)),
        )
        parent_clean_rows = sorted(
            clean_detections_by_candidate[parent_id],
            key=lambda row: (int(row.frame), str(row.detection_id)),
        )
        plan = _phase4b_plan_identity_purity_segments(
            parent_rows,
            probe_by_parent[parent_id],
            embeddings_by_detection_id,
            fps=fps,
            np=np,
        )
        parent_summary = {
            "parent_tracklet_id": parent_id,
            "parent_start_frame": int(base["start_frame"]),
            "parent_end_frame_inclusive": int(base["end_frame_inclusive"]),
            "parent_observation_count": len(parent_rows),
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "split_parent": bool(plan["split_parent"]),
            "split_reasons": list(plan["split_reasons"]),
            "change_points": list(plan["change_points"]),
            "segment_count": len(plan["segments"]),
            "maximum_review_window_frames": int(
                plan.get("maximum_review_window_frames") or 0
            ),
            "maximum_gap_frames": int(plan.get("maximum_gap_frames") or 0),
            "probe_count": int(plan.get("probe_count") or 0),
            "target_memory_used_for_segmentation": False,
            "segments": [],
            "automatic_target_confirmation": False,
        }
        parent_summaries.append(parent_summary)
        segment_count = len(plan["segments"])
        for segment in plan["segments"]:
            detection_ids = set(str(value) for value in segment["detection_ids"])
            segment_rows = [
                row for row in parent_rows if str(row.detection_id) in detection_ids
            ]
            if not segment_rows:
                continue
            segment_observability = _phase4b_identity_observability(
                segment_rows,
                runtime["by_frame"],
                frame_width=frame_width,
                frame_height=frame_height,
            )
            clean_ids = set(
                str(value)
                for value in segment_observability.get("clean_detection_ids") or []
            )
            segment_clean_rows = [
                row for row in segment_rows if str(row.detection_id) in clean_ids
            ]
            segment_index = int(segment["segment_index"])
            candidate_id = (
                parent_id
                if segment_count == 1 and not plan["split_parent"]
                else f"{parent_id}_seg_{segment_index:03d}"
            )
            purity = {
                "policy": PHASE4B_IDENTITY_PURITY_POLICY,
                "passed": bool(segment["passed"])
                and segment_observability.get("passed") is True
                and len(segment_clean_rows)
                >= PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_CLEAN_FRAMES,
                "classification": str(segment["classification"]),
                "parent_tracklet_id": parent_id,
                "parent_tracklet_split": bool(plan["split_parent"]),
                "parent_split_reasons": list(plan["split_reasons"]),
                "segment_index": segment_index,
                "segment_count": segment_count,
                "segment_start_frame": source_start + int(segment_rows[0].frame),
                "segment_end_frame_inclusive": source_start
                + int(segment_rows[-1].frame),
                "segment_observation_count": len(segment_rows),
                "segment_clean_frame_count": len(segment_clean_rows),
                "probe_count": int(segment["probe_count"]),
                "internal_pairwise_cosine_median": float(
                    segment["internal_pairwise_cosine_median"]
                ),
                "boundary_before": list(segment["boundary_before"]),
                "boundary_after": list(segment["boundary_after"]),
                "bounded_review_window": False,
                "forced_time_window_split": False,
                "target_memory_used_for_segmentation": False,
                "automatic_target_confirmation": False,
                "rejection_reasons": list(segment["rejection_reasons"]),
            }
            if segment_observability.get("passed") is not True:
                purity["rejection_reasons"].append(
                    "SEGMENT_IDENTITY_OBSERVABILITY_FAILED"
                )
            if (
                len(segment_clean_rows)
                < PHASE4B_IDENTITY_PURITY_MIN_SEGMENT_CLEAN_FRAMES
            ):
                purity["rejection_reasons"].append(
                    "INSUFFICIENT_CLEAN_FRAMES_IN_IDENTITY_SEGMENT"
                )
            purity["rejection_reasons"] = sorted(
                set(str(value) for value in purity["rejection_reasons"])
            )
            parent_summary["segments"].append(
                {
                    "candidate_id": candidate_id,
                    **purity,
                }
            )

            working_track = _phase4b_make_working_track(
                runtime["b0"], segment_rows, parent_number * 1000 + segment_index
            )
            runtime["b0"].make_tracklet_strip(
                runtime["clip"],
                candidate_id,
                working_track,
                strip_dir / f"{candidate_id}.jpg",
            )
            candidate_row = {
                **dict(base),
                "candidate_id": candidate_id,
                "parent_tracklet_id": parent_id,
                "start_frame": source_start + int(segment_rows[0].frame),
                "end_frame_inclusive": source_start + int(segment_rows[-1].frame),
                "analysis_local_start_frame": int(segment_rows[0].frame),
                "analysis_local_end_frame_inclusive": int(segment_rows[-1].frame),
                "detection_count": len(segment_rows),
                "identity_observability": segment_observability,
                "identity_purity": purity,
                "reid_eligible_detection_count": len(segment_clean_rows),
            }
            if purity["passed"] is not True:
                rejected_segments.append(candidate_row)
                continue

            final_candidates.append(candidate_row)
            final_detections[candidate_id] = segment_rows
            final_clean_detections[candidate_id] = segment_clean_rows
            observability_by_id = {
                str(item["detection_id"]): item
                for item in segment_observability.get("per_observation") or []
                if isinstance(item, Mapping)
            }
            for detection in segment_rows:
                evidence = observability_by_id.get(str(detection.detection_id), {})
                final_assignments.append(
                    {
                        "candidate_id": candidate_id,
                        "parent_tracklet_id": parent_id,
                        "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
                        "identity_purity_segment_index": segment_index,
                        "identity_purity_segment_count": segment_count,
                        "identity_purity_passed": True,
                        "identity_purity_boundary_before": "|".join(
                            str(value) for value in purity["boundary_before"]
                        ),
                        "identity_purity_boundary_after": "|".join(
                            str(value) for value in purity["boundary_after"]
                        ),
                        "frame_index": source_start + int(detection.frame),
                        "analysis_local_frame_index": int(detection.frame),
                        "detection_id": str(detection.detection_id),
                        "class_id": int(detection.class_id),
                        "class_name": str(detection.class_name or "").strip().lower(),
                        "confidence": float(detection.confidence),
                        "x1": float(detection.bbox[0]),
                        "y1": float(detection.bbox[1]),
                        "x2": float(detection.bbox[2]),
                        "y2": float(detection.bbox[3]),
                        "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                        "clean_for_reid": bool(evidence.get("clean_for_reid")),
                        "crowded": bool(evidence.get("crowded")),
                        "multi_person_contaminated": bool(
                            evidence.get("multi_person_contaminated")
                        ),
                        "maximum_candidate_overlap_ratio": float(
                            evidence.get("maximum_candidate_overlap_ratio") or 0.0
                        ),
                        "bbox_width_height_ratio": float(
                            evidence.get("bbox_width_height_ratio") or 0.0
                        ),
                        "observability_rejection_reasons": "|".join(
                            str(value)
                            for value in evidence.get("rejection_reasons") or []
                        ),
                    }
                )

    return {
        "assignments": final_assignments,
        "candidates": final_candidates,
        "detections_by_candidate": final_detections,
        "clean_detections_by_candidate": final_clean_detections,
        "parent_tracklet_summaries": parent_summaries,
        "rejected_identity_purity_segments": rejected_segments,
        "split_parent_tracklet_count": sum(
            bool(row.get("split_parent")) for row in parent_summaries
        ),
        "identity_segment_candidate_count": len(final_candidates),
        "purity_probe_embedding_count": len(embeddings_by_detection_id),
    }

def _phase4b_make_identity_purity_rejection_sheet(
    *,
    rejected: Sequence[Mapping[str, Any]],
    strip_dir: Path,
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    tile_width = 680
    tile_height = 280
    columns = 2
    rows = max(1, int(math.ceil(max(1, len(rejected)) / columns)))
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, row in enumerate(rejected):
        candidate_id = str(row.get("candidate_id") or "unknown")
        strip = cv2.imread(str(strip_dir / f"{candidate_id}.jpg"))
        column = index % columns
        row_index = index // columns
        x0 = column * tile_width
        y0 = row_index * tile_height
        if strip is not None:
            available_height = 170
            scale = min(
                (tile_width - 20) / max(strip.shape[1], 1),
                available_height / max(strip.shape[0], 1),
            )
            resized = cv2.resize(
                strip,
                (
                    max(1, int(round(strip.shape[1] * scale))),
                    max(1, int(round(strip.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            canvas[
                y0 + 96 : y0 + 96 + resized.shape[0],
                x0 + 10 : x0 + 10 + resized.shape[1],
            ] = resized
        purity = dict(row.get("identity_purity") or {})
        cv2.putText(
            canvas,
            f"{candidate_id}  {purity.get('classification')}",
            (x0 + 10, y0 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"parent={purity.get('parent_tracklet_id')} "
                f"clean={purity.get('segment_clean_frame_count', 0)}/"
                f"{purity.get('segment_observation_count', 0)} "
                f"coherence={float(purity.get('internal_pairwise_cosine_median') or 0.0):.3f}"
            ),
            (x0 + 10, y0 + 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        boundaries = list(purity.get("boundary_before") or []) + list(
            purity.get("boundary_after") or []
        )
        cv2.putText(
            canvas,
            "boundaries=" + ("|".join(str(value) for value in boundaries) or "none"),
            (x0 + 10, y0 + 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not rejected:
        cv2.putText(
            canvas,
            "NO IDENTITY-PURITY REJECTIONS",
            (40, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Cannot write identity-purity sheet: {output_path}")


def _phase4b_make_observability_rejection_sheet(
    *,
    rejected: Sequence[Mapping[str, Any]],
    strip_dir: Path,
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    tile_width = 620
    tile_height = 250
    columns = 2
    rows = max(1, int(math.ceil(max(1, len(rejected)) / columns)))
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, row in enumerate(rejected):
        candidate_id = str(row.get("candidate_id") or "unknown")
        strip = cv2.imread(str(strip_dir / f"{candidate_id}.jpg"))
        column = index % columns
        row_index = index // columns
        x0 = column * tile_width
        y0 = row_index * tile_height
        if strip is not None:
            available_height = 170
            scale = min(
                (tile_width - 20) / max(strip.shape[1], 1),
                available_height / max(strip.shape[0], 1),
            )
            resized = cv2.resize(
                strip,
                (
                    max(1, int(round(strip.shape[1] * scale))),
                    max(1, int(round(strip.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            canvas[
                y0 + 70 : y0 + 70 + resized.shape[0],
                x0 + 10 : x0 + 10 + resized.shape[1],
            ] = resized
        observability = dict(row.get("identity_observability") or {})
        cv2.putText(
            canvas,
            f"{candidate_id}  {observability.get('classification')}",
            (x0 + 10, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"clean={observability.get('clean_frame_count', 0)}/"
                f"{observability.get('observation_count', 0)} "
                f"crowded={float(observability.get('crowded_frame_ratio') or 0.0):.2f} "
                f"multi={float(observability.get('multi_person_contamination_ratio') or 0.0):.2f}"
            ),
            (x0 + 10, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not rejected:
        cv2.putText(
            canvas,
            "NO IDENTITY-OBSERVABILITY REJECTIONS",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Cannot write identity observability sheet: {output_path}")


def _phase4b_role_confusion_evidence(
    candidate_detections: Sequence[Any],
    negative_by_frame: Mapping[int, Sequence[Any]],
) -> dict[str, Any]:
    confused = 0
    best_iou = 0.0
    matched_roles: dict[str, int] = {}
    evidence_rows: list[dict[str, Any]] = []

    for detection in candidate_detections:
        frame = int(detection.frame)
        bbox = [float(value) for value in detection.bbox]
        local_best_iou = 0.0
        local_best_role: str | None = None
        local_best_frame: int | None = None
        for other_frame in range(
            frame - PHASE4B_ROLE_CONFUSION_TEMPORAL_WINDOW,
            frame + PHASE4B_ROLE_CONFUSION_TEMPORAL_WINDOW + 1,
        ):
            for negative in negative_by_frame.get(other_frame, ()):
                iou = _phase4b_bbox_iou(bbox, [float(value) for value in negative.bbox])
                if iou > local_best_iou:
                    local_best_iou = iou
                    local_best_role = str(negative.class_name or "").strip().lower()
                    local_best_frame = int(other_frame)
        best_iou = max(best_iou, local_best_iou)
        if local_best_iou >= PHASE4B_ROLE_CONFUSION_MIN_IOU:
            confused += 1
            if local_best_role:
                matched_roles[local_best_role] = matched_roles.get(local_best_role, 0) + 1
            evidence_rows.append(
                {
                    "candidate_frame": frame,
                    "negative_frame": local_best_frame,
                    "negative_role": local_best_role,
                    "iou": local_best_iou,
                }
            )

    observation_count = len(candidate_detections)
    ratio = confused / observation_count if observation_count else 0.0
    rejected = (
        confused >= PHASE4B_ROLE_CONFUSION_MIN_OBSERVATIONS
        and ratio >= PHASE4B_ROLE_CONFUSION_MIN_RATIO
    )
    return {
        "policy": "TEMPORAL_STAFF_REFEREE_LABEL_CONFLICT_R1",
        "temporal_window_frames": PHASE4B_ROLE_CONFUSION_TEMPORAL_WINDOW,
        "minimum_iou": PHASE4B_ROLE_CONFUSION_MIN_IOU,
        "candidate_observation_count": observation_count,
        "confused_observation_count": confused,
        "confused_observation_ratio": ratio,
        "maximum_confusion_iou": best_iou,
        "matched_negative_roles": dict(sorted(matched_roles.items())),
        "evidence": evidence_rows[:20],
        "passed": not rejected,
        "rejection_reason": (
            None if not rejected else "TEMPORAL_STAFF_REFEREE_ROLE_CONFLICT"
        ),
    }


def _phase4b_metric_value(
    row: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = row.get(key)
    return float(default if value is None else value)


def _phase4b_user_confirmed_role_review_gate(
    row: Mapping[str, Any],
    *,
    negative_memory_available: bool,
) -> dict[str, Any]:
    compatible_crop_count = int(
        row.get("user_role_negative_scale_compatible_crop_count") or 0
    )
    cluster_count = int(row.get("user_role_negative_compatible_cluster_count") or 0)
    evidence_available = bool(
        negative_memory_available
        and compatible_crop_count > 0
        and cluster_count > 0
    )
    if not evidence_available:
        return {
            "policy": PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY,
            "negative_memory_available": bool(negative_memory_available),
            "scale_compatible_evidence_available": False,
            "passed": True,
            "review_mode": "ASSISTED_WITHOUT_SCALE_COMPATIBLE_USER_ROLE_NEGATIVES",
            "gates": {},
            "rejection_reasons": [],
        }

    crop_margin = _phase4b_metric_value(
        row, "user_role_crop_margin_median", -1.0
    )
    support = _phase4b_metric_value(
        row, "user_role_positive_margin_support_ratio", 0.0
    )
    prototype_margin = _phase4b_metric_value(
        row, "user_role_prototype_negative_margin", -1.0
    )
    gates = {
        "robust_crop_margin_median": {
            "value": crop_margin,
            "required": f">={PHASE4B_USER_ROLE_MIN_CROP_MARGIN_MEDIAN}",
            "passed": crop_margin >= PHASE4B_USER_ROLE_MIN_CROP_MARGIN_MEDIAN,
        },
        "positive_margin_support_ratio": {
            "value": support,
            "required": f">={PHASE4B_USER_ROLE_MIN_POSITIVE_MARGIN_SUPPORT}",
            "passed": support >= PHASE4B_USER_ROLE_MIN_POSITIVE_MARGIN_SUPPORT,
        },
        "cluster_prototype_margin": {
            "value": prototype_margin,
            "required": f">={PHASE4B_USER_ROLE_MIN_PROTOTYPE_MARGIN}",
            "passed": prototype_margin >= PHASE4B_USER_ROLE_MIN_PROTOTYPE_MARGIN,
        },
    }
    reasons = [name for name, gate in gates.items() if not bool(gate["passed"])]
    return {
        "policy": PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY,
        "negative_memory_available": True,
        "scale_compatible_evidence_available": True,
        "compatible_crop_count": compatible_crop_count,
        "compatible_cluster_count": cluster_count,
        "top_k": PHASE4B_USER_ROLE_NEGATIVE_TOP_K,
        "passed": not reasons,
        "review_mode": "TARGET_VS_USER_CONFIRMED_NON_PLAYER_ROLE_HARD_GATE",
        "gates": gates,
        "rejection_reasons": reasons,
    }


def _phase4b_detector_role_soft_gate(
    row: Mapping[str, Any],
    *,
    negative_memory_available: bool,
) -> dict[str, Any]:
    """Precision-first detector-role gate with a conservative hard veto.

    Detector-produced referee/staff labels are useful negative identity
    evidence, but they are not trusted enough for a one-signal rejection.
    We keep the existing warning thresholds for diagnostics and only reject
    when at least two independent comparisons are actually negative:

    * median target-vs-role crop margin <= 0;
    * positive-margin crop support < 0.5;
    * target-vs-role prototype margin <= 0.

    This prevents a coach/staff tracklet that looks more like the role-negative
    gallery than the selected player from entering assisted review, while a
    single noisy detector-role observation remains reviewable.
    """

    if not negative_memory_available:
        return {
            "policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
            "negative_memory_available": False,
            "passed": True,
            "soft_warning": False,
            "hard_veto": False,
            "review_mode": "NO_DETECTOR_ROLE_EVIDENCE",
            "gates": {},
            "warning_reasons": [],
            "rejection_reasons": [],
        }

    crop_margin = _phase4b_metric_value(
        row, "detector_role_crop_margin_median", 1.0
    )
    support = _phase4b_metric_value(
        row, "detector_role_positive_margin_support_ratio", 1.0
    )
    prototype_margin = _phase4b_metric_value(
        row, "detector_role_prototype_negative_margin", 1.0
    )

    warnings = []
    if crop_margin < PHASE4B_REVIEW_MIN_CROP_MARGIN_MEDIAN:
        warnings.append("crop_margin_median")
    if support < PHASE4B_REVIEW_MIN_POSITIVE_MARGIN_SUPPORT:
        warnings.append("positive_margin_support_ratio")
    if prototype_margin < PHASE4B_REVIEW_MIN_PROTOTYPE_NEGATIVE_MARGIN:
        warnings.append("prototype_negative_margin")

    hard_failures = []
    if crop_margin <= PHASE4B_DETECTOR_ROLE_HARD_MAX_CROP_MARGIN_MEDIAN:
        hard_failures.append("crop_margin_median_nonpositive")
    if support < PHASE4B_DETECTOR_ROLE_HARD_MAX_POSITIVE_MARGIN_SUPPORT:
        hard_failures.append("positive_margin_support_below_half")
    if prototype_margin <= PHASE4B_DETECTOR_ROLE_HARD_MAX_PROTOTYPE_MARGIN:
        hard_failures.append("prototype_negative_margin_nonpositive")

    hard_veto = (
        len(hard_failures) >= PHASE4B_DETECTOR_ROLE_HARD_MIN_FAILED_SIGNALS
    )
    return {
        "policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
        "negative_memory_available": True,
        "passed": not hard_veto,
        "soft_warning": bool(warnings),
        "hard_veto": hard_veto,
        "hard_veto_min_failed_signals": (
            PHASE4B_DETECTOR_ROLE_HARD_MIN_FAILED_SIGNALS
        ),
        "review_mode": (
            "DETECTOR_ROLE_STRONG_MULTI_EVIDENCE_HARD_VETO"
            if hard_veto
            else "DETECTOR_ROLE_EVIDENCE_REVIEWABLE"
        ),
        "gates": {
            "crop_margin_median": {
                "value": crop_margin,
                "warning_below": PHASE4B_REVIEW_MIN_CROP_MARGIN_MEDIAN,
                "hard_veto_at_or_below": (
                    PHASE4B_DETECTOR_ROLE_HARD_MAX_CROP_MARGIN_MEDIAN
                ),
            },
            "positive_margin_support_ratio": {
                "value": support,
                "warning_below": PHASE4B_REVIEW_MIN_POSITIVE_MARGIN_SUPPORT,
                "hard_veto_below": (
                    PHASE4B_DETECTOR_ROLE_HARD_MAX_POSITIVE_MARGIN_SUPPORT
                ),
            },
            "prototype_negative_margin": {
                "value": prototype_margin,
                "warning_below": (
                    PHASE4B_REVIEW_MIN_PROTOTYPE_NEGATIVE_MARGIN
                ),
                "hard_veto_at_or_below": (
                    PHASE4B_DETECTOR_ROLE_HARD_MAX_PROTOTYPE_MARGIN
                ),
            },
        },
        "warning_reasons": warnings,
        "rejection_reasons": (
            [f"strong_detector_role:{name}" for name in hard_failures]
            if hard_veto
            else []
        ),
    }


def _phase4b_role_negative_review_gate(
    row: Mapping[str, Any],
    *,
    negative_memory_available: bool,
) -> dict[str, Any]:
    """Backward-compatible alias for the user-confirmed hard role gate."""

    return _phase4b_user_confirmed_role_review_gate(
        row,
        negative_memory_available=negative_memory_available,
    )


def _phase4b_identity_negative_review_gate(
    row: Mapping[str, Any],
    *,
    negative_memory_available: bool,
) -> dict[str, Any]:
    compatible_crop_count = int(
        row.get("identity_negative_scale_compatible_crop_count") or 0
    )
    cluster_count = int(row.get("identity_negative_compatible_cluster_count") or 0)
    evidence_available = bool(
        negative_memory_available
        and compatible_crop_count > 0
        and cluster_count > 0
    )
    if not evidence_available:
        return {
            "policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
            "negative_memory_available": bool(negative_memory_available),
            "scale_compatible_evidence_available": False,
            "passed": True,
            "review_mode": "ASSISTED_WITHOUT_SCALE_COMPATIBLE_IDENTITY_NEGATIVES",
            "gates": {},
            "rejection_reasons": [],
        }

    crop_margin = _phase4b_metric_value(row, "identity_crop_margin_median", -1.0)
    support = _phase4b_metric_value(row, "identity_positive_margin_support_ratio", 0.0)
    prototype_margin = _phase4b_metric_value(row, "identity_prototype_negative_margin", -1.0)
    gates = {
        "robust_crop_margin_median": {
            "value": crop_margin,
            "required": f">={PHASE4B_IDENTITY_NEGATIVE_MIN_CROP_MARGIN_MEDIAN}",
            "passed": crop_margin >= PHASE4B_IDENTITY_NEGATIVE_MIN_CROP_MARGIN_MEDIAN,
        },
        "positive_margin_support_ratio": {
            "value": support,
            "required": f">={PHASE4B_IDENTITY_NEGATIVE_MIN_POSITIVE_MARGIN_SUPPORT}",
            "passed": support >= PHASE4B_IDENTITY_NEGATIVE_MIN_POSITIVE_MARGIN_SUPPORT,
        },
        "cluster_prototype_margin": {
            "value": prototype_margin,
            "required": f">={PHASE4B_IDENTITY_NEGATIVE_MIN_PROTOTYPE_MARGIN}",
            "passed": prototype_margin >= PHASE4B_IDENTITY_NEGATIVE_MIN_PROTOTYPE_MARGIN,
        },
    }
    reasons = [name for name, gate in gates.items() if not bool(gate["passed"])]
    return {
        "policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
        "negative_memory_available": True,
        "scale_compatible_evidence_available": True,
        "compatible_crop_count": compatible_crop_count,
        "compatible_cluster_count": cluster_count,
        "top_k": PHASE4B_IDENTITY_NEGATIVE_TOP_K,
        "passed": not reasons,
        "review_mode": "TARGET_VS_SCALE_COMPATIBLE_CLUSTER_ROBUST_IDENTITY_NEGATIVES",
        "gates": gates,
        "rejection_reasons": reasons,
    }


def _phase4b_candidate_negative_review_gate(
    row: Mapping[str, Any],
    *,
    role_negative_memory_available: bool,
    identity_negative_memory_available: bool,
    detector_role_memory_available: bool | None = None,
) -> dict[str, Any]:
    user_role_gate = _phase4b_user_confirmed_role_review_gate(
        row,
        negative_memory_available=role_negative_memory_available,
    )
    detector_role_gate = _phase4b_detector_role_soft_gate(
        row,
        negative_memory_available=bool(detector_role_memory_available),
    )
    identity_gate = _phase4b_identity_negative_review_gate(
        row,
        negative_memory_available=identity_negative_memory_available,
    )
    reasons = [
        *(
            f"user_role:{name}"
            for name in user_role_gate.get("rejection_reasons") or []
        ),
        *(
            f"detector_role:{name}"
            for name in detector_role_gate.get("rejection_reasons") or []
        ),
        *(f"identity:{name}" for name in identity_gate.get("rejection_reasons") or []),
    ]
    return {
        "policy": PHASE4B_COMBINED_NEGATIVE_POLICY,
        "negative_memory_available": bool(
            role_negative_memory_available
            or detector_role_memory_available
            or identity_negative_memory_available
        ),
        "role_negative_gate": user_role_gate,
        "user_confirmed_role_gate": user_role_gate,
        # Keep the historical key for compatibility with stored UI/report
        # readers, but its `passed` value now participates in the hard gate.
        "detector_role_soft_gate": detector_role_gate,
        "identity_negative_gate": identity_gate,
        "passed": bool(user_role_gate.get("passed"))
        and bool(detector_role_gate.get("passed"))
        and bool(identity_gate.get("passed")),
        "review_mode": "USER_ROLE_HARD_DETECTOR_STRONG_VETO_IDENTITY_ROBUST",
        "rejection_reasons": reasons,
        "soft_warning_reasons": [
            f"detector_role:{name}"
            for name in detector_role_gate.get("warning_reasons") or []
        ],
    }


def _phase4b_strict_target_evidence_passed(
    row: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> bool:
    return (
        float(row.get("retrieval_score") or 0.0)
        >= float(policy["minimum_retrieval_score"])
        and float(row.get("prototype_target_similarity") or 0.0)
        >= float(policy["minimum_prototype_similarity"])
    )


def _phase4b_plausible_assisted_review_eligible(
    row: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> bool:
    review_gate = dict(row.get("negative_review_gate") or {})
    user_role_gate = dict(review_gate.get("user_confirmed_role_gate") or {})
    detector_role_gate = dict(review_gate.get("detector_role_soft_gate") or {})
    identity_gate = dict(review_gate.get("identity_negative_gate") or {})
    observability = dict(row.get("identity_observability") or {})
    purity = dict(row.get("identity_purity") or {})
    if observability.get("passed") is not True:
        return False
    if purity.get("passed") is not True:
        return False
    if user_role_gate.get("passed") is not True:
        return False
    if detector_role_gate.get("passed") is not True:
        return False
    if identity_gate.get("passed") is not True:
        return False
    return (
        float(row.get("retrieval_score") or 0.0)
        >= float(policy["minimum_plausible_score"])
        and float(row.get("prototype_target_similarity") or 0.0)
        >= float(policy["minimum_plausible_prototype"])
    )


def _phase4b_identity_rescue_eligible(
    row: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> bool:
    """Compatibility wrapper for R8 tests and older callers."""

    return _phase4b_plausible_assisted_review_eligible(row, policy=policy)


def _phase4b_track_duplicate_evidence(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_by_frame = {int(row["analysis_local_frame_index"]): row for row in left_rows}
    right_by_frame = {int(row["analysis_local_frame_index"]): row for row in right_rows}
    common = sorted(set(left_by_frame) & set(right_by_frame))
    ious = []
    for frame in common:
        left = left_by_frame[frame]
        right = right_by_frame[frame]
        ious.append(
            _phase4b_bbox_iou(
                [left["x1"], left["y1"], left["x2"], left["y2"]],
                [right["x1"], right["y1"], right["x2"], right["y2"]],
            )
        )
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    denominator = max(1, min(len(left_rows), len(right_rows)))
    temporal_overlap = len(common) / denominator
    duplicate = (
        len(common) >= PHASE4B_DUPLICATE_MIN_COMMON_FRAMES
        and mean_iou >= PHASE4B_DUPLICATE_MIN_MEAN_IOU
        and temporal_overlap >= PHASE4B_DUPLICATE_MIN_TEMPORAL_OVERLAP
    )
    return {
        "common_frame_count": len(common),
        "mean_iou_on_common_frames": mean_iou,
        "temporal_overlap_ratio": temporal_overlap,
        "duplicate": duplicate,
    }


def _phase4b_assignment_geometry(row: Mapping[str, Any]) -> tuple[float, float, float]:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1))


def _phase4b_candidate_continuation_evidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    assignments_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    np: Any,
) -> dict[str, Any]:
    if int(left.get("start_frame") or 0) > int(right.get("start_frame") or 0):
        left, right = right, left
    left_parent = str(left.get("parent_tracklet_id") or left.get("candidate_id") or "")
    right_parent = str(right.get("parent_tracklet_id") or right.get("candidate_id") or "")
    if left_parent and left_parent == right_parent and str(left.get("candidate_id")) != str(right.get("candidate_id")):
        return {
            "linked": False,
            "reason": "IDENTITY_PURITY_SEGMENT_BOUNDARY",
            "same_parent_tracklet": True,
            "parent_tracklet_id": left_parent,
        }
    left_end = int(left.get("end_frame_inclusive") or 0)
    right_start = int(right.get("start_frame") or 0)
    gap = max(0, right_start - left_end - 1)
    if gap > PHASE4B_CONTINUATION_MAX_GAP_FRAMES:
        return {"linked": False, "reason": "TEMPORAL_GAP", "gap_frames": gap}

    left_proto = np.asarray(left.get("_prototype_vector"), dtype=np.float32).reshape(-1)
    right_proto = np.asarray(right.get("_prototype_vector"), dtype=np.float32).reshape(-1)
    if left_proto.size == 0 or right_proto.size == 0 or left_proto.size != right_proto.size:
        return {"linked": False, "reason": "MISSING_PROTOTYPE", "gap_frames": gap}
    left_proto = left_proto / max(1e-12, float(np.linalg.norm(left_proto)))
    right_proto = right_proto / max(1e-12, float(np.linalg.norm(right_proto)))
    prototype_cosine = float(left_proto @ right_proto)
    if prototype_cosine < PHASE4B_CONTINUATION_MIN_PROTOTYPE_COSINE:
        return {
            "linked": False,
            "reason": "PROTOTYPE_MISMATCH",
            "gap_frames": gap,
            "prototype_cosine": prototype_cosine,
        }

    left_scales = dict(left.get("candidate_scale_class_counts") or {})
    right_scales = dict(right.get("candidate_scale_class_counts") or {})
    left_scale = max(left_scales, key=left_scales.get) if left_scales else "unknown"
    right_scale = max(right_scales, key=right_scales.get) if right_scales else "unknown"
    if not _phase4b_scale_compatible(left_scale, right_scale):
        return {
            "linked": False,
            "reason": "SCALE_MISMATCH",
            "gap_frames": gap,
            "prototype_cosine": prototype_cosine,
            "left_scale": left_scale,
            "right_scale": right_scale,
        }

    left_rows = [
        dict(row)
        for row in assignments_by_candidate.get(str(left.get("candidate_id") or ""), [])
    ]
    right_rows = [
        dict(row)
        for row in assignments_by_candidate.get(str(right.get("candidate_id") or ""), [])
    ]
    if not left_rows or not right_rows:
        return {
            "linked": False,
            "reason": "MISSING_ASSIGNMENTS",
            "gap_frames": gap,
            "prototype_cosine": prototype_cosine,
        }
    left_rows.sort(key=lambda row: int(row["analysis_local_frame_index"]))
    right_rows.sort(key=lambda row: int(row["analysis_local_frame_index"]))
    left_by_frame = {int(row["analysis_local_frame_index"]): row for row in left_rows}
    right_by_frame = {int(row["analysis_local_frame_index"]): row for row in right_rows}
    common = sorted(set(left_by_frame) & set(right_by_frame))
    best_iou = 0.0
    best_center_distance = float("inf")
    best_height_ratio = float("inf")
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    if common:
        pairs = [(left_by_frame[frame], right_by_frame[frame]) for frame in common]
    else:
        pairs = [(left_rows[-1], right_rows[0])]
    for left_row, right_row in pairs:
        iou = _phase4b_bbox_iou(
            [left_row["x1"], left_row["y1"], left_row["x2"], left_row["y2"]],
            [right_row["x1"], right_row["y1"], right_row["x2"], right_row["y2"]],
        )
        best_iou = max(best_iou, float(iou))
        left_cx, left_cy, left_h = _phase4b_assignment_geometry(left_row)
        right_cx, right_cy, right_h = _phase4b_assignment_geometry(right_row)
        mean_h = max(1.0, (left_h + right_h) / 2.0)
        center_distance = math.hypot(left_cx - right_cx, left_cy - right_cy) / mean_h
        height_ratio = max(left_h, right_h) / max(1.0, min(left_h, right_h))
        best_center_distance = min(best_center_distance, center_distance)
        best_height_ratio = min(best_height_ratio, height_ratio)
    spatial_pass = bool(
        best_iou >= PHASE4B_CONTINUATION_MIN_ENDPOINT_IOU
        or (
            best_center_distance
            <= PHASE4B_CONTINUATION_MAX_CENTER_DISTANCE_BY_HEIGHT
            and best_height_ratio <= PHASE4B_CONTINUATION_MAX_HEIGHT_RATIO
        )
    )
    return {
        "linked": spatial_pass,
        "reason": "CONTINUATION_MATCH" if spatial_pass else "SPATIAL_DISCONTINUITY",
        "gap_frames": gap,
        "common_frame_count": len(common),
        "prototype_cosine": prototype_cosine,
        "best_iou": best_iou,
        "best_center_distance_by_height": best_center_distance,
        "best_height_ratio": best_height_ratio,
        "left_scale": left_scale,
        "right_scale": right_scale,
    }


def _phase4b_assign_continuation_groups(
    rows: Sequence[dict[str, Any]],
    *,
    assignments_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    np: Any,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_evidence: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            evidence = _phase4b_candidate_continuation_evidence(
                left,
                right,
                assignments_by_candidate=assignments_by_candidate,
                np=np,
            )
            if evidence.get("linked") is True:
                union(left_index, right_index)
                edge_evidence.append(
                    {
                        "left_candidate_id": str(left.get("candidate_id") or ""),
                        "right_candidate_id": str(right.get("candidate_id") or ""),
                        **evidence,
                    }
                )

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped_indices.setdefault(find(index), []).append(index)
    summaries: list[dict[str, Any]] = []
    for group_number, indices in enumerate(
        sorted(grouped_indices.values(), key=lambda value: min(value)),
        start=1,
    ):
        members = [rows[index] for index in indices]
        member_ids = [str(row.get("candidate_id") or "") for row in members]
        start_frame = min(int(row.get("start_frame") or 0) for row in members)
        end_frame = max(int(row.get("end_frame_inclusive") or 0) for row in members)
        clean_frames = sum(
            int(dict(row.get("identity_observability") or {}).get("clean_frame_count") or 0)
            for row in members
        )
        group_id = f"continuation_group_{group_number:03d}"
        summary = {
            "group_id": group_id,
            "member_candidate_ids": member_ids,
            "member_count": len(member_ids),
            "start_frame": start_frame,
            "end_frame_inclusive": end_frame,
            "source_frame_span": end_frame - start_frame + 1,
            "aggregate_clean_frame_count": clean_frames,
            "edge_evidence": [
                edge
                for edge in edge_evidence
                if edge["left_candidate_id"] in member_ids
                and edge["right_candidate_id"] in member_ids
            ],
        }
        summaries.append(summary)
        for row in members:
            row["continuation_group_id"] = group_id
            row["continuation_group_member_ids"] = member_ids
            row["continuation_group_member_count"] = len(member_ids)
            row["continuation_group_start_frame"] = start_frame
            row["continuation_group_end_frame_inclusive"] = end_frame
            row["continuation_group_source_frame_span"] = end_frame - start_frame + 1
            row["continuation_group_aggregate_clean_frame_count"] = clean_frames
    return summaries


def _phase4b_candidate_corroboration_evidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    assignments_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    np: Any,
) -> dict[str, Any]:
    left_parent = str(left.get("parent_tracklet_id") or left.get("candidate_id") or "")
    right_parent = str(right.get("parent_tracklet_id") or right.get("candidate_id") or "")
    if not left_parent or not right_parent or left_parent == right_parent:
        return {
            "linked": False,
            "reason": "SAME_OR_MISSING_PARENT_TRACKLET",
            "left_parent_tracklet_id": left_parent,
            "right_parent_tracklet_id": right_parent,
        }

    left_proto = np.asarray(left.get("_prototype_vector"), dtype=np.float32).reshape(-1)
    right_proto = np.asarray(right.get("_prototype_vector"), dtype=np.float32).reshape(-1)
    if left_proto.size == 0 or right_proto.size == 0 or left_proto.size != right_proto.size:
        return {"linked": False, "reason": "MISSING_PROTOTYPE"}
    prototype_cosine = _phase4b_cosine(left_proto, right_proto, np)
    if prototype_cosine < PHASE4B_CORROBORATION_MIN_PROTOTYPE_COSINE:
        return {
            "linked": False,
            "reason": "PROTOTYPE_MISMATCH",
            "prototype_cosine": prototype_cosine,
        }

    left_context = float(left.get("target_context_similarity") or 0.0)
    right_context = float(right.get("target_context_similarity") or 0.0)
    context_distance = abs(left_context - right_context)
    if context_distance > PHASE4B_CORROBORATION_MAX_CONTEXT_DISTANCE:
        return {
            "linked": False,
            "reason": "TARGET_CONTEXT_MISMATCH",
            "prototype_cosine": prototype_cosine,
            "target_context_distance": context_distance,
        }

    left_rows = [
        dict(row)
        for row in assignments_by_candidate.get(str(left.get("candidate_id") or ""), [])
    ]
    right_rows = [
        dict(row)
        for row in assignments_by_candidate.get(str(right.get("candidate_id") or ""), [])
    ]
    left_by_frame = {int(row["analysis_local_frame_index"]): row for row in left_rows}
    right_by_frame = {int(row["analysis_local_frame_index"]): row for row in right_rows}
    common = sorted(set(left_by_frame) & set(right_by_frame))
    if len(common) < PHASE4B_CORROBORATION_MIN_COMMON_FRAMES:
        return {
            "linked": False,
            "reason": "INSUFFICIENT_COMMON_FRAMES",
            "common_frame_count": len(common),
            "prototype_cosine": prototype_cosine,
            "target_context_distance": context_distance,
        }

    ious: list[float] = []
    center_distances: list[float] = []
    height_ratios: list[float] = []
    for frame in common:
        left_row = left_by_frame[frame]
        right_row = right_by_frame[frame]
        ious.append(
            _phase4b_bbox_iou(
                [left_row["x1"], left_row["y1"], left_row["x2"], left_row["y2"]],
                [right_row["x1"], right_row["y1"], right_row["x2"], right_row["y2"]],
            )
        )
        left_cx, left_cy, left_h = _phase4b_assignment_geometry(left_row)
        right_cx, right_cy, right_h = _phase4b_assignment_geometry(right_row)
        mean_h = max(1.0, (left_h + right_h) / 2.0)
        center_distances.append(
            math.hypot(left_cx - right_cx, left_cy - right_cy) / mean_h
        )
        height_ratios.append(max(left_h, right_h) / max(1.0, min(left_h, right_h)))

    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    median_center_distance = float(np.median(np.asarray(center_distances, dtype=np.float32)))
    median_height_ratio = float(np.median(np.asarray(height_ratios, dtype=np.float32)))
    spatial_pass = bool(
        mean_iou >= PHASE4B_CORROBORATION_MIN_MEAN_IOU
        or (
            median_center_distance
            <= PHASE4B_CORROBORATION_MAX_CENTER_DISTANCE_BY_HEIGHT
            and median_height_ratio <= PHASE4B_CORROBORATION_MAX_HEIGHT_RATIO
        )
    )
    return {
        "linked": spatial_pass,
        "reason": "INDEPENDENT_TRACKLET_CORROBORATION" if spatial_pass else "SPATIAL_MISMATCH",
        "common_frame_count": len(common),
        "mean_iou_on_common_frames": mean_iou,
        "median_center_distance_by_height": median_center_distance,
        "median_height_ratio": median_height_ratio,
        "prototype_cosine": prototype_cosine,
        "target_context_distance": context_distance,
        "left_parent_tracklet_id": left_parent,
        "right_parent_tracklet_id": right_parent,
    }


def _phase4b_assign_corroboration_groups(
    rows: Sequence[dict[str, Any]],
    *,
    assignments_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    np: Any,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    evidence_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            evidence = _phase4b_candidate_corroboration_evidence(
                left,
                right,
                assignments_by_candidate=assignments_by_candidate,
                np=np,
            )
            if evidence.get("linked") is True:
                union(left_index, right_index)
                evidence_rows.append(
                    {
                        "left_candidate_id": str(left.get("candidate_id") or ""),
                        "right_candidate_id": str(right.get("candidate_id") or ""),
                        **evidence,
                    }
                )

    grouped: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped.setdefault(find(index), []).append(index)

    summaries: list[dict[str, Any]] = []
    for group_number, indices in enumerate(
        sorted(grouped.values(), key=lambda value: min(value)),
        start=1,
    ):
        members = [rows[index] for index in indices]
        member_ids = [str(row.get("candidate_id") or "") for row in members]
        parent_ids = sorted(
            {str(row.get("parent_tracklet_id") or row.get("candidate_id") or "") for row in members}
        )
        group_id = f"corroboration_group_{group_number:03d}"
        linked_evidence = [
            row
            for row in evidence_rows
            if row["left_candidate_id"] in member_ids
            and row["right_candidate_id"] in member_ids
        ]
        summary = {
            "group_id": group_id,
            "policy": PHASE4B_CORROBORATION_POLICY,
            "member_candidate_ids": member_ids,
            "member_count": len(member_ids),
            "independent_parent_tracklet_ids": parent_ids,
            "independent_parent_count": len(parent_ids),
            "corroborated": len(parent_ids) >= 2 and bool(linked_evidence),
            "edge_evidence": linked_evidence,
        }
        summaries.append(summary)
        for row in members:
            row["corroboration_policy"] = PHASE4B_CORROBORATION_POLICY
            row["corroboration_group_id"] = group_id
            row["corroboration_group_member_ids"] = member_ids
            row["corroboration_group_member_count"] = len(member_ids)
            row["corroboration_independent_parent_count"] = len(parent_ids)
            row["corroborated_by_independent_tracklet"] = bool(summary["corroborated"])
    return summaries


def _phase4b_normalized_interval(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    if maximum <= minimum:
        return 1.0 if value >= maximum else 0.0
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))


def _phase4b_stable_review_score(
    row: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> float:
    """Score review evidence without letting uncalibrated torso context own a slot.

    R11 treated target-context similarity as a major ranking signal. The live
    failure bundle proved that the true target could have lower torso-context
    similarity than a visually different player. R12 therefore retains context
    only as a small soft tie-break and gives more weight to identity purity,
    clean-frame support, and frozen ReID evidence.
    """
    observability = dict(row.get("identity_observability") or {})
    clean_count = int(observability.get("clean_frame_count") or 0)
    clean_ratio = float(observability.get("clean_frame_ratio") or 0.0)
    duration = int(row.get("tracklet_detection_count") or row.get("detection_count") or 0)
    retrieval_component = _phase4b_normalized_interval(
        float(row.get("retrieval_score") or 0.0),
        float(policy["minimum_plausible_score"]),
        float(policy["minimum_retrieval_score"]),
    )
    prototype_component = _phase4b_normalized_interval(
        float(row.get("prototype_target_similarity") or 0.0),
        float(policy["minimum_plausible_prototype"]),
        float(policy["minimum_prototype_similarity"]),
    )
    clean_component = min(
        1.0, clean_count / max(1.0, PHASE4B_STABLE_EVIDENCE_CLEAN_FRAME_CAP)
    )
    duration_component = min(
        1.0, duration / max(1.0, PHASE4B_STABLE_EVIDENCE_DURATION_CAP)
    )
    purity = dict(row.get("identity_purity") or {})
    purity_component = _phase4b_normalized_interval(
        float(purity.get("internal_pairwise_cosine_median") or 0.0),
        PHASE4B_IDENTITY_PURITY_MIN_INTERNAL_MEDIAN_COSINE,
        0.90,
    )
    group_count = int(row.get("continuation_group_member_count") or 1)
    continuation_component = min(1.0, max(0, group_count - 1) / 2.0)
    group_span = int(row.get("continuation_group_source_frame_span") or duration)
    span_component = min(1.0, group_span / 96.0)
    target_context_component = max(
        PHASE4B_TARGET_CONTEXT_MIN_SIMILARITY,
        min(1.0, float(row.get("target_context_similarity") or 0.0)),
    )
    corroboration_parent_count = int(
        row.get("corroboration_independent_parent_count") or 1
    )
    corroboration_component = min(
        1.0, max(0, corroboration_parent_count - 1) / 2.0
    )
    score = (
        0.25 * retrieval_component
        + 0.20 * prototype_component
        + 0.04 * target_context_component
        + 0.10 * corroboration_component
        + 0.14 * clean_component
        + 0.12 * max(0.0, min(1.0, clean_ratio))
        + 0.04 * duration_component
        + 0.08 * purity_component
        + 0.015 * continuation_component
        + 0.015 * span_component
    )
    return float(score)


def _phase4b_group_representative_score(row: Mapping[str, Any]) -> float:
    """Prefer the cleanest identity-pure member inside one evidence group."""
    observability = dict(row.get("identity_observability") or {})
    purity = dict(row.get("identity_purity") or {})
    clean_ratio = max(0.0, min(1.0, float(observability.get("clean_frame_ratio") or 0.0)))
    purity_median = max(
        0.0,
        min(1.0, float(purity.get("internal_pairwise_cosine_median") or 0.0)),
    )
    stable = max(0.0, min(1.0, float(row.get("stable_review_score") or 0.0)))
    return float(0.45 * purity_median + 0.35 * clean_ratio + 0.20 * stable)


def _phase4b_group_confidence(
    representative: Mapping[str, Any],
    group: Mapping[str, Any],
) -> float:
    """Rank corroborated groups by independent agreement, not torso color.

    This remains assisted-review evidence only. It cannot confirm a target.
    """
    edges = [
        dict(row)
        for row in group.get("edge_evidence") or []
        if isinstance(row, Mapping)
    ]
    prototype_agreement = max(
        [float(row.get("prototype_cosine") or 0.0) for row in edges] + [0.0]
    )
    common_frame_support = min(
        1.0,
        max([int(row.get("common_frame_count") or 0) for row in edges] + [0]) / 4.0,
    )
    observability = dict(representative.get("identity_observability") or {})
    purity = dict(representative.get("identity_purity") or {})
    clean_ratio = max(0.0, min(1.0, float(observability.get("clean_frame_ratio") or 0.0)))
    purity_median = max(
        0.0,
        min(1.0, float(purity.get("internal_pairwise_cosine_median") or 0.0)),
    )
    retrieval = max(0.0, min(1.0, float(representative.get("retrieval_score") or 0.0)))
    target_prototype = max(
        0.0,
        min(1.0, float(representative.get("prototype_target_similarity") or 0.0)),
    )
    return float(
        0.30 * prototype_agreement
        + 0.22 * purity_median
        + 0.18 * clean_ratio
        + 0.15 * retrieval
        + 0.10 * target_prototype
        + 0.05 * common_frame_support
    )


def _phase4b_select_assisted_review_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    policy: Mapping[str, Any],
    assignments_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    np: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    pool = [
        row
        for row in rows
        if _phase4b_plausible_assisted_review_eligible(row, policy=policy)
    ]
    corroboration_groups = _phase4b_assign_corroboration_groups(
        pool,
        assignments_by_candidate=assignments_by_candidate,
        np=np,
    )
    continuation_groups = _phase4b_assign_continuation_groups(
        pool,
        assignments_by_candidate=assignments_by_candidate,
        np=np,
    )
    for row in pool:
        row["assisted_review_selection_policy"] = (
            PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY
        )
        row["review_catalog_policy"] = PHASE4B_REVIEW_CATALOG_POLICY
        row["target_context_policy"] = PHASE4B_TARGET_CONTEXT_POLICY
        row["corroboration_policy"] = PHASE4B_CORROBORATION_POLICY
        row["group_representative_policy"] = PHASE4B_GROUP_REPRESENTATIVE_POLICY
        row["group_confidence_policy"] = PHASE4B_GROUP_CONFIDENCE_POLICY
        row["stable_review_score"] = _phase4b_stable_review_score(
            row, policy=policy
        )
        row["strict_target_evidence_passed"] = _phase4b_strict_target_evidence_passed(
            row, policy=policy
        )
        row["group_representative_score"] = _phase4b_group_representative_score(row)

    representatives: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        if row.get("corroborated_by_independent_tracklet") is True:
            group_key = str(row.get("corroboration_group_id") or "")
        else:
            group_key = str(
                row.get("continuation_group_id")
                or row.get("corroboration_group_id")
                or row.get("candidate_id")
                or ""
            )
        row["review_evidence_group_id"] = group_key
        by_group.setdefault(group_key, []).append(row)

    corroboration_by_id = {
        str(group.get("group_id") or ""): group for group in corroboration_groups
    }
    for group_key, members in by_group.items():
        representative = max(
            members,
            key=lambda row: (
                float(row.get("group_representative_score") or 0.0),
                float(
                    dict(row.get("identity_purity") or {}).get(
                        "internal_pairwise_cosine_median"
                    )
                    or 0.0
                ),
                float(
                    dict(row.get("identity_observability") or {}).get(
                        "clean_frame_ratio"
                    )
                    or 0.0
                ),
                float(row.get("stable_review_score") or 0.0),
                float(row.get("retrieval_score") or 0.0),
            ),
        )
        representative["review_evidence_group_representative"] = True
        group = corroboration_by_id.get(group_key)
        if group is not None and group.get("corroborated") is True:
            confidence = _phase4b_group_confidence(representative, group)
        else:
            confidence = float(representative.get("stable_review_score") or 0.0)
        representative["group_confidence_score"] = float(confidence)
        representatives.append(representative)
        if group is not None:
            group["representative_policy"] = PHASE4B_GROUP_REPRESENTATIVE_POLICY
            group["representative_candidate_id"] = str(
                representative.get("candidate_id") or ""
            )
            group["representative_score"] = float(
                representative.get("group_representative_score") or 0.0
            )
            group["group_confidence_policy"] = PHASE4B_GROUP_CONFIDENCE_POLICY
            group["group_confidence_score"] = float(confidence)
        for row in members:
            if row is not representative:
                row["review_evidence_group_representative"] = False
                row["group_confidence_score"] = float(confidence)

    review_limit = min(
        int(policy["review_candidate_count"]),
        PHASE4B_RESCUE_REVIEW_MAX_CANDIDATES,
        len(representatives),
    )
    selected: list[dict[str, Any]] = []
    selected_candidate_ids: set[str] = set()

    def add(row: dict[str, Any] | None) -> None:
        if row is None or len(selected) >= review_limit:
            return
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in selected_candidate_ids:
            return
        selected.append(row)
        selected_candidate_ids.add(candidate_id)

    # Slot 1: strongest independent-tracklet agreement. This is exactly the
    # evidence R11 computed but failed to prioritize.
    corroborated_representatives = [
        row
        for row in representatives
        if row.get("corroborated_by_independent_tracklet") is True
    ]
    if corroborated_representatives:
        add(
            max(
                corroborated_representatives,
                key=lambda row: (
                    float(row.get("group_confidence_score") or 0.0),
                    float(row.get("group_representative_score") or 0.0),
                    float(row.get("stable_review_score") or 0.0),
                ),
            )
        )

    # Slot 2: strongest strict frozen-ReID candidate. Target context is not a
    # slot owner; it remains recorded only as soft evidence.
    strict_representatives = [
        row for row in representatives if row.get("strict_target_evidence_passed") is True
    ]
    if strict_representatives:
        add(
            max(
                strict_representatives,
                key=lambda row: (
                    float(row.get("stable_review_score") or 0.0),
                    float(row.get("retrieval_score") or 0.0),
                    float(row.get("prototype_target_similarity") or 0.0),
                ),
            )
        )

    # Remaining slots: strongest clean/pure group representatives.
    for row in sorted(
        representatives,
        key=lambda row: (
            float(row.get("stable_review_score") or 0.0),
            float(row.get("group_confidence_score") or 0.0),
            float(row.get("group_representative_score") or 0.0),
            float(row.get("retrieval_score") or 0.0),
        ),
        reverse=True,
    ):
        add(row)

    selected_ids = {str(row.get("candidate_id") or "") for row in selected}
    rescue_selected: list[dict[str, Any]] = []
    for review_rank, row in enumerate(selected, start=1):
        row["review_rank"] = review_rank
        if row.get("strict_target_evidence_passed") is True:
            row["review_path"] = "STRICT_TARGET_AND_PROVENANCE_NEGATIVE_GATE"
            row["rescue_review_required"] = False
            row["rescue_review_policy"] = None
        else:
            row["review_path"] = "GROUP_CONFIDENCE_ASSISTED_REVIEW"
            row["rescue_review_required"] = True
            row["rescue_review_policy"] = PHASE4B_RESCUE_REVIEW_POLICY
            row["rescue_review_evidence"] = {
                "policy": PHASE4B_RESCUE_REVIEW_POLICY,
                "reason": "PLAUSIBLE_TARGET_WITH_INDEPENDENT_GROUP_CONFIDENCE",
                "stable_review_score": float(row.get("stable_review_score") or 0.0),
                "group_representative_policy": PHASE4B_GROUP_REPRESENTATIVE_POLICY,
                "group_representative_score": float(
                    row.get("group_representative_score") or 0.0
                ),
                "group_confidence_policy": PHASE4B_GROUP_CONFIDENCE_POLICY,
                "group_confidence_score": float(
                    row.get("group_confidence_score") or 0.0
                ),
                "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
                "target_context_similarity": float(
                    row.get("target_context_similarity") or 0.0
                ),
                "target_context_is_soft_evidence_only": True,
                "corroboration_policy": PHASE4B_CORROBORATION_POLICY,
                "corroborated_by_independent_tracklet": bool(
                    row.get("corroborated_by_independent_tracklet")
                ),
                "corroboration_group_member_ids": list(
                    row.get("corroboration_group_member_ids") or []
                ),
                "detector_role_evidence_is_soft_only": False,
                "user_confirmed_role_gate_passed": True,
                "identity_negative_gate_passed": True,
                "automatic_target_confirmation": False,
            }
            rescue_selected.append(row)

    unselected = [
        row
        for row in representatives
        if str(row.get("candidate_id") or "") not in selected_ids
    ]
    catalog = sorted(
        representatives,
        key=lambda row: (
            float(row.get("group_confidence_score") or 0.0),
            float(row.get("stable_review_score") or 0.0),
            float(row.get("group_representative_score") or 0.0),
        ),
        reverse=True,
    )
    for catalog_rank, row in enumerate(catalog, start=1):
        row["review_catalog_policy"] = PHASE4B_REVIEW_CATALOG_POLICY
        row["review_catalog_rank"] = catalog_rank
        row["manual_review_promotion_allowed"] = True
        row["automatic_target_confirmation"] = False

    return (
        selected,
        rescue_selected,
        unselected,
        continuation_groups,
        corroboration_groups,
    )


def _phase4b_normalize_embedding_rows(matrix: Any, np: Any) -> Any:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise RuntimeError("Phase 4-B identity-negative embeddings must be a non-empty 2D matrix.")
    if not np.isfinite(value).all():
        raise RuntimeError("Phase 4-B identity-negative embeddings contain non-finite values.")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("Phase 4-B identity-negative embeddings contain a zero-norm row.")
    return (value / norms).astype(np.float32, copy=False)


def _phase4b_load_embedding_matrix(
    *,
    path_value: object,
    expected_sha256: object,
    label: str,
    np: Any,
) -> tuple[Path, Any]:
    path = Path(str(path_value or "")).resolve()
    expected = str(expected_sha256 or "")
    validate_input_file(path, expected or None, label)
    try:
        with path.open("rb") as stream:
            matrix = np.load(stream, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot load {label}: {path}: {exc}") from exc
    return path, _phase4b_normalize_embedding_rows(matrix, np)


def _phase4b_deduplicate_negative_rows(
    matrix: Any,
    np: Any,
    *,
    duplicate_cosine: float,
    label: str,
) -> tuple[Any, list[int]]:
    normalized = _phase4b_normalize_embedding_rows(matrix, np)
    kept_rows: list[Any] = []
    kept_indices: list[int] = []
    for index, row in enumerate(normalized):
        if kept_rows:
            gallery = np.stack(kept_rows).astype(np.float32)
            if float(np.max(gallery @ row)) >= float(duplicate_cosine):
                continue
        kept_rows.append(row)
        kept_indices.append(index)
    if not kept_rows:
        raise RuntimeError(f"{label} produced no unique negative embeddings.")
    return np.stack(kept_rows).astype(np.float32), kept_indices


def _phase4b_deduplicate_identity_negative_rows(matrix: Any, np: Any) -> tuple[Any, list[int]]:
    return _phase4b_deduplicate_negative_rows(
        matrix,
        np,
        duplicate_cosine=PHASE4B_IDENTITY_NEGATIVE_DUPLICATE_COSINE,
        label="NONE_OF_THESE",
    )


def _phase4b_pending_ambiguity(
    state: Mapping[str, Any],
    *,
    ambiguity_id: str,
) -> dict[str, Any]:
    pending = state.get("pending_action")
    if not isinstance(pending, Mapping) or pending.get("type") != "CROSS_SHOT_CONFIRMATION":
        raise RuntimeError("NONE_OF_THESE requires a pending cross-shot confirmation.")
    if str(pending.get("ambiguity_id") or "") != ambiguity_id:
        raise RuntimeError("NONE_OF_THESE ambiguity_id does not match the pending action.")
    ambiguity = next(
        (
            dict(row)
            for row in state.get("ambiguities") or []
            if isinstance(row, Mapping)
            and str(row.get("ambiguity_id") or "") == ambiguity_id
        ),
        None,
    )
    if ambiguity is None:
        raise RuntimeError("Pending Phase 4-B ambiguity is missing from pipeline_state.json.")
    candidates = [
        dict(row)
        for row in ambiguity.get("review_candidates") or []
        if isinstance(row, Mapping)
        and str(row.get("status") or "PENDING") == "PENDING"
    ]
    if not candidates:
        raise RuntimeError("NONE_OF_THESE requires at least one pending review candidate.")
    ambiguity["review_candidates"] = candidates
    return ambiguity


def _phase4b_apply_none_of_these_review(
    *,
    output_dir: Path,
    state: dict[str, Any],
    ambiguity_id: str,
    reviewer: str,
    note: str,
    decision: str = "NONE_OF_THESE",
) -> dict[str, Any]:
    """Persist user-confirmed identity negatives and resume at the next shot.

    This decision means only that every surfaced candidate is a different person.
    It does not assert that the target is absent from the reviewed shot.
    """

    import numpy as np

    ambiguity = _phase4b_pending_ambiguity(state, ambiguity_id=ambiguity_id)
    candidates = list(ambiguity["review_candidates"])
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise RuntimeError("NONE_OF_THESE candidate list contains an empty candidate_id.")

    runtime = dict(state.get("runtime") or {})
    previous = dict(runtime.get("phase4b_identity_negative_memory") or {})
    matrices: list[Any] = []
    references: list[dict[str, Any]] = []
    previous_embedding_count = 0
    previous_revision_id = None
    previous_manifest_sha = None

    if previous.get("embeddings_path"):
        _, previous_matrix = _phase4b_load_embedding_matrix(
            path_value=previous.get("embeddings_path"),
            expected_sha256=previous.get("embeddings_sha256"),
            label="previous Phase 4-B identity-negative embeddings",
            np=np,
        )
        matrices.append(previous_matrix)
        previous_embedding_count = int(previous_matrix.shape[0])
        previous_revision_id = str(previous.get("revision_id") or "") or None
        previous_manifest_path = Path(str(previous.get("manifest_path") or "")).resolve()
        previous_manifest_sha = str(previous.get("manifest_sha256") or "") or None
        if previous_manifest_path.is_file():
            validate_input_file(
                previous_manifest_path,
                previous_manifest_sha,
                "previous Phase 4-B identity-negative manifest",
            )
            prior_document = read_object(previous_manifest_path)
            references.extend(
                dict(row)
                for row in prior_document.get("references") or []
                if isinstance(row, Mapping)
            )

    new_reference_rows: list[dict[str, Any]] = []
    manifest_fingerprints: list[dict[str, str]] = []
    embedding_dimension: int | None = None
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        manifest_path = _phase4b_resolve_output_artifact(
            output_dir=output_dir,
            path_value=candidate.get("manifest_path"),
            label=f"candidate manifest {candidate_id}",
        )
        manifest_sha = validate_input_file(
            manifest_path,
            str(candidate.get("manifest_sha256") or "") or None,
            f"NONE_OF_THESE candidate manifest {candidate_id}",
        )
        manifest = read_object(manifest_path)
        if str(manifest.get("candidate_id") or "") != candidate_id:
            raise RuntimeError("NONE_OF_THESE candidate manifest identity mismatch.")
        quality = dict(manifest.get("quality") or {})
        if "identity_pure" in quality and quality.get("identity_pure") is not True:
            raise RuntimeError(
                f"Identity-impure candidate cannot enter negative memory: {candidate_id}"
            )
        embeddings = manifest.get("candidate_embeddings")
        if not isinstance(embeddings, Mapping):
            raise RuntimeError(f"Candidate {candidate_id} has no immutable embeddings contract.")
        embeddings_path, matrix = _phase4b_load_embedding_matrix(
            path_value=embeddings.get("path"),
            expected_sha256=embeddings.get("sha256"),
            label=f"NONE_OF_THESE candidate embeddings {candidate_id}",
            np=np,
        )
        if embedding_dimension is None:
            embedding_dimension = int(matrix.shape[1])
        elif int(matrix.shape[1]) != embedding_dimension:
            raise RuntimeError("NONE_OF_THESE candidate embedding dimensions are inconsistent.")
        matrices.append(matrix)
        prototype_path = Path(str(embeddings.get("prototype_path") or "")).resolve()
        prototype_sha = validate_input_file(
            prototype_path,
            str(embeddings.get("prototype_sha256") or "") or None,
            f"NONE_OF_THESE candidate prototype {candidate_id}",
        )
        reference = {
            "candidate_id": candidate_id,
            "shot_id": str(ambiguity.get("shot_id") or candidate.get("shot_id") or ""),
            "ambiguity_id": ambiguity_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "embeddings_path": str(embeddings_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "embedding_count": int(matrix.shape[0]),
            "prototype_path": str(prototype_path),
            "prototype_sha256": prototype_sha,
            "review_decision": decision,
            "reviewer": reviewer,
            "review_note": note,
        }
        references.append(reference)
        new_reference_rows.append(reference)
        manifest_fingerprints.append(
            {"candidate_id": candidate_id, "manifest_sha256": manifest_sha}
        )

    if not matrices:
        raise RuntimeError("NONE_OF_THESE produced no candidate embeddings.")
    dimension_set = {int(matrix.shape[1]) for matrix in matrices}
    if len(dimension_set) != 1:
        raise RuntimeError("Previous and new identity-negative dimensions do not match.")
    combined = np.concatenate(matrices, axis=0).astype(np.float32)
    deduplicated, kept_indices = _phase4b_deduplicate_identity_negative_rows(combined, np)

    seed_payload = {
        "policy": PHASE4B_IDENTITY_NEGATIVE_MEMORY_POLICY,
        "previous_revision_id": previous_revision_id,
        "previous_manifest_sha256": previous_manifest_sha,
        "ambiguity_id": ambiguity_id,
        "candidate_manifests": sorted(
            manifest_fingerprints,
            key=lambda row: (row["candidate_id"], row["manifest_sha256"]),
        ),
    }
    seed = hashlib.sha256(
        json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    revision_id = f"ecneg_identity_{seed}"
    negative_dir = output_dir / "identity_negative_memory"
    negative_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = negative_dir / f"{revision_id}.embeddings.npy"
    manifest_path = negative_dir / f"{revision_id}.json"
    with embeddings_path.open("wb") as stream:
        np.save(stream, deduplicated, allow_pickle=False)
    embeddings_sha = sha256_file(embeddings_path)
    document = {
        "schema_version": PHASE4B_IDENTITY_NEGATIVE_MEMORY_SCHEMA,
        "immutable": True,
        "policy": PHASE4B_IDENTITY_NEGATIVE_MEMORY_POLICY,
        "revision_id": revision_id,
        "previous_revision_id": previous_revision_id,
        "previous_manifest_sha256": previous_manifest_sha,
        "source_ambiguity_id": ambiguity_id,
        "source_shot_id": str(ambiguity.get("shot_id") or ""),
        "decision": decision,
        "rejected_candidate_ids": candidate_ids,
        "reviewer": reviewer,
        "review_note": note,
        "created_at": now_iso(),
        "previous_embedding_count": previous_embedding_count,
        "new_candidate_embedding_count": sum(
            int(row["embedding_count"]) for row in new_reference_rows
        ),
        "combined_embedding_count_before_deduplication": int(combined.shape[0]),
        "embedding_count": int(deduplicated.shape[0]),
        "deduplicated_embedding_count": int(combined.shape[0] - deduplicated.shape[0]),
        "embedding_dimension": int(deduplicated.shape[1]),
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": embeddings_sha,
        "kept_source_row_indices": kept_indices,
        "references": references,
        "user_confirmed_only": True,
        "target_positive_memory_modified": False,
        "automatic_target_confirmation": False,
    }
    atomic_json(manifest_path, document)
    manifest_sha = sha256_file(manifest_path)
    identity_memory = {
        "policy": PHASE4B_IDENTITY_NEGATIVE_MEMORY_POLICY,
        "revision_id": revision_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": embeddings_sha,
        "embedding_count": int(deduplicated.shape[0]),
        "embedding_dimension": int(deduplicated.shape[1]),
        "rejected_candidate_ids": candidate_ids,
        "source_ambiguity_id": ambiguity_id,
        "source_shot_id": str(ambiguity.get("shot_id") or ""),
        "user_confirmed_only": True,
        "automatic_target_confirmation": False,
    }

    updated_ambiguities: list[dict[str, Any]] = []
    for raw in state.get("ambiguities") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if str(row.get("ambiguity_id") or "") == ambiguity_id:
            row["status"] = (
                "RESOLVED_REJECTED_BATCH"
                if decision == "DIFFERENT_PLAYER"
                else "RESOLVED_NONE_OF_THESE"
            )
            row["resolved_at"] = now_iso()
            row["review_decision"] = decision
            row["identity_negative_memory_revision_id"] = revision_id
            updated_candidates = []
            for candidate in row.get("review_candidates") or []:
                if isinstance(candidate, Mapping):
                    item = dict(candidate)
                    item["status"] = "EXCLUDED_BY_USER_NONE_OF_THESE"
                    updated_candidates.append(item)
            row["review_candidates"] = updated_candidates
        updated_ambiguities.append(row)
    state["ambiguities"] = updated_ambiguities

    shot_id = str(ambiguity.get("shot_id") or "")
    for shot in state.get("shots") or []:
        if isinstance(shot, dict) and str(shot.get("shot_id") or "") == shot_id:
            shot["status"] = PHASE4B_NONE_OF_THESE_SHOT_STATUS
            shot["phase4b_none_of_these_confirmed"] = True
            shot["identity_negative_memory_revision_id"] = revision_id

    shot_results = state.setdefault("shot_search_results", {})
    existing_result = dict(shot_results.get(shot_id) or {})
    existing_result.update(
        {
            "status": "USER_REJECTED_NONE_OF_THESE",
            "review_decision": decision,
            "rejected_candidate_ids": candidate_ids,
            "identity_negative_memory": identity_memory,
        }
    )
    shot_results[shot_id] = existing_result

    confirmations = [
        dict(row) for row in state.get("confirmations") or [] if isinstance(row, Mapping)
    ]
    confirmations.append(
        {
            "ambiguity_id": ambiguity_id,
            "shot_id": shot_id,
            "decision": decision,
            "rejected_candidate_ids": candidate_ids,
            "reviewer": reviewer,
            "review_note": note,
            "reviewed_at": now_iso(),
            "identity_negative_memory_revision_id": revision_id,
            "identity_negative_memory_manifest_sha256": manifest_sha,
            "automatic_target_confirmation": False,
        }
    )
    state["confirmations"] = confirmations
    runtime["phase4b_identity_negative_memory"] = identity_memory
    runtime["phase4b_none_of_these_last_ambiguity_id"] = ambiguity_id
    runtime["phase4b_none_of_these_last_candidate_ids"] = candidate_ids
    runtime["phase4b_next_shot_id"] = next(
        (
            str(shot.get("shot_id"))
            for shot in state.get("shots") or []
            if isinstance(shot, Mapping)
            and str(shot.get("status") or "").upper() == "SEARCHING_MEMORY_READY"
        ),
        None,
    )
    runtime["automatic_target_confirmation"] = False
    state["runtime"] = runtime
    state["pending_action"] = None
    if runtime["phase4b_next_shot_id"]:
        state["status"] = "RUNNING"
        state["decision"] = PHASE4B_CONTINUE_DECISION
    else:
        state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
        state["decision"] = "COMPLETE_PHASE4B_WITH_UNRESOLVED_GAPS"
    state["updated_at"] = now_iso()
    return identity_memory


def _phase4b_reviewed_candidates_from_decisions(
    *,
    output_dir: Path,
    current_decision_path: Path,
    current_decision_sha256: str,
    expected_candidate_id: str,
    expected_state: str,
) -> tuple[list[dict[str, Any]], str]:
    validate_input_file(
        current_decision_path,
        current_decision_sha256,
        "R14 review decision",
    )
    current = read_object(current_decision_path)
    if (
        str(current.get("candidate_id") or "") != expected_candidate_id
        or str(current.get("state") or "") != expected_state
        or not current.get("ambiguity_id")
    ):
        raise RuntimeError("R14 review decision does not match resume arguments.")

    current_ambiguity_id = str(current["ambiguity_id"])
    current_ambiguity_path = output_dir / "ambiguities" / f"{current_ambiguity_id}.json"
    if not current_ambiguity_path.is_file():
        raise RuntimeError(
            f"R14 ambiguity provenance artifact is missing: {current_ambiguity_id}"
        )
    current_ambiguity = read_object(current_ambiguity_path)
    shot_id = str(current_ambiguity.get("shot_id") or "")
    if not shot_id:
        raise RuntimeError("R14 current review decision has no shot provenance.")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted((output_dir / "review_decisions").glob("*.json")):
        decision = read_object(path)
        if str(decision.get("state") or "") != "DIFFERENT_PLAYER":
            continue
        candidate_id = str(decision.get("candidate_id") or "")
        ambiguity_id = str(decision.get("ambiguity_id") or "")
        if not candidate_id or not ambiguity_id or candidate_id in seen:
            continue
        ambiguity_path = output_dir / "ambiguities" / f"{ambiguity_id}.json"
        if not ambiguity_path.is_file():
            raise RuntimeError(
                f"R14 ambiguity provenance artifact is missing: {ambiguity_id}"
            )
        ambiguity = read_object(ambiguity_path)
        if str(ambiguity.get("ambiguity_id") or "") != ambiguity_id:
            raise RuntimeError("R14 ambiguity provenance identity mismatch.")
        raw_candidates = (
            ambiguity.get("candidates")
            or ambiguity.get("review_candidates")
            or []
        )
        candidate = next(
            (
                dict(row)
                for row in raw_candidates
                if isinstance(row, Mapping)
                and str(row.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(
                f"R14 decision candidate is absent from ambiguity provenance: {candidate_id}"
            )
        candidate_shot_id = str(
            ambiguity.get("shot_id") or candidate.get("shot_id") or ""
        )
        if not candidate_shot_id:
            raise RuntimeError("R14 rejected candidate has no reviewed shot provenance.")
        if candidate_shot_id != shot_id:
            continue
        candidate["status"] = "PENDING"
        candidates.append(candidate)
        seen.add(candidate_id)
    return candidates, shot_id


def _phase4b_resolve_output_artifact(
    *,
    output_dir: Path,
    path_value: object,
    label: str,
) -> Path:
    """Resolve a backend-storage artifact without trusting the adapter cwd.

    R14 ambiguity artifacts may contain project-relative ``storage/...`` paths,
    while the frozen adapter runs with the AI source tree as its project root.
    The job output directory is the provenance boundary shared by both sides.
    """

    raw = Path(str(path_value or ""))
    if raw.is_absolute():
        return raw.resolve()
    resolved_output = output_dir.resolve()
    parts = raw.parts
    try:
        job_segment = parts.index(resolved_output.name)
    except ValueError:
        candidate = resolved_output.joinpath(*parts)
    else:
        candidate = resolved_output.joinpath(*parts[job_segment + 1 :])
    candidate = candidate.resolve()
    try:
        candidate.relative_to(resolved_output)
    except ValueError as exc:
        raise RuntimeError(f"R14 {label} escapes the tracking output directory.") from exc
    return candidate


def _phase4b_resume_rejected_candidate(
    *,
    output_dir: Path,
    state: dict[str, Any],
    candidate_id: str,
    decision_state: str,
    decision_path: Path,
    decision_sha256: str,
    reviewer: str,
    note: str,
) -> dict[str, Any] | None:
    _phase4b_restore_memory_ready_shots(state)
    positive_before = (
        str(dict(state.get("runtime") or {}).get("memory_revision_sha256") or ""),
        str(dict(state.get("runtime") or {}).get("memory_revision_path") or ""),
    )
    rejected, shot_id = _phase4b_reviewed_candidates_from_decisions(
        output_dir=output_dir,
        current_decision_path=decision_path,
        current_decision_sha256=decision_sha256,
        expected_candidate_id=candidate_id,
        expected_state=decision_state,
    )
    current_decision = read_object(decision_path)
    current_ambiguity_id = str(current_decision["ambiguity_id"])
    if not shot_id:
        ambiguity = read_object(
            output_dir / "ambiguities" / f"{current_ambiguity_id}.json"
        )
        shot_id = str(ambiguity.get("shot_id") or "")
    if not shot_id:
        raise RuntimeError("R14 exhausted review batch has no shot provenance.")

    identity_memory: dict[str, Any] | None = None
    if rejected:
        synthetic_id = f"r14_rejected_{_safe_name(shot_id)}"
        synthetic = {
            "ambiguity_id": synthetic_id,
            "shot_id": shot_id,
            "status": "PENDING",
            "review_candidates": rejected,
            "automatic_target_confirmation": False,
        }
        state["ambiguities"] = [
            row
            for row in state.get("ambiguities") or []
            if not (
                isinstance(row, Mapping)
                and str(row.get("ambiguity_id") or "") == synthetic_id
            )
        ] + [synthetic]
        state["pending_action"] = {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": synthetic_id,
            "shot_id": shot_id,
            "candidate_ids": [row["candidate_id"] for row in rejected],
            "automatic_target_confirmation": False,
        }
        identity_memory = _phase4b_apply_none_of_these_review(
            output_dir=output_dir,
            state=state,
            ambiguity_id=synthetic_id,
            reviewer=reviewer,
            note=note,
            decision="DIFFERENT_PLAYER",
        )
    else:
        for shot in state.get("shots") or []:
            if isinstance(shot, dict) and str(shot.get("shot_id") or "") == shot_id:
                shot["status"] = "UNRESOLVED_LOW_RESOLUTION"
        runtime = dict(state.get("runtime") or {})
        runtime["phase4b_next_shot_id"] = next(
            (
                str(shot.get("shot_id"))
                for shot in state.get("shots") or []
                if isinstance(shot, Mapping)
                and str(shot.get("status") or "").upper()
                == "SEARCHING_MEMORY_READY"
            ),
            None,
        )
        state["runtime"] = runtime
        state["pending_action"] = None
        if runtime["phase4b_next_shot_id"]:
            state["status"] = "RUNNING"
            state["decision"] = PHASE4B_CONTINUE_DECISION
        else:
            state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
            state["decision"] = "COMPLETE_PHASE4B_WITH_UNRESOLVED_GAPS"

    positive_after = (
        str(dict(state.get("runtime") or {}).get("memory_revision_sha256") or ""),
        str(dict(state.get("runtime") or {}).get("memory_revision_path") or ""),
    )
    if positive_after != positive_before:
        raise RuntimeError("R14 rejection resume changed positive target memory.")
    state["updated_at"] = now_iso()
    return identity_memory


def _phase4b_restore_memory_ready_shots(state: dict[str, Any]) -> None:
    """Restore reviewed-shot search markers when durable positive memory exists."""

    runtime = dict(state.get("runtime") or {})
    positive_path = Path(str(runtime.get("memory_revision_path") or "")).resolve()
    positive_sha = str(runtime.get("memory_revision_sha256") or "")
    if not positive_path.is_file() or not positive_sha:
        return
    validate_input_file(
        positive_path,
        positive_sha,
        "R14 positive target memory",
    )
    for shot in state.get("shots") or []:
        if (
            isinstance(shot, dict)
            and str(shot.get("status") or "").upper() == "SEARCHING_NO_MEMORY"
        ):
            shot["status"] = "SEARCHING_MEMORY_READY"


def _phase4b_restore_authorization_from_report(
    *,
    output_dir: Path,
    state: dict[str, Any],
) -> None:
    """Recover Phase 4-B authorization only from its prior PASS report."""

    report_path = output_dir / "phase4b_first_cross_shot_report.json"
    if not report_path.is_file():
        return
    report = read_object(report_path)
    runtime = dict(state.get("runtime") or {})
    if (
        report.get("status") != "PASS"
        or report.get("policy") != PHASE4B_POLICY
        or report.get("automatic_target_confirmation") is not False
        or str(report.get("memory_revision_sha256") or "")
        != str(runtime.get("memory_revision_sha256") or "")
    ):
        return
    runtime["phase4a_initial_memory_review_status"] = "PASS"
    runtime["phase4b_cross_shot_scoring_authorized"] = True
    runtime["phase4b_authorization_recovered_from_report"] = str(report_path)
    state["runtime"] = runtime


def _phase4b_historical_detector_role_negative_sources(
    *,
    state: Mapping[str, Any],
    np: Any,
    expected_dimension: int | None,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load trusted detector-labeled staff/referee galleries from earlier shots."""

    matrices: list[Any] = []
    source_records: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    shot_results = state.get("shot_search_results")
    if not isinstance(shot_results, Mapping):
        return matrices, source_records, references

    for shot_id in sorted(str(value) for value in shot_results):
        raw_result = shot_results.get(shot_id)
        if not isinstance(raw_result, Mapping):
            continue
        memory = raw_result.get("negative_role_memory")
        if not isinstance(memory, Mapping) or memory.get("negative_memory_available") is not True:
            continue
        if str(memory.get("policy") or "") != PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY:
            raise RuntimeError("Historical role-negative policy mismatch.")
        embeddings_path, matrix = _phase4b_load_embedding_matrix(
            path_value=memory.get("embeddings_path"),
            expected_sha256=memory.get("embeddings_sha256"),
            label=f"historical detector role-negative embeddings {shot_id}",
            np=np,
        )
        if expected_dimension is not None and int(matrix.shape[1]) != int(expected_dimension):
            raise RuntimeError("Historical role-negative dimension mismatch.")
        manifest_path = Path(str(memory.get("manifest_path") or "")).resolve()
        manifest_sha = validate_input_file(
            manifest_path,
            str(memory.get("manifest_sha256") or "") or None,
            f"historical detector role-negative manifest {shot_id}",
        )
        manifest = read_object(manifest_path)
        if str(manifest.get("policy") or "") != PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY:
            raise RuntimeError("Historical role-negative manifest policy mismatch.")
        if manifest.get("automatic_target_confirmation") is not False:
            raise RuntimeError("Historical role-negative memory cannot confirm a target.")
        matrices.append(matrix)
        source_records.append(
            {
                "source_kind": "DETECTOR_LABELED_STAFF_REFEREE_GALLERY",
                "source_shot_id": shot_id,
                "policy": PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
                "embeddings_path": str(embeddings_path),
                "embeddings_sha256": sha256_file(embeddings_path),
                "embedding_count": int(matrix.shape[0]),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
            }
        )
        references.extend(
            {
                **dict(row),
                "source_kind": "DETECTOR_LABELED_STAFF_REFEREE",
                "source_shot_id": shot_id,
            }
            for row in manifest.get("references") or []
            if isinstance(row, Mapping)
        )
    return matrices, source_records, references


def _phase4b_apply_non_player_role_review(
    *,
    output_dir: Path,
    state: dict[str, Any],
    ambiguity_id: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Persist user-confirmed non-player role negatives and continue search.

    This is distinct from NONE_OF_THESE: the surfaced candidates are explicitly
    identified as coach/staff/referee/non-player roles, not merely other players.
    Positive target memory and identity-negative memory are left unchanged.
    """

    import numpy as np

    ambiguity = _phase4b_pending_ambiguity(state, ambiguity_id=ambiguity_id)
    candidates = list(ambiguity["review_candidates"])
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise RuntimeError("NON_PLAYER_ROLE candidate list contains an empty candidate_id.")

    runtime = dict(state.get("runtime") or {})
    previous = dict(runtime.get("phase4b_persistent_role_negative_memory") or {})
    matrices: list[Any] = []
    references: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    previous_embedding_count = 0
    previous_revision_id: str | None = None
    previous_manifest_sha: str | None = None
    embedding_dimension: int | None = None

    if previous.get("embeddings_path"):
        _, previous_matrix = _phase4b_load_embedding_matrix(
            path_value=previous.get("embeddings_path"),
            expected_sha256=previous.get("embeddings_sha256"),
            label="previous persistent role-negative embeddings",
            np=np,
        )
        matrices.append(previous_matrix)
        embedding_dimension = int(previous_matrix.shape[1])
        previous_embedding_count = int(previous_matrix.shape[0])
        previous_revision_id = str(previous.get("revision_id") or "") or None
        previous_manifest_path = Path(str(previous.get("manifest_path") or "")).resolve()
        previous_manifest_sha = str(previous.get("manifest_sha256") or "") or None
        validate_input_file(
            previous_manifest_path,
            previous_manifest_sha,
            "previous persistent role-negative manifest",
        )
        previous_document = read_object(previous_manifest_path)
        if str(previous_document.get("policy") or "") != PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY:
            raise RuntimeError("Previous persistent role-negative policy mismatch.")
        references.extend(
            dict(row)
            for row in previous_document.get("references") or []
            if isinstance(row, Mapping)
        )
        source_records.extend(
            dict(row)
            for row in previous_document.get("sources") or []
            if isinstance(row, Mapping)
        )
    else:
        historical_matrices, historical_sources, historical_references = (
            _phase4b_historical_detector_role_negative_sources(
                state=state,
                np=np,
                expected_dimension=None,
            )
        )
        if historical_matrices:
            embedding_dimension = int(historical_matrices[0].shape[1])
            if any(int(matrix.shape[1]) != embedding_dimension for matrix in historical_matrices):
                raise RuntimeError("Historical detector role-negative dimensions are inconsistent.")
            matrices.extend(historical_matrices)
            source_records.extend(historical_sources)
            references.extend(historical_references)

    new_reference_rows: list[dict[str, Any]] = []
    candidate_fingerprints: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        manifest_path = _phase4b_resolve_output_artifact(
            output_dir=output_dir,
            path_value=candidate.get("manifest_path"),
            label=f"candidate manifest {candidate_id}",
        )
        manifest_sha = validate_input_file(
            manifest_path,
            str(candidate.get("manifest_sha256") or "") or None,
            f"NON_PLAYER_ROLE candidate manifest {candidate_id}",
        )
        manifest = read_object(manifest_path)
        if str(manifest.get("candidate_id") or "") != candidate_id:
            raise RuntimeError("NON_PLAYER_ROLE candidate manifest identity mismatch.")
        embeddings = manifest.get("candidate_embeddings")
        if not isinstance(embeddings, Mapping):
            raise RuntimeError(f"Candidate {candidate_id} has no immutable embeddings contract.")
        embeddings_path, matrix = _phase4b_load_embedding_matrix(
            path_value=embeddings.get("path"),
            expected_sha256=embeddings.get("sha256"),
            label=f"NON_PLAYER_ROLE candidate embeddings {candidate_id}",
            np=np,
        )
        if embedding_dimension is None:
            embedding_dimension = int(matrix.shape[1])
        elif int(matrix.shape[1]) != embedding_dimension:
            raise RuntimeError("Persistent role-negative embedding dimensions are inconsistent.")
        matrices.append(matrix)
        prototype_path = Path(str(embeddings.get("prototype_path") or "")).resolve()
        prototype_sha = validate_input_file(
            prototype_path,
            str(embeddings.get("prototype_sha256") or "") or None,
            f"NON_PLAYER_ROLE candidate prototype {candidate_id}",
        )
        reference = {
            "source_kind": "USER_CONFIRMED_NON_PLAYER_ROLE_CANDIDATE",
            "candidate_id": candidate_id,
            "shot_id": str(ambiguity.get("shot_id") or candidate.get("shot_id") or ""),
            "ambiguity_id": ambiguity_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "embeddings_path": str(embeddings_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "embedding_count": int(matrix.shape[0]),
            "prototype_path": str(prototype_path),
            "prototype_sha256": prototype_sha,
            "review_decision": "NONE_OF_THESE_NON_PLAYER_ROLE",
            "reviewer": reviewer,
            "review_note": note,
            "user_confirmed": True,
        }
        references.append(reference)
        new_reference_rows.append(reference)
        candidate_fingerprints.append(
            {"candidate_id": candidate_id, "manifest_sha256": manifest_sha}
        )

    if not matrices:
        raise RuntimeError("NON_PLAYER_ROLE produced no role-negative embeddings.")
    dimension_set = {int(matrix.shape[1]) for matrix in matrices}
    if len(dimension_set) != 1:
        raise RuntimeError("Persistent role-negative source dimensions do not match.")
    combined = np.concatenate(matrices, axis=0).astype(np.float32)
    deduplicated, kept_indices = _phase4b_deduplicate_negative_rows(
        combined,
        np,
        duplicate_cosine=PHASE4B_PERSISTENT_ROLE_NEGATIVE_DUPLICATE_COSINE,
        label="NON_PLAYER_ROLE",
    )

    seed_payload = {
        "policy": PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY,
        "previous_revision_id": previous_revision_id,
        "previous_manifest_sha256": previous_manifest_sha,
        "ambiguity_id": ambiguity_id,
        "candidate_manifests": sorted(
            candidate_fingerprints,
            key=lambda row: (row["candidate_id"], row["manifest_sha256"]),
        ),
        "bootstrap_sources": sorted(
            str(row.get("manifest_sha256") or "") for row in source_records
        ),
    }
    seed = hashlib.sha256(
        json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    revision_id = f"ecneg_role_{seed}"
    memory_dir = output_dir / "persistent_role_negative_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = memory_dir / f"{revision_id}.embeddings.npy"
    manifest_path = memory_dir / f"{revision_id}.json"
    with embeddings_path.open("wb") as stream:
        np.save(stream, deduplicated, allow_pickle=False)
    embeddings_sha = sha256_file(embeddings_path)
    document = {
        "schema_version": PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_SCHEMA,
        "immutable": True,
        "policy": PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY,
        "revision_id": revision_id,
        "previous_revision_id": previous_revision_id,
        "previous_manifest_sha256": previous_manifest_sha,
        "source_ambiguity_id": ambiguity_id,
        "source_shot_id": str(ambiguity.get("shot_id") or ""),
        "decision": "NONE_OF_THESE_NON_PLAYER_ROLE",
        "rejected_candidate_ids": candidate_ids,
        "reviewer": reviewer,
        "review_note": note,
        "created_at": now_iso(),
        "previous_embedding_count": previous_embedding_count,
        "historical_detector_source_count": len(source_records),
        "new_candidate_embedding_count": sum(
            int(row["embedding_count"]) for row in new_reference_rows
        ),
        "combined_embedding_count_before_deduplication": int(combined.shape[0]),
        "embedding_count": int(deduplicated.shape[0]),
        "deduplicated_embedding_count": int(combined.shape[0] - deduplicated.shape[0]),
        "embedding_dimension": int(deduplicated.shape[1]),
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": embeddings_sha,
        "kept_source_row_indices": kept_indices,
        "sources": source_records,
        "references": references,
        "user_confirmed_candidate_roles_only": True,
        "detector_labeled_role_sources_allowed": True,
        "identity_negative_memory_modified": False,
        "target_positive_memory_modified": False,
        "automatic_target_confirmation": False,
    }
    atomic_json(manifest_path, document)
    manifest_sha = sha256_file(manifest_path)
    persistent_memory = {
        "policy": PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY,
        "revision_id": revision_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": embeddings_sha,
        "embedding_count": int(deduplicated.shape[0]),
        "embedding_dimension": int(deduplicated.shape[1]),
        "rejected_candidate_ids": candidate_ids,
        "source_ambiguity_id": ambiguity_id,
        "source_shot_id": str(ambiguity.get("shot_id") or ""),
        "user_confirmed_non_player_role": True,
        "automatic_target_confirmation": False,
    }

    updated_ambiguities: list[dict[str, Any]] = []
    for raw in state.get("ambiguities") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if str(row.get("ambiguity_id") or "") == ambiguity_id:
            row["status"] = "RESOLVED_NON_PLAYER_ROLE"
            row["resolved_at"] = now_iso()
            row["review_decision"] = "NONE_OF_THESE_NON_PLAYER_ROLE"
            row["persistent_role_negative_memory_revision_id"] = revision_id
            updated_candidates = []
            for candidate in row.get("review_candidates") or []:
                if isinstance(candidate, Mapping):
                    item = dict(candidate)
                    item["status"] = "EXCLUDED_BY_USER_NON_PLAYER_ROLE"
                    updated_candidates.append(item)
            row["review_candidates"] = updated_candidates
        updated_ambiguities.append(row)
    state["ambiguities"] = updated_ambiguities

    shot_id = str(ambiguity.get("shot_id") or "")
    for shot in state.get("shots") or []:
        if isinstance(shot, dict) and str(shot.get("shot_id") or "") == shot_id:
            shot["status"] = PHASE4B_NON_PLAYER_ROLE_SHOT_STATUS
            shot["phase4b_non_player_role_confirmed"] = True
            shot["persistent_role_negative_memory_revision_id"] = revision_id

    shot_results = state.setdefault("shot_search_results", {})
    existing_result = dict(shot_results.get(shot_id) or {})
    existing_result.update(
        {
            "status": "USER_REJECTED_NON_PLAYER_ROLE",
            "review_decision": "NONE_OF_THESE_NON_PLAYER_ROLE",
            "rejected_candidate_ids": candidate_ids,
            "persistent_role_negative_memory": persistent_memory,
        }
    )
    shot_results[shot_id] = existing_result

    confirmations = [
        dict(row) for row in state.get("confirmations") or [] if isinstance(row, Mapping)
    ]
    confirmations.append(
        {
            "ambiguity_id": ambiguity_id,
            "shot_id": shot_id,
            "decision": "NONE_OF_THESE_NON_PLAYER_ROLE",
            "rejected_candidate_ids": candidate_ids,
            "reviewer": reviewer,
            "review_note": note,
            "reviewed_at": now_iso(),
            "persistent_role_negative_memory_revision_id": revision_id,
            "persistent_role_negative_memory_manifest_sha256": manifest_sha,
            "automatic_target_confirmation": False,
        }
    )
    state["confirmations"] = confirmations
    runtime["phase4b_persistent_role_negative_memory"] = persistent_memory
    runtime["phase4b_non_player_role_last_ambiguity_id"] = ambiguity_id
    runtime["phase4b_non_player_role_last_candidate_ids"] = candidate_ids
    runtime["phase4b_next_shot_id"] = next(
        (
            str(shot.get("shot_id"))
            for shot in state.get("shots") or []
            if isinstance(shot, Mapping)
            and str(shot.get("status") or "").upper() == "SEARCHING_MEMORY_READY"
        ),
        None,
    )
    runtime["automatic_target_confirmation"] = False
    state["runtime"] = runtime
    state["pending_action"] = None
    state["status"] = "RUNNING"
    state["decision"] = PHASE4B_CONTINUE_DECISION
    state["updated_at"] = now_iso()
    return persistent_memory


def _phase4b_invalidate_stale_candidate_review(state: dict[str, Any]) -> bool:
    """Invalidate only a prior Phase 4-B review produced by an older policy."""

    pending = state.get("pending_action")
    runtime = dict(state.get("runtime") or {})
    if not isinstance(pending, Mapping):
        return False
    if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
        return False
    if runtime.get("phase4b_policy") == PHASE4B_POLICY:
        return False

    ambiguity_id = str(pending.get("ambiguity_id") or "")
    shot_id = str(pending.get("shot_id") or "")
    state["ambiguities"] = [
        row
        for row in state.get("ambiguities") or []
        if not (
            isinstance(row, Mapping)
            and (
                str(row.get("ambiguity_id") or "") == ambiguity_id
                or str(row.get("shot_id") or "") == shot_id
            )
        )
    ]
    for shot in state.get("shots") or []:
        if isinstance(shot, dict) and str(shot.get("shot_id") or "") == shot_id:
            shot["status"] = "SEARCHING_MEMORY_READY"

    state["pending_action"] = None
    state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
    state["decision"] = "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING"
    runtime.update(
        {
            "phase4b_first_cross_shot_scoring_complete": False,
            "phase4b_stale_review_invalidated": True,
            "phase4b_stale_review_policy": runtime.get("phase4b_policy"),
            "phase4b_policy": PHASE4B_POLICY,
            "automatic_target_confirmation": False,
        }
    )
    state["runtime"] = runtime
    return True


def _phase4b_reopen_stale_terminal_results_for_rescoring(
    state: dict[str, Any],
) -> list[str]:
    """Reopen terminal R7/R8/R9/R10 attempts for R11 rescoring.

    Explicit user decisions such as NONE_OF_THESE and NON_PLAYER_ROLE are never
    reopened. Detector caches and shot boundaries remain untouched.
    """

    runtime = dict(state.get("runtime") or {})
    if runtime.get("phase4b_policy") == PHASE4B_POLICY:
        return []
    if state.get("pending_action") is not None:
        return []

    results = state.get("shot_search_results")
    if not isinstance(results, dict):
        return []
    stale_ids = [
        str(shot_id)
        for shot_id, raw in results.items()
        if isinstance(raw, Mapping)
        and str(raw.get("phase4b_policy") or "")
        in PHASE4B_RESCORABLE_PREVIOUS_POLICIES
        and str(raw.get("identity_observability_policy") or "")
        == PHASE4B_IDENTITY_OBSERVABILITY_POLICY
        and str(raw.get("review_decision") or "") == ""
    ]
    if not stale_ids:
        return []

    shot_index = {
        str(row.get("shot_id") or ""): int(row.get("shot_index") or 0)
        for row in state.get("shots") or []
        if isinstance(row, Mapping)
    }
    stale_ids.sort(key=lambda shot_id: shot_index.get(shot_id, 10**9))
    stale_set = set(stale_ids)
    for shot in state.get("shots") or []:
        if isinstance(shot, dict) and str(shot.get("shot_id") or "") in stale_set:
            shot["status"] = "SEARCHING_MEMORY_READY"
            shot.pop("phase4b_exhausted_without_reviewable_candidate", None)
            shot.pop("phase4b_unreviewable_group_occlusion", None)
            shot.pop("phase4b_unreviewable_mixed_identity_tracklets", None)

    state["shot_search_results"] = {
        str(shot_id): value
        for shot_id, value in results.items()
        if str(shot_id) not in stale_set
    }
    state["ambiguities"] = [
        row
        for row in state.get("ambiguities") or []
        if not (
            isinstance(row, Mapping)
            and str(row.get("shot_id") or "") in stale_set
        )
    ]
    state["pending_action"] = None
    state["status"] = "RUNNING"
    state["decision"] = "AUTHORIZE_PHASE4B_CROSS_SHOT_SCORING"
    runtime.update(
        {
            "phase4b_first_cross_shot_scoring_complete": False,
            "phase4b_stale_terminal_results_reopened": True,
            "phase4b_stale_terminal_policy": sorted(
                PHASE4B_RESCORABLE_PREVIOUS_POLICIES
            ),
            "phase4b_reopened_shot_ids": stale_ids,
            "phase4b_policy": PHASE4B_POLICY,
            "automatic_target_confirmation": False,
        }
    )
    state["runtime"] = runtime
    state["updated_at"] = now_iso()
    return stale_ids


def _phase4b_scale_class(bbox: Sequence[float], frame_height: int) -> str:
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    ratio = height / max(1.0, float(frame_height))
    if ratio >= 0.45:
        return "close-up"
    if ratio >= 0.20:
        return "medium"
    return "wide"



def _phase4b_normalize_scale_class(value: object) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "closeup": "close-up",
        "close-up": "close-up",
        "medium": "medium",
        "mid": "medium",
        "wide": "wide",
        "same-shot-active": "medium",
        "phase3c-active": "medium",
    }
    return aliases.get(raw, "unknown")


def _phase4b_scale_compatible(left: object, right: object) -> bool:
    left_name = _phase4b_normalize_scale_class(left)
    right_name = _phase4b_normalize_scale_class(right)
    if "unknown" in {left_name, right_name}:
        return False
    return left_name == right_name


def _phase4b_top_k_mean(values: Any, np: Any, k: int) -> float:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if flattened.size == 0:
        return -1.0
    use = max(1, min(int(k), int(flattened.size)))
    partitioned = np.partition(flattened, int(flattened.size) - use)
    return float(np.mean(partitioned[-use:]))


def _phase4b_load_identity_negative_scoring_bank(
    *,
    memory: Mapping[str, Any],
    fallback_gallery: Any,
    target_dimension: int,
    np: Any,
) -> dict[str, Any]:
    fallback_value = np.asarray(fallback_gallery, dtype=np.float32)
    if fallback_value.ndim == 2 and int(fallback_value.shape[0]) > 0:
        fallback = _phase4b_normalize_embedding_rows(fallback_value, np)
    else:
        fallback = np.empty((0, int(target_dimension)), dtype=np.float32)
    rows: list[Any] = []
    scale_classes: list[str] = []
    clusters: list[dict[str, Any]] = []
    manifest_path_value = memory.get("manifest_path")
    manifest_sha_value = memory.get("manifest_sha256")
    source_manifest_path: Path | None = None
    if manifest_path_value:
        source_manifest_path = Path(str(manifest_path_value)).resolve()
        validate_input_file(
            source_manifest_path,
            str(manifest_sha_value or "") or None,
            "Phase 4-B identity-negative scoring manifest",
        )
        document = read_object(source_manifest_path)
        for reference in document.get("references") or []:
            if not isinstance(reference, Mapping):
                continue
            candidate_id = str(reference.get("candidate_id") or "")
            embeddings_path_value = reference.get("embeddings_path")
            if not candidate_id or not embeddings_path_value:
                continue
            _, matrix = _phase4b_load_embedding_matrix(
                path_value=embeddings_path_value,
                expected_sha256=reference.get("embeddings_sha256"),
                label=f"Phase 4-B identity-negative candidate bank {candidate_id}",
                np=np,
            )
            if int(matrix.shape[1]) != int(target_dimension):
                raise RuntimeError(
                    "Phase 4-B identity-negative candidate bank dimension mismatch."
                )
            candidate_scales = ["unknown"] * int(matrix.shape[0])
            candidate_manifest_value = reference.get("manifest_path")
            if candidate_manifest_value:
                candidate_manifest_path = Path(str(candidate_manifest_value)).resolve()
                expected_manifest_sha = str(reference.get("manifest_sha256") or "")
                if candidate_manifest_path.is_file():
                    validate_input_file(
                        candidate_manifest_path,
                        expected_manifest_sha or None,
                        f"Phase 4-B identity-negative candidate manifest {candidate_id}",
                    )
                    candidate_manifest = read_object(candidate_manifest_path)
                    gallery_rows = [
                        dict(row)
                        for row in candidate_manifest.get("reference_gallery") or []
                        if isinstance(row, Mapping)
                    ]
                    if len(gallery_rows) == int(matrix.shape[0]):
                        candidate_scales = [
                            _phase4b_normalize_scale_class(row.get("scale_class"))
                            for row in gallery_rows
                        ]
            rows.extend(matrix)
            scale_classes.extend(candidate_scales)
            grouped: dict[str, list[Any]] = {}
            for embedding, scale_class in zip(matrix, candidate_scales):
                grouped.setdefault(scale_class, []).append(embedding)
            for scale_class, embeddings in sorted(grouped.items()):
                cluster_matrix = np.stack(embeddings).astype(np.float32)
                cluster_prototype = _phase4b_normalize_embedding_rows(
                    np.mean(cluster_matrix, axis=0, keepdims=True).astype(np.float32),
                    np,
                )[0]
                clusters.append(
                    {
                        "candidate_id": candidate_id,
                        "scale_class": scale_class,
                        "embedding_count": int(cluster_matrix.shape[0]),
                        "prototype": cluster_prototype,
                    }
                )

    if rows:
        gallery = _phase4b_normalize_embedding_rows(
            np.stack(rows).astype(np.float32),
            np,
        )
        source = "REFERENCE_CANDIDATE_BANKS"
    else:
        gallery = fallback
        scale_classes = ["unknown"] * int(fallback.shape[0])
        if int(fallback.shape[0]) > 0:
            prototype = _phase4b_normalize_embedding_rows(
                np.mean(fallback, axis=0, keepdims=True).astype(np.float32),
                np,
            )[0]
            clusters = [
                {
                    "candidate_id": "fallback_combined_identity_negative",
                    "scale_class": "unknown",
                    "embedding_count": int(fallback.shape[0]),
                    "prototype": prototype,
                }
            ]
        source = "FALLBACK_COMBINED_MEMORY"

    return {
        "policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
        "source": source,
        "source_manifest_path": (
            str(source_manifest_path) if source_manifest_path is not None else None
        ),
        "gallery": gallery,
        "scale_classes": scale_classes,
        "clusters": clusters,
        "embedding_count": int(gallery.shape[0]),
        "cluster_count": len(clusters),
        "automatic_target_confirmation": False,
    }


def _phase4b_load_user_confirmed_role_scoring_bank(
    *,
    memory: Mapping[str, Any],
    target_dimension: int,
    np: Any,
) -> dict[str, Any]:
    rows: list[Any] = []
    scale_classes: list[str] = []
    clusters: list[dict[str, Any]] = []
    source_manifest_path: Path | None = None
    manifest_path_value = memory.get("manifest_path")
    if manifest_path_value:
        source_manifest_path = Path(str(manifest_path_value)).resolve()
        validate_input_file(
            source_manifest_path,
            str(memory.get("manifest_sha256") or "") or None,
            "Phase 4-B persistent role-negative scoring manifest",
        )
        document = read_object(source_manifest_path)
        for reference in document.get("references") or []:
            if not isinstance(reference, Mapping):
                continue
            if str(reference.get("source_kind") or "") != (
                "USER_CONFIRMED_NON_PLAYER_ROLE_CANDIDATE"
            ):
                continue
            candidate_id = str(reference.get("candidate_id") or "")
            embeddings_path_value = reference.get("embeddings_path")
            if not candidate_id or not embeddings_path_value:
                continue
            _, matrix = _phase4b_load_embedding_matrix(
                path_value=embeddings_path_value,
                expected_sha256=reference.get("embeddings_sha256"),
                label=f"Phase 4-B user-confirmed role bank {candidate_id}",
                np=np,
            )
            if int(matrix.shape[1]) != int(target_dimension):
                raise RuntimeError(
                    "Phase 4-B user-confirmed role bank dimension mismatch."
                )
            candidate_scales = ["unknown"] * int(matrix.shape[0])
            candidate_manifest_value = reference.get("manifest_path")
            if candidate_manifest_value:
                candidate_manifest_path = Path(str(candidate_manifest_value)).resolve()
                if candidate_manifest_path.is_file():
                    validate_input_file(
                        candidate_manifest_path,
                        str(reference.get("manifest_sha256") or "") or None,
                        f"Phase 4-B user role candidate manifest {candidate_id}",
                    )
                    candidate_manifest = read_object(candidate_manifest_path)
                    gallery_rows = [
                        dict(row)
                        for row in candidate_manifest.get("reference_gallery") or []
                        if isinstance(row, Mapping)
                    ]
                    if len(gallery_rows) == int(matrix.shape[0]):
                        candidate_scales = [
                            _phase4b_normalize_scale_class(row.get("scale_class"))
                            for row in gallery_rows
                        ]
            rows.extend(matrix)
            scale_classes.extend(candidate_scales)
            grouped: dict[str, list[Any]] = {}
            for embedding, scale_class in zip(matrix, candidate_scales):
                grouped.setdefault(scale_class, []).append(embedding)
            for scale_class, embeddings in sorted(grouped.items()):
                cluster_matrix = np.stack(embeddings).astype(np.float32)
                prototype = _phase4b_normalize_embedding_rows(
                    np.mean(cluster_matrix, axis=0, keepdims=True).astype(np.float32),
                    np,
                )[0]
                clusters.append(
                    {
                        "candidate_id": candidate_id,
                        "scale_class": scale_class,
                        "embedding_count": int(cluster_matrix.shape[0]),
                        "prototype": prototype,
                    }
                )
    gallery = (
        _phase4b_normalize_embedding_rows(np.stack(rows).astype(np.float32), np)
        if rows
        else np.empty((0, int(target_dimension)), dtype=np.float32)
    )
    return {
        "policy": PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY,
        "source": "USER_CONFIRMED_REFERENCE_BANKS" if rows else "NO_USER_CONFIRMED_ROLE_REFERENCES",
        "source_manifest_path": (
            str(source_manifest_path) if source_manifest_path is not None else None
        ),
        "gallery": gallery,
        "scale_classes": scale_classes,
        "clusters": clusters,
        "embedding_count": int(gallery.shape[0]),
        "cluster_count": len(clusters),
        "automatic_target_confirmation": False,
    }


def _phase4b_load_detector_role_gallery(
    *,
    persistent_memory: Mapping[str, Any],
    same_shot_gallery: Any,
    target_dimension: int,
    np: Any,
) -> dict[str, Any]:
    parts: list[Any] = []
    sources: list[dict[str, Any]] = []
    same_shot_value = np.asarray(same_shot_gallery, dtype=np.float32)
    if same_shot_value.ndim == 2 and int(same_shot_value.shape[0]) > 0:
        if int(same_shot_value.shape[1]) != int(target_dimension):
            raise RuntimeError("Same-shot detector role-negative dimension mismatch.")
        parts.append(_phase4b_normalize_embedding_rows(same_shot_value, np))
        sources.append(
            {
                "source_kind": "SAME_SHOT_DETECTOR_LABELED_STAFF_REFEREE",
                "embedding_count": int(same_shot_value.shape[0]),
            }
        )

    manifest_value = persistent_memory.get("manifest_path")
    if manifest_value:
        manifest_path = Path(str(manifest_value)).resolve()
        validate_input_file(
            manifest_path,
            str(persistent_memory.get("manifest_sha256") or "") or None,
            "Phase 4-B persistent role provenance manifest",
        )
        document = read_object(manifest_path)
        for source in document.get("sources") or []:
            if not isinstance(source, Mapping):
                continue
            if str(source.get("source_kind") or "") != (
                "DETECTOR_LABELED_STAFF_REFEREE_GALLERY"
            ):
                continue
            embeddings_path_value = source.get("embeddings_path")
            if not embeddings_path_value:
                continue
            embeddings_path, matrix = _phase4b_load_embedding_matrix(
                path_value=embeddings_path_value,
                expected_sha256=source.get("embeddings_sha256"),
                label="Phase 4-B detector role provenance gallery",
                np=np,
            )
            if int(matrix.shape[1]) != int(target_dimension):
                raise RuntimeError("Detector role provenance dimension mismatch.")
            parts.append(matrix)
            sources.append(
                {
                    "source_kind": str(source.get("source_kind") or ""),
                    "source_shot_id": str(source.get("source_shot_id") or ""),
                    "embeddings_path": str(embeddings_path),
                    "embeddings_sha256": sha256_file(embeddings_path),
                    "embedding_count": int(matrix.shape[0]),
                }
            )
    gallery = (
        _phase4b_normalize_embedding_rows(
            np.concatenate(parts, axis=0).astype(np.float32),
            np,
        )
        if parts
        else np.empty((0, int(target_dimension)), dtype=np.float32)
    )
    return {
        "policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
        "gallery": gallery,
        "embedding_count": int(gallery.shape[0]),
        "sources": sources,
        "automatic_target_confirmation": False,
    }


def _phase4b_scale_compatible_negative_metrics(
    *,
    prefix: str,
    candidate_embeddings: Any,
    candidate_scale_classes: Sequence[str],
    candidate_prototype: Any,
    target_gallery: Any,
    bank: Mapping[str, Any],
    top_k: int,
    np: Any,
) -> dict[str, Any]:
    matrix = _phase4b_normalize_embedding_rows(candidate_embeddings, np)
    prototype = _phase4b_normalize_embedding_rows(
        np.asarray(candidate_prototype, dtype=np.float32).reshape(1, -1),
        np,
    )[0]
    target_best = np.max(matrix @ target_gallery.T, axis=1)
    gallery = np.asarray(bank.get("gallery"), dtype=np.float32)
    scales = [
        _phase4b_normalize_scale_class(value)
        for value in bank.get("scale_classes") or []
    ]
    compatible_negative_scores: list[float] = []
    compatible_target_scores: list[float] = []
    compatible_scale_classes: list[str] = []
    if gallery.ndim == 2 and int(gallery.shape[0]) > 0:
        gallery = _phase4b_normalize_embedding_rows(gallery, np)
        for index, candidate_scale in enumerate(candidate_scale_classes):
            normalized = _phase4b_normalize_scale_class(candidate_scale)
            compatible_indices = [
                gallery_index
                for gallery_index, negative_scale in enumerate(scales)
                if _phase4b_scale_compatible(normalized, negative_scale)
            ]
            if not compatible_indices:
                continue
            similarities = matrix[index] @ gallery[compatible_indices].T
            compatible_negative_scores.append(
                _phase4b_top_k_mean(similarities, np, top_k)
            )
            compatible_target_scores.append(float(target_best[index]))
            compatible_scale_classes.append(normalized)

    clusters = [
        dict(cluster)
        for cluster in bank.get("clusters") or []
        if isinstance(cluster, Mapping)
    ]
    candidate_scale_set = {
        _phase4b_normalize_scale_class(value)
        for value in candidate_scale_classes
        if _phase4b_normalize_scale_class(value) != "unknown"
    }
    compatible_clusters = [
        cluster
        for cluster in clusters
        if _phase4b_normalize_scale_class(cluster.get("scale_class"))
        in candidate_scale_set
    ]
    if compatible_negative_scores and compatible_clusters:
        target_scores = np.asarray(compatible_target_scores, dtype=np.float32)
        negative_scores = np.asarray(compatible_negative_scores, dtype=np.float32)
        margins = target_scores - negative_scores
        prototypes = _phase4b_normalize_embedding_rows(
            np.stack(
                [
                    np.asarray(cluster["prototype"], dtype=np.float32)
                    for cluster in compatible_clusters
                ]
            ).astype(np.float32),
            np,
        )
        negative_prototype_best = float(np.max(prototype @ prototypes.T))
        target_prototype = _phase4b_normalize_embedding_rows(
            np.mean(target_gallery, axis=0, keepdims=True).astype(np.float32),
            np,
        )[0]
        return {
            f"{prefix}_negative_scale_compatible_evidence_available": True,
            f"{prefix}_negative_scale_compatible_crop_count": len(
                compatible_negative_scores
            ),
            f"{prefix}_negative_compatible_cluster_count": len(compatible_clusters),
            f"{prefix}_negative_compatible_scale_classes": sorted(
                set(compatible_scale_classes)
            ),
            f"{prefix}_crop_negative_robust_max": float(np.max(negative_scores)),
            f"{prefix}_crop_margin_mean": float(np.mean(margins)),
            f"{prefix}_crop_margin_median": float(np.median(margins)),
            f"{prefix}_crop_margin_min": float(np.min(margins)),
            f"{prefix}_positive_margin_support_ratio": float(
                np.mean(margins > 0.0)
            ),
            f"{prefix}_prototype_negative_best": negative_prototype_best,
            f"{prefix}_prototype_negative_margin": float(
                float(prototype @ target_prototype) - negative_prototype_best
            ),
        }
    return {
        f"{prefix}_negative_scale_compatible_evidence_available": False,
        f"{prefix}_negative_scale_compatible_crop_count": 0,
        f"{prefix}_negative_compatible_cluster_count": 0,
        f"{prefix}_negative_compatible_scale_classes": [],
        f"{prefix}_crop_negative_robust_max": -1.0,
        f"{prefix}_crop_margin_mean": 1.0,
        f"{prefix}_crop_margin_median": 1.0,
        f"{prefix}_crop_margin_min": 1.0,
        f"{prefix}_positive_margin_support_ratio": 1.0,
        f"{prefix}_prototype_negative_best": -1.0,
        f"{prefix}_prototype_negative_margin": 1.0,
    }


def _phase4b_compute_banked_negative_metrics(
    *,
    candidate_embeddings: Any,
    candidate_scale_classes: Sequence[str],
    candidate_prototype: Any,
    target_gallery: Any,
    identity_bank: Mapping[str, Any],
    user_confirmed_role_bank: Mapping[str, Any],
    detector_role_negative_gallery: Any,
    np: Any,
) -> dict[str, Any]:
    matrix = _phase4b_normalize_embedding_rows(candidate_embeddings, np)
    prototype = _phase4b_normalize_embedding_rows(
        np.asarray(candidate_prototype, dtype=np.float32).reshape(1, -1),
        np,
    )[0]
    target_best = np.max(matrix @ target_gallery.T, axis=1)

    detector_value = np.asarray(detector_role_negative_gallery, dtype=np.float32)
    if detector_value.ndim == 2 and int(detector_value.shape[0]) > 0:
        detector_gallery = _phase4b_normalize_embedding_rows(detector_value, np)
        detector_best = np.max(matrix @ detector_gallery.T, axis=1)
        detector_margin = target_best - detector_best
        detector_prototype_best = float(np.max(prototype @ detector_gallery.T))
        target_prototype = _phase4b_normalize_embedding_rows(
            np.mean(target_gallery, axis=0, keepdims=True).astype(np.float32),
            np,
        )[0]
        detector_metrics = {
            "detector_role_negative_memory_available": True,
            "detector_role_crop_negative_best_max": float(np.max(detector_best)),
            "detector_role_crop_margin_mean": float(np.mean(detector_margin)),
            "detector_role_crop_margin_median": float(np.median(detector_margin)),
            "detector_role_crop_margin_min": float(np.min(detector_margin)),
            "detector_role_positive_margin_support_ratio": float(
                np.mean(detector_margin > 0.0)
            ),
            "detector_role_prototype_negative_best": detector_prototype_best,
            "detector_role_prototype_negative_margin": float(
                float(prototype @ target_prototype) - detector_prototype_best
            ),
        }
    else:
        detector_metrics = {
            "detector_role_negative_memory_available": False,
            "detector_role_crop_negative_best_max": -1.0,
            "detector_role_crop_margin_mean": 1.0,
            "detector_role_crop_margin_median": 1.0,
            "detector_role_crop_margin_min": 1.0,
            "detector_role_positive_margin_support_ratio": 1.0,
            "detector_role_prototype_negative_best": -1.0,
            "detector_role_prototype_negative_margin": 1.0,
        }

    user_role_metrics = _phase4b_scale_compatible_negative_metrics(
        prefix="user_role",
        candidate_embeddings=matrix,
        candidate_scale_classes=candidate_scale_classes,
        candidate_prototype=prototype,
        target_gallery=target_gallery,
        bank=user_confirmed_role_bank,
        top_k=PHASE4B_USER_ROLE_NEGATIVE_TOP_K,
        np=np,
    )
    identity_metrics = _phase4b_scale_compatible_negative_metrics(
        prefix="identity",
        candidate_embeddings=matrix,
        candidate_scale_classes=candidate_scale_classes,
        candidate_prototype=prototype,
        target_gallery=target_gallery,
        bank=identity_bank,
        top_k=PHASE4B_IDENTITY_NEGATIVE_TOP_K,
        np=np,
    )

    # Preserve R8-compatible field names for diagnostics. They now represent
    # detector-derived role evidence only and are never used as a hard reject.
    legacy_role_metrics = {
        "role_negative_memory_available": detector_metrics[
            "detector_role_negative_memory_available"
        ],
        "role_crop_negative_best_max": detector_metrics[
            "detector_role_crop_negative_best_max"
        ],
        "role_crop_margin_mean": detector_metrics[
            "detector_role_crop_margin_mean"
        ],
        "role_crop_margin_median": detector_metrics[
            "detector_role_crop_margin_median"
        ],
        "role_crop_margin_min": detector_metrics[
            "detector_role_crop_margin_min"
        ],
        "role_positive_margin_support_ratio": detector_metrics[
            "detector_role_positive_margin_support_ratio"
        ],
        "role_prototype_negative_best": detector_metrics[
            "detector_role_prototype_negative_best"
        ],
        "role_prototype_negative_margin": detector_metrics[
            "detector_role_prototype_negative_margin"
        ],
    }
    return {
        **legacy_role_metrics,
        **detector_metrics,
        **user_role_metrics,
        **identity_metrics,
        "identity_negative_scoring_policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
        "role_negative_scoring_policy": PHASE4B_ROLE_NEGATIVE_SCORING_POLICY,
        "user_confirmed_role_scoring_policy": (
            PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY
        ),
        "detector_role_scoring_policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
    }


def _phase4b_prepare_detector_source(
    *,
    root: Path,
    output_dir: Path,
    state: Mapping[str, Any],
    shot: Mapping[str, Any],
    device: str,
    overwrite: bool,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    import math

    video = dict(state["video"])
    source_video = Path(str(video["path"])).resolve()
    validate_input_file(source_video, str(video["sha256"]), "Phase 4-B source video")
    fps = float(video["fps"])
    start = int(shot["start_frame"])
    minimum_frames = int(math.ceil(10.0 * fps))
    end = min(
        int(video["frame_count"]) - 1,
        max(int(shot["end_frame_inclusive"]), start + minimum_frames - 1),
    )
    if end - start + 1 < minimum_frames:
        raise RuntimeError("Phase 4-B detector audit requires at least 10 seconds after shot start.")
    work_root = output_dir / "phase4b_cross_shot" / _safe_name(shot["shot_id"])
    clip = work_root / "analysis_clip.mp4"
    if overwrite:
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if not clip.is_file():
        _write_video_range(
            source_video,
            clip,
            start_frame=start,
            end_frame_inclusive=end,
            metadata=video,
        )
    clip_meta = _video_metadata(clip)
    detector_output_root = work_root / "phase1_detector"
    detector_test_name = f"p4b_{_safe_name(shot['shot_id'])}"
    detector_dir = detector_output_root / detector_test_name
    detections_path = detector_dir / "detections.csv"
    if not detections_path.is_file():
        stage0 = root / "target_centric_tracking_v1" / "stage0_audit_inputs.py"
        stage1 = root / "target_centric_tracking_v1" / "stage1_generate_rfdetr_detections.py"
        stage0_artifacts = (
            detector_dir / "input_manifest.json",
            detector_dir / "audit.json",
            detector_dir / "target_initialization.json",
        )
        if not all(path.is_file() for path in stage0_artifacts):
            _phase4b_run_command(
                [
                    sys.executable,
                    str(stage0),
                    "--project-root",
                    str(root),
                    "--video",
                    str(clip),
                    "--test-name",
                    detector_test_name,
                    "--initial-bbox",
                    "0",
                    "0",
                    str(int(clip_meta["width"])),
                    str(int(clip_meta["height"])),
                    "--bbox-format",
                    "xyxy_pixels",
                    "--output-root",
                    str(detector_output_root),
                    *( ["--overwrite-audit"] if overwrite else [] ),
                ],
                cwd=root,
            )
        _phase4b_accept_no_preview_ffmpeg_warning(detector_dir)
        _phase4b_run_command(
            [
                sys.executable,
                str(stage1),
                "--project-root",
                str(root),
                "--test-name",
                detector_test_name,
                "--output-root",
                str(detector_output_root),
                "--device",
                "cpu" if device == "mps" else device,
                "--no-preview",
                *( ["--overwrite-stage1"] if overwrite else [] ),
            ],
            cwd=root,
            accepted_return_codes=frozenset({0, 3}),
        )
    if not detections_path.is_file():
        raise FileNotFoundError(detections_path)
    detector_contract = {
        "analysis_clip_path": str(clip),
        "analysis_clip_sha256": sha256_file(clip),
        "analysis_source_start_frame": start,
        "analysis_source_end_frame_inclusive": end,
        "analysis_clip_frame_count": int(clip_meta["frame_count"]),
        "candidate_generation_source_start_frame": start,
        "candidate_generation_source_end_frame_inclusive": int(shot["end_frame_inclusive"]),
        "candidate_generation_uses_reviewed_shot_only": True,
        "analysis_tail_frames_excluded_from_candidate_generation": max(
            0, end - int(shot["end_frame_inclusive"])
        ),
        "detections_path": str(detections_path),
        "detections_sha256": sha256_file(detections_path),
        "target_agnostic_full_frame_audit_seed": True,
        "stage1_initial_anchor_is_target_identity_evidence": False,
    }
    return clip, clip_meta, detections_path, detector_contract


def _phase4b_accept_no_preview_ffmpeg_warning(detector_dir: Path) -> None:
    """Bridge Stage-0's warning status to Stage-1 for a no-preview run only."""

    audit_path = detector_dir / "audit.json"
    manifest_path = detector_dir / "input_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        return
    audit = read_object(audit_path)
    manifest = read_object(manifest_path)
    findings = [
        dict(row)
        for row in audit.get("findings") or []
        if isinstance(row, Mapping)
    ]
    warning_codes = {
        str(row.get("code") or "")
        for row in findings
        if str(row.get("severity") or "").upper() == "WARNING"
    }
    has_non_warning = any(
        str(row.get("severity") or "").upper() != "WARNING"
        for row in findings
    )
    if (
        audit.get("status") != "PASS_WITH_WARNINGS"
        or manifest.get("status") != "PASS_WITH_WARNINGS"
        or has_non_warning
        or warning_codes != {"FFMPEG_NOT_FOUND"}
    ):
        return
    compatibility = {
        "policy": "NO_PREVIEW_FFMPEG_WARNING_COMPATIBILITY_R1",
        "preserved_warning_codes": sorted(warning_codes),
        "preview_requested": False,
    }
    audit["status"] = "PASS"
    audit["adapter_warning_compatibility"] = compatibility
    manifest["status"] = "PASS"
    manifest["adapter_warning_compatibility"] = compatibility
    atomic_json(audit_path, audit)
    atomic_json(manifest_path, manifest)


def _phase4b_load_runtime(
    *,
    root: Path,
    state: Mapping[str, Any],
    shot: Mapping[str, Any],
    clip: Path,
    clip_meta: Mapping[str, Any],
    detections_path: Path,
    memory: Mapping[str, Any],
    scoring_references: Sequence[Mapping[str, Any]],
    device_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    stage2 = _load_module(
        "kickclip_phase4b_stage2",
        root / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
    )
    stage2b = _load_module(
        "kickclip_phase4b_stage2b",
        root / "target_centric_tracking_v1" / "stage2b_run_same_shot_reentry_reacquisition.py",
    )
    b0 = _load_module(
        "kickclip_phase4b_b0",
        root / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
    )
    b1 = _load_module(
        "kickclip_phase4b_b1",
        root / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
    )
    b3 = _load_module(
        "kickclip_phase4b_b3",
        root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
    )
    e2e = _load_module(
        "kickclip_phase4b_e2e_policy",
        root / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
    )
    reid = _load_module(
        "kickclip_phase4b_reid",
        root / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    )
    raw_by_frame, raw_detection_count = stage2.load_detections(
        detections_path,
        int(clip_meta["frame_count"]),
        int(clip_meta["width"]),
        int(clip_meta["height"]),
    )
    shot_local_end = int(shot["end_frame_inclusive"]) - int(shot["start_frame"])
    by_frame, negative_role_by_frame, candidate_role_filter = (
        _phase4b_filter_candidate_detections(
            raw_by_frame,
            end_frame_inclusive=shot_local_end,
        )
    )
    detection_count = sum(len(rows) for rows in by_frame.values())
    detection_by_id = {
        str(det.detection_id): det
        for rows in by_frame.values()
        for det in rows
    }
    manifest_path = Path(str((state.get("phase1_manifest") or {}).get("path") or "")).resolve()
    manifest_sha = str((state.get("phase1_manifest") or {}).get("sha256") or "")
    validate_input_file(manifest_path, manifest_sha, "Phase-1 frozen manifest")
    manifest = read_object(manifest_path)
    checkpoint = root / str(manifest["models"]["sports_osnet"]["path"]).replace("\\", "/")
    checkpoint = checkpoint.resolve()
    validate_input_file(
        checkpoint,
        str(manifest["models"]["sports_osnet"].get("sha256") or ""),
        "Sports OSNet checkpoint",
    )
    import torch

    selected_device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if selected_device in {"auto", "mps"}:
        selected_device = "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable for Phase 4-B.")
    torch_device = torch.device(selected_device)
    if hasattr(reid, "configure_determinism"):
        reid.configure_determinism(torch)
    deep_root, models, reid_root = stage2b.discover_deep_eiou(root, reid, None)
    safe_loader_compat = _register_sports_osnet_numpy_safe_globals(torch, np)
    model, model_contract = reid.build_model(torch, models, checkpoint, torch_device)
    model_contract = dict(model_contract)
    model_contract["product_numpy_safe_globals_compat"] = safe_loader_compat
    transform = stage2b.build_transform()
    reference_crops: dict[str, Any] = {}
    for index, row in enumerate(scoring_references, start=1):
        path = Path(str(row["path"])).resolve()
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            raise RuntimeError(f"Cannot read Phase 4-B scoring reference: {path}")
        reference_crops[f"memory_ref_{index:03d}"] = image
    target_context_descriptor = _phase4b_aggregate_target_context_descriptor(
        [reference_crops[key] for key in sorted(reference_crops)],
        cv2=cv2,
        np=np,
    )
    reference_embeddings = stage2b.embed_crops(
        reference_crops, model, transform, torch, torch_device, 32
    )
    target_gallery = np.stack(
        [reference_embeddings[key] for key in sorted(reference_embeddings)]
    ).astype(np.float32)
    target_gallery = b1.l2_normalize_rows(target_gallery)
    target_prototype = b1.l2_normalize_vector(
        np.mean(target_gallery, axis=0).astype(np.float32)
    )

    state_runtime = dict(state.get("runtime") or {})
    identity_negative_memory = dict(
        state_runtime.get("phase4b_identity_negative_memory") or {}
    )
    if identity_negative_memory.get("embeddings_path"):
        _, identity_negative_gallery = _phase4b_load_embedding_matrix(
            path_value=identity_negative_memory.get("embeddings_path"),
            expected_sha256=identity_negative_memory.get("embeddings_sha256"),
            label="Phase 4-B user-confirmed identity-negative memory",
            np=np,
        )
        if int(identity_negative_gallery.shape[1]) != int(target_gallery.shape[1]):
            raise RuntimeError(
                "Phase 4-B identity-negative memory dimension does not match target memory."
            )
        identity_negative_memory_available = True
    else:
        identity_negative_gallery = np.empty(
            (0, target_gallery.shape[1]), dtype=np.float32
        )
        identity_negative_memory_available = False
        identity_negative_memory = {
            "policy": PHASE4B_IDENTITY_NEGATIVE_MEMORY_POLICY,
            "revision_id": None,
            "embedding_count": 0,
            "user_confirmed_only": True,
            "automatic_target_confirmation": False,
        }

    persistent_role_negative_memory = dict(
        state_runtime.get("phase4b_persistent_role_negative_memory") or {}
    )
    if persistent_role_negative_memory.get("embeddings_path"):
        _, persistent_role_negative_gallery = _phase4b_load_embedding_matrix(
            path_value=persistent_role_negative_memory.get("embeddings_path"),
            expected_sha256=persistent_role_negative_memory.get("embeddings_sha256"),
            label="Phase 4-B persistent role-negative memory",
            np=np,
        )
        if int(persistent_role_negative_gallery.shape[1]) != int(target_gallery.shape[1]):
            raise RuntimeError(
                "Phase 4-B persistent role-negative dimension does not match target memory."
            )
        persistent_role_negative_memory_available = True
    else:
        persistent_role_negative_gallery = np.empty(
            (0, target_gallery.shape[1]), dtype=np.float32
        )
        persistent_role_negative_memory_available = False
        persistent_role_negative_memory = {
            "policy": PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY,
            "revision_id": None,
            "embedding_count": 0,
            "user_confirmed_non_player_role": False,
            "automatic_target_confirmation": False,
        }

    # Build an explicit same-shot negative gallery from RF-DETR referee/staff
    # detections. This does not label the target automatically; it only prevents
    # staff-like candidates from being surfaced as plausible player matches.
    negative_selected: list[Any] = []
    negative_selection_by_role: dict[str, int] = {}
    minimum_gap = int(dict(e2e.B1_POLICY).get("minimum_crop_gap") or 2)
    for class_name in sorted(PHASE4B_NEGATIVE_ROLE_CLASS_NAMES):
        role_rows = [
            detection
            for frame_index in sorted(negative_role_by_frame)
            for detection in negative_role_by_frame[frame_index]
            if str(detection.class_name or "").strip().lower() == class_name
        ]
        selected = b1.select_diverse_detections(
            role_rows,
            PHASE4B_NEGATIVE_ROLE_MAX_CROPS_PER_CLASS,
            minimum_gap,
        )
        negative_selected.extend(selected)
        negative_selection_by_role[class_name] = len(selected)

    attempt_memory_dir = output_dir / "phase4b_cross_shot" / _safe_name(shot["shot_id"]) / "memory_contract"
    negative_role_dir = attempt_memory_dir / "same_shot_role_negatives"
    negative_role_dir.mkdir(parents=True, exist_ok=True)
    negative_reference_rows: list[dict[str, Any]] = []
    if negative_selected:
        negative_crops = stage2b.collect_crops(
            clip,
            negative_selected,
            int(clip_meta["frame_count"]),
            int(clip_meta["width"]),
            int(clip_meta["height"]),
        )
        negative_embeddings_by_id = stage2b.embed_crops(
            negative_crops,
            model,
            transform,
            torch,
            torch_device,
            32,
        )
        negative_gallery = np.stack(
            [
                negative_embeddings_by_id[str(detection.detection_id)]
                for detection in negative_selected
            ]
        ).astype(np.float32)
        negative_gallery = b1.l2_normalize_rows(negative_gallery)
        for index, detection in enumerate(negative_selected, start=1):
            crop = negative_crops[str(detection.detection_id)]
            path = negative_role_dir / (
                f"negative_{index:02d}_{str(detection.class_name).lower()}_"
                f"frame_{int(detection.frame):06d}.jpg"
            )
            if not cv2.imwrite(str(path), crop):
                raise RuntimeError(f"Cannot write Phase 4-B role-negative crop: {path}")
            negative_reference_rows.append(
                {
                    "detection_id": str(detection.detection_id),
                    "analysis_local_frame_index": int(detection.frame),
                    "source_frame": int(shot["start_frame"]) + int(detection.frame),
                    "class_id": int(detection.class_id),
                    "class_name": str(detection.class_name or "").strip().lower(),
                    "confidence": float(detection.confidence),
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    else:
        negative_gallery = np.empty((0, target_gallery.shape[1]), dtype=np.float32)

    role_negative_memory_available = (
        int(negative_gallery.shape[0]) >= PHASE4B_NEGATIVE_ROLE_MIN_GALLERY_SIZE
    )
    identity_negative_scoring_bank = _phase4b_load_identity_negative_scoring_bank(
        memory=identity_negative_memory,
        fallback_gallery=identity_negative_gallery,
        target_dimension=int(target_gallery.shape[1]),
        np=np,
    )
    user_confirmed_role_scoring_bank = (
        _phase4b_load_user_confirmed_role_scoring_bank(
            memory=persistent_role_negative_memory,
            target_dimension=int(target_gallery.shape[1]),
            np=np,
        )
    )
    detector_role_scoring_bank = _phase4b_load_detector_role_gallery(
        persistent_memory=persistent_role_negative_memory,
        same_shot_gallery=negative_gallery,
        target_dimension=int(target_gallery.shape[1]),
        np=np,
    )
    detector_role_negative_gallery = np.asarray(
        detector_role_scoring_bank["gallery"], dtype=np.float32
    )
    # Combined gallery is retained only for immutable audit compatibility.
    # R9 never uses detector-labeled role embeddings as a hard rejection gate.
    role_negative_parts = [
        gallery
        for gallery in (
            persistent_role_negative_gallery,
            negative_gallery,
        )
        if int(gallery.shape[0]) > 0
    ]
    role_negative_gallery = (
        np.concatenate(role_negative_parts, axis=0).astype(np.float32)
        if role_negative_parts
        else np.empty((0, target_gallery.shape[1]), dtype=np.float32)
    )
    user_confirmed_role_bank_available = bool(
        int(user_confirmed_role_scoring_bank.get("embedding_count") or 0) > 0
    )
    detector_role_bank_available = bool(
        int(detector_role_scoring_bank.get("embedding_count") or 0) > 0
    )
    negative_parts = [
        gallery
        for gallery in (
            identity_negative_gallery,
            persistent_role_negative_gallery,
            negative_gallery,
        )
        if int(gallery.shape[0]) > 0
    ]
    combined_negative_gallery = (
        np.concatenate(negative_parts, axis=0).astype(np.float32)
        if negative_parts
        else np.empty((0, target_gallery.shape[1]), dtype=np.float32)
    )
    negative_memory_available = bool(
        identity_negative_memory_available
        or persistent_role_negative_memory_available
        or role_negative_memory_available
    )
    memory_dir = attempt_memory_dir
    memory_dir.mkdir(parents=True, exist_ok=True)
    target_path = memory_dir / "target_embeddings.npy"
    proto_path = memory_dir / "target_prototype.npy"
    negative_path = memory_dir / "same_shot_role_negative_embeddings.npy"
    combined_negative_path = memory_dir / "combined_negative_embeddings.npy"
    negative_manifest_path = memory_dir / "same_shot_role_negative_manifest.json"
    with target_path.open("wb") as stream:
        np.save(stream, target_gallery, allow_pickle=False)
    with proto_path.open("wb") as stream:
        np.save(stream, target_prototype, allow_pickle=False)
    with negative_path.open("wb") as stream:
        np.save(stream, negative_gallery, allow_pickle=False)
    with combined_negative_path.open("wb") as stream:
        np.save(stream, combined_negative_gallery, allow_pickle=False)
    atomic_json(
        negative_manifest_path,
        {
            "schema_version": "kickclip.phase4b_same_shot_role_negative_memory.v1",
            "policy": PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
            "shot_id": str(shot["shot_id"]),
            "minimum_gallery_size": PHASE4B_NEGATIVE_ROLE_MIN_GALLERY_SIZE,
            "negative_memory_available": role_negative_memory_available,
            "selected_count": int(negative_gallery.shape[0]),
            "selected_by_role": dict(sorted(negative_selection_by_role.items())),
            "references": negative_reference_rows,
            "automatic_target_confirmation": False,
        },
    )
    return {
        "stage2": stage2,
        "stage2b": stage2b,
        "b0": b0,
        "b1": b1,
        "b3": b3,
        "torch": torch,
        "device": torch_device,
        "model": model,
        "transform": transform,
        "model_contract": model_contract,
        "checkpoint": checkpoint,
        "deep_eiou_root": deep_root,
        "reid_root": reid_root,
        "by_frame": by_frame,
        "negative_role_by_frame": negative_role_by_frame,
        "detection_by_id": detection_by_id,
        "detection_count": detection_count,
        "raw_detection_count": int(candidate_role_filter["raw_detection_count"]),
        "analysis_raw_detection_count": raw_detection_count,
        "candidate_role_filter": candidate_role_filter,
        "target_gallery": target_gallery,
        "target_prototype": target_prototype,
        "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
        "target_context_descriptor": target_context_descriptor,
        "negative_gallery": combined_negative_gallery,
        "identity_negative_scoring_bank": identity_negative_scoring_bank,
        "identity_negative_scoring_policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
        "role_negative_gallery": role_negative_gallery,
        "user_confirmed_role_scoring_bank": user_confirmed_role_scoring_bank,
        "user_confirmed_role_memory_available": user_confirmed_role_bank_available,
        "detector_role_negative_gallery": detector_role_negative_gallery,
        "detector_role_scoring_bank": detector_role_scoring_bank,
        "detector_role_memory_available": detector_role_bank_available,
        "role_negative_memory_available": user_confirmed_role_bank_available,
        "role_negative_scoring_policy": PHASE4B_ROLE_NEGATIVE_SCORING_POLICY,
        "negative_memory_available": negative_memory_available,
        "negative_memory_policy": PHASE4B_COMBINED_NEGATIVE_POLICY,
        "combined_negative_embeddings_path": str(combined_negative_path),
        "combined_negative_embeddings_sha256": sha256_file(combined_negative_path),
        "identity_negative_memory": {
            **identity_negative_memory,
            "negative_memory_available": identity_negative_memory_available,
            "embedding_count": int(identity_negative_gallery.shape[0]),
            "scoring_policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
            "scoring_source": str(identity_negative_scoring_bank.get("source") or ""),
            "scoring_embedding_count": int(
                identity_negative_scoring_bank.get("embedding_count") or 0
            ),
            "scoring_cluster_count": int(
                identity_negative_scoring_bank.get("cluster_count") or 0
            ),
        },
        "persistent_role_negative_memory": {
            **persistent_role_negative_memory,
            "negative_memory_available": persistent_role_negative_memory_available,
            "embedding_count": int(persistent_role_negative_gallery.shape[0]),
        },
        "negative_role_memory": {
            "policy": PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
            "negative_memory_available": role_negative_memory_available,
            "selected_count": int(negative_gallery.shape[0]),
            "selected_by_role": dict(sorted(negative_selection_by_role.items())),
            "embeddings_path": str(negative_path),
            "embeddings_sha256": sha256_file(negative_path),
            "manifest_path": str(negative_manifest_path),
            "manifest_sha256": sha256_file(negative_manifest_path),
        },
        "b0_policy": dict(e2e.B0_POLICY),
        "b1_policy": dict(e2e.B1_POLICY),
        "b2_policy": dict(e2e.B2_POLICY),
        "target_embeddings_path": target_path,
        "target_embeddings_sha256": sha256_file(target_path),
        "target_prototype_path": proto_path,
        "target_prototype_sha256": sha256_file(proto_path),
        "clip": clip,
        "clip_meta": dict(clip_meta),
    }


def _phase4b_process_candidates(
    *,
    output_dir: Path,
    state: Mapping[str, Any],
    shot: Mapping[str, Any],
    runtime: Mapping[str, Any],
    memory: Mapping[str, Any],
    memory_path: Path,
    memory_sha: str,
    generation: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    b0 = runtime["b0"]
    b1 = runtime["b1"]
    stage2b = runtime["stage2b"]
    b3 = runtime["b3"]
    local_start = 0
    local_end = int(shot["end_frame_inclusive"]) - int(shot["start_frame"])
    p0 = runtime["b0_policy"]
    raw_tracks, _ = b0.build_tracklets(
        by_frame=runtime["by_frame"],
        start_frame=local_start,
        end_frame_inclusive=local_end,
        max_age=int(p0["max_age"]),
        minimum_predicted_iou=float(p0["minimum_predicted_iou"]),
        maximum_center_distance=float(p0["maximum_center_distance"]),
        minimum_area_ratio=float(p0["minimum_area_ratio"]),
        maximum_area_ratio=float(p0["maximum_area_ratio"]),
        minimum_match_score=float(p0["minimum_match_score"]),
    )
    kept = [
        track
        for track in raw_tracks
        if len(track.observations) >= int(p0["minimum_tracklet_frames"])
    ]
    kept.sort(key=lambda track: (track.start_frame, track.end_frame, track.internal_id))
    shot_root = output_dir / "phase4b_cross_shot" / _safe_name(shot["shot_id"])
    candidate_root = shot_root / "candidates"
    strip_dir = shot_root / "candidate_strips"
    shutil.rmtree(candidate_root, ignore_errors=True)
    shutil.rmtree(strip_dir, ignore_errors=True)
    for stale_name in (
        "candidate_assignments.csv",
        "candidate_tracklets.json",
        "ranked_candidates.csv",
        "ranked_candidates_all.csv",
        "ranked_candidates.json",
        "ranked_candidates.jpg",
        "ranked_candidates_all.jpg",
        "identity_observability_rejected.jpg",
        "identity_purity_rejected.jpg",
        "safe_gate.json",
    ):
        (shot_root / stale_name).unlink(missing_ok=True)
    candidate_root.mkdir(parents=True, exist_ok=True)
    strip_dir.mkdir(parents=True, exist_ok=True)

    assignments: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    rejected_role_confusion_tracklets: list[dict[str, Any]] = []
    rejected_identity_observability_candidates: list[dict[str, Any]] = []
    detections_by_candidate: dict[str, list[Any]] = {}
    clean_detections_by_candidate: dict[str, list[Any]] = {}
    source_start = int(shot["start_frame"])
    frame_width = int(state["video"]["width"])
    frame_height = int(state["video"]["height"])

    for number, track in enumerate(kept, start=1):
        candidate_id = f"{shot['shot_id']}_track_{number:04d}"
        rows: list[Any] = []
        candidate_assignment_rows: list[dict[str, Any]] = []
        for observation in track.observations:
            detection = runtime["detection_by_id"].get(str(observation.detection_id))
            if detection is None:
                raise RuntimeError(
                    f"Phase 4-B detection is missing: {observation.detection_id}"
                )
            class_id = int(detection.class_id)
            class_name = str(detection.class_name or "").strip().lower()
            if (
                class_id not in PHASE4B_ALLOWED_CANDIDATE_CLASS_IDS
                or class_name not in PHASE4B_ALLOWED_CANDIDATE_CLASS_NAMES
            ):
                raise RuntimeError(
                    "Non-player role reached Phase 4-B candidate generation: "
                    f"{class_id}/{class_name}"
                )
            rows.append(detection)
            candidate_assignment_rows.append(
                {
                    "candidate_id": candidate_id,
                    "frame_index": source_start + int(observation.frame_index),
                    "analysis_local_frame_index": int(observation.frame_index),
                    "detection_id": str(observation.detection_id),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(observation.confidence),
                    "x1": float(observation.bbox_xyxy[0]),
                    "y1": float(observation.bbox_xyxy[1]),
                    "x2": float(observation.bbox_xyxy[2]),
                    "y2": float(observation.bbox_xyxy[3]),
                }
            )
        role_confusion = _phase4b_role_confusion_evidence(
            rows,
            runtime.get("negative_role_by_frame") or {},
        )
        if role_confusion.get("passed") is not True:
            rejected_role_confusion_tracklets.append(
                {
                    "candidate_id": candidate_id,
                    "start_frame": source_start + int(track.start_frame),
                    "end_frame_inclusive": source_start + int(track.end_frame),
                    "detection_count": len(rows),
                    "role_confusion": role_confusion,
                }
            )
            continue

        observability = _phase4b_identity_observability(
            rows,
            runtime["by_frame"],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        observability_by_id = {
            str(item["detection_id"]): item
            for item in observability.get("per_observation") or []
            if isinstance(item, Mapping)
        }
        for assignment in candidate_assignment_rows:
            evidence = observability_by_id.get(str(assignment["detection_id"]), {})
            assignment.update(
                {
                    "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                    "clean_for_reid": bool(evidence.get("clean_for_reid")),
                    "crowded": bool(evidence.get("crowded")),
                    "multi_person_contaminated": bool(
                        evidence.get("multi_person_contaminated")
                    ),
                    "maximum_candidate_overlap_ratio": float(
                        evidence.get("maximum_candidate_overlap_ratio") or 0.0
                    ),
                    "bbox_width_height_ratio": float(
                        evidence.get("bbox_width_height_ratio") or 0.0
                    ),
                    "observability_rejection_reasons": "|".join(
                        str(value)
                        for value in evidence.get("rejection_reasons") or []
                    ),
                }
            )
        assignments.extend(candidate_assignment_rows)
        b0.make_tracklet_strip(
            runtime["clip"], candidate_id, track, strip_dir / f"{candidate_id}.jpg"
        )

        base_candidate = {
            "candidate_id": candidate_id,
            "start_frame": source_start + int(track.start_frame),
            "end_frame_inclusive": source_start + int(track.end_frame),
            "analysis_local_start_frame": int(track.start_frame),
            "analysis_local_end_frame_inclusive": int(track.end_frame),
            "detection_count": len(rows),
            "candidate_role_filter_policy": PHASE4B_CANDIDATE_ROLE_FILTER_POLICY,
            "candidate_class_ids": sorted({int(row.class_id) for row in rows}),
            "candidate_class_names": sorted(
                {str(row.class_name or "").strip().lower() for row in rows}
            ),
            "role_confusion": role_confusion,
            "identity_observability": observability,
        }
        if observability.get("passed") is not True:
            rejected_identity_observability_candidates.append(base_candidate)
            continue

        clean_ids = set(str(value) for value in observability["clean_detection_ids"])
        clean_rows = [
            detection
            for detection in rows
            if str(detection.detection_id) in clean_ids
        ]
        if len(clean_rows) < int(observability["minimum_clean_frame_count"]):
            raise RuntimeError(
                "Identity observability passed without enough clean detection rows."
            )
        detections_by_candidate[candidate_id] = rows
        clean_detections_by_candidate[candidate_id] = clean_rows
        candidates.append(
            {
                **base_candidate,
                "reid_eligible_detection_count": len(clean_rows),
            }
        )

    identity_purity = _phase4b_segment_candidates_by_identity_purity(
        state=state,
        shot=shot,
        runtime=runtime,
        base_candidates=candidates,
        detections_by_candidate=detections_by_candidate,
        clean_detections_by_candidate=clean_detections_by_candidate,
        strip_dir=strip_dir,
    )
    assignments = list(identity_purity["assignments"])
    candidates = list(identity_purity["candidates"])
    detections_by_candidate = dict(identity_purity["detections_by_candidate"])
    clean_detections_by_candidate = dict(
        identity_purity["clean_detections_by_candidate"]
    )
    parent_tracklet_summaries = list(
        identity_purity["parent_tracklet_summaries"]
    )
    rejected_identity_purity_segments = list(
        identity_purity["rejected_identity_purity_segments"]
    )
    split_parent_tracklet_count = int(
        identity_purity["split_parent_tracklet_count"]
    )
    identity_segment_candidate_count = int(
        identity_purity["identity_segment_candidate_count"]
    )
    purity_probe_embedding_count = int(
        identity_purity["purity_probe_embedding_count"]
    )

    assignments_path = shot_root / "candidate_assignments.csv"
    _write_csv(assignments_path, assignments)
    atomic_json(
        shot_root / "candidate_tracklets.json",
        {
            "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "parent_tracklet_summaries": parent_tracklet_summaries,
            "split_parent_tracklet_count": split_parent_tracklet_count,
            "identity_segment_candidate_count": identity_segment_candidate_count,
            "purity_probe_embedding_count": purity_probe_embedding_count,
            "candidates": candidates,
            "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
            "rejected_identity_purity_segments": rejected_identity_purity_segments,
            "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
        },
    )

    observability_rejection_sheet = shot_root / "identity_observability_rejected.jpg"
    if rejected_identity_observability_candidates:
        _phase4b_make_observability_rejection_sheet(
            rejected=rejected_identity_observability_candidates,
            strip_dir=strip_dir,
            output_path=observability_rejection_sheet,
        )
    purity_rejection_sheet = shot_root / "identity_purity_rejected.jpg"
    if rejected_identity_purity_segments:
        _phase4b_make_identity_purity_rejection_sheet(
            rejected=rejected_identity_purity_segments,
            strip_dir=strip_dir,
            output_path=purity_rejection_sheet,
        )

    rejected_group_occlusion_count = sum(
        str((row.get("identity_observability") or {}).get("classification") or "")
        == "UNREVIEWABLE_GROUP_OCCLUSION"
        for row in rejected_identity_observability_candidates
    )
    all_observability_rejected = bool(kept) and not candidates and bool(
        rejected_identity_observability_candidates
    )
    all_identity_purity_rejected = bool(parent_tracklet_summaries) and not candidates and bool(
        rejected_identity_purity_segments
    )
    if all_identity_purity_rejected:
        exhaustion_reason = "UNREVIEWABLE_MIXED_IDENTITY_TRACKLETS"
    elif (
        all_observability_rejected
        and rejected_group_occlusion_count
        >= max(1, math.ceil(len(rejected_identity_observability_candidates) / 2))
    ):
        exhaustion_reason = "UNREVIEWABLE_GROUP_OCCLUSION"
    else:
        exhaustion_reason = None

    if not candidates:
        contact_sheet = shot_root / "ranked_candidates.jpg"
        canvas = np.zeros((360, 1280, 3), dtype=np.uint8)
        headline = (
            "NO REVIEWABLE CANDIDATES: GROUP OCCLUSION"
            if exhaustion_reason == "UNREVIEWABLE_GROUP_OCCLUSION"
            else (
                "NO REVIEWABLE CANDIDATES: MIXED IDENTITY TRACKLETS"
                if exhaustion_reason == "UNREVIEWABLE_MIXED_IDENTITY_TRACKLETS"
                else "NO REVIEWABLE CROSS-SHOT CANDIDATES"
            )
        )
        cv2.putText(
            canvas,
            headline,
            (55, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                "SAFE CONTINUE: TRACKLETS COULD NOT FORM IDENTITY-PURE SEGMENTS"
                if exhaustion_reason == "UNREVIEWABLE_MIXED_IDENTITY_TRACKLETS"
                else "SAFE CONTINUE: NO IDENTITY-USABLE SINGLE-PERSON CROP"
            ),
            (55, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(contact_sheet), canvas):
            raise RuntimeError(f"Cannot write safe-block contact sheet: {contact_sheet}")
        gate = _phase4b_gate_candidate_rows(
            [],
            policy=runtime["b2_policy"],
            negative_memory_available=bool(runtime["negative_memory_available"]),
        )
        if exhaustion_reason == "UNREVIEWABLE_GROUP_OCCLUSION":
            gate.update(
                {
                    "operational_state": "SAFE_REJECTED_UNREVIEWABLE_GROUP_OCCLUSION",
                    "decision": "RETAIN_SEARCHING_UNREVIEWABLE_IDENTITY",
                    "plausible_candidate_exists": False,
                }
            )
        elif exhaustion_reason == "UNREVIEWABLE_MIXED_IDENTITY_TRACKLETS":
            gate.update(
                {
                    "operational_state": "SAFE_REJECTED_MIXED_IDENTITY_TRACKLETS",
                    "decision": "RETAIN_SEARCHING_IDENTITY_PURITY_FAILED",
                    "plausible_candidate_exists": False,
                }
            )
        gate.update(
            {
                "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
                "parent_tracklet_count_before_purity_segmentation": len(parent_tracklet_summaries),
                "split_parent_tracklet_count": split_parent_tracklet_count,
                "identity_segment_candidate_count": identity_segment_candidate_count,
                "purity_probe_embedding_count": purity_probe_embedding_count,
                "rejected_identity_purity_segment_count": len(rejected_identity_purity_segments),
                "identity_observability_raw_candidate_count": len(kept),
                "identity_observability_passed_candidate_count": 0,
                "rejected_low_observability_candidate_count": len(
                    rejected_identity_observability_candidates
                ),
                "unreviewable_group_occlusion_candidate_count": rejected_group_occlusion_count,
                "negative_role_memory": dict(runtime.get("negative_role_memory") or {}),
                "identity_negative_memory": dict(runtime.get("identity_negative_memory") or {}),
                "persistent_role_negative_memory": dict(
                    runtime.get("persistent_role_negative_memory") or {}
                ),
                "negative_memory_policy": str(
                    runtime.get("negative_memory_policy") or ""
                ),
                "rejected_role_confusion_tracklet_count": len(
                    rejected_role_confusion_tracklets
                ),
                "rejected_identity_observability_candidate_count": len(
                    rejected_identity_observability_candidates
                ),
                "rejected_negative_margin_candidate_count": 0,
                "deduplicated_candidate_count": 0,
                "automatic_target_confirmation": False,
            }
        )
        atomic_json(shot_root / "safe_gate.json", gate)
        ranked_document = {
            "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "parent_tracklet_summaries": parent_tracklet_summaries,
            "rejected_identity_purity_segments": rejected_identity_purity_segments,
            "reviewable_candidates": [],
            "all_unique_candidates": [],
            "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
            "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
            "rejected_negative_margin_candidates": [],
            "deduplicated_candidates": [],
        }
        ranked_path = shot_root / "ranked_candidates.json"
        atomic_json(ranked_path, ranked_document)
        return {
            "candidate_count": 0,
            "ranked_candidates": [],
            "review_candidates": [],
            "gate": gate,
            "assignments_path": str(assignments_path),
            "assignments_sha256": sha256_file(assignments_path),
            "contact_sheet_path": str(contact_sheet),
            "contact_sheet_sha256": sha256_file(contact_sheet),
            "all_contact_sheet_path": (
                str(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "all_contact_sheet_sha256": (
                sha256_file(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "identity_observability_rejection_sheet_path": (
                str(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "identity_observability_rejection_sheet_sha256": (
                sha256_file(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "identity_purity_rejection_sheet_path": (
                str(purity_rejection_sheet)
                if purity_rejection_sheet.is_file()
                else None
            ),
            "identity_purity_rejection_sheet_sha256": (
                sha256_file(purity_rejection_sheet)
                if purity_rejection_sheet.is_file()
                else None
            ),
            "ranked_candidates_path": str(ranked_path),
            "ranked_candidates_sha256": sha256_file(ranked_path),
            "candidate_role_filter": dict(runtime["candidate_role_filter"]),
            "negative_role_memory": dict(runtime.get("negative_role_memory") or {}),
            "identity_negative_memory": dict(runtime.get("identity_negative_memory") or {}),
            "persistent_role_negative_memory": dict(
                runtime.get("persistent_role_negative_memory") or {}
            ),
            "negative_memory_policy": str(runtime.get("negative_memory_policy") or ""),
            "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "parent_tracklet_summaries": parent_tracklet_summaries,
            "split_parent_tracklet_count": split_parent_tracklet_count,
            "identity_segment_candidate_count": identity_segment_candidate_count,
            "purity_probe_embedding_count": purity_probe_embedding_count,
            "rejected_identity_purity_segments": rejected_identity_purity_segments,
            "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
            "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
            "rejected_negative_margin_candidates": [],
            "deduplicated_candidates": [],
            "raw_tracklet_count": len(raw_tracks),
            "unique_candidate_count_before_negative_gate": 0,
            "exhaustion_reason": exhaustion_reason,
        }

    selected_by_candidate: dict[str, list[Any]] = {}
    all_selected: dict[str, Any] = {}
    p1 = runtime["b1_policy"]
    for candidate_id, detections in clean_detections_by_candidate.items():
        selected = b1.select_diverse_detections(
            detections,
            int(p1["max_crops_per_tracklet"]),
            int(p1["minimum_crop_gap"]),
        )
        minimum_clean = int(
            next(
                row["identity_observability"]["minimum_clean_frame_count"]
                for row in candidates
                if row["candidate_id"] == candidate_id
            )
        )
        if len(selected) < min(3, minimum_clean):
            rejected = next(
                dict(row) for row in candidates if row["candidate_id"] == candidate_id
            )
            rejected["identity_observability"] = {
                **dict(rejected["identity_observability"]),
                "passed": False,
                "classification": "UNREVIEWABLE_LOW_IDENTITY_OBSERVABILITY",
                "rejection_reasons": [
                    *list(
                        rejected["identity_observability"].get("rejection_reasons")
                        or []
                    ),
                    "INSUFFICIENT_DIVERSE_CLEAN_REID_CROPS",
                ],
            }
            rejected_identity_observability_candidates.append(rejected)
            continue
        selected_by_candidate[candidate_id] = selected
        for detection in selected:
            all_selected[str(detection.detection_id)] = detection

    candidates = [
        row for row in candidates if row["candidate_id"] in selected_by_candidate
    ]
    detections_by_candidate = {
        key: value
        for key, value in detections_by_candidate.items()
        if key in selected_by_candidate
    }
    if not candidates:
        # This path is rare: the frame-level gate passed but diversity sampling
        # could not yield at least three clean identity crops. Re-enter through
        # the same safe no-candidate contract without embedding anything.
        contact_sheet = shot_root / "ranked_candidates.jpg"
        canvas = np.zeros((360, 1280, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "NO REVIEWABLE CANDIDATES: INSUFFICIENT CLEAN REID CROPS",
            (45, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(contact_sheet), canvas):
            raise RuntimeError(f"Cannot write safe-block contact sheet: {contact_sheet}")
        gate = _phase4b_gate_candidate_rows(
            [],
            policy=runtime["b2_policy"],
            negative_memory_available=bool(runtime["negative_memory_available"]),
        )
        gate.update(
            {
                "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
                "parent_tracklet_count_before_purity_segmentation": len(parent_tracklet_summaries),
                "split_parent_tracklet_count": split_parent_tracklet_count,
                "identity_segment_candidate_count": identity_segment_candidate_count,
                "purity_probe_embedding_count": purity_probe_embedding_count,
                "rejected_identity_purity_segment_count": len(rejected_identity_purity_segments),
                "identity_observability_raw_candidate_count": len(kept),
                "identity_observability_passed_candidate_count": 0,
                "rejected_low_observability_candidate_count": len(
                    rejected_identity_observability_candidates
                ),
                "unreviewable_group_occlusion_candidate_count": rejected_group_occlusion_count,
                "automatic_target_confirmation": False,
            }
        )
        atomic_json(shot_root / "safe_gate.json", gate)
        ranked_path = shot_root / "ranked_candidates.json"
        atomic_json(
            ranked_path,
            {
                "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
                "parent_tracklet_summaries": parent_tracklet_summaries,
                "rejected_identity_purity_segments": rejected_identity_purity_segments,
                "reviewable_candidates": [],
                "all_unique_candidates": [],
                "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
                "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
                "rejected_negative_margin_candidates": [],
                "deduplicated_candidates": [],
            },
        )
        return {
            "candidate_count": 0,
            "ranked_candidates": [],
            "review_candidates": [],
            "gate": gate,
            "assignments_path": str(assignments_path),
            "assignments_sha256": sha256_file(assignments_path),
            "contact_sheet_path": str(contact_sheet),
            "contact_sheet_sha256": sha256_file(contact_sheet),
            "all_contact_sheet_path": (
                str(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "all_contact_sheet_sha256": (
                sha256_file(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "identity_observability_rejection_sheet_path": (
                str(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "identity_observability_rejection_sheet_sha256": (
                sha256_file(observability_rejection_sheet)
                if observability_rejection_sheet.is_file()
                else None
            ),
            "identity_purity_rejection_sheet_path": (
                str(purity_rejection_sheet)
                if purity_rejection_sheet.is_file()
                else None
            ),
            "identity_purity_rejection_sheet_sha256": (
                sha256_file(purity_rejection_sheet)
                if purity_rejection_sheet.is_file()
                else None
            ),
            "ranked_candidates_path": str(ranked_path),
            "ranked_candidates_sha256": sha256_file(ranked_path),
            "candidate_role_filter": dict(runtime["candidate_role_filter"]),
            "negative_role_memory": dict(runtime.get("negative_role_memory") or {}),
            "identity_negative_memory": dict(runtime.get("identity_negative_memory") or {}),
            "persistent_role_negative_memory": dict(
                runtime.get("persistent_role_negative_memory") or {}
            ),
            "negative_memory_policy": str(runtime.get("negative_memory_policy") or ""),
            "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "parent_tracklet_summaries": parent_tracklet_summaries,
            "split_parent_tracklet_count": split_parent_tracklet_count,
            "identity_segment_candidate_count": identity_segment_candidate_count,
            "purity_probe_embedding_count": purity_probe_embedding_count,
            "rejected_identity_purity_segments": rejected_identity_purity_segments,
            "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
            "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
            "rejected_negative_margin_candidates": [],
            "deduplicated_candidates": [],
            "raw_tracklet_count": len(raw_tracks),
            "unique_candidate_count_before_negative_gate": 0,
            "exhaustion_reason": None,
        }

    crops = stage2b.collect_crops(
        runtime["clip"],
        list(all_selected.values()),
        int(runtime["clip_meta"]["frame_count"]),
        int(runtime["clip_meta"]["width"]),
        int(runtime["clip_meta"]["height"]),
    )
    embeddings = stage2b.embed_crops(
        crops,
        runtime["model"],
        runtime["transform"],
        runtime["torch"],
        runtime["device"],
        32,
    )
    ranked: list[dict[str, Any]] = []
    by_id = {str(row["candidate_id"]): row for row in candidates}
    for candidate_id, selected in selected_by_candidate.items():
        matrix = np.stack(
            [embeddings[str(det.detection_id)] for det in selected]
        ).astype(np.float32)
        empty_negative_gallery = np.empty(
            (0, runtime["target_gallery"].shape[1]),
            dtype=np.float32,
        )
        metrics = b1.compute_candidate_metrics(
            matrix,
            runtime["target_gallery"],
            runtime["target_prototype"],
            empty_negative_gallery,
        )
        prototype = metrics.pop("prototype")
        candidate_context_descriptor = _phase4b_aggregate_target_context_descriptor(
            [crops[str(det.detection_id)] for det in selected],
            cv2=cv2,
            np=np,
        )
        target_context_similarity = _phase4b_target_context_similarity(
            candidate_context_descriptor,
            runtime["target_context_descriptor"],
            np=np,
        )
        candidate_scale_classes = [
            _phase4b_scale_class(
                [float(value) for value in det.bbox],
                frame_height,
            )
            for det in selected
        ]
        banked_metrics = _phase4b_compute_banked_negative_metrics(
            candidate_embeddings=matrix,
            candidate_scale_classes=candidate_scale_classes,
            candidate_prototype=prototype,
            target_gallery=runtime["target_gallery"],
            identity_bank=runtime["identity_negative_scoring_bank"],
            user_confirmed_role_bank=runtime[
                "user_confirmed_role_scoring_bank"
            ],
            detector_role_negative_gallery=runtime[
                "detector_role_negative_gallery"
            ],
            np=np,
        )
        compatible_negative_values = [
            _phase4b_metric_value(
                banked_metrics, "user_role_crop_negative_robust_max", -1.0
            ),
            _phase4b_metric_value(
                banked_metrics, "identity_crop_negative_robust_max", -1.0
            ),
        ]
        prototype_negative_values = [
            _phase4b_metric_value(
                banked_metrics, "user_role_prototype_negative_best", -1.0
            ),
            _phase4b_metric_value(
                banked_metrics, "identity_prototype_negative_best", -1.0
            ),
        ]
        metrics["crop_negative_best_max"] = max(compatible_negative_values)
        metrics["crop_margin_mean"] = min(
            _phase4b_metric_value(banked_metrics, "user_role_crop_margin_mean", 1.0),
            _phase4b_metric_value(banked_metrics, "identity_crop_margin_mean", 1.0),
        )
        metrics["crop_margin_median"] = min(
            _phase4b_metric_value(banked_metrics, "user_role_crop_margin_median", 1.0),
            _phase4b_metric_value(banked_metrics, "identity_crop_margin_median", 1.0),
        )
        metrics["crop_margin_min"] = min(
            _phase4b_metric_value(banked_metrics, "user_role_crop_margin_min", 1.0),
            _phase4b_metric_value(banked_metrics, "identity_crop_margin_min", 1.0),
        )
        metrics["positive_margin_support_ratio"] = min(
            _phase4b_metric_value(
                banked_metrics, "user_role_positive_margin_support_ratio", 1.0
            ),
            _phase4b_metric_value(
                banked_metrics, "identity_positive_margin_support_ratio", 1.0
            ),
        )
        metrics["prototype_negative_best"] = max(prototype_negative_values)
        metrics["prototype_negative_margin"] = min(
            _phase4b_metric_value(
                banked_metrics, "user_role_prototype_negative_margin", 1.0
            ),
            _phase4b_metric_value(
                banked_metrics, "identity_prototype_negative_margin", 1.0
            ),
        )
        row = {
            **by_id[candidate_id],
            "tracklet_detection_count": int(by_id[candidate_id]["detection_count"]),
            "reid_eligible_detection_count": len(
                clean_detections_by_candidate[candidate_id]
            ),
            "embedded_crop_count": int(matrix.shape[0]),
            **metrics,
            **banked_metrics,
            "candidate_scale_classes": candidate_scale_classes,
            "candidate_scale_class_counts": {
                scale_class: candidate_scale_classes.count(scale_class)
                for scale_class in sorted(set(candidate_scale_classes))
            },
            "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
            "target_context_similarity": target_context_similarity,
            "target_context_is_soft_evidence_only": True,
            "negative_margin_evidence_valid": False,
        }
        ranked.append(row)
        candidate_dir = candidate_root / _safe_name(candidate_id)
        refs_dir = candidate_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        reference_gallery: list[dict[str, Any]] = []
        for index, det in enumerate(selected, start=1):
            crop = crops[str(det.detection_id)]
            source_frame = source_start + int(det.frame)
            ref_path = refs_dir / f"reference_{index:02d}_frame_{source_frame:06d}.jpg"
            if not cv2.imwrite(str(ref_path), crop):
                raise RuntimeError(f"Cannot write candidate reference crop: {ref_path}")
            reference_gallery.append(
                {
                    "frame": source_frame,
                    "path": ref_path.relative_to(candidate_dir).as_posix(),
                    "crop_sha256": sha256_file(ref_path),
                    "scale_class": _phase4b_scale_class(
                        [float(value) for value in det.bbox],
                        frame_height,
                    ),
                    "identity_observability_clean": True,
                }
            )
        embedding_path = candidate_dir / "candidate_embeddings.npy"
        prototype_path = candidate_dir / "candidate_prototype.npy"
        with embedding_path.open("wb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        with prototype_path.open("wb") as stream:
            np.save(stream, prototype.astype(np.float32), allow_pickle=False)
        row["_candidate_dir"] = str(candidate_dir)
        row["_prototype_vector"] = np.asarray(prototype, dtype=np.float32)
        row["_target_context_descriptor"] = np.asarray(
            candidate_context_descriptor, dtype=np.float32
        )
        row["_reference_gallery"] = reference_gallery
        row["_embedding_path"] = str(embedding_path)
        row["_embedding_sha256"] = sha256_file(embedding_path)
        row["_prototype_path"] = str(prototype_path)
        row["_prototype_sha256"] = sha256_file(prototype_path)

    ranked.sort(
        key=lambda row: (
            float(row["retrieval_score"]),
            float(row["prototype_target_similarity"]),
            float(row["crop_target_best_median"]),
        ),
        reverse=True,
    )
    assignments_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        assignments_by_candidate.setdefault(
            str(assignment["candidate_id"]),
            [],
        ).append(dict(assignment))

    deduplicated_candidates: list[dict[str, Any]] = []
    unique_ranked: list[dict[str, Any]] = []
    for row in ranked:
        duplicate_of: str | None = None
        duplicate_evidence: dict[str, Any] | None = None
        for kept_row in unique_ranked:
            evidence = _phase4b_track_duplicate_evidence(
                assignments_by_candidate.get(str(row["candidate_id"]), []),
                assignments_by_candidate.get(str(kept_row["candidate_id"]), []),
            )
            if evidence["duplicate"]:
                duplicate_of = str(kept_row["candidate_id"])
                duplicate_evidence = evidence
                break
        if duplicate_of is not None:
            deduplicated_candidates.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "duplicate_of": duplicate_of,
                    "duplicate_evidence": duplicate_evidence,
                }
            )
            continue
        unique_ranked.append(row)

    rejected_negative_margin_candidates: list[dict[str, Any]] = []
    eligible_assisted_rows: list[dict[str, Any]] = []
    for row in unique_ranked:
        review_gate = _phase4b_candidate_negative_review_gate(
            row,
            role_negative_memory_available=bool(
                runtime.get("user_confirmed_role_memory_available")
            ),
            detector_role_memory_available=bool(
                runtime.get("detector_role_memory_available")
            ),
            identity_negative_memory_available=bool(
                dict(runtime.get("identity_negative_memory") or {}).get(
                    "negative_memory_available"
                )
            ),
        )
        row["negative_review_gate"] = review_gate
        row["negative_role_review_gate"] = dict(
            review_gate.get("user_confirmed_role_gate") or {}
        )
        row["user_confirmed_role_review_gate"] = dict(
            review_gate.get("user_confirmed_role_gate") or {}
        )
        row["detector_role_soft_gate"] = dict(
            review_gate.get("detector_role_soft_gate") or {}
        )
        row["identity_negative_review_gate"] = dict(
            review_gate.get("identity_negative_gate") or {}
        )
        row["negative_margin_evidence_valid"] = bool(
            review_gate.get("negative_memory_available")
        )
        row["rescue_review_required"] = False
        row["rescue_review_policy"] = None
        if review_gate.get("passed") is True and _phase4b_plausible_assisted_review_eligible(
            row, policy=runtime["b2_policy"]
        ):
            eligible_assisted_rows.append(row)
            continue

        rejection_reason = (
            "PROVENANCE_NEGATIVE_HARD_GATE"
            if review_gate.get("passed") is not True
            else "BELOW_PLAUSIBLE_TARGET_EVIDENCE"
        )
        rejected_negative_margin_candidates.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "retrieval_score": float(row["retrieval_score"]),
                "prototype_target_similarity": float(
                    row["prototype_target_similarity"]
                ),
                "prototype_negative_best": float(
                    row["prototype_negative_best"]
                ),
                "prototype_negative_margin": float(
                    row["prototype_negative_margin"]
                ),
                "crop_margin_median": float(row["crop_margin_median"]),
                "positive_margin_support_ratio": float(
                    row["positive_margin_support_ratio"]
                ),
                "candidate_scale_classes": list(
                    row.get("candidate_scale_classes") or []
                ),
                "detector_role_crop_margin_median": float(
                    row.get("detector_role_crop_margin_median") or -1.0
                ),
                "detector_role_positive_margin_support_ratio": float(
                    row.get("detector_role_positive_margin_support_ratio") or 0.0
                ),
                "detector_role_prototype_negative_margin": float(
                    row.get("detector_role_prototype_negative_margin") or -1.0
                ),
                "user_role_crop_margin_median": float(
                    row.get("user_role_crop_margin_median") or -1.0
                ),
                "user_role_positive_margin_support_ratio": float(
                    row.get("user_role_positive_margin_support_ratio") or 0.0
                ),
                "user_role_prototype_negative_margin": float(
                    row.get("user_role_prototype_negative_margin") or -1.0
                ),
                "identity_crop_margin_median": float(
                    row.get("identity_crop_margin_median") or -1.0
                ),
                "identity_positive_margin_support_ratio": float(
                    row.get("identity_positive_margin_support_ratio") or 0.0
                ),
                "identity_prototype_negative_margin": float(
                    row.get("identity_prototype_negative_margin") or -1.0
                ),
                "identity_observability": dict(
                    row.get("identity_observability") or {}
                ),
                "negative_review_gate": review_gate,
                "user_confirmed_role_review_gate": dict(
                    review_gate.get("user_confirmed_role_gate") or {}
                ),
                "detector_role_soft_gate": dict(
                    review_gate.get("detector_role_soft_gate") or {}
                ),
                "identity_negative_review_gate": dict(
                    review_gate.get("identity_negative_gate") or {}
                ),
                "rescue_review_eligible": False,
                "rejection_reason": rejection_reason,
            }
        )

    (
        reviewable_ranked,
        rescue_reviewable_ranked,
        unselected_plausible_candidates,
        continuation_groups,
        corroboration_groups,
    ) = _phase4b_select_assisted_review_candidates(
        eligible_assisted_rows,
        policy=runtime["b2_policy"],
        assignments_by_candidate=assignments_by_candidate,
        np=np,
    )
    strict_reviewable_ranked = [
        row
        for row in eligible_assisted_rows
        if row.get("strict_target_evidence_passed") is True
    ]
    review_catalog_ranked = sorted(
        [*reviewable_ranked, *unselected_plausible_candidates],
        key=lambda row: int(row.get("review_catalog_rank") or 10**9),
    )
    for row in unselected_plausible_candidates:
        row["unselected_review_reason"] = "ASSISTED_REVIEW_RANK_LIMIT"

    for rank, row in enumerate(unique_ranked, start=1):
        row["retrieval_rank"] = rank
    for rank, row in enumerate(reviewable_ranked, start=1):
        row["review_rank"] = rank

    public_all_ranked = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in unique_ranked
    ]
    public_reviewable_ranked = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in reviewable_ranked
    ]
    public_review_catalog_ranked = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in review_catalog_ranked
    ]
    ranked_csv = shot_root / "ranked_candidates.csv"
    _write_csv(ranked_csv, public_reviewable_ranked)
    all_ranked_csv = shot_root / "ranked_candidates_all.csv"
    _write_csv(all_ranked_csv, public_all_ranked)
    review_catalog_csv = shot_root / "review_catalog.csv"
    _write_csv(review_catalog_csv, public_review_catalog_ranked)
    review_catalog_contact_sheet = shot_root / "review_catalog.jpg"
    if public_review_catalog_ranked:
        b1.make_ranked_contact_sheet(
            public_review_catalog_ranked,
            strip_dir,
            review_catalog_contact_sheet,
            2,
        )
    all_contact_sheet = shot_root / "ranked_candidates_all.jpg"
    if public_all_ranked:
        b1.make_ranked_contact_sheet(public_all_ranked, strip_dir, all_contact_sheet, 2)

    contact_sheet = shot_root / "ranked_candidates.jpg"
    if public_reviewable_ranked:
        b1.make_ranked_contact_sheet(
            public_reviewable_ranked,
            strip_dir,
            contact_sheet,
            2,
        )
    else:
        canvas = np.zeros((360, 1280, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "NO REVIEWABLE CROSS-SHOT CANDIDATES",
            (60, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "SAFE CONTINUE: OBSERVABILITY OR NEGATIVE-MARGIN GATE",
            (60, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(contact_sheet), canvas):
            raise RuntimeError(f"Cannot write safe-block contact sheet: {contact_sheet}")

    gate = _phase4b_gate_candidate_rows(
        public_reviewable_ranked,
        policy=runtime["b2_policy"],
        negative_memory_available=bool(runtime["negative_memory_available"]),
    )
    gate["identity_observability_policy"] = PHASE4B_IDENTITY_OBSERVABILITY_POLICY
    gate["identity_purity_policy"] = PHASE4B_IDENTITY_PURITY_POLICY
    gate["parent_tracklet_count_before_purity_segmentation"] = len(parent_tracklet_summaries)
    gate["split_parent_tracklet_count"] = split_parent_tracklet_count
    gate["identity_segment_candidate_count"] = identity_segment_candidate_count
    gate["purity_probe_embedding_count"] = purity_probe_embedding_count
    gate["rejected_identity_purity_segment_count"] = len(
        rejected_identity_purity_segments
    )
    gate["identity_purity_parent_summaries"] = parent_tracklet_summaries
    gate["identity_observability_raw_candidate_count"] = len(kept)
    gate["identity_observability_passed_candidate_count"] = len(candidates)
    gate["rejected_low_observability_candidate_count"] = len(
        rejected_identity_observability_candidates
    )
    gate["unreviewable_group_occlusion_candidate_count"] = rejected_group_occlusion_count
    gate["negative_role_memory"] = dict(runtime.get("negative_role_memory") or {})
    gate["identity_negative_memory"] = dict(
        runtime.get("identity_negative_memory") or {}
    )
    gate["persistent_role_negative_memory"] = dict(
        runtime.get("persistent_role_negative_memory") or {}
    )
    gate["negative_memory_policy"] = str(
        runtime.get("negative_memory_policy") or ""
    )
    gate["rejected_role_confusion_tracklet_count"] = len(
        rejected_role_confusion_tracklets
    )
    gate["rejected_identity_observability_candidate_count"] = len(
        rejected_identity_observability_candidates
    )
    gate["rejected_negative_margin_candidate_count"] = len(
        rejected_negative_margin_candidates
    )
    gate["rescue_review_policy"] = PHASE4B_RESCUE_REVIEW_POLICY
    gate["rescue_review_candidate_count"] = len(rescue_reviewable_ranked)
    gate["rescue_review_candidate_ids"] = [
        str(row.get("candidate_id") or "")
        for row in rescue_reviewable_ranked
    ]
    gate["strict_review_candidate_count"] = len(strict_reviewable_ranked)
    gate["eligible_assisted_candidate_count"] = len(eligible_assisted_rows)
    gate["selected_review_candidate_count"] = len(reviewable_ranked)
    gate["selected_review_candidate_ids"] = [
        str(row.get("candidate_id") or "") for row in reviewable_ranked
    ]
    gate["review_catalog_policy"] = PHASE4B_REVIEW_CATALOG_POLICY
    gate["review_catalog_candidate_count"] = len(review_catalog_ranked)
    gate["review_catalog_candidate_ids"] = [
        str(row.get("candidate_id") or "") for row in review_catalog_ranked
    ]
    gate["manual_review_promotion_allowed"] = True
    gate["group_representative_policy"] = PHASE4B_GROUP_REPRESENTATIVE_POLICY
    gate["group_confidence_policy"] = PHASE4B_GROUP_CONFIDENCE_POLICY
    gate["unselected_plausible_candidate_count"] = len(
        unselected_plausible_candidates
    )
    gate["continuation_group_count"] = len(continuation_groups)
    gate["continuation_groups"] = continuation_groups
    gate["target_context_policy"] = PHASE4B_TARGET_CONTEXT_POLICY
    gate["target_context_is_soft_evidence_only"] = True
    gate["corroboration_policy"] = PHASE4B_CORROBORATION_POLICY
    gate["corroboration_group_count"] = len(corroboration_groups)
    gate["corroboration_groups"] = corroboration_groups
    gate["corroborated_candidate_count"] = sum(
        row.get("corroborated_by_independent_tracklet") is True
        for row in eligible_assisted_rows
    )
    gate["assisted_review_selection_policy"] = (
        PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY
    )
    gate["identity_negative_scoring_policy"] = (
        PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY
    )
    gate["user_confirmed_role_scoring_policy"] = (
        PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY
    )
    gate["detector_role_scoring_policy"] = PHASE4B_DETECTOR_ROLE_SCORING_POLICY
    gate["detector_role_evidence_is_soft_only"] = False
    gate["role_negative_scoring_policy"] = PHASE4B_ROLE_NEGATIVE_SCORING_POLICY
    gate["deduplicated_candidate_count"] = len(deduplicated_candidates)
    atomic_json(shot_root / "safe_gate.json", gate)
    atomic_json(
        shot_root / "ranked_candidates.json",
        {
            "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "parent_tracklet_summaries": parent_tracklet_summaries,
            "split_parent_tracklet_count": split_parent_tracklet_count,
            "identity_segment_candidate_count": identity_segment_candidate_count,
            "purity_probe_embedding_count": purity_probe_embedding_count,
            "rejected_identity_purity_segments": rejected_identity_purity_segments,
            "reviewable_candidates": public_reviewable_ranked,
            "review_catalog_policy": PHASE4B_REVIEW_CATALOG_POLICY,
            "review_catalog_candidates": public_review_catalog_ranked,
            "manual_review_promotion_allowed": True,
            "group_representative_policy": PHASE4B_GROUP_REPRESENTATIVE_POLICY,
            "group_confidence_policy": PHASE4B_GROUP_CONFIDENCE_POLICY,
            "all_unique_candidates": public_all_ranked,
            "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
            "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
            "rejected_negative_margin_candidates": rejected_negative_margin_candidates,
            "rescue_review_policy": PHASE4B_RESCUE_REVIEW_POLICY,
            "rescue_review_candidates": [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in rescue_reviewable_ranked
            ],
            "strict_review_candidates": [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in strict_reviewable_ranked
            ],
            "unselected_plausible_candidates": [
                {key: value for key, value in row.items() if not key.startswith("_")}
                for row in unselected_plausible_candidates
            ],
            "continuation_groups": continuation_groups,
            "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
            "target_context_is_soft_evidence_only": True,
            "corroboration_policy": PHASE4B_CORROBORATION_POLICY,
            "corroboration_groups": corroboration_groups,
            "assisted_review_selection_policy": PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY,
            "identity_negative_scoring_policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
            "user_confirmed_role_scoring_policy": PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY,
            "detector_role_scoring_policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
            "detector_role_evidence_is_soft_only": False,
            "role_negative_scoring_policy": PHASE4B_ROLE_NEGATIVE_SCORING_POLICY,
            "deduplicated_candidates": deduplicated_candidates,
        },
    )
    review_count = min(
        int(runtime["b2_policy"]["review_candidate_count"]),
        len(reviewable_ranked),
    )

    # Materialize immutable evidence for the whole safe review catalog once.
    # R14 can then page through later batches without rerunning detector/ReID.
    materialized_review_catalog: list[dict[str, Any]] = []
    source_video = Path(str(state["video"]["path"])).resolve()
    shot_clip = shot_root / "shot_clip.mp4"
    if not shot_clip.is_file():
        _write_video_range(
            source_video,
            shot_clip,
            start_frame=int(shot["start_frame"]),
            end_frame_inclusive=int(shot["end_frame_inclusive"]),
            metadata=state["video"],
        )
    assignment_rows = _read_csv_rows(assignments_path)
    for raw in review_catalog_ranked:
        candidate_id = str(raw["candidate_id"])
        candidate_dir = Path(str(raw["_candidate_dir"]))
        candidate_assignments = [
            row
            for row in assignment_rows
            if str(row.get("candidate_id") or "") == candidate_id
            and str(row.get("clean_for_reid") or "").strip().lower()
            in {"true", "1", "yes"}
        ]
        selected_anchor, scored_anchor = b3.choose_anchor(candidate_assignments)
        anchor_frame = int(selected_anchor["frame_index"])
        full_frame = candidate_dir / "full_frame_context.jpg"
        _write_full_frame(source_video, anchor_frame, full_frame)
        strip = strip_dir / f"{candidate_id}.jpg"
        gallery = candidate_dir / "reference_gallery.jpg"
        shutil.copy2(strip, gallery)
        score_evidence = {
            "phase4b_policy": PHASE4B_POLICY,
            "candidate_role_filter_policy": PHASE4B_CANDIDATE_ROLE_FILTER_POLICY,
            "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
            "identity_observability": dict(raw.get("identity_observability") or {}),
            "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
            "identity_purity": dict(raw.get("identity_purity") or {}),
            "parent_tracklet_id": raw.get("parent_tracklet_id"),
            "memory_revision_id": str(memory["memory_revision_id"]),
            "memory_revision_path": str(memory_path),
            "memory_revision_sha256": memory_sha,
            "memory_source_reference_count": int(
                memory.get("reference_count") or len(memory.get("references") or [])
            ),
            "memory_scoring_reference_count": int(
                memory.get("scoring_reference_count") or len(runtime["target_gallery"])
            ),
            "memory_pending_review_reference_count": int(
                memory.get("pending_review_reference_count") or 0
            ),
            "memory_scale_banks_used": list(memory.get("scale_banks_used") or []),
            "candidate_scoring_generation": generation,
            "backend_memory_used_by_phase4b_scoring": True,
            "backend_memory_used_by_provided_e2e_scoring": False,
            "negative_memory_available": bool(runtime["negative_memory_available"]),
            "negative_memory_policy": str(runtime.get("negative_memory_policy") or ""),
            "identity_negative_memory": dict(
                runtime.get("identity_negative_memory") or {}
            ),
            "persistent_role_negative_memory": dict(
                runtime.get("persistent_role_negative_memory") or {}
            ),
            "negative_role_memory_policy": PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
            "negative_role_memory": dict(runtime.get("negative_role_memory") or {}),
            "negative_review_gate": dict(raw.get("negative_review_gate") or {}),
            "negative_role_review_gate": dict(
                raw.get("negative_role_review_gate") or {}
            ),
            "user_confirmed_role_review_gate": dict(
                raw.get("user_confirmed_role_review_gate") or {}
            ),
            "detector_role_soft_gate": dict(
                raw.get("detector_role_soft_gate") or {}
            ),
            "identity_negative_review_gate": dict(
                raw.get("identity_negative_review_gate") or {}
            ),
            "assisted_review_selection_policy": PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY,
            "review_catalog_policy": PHASE4B_REVIEW_CATALOG_POLICY,
            "group_representative_policy": PHASE4B_GROUP_REPRESENTATIVE_POLICY,
            "group_representative_score": float(raw.get("group_representative_score") or 0.0),
            "group_confidence_policy": PHASE4B_GROUP_CONFIDENCE_POLICY,
            "group_confidence_score": float(raw.get("group_confidence_score") or 0.0),
            "stable_review_score": float(raw.get("stable_review_score") or 0.0),
            "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
            "target_context_similarity": float(
                raw.get("target_context_similarity") or 0.0
            ),
            "target_context_is_soft_evidence_only": True,
            "corroboration_policy": PHASE4B_CORROBORATION_POLICY,
            "corroboration_group_id": raw.get("corroboration_group_id"),
            "corroboration_group_member_ids": list(
                raw.get("corroboration_group_member_ids") or []
            ),
            "corroboration_independent_parent_count": int(
                raw.get("corroboration_independent_parent_count") or 1
            ),
            "corroborated_by_independent_tracklet": bool(
                raw.get("corroborated_by_independent_tracklet")
            ),
            "continuation_group_id": raw.get("continuation_group_id"),
            "continuation_group_member_ids": list(
                raw.get("continuation_group_member_ids") or []
            ),
            "identity_negative_scoring_policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
            "user_confirmed_role_scoring_policy": PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY,
            "detector_role_scoring_policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
            "role_negative_scoring_policy": PHASE4B_ROLE_NEGATIVE_SCORING_POLICY,
            "rescue_review_policy": raw.get("rescue_review_policy"),
            "rescue_review_required": bool(raw.get("rescue_review_required")),
            "rescue_review_evidence": dict(
                raw.get("rescue_review_evidence") or {}
            ),
            "automatic_target_confirmation": False,
        }
        manifest = {
            "schema_version": "kickclip.runtime_candidate_manifest.r1_2",
            "candidate_id": candidate_id,
            "candidate_media_id": candidate_id,
            "shot_id": str(shot["shot_id"]),
            "tracklet_id": candidate_id,
            "quality": {
                "identity_pure": bool(
                    dict(raw.get("identity_purity") or {}).get("passed") is True
                ),
                "identity_purity_gate_passed": bool(
                    dict(raw.get("identity_purity") or {}).get("passed") is True
                ),
                "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
                "reviewability": "PHASE4B_USER_CONFIRMATION_REQUIRED",
                "identity_confirmation": False,
                "candidate_role_filter_passed": True,
                "allowed_player_goalkeeper_only": True,
                "identity_observability_gate_passed": True,
                "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
                "reid_uses_clean_observations_only": True,
                "same_shot_role_negative_gate_passed": True,
                "combined_negative_gate_passed": bool(
                    dict(raw.get("negative_review_gate") or {}).get("passed") is True
                ),
                "role_negative_hard_gate_passed": bool(
                    dict(raw.get("negative_role_review_gate") or {}).get("passed")
                    is True
                ),
                "identity_negative_robust_gate_passed": bool(
                    dict(raw.get("identity_negative_review_gate") or {}).get("passed")
                    is True
                ),
                "identity_negative_rescue_review": bool(
                    raw.get("rescue_review_required")
                ),
                "rescue_review_policy": raw.get("rescue_review_policy"),
                "user_confirmed_identity_negative_memory_used": bool(
                    dict(runtime.get("identity_negative_memory") or {}).get(
                        "negative_memory_available"
                    )
                ),
                "persistent_role_negative_memory_used": bool(
                    dict(runtime.get("persistent_role_negative_memory") or {}).get(
                        "negative_memory_available"
                    )
                ),
                "temporal_role_confusion_gate_passed": bool(
                    dict(raw.get("role_confusion") or {}).get("passed") is True
                ),
            },
            "identity_observability": dict(raw.get("identity_observability") or {}),
            "identity_purity": dict(raw.get("identity_purity") or {}),
            "parent_tracklet_id": raw.get("parent_tracklet_id"),
            "assisted_review_selection": {
                "policy": PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY,
                "review_catalog_policy": PHASE4B_REVIEW_CATALOG_POLICY,
                "manual_review_promotion_allowed": True,
                "group_representative_policy": PHASE4B_GROUP_REPRESENTATIVE_POLICY,
                "group_representative_score": float(raw.get("group_representative_score") or 0.0),
                "group_confidence_policy": PHASE4B_GROUP_CONFIDENCE_POLICY,
                "group_confidence_score": float(raw.get("group_confidence_score") or 0.0),
                "stable_review_score": float(raw.get("stable_review_score") or 0.0),
                "strict_target_evidence_passed": bool(
                    raw.get("strict_target_evidence_passed")
                ),
                "continuation_group_id": raw.get("continuation_group_id"),
                "continuation_group_member_ids": list(
                    raw.get("continuation_group_member_ids") or []
                ),
                "continuation_group_representative": bool(
                    raw.get("continuation_group_representative")
                ),
                "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
                "target_context_similarity": float(
                    raw.get("target_context_similarity") or 0.0
                ),
                "target_context_is_soft_evidence_only": True,
                "corroboration_policy": PHASE4B_CORROBORATION_POLICY,
                "corroboration_group_id": raw.get("corroboration_group_id"),
                "corroboration_group_member_ids": list(
                    raw.get("corroboration_group_member_ids") or []
                ),
                "corroboration_independent_parent_count": int(
                    raw.get("corroboration_independent_parent_count") or 1
                ),
                "corroborated_by_independent_tracklet": bool(
                    raw.get("corroborated_by_independent_tracklet")
                ),
                "automatic_target_confirmation": False,
            },
            "start_frame": int(raw["start_frame"]),
            "end_frame_inclusive": int(raw["end_frame_inclusive"]),
            "best_observation": {
                "candidate_id": candidate_id,
                "frame": anchor_frame,
                "bbox_xyxy": [
                    float(selected_anchor["x1"]),
                    float(selected_anchor["y1"]),
                    float(selected_anchor["x2"]),
                    float(selected_anchor["y2"]),
                ],
                "stage3b3_scored_observations": scored_anchor,
            },
            "reference_gallery": raw["_reference_gallery"],
            "ranking_metrics": {
                key: value for key, value in raw.items() if not key.startswith("_")
            },
            "candidate_embeddings": {
                "path": str(raw["_embedding_path"]),
                "sha256": str(raw["_embedding_sha256"]),
                "prototype_path": str(raw["_prototype_path"]),
                "prototype_sha256": str(raw["_prototype_sha256"]),
            },
            "score_evidence": score_evidence,
            "automatic_target_confirmation": False,
        }
        manifest_path = candidate_dir / "candidate_manifest.json"
        atomic_json(manifest_path, manifest)
        materialized_review_catalog.append(
            {
                **{key: value for key, value in raw.items() if not key.startswith("_")},
                "candidate_id": candidate_id,
                "shot_id": str(shot["shot_id"]),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "full_frame_context_path": str(full_frame),
                "full_frame_context_sha256": sha256_file(full_frame),
                "shot_clip_path": str(shot_clip),
                "shot_clip_sha256": sha256_file(shot_clip),
                "reference_gallery_path": str(gallery),
                "reference_gallery_sha256": sha256_file(gallery),
                "best_frame": anchor_frame,
                "best_bbox_xyxy": manifest["best_observation"]["bbox_xyxy"],
                "score_evidence": score_evidence,
                "status": "PENDING",
            }
        )
    selected_review_ids = {
        str(row.get("candidate_id") or "")
        for row in reviewable_ranked[:review_count]
    }
    review_candidates = [
        row
        for row in materialized_review_catalog
        if str(row.get("candidate_id") or "") in selected_review_ids
    ]
    return {
        "candidate_count": len(public_reviewable_ranked),
        "raw_tracklet_count": len(raw_tracks),
        "unique_candidate_count_before_negative_gate": len(public_all_ranked),
        "ranked_candidates": public_reviewable_ranked,
        "all_unique_candidates": public_all_ranked,
        "review_catalog_policy": PHASE4B_REVIEW_CATALOG_POLICY,
        "review_catalog_candidates": materialized_review_catalog,
        "review_catalog_path": str(review_catalog_csv),
        "review_catalog_sha256": sha256_file(review_catalog_csv),
        "review_catalog_contact_sheet_path": str(review_catalog_contact_sheet) if review_catalog_contact_sheet.is_file() else None,
        "review_catalog_contact_sheet_sha256": sha256_file(review_catalog_contact_sheet) if review_catalog_contact_sheet.is_file() else None,
        "manual_review_promotion_allowed": True,
        "group_representative_policy": PHASE4B_GROUP_REPRESENTATIVE_POLICY,
        "group_confidence_policy": PHASE4B_GROUP_CONFIDENCE_POLICY,
        "review_candidates": review_candidates,
        "gate": gate,
        "assignments_path": str(assignments_path),
        "assignments_sha256": sha256_file(assignments_path),
        "contact_sheet_path": str(contact_sheet),
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "all_contact_sheet_path": (
            str(all_contact_sheet) if all_contact_sheet.is_file() else None
        ),
        "all_contact_sheet_sha256": (
            sha256_file(all_contact_sheet) if all_contact_sheet.is_file() else None
        ),
        "identity_observability_rejection_sheet_path": (
            str(observability_rejection_sheet)
            if observability_rejection_sheet.is_file()
            else None
        ),
        "identity_observability_rejection_sheet_sha256": (
            sha256_file(observability_rejection_sheet)
            if observability_rejection_sheet.is_file()
            else None
        ),
        "identity_purity_rejection_sheet_path": (
            str(purity_rejection_sheet)
            if purity_rejection_sheet.is_file()
            else None
        ),
        "identity_purity_rejection_sheet_sha256": (
            sha256_file(purity_rejection_sheet)
            if purity_rejection_sheet.is_file()
            else None
        ),
        "ranked_candidates_path": str(shot_root / "ranked_candidates.json"),
        "ranked_candidates_sha256": sha256_file(
            shot_root / "ranked_candidates.json"
        ),
        "candidate_role_filter": dict(runtime["candidate_role_filter"]),
        "negative_role_memory": dict(runtime.get("negative_role_memory") or {}),
        "identity_negative_memory": dict(runtime.get("identity_negative_memory") or {}),
        "persistent_role_negative_memory": dict(
            runtime.get("persistent_role_negative_memory") or {}
        ),
        "negative_memory_policy": str(runtime.get("negative_memory_policy") or ""),
        "identity_observability_policy": PHASE4B_IDENTITY_OBSERVABILITY_POLICY,
        "identity_purity_policy": PHASE4B_IDENTITY_PURITY_POLICY,
        "parent_tracklet_summaries": parent_tracklet_summaries,
        "split_parent_tracklet_count": split_parent_tracklet_count,
        "identity_segment_candidate_count": identity_segment_candidate_count,
        "purity_probe_embedding_count": purity_probe_embedding_count,
        "rejected_identity_purity_segments": rejected_identity_purity_segments,
        "identity_observability_raw_candidate_count": len(kept),
        "identity_observability_passed_candidate_count": len(candidates),
        "rejected_identity_observability_candidates": rejected_identity_observability_candidates,
        "rejected_role_confusion_tracklets": rejected_role_confusion_tracklets,
        "rejected_negative_margin_candidates": rejected_negative_margin_candidates,
        "rescue_review_policy": PHASE4B_RESCUE_REVIEW_POLICY,
        "rescue_review_candidates": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rescue_reviewable_ranked
        ],
        "strict_review_candidates": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in strict_reviewable_ranked
        ],
        "unselected_plausible_candidates": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in unselected_plausible_candidates
        ],
        "continuation_groups": continuation_groups,
        "target_context_policy": PHASE4B_TARGET_CONTEXT_POLICY,
        "target_context_is_soft_evidence_only": True,
        "corroboration_policy": PHASE4B_CORROBORATION_POLICY,
        "corroboration_groups": corroboration_groups,
        "assisted_review_selection_policy": PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY,
        "identity_negative_scoring_policy": PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY,
        "user_confirmed_role_scoring_policy": PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY,
        "detector_role_scoring_policy": PHASE4B_DETECTOR_ROLE_SCORING_POLICY,
        "detector_role_evidence_is_soft_only": False,
        "role_negative_scoring_policy": PHASE4B_ROLE_NEGATIVE_SCORING_POLICY,
        "deduplicated_candidates": deduplicated_candidates,
        "exhaustion_reason": None,
    }


def _phase4b_apply_result_to_state(
    *,
    state: dict[str, Any],
    shot: Mapping[str, Any],
    result: Mapping[str, Any],
    memory: Mapping[str, Any],
    memory_path: Path,
    memory_sha: str,
    generation: int,
    report_path: Path,
    remaining_ready_shot_ids: Sequence[str] = (),
) -> dict[str, Any]:
    review_candidates = [
        dict(row)
        for row in result.get("review_candidates") or []
        if isinstance(row, Mapping)
    ]
    runtime = dict(state.get("runtime") or {})
    runtime.update(
        {
            "phase4b_policy": PHASE4B_POLICY,
            "phase4b_cross_shot_scoring_authorized": True,
            "phase4b_first_cross_shot_scoring_complete": True,
            "phase4b_scored_shot_id": str(shot["shot_id"]),
            "phase4b_candidate_count": int(result.get("candidate_count") or 0),
            "phase4b_candidate_role_filter": dict(
                result.get("candidate_role_filter") or {}
            ),
            "phase4b_negative_role_memory": dict(
                result.get("negative_role_memory") or {}
            ),
            "phase4b_identity_negative_memory": dict(
                result.get("identity_negative_memory")
                or runtime.get("phase4b_identity_negative_memory")
                or {}
            ),
            "phase4b_persistent_role_negative_memory": dict(
                result.get("persistent_role_negative_memory")
                or runtime.get("phase4b_persistent_role_negative_memory")
                or {}
            ),
            "phase4b_negative_memory_policy": str(
                result.get("negative_memory_policy") or ""
            ),
            "phase4b_raw_tracklet_count": int(
                result.get("raw_tracklet_count") or 0
            ),
            "phase4b_unique_candidate_count_before_negative_gate": int(
                result.get("unique_candidate_count_before_negative_gate") or 0
            ),
            "phase4b_rejected_role_confusion_tracklet_count": len(
                result.get("rejected_role_confusion_tracklets") or []
            ),
            "phase4b_rejected_negative_margin_candidate_count": len(
                result.get("rejected_negative_margin_candidates") or []
            ),
            "phase4b_rescue_review_policy": str(
                result.get("rescue_review_policy") or ""
            ),
            "phase4b_rescue_review_candidate_count": len(
                result.get("rescue_review_candidates") or []
            ),
            "phase4b_rescue_review_candidate_ids": [
                str(row.get("candidate_id") or "")
                for row in result.get("rescue_review_candidates") or []
                if isinstance(row, Mapping)
            ],
            "phase4b_identity_negative_scoring_policy": str(
                result.get("identity_negative_scoring_policy") or ""
            ),
            "phase4b_user_confirmed_role_scoring_policy": str(
                result.get("user_confirmed_role_scoring_policy") or ""
            ),
            "phase4b_detector_role_scoring_policy": str(
                result.get("detector_role_scoring_policy") or ""
            ),
            "phase4b_detector_role_evidence_is_soft_only": bool(
                result.get("detector_role_evidence_is_soft_only") is True
            ),
            "phase4b_assisted_review_selection_policy": str(
                result.get("assisted_review_selection_policy") or ""
            ),
            "phase4b_selected_review_candidate_ids": [
                str(row.get("candidate_id") or "")
                for row in result.get("review_candidates") or []
                if isinstance(row, Mapping)
            ],
            "phase4b_review_catalog_policy": str(
                result.get("review_catalog_policy") or PHASE4B_REVIEW_CATALOG_POLICY
            ),
            "phase4b_review_catalog_candidate_count": len(
                result.get("review_catalog_candidates") or []
            ),
            "phase4b_review_catalog_candidate_ids": [
                str(row.get("candidate_id") or "")
                for row in result.get("review_catalog_candidates") or []
                if isinstance(row, Mapping)
            ],
            "phase4b_review_catalog_path": result.get("review_catalog_path"),
            "phase4b_review_catalog_sha256": result.get("review_catalog_sha256"),
            "phase4b_review_catalog_contact_sheet_path": result.get("review_catalog_contact_sheet_path"),
            "phase4b_review_catalog_contact_sheet_sha256": result.get("review_catalog_contact_sheet_sha256"),
            "phase4b_manual_review_promotion_allowed": bool(
                result.get("manual_review_promotion_allowed") is True
            ),
            "phase4b_group_representative_policy": str(
                result.get("group_representative_policy") or PHASE4B_GROUP_REPRESENTATIVE_POLICY
            ),
            "phase4b_group_confidence_policy": str(
                result.get("group_confidence_policy") or PHASE4B_GROUP_CONFIDENCE_POLICY
            ),
            "phase4b_continuation_group_count": len(
                result.get("continuation_groups") or []
            ),
            "phase4b_target_context_policy": str(
                result.get("target_context_policy") or PHASE4B_TARGET_CONTEXT_POLICY
            ),
            "phase4b_target_context_is_soft_evidence_only": bool(
                result.get("target_context_is_soft_evidence_only") is True
            ),
            "phase4b_corroboration_policy": str(
                result.get("corroboration_policy") or PHASE4B_CORROBORATION_POLICY
            ),
            "phase4b_corroboration_group_count": len(
                result.get("corroboration_groups") or []
            ),
            "phase4b_role_negative_scoring_policy": str(
                result.get("role_negative_scoring_policy") or ""
            ),
            "phase4b_deduplicated_candidate_count": len(
                result.get("deduplicated_candidates") or []
            ),
            "phase4b_identity_observability_policy": str(
                result.get("identity_observability_policy") or ""
            ),
            "phase4b_identity_observability_raw_candidate_count": int(
                result.get("identity_observability_raw_candidate_count") or 0
            ),
            "phase4b_identity_observability_passed_candidate_count": int(
                result.get("identity_observability_passed_candidate_count") or 0
            ),
            "phase4b_rejected_identity_observability_candidate_count": len(
                result.get("rejected_identity_observability_candidates") or []
            ),
            "phase4b_identity_purity_policy": str(
                result.get("identity_purity_policy") or ""
            ),
            "phase4b_parent_tracklet_count_before_purity_segmentation": len(
                result.get("parent_tracklet_summaries") or []
            ),
            "phase4b_split_parent_tracklet_count": int(
                result.get("split_parent_tracklet_count") or 0
            ),
            "phase4b_identity_segment_candidate_count": int(
                result.get("identity_segment_candidate_count") or 0
            ),
            "phase4b_purity_probe_embedding_count": int(
                result.get("purity_probe_embedding_count") or 0
            ),
            "phase4b_rejected_identity_purity_segment_count": len(
                result.get("rejected_identity_purity_segments") or []
            ),
            "phase4b_exhaustion_reason": result.get("exhaustion_reason"),
            "memory_revision_path": str(memory_path),
            "memory_revision_sha256": memory_sha,
            "reference_count": int(memory.get("reference_count") or len(memory.get("references") or [])),
            "scoring_reference_count": int(memory.get("scoring_reference_count") or 0),
            "candidate_scoring_generation": generation,
            "backend_memory_used_by_phase4b_scoring": True,
            "backend_memory_used_by_provided_e2e_scoring": False,
            "cross_shot_scoring_performed": True,
            "automatic_target_confirmation": False,
            "phase4b_report_path": str(report_path),
            "phase4b_multi_shot_continuation_enabled": True,
            "phase4b_next_shot_id": (
                str(remaining_ready_shot_ids[0]) if remaining_ready_shot_ids else None
            ),
        }
    )
    state["runtime"] = runtime
    state.setdefault("shot_search_results", {})[str(shot["shot_id"])] = {
        "status": str((result.get("gate") or {}).get("operational_state") or ""),
        "candidate_count": int(result.get("candidate_count") or 0),
        "ranked_candidates": list(result.get("ranked_candidates") or []),
        "review_catalog_policy": str(
            result.get("review_catalog_policy") or PHASE4B_REVIEW_CATALOG_POLICY
        ),
        "review_catalog_candidates": list(
            result.get("review_catalog_candidates") or []
        ),
        "review_catalog_path": result.get("review_catalog_path"),
        "review_catalog_contact_sheet": result.get("review_catalog_contact_sheet_path"),
        "manual_review_promotion_allowed": bool(
            result.get("manual_review_promotion_allowed") is True
        ),
        "group_representative_policy": str(
            result.get("group_representative_policy") or PHASE4B_GROUP_REPRESENTATIVE_POLICY
        ),
        "group_confidence_policy": str(
            result.get("group_confidence_policy") or PHASE4B_GROUP_CONFIDENCE_POLICY
        ),
        "gate": dict(result.get("gate") or {}),
        "assignments": result.get("assignments_path"),
        "contact_sheet": result.get("contact_sheet_path"),
        "all_contact_sheet": result.get("all_contact_sheet_path"),
        "negative_role_memory": dict(result.get("negative_role_memory") or {}),
        "identity_negative_memory": dict(
            result.get("identity_negative_memory") or {}
        ),
        "persistent_role_negative_memory": dict(
            result.get("persistent_role_negative_memory") or {}
        ),
        "negative_memory_policy": str(result.get("negative_memory_policy") or ""),
        "rejected_role_confusion_tracklets": list(
            result.get("rejected_role_confusion_tracklets") or []
        ),
        "rejected_negative_margin_candidates": list(
            result.get("rejected_negative_margin_candidates") or []
        ),
        "rescue_review_policy": str(
            result.get("rescue_review_policy") or ""
        ),
        "rescue_review_candidates": list(
            result.get("rescue_review_candidates") or []
        ),
        "strict_review_candidates": list(
            result.get("strict_review_candidates") or []
        ),
        "unselected_plausible_candidates": list(
            result.get("unselected_plausible_candidates") or []
        ),
        "continuation_groups": list(result.get("continuation_groups") or []),
        "target_context_policy": str(
            result.get("target_context_policy") or PHASE4B_TARGET_CONTEXT_POLICY
        ),
        "target_context_is_soft_evidence_only": bool(
            result.get("target_context_is_soft_evidence_only") is True
        ),
        "corroboration_policy": str(
            result.get("corroboration_policy") or PHASE4B_CORROBORATION_POLICY
        ),
        "corroboration_groups": list(result.get("corroboration_groups") or []),
        "assisted_review_selection_policy": str(
            result.get("assisted_review_selection_policy") or ""
        ),
        "identity_negative_scoring_policy": str(
            result.get("identity_negative_scoring_policy") or ""
        ),
        "user_confirmed_role_scoring_policy": str(
            result.get("user_confirmed_role_scoring_policy") or ""
        ),
        "detector_role_scoring_policy": str(
            result.get("detector_role_scoring_policy") or ""
        ),
        "detector_role_evidence_is_soft_only": bool(
            result.get("detector_role_evidence_is_soft_only") is True
        ),
        "role_negative_scoring_policy": str(
            result.get("role_negative_scoring_policy") or ""
        ),
        "deduplicated_candidates": list(
            result.get("deduplicated_candidates") or []
        ),
        "identity_observability_policy": str(
            result.get("identity_observability_policy") or ""
        ),
        "rejected_identity_observability_candidates": list(
            result.get("rejected_identity_observability_candidates") or []
        ),
        "identity_purity_policy": str(
            result.get("identity_purity_policy") or ""
        ),
        "parent_tracklet_summaries": list(
            result.get("parent_tracklet_summaries") or []
        ),
        "split_parent_tracklet_count": int(
            result.get("split_parent_tracklet_count") or 0
        ),
        "identity_segment_candidate_count": int(
            result.get("identity_segment_candidate_count") or 0
        ),
        "purity_probe_embedding_count": int(
            result.get("purity_probe_embedding_count") or 0
        ),
        "rejected_identity_purity_segments": list(
            result.get("rejected_identity_purity_segments") or []
        ),
        "exhaustion_reason": result.get("exhaustion_reason"),
        "phase4b_policy": PHASE4B_POLICY,
        "memory_revision_id": str(memory["memory_revision_id"]),
        "memory_revision_sha256": memory_sha,
        "candidate_scoring_generation": generation,
    }
    shot_row = next(
        (row for row in state.get("shots") or [] if isinstance(row, dict) and str(row.get("shot_id")) == str(shot["shot_id"])),
        None,
    )
    if review_candidates:
        ambiguity_id = (
            f"ambiguity_phase4b_{_safe_name(shot['shot_id'])}_g{generation:03d}"
        )
        ambiguity = {
            "ambiguity_id": ambiguity_id,
            "shot_id": str(shot["shot_id"]),
            "shot_index": int(shot["shot_index"]),
            "start_frame": int(shot["start_frame"]),
            "end_frame_inclusive": int(shot["end_frame_inclusive"]),
            "status": "PENDING",
            "operational_state": str((result.get("gate") or {}).get("operational_state") or "AMBIGUOUS"),
            "decision": "AUTHORIZE_USER_CONFIRMATION_FALLBACK",
            "recommended_candidate": None,
            "review_candidates": review_candidates,
            "all_ranked_candidate_ids": [
                str(row.get("candidate_id")) for row in result.get("ranked_candidates") or []
            ],
            "review_catalog_policy": PHASE4B_REVIEW_CATALOG_POLICY,
            "review_catalog_candidate_ids": [
                str(row.get("candidate_id") or "")
                for row in result.get("review_catalog_candidates") or []
                if isinstance(row, Mapping)
            ],
            "review_catalog_path": result.get("review_catalog_path"),
            "review_catalog_contact_sheet": result.get("review_catalog_contact_sheet_path"),
            "manual_review_promotion_allowed": True,
            "review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
            "contact_sheet": result.get("contact_sheet_path"),
            "assignments": result.get("assignments_path"),
            "safe_gate": dict(result.get("gate") or {}),
            "score_evidence": {
                "memory_revision_id": str(memory["memory_revision_id"]),
                "memory_revision_sha256": memory_sha,
                "memory_source_reference_count": int(memory.get("reference_count") or len(memory.get("references") or [])),
                "candidate_scoring_generation": generation,
                "backend_memory_used_by_phase4b_scoring": True,
                "negative_role_memory": dict(
                    result.get("negative_role_memory") or {}
                ),
                "identity_negative_memory": dict(
                    result.get("identity_negative_memory") or {}
                ),
                "persistent_role_negative_memory": dict(
                    result.get("persistent_role_negative_memory") or {}
                ),
                "negative_memory_policy": str(
                    result.get("negative_memory_policy") or ""
                ),
                "identity_negative_scoring_policy": str(
                    result.get("identity_negative_scoring_policy") or ""
                ),
                "user_confirmed_role_scoring_policy": str(
                    result.get("user_confirmed_role_scoring_policy") or ""
                ),
                "detector_role_scoring_policy": str(
                    result.get("detector_role_scoring_policy") or ""
                ),
                "detector_role_evidence_is_soft_only": bool(
                    result.get("detector_role_evidence_is_soft_only") is True
                ),
                "assisted_review_selection_policy": str(
                    result.get("assisted_review_selection_policy") or ""
                ),
                "continuation_groups": list(
                    result.get("continuation_groups") or []
                ),
                "target_context_policy": str(
                    result.get("target_context_policy") or PHASE4B_TARGET_CONTEXT_POLICY
                ),
                "target_context_is_soft_evidence_only": bool(
                    result.get("target_context_is_soft_evidence_only") is True
                ),
                "corroboration_policy": str(
                    result.get("corroboration_policy") or PHASE4B_CORROBORATION_POLICY
                ),
                "corroboration_groups": list(
                    result.get("corroboration_groups") or []
                ),
                "role_negative_scoring_policy": str(
                    result.get("role_negative_scoring_policy") or ""
                ),
                "rescue_review_policy": str(
                    result.get("rescue_review_policy") or ""
                ),
                "rescue_review_candidate_ids": [
                    str(row.get("candidate_id") or "")
                    for row in result.get("rescue_review_candidates") or []
                    if isinstance(row, Mapping)
                ],
                "identity_purity_policy": str(
                    result.get("identity_purity_policy") or ""
                ),
                "split_parent_tracklet_count": int(
                    result.get("split_parent_tracklet_count") or 0
                ),
                "identity_segment_candidate_count": int(
                    result.get("identity_segment_candidate_count") or 0
                ),
                "rejected_identity_purity_segment_count": len(
                    result.get("rejected_identity_purity_segments") or []
                ),
            },
            "automatic_target_confirmation": False,
            "created_at": now_iso(),
        }
        prior = [
            row
            for row in state.get("ambiguities") or []
            if not (isinstance(row, Mapping) and str(row.get("ambiguity_id")) == ambiguity_id)
        ]
        state["ambiguities"] = [*prior, ambiguity]
        state["pending_action"] = {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": ambiguity_id,
            "shot_id": str(shot["shot_id"]),
            "contact_sheet": result.get("contact_sheet_path"),
            "candidate_ids": [str(row["candidate_id"]) for row in review_candidates],
            "recommended_candidate": None,
            "rescue_review_policy": str(
                result.get("rescue_review_policy") or ""
            ),
            "assisted_review_selection_policy": str(
                result.get("assisted_review_selection_policy") or ""
            ),
            "rescue_review_candidate_ids": [
                str(row.get("candidate_id") or "")
                for row in result.get("rescue_review_candidates") or []
                if isinstance(row, Mapping)
            ],
            "review_catalog_policy": PHASE4B_REVIEW_CATALOG_POLICY,
            "review_catalog_candidate_ids": [
                str(row.get("candidate_id") or "")
                for row in result.get("review_catalog_candidates") or []
                if isinstance(row, Mapping)
            ],
            "review_catalog_path": result.get("review_catalog_path"),
            "review_catalog_contact_sheet": result.get("review_catalog_contact_sheet_path"),
            "manual_review_promotion_allowed": True,
            "review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
            "automatic_target_confirmation": False,
        }
        state["status"] = "NEEDS_CONFIRMATION"
        state["decision"] = "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION"
        if shot_row is not None:
            shot_row["status"] = "AMBIGUOUS_REVIEW_REQUIRED"
    else:
        state["pending_action"] = None
        if shot_row is not None:
            if result.get("exhaustion_reason") == "UNREVIEWABLE_GROUP_OCCLUSION":
                shot_row["status"] = PHASE4B_UNREVIEWABLE_GROUP_OCCLUSION_STATUS
                shot_row["phase4b_unreviewable_group_occlusion"] = True
            else:
                shot_row["status"] = PHASE4B_SEARCH_EXHAUSTED_STATUS
                if (
                    result.get("exhaustion_reason")
                    == "UNREVIEWABLE_MIXED_IDENTITY_TRACKLETS"
                ):
                    shot_row["phase4b_unreviewable_mixed_identity_tracklets"] = True
            shot_row["phase4b_exhausted_without_reviewable_candidate"] = True
        if remaining_ready_shot_ids:
            state["status"] = "RUNNING"
            state["decision"] = PHASE4B_CONTINUE_DECISION
        else:
            state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
            state["decision"] = PHASE4B_ALL_EXHAUSTED_DECISION
    state["updated_at"] = now_iso()
    return state


def _phase4b_build_attempt_report(
    *,
    shot: Mapping[str, Any],
    memory: Mapping[str, Any],
    memory_path: Path,
    memory_sha: str,
    all_refs: Sequence[Mapping[str, Any]],
    scoring_refs: Sequence[Mapping[str, Any]],
    generation: int,
    detector_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    result: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": PHASE4B_ATTEMPT_SCHEMA,
        "created_at": now_iso(),
        "status": "PASS",
        "decision": (
            "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION"
            if result.get("review_candidates")
            else "EXHAUST_SHOT_AND_CONTINUE_PHASE4B"
        ),
        "policy": PHASE4B_POLICY,
        "shot_id": str(shot["shot_id"]),
        "shot_index": int(shot["shot_index"]),
        "shot_start_frame": int(shot["start_frame"]),
        "shot_end_frame_inclusive": int(shot["end_frame_inclusive"]),
        "memory_revision_id": str(memory["memory_revision_id"]),
        "memory_revision_path": str(memory_path),
        "memory_revision_sha256": memory_sha,
        "memory_total_reference_count": len(all_refs),
        "memory_scoring_reference_count": len(scoring_refs),
        "memory_pending_review_reference_count": len(all_refs) - len(scoring_refs),
        "negative_memory_available": bool(runtime["negative_memory_available"]),
        "negative_memory_policy": str(runtime.get("negative_memory_policy") or ""),
        "identity_negative_memory": dict(
            runtime.get("identity_negative_memory") or {}
        ),
        "persistent_role_negative_memory": dict(
            runtime.get("persistent_role_negative_memory") or {}
        ),
        "negative_role_memory": dict(runtime.get("negative_role_memory") or {}),
        "combined_negative_embeddings_path": str(
            runtime.get("combined_negative_embeddings_path") or ""
        ),
        "combined_negative_embeddings_sha256": str(
            runtime.get("combined_negative_embeddings_sha256") or ""
        ),
        "candidate_scoring_generation": generation,
        "candidate_count": int(result.get("candidate_count") or 0),
        "raw_tracklet_count": int(result.get("raw_tracklet_count") or 0),
        "unique_candidate_count_before_negative_gate": int(
            result.get("unique_candidate_count_before_negative_gate") or 0
        ),
        "review_candidate_count": len(result.get("review_candidates") or []),
        "review_candidates": list(result.get("review_candidates") or []),
        "selected_review_candidate_ids": [
            str(row.get("candidate_id") or "")
            for row in result.get("review_candidates") or []
            if isinstance(row, Mapping)
        ],
        "review_catalog_policy": str(
            result.get("review_catalog_policy") or PHASE4B_REVIEW_CATALOG_POLICY
        ),
        "review_catalog_candidate_count": len(
            result.get("review_catalog_candidates") or []
        ),
        "review_catalog_candidates": list(
            result.get("review_catalog_candidates") or []
        ),
        "review_catalog_candidate_ids": [
            str(row.get("candidate_id") or "")
            for row in result.get("review_catalog_candidates") or []
            if isinstance(row, Mapping)
        ],
        "review_catalog_path": result.get("review_catalog_path"),
        "review_catalog_sha256": result.get("review_catalog_sha256"),
        "review_catalog_contact_sheet_path": result.get("review_catalog_contact_sheet_path"),
        "review_catalog_contact_sheet_sha256": result.get("review_catalog_contact_sheet_sha256"),
        "manual_review_promotion_allowed": bool(
            result.get("manual_review_promotion_allowed") is True
        ),
        "review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
        "group_representative_policy": str(
            result.get("group_representative_policy") or PHASE4B_GROUP_REPRESENTATIVE_POLICY
        ),
        "group_confidence_policy": str(
            result.get("group_confidence_policy") or PHASE4B_GROUP_CONFIDENCE_POLICY
        ),
        "rejected_role_confusion_tracklet_count": len(
            result.get("rejected_role_confusion_tracklets") or []
        ),
        "rejected_negative_margin_candidate_count": len(
            result.get("rejected_negative_margin_candidates") or []
        ),
        "rejected_negative_margin_candidates": list(
            result.get("rejected_negative_margin_candidates") or []
        ),
        "rescue_review_policy": str(
            result.get("rescue_review_policy") or PHASE4B_RESCUE_REVIEW_POLICY
        ),
        "rescue_review_candidate_count": len(
            result.get("rescue_review_candidates") or []
        ),
        "rescue_review_candidates": list(
            result.get("rescue_review_candidates") or []
        ),
        "strict_review_candidate_count": len(
            result.get("strict_review_candidates") or []
        ),
        "unselected_plausible_candidate_count": len(
            result.get("unselected_plausible_candidates") or []
        ),
        "unselected_plausible_candidates": list(
            result.get("unselected_plausible_candidates") or []
        ),
        "continuation_group_count": len(
            result.get("continuation_groups") or []
        ),
        "continuation_groups": list(result.get("continuation_groups") or []),
        "target_context_policy": str(
            result.get("target_context_policy") or PHASE4B_TARGET_CONTEXT_POLICY
        ),
        "target_context_is_soft_evidence_only": bool(
            result.get("target_context_is_soft_evidence_only") is True
        ),
        "corroboration_policy": str(
            result.get("corroboration_policy") or PHASE4B_CORROBORATION_POLICY
        ),
        "corroboration_group_count": len(
            result.get("corroboration_groups") or []
        ),
        "corroboration_groups": list(result.get("corroboration_groups") or []),
        "assisted_review_selection_policy": str(
            result.get("assisted_review_selection_policy")
            or PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY
        ),
        "identity_negative_scoring_policy": str(
            result.get("identity_negative_scoring_policy")
            or PHASE4B_IDENTITY_NEGATIVE_SCORING_POLICY
        ),
        "user_confirmed_role_scoring_policy": str(
            result.get("user_confirmed_role_scoring_policy")
            or PHASE4B_USER_CONFIRMED_ROLE_SCORING_POLICY
        ),
        "detector_role_scoring_policy": str(
            result.get("detector_role_scoring_policy")
            or PHASE4B_DETECTOR_ROLE_SCORING_POLICY
        ),
        "detector_role_evidence_is_soft_only": bool(
            result.get("detector_role_evidence_is_soft_only") is True
        ),
        "role_negative_scoring_policy": str(
            result.get("role_negative_scoring_policy")
            or PHASE4B_ROLE_NEGATIVE_SCORING_POLICY
        ),
        "deduplicated_candidate_count": len(
            result.get("deduplicated_candidates") or []
        ),
        "identity_observability_policy": str(
            result.get("identity_observability_policy") or ""
        ),
        "identity_observability_raw_candidate_count": int(
            result.get("identity_observability_raw_candidate_count") or 0
        ),
        "identity_observability_passed_candidate_count": int(
            result.get("identity_observability_passed_candidate_count") or 0
        ),
        "rejected_identity_observability_candidate_count": len(
            result.get("rejected_identity_observability_candidates") or []
        ),
        "rejected_identity_observability_candidates": list(
            result.get("rejected_identity_observability_candidates") or []
        ),
        "identity_purity_policy": str(
            result.get("identity_purity_policy") or PHASE4B_IDENTITY_PURITY_POLICY
        ),
        "parent_tracklet_count_before_purity_segmentation": len(
            result.get("parent_tracklet_summaries") or []
        ),
        "split_parent_tracklet_count": int(
            result.get("split_parent_tracklet_count") or 0
        ),
        "identity_segment_candidate_count": int(
            result.get("identity_segment_candidate_count") or 0
        ),
        "purity_probe_embedding_count": int(
            result.get("purity_probe_embedding_count") or 0
        ),
        "rejected_identity_purity_segment_count": len(
            result.get("rejected_identity_purity_segments") or []
        ),
        "parent_tracklet_summaries": list(
            result.get("parent_tracklet_summaries") or []
        ),
        "rejected_identity_purity_segments": list(
            result.get("rejected_identity_purity_segments") or []
        ),
        "exhaustion_reason": result.get("exhaustion_reason"),
        "candidate_role_filter": dict(result.get("candidate_role_filter") or {}),
        "raw_detection_count": int(runtime.get("raw_detection_count") or 0),
        "analysis_raw_detection_count": int(
            runtime.get("analysis_raw_detection_count") or 0
        ),
        "eligible_candidate_detection_count": int(runtime.get("detection_count") or 0),
        "detector_contract": dict(detector_contract),
        "model_contract": dict(runtime.get("model_contract") or {}),
        "sports_osnet_checkpoint_path": str(runtime["checkpoint"]),
        "sports_osnet_checkpoint_sha256": sha256_file(runtime["checkpoint"]),
        "target_embeddings_path": str(runtime["target_embeddings_path"]),
        "target_embeddings_sha256": str(runtime["target_embeddings_sha256"]),
        "target_prototype_path": str(runtime["target_prototype_path"]),
        "target_prototype_sha256": str(runtime["target_prototype_sha256"]),
        "ranked_candidates_path": result.get("ranked_candidates_path"),
        "ranked_candidates_sha256": result.get("ranked_candidates_sha256"),
        "contact_sheet_path": result.get("contact_sheet_path"),
        "contact_sheet_sha256": result.get("contact_sheet_sha256"),
        "all_contact_sheet_path": result.get("all_contact_sheet_path"),
        "all_contact_sheet_sha256": result.get("all_contact_sheet_sha256"),
        "identity_observability_rejection_sheet_path": result.get(
            "identity_observability_rejection_sheet_path"
        ),
        "identity_observability_rejection_sheet_sha256": result.get(
            "identity_observability_rejection_sheet_sha256"
        ),
        "identity_purity_rejection_sheet_path": result.get(
            "identity_purity_rejection_sheet_path"
        ),
        "identity_purity_rejection_sheet_sha256": result.get(
            "identity_purity_rejection_sheet_sha256"
        ),
        "safe_gate": dict(result.get("gate") or {}),
        "backend_memory_used_by_phase4b_scoring": True,
        "backend_memory_used_by_provided_e2e_scoring": False,
        "cross_shot_scoring_performed": True,
        "candidate_link_created": False,
        "synthetic_tracking_used": False,
        "automatic_target_confirmation": False,
        "runtime_seconds": runtime_seconds,
    }


def run_phase4b_first_cross_shot(
    *,
    root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    memory_path: Path,
    memory_sha: str,
) -> dict[str, Any]:
    """Search reviewed shots in order until confirmation or safe exhaustion.

    The historical function name and CLI flag are kept for backend compatibility.
    R11 keeps the provenance-aware negative gates and target-agnostic identity
    purity segmentation, removes forced time slicing, and adds two soft review
    signals: approved-reference torso context and independent-tracklet
    spatiotemporal corroboration. Neither signal can auto-confirm the target.
    """

    import time

    report_path = output_dir / "phase4b_first_cross_shot_report.json"
    state_path = output_dir / "pipeline_state.json"
    state = read_object(state_path)

    if state.get("pending_action") is not None and report_path.is_file():
        existing = read_object(report_path)
        if existing.get("status") == "PASS" and existing.get("policy") == PHASE4B_POLICY:
            return existing

    stale_review_invalidated = _phase4b_invalidate_stale_candidate_review(state)
    reopened_stale_terminal_ids = (
        _phase4b_reopen_stale_terminal_results_for_rescoring(state)
    )
    migrated_no_candidate = _phase4b_migrate_prior_no_candidate_results(state)
    if (
        stale_review_invalidated
        or reopened_stale_terminal_ids
        or migrated_no_candidate
    ):
        atomic_json(state_path, state)

    runtime_state = dict(state.get("runtime") or {})
    if runtime_state.get("phase4a_initial_memory_review_status") != "PASS":
        raise RuntimeError("Phase 4-B requires an approved Phase 4-A memory review.")
    if runtime_state.get("phase4b_cross_shot_scoring_authorized") is not True:
        raise RuntimeError("Phase 4-B scoring has not been authorized.")
    if state.get("pending_action") is not None:
        raise RuntimeError("Phase 4-B cannot start while another review is pending.")

    memory_path = memory_path.resolve()
    validate_input_file(memory_path, memory_sha, "approved Phase 4-A memory")
    memory = read_object(memory_path)
    if memory.get("schema_version") != PHASE4A_MEMORY_SCHEMA:
        raise RuntimeError("Unsupported Phase 4-A memory schema.")
    if memory.get("automatic_target_confirmation") is not False:
        raise RuntimeError("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")
    scoring_refs, all_refs = _phase4b_scoring_references(memory)

    ready_shots = _phase4b_ready_shots(state)
    if not ready_shots:
        raise RuntimeError("No remaining reviewed shot is ready for Phase 4-B continuation.")

    generation = max(
        int(args.candidate_scoring_generation or 1),
        int(runtime_state.get("candidate_scoring_generation") or 1),
    )
    attempts: list[dict[str, Any]] = []
    exhausted_shot_ids: list[str] = []
    stop_reason = "ALL_REMAINING_SHOTS_EXHAUSTED"

    for position, shot in enumerate(ready_shots):
        generation += 1
        started = time.perf_counter()
        clip, clip_meta, detections_path, detector_contract = _phase4b_prepare_detector_source(
            root=root,
            output_dir=output_dir,
            state=state,
            shot=shot,
            device=args.device,
            overwrite=args.overwrite,
        )
        runtime = _phase4b_load_runtime(
            root=root,
            state=state,
            shot=shot,
            clip=clip,
            clip_meta=clip_meta,
            detections_path=detections_path,
            memory=memory,
            scoring_references=scoring_refs,
            device_name=args.device,
            output_dir=output_dir,
        )
        result = _phase4b_process_candidates(
            output_dir=output_dir,
            state=state,
            shot=shot,
            runtime=runtime,
            memory=memory,
            memory_path=memory_path,
            memory_sha=memory_sha,
            generation=generation,
        )
        attempt = _phase4b_build_attempt_report(
            shot=shot,
            memory=memory,
            memory_path=memory_path,
            memory_sha=memory_sha,
            all_refs=all_refs,
            scoring_refs=scoring_refs,
            generation=generation,
            detector_contract=detector_contract,
            runtime=runtime,
            result=result,
            runtime_seconds=time.perf_counter() - started,
        )
        attempt_path = (
            output_dir
            / "phase4b_cross_shot"
            / _safe_name(shot["shot_id"])
            / "phase4b_attempt_report.json"
        )
        atomic_json(attempt_path, attempt)
        attempt["attempt_report_path"] = str(attempt_path)
        attempt["attempt_report_sha256"] = sha256_file(attempt_path)
        attempts.append(attempt)

        remaining_ids = [
            str(row["shot_id"])
            for row in ready_shots[position + 1 :]
        ]
        state = _phase4b_apply_result_to_state(
            state=state,
            shot=shot,
            result=result,
            memory=memory,
            memory_path=memory_path,
            memory_sha=memory_sha,
            generation=generation,
            report_path=attempt_path,
            remaining_ready_shot_ids=remaining_ids,
        )
        atomic_json(state_path, state)

        if result.get("review_candidates"):
            stop_reason = "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION"
            break
        exhausted_shot_ids.append(str(shot["shot_id"]))

    last = attempts[-1]
    report = {
        **last,
        "schema_version": PHASE4B_REPORT_SCHEMA,
        "created_at": now_iso(),
        "decision": str(state.get("decision") or ""),
        "policy": PHASE4B_POLICY,
        "multi_shot_continuation": True,
        "reopened_stale_terminal_shot_ids": list(
            reopened_stale_terminal_ids
        ),
        "attempted_shot_count": len(attempts),
        "attempted_shot_ids": [str(row["shot_id"]) for row in attempts],
        "exhausted_shot_ids": exhausted_shot_ids,
        "stop_reason": stop_reason,
        "first_reviewable_candidate_shot_id": (
            str(last["shot_id"])
            if int(last.get("review_candidate_count") or 0) > 0
            else None
        ),
        "all_remaining_shots_exhausted": stop_reason == "ALL_REMAINING_SHOTS_EXHAUSTED",
        "attempts": attempts,
        "candidate_scoring_generation_start": int(attempts[0]["candidate_scoring_generation"]),
        "candidate_scoring_generation_end": int(attempts[-1]["candidate_scoring_generation"]),
        "automatic_target_confirmation": False,
        "candidate_link_created": False,
        "synthetic_tracking_used": False,
    }
    atomic_json(report_path, report)
    report_sha = sha256_file(report_path)

    state_runtime = dict(state.get("runtime") or {})
    state_runtime.update(
        {
            "phase4b_policy": PHASE4B_POLICY,
            "phase4b_report_path": str(report_path),
            "phase4b_report_sha256": report_sha,
            "phase4b_multi_shot_continuation_enabled": True,
            "phase4b_attempted_shot_ids": list(report["attempted_shot_ids"]),
            "phase4b_exhausted_shot_ids": list(report["exhausted_shot_ids"]),
            "phase4b_stop_reason": stop_reason,
            "phase4b_identity_purity_policy": str(
                last.get("identity_purity_policy") or PHASE4B_IDENTITY_PURITY_POLICY
            ),
            "phase4b_parent_tracklet_count_before_purity_segmentation": int(
                last.get("parent_tracklet_count_before_purity_segmentation") or 0
            ),
            "phase4b_split_parent_tracklet_count": int(
                last.get("split_parent_tracklet_count") or 0
            ),
            "phase4b_identity_segment_candidate_count": int(
                last.get("identity_segment_candidate_count") or 0
            ),
            "phase4b_rejected_identity_purity_segment_count": int(
                last.get("rejected_identity_purity_segment_count") or 0
            ),
            "candidate_scoring_generation": generation,
            "backend_memory_used_by_phase4b_scoring": True,
            "automatic_target_confirmation": False,
        }
    )
    state["runtime"] = state_runtime
    atomic_json(state_path, state)

    summary_path = output_dir / "pipeline_summary.json"
    summary = read_object(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "status": state["status"],
            "decision": state["decision"],
            "phase4b_policy": PHASE4B_POLICY,
            "phase4b_multi_shot_continuation": True,
            "phase4b_attempted_shot_ids": list(report["attempted_shot_ids"]),
            "phase4b_exhausted_shot_ids": list(report["exhausted_shot_ids"]),
            "phase4b_stop_reason": stop_reason,
            "phase4b_scored_shot_id": str(last["shot_id"]),
            "phase4b_candidate_count": int(last.get("candidate_count") or 0),
            "phase4b_review_candidate_count": int(last.get("review_candidate_count") or 0),
            "phase4b_identity_purity_policy": str(
                last.get("identity_purity_policy") or PHASE4B_IDENTITY_PURITY_POLICY
            ),
            "phase4b_parent_tracklet_count_before_purity_segmentation": int(
                last.get("parent_tracklet_count_before_purity_segmentation") or 0
            ),
            "phase4b_split_parent_tracklet_count": int(
                last.get("split_parent_tracklet_count") or 0
            ),
            "phase4b_identity_segment_candidate_count": int(
                last.get("identity_segment_candidate_count") or 0
            ),
            "phase4b_rejected_identity_purity_segment_count": int(
                last.get("rejected_identity_purity_segment_count") or 0
            ),
            "candidate_scoring_generation": generation,
            "backend_memory_used_by_phase4b_scoring": True,
            "automatic_target_confirmation": False,
        }
    )
    atomic_json(summary_path, summary)
    report["report_path"] = str(report_path)
    report["report_sha256"] = report_sha
    return report

def _candidate_evidence(
    *,
    root: Path,
    output_dir: Path,
    raw_dir: Path,
    raw_state: Mapping[str, Any],
    view: Mapping[str, Any],
    ambiguity: Mapping[str, Any],
    candidate: Mapping[str, Any],
    memory_path: Path | None,
    memory_sha: str | None,
    generation: int,
) -> dict[str, Any]:
    ambiguity_id = str(ambiguity["ambiguity_id"])
    candidate_id = str(candidate["candidate_id"])
    raw_shot_id = str(ambiguity["shot_id"])
    mapping = _shot_map_by_raw(view).get(raw_shot_id)
    if mapping is None:
        raise RuntimeError(f"No reviewed-shot mapping for runtime shot: {raw_shot_id}")
    assignments_path = Path(str(ambiguity.get("assignments") or "")).resolve()
    rows = [
        row for row in _read_csv_rows(assignments_path)
        if str(row.get("candidate_id") or "") == candidate_id
    ]
    if not rows:
        raise RuntimeError(f"Candidate has no runtime assignments: {candidate_id}")
    anchor_module = _load_module(
        "kickclip_r1_evidence_anchor",
        root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
    )
    selected, scored = anchor_module.choose_anchor(rows)
    offset = int(view["source_offset_frame"])
    observations = []
    for row in rows:
        local_frame = int(row["frame_index"])
        observations.append(
            {
                "frame_index": local_frame + offset,
                "runtime_local_frame_index": local_frame,
                "detection_id": str(row.get("detection_id") or ""),
                "confidence": float(row.get("confidence") or 0.0),
                "bbox_xyxy": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
            }
        )
    anchor_local = int(selected["frame_index"])
    anchor_global = anchor_local + offset
    evidence_dir = output_dir / "runtime_candidate_evidence" / _safe_name(ambiguity_id) / _safe_name(candidate_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source_video = Path(str(view["source_video"]["path"])).resolve()
    full_frame = evidence_dir / "full_frame_context.jpg"
    if not full_frame.is_file():
        _write_full_frame(source_video, anchor_global, full_frame)
    shot_clip = output_dir / "runtime_candidate_evidence" / "shots" / f"{_safe_name(mapping['source_shot_id'])}.mp4"
    if not shot_clip.is_file():
        _write_video_range(
            source_video,
            shot_clip,
            start_frame=int(mapping["source_start_frame"]),
            end_frame_inclusive=int(mapping["source_end_frame_inclusive"]),
            metadata=view["source_video"],
        )
    strip = raw_dir / "work" / "shots" / raw_shot_id / "candidate_strips" / f"{candidate_id}.jpg"
    gallery = evidence_dir / "reference_gallery.jpg"
    if strip.is_file():
        shutil.copy2(strip, gallery)
    else:
        contact_sheet = Path(str(ambiguity.get("contact_sheet") or "")).resolve()
        if not contact_sheet.is_file():
            raise RuntimeError(f"Candidate visual evidence is missing: {candidate_id}")
        shutil.copy2(contact_sheet, gallery)

    backend_memory = read_object(memory_path) if memory_path and memory_path.is_file() else {}
    runtime_memory = raw_state.get("memory") if isinstance(raw_state.get("memory"), Mapping) else {}
    manifest = {
        "schema_version": "kickclip.runtime_candidate_manifest.r1",
        "candidate_id": candidate_id,
        "ambiguity_id": ambiguity_id,
        "shot_id": str(mapping["source_shot_id"]),
        "runtime_shot_id": raw_shot_id,
        "tracklet_id": candidate_id,
        "parent_tracklet_id": candidate.get("parent_tracklet_id") or candidate_id,
        "identity_pure": bool(
            dict(candidate.get("identity_purity") or {}).get("passed") is True
        ),
        "identity_purity_source": str(
            dict(candidate.get("identity_purity") or {}).get("policy")
            or PHASE4B_IDENTITY_PURITY_POLICY
        ),
        "identity_purity": dict(candidate.get("identity_purity") or {}),
        "start_frame": int(candidate.get("start_frame", rows[0]["frame_index"])) + offset,
        "end_frame_inclusive": int(candidate.get("end_frame_inclusive", rows[-1]["frame_index"])) + offset,
        "best_observation": {
            "frame_index": anchor_global,
            "runtime_local_frame_index": anchor_local,
            "bbox_xyxy": [
                float(selected["x1"]),
                float(selected["y1"]),
                float(selected["x2"]),
                float(selected["y2"]),
            ],
            "stage3b3_scored_observations": scored,
        },
        "observations": observations,
        "ranking_metrics": dict(candidate),
        "evidence": {
            "full_frame_context_path": str(full_frame),
            "full_frame_context_sha256": sha256_file(full_frame),
            "shot_clip_path": str(shot_clip),
            "shot_clip_sha256": sha256_file(shot_clip),
            "reference_gallery_path": str(gallery),
            "reference_gallery_sha256": sha256_file(gallery),
            "assignments_path": str(assignments_path),
            "assignments_sha256": sha256_file(assignments_path),
        },
        "score_evidence": {
            "candidate_scoring_generation": generation,
            "runtime_memory_source": dict(runtime_memory),
            "backend_memory_revision_path": str(memory_path) if memory_path else None,
            "backend_memory_revision_sha256": memory_sha,
            "backend_memory_reference_count": len(backend_memory.get("references") or []),
            "backend_memory_used_by_provided_e2e_scoring": bool(
                runtime_memory.get("backend_memory_used_by_provided_e2e_scoring")
            ),
        },
        "automatic_target_confirmation": False,
    }
    manifest_path = evidence_dir / "candidate_manifest.json"
    manifest_sha = None
    atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    return {
        **dict(candidate),
        "candidate_id": candidate_id,
        "shot_id": str(mapping["source_shot_id"]),
        "runtime_shot_id": raw_shot_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "full_frame_context_path": str(full_frame),
        "full_frame_context_sha256": sha256_file(full_frame),
        "shot_clip_path": str(shot_clip),
        "shot_clip_sha256": sha256_file(shot_clip),
        "reference_gallery_path": str(gallery),
        "reference_gallery_sha256": sha256_file(gallery),
        "best_frame": anchor_global,
        "best_bbox_xyxy": manifest["best_observation"]["bbox_xyxy"],
        "score_evidence": manifest["score_evidence"],
        "status": "PENDING",
    }


def _normalize_timeline(
    *,
    raw_timeline: Path,
    output_timeline: Path,
    view: Mapping[str, Any],
) -> None:
    raw = read_object(raw_timeline)
    source = view["source_video"]
    frame_count = int(source["frame_count"])
    fps = float(source["fps"])
    frames = [
        {
            "frame_index": index,
            "time_seconds": index / fps,
            "shot_id": _shot_for_global_frame(view, index),
            "state": "SEARCHING",
            "bbox_xyxy": None,
            "tracking_confidence": 0.0,
            "identity_confidence": 0.0,
            "identity_source": "NONE",
            "selected_detection_id": None,
            "decision_reason": "OUTSIDE_SELECTION_ANCHOR_VIEW",
            "review_required": False,
            "ambiguity_id": None,
        }
        for index in range(frame_count)
    ]
    offset = int(view["source_offset_frame"])
    for raw_row in raw.get("frames") or []:
        if not isinstance(raw_row, Mapping):
            continue
        local = int(raw_row.get("frame_index", -1))
        global_index = local + offset
        if not 0 <= global_index < frame_count:
            continue
        row = dict(raw_row)
        row["frame_index"] = global_index
        row["time_seconds"] = global_index / fps
        row["shot_id"] = _shot_for_global_frame(view, global_index)
        row["runtime_local_frame_index"] = local
        frames[global_index] = row
    normalized_shots = _normalize_source_shots(
        raw_shots=list(raw.get("shots") or []),
        view=view,
    )
    result = {
        **raw,
        "video": dict(source),
        "shots": normalized_shots,
        "frames": frames,
        "provenance": {
            **(dict(raw.get("provenance") or {}) if isinstance(raw.get("provenance"), Mapping) else {}),
            "selection_anchor_view": {
                "source_offset_frame": offset,
                "unresolved_prefix_frame_count": int(view.get("unresolved_prefix_frame_count") or 0),
                "frame_zero_fallback_used": False,
                "reviewed_shot_contract_schema": str(
                    view.get("reviewed_shot_contract_schema") or ""
                ),
                "reviewed_shot_count": int(
                    view.get("reviewed_shot_count") or 0
                ),
                "reviewed_shot_boundaries_sha256": str(
                    view.get("reviewed_shot_boundaries_sha256") or ""
                ),
                "full_reviewed_shot_coverage_used": bool(
                    view.get("reviewed_shots")
                ),
            },
            "observation_copy_used_as_tracking_success": False,
            "synthetic_tracking_used": False,
        },
    }
    atomic_json(output_timeline, result)


def normalize_state(
    *,
    root: Path,
    output_dir: Path,
    raw_dir: Path,
    launch: Mapping[str, Any],
    view: Mapping[str, Any],
    memory_path: Path | None,
    memory_sha: str | None,
    generation: int,
) -> None:
    raw_state_path = raw_dir / "pipeline_state.json"
    if not raw_state_path.is_file():
        raise FileNotFoundError(raw_state_path)
    raw = read_object(raw_state_path)
    phase3c_contract = _phase3c_runtime_contract(output_dir=output_dir)
    initial_memory = build_initial_memory_revision(
        root=root,
        output_dir=output_dir,
        raw_state=raw,
        launch=launch,
        view=view,
    )
    phase4a: dict[str, Any] | None = None
    if (
        memory_path is None
        and initial_memory is None
        and phase3c_contract["complete"]
        and str(raw.get("decision") or "")
        == "SAFE_BLOCK_NO_STABLE_PRECUT_TARGET_MEMORY"
    ):
        phase4a = build_phase4a_initial_memory_from_phase3c(
            output_dir=output_dir,
            launch=launch,
            view=view,
        )
    effective_memory_path = (
        memory_path
        or (initial_memory[0] if initial_memory else None)
        or (
            Path(str(phase4a["memory_revision_path"])).resolve()
            if phase4a is not None
            else None
        )
    )
    effective_memory_sha = (
        memory_sha
        or (initial_memory[1] if initial_memory else None)
        or (
            str(phase4a["memory_revision_sha256"])
            if phase4a is not None
            else None
        )
    )
    mapping = _shot_map_by_raw(view)
    normalized_shots = _normalize_source_shots(
        raw_shots=list(raw.get("shots") or []),
        view=view,
    )
    if (
        not phase3c_contract["complete"]
        and int(view.get("unresolved_prefix_frame_count") or 0) > 0
    ):
        for row in normalized_shots:
            if (
                str(row.get("shot_id") or "")
                == str(view.get("selected_shot_id") or "")
                and str(row.get("status") or "").upper()
                in {"TARGET_CONFIRMED_AND_TRACKED", "ACCEPTED", "TRACKED"}
            ):
                row["runtime_forward_status"] = row["status"]
                row["status"] = "UNRESOLVED_SELECTION_PREFIX"

    normalized_ambiguities: list[dict[str, Any]] = []
    for raw_ambiguity in raw.get("ambiguities") or []:
        if not isinstance(raw_ambiguity, Mapping):
            continue
        ambiguity = dict(raw_ambiguity)
        raw_shot_id = str(ambiguity.get("shot_id") or "")
        source = mapping.get(raw_shot_id)
        if source:
            ambiguity["runtime_shot_id"] = raw_shot_id
            ambiguity["shot_id"] = source["source_shot_id"]
            ambiguity["start_frame"] = source["source_start_frame"]
            ambiguity["end_frame_inclusive"] = source["source_end_frame_inclusive"]
        candidates = []
        for candidate in ambiguity.get("review_candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            candidates.append(
                _candidate_evidence(
                    root=root,
                    output_dir=output_dir,
                    raw_dir=raw_dir,
                    raw_state=raw,
                    view=view,
                    ambiguity=raw_ambiguity,
                    candidate=candidate,
                    memory_path=effective_memory_path,
                    memory_sha=effective_memory_sha,
                    generation=generation,
                )
            )
        if candidates:
            ambiguity["review_candidates"] = candidates
        normalized_ambiguities.append(ambiguity)

    pending = dict(raw.get("pending_action") or {}) if isinstance(raw.get("pending_action"), Mapping) else None
    if pending:
        raw_shot_id = str(pending.get("shot_id") or "")
        source = mapping.get(raw_shot_id)
        if source:
            pending["runtime_shot_id"] = raw_shot_id
            pending["shot_id"] = source["source_shot_id"]

    phase3b_contract = _phase3b_runtime_contract(output_dir=output_dir)

    state = dict(raw)
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "execution_kind": "EVENT_CANDIDATE_HANDOFF_R1",
            "video": dict(view["source_video"]),
            "shots": normalized_shots,
            "runtime": {
                **(dict(raw.get("runtime") or {}) if isinstance(raw.get("runtime"), Mapping) else {}),
                "integration_path": "B.ADD_THIN_BACKEND_ADAPTER_TO_V1_V2_STAGES",
                "provided_e2e_runner_used": True,
                "selection_aware_video_adapter_used": True,
                "selection_anchor_view_path": str(output_dir / "_selection_anchor_view" / "selection_view.json"),
                "selection_anchor_view_sha256": sha256_file(output_dir / "_selection_anchor_view" / "selection_view.json"),
                "source_offset_frame": int(view["source_offset_frame"]),
                "unresolved_prefix_frame_count": int(view.get("unresolved_prefix_frame_count") or 0),
                "selected_shot_bidirectional_inputs_materialized": bool(
                    view.get("selected_shot_bidirectional_inputs_materialized")
                ),
                "selected_shot_forward_source_sha256": str(
                    (view.get("selected_shot_forward_source") or {}).get("sha256") or ""
                ),
                "selected_shot_backward_source_sha256": str(
                    (view.get("selected_shot_backward_source") or {}).get("sha256") or ""
                ),
                "reverse_phase1_execution_complete": bool(
                    phase3b_contract["complete"]
                ),
                "forward_phase1_execution_complete": bool(
                    phase3b_contract["complete"]
                ),
                "phase3b_bidirectional_phase1_report_path": (
                    phase3b_contract["report_path"]
                ),
                "phase3b_bidirectional_phase1_report_sha256": (
                    phase3b_contract["report_sha256"]
                ),
                "phase3b_pending_visual_review_count": (
                    phase3b_contract.get("pending_visual_review_count", 0)
                ),
                "timeline_merge_complete": bool(phase3c_contract["complete"]),
                "phase3c_bidirectional_timeline_merge_report_path": (
                    phase3c_contract["report_path"]
                ),
                "phase3c_bidirectional_timeline_merge_report_sha256": (
                    phase3c_contract["report_sha256"]
                ),
                "phase3c_selected_shot_timeline_path": (
                    phase3c_contract["timeline_path"]
                ),
                "phase3c_selected_shot_timeline_sha256": (
                    phase3c_contract["timeline_sha256"]
                ),
                "resolved_prefix_frame_count": int(
                    phase3c_contract.get("resolved_prefix_frame_count", 0)
                ),
                "reviewed_shot_boundaries_translated": True,
                "full_reviewed_shot_contract_emitted": bool(
                    view.get("reviewed_shots")
                ),
                "reviewed_shot_count": int(
                    view.get("reviewed_shot_count") or 0
                ),
                "frame_shot_ids_assigned_from_reviewed_boundaries": bool(
                    view.get("reviewed_shots")
                ),
                "synthetic_tracking_used": False,
                "observation_copy_used_as_success": False,
                "frame_zero_fallback_used": False,
                "automatic_target_confirmation": False,
                "memory_revision_path": (
                    str(effective_memory_path) if effective_memory_path else None
                ),
                "memory_revision_sha256": effective_memory_sha,
                "reference_count": (
                    len((read_object(effective_memory_path).get("references") or []))
                    if effective_memory_path and effective_memory_path.is_file()
                    else len(read_object(Path(str(launch["target_reference_set"]["path"]))).get("references") or [])
                ),
                "candidate_scoring_generation": generation,
                "backend_memory_used_by_provided_e2e_scoring": bool(
                    (raw.get("memory") or {}).get(
                        "backend_memory_used_by_provided_e2e_scoring"
                    )
                    if isinstance(raw.get("memory"), Mapping)
                    else False
                ),
            },
            "pending_action": pending,
            "ambiguities": normalized_ambiguities,
        }
    )
    if phase4a is not None:
        _activate_phase4a_memory_review_state(
            state=state,
            output_dir=output_dir,
            phase4a=phase4a,
            view=view,
        )

    if (
        str(state.get("status") or "") == "COMPLETE"
        and not phase3c_contract["complete"]
        and int(view.get("unresolved_prefix_frame_count") or 0) > 0
    ):
        state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
        state["decision"] = "E2E_COMPLETE_WITH_UNRESOLVED_SELECTION_PREFIX"
    atomic_json(output_dir / "pipeline_state.json", state)

    raw_timeline = raw_dir / "target_timeline.json"
    if raw_timeline.is_file():
        normalized_timeline_path = output_dir / "target_timeline.json"
        _normalize_timeline(
            raw_timeline=raw_timeline,
            output_timeline=normalized_timeline_path,
            view=view,
        )
        if phase3c_contract["complete"]:
            _overlay_phase3c_selected_shot_timeline(
                full_timeline_path=normalized_timeline_path,
                selected_shot_timeline_path=Path(
                    str(phase3c_contract["timeline_path"])
                ).resolve(),
                view=view,
            )
    for name in (
        "target_timeline.csv",
        "pipeline_summary.json",
        "pipeline_manifest.json",
        "full_frame_tracking_preview.mp4",
        "target_centered_preview.mp4",
    ):
        source = raw_dir / name
        target = output_dir / name
        if source.is_file():
            shutil.copy2(source, target)
    summary_path = output_dir / "pipeline_summary.json"
    summary = read_object(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "status": state.get("status"),
            "decision": state.get("decision"),
            "selection_anchor_view": True,
            "source_offset_frame": int(view["source_offset_frame"]),
            "unresolved_prefix_frame_count": int(view.get("unresolved_prefix_frame_count") or 0),
            "selected_shot_bidirectional_inputs_materialized": bool(
                view.get("selected_shot_bidirectional_inputs_materialized")
            ),
            "reverse_phase1_execution_complete": bool(
                phase3b_contract["complete"]
            ),
            "forward_phase1_execution_complete": bool(
                phase3b_contract["complete"]
            ),
            "phase3b_pending_visual_review_count": (
                phase3b_contract.get("pending_visual_review_count", 0)
            ),
            "timeline_merge_complete": bool(phase3c_contract["complete"]),
            "resolved_prefix_frame_count": int(
                phase3c_contract.get("resolved_prefix_frame_count", 0)
            ),
            "phase3c_selected_shot_timeline_sha256": (
                phase3c_contract.get("timeline_sha256")
            ),
            "backend_memory_used_by_provided_e2e_scoring": state["runtime"][
                "backend_memory_used_by_provided_e2e_scoring"
            ],
            "full_event_recommendation_e2e": "NOT_RUN",
        }
    )
    atomic_json(summary_path, summary)



def _phase4b_promote_review_candidate(
    *,
    output_dir: Path,
    ambiguity_id: str,
    candidate_id: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Promote a safe catalog candidate into the pending review set.

    Promotion is a review-navigation action only. It never confirms identity,
    creates a candidate link, or changes positive/negative memory.
    """
    state_path = output_dir / "pipeline_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = read_object(state_path)
    pending = dict(state.get("pending_action") or {})
    if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
        raise RuntimeError("Candidate promotion requires a pending cross-shot confirmation.")
    if str(pending.get("ambiguity_id") or "") != str(ambiguity_id):
        raise RuntimeError("Candidate promotion ambiguity does not match the pending action.")

    ambiguities = [
        row for row in state.get("ambiguities") or [] if isinstance(row, dict)
    ]
    ambiguity = next(
        (row for row in ambiguities if str(row.get("ambiguity_id") or "") == ambiguity_id),
        None,
    )
    if ambiguity is None or str(ambiguity.get("status") or "") != "PENDING":
        raise RuntimeError("Pending ambiguity record is missing or no longer reviewable.")
    shot_id = str(ambiguity.get("shot_id") or pending.get("shot_id") or "")
    if not shot_id:
        raise RuntimeError("Pending ambiguity has no shot id.")
    shot_root = output_dir / "phase4b_cross_shot" / _safe_name(shot_id)
    ranked_path = shot_root / "ranked_candidates.json"
    attempt_path = shot_root / "phase4b_attempt_report.json"
    if not ranked_path.is_file() or not attempt_path.is_file():
        raise FileNotFoundError("Current Phase 4-B ranked/attempt artifacts are missing.")
    attempt = read_object(attempt_path)
    expected_ranked_sha = str(attempt.get("ranked_candidates_sha256") or "")
    actual_ranked_sha = sha256_file(ranked_path)
    if expected_ranked_sha and expected_ranked_sha != actual_ranked_sha:
        raise RuntimeError("Ranked candidate artifact SHA-256 does not match the attempt report.")
    ranked = read_object(ranked_path)
    rows = [
        dict(row)
        for row in ranked.get("all_unique_candidates") or []
        if isinstance(row, Mapping)
    ]
    candidate = next(
        (row for row in rows if str(row.get("candidate_id") or "") == candidate_id),
        None,
    )
    if candidate is None:
        raise RuntimeError(f"Candidate is not present in the current ranked artifact: {candidate_id}")

    role_confusion = dict(candidate.get("role_confusion") or {})
    observability = dict(candidate.get("identity_observability") or {})
    purity = dict(candidate.get("identity_purity") or {})
    negative_gate = dict(candidate.get("negative_review_gate") or {})
    failures: list[str] = []
    if role_confusion.get("passed") is not True:
        failures.append("ROLE_CONFUSION_GATE")
    if observability.get("passed") is not True:
        failures.append("IDENTITY_OBSERVABILITY_GATE")
    if purity.get("passed") is not True:
        failures.append("IDENTITY_PURITY_GATE")
    if negative_gate.get("passed") is not True:
        failures.append("PROVENANCE_NEGATIVE_GATE")
    if float(candidate.get("retrieval_score") or 0.0) < 0.45:
        failures.append("PLAUSIBLE_RETRIEVAL_MINIMUM")
    if float(candidate.get("prototype_target_similarity") or 0.0) < 0.45:
        failures.append("PLAUSIBLE_PROTOTYPE_MINIMUM")
    if failures:
        raise RuntimeError(
            "Candidate cannot be promoted because safe review gates failed: "
            + ",".join(failures)
        )

    assignments_path = shot_root / "candidate_assignments.csv"
    assignment_rows = [
        row
        for row in _read_csv_rows(assignments_path)
        if str(row.get("candidate_id") or "") == candidate_id
    ]
    clean_rows = [
        row
        for row in assignment_rows
        if str(row.get("clean_for_reid") or "").strip().lower() in {"true", "1", "yes"}
    ]
    if not clean_rows:
        raise RuntimeError("Candidate has no clean assignment for assisted review.")
    selected_anchor = max(
        clean_rows,
        key=lambda row: (
            float(row.get("confidence") or 0.0),
            -int(row.get("frame_index") or 0),
        ),
    )
    anchor_frame = int(selected_anchor["frame_index"])
    bbox = [
        float(selected_anchor["x1"]),
        float(selected_anchor["y1"]),
        float(selected_anchor["x2"]),
        float(selected_anchor["y2"]),
    ]

    candidate_dir = shot_root / "candidates" / _safe_name(candidate_id)
    references_dir = candidate_dir / "references"
    references = sorted(references_dir.glob("reference_*.jpg"))
    if not references:
        raise RuntimeError(f"Candidate reference images are missing: {candidate_id}")
    gallery = candidate_dir / "reference_gallery.jpg"
    strip = shot_root / "candidate_strips" / f"{candidate_id}.jpg"
    shutil.copy2(strip if strip.is_file() else references[0], gallery)

    source_video = Path(str(dict(state.get("video") or {}).get("path") or "")).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    full_frame = candidate_dir / "full_frame_context.jpg"
    _write_full_frame(source_video, anchor_frame, full_frame)
    shot_row = next(
        (
            row
            for row in state.get("shots") or []
            if isinstance(row, Mapping) and str(row.get("shot_id") or "") == shot_id
        ),
        None,
    )
    if shot_row is None:
        raise RuntimeError(f"Reviewed shot is missing from pipeline state: {shot_id}")
    shot_clip = shot_root / "shot_clip.mp4"
    if not shot_clip.is_file():
        _write_video_range(
            source_video,
            shot_clip,
            start_frame=int(shot_row["start_frame"]),
            end_frame_inclusive=int(shot_row["end_frame_inclusive"]),
            metadata=dict(state.get("video") or {}),
        )

    manifest = {
        "schema_version": "kickclip.runtime_candidate_manifest.r1_3",
        "candidate_id": candidate_id,
        "candidate_media_id": candidate_id,
        "shot_id": shot_id,
        "tracklet_id": candidate_id,
        "parent_tracklet_id": str(candidate.get("parent_tracklet_id") or candidate_id),
        "quality": {
            "identity_pure": True,
            "identity_purity_gate_passed": True,
            "identity_observability_gate_passed": True,
            "combined_negative_gate_passed": True,
            "reviewability": "PHASE4B_USER_PROMOTED_CONFIRMATION_REQUIRED",
            "identity_confirmation": False,
            "manual_review_promotion": True,
            "automatic_target_confirmation": False,
        },
        "best_observation": {
            "frame_index": anchor_frame,
            "detection_id": str(selected_anchor.get("detection_id") or ""),
            "confidence": float(selected_anchor.get("confidence") or 0.0),
            "bbox_xyxy": bbox,
        },
        "reference_paths": [str(path) for path in references],
        "review_promotion": {
            "policy": PHASE4B_REVIEW_PROMOTION_POLICY,
            "reviewer": reviewer,
            "note": note,
            "source_ambiguity_id": ambiguity_id,
            "source_ranked_candidates_path": str(ranked_path),
            "source_ranked_candidates_sha256": actual_ranked_sha,
            "positive_memory_modified": False,
            "negative_memory_modified": False,
            "candidate_link_created": False,
            "automatic_target_confirmation": False,
        },
        "automatic_target_confirmation": False,
    }
    manifest_path = candidate_dir / "candidate_manifest.json"
    atomic_json(manifest_path, manifest)
    review_candidate = {
        **candidate,
        "candidate_id": candidate_id,
        "shot_id": shot_id,
        "review_rank": 1,
        "review_path": "USER_SELECTED_SAFE_CANDIDATE_PROMOTION",
        "manual_review_promoted": True,
        "review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "full_frame_context_path": str(full_frame),
        "full_frame_context_sha256": sha256_file(full_frame),
        "shot_clip_path": str(shot_clip),
        "shot_clip_sha256": sha256_file(shot_clip),
        "reference_gallery_path": str(gallery),
        "reference_gallery_sha256": sha256_file(gallery),
        "best_frame": anchor_frame,
        "best_bbox_xyxy": bbox,
        "score_evidence": {
            "phase4b_policy": PHASE4B_POLICY,
            "review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
            "retrieval_score": float(candidate.get("retrieval_score") or 0.0),
            "prototype_target_similarity": float(
                candidate.get("prototype_target_similarity") or 0.0
            ),
            "identity_observability": observability,
            "identity_purity": purity,
            "negative_review_gate": negative_gate,
            "corroboration_group_id": candidate.get("corroboration_group_id"),
            "corroboration_group_member_ids": list(
                candidate.get("corroboration_group_member_ids") or []
            ),
            "group_confidence_score": float(
                candidate.get("group_confidence_score") or 0.0
            ),
            "target_context_is_soft_evidence_only": True,
            "positive_memory_modified": False,
            "negative_memory_modified": False,
            "candidate_link_created": False,
            "automatic_target_confirmation": False,
        },
        "status": "PENDING",
        "automatic_target_confirmation": False,
    }

    promotion_dir = output_dir / "phase4b_review_promotions"
    promotion_dir.mkdir(parents=True, exist_ok=True)
    promotion_path = promotion_dir / (
        f"{_safe_name(ambiguity_id)}__{_safe_name(candidate_id)}.json"
    )
    promotion = {
        "schema_version": "kickclip.phase4b_review_candidate_promotion.v1",
        "created_at": now_iso(),
        "policy": PHASE4B_REVIEW_PROMOTION_POLICY,
        "ambiguity_id": ambiguity_id,
        "shot_id": shot_id,
        "candidate_id": candidate_id,
        "reviewer": reviewer,
        "note": note,
        "source_ranked_candidates_path": str(ranked_path),
        "source_ranked_candidates_sha256": actual_ranked_sha,
        "safe_gate_validation": {
            "role_confusion_passed": True,
            "identity_observability_passed": True,
            "identity_purity_passed": True,
            "negative_gate_passed": True,
            "plausible_retrieval_passed": True,
            "plausible_prototype_passed": True,
        },
        "positive_memory_modified": False,
        "negative_memory_modified": False,
        "candidate_link_created": False,
        "automatic_target_confirmation": False,
    }
    atomic_json(promotion_path, promotion)

    ambiguity["review_candidates"] = [review_candidate]
    ambiguity["manual_review_promoted_candidate_id"] = candidate_id
    ambiguity["review_promotion_policy"] = PHASE4B_REVIEW_PROMOTION_POLICY
    ambiguity["review_promotion_path"] = str(promotion_path)
    ambiguity["review_promotion_sha256"] = sha256_file(promotion_path)
    ambiguity["recommended_candidate"] = None
    ambiguity["automatic_target_confirmation"] = False
    pending.update(
        {
            "candidate_ids": [candidate_id],
            "manual_review_promoted_candidate_id": candidate_id,
            "review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
            "review_promotion_path": str(promotion_path),
            "review_promotion_sha256": sha256_file(promotion_path),
            "recommended_candidate": None,
            "automatic_target_confirmation": False,
        }
    )
    state["pending_action"] = pending
    runtime = dict(state.get("runtime") or {})
    runtime.update(
        {
            "phase4b_manual_review_promotion_allowed": True,
            "phase4b_manual_review_promoted_candidate_id": candidate_id,
            "phase4b_review_promotion_policy": PHASE4B_REVIEW_PROMOTION_POLICY,
            "phase4b_review_promotion_path": str(promotion_path),
            "phase4b_review_promotion_sha256": sha256_file(promotion_path),
            "automatic_target_confirmation": False,
        }
    )
    state["runtime"] = runtime
    state["updated_at"] = now_iso()
    atomic_json(state_path, state)
    return {
        **promotion,
        "status": "PASS",
        "decision": "PAUSE_FOR_USER_CONFIRMATION_OF_PROMOTED_CANDIDATE",
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "pipeline_state_path": str(state_path),
        "pipeline_state_sha256": sha256_file(state_path),
    }


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_path = args.tracking_launch_manifest.resolve()
    launch = read_object(launch_path)
    launch_wrapper = {"path": str(launch_path)}
    validate_input_file(launch_path, None, "tracking launch manifest")
    validate_input_file(args.target_selection, launch["target_selection"]["sha256"], "target selection")
    validate_input_file(args.target_reference_set, launch["target_reference_set"]["sha256"], "target reference set")
    validate_input_file(args.earlier_anchor_decision, launch["earlier_anchor_decision"]["sha256"], "earlier anchor decision")
    validate_input_file(args.shot_boundaries, launch["shot_boundaries"]["sha256"], "reviewed shot boundaries")
    target_selection = read_object(args.target_selection)
    anchor_decision = read_object(args.earlier_anchor_decision)
    boundaries = read_object(args.shot_boundaries)
    if anchor_decision.get("frame_zero_fallback_used") is not False:
        raise ValueError("FRAME_ZERO_FALLBACK_FORBIDDEN")
    if target_selection.get("automatic_target_confirmation") is not False:
        raise ValueError("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")

    memory_sha: str | None = None
    if args.target_memory_revision is not None:
        memory_sha = validate_input_file(
            args.target_memory_revision,
            args.target_memory_sha256,
            "target memory revision",
        )

    standalone_modes = sum(
        bool(value)
        for value in (
            args.materialize_selection_view_only,
            args.execute_bidirectional_phase1_only,
            args.merge_bidirectional_timelines_only,
            args.build_phase4a_initial_memory_only,
            args.run_phase4b_first_cross_shot_only,
        )
    )
    if standalone_modes > 1:
        raise ValueError("Choose only one standalone Phase 3/4 mode.")

    if args.materialize_selection_view_only:
        if args.resume:
            raise ValueError("Selection-view-only mode does not support --resume.")
        if args.video is None or args.initial_bbox is None:
            raise ValueError(
                "Selection-view-only mode requires --video and --initial-bbox "
                "to preserve the normal new-run command contract."
            )
        view = materialize_selection_view(
            output_dir=output_dir,
            launch=launch,
            target_selection=target_selection,
            boundaries=boundaries,
            overwrite=args.overwrite,
        )
        report = {
            "schema_version": "kickclip.phase3a_selection_view_report.v1",
            "status": "PASS",
            "decision": "AUTHORIZE_PHASE3B_REVERSE_PHASE1_EXECUTION",
            "selection_view_path": str(
                output_dir / "_selection_anchor_view" / "selection_view.json"
            ),
            "selection_view_sha256": sha256_file(
                output_dir / "_selection_anchor_view" / "selection_view.json"
            ),
            "selected_shot_id": view["selected_shot_id"],
            "selected_shot_start_frame": view["selected_shot_start_frame"],
            "anchor_frame": view["source_offset_frame"],
            "selected_shot_end_frame_inclusive": view[
                "selected_shot_end_frame_inclusive"
            ],
            "forward_frame_count": view["forward_frame_count"],
            "backward_frame_count": view["backward_frame_count"],
            "unresolved_prefix_frame_count": view[
                "unresolved_prefix_frame_count"
            ],
            "forward_source_sha256": view[
                "selected_shot_forward_source"
            ]["sha256"],
            "backward_source_sha256": view[
                "selected_shot_backward_source"
            ]["sha256"],
            "reverse_phase1_execution_complete": False,
            "timeline_merge_complete": False,
        }
        report_path = output_dir / "phase3a_selection_view_report.json"
        atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.merge_bidirectional_timelines_only:
        if args.resume:
            raise ValueError(
                "Bidirectional merge-only mode does not support --resume."
            )
        view_path = output_dir / "_selection_anchor_view" / "selection_view.json"
        if not view_path.is_file():
            raise FileNotFoundError(view_path)
        if args.initial_bbox is None:
            raise ValueError(
                "Bidirectional merge-only mode requires --initial-bbox."
            )
        view = read_object(view_path)
        report = merge_bidirectional_selected_shot_timelines(
            output_dir=output_dir,
            view=view,
            initial_bbox=args.initial_bbox,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.build_phase4a_initial_memory_only:
        if args.resume:
            raise ValueError(
                "Phase 4-A initial-memory-only mode does not support --resume."
            )
        view_path = output_dir / "_selection_anchor_view" / "selection_view.json"
        if not view_path.is_file():
            raise FileNotFoundError(view_path)
        view = read_object(view_path)
        phase4a = build_phase4a_initial_memory_from_phase3c(
            output_dir=output_dir,
            launch=launch,
            view=view,
        )
        # Artifact-only validation must not mutate a completed job's durable
        # pipeline_state.json or imply that the backend DB was synchronized.
        print(json.dumps(phase4a, ensure_ascii=False, indent=2))
        return 0

    phase4a_review_result = _handle_phase4a_memory_review_resume(
        args=args,
        output_dir=output_dir,
    )
    phase4b_from_memory_review = (
        phase4a_review_result == PHASE4B_APPROVED_CONTINUE
    )
    if phase4a_review_result is not None and not phase4b_from_memory_review:
        return int(phase4a_review_result)

    if args.resume and args.promote_review_candidate:
        if not args.ambiguity_id:
            raise ValueError("--promote-review-candidate requires --ambiguity-id.")
        promotion = _phase4b_promote_review_candidate(
            output_dir=output_dir,
            ambiguity_id=str(args.ambiguity_id),
            candidate_id=str(args.promote_review_candidate),
            reviewer=str(args.reviewer or "USER"),
            note=str(args.review_note or ""),
        )
        print(json.dumps(promotion, ensure_ascii=False, indent=2))
        return 0

    dependencies = dependency_report(root)
    if args.verify_only:
        print(json.dumps(dependencies, ensure_ascii=False, indent=2))
        return 0 if not dependencies["missing"] else 2
    if dependencies["missing"]:
        write_blocked_state(
            output_dir,
            decision="BLOCK_MISSING_FROZEN_RUNTIME_DEPENDENCY",
            failure_code="MISSING_FROZEN_RUNTIME_DEPENDENCY",
            message="The supplied research source is present, but required helper/model files are missing.",
            launch=launch_wrapper,
            dependencies=dependencies,
        )
        return 2

    if args.resume and args.reject_all_candidates_as_non_player_role:
        if not args.ambiguity_id:
            raise ValueError(
                "--reject-all-candidates-as-non-player-role requires --ambiguity-id."
            )
        state_path = output_dir / "pipeline_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        durable_state = read_object(state_path)
        role_memory = _phase4b_apply_non_player_role_review(
            output_dir=output_dir,
            state=durable_state,
            ambiguity_id=str(args.ambiguity_id),
            reviewer=str(args.reviewer or "USER"),
            note=str(args.review_note or ""),
        )
        atomic_json(state_path, durable_state)
        summary_path = output_dir / "pipeline_summary.json"
        summary = read_object(summary_path) if summary_path.is_file() else {}
        summary.update(
            {
                "status": durable_state["status"],
                "decision": durable_state["decision"],
                "phase4b_non_player_role_applied": True,
                "phase4b_non_player_role_ambiguity_id": str(args.ambiguity_id),
                "phase4b_persistent_role_negative_memory_revision_id": role_memory[
                    "revision_id"
                ],
                "phase4b_persistent_role_negative_embedding_count": int(
                    role_memory["embedding_count"]
                ),
                "automatic_target_confirmation": False,
            }
        )
        atomic_json(summary_path, summary)
        durable_runtime = dict(durable_state.get("runtime") or {})
        effective_memory_path = (
            args.target_memory_revision.resolve()
            if args.target_memory_revision is not None
            else Path(str(durable_runtime.get("memory_revision_path") or "")).resolve()
        )
        effective_memory_sha = (
            memory_sha or str(durable_runtime.get("memory_revision_sha256") or "")
        )
        report = run_phase4b_first_cross_shot(
            root=root,
            output_dir=output_dir,
            args=args,
            memory_path=effective_memory_path,
            memory_sha=effective_memory_sha,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.resume and args.reject_all_candidates:
        if not args.ambiguity_id:
            raise ValueError("--reject-all-candidates requires --ambiguity-id.")
        state_path = output_dir / "pipeline_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        durable_state = read_object(state_path)
        identity_memory = _phase4b_apply_none_of_these_review(
            output_dir=output_dir,
            state=durable_state,
            ambiguity_id=str(args.ambiguity_id),
            reviewer=str(args.reviewer or "USER"),
            note=str(args.review_note or ""),
        )
        atomic_json(state_path, durable_state)
        summary_path = output_dir / "pipeline_summary.json"
        summary = read_object(summary_path) if summary_path.is_file() else {}
        summary.update(
            {
                "status": durable_state["status"],
                "decision": durable_state["decision"],
                "phase4b_none_of_these_applied": True,
                "phase4b_none_of_these_ambiguity_id": str(args.ambiguity_id),
                "phase4b_identity_negative_memory_revision_id": identity_memory[
                    "revision_id"
                ],
                "phase4b_identity_negative_embedding_count": int(
                    identity_memory["embedding_count"]
                ),
                "automatic_target_confirmation": False,
            }
        )
        atomic_json(summary_path, summary)
        if durable_state.get("status") == "COMPLETE_WITH_UNRESOLVED_GAPS":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        durable_runtime = dict(durable_state.get("runtime") or {})
        effective_memory_path = (
            args.target_memory_revision.resolve()
            if args.target_memory_revision is not None
            else Path(str(durable_runtime.get("memory_revision_path") or "")).resolve()
        )
        effective_memory_sha = (
            memory_sha
            or str(durable_runtime.get("memory_revision_sha256") or "")
        )
        report = run_phase4b_first_cross_shot(
            root=root,
            output_dir=output_dir,
            args=args,
            memory_path=effective_memory_path,
            memory_sha=effective_memory_sha,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.run_phase4b_first_cross_shot_only or phase4b_from_memory_review:
        state_path = output_dir / "pipeline_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        durable_state = read_object(state_path)
        durable_runtime = dict(durable_state.get("runtime") or {})
        effective_memory_path = (
            args.target_memory_revision.resolve()
            if args.target_memory_revision is not None
            else Path(str(durable_runtime.get("memory_revision_path") or "")).resolve()
        )
        effective_memory_sha = (
            memory_sha
            or str(durable_runtime.get("memory_revision_sha256") or "")
        )
        report = run_phase4b_first_cross_shot(
            root=root,
            output_dir=output_dir,
            args=args,
            memory_path=effective_memory_path,
            memory_sha=effective_memory_sha,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # Phase 4-C commits only the real RF-DETR observations from the
    # user-confirmed identity-pure candidate. It does not invoke the supplied E2E
    # runner, interpolate missing frames, or auto-confirm any identity.
    if args.resume and args.confirmed_candidate and args.target_memory_revision is not None:
        if not args.ambiguity_id:
            raise ValueError(
                "Confirmed-candidate Phase 4-C resume requires --ambiguity-id."
            )
        if not memory_sha:
            raise ValueError(
                "Confirmed-candidate Phase 4-C resume requires a verified memory SHA."
            )
        from app.domains.candidate_handoff_r1.runtime.post_confirmation_finalizer import (
            finalize_confirmed_candidate,
        )

        report = finalize_confirmed_candidate(
            output_dir=output_dir,
            candidate_id=str(args.confirmed_candidate),
            ambiguity_id=str(args.ambiguity_id),
            memory_path=args.target_memory_revision.resolve(),
            memory_sha256=str(memory_sha),
            no_preview=bool(args.no_preview),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.resume and (args.rejected_candidate or args.unreviewable_candidate):
        if args.review_decision_artifact is None or not args.review_decision_sha256:
            raise ValueError(
                "R14 rejection resume requires an immutable review decision artifact."
            )
        state_path = output_dir / "pipeline_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        durable_state = read_object(state_path)
        decision_state = (
            "DIFFERENT_PLAYER"
            if args.rejected_candidate
            else "UNREVIEWABLE_LOW_RESOLUTION"
        )
        candidate_id = str(args.rejected_candidate or args.unreviewable_candidate)
        identity_memory = _phase4b_resume_rejected_candidate(
            output_dir=output_dir,
            state=durable_state,
            candidate_id=candidate_id,
            decision_state=decision_state,
            decision_path=args.review_decision_artifact.resolve(),
            decision_sha256=str(args.review_decision_sha256),
            reviewer=str(args.reviewer or "USER"),
            note=str(args.review_note or ""),
        )
        _phase4b_restore_authorization_from_report(
            output_dir=output_dir,
            state=durable_state,
        )
        atomic_json(state_path, durable_state)
        summary_path = output_dir / "pipeline_summary.json"
        summary = read_object(summary_path) if summary_path.is_file() else {}
        summary.update(
            {
                "status": durable_state["status"],
                "decision": durable_state["decision"],
                "phase4b_r14_rejection_resume_applied": True,
                "phase4b_r14_review_state": decision_state,
                "phase4b_r14_candidate_id": candidate_id,
                "phase4b_identity_negative_memory_revision_id": (
                    identity_memory.get("revision_id") if identity_memory else None
                ),
                "automatic_target_confirmation": False,
            }
        )
        atomic_json(summary_path, summary)
        if durable_state.get("status") == "COMPLETE_WITH_UNRESOLVED_GAPS":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        durable_runtime = dict(durable_state.get("runtime") or {})
        effective_memory_path = (
            args.target_memory_revision.resolve()
            if args.target_memory_revision is not None
            else Path(str(durable_runtime.get("memory_revision_path") or "")).resolve()
        )
        effective_memory_sha = (
            memory_sha or str(durable_runtime.get("memory_revision_sha256") or "")
        )
        report = run_phase4b_first_cross_shot(
            root=root,
            output_dir=output_dir,
            args=args,
            memory_path=effective_memory_path,
            memory_sha=effective_memory_sha,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.resume:
        state_path = output_dir / "pipeline_state.json"
        if state_path.is_file():
            durable_state = read_object(state_path)
            durable_runtime = dict(durable_state.get("runtime") or {})
            if (
                str(durable_state.get("status") or "").upper() == "RUNNING"
                and durable_state.get("pending_action") is None
                and durable_state.get("decision") == PHASE4B_CONTINUE_DECISION
                and durable_runtime.get("phase4b_next_shot_id")
            ):
                effective_memory_path = (
                    args.target_memory_revision.resolve()
                    if args.target_memory_revision is not None
                    else Path(
                        str(durable_runtime.get("memory_revision_path") or "")
                    ).resolve()
                )
                effective_memory_sha = (
                    memory_sha
                    or str(durable_runtime.get("memory_revision_sha256") or "")
                )
                report = run_phase4b_first_cross_shot(
                    root=root,
                    output_dir=output_dir,
                    args=args,
                    memory_path=effective_memory_path,
                    memory_sha=effective_memory_sha,
                )
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0

    view_path = output_dir / "_selection_anchor_view" / "selection_view.json"
    if args.resume:
        if not view_path.is_file():
            raise FileNotFoundError(f"Selection anchor view is missing for resume: {view_path}")
        view = read_object(view_path)
    else:
        if args.video is None or args.initial_bbox is None:
            raise ValueError("New run requires video and initial bbox.")
        view = materialize_selection_view(
            output_dir=output_dir,
            launch=launch,
            target_selection=target_selection,
            boundaries=boundaries,
            overwrite=args.overwrite,
        )

    e2e = Path(dependencies["files"]["e2e_runner"])
    raw_root = output_dir / "_provided_e2e_output"
    phase3b_report: dict[str, Any] | None = None
    if not args.resume and not args.contract_test_skip_phase1_compatibility:
        compatibility_module = _load_module(
            "kickclip_phase1_compatibility_executor",
            Path(__file__).with_name("phase1_compatibility_executor.py"),
        )
        phase3b_report = execute_bidirectional_selected_shot_phase1(
            project_root=root,
            output_dir=output_dir,
            test_name=args.test_name,
            view=view,
            initial_bbox=args.initial_bbox,
            device=args.device,
            overwrite=args.overwrite,
            compatibility_module=compatibility_module,
        )
        view = read_object(view_path)

    if args.execute_bidirectional_phase1_only:
        if args.resume:
            raise ValueError(
                "Bidirectional Phase-1-only mode does not support --resume."
            )
        if args.contract_test_skip_phase1_compatibility:
            raise ValueError(
                "Bidirectional Phase-1-only mode cannot skip Phase-1 compatibility."
            )
        if phase3b_report is None:
            raise RuntimeError(
                "Phase 3-B report was not generated."
            )
        print(json.dumps(phase3b_report, ensure_ascii=False, indent=2))
        return 0
    if (
        not args.resume
        and not args.contract_test_skip_phase1_compatibility
    ):
        if phase3b_report is None:
            raise RuntimeError("Phase 3-B report was not generated.")
        if args.initial_bbox is None:
            raise ValueError("New run requires initial bbox for Phase 3-C.")
        merge_bidirectional_selected_shot_timelines(
            output_dir=output_dir,
            view=view,
            initial_bbox=args.initial_bbox,
        )
        view = read_object(view_path)
    command = [
        sys.executable,
        str(e2e),
        "--project-root",
        str(root),
        "--test-name",
        args.test_name,
        "--device",
        "cpu" if args.device == "mps" else args.device,
        "--reacquisition-mode",
        args.reacquisition_mode,
        "--output-root",
        str(raw_root),
        "--backend-safe-weights-runner",
        str(Path(__file__).with_name("safe_weights_subprocess.py").resolve()),
    ]
    if args.resume:
        command.append("--resume")
        if args.target_memory_revision is not None and memory_sha:
            command.extend(
                [
                    "--backend-memory-revision",
                    str(args.target_memory_revision.resolve()),
                    "--backend-memory-sha256",
                    memory_sha,
                ]
            )
        if args.ambiguity_id:
            command.extend(["--ambiguity-id", args.ambiguity_id])
        if args.approve_review:
            command.extend(["--approve-review", args.approve_review])
        elif args.reject_review:
            command.extend(["--reject-review", args.reject_review])
        elif args.confirm_absent:
            command.append("--confirm-absent")
        elif args.confirmed_candidate:
            command.extend(["--confirmed-candidate", args.confirmed_candidate])
    else:
        suffix_video = Path(str(view["selection_view_video"]["path"])).resolve()
        command.extend(["--video", str(suffix_video), "--initial-bbox", *map(str, args.initial_bbox)])
        cut_frames = [int(value) for value in view.get("translated_cut_frames") or []]
        if cut_frames:
            command.extend(["--cut-frames", *map(str, cut_frames)])
        if args.overwrite:
            command.append("--overwrite")
    if args.reviewer:
        command.extend(["--reviewer", args.reviewer])
    if args.review_note:
        command.extend(["--review-note", args.review_note])
    if args.no_preview:
        command.append("--no-preview")

    rc = invoke(command, root)
    raw_dir = raw_root / args.test_name
    if raw_dir.is_dir():
        normalize_state(
            root=root,
            output_dir=output_dir,
            raw_dir=raw_dir,
            launch=launch,
            view=view,
            memory_path=args.target_memory_revision.resolve() if args.target_memory_revision else None,
            memory_sha=memory_sha,
            generation=args.candidate_scoring_generation,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
