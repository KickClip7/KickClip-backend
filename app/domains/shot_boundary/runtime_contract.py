from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SceneTargetShotBoundaryContractError(ValueError):
    reason: str
    runtime_message: str

    def __str__(self) -> str:
        return self.runtime_message


def _reason(message: str) -> str:
    normalized = message.lower()
    if "indexes are not contiguous" in normalized:
        return "SHOT_INDEXES_NOT_CONTIGUOUS"
    if "frame coverage is not contiguous" in normalized:
        return "SHOT_FRAME_COVERAGE_NOT_CONTIGUOUS"
    if "cover every scene frame" in normalized:
        return "SHOT_FRAME_COVERAGE_INCOMPLETE"
    if "every shot boundary must be reviewed" in normalized:
        return "SHOT_REVIEW_STATE_INVALID"
    if "video sha-256" in normalized:
        return "SCENE_VIDEO_SHA256_MISMATCH"
    if "frame count" in normalized:
        return "SCENE_FRAME_COUNT_MISMATCH"
    if "waiting_shot_boundary_review" in normalized:
        return "WAITING_SHOT_BOUNDARY_REVIEW"
    if "no shots" in normalized:
        return "SHOTS_EMPTY"
    return "FROZEN_RUNTIME_CONTRACT_REJECTED"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneTargetShotBoundaryContractError(
            "SHOT_BOUNDARY_JSON_INVALID",
            "Shot-boundary artifact is not readable JSON.",
        ) from exc
    if not isinstance(value, dict):
        raise SceneTargetShotBoundaryContractError(
            "SHOT_BOUNDARY_JSON_INVALID",
            "Shot-boundary artifact root must be a JSON object.",
        )
    return value


def _validate_auto_detected_contract(
    document: Mapping[str, Any],
    *,
    video_sha256: str,
    frame_count: int,
) -> None:
    """Validate automatic cut boundaries without inventing human approval.

    Automatic camera cuts are allowed as authoritative *segmentation* input.
    They never confirm target identity.  The contract therefore verifies only
    immutable video provenance and exact contiguous shot coverage.
    """

    if str(document.get("boundary_origin") or "") != "AUTO_DETECTED":
        raise SceneTargetShotBoundaryContractError(
            "AUTO_BOUNDARY_ORIGIN_INVALID",
            "Automatic shot-boundary artifact has an invalid boundary origin.",
        )
    if document.get("human_reviewed") is not False:
        raise SceneTargetShotBoundaryContractError(
            "AUTO_BOUNDARY_HUMAN_REVIEW_FLAG_INVALID",
            "Automatic shot boundaries must explicitly declare human_reviewed=false.",
        )
    if document.get("automatic_target_confirmation") is not False:
        raise SceneTargetShotBoundaryContractError(
            "AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN",
            "Automatic cut detection must not confirm target identity.",
        )

    video = document.get("video")
    if not isinstance(video, Mapping):
        raise SceneTargetShotBoundaryContractError(
            "AUTO_BOUNDARY_VIDEO_CONTRACT_INVALID",
            "Automatic shot boundaries have no valid video contract.",
        )
    if str(video.get("sha256") or "") != video_sha256:
        raise SceneTargetShotBoundaryContractError(
            "SCENE_VIDEO_SHA256_MISMATCH",
            "Automatic shot-boundary video SHA-256 does not match the scene video.",
        )
    try:
        declared_frame_count = int(video.get("frame_count"))
    except (TypeError, ValueError) as exc:
        raise SceneTargetShotBoundaryContractError(
            "SCENE_FRAME_COUNT_MISMATCH",
            "Automatic shot-boundary frame count is invalid.",
        ) from exc
    if declared_frame_count != int(frame_count) or frame_count <= 0:
        raise SceneTargetShotBoundaryContractError(
            "SCENE_FRAME_COUNT_MISMATCH",
            "Automatic shot-boundary frame count does not match the scene video.",
        )

    structural = document.get("structural_validation")
    if not isinstance(structural, Mapping) or structural.get("status") != "PASS":
        raise SceneTargetShotBoundaryContractError(
            "AUTO_BOUNDARY_STRUCTURAL_GATE_NOT_PASSED",
            "Automatic shot boundaries did not pass the structural safety gate.",
        )

    rows = document.get("shots")
    if not isinstance(rows, list) or not rows:
        raise SceneTargetShotBoundaryContractError(
            "SHOTS_EMPTY",
            "Automatic shot-boundary artifact has no shots.",
        )

    expected_start = 0
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise SceneTargetShotBoundaryContractError(
                "SHOT_ROW_INVALID",
                "Automatic shot boundary contains a non-object row.",
            )
        try:
            shot_index = int(raw.get("shot_index", -1))
            start_frame = int(raw.get("start_frame", -1))
            end_frame = int(
                raw.get("end_frame_inclusive", raw.get("end_frame", -1))
            )
        except (TypeError, ValueError) as exc:
            raise SceneTargetShotBoundaryContractError(
                "SHOT_FRAME_VALUES_INVALID",
                "Automatic shot frame values must be integers.",
            ) from exc

        if shot_index != index:
            raise SceneTargetShotBoundaryContractError(
                "SHOT_INDEXES_NOT_CONTIGUOUS",
                "Automatic shot indexes are not contiguous.",
            )
        if start_frame != expected_start or end_frame < start_frame:
            raise SceneTargetShotBoundaryContractError(
                "SHOT_FRAME_COVERAGE_NOT_CONTIGUOUS",
                "Automatic shot frame coverage is not contiguous.",
            )
        if int(raw.get("frame_count", end_frame - start_frame + 1)) != (
            end_frame - start_frame + 1
        ):
            raise SceneTargetShotBoundaryContractError(
                "SHOT_FRAME_COUNT_INVALID",
                "Automatic shot frame_count does not match its frame range.",
            )

        expected_cut_in = None if index == 0 else start_frame
        expected_cut_out = None if end_frame == frame_count - 1 else end_frame + 1
        if raw.get("cut_in_frame") != expected_cut_in:
            raise SceneTargetShotBoundaryContractError(
                "SHOT_CUT_IN_INVALID",
                "Automatic shot cut_in_frame is inconsistent with its frame range.",
            )
        if raw.get("cut_out_frame") != expected_cut_out:
            raise SceneTargetShotBoundaryContractError(
                "SHOT_CUT_OUT_INVALID",
                "Automatic shot cut_out_frame is inconsistent with its frame range.",
            )
        expected_start = end_frame + 1

    if expected_start != frame_count:
        raise SceneTargetShotBoundaryContractError(
            "SHOT_FRAME_COVERAGE_INCOMPLETE",
            "Automatic shot boundaries do not cover every scene frame exactly once.",
        )


def validate_scene_target_selection_contract(
    *,
    package_root: Path,
    artifact_path: Path,
    video_sha256: str,
    frame_count: int,
) -> None:
    """Validate automatic or human-reviewed shot boundaries.

    Product policy:
    - AUTO_SHOT_BOUNDARIES are authoritative for camera-cut segmentation once
      the backend structural gate passes. No user approval is required.
    - Human-reviewed artifacts retain the original frozen validator path.
    - Neither path performs or implies automatic target-identity confirmation.
    """

    artifact = artifact_path.resolve()
    document = _load_object(artifact)
    artifact_type = str(document.get("artifact_type") or "")
    boundary_origin = str(document.get("boundary_origin") or "")
    if artifact_type == "AUTO_SHOT_BOUNDARIES" or boundary_origin == "AUTO_DETECTED":
        _validate_auto_detected_contract(
            document,
            video_sha256=video_sha256,
            frame_count=frame_count,
        )
        return

    package = package_root.resolve()
    contracts = package / "contracts.py"
    if not contracts.is_file():
        raise SceneTargetShotBoundaryContractError(
            "FROZEN_RUNTIME_CONTRACT_UNAVAILABLE",
            "Scene Target Selection frozen runtime contract is unavailable.",
        )
    verifier = """
import json
import sys
from pathlib import Path

package_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(package_root))
from contracts import validate_reviewed_shot_boundaries

validate_reviewed_shot_boundaries(
    path=Path(sys.argv[2]).resolve(),
    video_sha256=sys.argv[3],
    frame_count=int(sys.argv[4]),
)
print(json.dumps({"valid": True}))
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            verifier,
            str(package),
            str(artifact),
            video_sha256,
            str(frame_count),
        ],
        cwd=str(package.parent),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if process.returncode == 0:
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise SceneTargetShotBoundaryContractError(
                "FROZEN_RUNTIME_CONTRACT_OUTPUT_INVALID",
                "Scene Target Selection contract verifier returned invalid output.",
            ) from exc
        if result == {"valid": True}:
            return
    message = (process.stderr or process.stdout or "Frozen runtime rejected artifact.")[
        -3000:
    ].strip()
    raise SceneTargetShotBoundaryContractError(_reason(message), message)
