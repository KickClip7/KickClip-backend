from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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


def validate_scene_target_selection_contract(
    *,
    package_root: Path,
    artifact_path: Path,
    video_sha256: str,
    frame_count: int,
) -> None:
    """Execute the installed frozen runtime's validator in an isolated process."""

    package = package_root.resolve()
    artifact = artifact_path.resolve()
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
