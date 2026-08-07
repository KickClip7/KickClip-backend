from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.domains.tracking.artifacts import sha256_file
from app.domains.tracking.timeline import (
    TrackingTimelineService,
    _canonical_reviewed_shots,
    _video_frame_count,
)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed tracking job's public shot/frame contract "
            "against its immutable reviewed-shot launch input."
        )
    )
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()

    job_root = args.job_root.expanduser().resolve()
    timeline_path = job_root / "target_timeline.json"
    launch_path = job_root / "r3_inputs" / "tracking_launch_manifest.json"
    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    if not launch_path.is_file():
        raise FileNotFoundError(launch_path)

    timeline = _read_object(timeline_path)
    launch = _read_object(launch_path)
    record = launch.get("shot_boundaries")
    if not isinstance(record, dict):
        raise ValueError("Launch manifest has no shot_boundaries contract.")
    boundaries_path = Path(str(record.get("path") or "")).expanduser().resolve()
    expected_sha = str(record.get("sha256") or "").lower()
    if not boundaries_path.is_file():
        raise FileNotFoundError(boundaries_path)
    actual_sha = sha256_file(boundaries_path)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ValueError("Reviewed shot boundary SHA-256 mismatch.")

    reviewed = _canonical_reviewed_shots(
        _read_object(boundaries_path),
        frame_count=_video_frame_count(timeline),
    )
    normalized = TrackingTimelineService._normalize_compatible_payload(timeline)
    normalized = TrackingTimelineService._normalize_reviewed_shot_contract(
        normalized,
        reviewed_shots=reviewed,
        reviewed_source={
            "source": "SERVER_OWNED_R1_LAUNCH_MANIFEST",
            "sha256": actual_sha,
            "path": str(boundaries_path),
        },
    )
    TrackingTimelineService._validate(normalized)

    provenance = normalized["provenance"]["reviewed_shot_contract_normalization"]
    print("status=PASS")
    print("decision=AUTHORIZE_PUBLIC_TIMELINE_SHOT_CONTRACT")
    print(f"frame_count={len(normalized['frames'])}")
    print(f"reviewed_shot_count={len(normalized['shots'])}")
    print(
        "corrected_frame_shot_id_count="
        f"{provenance['corrected_frame_shot_id_count']}"
    )
    print(
        "runtime_shot_translation_count="
        f"{provenance['runtime_shot_translation_count']}"
    )
    print(f"reviewed_shot_boundaries_sha256={actual_sha}")
    print("metadata_only=True")
    print("tracking_state_modified=False")
    print("bbox_modified=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
