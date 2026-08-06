from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_POLICY = "USER_CONFIRMED_CANDIDATE_TIMELINE_COMMIT_R1"
EXPECTED_DECISION = "TRACKING_COMPLETED_WITH_USER_CONFIRMED_CROSS_SHOT_LINK"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rows(value: object) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ] if isinstance(value, list) else []


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-id",
        default="shot_0010_track_0047",
    )
    parser.add_argument(
        "--decision-id",
        default="ecdecr1_b6fb20b6446b",
    )
    parser.add_argument(
        "--memory-revision-id",
        default="ecmem_5339a6e1eecf",
    )
    parser.add_argument(
        "--expected-observation-count",
        type=int,
        default=29,
    )
    parser.add_argument(
        "--require-previews",
        action="store_true",
    )
    args = parser.parse_args()

    root = args.job_root.resolve()
    report_path = (
        root
        / "phase4c_post_confirmation"
        / "phase4c_post_confirmation_report.json"
    )
    link_path = (
        root / "phase4c_post_confirmation" / "candidate_link.json"
    )
    state_path = root / "pipeline_state.json"
    timeline_path = root / "target_timeline.json"
    summary_path = root / "pipeline_summary.json"
    errors: list[str] = []

    required = [
        report_path,
        link_path,
        state_path,
        timeline_path,
        summary_path,
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"MISSING:{path}")

    if errors:
        for error in errors:
            print(f"error={error}")
        return 2

    report = _read(report_path)
    link = _read(link_path)
    state = _read(state_path)
    timeline = _read(timeline_path)
    summary = _read(summary_path)

    if report.get("status") != "PASS":
        errors.append("REPORT_STATUS_NOT_PASS")
    if report.get("policy") != EXPECTED_POLICY:
        errors.append("REPORT_POLICY_MISMATCH")
    if report.get("decision") != EXPECTED_DECISION:
        errors.append("REPORT_DECISION_MISMATCH")
    if report.get("candidate_id") != args.candidate_id:
        errors.append("REPORT_CANDIDATE_MISMATCH")
    if report.get("decision_id") != args.decision_id:
        errors.append("REPORT_DECISION_ID_MISMATCH")
    if report.get("memory_revision_id") != args.memory_revision_id:
        errors.append("REPORT_MEMORY_ID_MISMATCH")
    if int(report.get("real_detector_observation_count") or 0) != (
        args.expected_observation_count
    ):
        errors.append("REPORT_OBSERVATION_COUNT_MISMATCH")
    if report.get("interpolation_used") is not False:
        errors.append("INTERPOLATION_MUST_BE_FALSE")
    if report.get("synthetic_tracking_used") is not False:
        errors.append("SYNTHETIC_TRACKING_MUST_BE_FALSE")
    if report.get("automatic_target_confirmation") is not False:
        errors.append("AUTOMATIC_CONFIRMATION_MUST_BE_FALSE")
    if report.get("candidate_link_created") is not True:
        errors.append("CANDIDATE_LINK_NOT_CREATED")

    if link.get("candidate_id") != args.candidate_id:
        errors.append("LINK_CANDIDATE_MISMATCH")
    if int(link.get("observation_count") or 0) != (
        args.expected_observation_count
    ):
        errors.append("LINK_OBSERVATION_COUNT_MISMATCH")
    expected_frames = [
        int(value) for value in link.get("observation_frame_ids") or []
    ]
    if len(expected_frames) != args.expected_observation_count:
        errors.append("LINK_FRAME_COUNT_MISMATCH")
    if link.get("real_detector_observations_only") is not True:
        errors.append("LINK_NOT_REAL_DETECTIONS_ONLY")
    if link.get("automatic_target_confirmation") is not False:
        errors.append("LINK_AUTOMATIC_CONFIRMATION_MISMATCH")

    if state.get("status") != "COMPLETE_WITH_UNRESOLVED_GAPS":
        errors.append("STATE_NOT_TERMINAL_WITH_GAPS")
    if state.get("decision") != EXPECTED_DECISION:
        errors.append("STATE_DECISION_MISMATCH")
    if state.get("pending_action") is not None:
        errors.append("STATE_PENDING_ACTION_NOT_CLEARED")
    runtime = dict(state.get("runtime") or {})
    if runtime.get("phase4c_post_confirmation_complete") is not True:
        errors.append("PHASE4C_RUNTIME_FLAG_MISSING")
    if runtime.get("memory_revision_id") != args.memory_revision_id:
        errors.append("RUNTIME_MEMORY_ID_MISMATCH")
    if runtime.get("candidate_link_created") is not True:
        errors.append("RUNTIME_LINK_FLAG_MISSING")
    if runtime.get("automatic_target_confirmation") is not False:
        errors.append("RUNTIME_AUTOMATIC_CONFIRMATION_MISMATCH")

    frames = _rows(timeline.get("frames"))
    candidate_frames = [
        row
        for row in frames
        if row.get("candidate_id") == args.candidate_id
    ]
    actual_frame_ids = [
        int(row.get("frame_index", -1))
        for row in candidate_frames
    ]
    if actual_frame_ids != expected_frames:
        errors.append("TIMELINE_CANDIDATE_FRAMES_MISMATCH")
    for row in candidate_frames:
        if row.get("state") not in {"ACTIVE", "REACQUIRED"}:
            errors.append("TIMELINE_CONFIRMED_FRAME_NOT_ACTIVE")
            break
        bbox = row.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append("TIMELINE_CONFIRMED_FRAME_MISSING_BBOX")
            break
        if row.get("identity_source") != "USER_CONFIRMED_SAME_PLAYER":
            errors.append("TIMELINE_IDENTITY_SOURCE_MISMATCH")
            break
        if row.get("synthetic_tracking_used") is not False:
            errors.append("TIMELINE_SYNTHETIC_FLAG_MISMATCH")
            break
        if row.get("automatic_target_confirmation") is not False:
            errors.append("TIMELINE_AUTOMATIC_CONFIRMATION_MISMATCH")
            break

    confirmations = _rows(timeline.get("confirmations"))
    if not any(
        row.get("decision_id") == args.decision_id
        and row.get("candidate_id") == args.candidate_id
        and row.get("decision") == "SAME_PLAYER"
        for row in confirmations
    ):
        errors.append("TIMELINE_CONFIRMATION_MISSING")

    if _sha(timeline_path) != report.get("timeline_sha256"):
        errors.append("TIMELINE_SHA_MISMATCH")
    if _sha(link_path) != report.get("candidate_link_sha256"):
        errors.append("LINK_SHA_MISMATCH")
    if summary.get("phase4c_post_confirmation_complete") is not True:
        errors.append("SUMMARY_PHASE4C_FLAG_MISSING")

    full_preview = root / "full_frame_tracking_preview.mp4"
    centered_preview = root / "target_centered_preview.mp4"
    previews_exist = (
        full_preview.is_file()
        and full_preview.stat().st_size > 0
        and centered_preview.is_file()
        and centered_preview.stat().st_size > 0
    )
    if args.require_previews and not previews_exist:
        errors.append("PREVIEWS_MISSING")

    print(f"status={'PASS' if not errors else 'FAIL'}")
    print(f"policy={report.get('policy')}")
    print(f"decision={report.get('decision')}")
    print(f"candidate_id={report.get('candidate_id')}")
    print(f"decision_id={report.get('decision_id')}")
    print(f"memory_revision_id={report.get('memory_revision_id')}")
    print(
        "real_detector_observation_count="
        f"{report.get('real_detector_observation_count')}"
    )
    print(
        "active_or_reacquired_bbox_frames="
        f"{report.get('active_or_reacquired_bbox_frames')}"
    )
    print(f"source_frame_ids={expected_frames}")
    print(f"previews_exist={previews_exist}")
    print("interpolation_used=False")
    print("synthetic_tracking_used=False")
    print("automatic_target_confirmation=False")
    print(f"errors={errors}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
