#!/usr/bin/env python
"""Test-only process fixture that mirrors the E2E file/exit contract."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--video")
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--initial-bbox", nargs=4)
    parser.add_argument("--device")
    parser.add_argument("--reacquisition-mode")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-note")
    reviews = parser.add_mutually_exclusive_group()
    reviews.add_argument("--approve-review")
    reviews.add_argument("--reject-review")
    decisions = parser.add_mutually_exclusive_group()
    decisions.add_argument("--confirmed-candidate")
    decisions.add_argument("--confirm-absent", action="store_true")
    parser.add_argument("--ambiguity-id")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def state_payload(
    args: argparse.Namespace,
    *,
    status: str,
    decision: str,
    pending: dict | None,
) -> dict:
    ambiguity = {
        "ambiguity_id": "ambiguity_0001",
        "shot_id": "shot_0001",
        "status": "PENDING",
        "review_candidates": [
            {"candidate_id": "shot_0001_track_0001"},
            {"candidate_id": "shot_0001_track_0002"},
        ],
        "contact_sheet": str(
            args.output_root
            / args.test_name
            / "ambiguity_candidates"
            / "ambiguity_0001"
            / "candidates.jpg"
        ),
    }
    return {
        "schema_version": "kickclip.target_centric_e2e_state.v1",
        "pipeline_version": "1.0.0",
        "test_name": args.test_name,
        "status": status,
        "decision": decision,
        "updated_at": "2026-07-27T00:00:00+00:00",
        "video": {
            "path": str(args.video or args.project_root / "video.mp4"),
            "sha256": "a" * 64,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 2,
        },
        "initial_bbox_xyxy": [10, 10, 30, 50],
        "phase1_manifest": {
            "path": str(args.project_root / "phase1_frozen_manifest.json"),
            "sha256": "b" * 64,
        },
        "reacquisition_mode": "assisted",
        "shots": [
            {
                "shot_index": 0,
                "shot_id": "shot_0000",
                "start_frame": 0,
                "end_frame_inclusive": 0,
                "frame_count": 1,
                "status": "INITIAL_TARGET_SHOT",
            },
            {
                "shot_index": 1,
                "shot_id": "shot_0001",
                "start_frame": 1,
                "end_frame_inclusive": 1,
                "frame_count": 1,
                "status": "UNPROCESSED",
            },
        ],
        "memory": {"source_timeline": "fixture"},
        "segments": [],
        "ambiguities": [ambiguity] if pending and pending["type"] == "CROSS_SHOT_CONFIRMATION" else [],
        "confirmations": [],
        "pending_action": pending,
    }


def write_contract(output: Path, state: dict) -> None:
    write_json(output / "pipeline_state.json", state)
    write_json(
        output / "pipeline_manifest.json",
        {
            "schema_version": "kickclip.target_centric_e2e_manifest.v1",
            "pipeline_version": "1.0.0",
            "frozen_files": [],
            "policies": {"stage3b2": {"minimum_retrieval_score": 0.65}},
        },
    )
    timeline = {
        "schema_version": "kickclip.target_centric_e2e.v1",
        "pipeline_version": "1.0.0",
        "test_name": state["test_name"],
        "target_id": "target_001",
        "status": state["status"],
        "video": state["video"],
        "shots": state["shots"],
        "frames": [
            {
                "frame_index": 0,
                "time_seconds": 0,
                "shot_id": "shot_0000",
                "state": "ACTIVE",
                "bbox_xyxy": [10, 10, 30, 50],
                "tracking_confidence": 0.9,
                "identity_confidence": 1.0,
                "identity_source": "USER",
                "review_required": False,
                "ambiguity_id": None,
            },
            {
                "frame_index": 1,
                "time_seconds": 1 / 30,
                "shot_id": "shot_0001",
                "state": (
                    "AMBIGUOUS"
                    if state["status"] == "NEEDS_CONFIRMATION"
                    else "SEARCHING"
                ),
                "bbox_xyxy": None,
                "tracking_confidence": 0,
                "identity_confidence": 0,
                "identity_source": "NONE",
                "review_required": state["status"] == "NEEDS_CONFIRMATION",
                "ambiguity_id": (
                    "ambiguity_0001"
                    if state["status"] == "NEEDS_CONFIRMATION"
                    else None
                ),
            },
        ],
        "ambiguities": state["ambiguities"],
        "confirmations": state["confirmations"],
        "provenance": {
            "phase1_manifest": state["phase1_manifest"],
            "reacquisition_mode": "assisted",
        },
    }
    write_json(output / "target_timeline.json", timeline)
    write_json(
        output / "pipeline_summary.json",
        {
            "schema_version": "kickclip.target_centric_e2e_summary.v1",
            "status": state["status"],
            "decision": state["decision"],
        },
    )
    for name in (
        "target_segments.json",
        "ambiguities.json",
        "confirmations.json",
    ):
        write_json(output / name, {})
    (output / "target_timeline.csv").write_text("frame_index,state\n", encoding="utf-8")
    (output / "shot_boundaries.csv").write_text("shot_id\n", encoding="utf-8")
    (output / "report.md").write_text("# fixture\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if "timeout" in args.test_name:
        time.sleep(5)
        return 0
    if "fatal" in args.test_name:
        print("fixture fatal", flush=True)
        return 2

    output = args.output_root / args.test_name
    output.mkdir(parents=True, exist_ok=True)
    if "safe_block" in args.test_name:
        state = state_payload(
            args,
            status="COMPLETE_WITH_SAFE_BLOCK",
            decision="SAFE_BLOCK_NO_STABLE_PRECUT_TARGET_MEMORY",
            pending=None,
        )
        write_contract(output, state)
        return 0

    if not args.resume:
        external = args.project_root / "runs" / "target_centric_tracking_v2" / args.test_name
        external.mkdir(parents=True, exist_ok=True)
        contact = external / "memory_contact_sheet.jpg"
        preview = external / "memory_tracking_preview.mp4"
        contact.write_bytes(b"fake-jpeg")
        preview.write_bytes(b"fake-mp4")
        pending = {
            "type": "MEMORY_REVIEW",
            "review_stage": "MEMORY",
            "contact_sheet": str(contact),
            "preview": str(preview),
        }
        state = state_payload(
            args,
            status="NEEDS_CONFIRMATION",
            decision="PAUSE_FOR_PRECUT_TARGET_MEMORY_VISUAL_REVIEW",
            pending=pending,
        )
        write_contract(output, state)
        return 3

    current = json.loads((output / "pipeline_state.json").read_text(encoding="utf-8"))
    if args.reject_review:
        current.update(
            status="BLOCKED",
            decision=f"BLOCK_{args.reject_review}_VISUAL_REVIEW_FAIL",
            pending_action=None,
        )
        write_contract(output, current)
        return 2
    if args.approve_review == "MEMORY":
        sheet = (
            output
            / "ambiguity_candidates"
            / "ambiguity_0001"
            / "candidates.jpg"
        )
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_bytes(b"candidate-sheet")
        pending = {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": "ambiguity_0001",
            "shot_id": "shot_0001",
            "candidate_ids": [
                "shot_0001_track_0001",
                "shot_0001_track_0002",
            ],
            "recommended_candidate": None,
            "contact_sheet": str(sheet),
        }
        current = state_payload(
            args,
            status="NEEDS_CONFIRMATION",
            decision="PAUSE_AT_CROSS_SHOT_AMBIGUITY",
            pending=pending,
        )
        write_contract(output, current)
        return 3
    if args.confirmed_candidate or args.confirm_absent:
        current.update(
            status="COMPLETE",
            decision=(
                "E2E_TARGET_TIMELINE_COMPLETE"
                if args.confirmed_candidate
                else "USER_CONFIRMED_ABSENT_COMPLETE"
            ),
            pending_action=None,
        )
        current["confirmations"] = [
            {
                "ambiguity_id": args.ambiguity_id,
                "decision": (
                    "CANDIDATE_CONFIRMED"
                    if args.confirmed_candidate
                    else "TARGET_ABSENT"
                ),
            }
        ]
        write_contract(output, current)
        (output / "full_frame_tracking_preview.mp4").write_bytes(b"preview")
        (output / "target_centered_preview.mp4").write_bytes(b"centered")
        return 0
    return 2


raise SystemExit(main())

