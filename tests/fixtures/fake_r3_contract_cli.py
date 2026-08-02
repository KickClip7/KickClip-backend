#!/usr/bin/env python
"""R3_SUBPROCESS_CONTRACT_TEST fixture only; never represents AI tracking quality."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def evidence(root: Path, ambiguity_id: str, candidate_id: str) -> dict:
    directory = root / "contract_evidence" / ambiguity_id / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, suffix in (
        ("manifest", ".json"),
        ("full_frame_context", ".jpg"),
        ("shot_clip", ".mp4"),
        ("reference_gallery", ".jpg"),
    ):
        path = directory / f"{name}{suffix}"
        path.write_bytes(f"contract:{ambiguity_id}:{candidate_id}:{name}".encode())
        files[name] = path
    return {
        "candidate_id": candidate_id,
        "shot_id": "shot_contract_next",
        "tracklet_id": "track_contract_dynamic",
        "status": "PENDING",
        "manifest_path": str(files["manifest"]),
        "manifest_sha256": sha(files["manifest"]),
        "full_frame_context_path": str(files["full_frame_context"]),
        "shot_clip_path": str(files["shot_clip"]),
        "reference_gallery_path": str(files["reference_gallery"]),
        "score_evidence": {
            "memory_revision_id": "contract_memory_r1",
            "memory_source_reference_count": 6,
            "candidate_scoring_generation": 2,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root")
    p.add_argument("--video")
    p.add_argument("--test-name", required=True)
    p.add_argument("--initial-bbox", nargs=4)
    p.add_argument("--device")
    p.add_argument("--reacquisition-mode")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ambiguity-id")
    p.add_argument("--confirmed-candidate")
    p.add_argument("--confirm-absent", action="store_true")
    p.add_argument("--rejected-candidate")
    p.add_argument("--unreviewable-candidate")
    p.add_argument("--reviewer")
    p.add_argument("--review-note")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--tracking-launch-manifest")
    p.add_argument("--shot-boundaries")
    p.add_argument("--target-selection")
    p.add_argument("--target-reference-set")
    p.add_argument("--earlier-anchor-decision")
    p.add_argument("--target-memory-revision")
    p.add_argument("--target-memory-sha256")
    p.add_argument("--candidate-scoring-generation", type=int, default=1)
    args = p.parse_args()
    root = args.output_root / args.test_name
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    timeline_path = root / "target_timeline.json"
    if not args.resume:
        candidate = evidence(root, "ambiguity_contract_0001", "runtime_dynamic_candidate_0001")
        state = {
            "schema_version": "contract.only",
            "pipeline_version": "R3_SUBPROCESS_CONTRACT_TEST",
            "status": "NEEDS_CONFIRMATION",
            "decision": "PAUSE_AT_CROSS_SHOT_AMBIGUITY",
            "pending_action": {
                "type": "CROSS_SHOT_CONFIRMATION",
                "ambiguity_id": "ambiguity_contract_0001",
                "shot_id": "shot_contract_next",
                "candidate_ids": [candidate["candidate_id"]],
            },
            "shots": [
                {"shot_id": "shot_selected", "status": "ACCEPTED"},
                {"shot_id": "shot_contract_next", "status": "AMBIGUOUS_REVIEW_REQUIRED"},
            ],
            "ambiguities": [
                {
                    "ambiguity_id": "ambiguity_contract_0001",
                    "shot_id": "shot_contract_next",
                    "status": "PENDING",
                    "review_candidates": [candidate],
                }
            ],
            "confirmations": [],
            "runtime": {
                "contract_test_only": True,
                "synthetic_tracking_used": True,
                "full_event_recommendation_e2e": "NOT_RUN",
            },
        }
        frames = [
            {"frame_index": i, "state": "ACTIVE", "bbox_xyxy": [1, 1, 10, 10]}
            for i in range(45)
        ]
        write(state_path, state)
        write(timeline_path, {"frames": frames, "provenance": {"silent_wrong_player_switches": 0}})
        return 3
    state = json.loads(state_path.read_text())
    if args.confirm_absent:
        state["status"] = "COMPLETE"
        state["decision"] = "CONTRACT_ABSENT_RESUME_COMPLETE"
        state["shots"][1]["status"] = "ABSENT_CONFIRMED"
    elif args.confirmed_candidate:
        timeline = json.loads(timeline_path.read_text())
        start = len(timeline["frames"])
        timeline["frames"].extend(
            {
                "frame_index": i,
                "state": "REACQUIRED",
                "bbox_xyxy": [2, 2, 12, 12],
            }
            for i in range(start, start + 20)
        )
        write(timeline_path, timeline)
        state["status"] = "COMPLETE"
        state["decision"] = "CONTRACT_CONFIRMED_RESUME_COMPLETE"
        state["shots"][1]["status"] = "ACCEPTED"
        state["confirmations"].append(
            {
                "ambiguity_id": args.ambiguity_id,
                "candidate_id": args.confirmed_candidate,
                "decision": "CANDIDATE_CONFIRMED",
            }
        )
    elif args.rejected_candidate:
        state["status"] = "COMPLETE_WITH_SAFE_BLOCK"
        state["decision"] = "CONTRACT_REJECTION_REPORTED"
        state["failure_code"] = "RUNTIME_REJECTION_RESUME_UNSUPPORTED"
    elif args.unreviewable_candidate:
        state["status"] = "COMPLETE_WITH_UNRESOLVED_GAPS"
        state["decision"] = "CONTRACT_UNREVIEWABLE_REPORTED"
        state["shots"][1]["status"] = "UNRESOLVED_LOW_RESOLUTION"
    state["pending_action"] = None
    write(state_path, state)
    (root / "full_frame_tracking_preview.mp4").write_bytes(b"contract-preview")
    (root / "target_centered_preview.mp4").write_bytes(b"contract-preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
