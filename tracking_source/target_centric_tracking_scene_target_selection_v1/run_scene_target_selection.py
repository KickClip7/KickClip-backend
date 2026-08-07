#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from appearance import frozen_sports_osnet_score_function
from common import atomic_json, read_json, sha256_file
from contracts import candidate_cache_key
from discovery import discover_from_frozen_detections
from earlier_anchor import (
    confirm_earlier_anchor,
    propose_earlier_candidates,
    selected_shot_fallback,
)
from gallery import render_gallery_html
from launch import create_tracking_launch_manifest
from selection import create_target_reference_set, create_target_selection


PACKAGE = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover")
    discover.add_argument("--project-root", type=Path, required=True)
    discover.add_argument("--scene-id", required=True)
    discover.add_argument("--discovery-id")
    discover.add_argument("--video", type=Path, required=True)
    discover.add_argument("--detections-csv", type=Path, required=True)
    discover.add_argument("--shot-boundaries", type=Path, required=True)
    discover.add_argument("--output-root", type=Path, required=True)

    select = sub.add_parser("select")
    select.add_argument("--output-root", type=Path, required=True)
    select.add_argument("--video", type=Path, required=True)
    select.add_argument("--candidate-id", required=True)
    select.add_argument("--reviewer", required=True)
    select.add_argument("--selection-revision", type=int)

    propose = sub.add_parser("propose-earlier")
    propose.add_argument("--project-root", type=Path, required=True)
    propose.add_argument("--output-root", type=Path, required=True)
    propose.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")

    confirm = sub.add_parser("confirm-earlier")
    confirm.add_argument("--output-root", type=Path, required=True)
    confirm.add_argument("--candidate-id")
    confirm.add_argument("--reviewer", required=True)
    confirm.add_argument("--reject-all", action="store_true")

    fallback = sub.add_parser("start-selected-shot")
    fallback.add_argument("--output-root", type=Path, required=True)
    fallback.add_argument("--reviewer", required=True)

    launch = sub.add_parser("prepare-tracking")
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--video", type=Path, required=True)
    launch.add_argument("--shot-boundaries", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    policy_path = PACKAGE / "scene_target_selection_policy.json"
    policy = read_json(policy_path)
    package_manifest = PACKAGE / "scene_target_selection_frozen_manifest.json"
    package_sha = (
        sha256_file(package_manifest)
        if package_manifest.is_file()
        else "UNFROZEN_DEVELOPMENT_PACKAGE"
    )
    if args.command == "discover":
        output = args.output_root.resolve()
        result = discover_from_frozen_detections(
            project_root=args.project_root.resolve(),
            scene_id=args.scene_id,
            candidate_namespace=args.discovery_id or args.scene_id,
            video=args.video.resolve(),
            detections_csv=args.detections_csv.resolve(),
            shot_boundaries_path=args.shot_boundaries.resolve(),
            output_root=output,
            policy=policy,
            candidate_policy_sha256=sha256_file(policy_path),
            package_manifest_sha256=package_sha,
        )
        gallery = read_json(output / "candidate_gallery.json")
        render_gallery_html(
            output_root=output,
            candidates=result,
            gallery=gallery,
        )
        cache_key = candidate_cache_key(
            video_sha256=result["video"]["sha256"],
            scene_start_sec=0.0,
            scene_end_sec=result["video"]["frame_count"]
            / result["video"]["fps"],
            shot_boundary_sha256=result["shot_boundary"]["sha256"],
            rfdetr_checkpoint_sha256=result["candidates"][0]["provenance"][
                "detector_checkpoint_sha256"
            ]
            if result["candidates"]
            else sha256_file(
                args.project_root.resolve()
                / "weights/rfdetr/checkpoint_best_regular.pth"
            ),
            policy_sha256=sha256_file(policy_path),
            package_manifest_sha256=package_sha,
            schema_version=result["schema_version"],
        )
        atomic_json(
            output / "scene_candidate_manifest.json",
            {
                "schema_version": "kickclip.scene_candidate_manifest.v1",
                "scene_id": args.scene_id,
                "candidate_cache_key": cache_key,
                "candidate_count": len(result["candidates"]),
                "shot_count": result["shot_boundary"]["shot_count"],
                "package_manifest_sha256": package_sha,
                "policy_sha256": sha256_file(policy_path),
                "portable_artifacts": True,
                "absolute_paths_exposed": False,
            },
        )
        print(json.dumps({"candidate_count": len(result["candidates"]), "cache_key": cache_key}))
        return 0
    output = args.output_root.resolve()
    if args.command == "select":
        selection = create_target_selection(
            output_root=output,
            selected_candidate_id=args.candidate_id,
            reviewer=args.reviewer,
            production_manifest_sha256=(
                "751338f51c4f7c08bb24576e5afea7d82b9cbbdefbb651ff1a3c1d5c45ffcc86"
            ),
            selection_revision=args.selection_revision,
        )
        create_target_reference_set(
            output_root=output,
            video=args.video.resolve(),
            selection=selection,
            maximum_references=int(policy["references"]["maximum_references"]),
        )
        print(json.dumps(selection))
        return 0
    selection = read_json(output / "target_selection.json")
    if args.command == "propose-earlier":
        references = read_json(output / "target_reference_set.json")
        score = frozen_sports_osnet_score_function(
            project_root=args.project_root.resolve(),
            output_root=output,
            reference_set=references,
            requested_device=args.device,
        )
        result = propose_earlier_candidates(
            output_root=output,
            selection=selection,
            score_candidates=score,
            maximum_candidates=int(
                policy["earlier_anchor"]["maximum_review_candidates"]
            ),
        )
    elif args.command == "confirm-earlier":
        result = confirm_earlier_anchor(
            output_root=output,
            selection=selection,
            candidate_id=args.candidate_id,
            reviewer=args.reviewer,
            reject_all=args.reject_all,
        )
    elif args.command == "start-selected-shot":
        result = selected_shot_fallback(
            output_root=output,
            selection=selection,
            reviewer=args.reviewer,
        )
    else:
        manifest = read_json(output / "scene_candidate_manifest.json")
        r3_manifest = (
            PACKAGE.parent
            / "target_centric_tracking_v2_production_r3"
            / "target_centric_tracking_v2_production_r3_manifest.json"
        )
        result = create_tracking_launch_manifest(
            output_root=output,
            scene_video=args.video.resolve(),
            shot_boundaries_path=args.shot_boundaries.resolve(),
            selection_path=output / "target_selection.json",
            reference_set_path=output / "target_reference_set.json",
            earlier_decision_path=output / "earlier_anchor_decision.json",
            candidate_cache_key_value=manifest["candidate_cache_key"],
            r3_manifest_sha256=(
                sha256_file(r3_manifest)
                if r3_manifest.is_file()
                else None
            ),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
