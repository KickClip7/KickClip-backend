from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import (
    PACKAGE_NAME,
    STATUS,
    UNSUPPORTED_EVENT_CLASS,
    ImmutableCandidateInput,
    canonical_event_label,
    sha256_file,
)
from .diversity import diverse_shortlist
from .feature_extractor import RawFeatureExtractor, load_detections
from .policy import load_policy, score_candidate


class EventCandidateRankingV11Engine:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.policy_path = self.package_root / "event_candidate_ranking_policy.json"
        self.feature_schema_path = (
            self.package_root / "event_candidate_feature_schema.json"
        )
        self.input_schema_path = (
            self.package_root / "event_candidate_input_schema.json"
        )
        self.annotation_schema_path = (
            self.package_root / "event_annotation_schema.json"
        )
        self.manifest_path = self.package_root / "manifest.json"

    def run(
        self,
        *,
        immutable_input: ImmutableCandidateInput,
        event_id: str,
        event_label: str,
        event_time_sec: float,
        event_confidence: float | None,
        scene_id: str,
        scene_start_sec: float,
        scene_end_sec: float,
        shortlist_size: int,
    ) -> dict[str, Any]:
        policy = load_policy(self.policy_path)
        canonical_label = canonical_event_label(event_label)
        event_context = {
            "event_id": event_id,
            "input_event_label": event_label,
            "canonical_event_label": canonical_label,
            "event_time_sec": event_time_sec,
            "event_confidence": event_confidence,
            "scene_id": scene_id,
            "scene_start_sec": scene_start_sec,
            "scene_end_sec": scene_end_sec,
            "event_scene_local_time_sec": event_time_sec - scene_start_sec,
            "time_coordinate_system": "SOURCE_VIDEO_SECONDS",
            "candidate_time_coordinate_system": "SCENE_LOCAL_SECONDS",
        }
        immutable_metadata = {
            "scene_candidates_relative_path": immutable_input.scene_candidates_path.relative_to(
                immutable_input.discovery_root
            ).as_posix(),
            "scene_candidates_sha256": immutable_input.scene_candidates_sha256,
            "detections_relative_path": immutable_input.detections_path.relative_to(
                immutable_input.discovery_root
            ).as_posix(),
            "detections_sha256": immutable_input.detections_sha256,
            "source_video_relative_path": immutable_input.source_video_path.relative_to(
                immutable_input.discovery_root
            ).as_posix(),
            "source_video_sha256": immutable_input.source_video_sha256,
            "video_width": immutable_input.video_width,
            "video_height": immutable_input.video_height,
            "video_fps": immutable_input.video_fps,
            "video_frame_count": immutable_input.video_frame_count,
            "shot_boundaries_relative_path": immutable_input.shot_boundaries_path.relative_to(
                immutable_input.discovery_root
            ).as_posix(),
            "shot_boundaries_sha256": immutable_input.shot_boundaries_sha256,
            "candidate_manifest_sha256": immutable_input.candidate_manifest_sha256,
        }
        base = {
            "schema_version": "kickclip.event_candidate_ranking.v1_1",
            "package": PACKAGE_NAME,
            "status": STATUS,
            "automatic_target_confirmation": False,
            "event_context": event_context,
            "immutable_candidate_input": immutable_metadata,
            "freeze": {
                "source_manifest_sha256": sha256_file(self.manifest_path),
                "policy_sha256": sha256_file(self.policy_path),
                "feature_schema_sha256": sha256_file(self.feature_schema_path),
                "input_schema_sha256": sha256_file(self.input_schema_path),
                "annotation_schema_sha256": sha256_file(
                    self.annotation_schema_path
                ),
            },
            "full_gallery_fallback": [
                candidate.candidate_id for candidate in immutable_input.candidates
            ],
        }
        if canonical_label is None:
            return {
                **base,
                "ranking_status": UNSUPPORTED_EVENT_CLASS,
                "feature_availability": {
                    group: {
                        "available": False,
                        "coverage": None,
                        "reason": UNSUPPORTED_EVENT_CLASS,
                    }
                    for group in (
                        "temporal",
                        "visual",
                        "motion",
                        "broadcast",
                        "ball",
                        "field_context",
                    )
                },
                "shortlist": [],
                "all_candidates": [],
                "shortlist_decisions": [],
            }
        event_local_sec = event_time_sec - scene_start_sec
        extractor = RawFeatureExtractor(
            width=immutable_input.video_width,
            height=immutable_input.video_height,
            event_scene_local_sec=event_local_sec,
            event_window_before_sec=float(policy["event_window"]["before_sec"]),
            event_window_after_sec=float(policy["event_window"]["after_sec"]),
            closeup_area_ratio=float(policy["visual"]["closeup_area_ratio"]),
            ball_labels=set(policy["ball"]["labels"]),
            ball_low_coverage_threshold=float(
                policy["ball"]["low_coverage_threshold"]
            ),
        )
        raw_rows, availability_summary = extractor.extract_all(
            candidates=immutable_input.candidates,
            detections=load_detections(immutable_input.detections_path),
            video_path=immutable_input.source_video_path,
        )
        scored = [
            score_candidate(
                row,
                canonical_label=canonical_label,
                policy=policy,
            )
            for row in raw_rows
        ]
        shortlist, decisions = diverse_shortlist(
            scored,
            size=shortlist_size,
            width=immutable_input.video_width,
            height=immutable_input.video_height,
            policy=policy,
        )
        availability = {
            group: {
                "available": count > 0,
                "available_candidate_count": count,
                "candidate_count": len(scored),
                "coverage": (
                    count / len(scored) if scored else None
                ),
            }
            for group, count in availability_summary[
                "group_available_candidate_counts"
            ].items()
        }
        persisted_rows = [
            {key: value for key, value in row.items() if key != "_trajectory"}
            for row in sorted(scored, key=lambda item: item["rank"])
        ]
        persisted_by_id = {
            row["candidate_id"]: row for row in persisted_rows
        }
        return {
            **base,
            "ranking_status": STATUS,
            "event_policy_group": policy["event_policy_map"][canonical_label],
            "feature_availability": availability,
            "candidate_feature_availability_summary": availability_summary,
            "shortlist": [
                persisted_by_id[row["candidate_id"]] for row in shortlist
            ],
            "all_candidates": persisted_rows,
            "shortlist_decisions": decisions,
            "human_annotation_template": json.loads(
                (self.package_root / "annotation_template.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
