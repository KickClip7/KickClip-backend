from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    STATUS,
    UNSUPPORTED_EVENT_CLASS,
    ImmutableCandidateInput,
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1.feature_extractor import (
    load_detections,
)
from app.domains.highlight.event_candidate_ranking_v1_1.policy import (
    load_policy,
    score_candidate,
)

from . import PACKAGE_NAME
from .contract import ResolvedEventContext, audit_shot_contract
from .diversity import diverse_shortlist_v111
from .feature_extractor import RawFeatureExtractorV111


class EventCandidateRankingV111Engine:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.project_root = self.package_root.parents[3]
        self.v11_package_root = (
            self.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1"
        )
        self.ranking_policy_path = (
            self.v11_package_root / "event_candidate_ranking_policy.json"
        )
        self.safety_policy_path = self.package_root / "safety_policy.json"
        self.feature_schema_path = (
            self.package_root / "event_candidate_feature_schema.json"
        )
        self.manifest_path = self.package_root / "manifest.json"

    def run(
        self,
        *,
        immutable_input: ImmutableCandidateInput,
        resolved_event: ResolvedEventContext,
        shortlist_size: int,
    ) -> dict[str, Any]:
        ranking_policy = load_policy(self.ranking_policy_path)
        safety_policy = json.loads(
            self.safety_policy_path.read_text(encoding="utf-8")
        )
        if safety_policy.get("ranking_weights_changed") is not False:
            raise ValueError("V1.1.1 may not change ranking weights.")
        canonical_label = resolved_event.canonical_event_label
        base = {
            "schema_version": "kickclip.event_candidate_ranking.v1_1_1",
            "package": PACKAGE_NAME,
            "status": STATUS,
            "automatic_target_confirmation": False,
            "production_recommendation_ui": "BLOCKED",
            "event_context": {
                **resolved_event.to_dict(),
                "source": "SERVER_RESOLVED_TIMELINE_EVENT",
                "client_event_label_accepted": False,
                "client_event_time_accepted": False,
                "client_scene_bounds_accepted": False,
                "event_scene_local_time_sec": (
                    resolved_event.event_time_sec
                    - resolved_event.scene_start_sec
                ),
            },
            "freeze": {
                "v1_1_baseline_manifest_sha256": sha256_file(
                    self.v11_package_root / "manifest.json"
                ),
                "ranking_policy_sha256": sha256_file(
                    self.ranking_policy_path
                ),
                "safety_policy_sha256": sha256_file(self.safety_policy_path),
                "feature_schema_sha256": sha256_file(
                    self.feature_schema_path
                ),
                "source_manifest_sha256": sha256_file(self.manifest_path),
            },
            "full_gallery_fallback": [
                candidate.candidate_id
                for candidate in immutable_input.candidates
            ],
        }
        if canonical_label is None:
            return {
                **base,
                "ranking_status": UNSUPPORTED_EVENT_CLASS,
                "shot_boundary_input_audit": None,
                "feature_completeness_state": "NO_RELIABLE_SHORTLIST",
                "shortlist": [],
                "all_candidates": [],
                "shortlist_decisions": [],
            }
        shot_audit = audit_shot_contract(
            immutable_input.shot_boundaries_path,
            candidates=immutable_input.candidates,
            frame_count=immutable_input.video_frame_count,
        )
        if shot_audit["status"] != "PASS":
            return {
                **base,
                "ranking_status": "SHOT_CONTRACT_INVALID",
                "shot_boundary_input_audit": shot_audit,
                "feature_completeness_state": "NO_RELIABLE_SHORTLIST",
                "shortlist": [],
                "all_candidates": [],
                "shortlist_decisions": [],
            }
        extractor = RawFeatureExtractorV111(
            canonical_event_label=canonical_label,
            safety_policy=safety_policy,
            width=immutable_input.video_width,
            height=immutable_input.video_height,
            event_scene_local_sec=(
                resolved_event.event_time_sec
                - resolved_event.scene_start_sec
            ),
            event_window_before_sec=float(
                ranking_policy["event_window"]["before_sec"]
            ),
            event_window_after_sec=float(
                ranking_policy["event_window"]["after_sec"]
            ),
            closeup_area_ratio=float(
                ranking_policy["visual"]["closeup_area_ratio"]
            ),
            ball_labels=set(ranking_policy["ball"]["labels"]),
            ball_low_coverage_threshold=float(
                ranking_policy["ball"]["low_coverage_threshold"]
            ),
        )
        raw_rows, summary = extractor.extract_all(
            candidates=immutable_input.candidates,
            detections=load_detections(immutable_input.detections_path),
            video_path=immutable_input.source_video_path,
        )
        scored = [
            score_candidate(
                row,
                canonical_label=canonical_label,
                policy=ranking_policy,
            )
            for row in raw_rows
        ]
        shortlist, decisions = diverse_shortlist_v111(
            scored,
            size=shortlist_size,
            width=immutable_input.video_width,
            height=immutable_input.video_height,
            policy=ranking_policy,
        )
        minimum = int(
            safety_policy["feature_completeness"][
                "minimum_reliable_shortlist_size"
            ]
        )
        if len(shortlist) < minimum:
            completeness_state = "NO_RELIABLE_SHORTLIST"
            for row in decisions:
                if row["decision"] == "INCLUDED":
                    row["decision"] = "EXCLUDED"
                    row["reason"] = "INSUFFICIENT_RELIABLE_CANDIDATES"
            shortlist = []
        elif all(
            row["reliability_state"] == "READY" for row in shortlist
        ):
            completeness_state = "READY"
        else:
            completeness_state = "PARTIAL_FEATURES"
        persisted = [
            {key: value for key, value in row.items() if key != "_trajectory"}
            for row in sorted(scored, key=lambda item: item["rank"])
        ]
        by_id = {row["candidate_id"]: row for row in persisted}
        return {
            **base,
            "ranking_status": STATUS,
            "shot_boundary_input_audit": shot_audit,
            "feature_completeness_state": completeness_state,
            "feature_completeness_score": (
                sum(
                    row["feature_completeness_score"]
                    for row in raw_rows
                )
                / len(raw_rows)
                if raw_rows
                else None
            ),
            "candidate_feature_availability_summary": summary,
            "shortlist": [
                by_id[row["candidate_id"]] for row in shortlist
            ],
            "all_candidates": persisted,
            "shortlist_decisions": decisions,
        }
