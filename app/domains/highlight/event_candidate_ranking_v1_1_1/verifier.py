from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    ImmutableCandidateInput,
    sha256_file,
)

from .contract import ResolvedEventContext
from .service import EventCandidateRankingV111Engine


@dataclass(frozen=True)
class EventRankingSafetyVerification:
    event_ranking_safety_runtime_verified: bool
    full_event_recommendation_e2e_verified: bool
    code: str
    message: str
    manifest_sha256: str | None = None


class EventCandidateRankingV111Verifier:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.project_root = self.package_root.parents[3]

    def check(self) -> EventRankingSafetyVerification:
        manifest_path = self.package_root / "manifest.json"
        if not manifest_path.is_file():
            return self._failed("EVENT_RANKING_V1_1_1_MANIFEST_MISSING")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._failed("EVENT_RANKING_V1_1_1_MANIFEST_INVALID")
        if (
            manifest.get("package")
            != "target_centric_tracking_event_candidate_ranking_v1_1_1"
            or manifest.get("status") != "PROVISIONAL_SHADOW_ONLY"
            or manifest.get("ranking_weights_changed") is not False
            or manifest.get("automatic_target_confirmation") is not False
            or manifest.get("full_event_recommendation_e2e_verified")
            is not False
        ):
            return self._failed("EVENT_RANKING_V1_1_1_FREEZE_INVALID")
        for relative, expected in (manifest.get("sha256") or {}).items():
            path = (self.project_root / relative).resolve()
            if (
                not path.is_relative_to(self.project_root)
                or not path.is_file()
                or sha256_file(path) != expected
            ):
                return self._failed(
                    "EVENT_RANKING_V1_1_1_SOURCE_HASH_MISMATCH"
                )
        baseline = json.loads(
            (self.package_root / "frozen_v1_1_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        for relative, expected in baseline["sha256"].items():
            if sha256_file(self.project_root / relative) != expected:
                return self._failed("EVENT_RANKING_V1_1_BASELINE_CHANGED")
        try:
            self._smoke()
        except Exception as exc:
            return EventRankingSafetyVerification(
                event_ranking_safety_runtime_verified=False,
                full_event_recommendation_e2e_verified=False,
                code="EVENT_RANKING_V1_1_1_SMOKE_FAILED",
                message=str(exc),
                manifest_sha256=sha256_file(manifest_path),
            )
        return EventRankingSafetyVerification(
            event_ranking_safety_runtime_verified=True,
            full_event_recommendation_e2e_verified=False,
            code="EVENT_RANKING_V1_1_1_SAFETY_RUNTIME_VERIFIED",
            message=(
                "V1.1 baseline, V1.1.1 hashes, server-event output contract, "
                "shot audit, bounded decoding, and reliability smoke passed."
            ),
            manifest_sha256=sha256_file(manifest_path),
        )

    def _failed(self, code: str) -> EventRankingSafetyVerification:
        return EventRankingSafetyVerification(
            event_ranking_safety_runtime_verified=False,
            full_event_recommendation_e2e_verified=False,
            code=code,
            message="Event ranking V1.1.1 safety verification failed.",
        )

    def _smoke(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kickclip-v111-") as temp:
            root = Path(temp)
            video = root / "scene.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (100, 80),
            )
            if not writer.isOpened():
                raise ValueError("Synthetic video writer is unavailable.")
            for frame_index in range(8):
                frame = np.zeros((80, 100, 3), dtype=np.uint8)
                for offset in (0, 25, 50):
                    cv2.rectangle(
                        frame,
                        (5 + offset + frame_index, 10),
                        (22 + offset + frame_index, 70),
                        (255, 255, 255),
                        -1,
                    )
                writer.write(frame)
            writer.release()
            candidates = root / "scene_candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": f"candidate_{index}",
                                "shot_id": "shot_1",
                                "shot_index": 0,
                                "local_tracklet_id": f"track_{index}",
                                "quality": {"trackability_score": 0.8},
                                "observations": [
                                    {
                                        "global_frame": frame,
                                        "scene_local_frame": frame,
                                        "scene_local_time_sec": frame / 5,
                                        "bbox_xyxy": [
                                            5 + 25 * index + frame,
                                            10,
                                            22 + 25 * index + frame,
                                            70,
                                        ],
                                        "detector_confidence": 0.9,
                                        "detection_id": f"d_{index}_{frame}",
                                        "shot_id": "shot_1",
                                    }
                                    for frame in range(8)
                                ],
                            }
                            for index in range(3)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            detections = root / "detections.json"
            detections.write_text('{"detections":[]}', encoding="utf-8")
            shots = root / "shots.json"
            shots.write_text(
                '{"shots":[{"shot_id":"shot_1","start_frame":0,'
                '"end_frame":7,"status":"REVIEWED"}]}',
                encoding="utf-8",
            )
            immutable = ImmutableCandidateInput.load(
                discovery_root=root,
                scene_candidates_relative_path=candidates.name,
                scene_candidates_sha256=sha256_file(candidates),
                detections_relative_path=detections.name,
                detections_sha256=sha256_file(detections),
                source_video_relative_path=video.name,
                source_video_sha256=sha256_file(video),
                shot_boundaries_relative_path=shots.name,
                shot_boundaries_sha256=sha256_file(shots),
                candidate_manifest_sha256="0" * 64,
                video_width=100,
                video_height=80,
                video_fps=5.0,
                video_frame_count=8,
            )
            resolved = ResolvedEventContext(
                event_id="event",
                event_label="goal",
                canonical_event_label="goal",
                event_time_sec=0.8,
                event_confidence=0.8,
                event_source_job_id="job",
                event_source_artifact_id="artifact",
                scene_id="scene",
                scene_start_sec=0.0,
                scene_end_sec=1.6,
                match_id="match",
                project_id="project",
                revision_id="revision",
            )
            result = EventCandidateRankingV111Engine(
                self.package_root
            ).run(
                immutable_input=immutable,
                resolved_event=resolved,
                shortlist_size=3,
            )
            first = result["all_candidates"][0]
            if (
                result["automatic_target_confirmation"] is not False
                or result["shot_boundary_input_audit"]["status"] != "PASS"
                or result["event_context"]["source"]
                != "SERVER_RESOLVED_TIMELINE_EVENT"
                or len(result["shortlist"]) != 3
                or first["raw_features"]["broadcast"][
                    "cross_shot_repeated_focus"
                ]
                is not None
                or result["candidate_feature_availability_summary"][
                    "frame_decoding"
                ]["peak_cached_frames"]
                > 4
            ):
                raise ValueError("V1.1.1 safety smoke assertions failed.")
