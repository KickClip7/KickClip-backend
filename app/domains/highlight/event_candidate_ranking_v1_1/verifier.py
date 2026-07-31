from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .contract import ImmutableCandidateInput, canonical_event_label, sha256_file
from .service import EventCandidateRankingV11Engine


@dataclass(frozen=True)
class EventRankingShadowVerification:
    event_ranking_shadow_runtime_verified: bool
    full_event_recommendation_e2e_verified: bool
    code: str
    message: str
    manifest_sha256: str | None = None


class EventCandidateRankingV11Verifier:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()

    def check(self) -> EventRankingShadowVerification:
        manifest_path = self.package_root / "manifest.json"
        if not manifest_path.is_file():
            return self._failed("EVENT_RANKING_V1_1_MANIFEST_MISSING")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._failed("EVENT_RANKING_V1_1_MANIFEST_INVALID")
        if (
            manifest.get("package")
            != "target_centric_tracking_event_candidate_ranking_v1_1"
            or manifest.get("status") != "PROVISIONAL_SHADOW_ONLY"
            or manifest.get("automatic_target_confirmation") is not False
            or manifest.get("full_event_recommendation_e2e_verified")
            is not False
        ):
            return self._failed("EVENT_RANKING_V1_1_FREEZE_INVALID")
        project_root = self.package_root.parents[3]
        for relative_path, expected in (manifest.get("sha256") or {}).items():
            path = (project_root / relative_path).resolve()
            if (
                not path.is_relative_to(project_root)
                or not path.is_file()
                or sha256_file(path) != expected
            ):
                return self._failed(
                    "EVENT_RANKING_V1_1_SOURCE_HASH_MISMATCH"
                )
        try:
            self._synthetic_shadow_smoke()
        except Exception as exc:
            return EventRankingShadowVerification(
                event_ranking_shadow_runtime_verified=False,
                full_event_recommendation_e2e_verified=False,
                code="EVENT_RANKING_V1_1_SHADOW_SMOKE_FAILED",
                message=str(exc),
                manifest_sha256=sha256_file(manifest_path),
            )
        return EventRankingShadowVerification(
            event_ranking_shadow_runtime_verified=True,
            full_event_recommendation_e2e_verified=False,
            code="EVENT_RANKING_V1_1_SHADOW_RUNTIME_VERIFIED",
            message="V1.1 source/policy/schema hashes and synthetic shadow smoke passed.",
            manifest_sha256=sha256_file(manifest_path),
        )

    def _failed(self, code: str) -> EventRankingShadowVerification:
        return EventRankingShadowVerification(
            event_ranking_shadow_runtime_verified=False,
            full_event_recommendation_e2e_verified=False,
            code=code,
            message="Event ranking V1.1 shadow runtime verification failed.",
        )

    def _synthetic_shadow_smoke(self) -> None:
        if canonical_event_label("Penalty") is not None:
            raise ValueError("Unsupported-event guard failed.")
        with tempfile.TemporaryDirectory(prefix="kickclip-ecr-v11-") as tmp:
            root = Path(tmp).resolve()
            video_path = root / "scene.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (64, 48),
            )
            if not writer.isOpened():
                raise ValueError("Synthetic video writer is unavailable.")
            for index in range(4):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                cv2.rectangle(
                    frame,
                    (10 + index, 8),
                    (30 + index, 44),
                    (255, 255, 255),
                    -1,
                )
                writer.write(frame)
            writer.release()
            candidates_path = root / "scene_candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": "candidate_smoke",
                                "shot_id": "shot_1",
                                "shot_index": 0,
                                "local_tracklet_id": "track_1",
                                "quality": {"trackability_score": 0.8},
                                "observations": [
                                    {
                                        "global_frame": index,
                                        "scene_local_frame": index,
                                        "scene_local_time_sec": index / 5,
                                        "bbox_xyxy": [
                                            10 + index,
                                            8,
                                            30 + index,
                                            44,
                                        ],
                                        "detector_confidence": 0.9,
                                        "detection_id": f"det_{index}",
                                    }
                                    for index in range(4)
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            detections_path = root / "detections.json"
            detections_path.write_text('{"detections":[]}', encoding="utf-8")
            boundaries_path = root / "boundaries.json"
            boundaries_path.write_text('{"shots":[]}', encoding="utf-8")
            immutable = ImmutableCandidateInput.load(
                discovery_root=root,
                scene_candidates_relative_path=candidates_path.name,
                scene_candidates_sha256=sha256_file(candidates_path),
                detections_relative_path=detections_path.name,
                detections_sha256=sha256_file(detections_path),
                source_video_relative_path=video_path.name,
                source_video_sha256=sha256_file(video_path),
                shot_boundaries_relative_path=boundaries_path.name,
                shot_boundaries_sha256=sha256_file(boundaries_path),
                candidate_manifest_sha256="0" * 64,
                video_width=64,
                video_height=48,
                video_fps=5.0,
                video_frame_count=4,
            )
            result = EventCandidateRankingV11Engine(self.package_root).run(
                immutable_input=immutable,
                event_id="event_smoke",
                event_label="Goal",
                event_time_sec=0.4,
                event_confidence=0.8,
                scene_id="scene_smoke",
                scene_start_sec=0.0,
                scene_end_sec=1.0,
                shortlist_size=3,
            )
            if (
                result["automatic_target_confirmation"] is not False
                or not result["all_candidates"]
                or result["all_candidates"][0]["raw_features"]["visual"][
                    "bbox_area_ratio"
                ]
                is None
                or result["all_candidates"][0]["raw_features"]["motion"][
                    "center_velocity"
                ]
                is None
            ):
                raise ValueError("Synthetic raw feature smoke is incomplete.")
