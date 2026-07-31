from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    ImmutableCandidateInput,
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.contract import (
    ResolvedEventContext,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.verifier import (
    EventCandidateRankingV112Verifier,
)

from .service import EventCandidateRankingV112aEngine


@dataclass(frozen=True)
class EventRankingCompatibilityVerification:
    event_ranking_compatibility_runtime_verified: bool
    full_event_recommendation_e2e_verified: bool
    code: str
    message: str
    manifest_sha256: str | None = None


class EventCandidateRankingV112aVerifier:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()
        self.project_root = self.package_root.parents[3]

    @staticmethod
    def _load_json(path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path.name}.")
        return value

    def check(self) -> EventRankingCompatibilityVerification:
        manifest_path = self.package_root / "manifest.json"
        if not manifest_path.is_file():
            return self._failed("EVENT_RANKING_V1_1_2A_MANIFEST_MISSING")
        try:
            manifest = self._load_json(manifest_path)
            self._verify_freeze(manifest)
            schemas = self._verify_schemas()
            self._smoke(schemas)
        except Exception as exc:
            return EventRankingCompatibilityVerification(
                event_ranking_compatibility_runtime_verified=False,
                full_event_recommendation_e2e_verified=False,
                code=(
                    "EVENT_RANKING_V1_1_2A_COMPATIBILITY_"
                    "VERIFICATION_FAILED"
                ),
                message=str(exc),
                manifest_sha256=sha256_file(manifest_path),
            )
        return EventRankingCompatibilityVerification(
            event_ranking_compatibility_runtime_verified=True,
            full_event_recommendation_e2e_verified=False,
            code="EVENT_RANKING_V1_1_2A_COMPATIBILITY_RUNTIME_VERIFIED",
            message=(
                "Frozen baselines, reviewed-shot aliases, reset-segment "
                "ball continuity, snapshot/cache contracts, and tightened "
                "input/candidate/output schemas passed."
            ),
            manifest_sha256=sha256_file(manifest_path),
        )

    def _failed(
        self,
        code: str,
    ) -> EventRankingCompatibilityVerification:
        return EventRankingCompatibilityVerification(
            event_ranking_compatibility_runtime_verified=False,
            full_event_recommendation_e2e_verified=False,
            code=code,
            message="Event ranking V1.1.2a verification failed.",
        )

    def _verify_freeze(self, manifest: dict) -> None:
        if (
            manifest.get("package")
            != "target_centric_tracking_event_candidate_ranking_v1_1_2a"
            or manifest.get("status") != "PROVISIONAL_SHADOW_ONLY"
            or manifest.get("ranking_weights_changed") is not False
            or manifest.get("automatic_target_confirmation") is not False
            or manifest.get("full_event_recommendation_e2e_verified")
            is not False
        ):
            raise ValueError("V1.1.2a freeze contract is invalid.")
        baseline = self._load_json(
            self.package_root
            / "frozen_v1_1_v1_1_1_v1_1_2_baseline.json"
        )
        for relative, expected in baseline["sha256"].items():
            if sha256_file(self.project_root / relative) != expected:
                raise ValueError(f"Frozen baseline changed: {relative}.")
        for relative, expected in (manifest.get("sha256") or {}).items():
            path = (self.project_root / relative).resolve()
            if (
                not path.is_relative_to(self.project_root)
                or not path.is_file()
                or sha256_file(path) != expected
            ):
                raise ValueError(
                    f"V1.1.2a source hash mismatch: {relative}."
                )
        v112_root = (
            self.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_2"
        )
        prior = EventCandidateRankingV112Verifier(v112_root).check()
        if not prior.event_ranking_contract_runtime_verified:
            raise ValueError(f"V1.1.2 baseline failed: {prior.code}.")

    def _verify_schemas(self) -> dict[str, dict]:
        result = {}
        for name in (
            "input_schema.json",
            "candidate_feature_schema.json",
            "ranking_output_schema.json",
        ):
            schema = self._load_json(self.package_root / name)
            Draft202012Validator.check_schema(schema)
            result[name] = schema
        return result

    def _smoke(self, schemas: dict[str, dict]) -> None:
        with tempfile.TemporaryDirectory(prefix="kickclip-v112a-") as temp:
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
            for frame_index in range(10):
                frame = np.zeros((80, 100, 3), dtype=np.uint8)
                cv2.rectangle(
                    frame,
                    (10 + frame_index, 10),
                    (35 + frame_index, 70),
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
                                "shot_id": shot_id,
                                "shot_index": shot_index,
                                "local_tracklet_id": f"track_{index}",
                                "quality": {"trackability_score": 0.8},
                                "observations": [
                                    {
                                        "global_frame": frame,
                                        "scene_local_frame": frame,
                                        "scene_local_time_sec": frame / 5,
                                        "bbox_xyxy": [
                                            5 + index * 15 + frame,
                                            10,
                                            17 + index * 15 + frame,
                                            70,
                                        ],
                                        "detector_confidence": 0.9,
                                        "detection_id": f"d_{index}_{frame}",
                                        "shot_id": shot_id,
                                    }
                                    for frame in frames
                                ],
                            }
                            for index, (shot_index, shot_id, frames) in enumerate(
                                (
                                    (0, "shot_0000", range(0, 5)),
                                    (0, "shot_0000", range(0, 5)),
                                    (1, "shot_0001", range(5, 10)),
                                    (1, "shot_0001", range(5, 10)),
                                )
                            )
                        ]
                    }
                ),
                encoding="utf-8",
            )
            detections = root / "detections.json"
            detections.write_text(
                json.dumps(
                    {
                        "detections": [
                            {
                                "frame": frame,
                                "label": "ball",
                                "bbox_xyxy": [
                                    25 + frame,
                                    62,
                                    29 + frame,
                                    66,
                                ],
                                "confidence": 0.95,
                            }
                            for frame in range(10)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            shots = root / "shots.json"
            fixture = (
                self.project_root
                / "tests/fixtures/r2_r3_reviewed_shot_artifact.json"
            )
            shots.write_text(
                fixture.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            candidate_sha = sha256_file(candidates)
            immutable = ImmutableCandidateInput.load(
                discovery_root=root,
                scene_candidates_relative_path=candidates.name,
                scene_candidates_sha256=candidate_sha,
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
                video_frame_count=10,
            )
            resolved = ResolvedEventContext(
                event_id="event",
                event_label="goal",
                canonical_event_label="goal",
                event_time_sec=1.2,
                event_confidence=0.8,
                event_source_job_id="job",
                event_source_artifact_id="artifact",
                scene_id="scene",
                scene_start_sec=0.0,
                scene_end_sec=2.0,
                match_id="match",
                project_id="project",
                revision_id="revision",
            )
            input_document = {
                "event_id": "event",
                "scene_id": "scene",
                "revision_id": "revision",
                "project_id": "project",
                "match_id": "match",
                "event_label": "goal",
                "event_time_sec": 1.2,
                "scene_start_sec": 0.0,
                "scene_end_sec": 2.0,
                "scene_candidates_sha256": candidate_sha,
                "stored_discovery_scene_candidates_sha256": candidate_sha,
                "manifest_declared_scene_candidates_sha256": candidate_sha,
                "scene_candidate_manifest_sha256": "1" * 64,
                "detections_sha256": sha256_file(detections),
                "source_video_sha256": sha256_file(video),
                "shot_boundaries_sha256": sha256_file(shots),
                "discovery_id": "discovery",
                "discovery_artifact_root": "discovery",
                "shortlist_size": 3,
            }
            Draft202012Validator(
                schemas["input_schema.json"]
            ).validate(input_document)
            result = EventCandidateRankingV112aEngine(
                self.package_root
            ).run(
                immutable_input=immutable,
                resolved_event=resolved,
                shortlist_size=3,
                immutable_sha_contract={
                    "current_scene_candidates_sha256": candidate_sha,
                    "stored_discovery_scene_candidates_sha256": (
                        candidate_sha
                    ),
                    "manifest_declared_scene_candidates_sha256": (
                        candidate_sha
                    ),
                },
                scene_video_duration_contract={
                    "scene_duration_sec": 2.0,
                    "video_duration_sec": 2.0,
                    "difference_sec": 0.0,
                    "tolerance_sec": 0.25,
                },
                discovery_snapshot={
                    "discovery_id": "discovery",
                    "discovery_artifact_root": "discovery",
                    "scene_id": "scene",
                    "scene_candidate_manifest_sha256": "1" * 64,
                },
            )
            candidate_validator = Draft202012Validator(
                schemas["candidate_feature_schema.json"]
            )
            for candidate in result["all_candidates"]:
                candidate_validator.validate(candidate)
            Draft202012Validator(
                schemas["ranking_output_schema.json"]
            ).validate(result)
            if (
                result["automatic_target_confirmation"] is not False
                or result["shot_boundary_input_audit"]["status"] != "PASS"
                or result["shot_boundary_input_audit"][
                    "unapproved_review_count"
                ]
                != 0
                or not result["all_candidates"]
                or any(
                    "selected_ball_observation_count"
                    not in row["raw_features"]["ball"]
                    for row in result["all_candidates"]
                )
            ):
                raise ValueError(
                    "V1.1.2a compatibility smoke assertions failed."
                )
