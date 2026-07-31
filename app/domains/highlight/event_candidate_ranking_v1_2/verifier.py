from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .contract import sha256_file
from .service import EventCandidateRankingV12ShortlistPatch


@dataclass(frozen=True)
class V12VerificationResult:
    verified: bool
    package: str
    manifest_sha256: str
    shortlist_policy_sha256: str
    synthetic_smoke: bool
    ranking_weights_changed: bool
    automatic_target_confirmation: bool


class EventCandidateRankingV12Verifier:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(path)
        return value

    def _verify_manifest(self) -> dict[str, Any]:
        manifest = self._load(self.package_root / "manifest.json")
        project_root = self.package_root.parents[3]
        for relative, expected in (manifest.get("sha256") or {}).items():
            path = project_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"V1.2 manifest SHA mismatch: {relative}")
        if (
            manifest.get("ranking_weights_changed") is not False
            or manifest.get("scores_recomputed") is not False
            or manifest.get("human_labels_used_for_shortlist") is not False
            or manifest.get("automatic_target_confirmation") is not False
        ):
            raise ValueError("V1.2 safety flags are invalid.")
        return manifest

    @staticmethod
    def _row(rank: int, shot: int, score: float) -> dict[str, Any]:
        return {
            "candidate_id": f"candidate_{rank:02d}",
            "rank": rank,
            "recommendation_score": score,
            "shot_id": f"shot_{shot:04d}",
            "reliability_state": "PARTIAL_FEATURES",
        }

    def _smoke(self) -> bool:
        rows = [
            self._row(1, 2, 0.90),
            self._row(2, 2, 0.89),
            self._row(3, 2, 0.88),
            self._row(4, 1, 0.87),
            self._row(5, 1, 0.86),
            self._row(6, 0, 0.85),
        ]
        source = {
            "package": (
                "target_centric_tracking_event_candidate_ranking_v1_1_2a"
            ),
            "schema_version": "kickclip.event_candidate_ranking.v1_1_2a",
            "automatic_target_confirmation": False,
            "event_context": {"event_scene_local_time_sec": 2.4},
            "all_candidates": rows,
            "full_gallery_fallback": [
                row["candidate_id"] for row in rows
            ],
            "freeze": {
                "source_manifest_sha256": "1" * 64,
                "ranking_policy_sha256": "2" * 64,
            },
        }
        shots = {
            "video": {"frame_count": 90},
            "shots": [
                {
                    "shot_id": "shot_0000",
                    "shot_index": 0,
                    "start_frame": 0,
                    "end_frame_inclusive": 29,
                    "start_time_sec": 0.0,
                    "end_time_sec": 1.0,
                    "review_state": "REVIEWED_PASS",
                },
                {
                    "shot_id": "shot_0001",
                    "shot_index": 1,
                    "start_frame": 30,
                    "end_frame_inclusive": 59,
                    "start_time_sec": 1.0,
                    "end_time_sec": 2.0,
                    "review_state": "REVIEWED_PASS",
                },
                {
                    "shot_id": "shot_0002",
                    "shot_index": 2,
                    "start_frame": 60,
                    "end_frame_inclusive": 89,
                    "start_time_sec": 2.0,
                    "end_time_sec": 3.0,
                    "review_state": "REVIEWED_PASS",
                },
            ],
        }
        output = EventCandidateRankingV12ShortlistPatch(
            self.package_root
        ).run(
            source_ranking=source,
            reviewed_shots=shots,
            source_ranking_artifact_id="artifact",
            source_ranking_sha256="3" * 64,
            shortlist_size=5,
        )
        Draft202012Validator(
            self._load(self.package_root / "output_schema.json")
        ).validate(output)
        ids = [row["candidate_id"] for row in output["shortlist"]]
        return bool(
            len(ids) == 5
            and max(output["shot_counts"].values()) <= 2
            and output["shot_context"]["action_adjacent_slot_active"]
            and "candidate_04" in ids
            and output["global_ranking_invariant"][
                "original_global_ranks_unchanged"
            ]
            and output["automatic_target_confirmation"] is False
        )

    def check(self) -> V12VerificationResult:
        manifest = self._verify_manifest()
        smoke = self._smoke()
        if not smoke:
            raise ValueError("V1.2 synthetic shortlist smoke failed.")
        return V12VerificationResult(
            verified=True,
            package=manifest["package"],
            manifest_sha256=sha256_file(
                self.package_root / "manifest.json"
            ),
            shortlist_policy_sha256=sha256_file(
                self.package_root / "shortlist_policy.json"
            ),
            synthetic_smoke=True,
            ranking_weights_changed=False,
            automatic_target_confirmation=False,
        )

