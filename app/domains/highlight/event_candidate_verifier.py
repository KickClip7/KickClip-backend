from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.domains.highlight.runtime_contract import sha256_file


@dataclass(frozen=True)
class EventCandidateRankingVerification:
    available: bool
    code: str
    message: str
    policy_sha256: str | None = None
    feature_schema_sha256: str | None = None
    manifest_sha256: str | None = None


class EventCandidateRankingVerifier:
    """Verify the backend-native shadow package and feature blacklist."""

    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root.resolve()

    def check(self) -> EventCandidateRankingVerification:
        policy_path = self.package_root / "event_candidate_ranking_policy.json"
        schema_path = self.package_root / "event_candidate_feature_schema.json"
        manifest_path = self.package_root / "manifest.json"
        if any(
            not path.is_file()
            for path in (policy_path, schema_path, manifest_path)
        ):
            return EventCandidateRankingVerification(
                available=False,
                code="EVENT_RANKING_RUNTIME_MISSING",
                message="Event candidate ranking package files are missing.",
            )
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return EventCandidateRankingVerification(
                available=False,
                code="EVENT_RANKING_RUNTIME_INVALID",
                message="Event candidate ranking package JSON is invalid.",
            )
        expected_blacklist = {
            "candidate_id",
            "jersey_number",
            "hardcoded_frame",
            "hardcoded_shot",
        }
        weights = policy.get("event_relevance_weights")
        if (
            policy.get("policy_id")
            != "target_centric_tracking_event_candidate_ranking_v1"
            or policy.get("status") != "PROVISIONAL_SHADOW_ONLY"
            or policy.get("automatic_target_confirmation") is not False
            or manifest.get("automatic_target_confirmation") is not False
            or not expected_blacklist.issubset(
                set(manifest.get("production_feature_blacklist") or [])
            )
            or not isinstance(weights, dict)
            or expected_blacklist.intersection(weights)
            or schema.get("$id") != "kickclip.event_candidate_features.v1"
        ):
            return EventCandidateRankingVerification(
                available=False,
                code="EVENT_RANKING_CONTRACT_INVALID",
                message="Event ranking policy/schema contract is invalid.",
            )
        return EventCandidateRankingVerification(
            available=True,
            code="EVENT_RANKING_RUNTIME_VERIFIED",
            message="Shadow event candidate ranking package verified.",
            policy_sha256=sha256_file(policy_path),
            feature_schema_sha256=sha256_file(schema_path),
            manifest_sha256=sha256_file(manifest_path),
        )
