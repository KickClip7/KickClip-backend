from __future__ import annotations

from typing import Any

from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1_2.annotation_service import (
    EventAnnotationCompatibilityService,
    RANKING_ARTIFACT_ALLOWLIST,
)


HOTFIX_RANKING_ARTIFACT_ALLOWLIST = {
    **RANKING_ARTIFACT_ALLOWLIST,
    "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW": (
        "target_centric_tracking_event_candidate_ranking_v1_1_2a",
        "kickclip.event_candidate_ranking.v1_1_2a",
    ),
}


class EventAnnotationCompatibilityV112aService(
    EventAnnotationCompatibilityService
):
    def _read_json_artifact(
        self,
        artifact: Artifact,
        *,
        expected_types: set[str],
    ) -> dict[str, Any]:
        if expected_types in (
            {"EVENT_CANDIDATE_RANKING_V1_1_SHADOW"},
            set(RANKING_ARTIFACT_ALLOWLIST),
        ):
            expected_types = set(HOTFIX_RANKING_ARTIFACT_ALLOWLIST)
        return super()._read_json_artifact(
            artifact,
            expected_types=expected_types,
        )

    def _owned_ranking(self, artifact_id: str, user: User) -> Artifact:
        artifact = self.db.get(Artifact, artifact_id)
        if (
            artifact is None
            or artifact.artifact_type
            not in HOTFIX_RANKING_ARTIFACT_ALLOWLIST
        ):
            raise ValueError("Compatible event ranking artifact was not found.")
        metadata = artifact.metadata_ or {}
        if (
            metadata.get("owner_id") != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Event ranking artifact is not accessible.")
        self._ranking_identity(artifact)
        return artifact

    def _ranking_identity(self, artifact: Artifact) -> dict[str, str]:
        document = self._read_json_artifact(
            artifact,
            expected_types=set(HOTFIX_RANKING_ARTIFACT_ALLOWLIST),
        )
        expected_package, expected_schema = (
            HOTFIX_RANKING_ARTIFACT_ALLOWLIST[artifact.artifact_type]
        )
        if (
            document.get("package") != expected_package
            or document.get("schema_version") != expected_schema
        ):
            raise ValueError("Ranking artifact package/schema mismatch.")
        freeze = document.get("freeze") or {}
        hashes = {
            "ranking_source_manifest_sha256": freeze.get(
                "source_manifest_sha256"
            ),
            "ranking_feature_schema_sha256": (
                freeze.get("candidate_feature_schema_sha256")
                or freeze.get("feature_schema_sha256")
            ),
            "ranking_policy_sha256": (
                freeze.get("ranking_policy_sha256")
                or freeze.get("policy_sha256")
            ),
        }
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in hashes.values()
        ):
            raise ValueError(
                "Ranking artifact manifest/schema/policy SHA is missing."
            )
        return {
            "ranking_package": expected_package,
            "ranking_schema_version": expected_schema,
            **hashes,
        }
