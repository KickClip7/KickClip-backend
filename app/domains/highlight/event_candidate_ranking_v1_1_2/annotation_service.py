from __future__ import annotations

import json
from typing import Any

from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1.annotation_service import (
    EventAnnotationV11Service,
)
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1.schema import (
    EventAnnotationFinalizeRequest,
)


RANKING_ARTIFACT_ALLOWLIST = {
    "EVENT_CANDIDATE_RANKING_V1_1_SHADOW": (
        "target_centric_tracking_event_candidate_ranking_v1_1",
        "kickclip.event_candidate_ranking.v1_1",
    ),
    "EVENT_CANDIDATE_RANKING_V1_1_1_SHADOW": (
        "target_centric_tracking_event_candidate_ranking_v1_1_1",
        "kickclip.event_candidate_ranking.v1_1_1",
    ),
    "EVENT_CANDIDATE_RANKING_V1_1_2_SHADOW": (
        "target_centric_tracking_event_candidate_ranking_v1_1_2",
        "kickclip.event_candidate_ranking.v1_1_2",
    ),
}


class EventAnnotationCompatibilityService(EventAnnotationV11Service):
    """Version-aware annotation bridge without changing frozen V1.1 code."""

    def _read_json_artifact(
        self,
        artifact: Artifact,
        *,
        expected_types: set[str],
    ) -> dict[str, Any]:
        # The frozen V1.1 implementation passes its single ranking type from
        # submit/finalize/evaluate. Widen only that exact ranking read; review
        # and final-annotation artifact checks remain unchanged.
        if expected_types == {"EVENT_CANDIDATE_RANKING_V1_1_SHADOW"}:
            expected_types = set(RANKING_ARTIFACT_ALLOWLIST)
        return super()._read_json_artifact(
            artifact,
            expected_types=expected_types,
        )

    def _owned_ranking(self, artifact_id: str, user: User) -> Artifact:
        artifact = self.db.get(Artifact, artifact_id)
        if artifact is None or artifact.artifact_type not in RANKING_ARTIFACT_ALLOWLIST:
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
            expected_types=set(RANKING_ARTIFACT_ALLOWLIST),
        )
        expected_package, expected_schema = RANKING_ARTIFACT_ALLOWLIST[
            artifact.artifact_type
        ]
        if (
            document.get("package") != expected_package
            or document.get("schema_version") != expected_schema
        ):
            raise ValueError("Ranking artifact package/schema mismatch.")
        freeze = document.get("freeze") or {}
        policy_sha = (
            freeze.get("ranking_policy_sha256")
            or freeze.get("policy_sha256")
        )
        manifest_sha = freeze.get("source_manifest_sha256")
        schema_sha = (
            freeze.get("candidate_feature_schema_sha256")
            or freeze.get("feature_schema_sha256")
        )
        hashes = {
            "ranking_source_manifest_sha256": manifest_sha,
            "ranking_feature_schema_sha256": schema_sha,
            "ranking_policy_sha256": policy_sha,
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

    def _stamp_annotation(
        self,
        artifact: Artifact,
        *,
        identity: dict[str, str],
    ) -> None:
        path = self.storage.resolve_path(artifact.file_path)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["ranking_identity"] = identity
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        metadata = dict(artifact.metadata_ or {})
        metadata.update(identity)
        metadata["sha256"] = sha256_file(path)
        artifact.metadata_ = metadata
        self.db.commit()

    def submit_review(self, *, ranking_artifact_id: str, user: User, payload):
        ranking = self._owned_ranking(ranking_artifact_id, user)
        identity = self._ranking_identity(ranking)
        artifact = super().submit_review(
            ranking_artifact_id=ranking_artifact_id,
            user=user,
            payload=payload,
        )
        self._stamp_annotation(artifact, identity=identity)
        return artifact

    def finalize(
        self,
        *,
        ranking_artifact_id: str,
        user: User,
        payload: EventAnnotationFinalizeRequest,
    ) -> Artifact:
        ranking = self._owned_ranking(ranking_artifact_id, user)
        identity = self._ranking_identity(ranking)
        for artifact_id in payload.review_artifact_ids:
            review = self.db.get(Artifact, artifact_id)
            if review is None:
                raise ValueError("Annotation review artifact was not found.")
            document = self._read_json_artifact(
                review,
                expected_types={"EVENT_ACTOR_ANNOTATION_V1_1_REVIEW"},
            )
            if document.get("ranking_identity") != identity:
                raise ValueError(
                    "Annotation review ranking version does not match."
                )
        artifact = super().finalize(
            ranking_artifact_id=ranking_artifact_id,
            user=user,
            payload=payload,
        )
        self._stamp_annotation(artifact, identity=identity)
        return artifact

    def evaluate(
        self,
        *,
        ranking_artifact_id: str,
        annotation_artifact_id: str,
        user: User,
    ) -> dict[str, Any]:
        ranking = self._owned_ranking(ranking_artifact_id, user)
        identity = self._ranking_identity(ranking)
        annotation = self.db.get(Artifact, annotation_artifact_id)
        if annotation is None:
            raise ValueError("Final annotation artifact was not found.")
        document = self._read_json_artifact(
            annotation,
            expected_types={"EVENT_ACTOR_ANNOTATION_V1_1_FINAL"},
        )
        if document.get("ranking_identity") != identity:
            raise ValueError(
                "Evaluation cannot mix ranking package/schema/policy."
            )
        result = super().evaluate(
            ranking_artifact_id=ranking_artifact_id,
            annotation_artifact_id=annotation_artifact_id,
            user=user,
        )
        return {
            **result,
            "ranking_identity": identity,
            "cross_version_metrics_mixed": False,
        }
