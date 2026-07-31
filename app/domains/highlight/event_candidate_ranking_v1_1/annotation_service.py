from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .contract import sha256_file
from .evaluation import ApprovedAnnotation, evaluate_event, finalize_reviews
from .schema import (
    EventAnnotationFinalizeRequest,
    EventAnnotationReviewRequest,
)


class EventAnnotationV11Service:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)

    def _owned_ranking(self, artifact_id: str, user: User) -> Artifact:
        artifact = self.db.get(Artifact, artifact_id)
        if (
            artifact is None
            or artifact.artifact_type
            != "EVENT_CANDIDATE_RANKING_V1_1_SHADOW"
        ):
            raise ValueError("V1.1 ranking artifact was not found.")
        metadata = artifact.metadata_ or {}
        if (
            metadata.get("owner_id") != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("V1.1 ranking artifact is not accessible.")
        return artifact

    def _read_json_artifact(
        self,
        artifact: Artifact,
        *,
        expected_types: set[str],
    ) -> dict[str, Any]:
        if artifact.artifact_type not in expected_types:
            raise ValueError("Annotation artifact type is invalid.")
        path = self.storage.resolve_path(artifact.file_path)
        expected = (artifact.metadata_ or {}).get("sha256")
        if not expected or sha256_file(path) != expected:
            raise ValueError("Annotation artifact SHA-256 mismatch.")
        return json.loads(path.read_text(encoding="utf-8"))

    def submit_review(
        self,
        *,
        ranking_artifact_id: str,
        user: User,
        payload: EventAnnotationReviewRequest,
    ) -> Artifact:
        ranking_artifact = self._owned_ranking(ranking_artifact_id, user)
        ranking = self._read_json_artifact(
            ranking_artifact,
            expected_types={"EVENT_CANDIDATE_RANKING_V1_1_SHADOW"},
        )
        candidate_ids = {
            row["candidate_id"] for row in ranking["all_candidates"]
        }
        referenced = set(payload.primary_actor_candidate_ids) | set(
            payload.directly_related_candidate_ids
        ) | set(payload.candidate_roles)
        if not referenced.issubset(candidate_ids):
            raise ValueError("Annotation review references unknown candidates.")
        review_id = generate_prefixed_id("ecrv11review")
        reviewed_at = datetime.now(timezone.utc).isoformat()
        document = {
            "schema_version": "kickclip.event_actor_annotation_review.v1_1",
            "review_id": review_id,
            "ranking_id": (ranking_artifact.metadata_ or {})["ranking_id"],
            "ranking_artifact_id": ranking_artifact_id,
            "annotation_status": "IN_REVIEW",
            "reviewer": user.user_id,
            "reviewed_at": reviewed_at,
            **payload.model_dump(),
        }
        root = self.storage.resolve_path(ranking_artifact.file_path).parent
        output_root = (root / "annotations").resolve()
        if not output_root.is_relative_to(root):
            raise ValueError("Annotation path escapes ranking root.")
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / f"{review_id}.json"
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=ranking_artifact.match_id,
            project_id=ranking_artifact.project_id,
            analysis_job_id=None,
            artifact_type="EVENT_ACTOR_ANNOTATION_V1_1_REVIEW",
            file_path=path.relative_to(self.storage.project_root).as_posix(),
            mime_type="application/json",
            metadata_={
                "sha256": sha256_file(path),
                "ranking_id": document["ranking_id"],
                "ranking_artifact_id": ranking_artifact_id,
                "reviewer": user.user_id,
                "annotation_status": "IN_REVIEW",
            },
        )
        self.db.commit()
        return artifact

    def finalize(
        self,
        *,
        ranking_artifact_id: str,
        user: User,
        payload: EventAnnotationFinalizeRequest,
    ) -> Artifact:
        ranking_artifact = self._owned_ranking(ranking_artifact_id, user)
        ranking = self._read_json_artifact(
            ranking_artifact,
            expected_types={"EVENT_CANDIDATE_RANKING_V1_1_SHADOW"},
        )
        reviews: list[dict[str, Any]] = []
        for artifact_id in payload.review_artifact_ids:
            artifact = self.db.get(Artifact, artifact_id)
            if artifact is None:
                raise ValueError("Annotation review artifact was not found.")
            document = self._read_json_artifact(
                artifact,
                expected_types={"EVENT_ACTOR_ANNOTATION_V1_1_REVIEW"},
            )
            if document["ranking_artifact_id"] != ranking_artifact_id:
                raise ValueError("Annotation reviews must target one ranking.")
            reviews.append(document)
        annotation = finalize_reviews(
            reviews,
            approval_mode=payload.approval_mode,
            approved_reviewer=payload.approved_reviewer,
        )
        annotation.validate(
            {row["candidate_id"] for row in ranking["all_candidates"]}
        )
        selected_review = next(
            (
                review
                for review in reviews
                if payload.approval_mode == "FINAL_APPROVED"
                and review["reviewer"] == payload.approved_reviewer
            ),
            reviews[0],
        )
        document = {
            "schema_version": "kickclip.event_actor_annotation.v1_1",
            "ranking_id": (ranking_artifact.metadata_ or {})["ranking_id"],
            "ranking_artifact_id": ranking_artifact_id,
            **annotation.__dict__,
            "candidate_roles": selected_review.get("candidate_roles") or {},
            "source_review_artifact_ids": payload.review_artifact_ids,
            "union_of_reviewer_labels": False,
        }
        root = self.storage.resolve_path(ranking_artifact.file_path).parent
        output_root = (root / "annotations").resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / (
            f"final_{generate_prefixed_id('ecrv11annotation')}.json"
        )
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=ranking_artifact.match_id,
            project_id=ranking_artifact.project_id,
            analysis_job_id=None,
            artifact_type="EVENT_ACTOR_ANNOTATION_V1_1_FINAL",
            file_path=path.relative_to(self.storage.project_root).as_posix(),
            mime_type="application/json",
            metadata_={
                "sha256": sha256_file(path),
                "ranking_id": document["ranking_id"],
                "ranking_artifact_id": ranking_artifact_id,
                "annotation_status": "COMPLETE",
                "approval_mode": payload.approval_mode,
            },
        )
        self.db.commit()
        return artifact

    def evaluate(
        self,
        *,
        ranking_artifact_id: str,
        annotation_artifact_id: str,
        user: User,
    ) -> dict[str, Any]:
        ranking_artifact = self._owned_ranking(ranking_artifact_id, user)
        ranking = self._read_json_artifact(
            ranking_artifact,
            expected_types={"EVENT_CANDIDATE_RANKING_V1_1_SHADOW"},
        )
        annotation_artifact = self.db.get(Artifact, annotation_artifact_id)
        if annotation_artifact is None:
            raise ValueError("Final annotation artifact was not found.")
        document = self._read_json_artifact(
            annotation_artifact,
            expected_types={"EVENT_ACTOR_ANNOTATION_V1_1_FINAL"},
        )
        if document["ranking_artifact_id"] != ranking_artifact_id:
            raise ValueError("Final annotation belongs to another ranking.")
        annotation = ApprovedAnnotation(
            annotation_status=document["annotation_status"],
            approval_mode=document["approval_mode"],
            primary_actor_visible=document["primary_actor_visible"],
            primary_actor_in_candidate_set=document[
                "primary_actor_in_candidate_set"
            ],
            primary_actor_candidate_ids=tuple(
                document["primary_actor_candidate_ids"]
            ),
            directly_related_candidate_ids=tuple(
                document["directly_related_candidate_ids"]
            ),
            actor_missing_reason=document.get("actor_missing_reason"),
            reviewer=document["reviewer"],
            reviewed_at=document["reviewed_at"],
        )
        return evaluate_event(
            ranked_candidate_ids=[
                row["candidate_id"] for row in ranking["all_candidates"]
            ],
            shortlist_candidate_ids=[
                row["candidate_id"] for row in ranking["shortlist"]
            ],
            annotation=annotation,
            candidate_roles=document.get("candidate_roles"),
        )
