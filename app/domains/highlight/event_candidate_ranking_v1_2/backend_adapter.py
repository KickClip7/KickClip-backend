from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.project.model import Project
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .contract import canonical_sha256, sha256_file
from .service import EventCandidateRankingV12ShortlistPatch


ARTIFACT_TYPE = "EVENT_CANDIDATE_RANKING_V1_2_SHADOW_SHORTLIST_PATCH"


class EventCandidateRankingV12BackendAdapter:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.package_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_2"
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object artifact.")
        return value

    def _owned_source(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        project: Project,
        user: User,
    ) -> Artifact:
        artifact = self.db.get(Artifact, artifact_id)
        if (
            artifact is None
            or artifact.artifact_type != artifact_type
            or artifact.project_id != project.project_id
            or artifact.match_id != project.match_id
        ):
            raise ValueError(f"Required artifact is missing: {artifact_type}")
        metadata = artifact.metadata_ or {}
        if (
            artifact_type
            == "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW"
            and metadata.get("owner_id") != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Source ranking artifact is not accessible.")
        path = self.storage.resolve_path(artifact.file_path)
        expected = str(metadata.get("sha256") or "")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError("Source artifact SHA-256 mismatch.")
        return artifact

    def run(
        self,
        *,
        project: Project,
        user: User,
        source_ranking_artifact_id: str,
        reviewed_shots_artifact_id: str,
        shortlist_size: int = 5,
    ) -> Artifact:
        source = self._owned_source(
            artifact_id=source_ranking_artifact_id,
            artifact_type="EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW",
            project=project,
            user=user,
        )
        shots = self._owned_source(
            artifact_id=reviewed_shots_artifact_id,
            artifact_type="REVIEWED_SHOT_BOUNDARIES",
            project=project,
            user=user,
        )
        source_path = self.storage.resolve_path(source.file_path)
        shots_path = self.storage.resolve_path(shots.file_path)
        source_sha = sha256_file(source_path)
        shots_sha = sha256_file(shots_path)
        cache_key = canonical_sha256(
            {
                "source_ranking_artifact_id": source.artifact_id,
                "source_ranking_sha256": source_sha,
                "reviewed_shots_artifact_id": shots.artifact_id,
                "reviewed_shots_sha256": shots_sha,
                "shortlist_size": shortlist_size,
                "source_manifest_sha256": sha256_file(
                    self.package_root / "manifest.json"
                ),
                "shortlist_policy_sha256": sha256_file(
                    self.package_root / "shortlist_policy.json"
                ),
            }
        )
        existing = self.db.scalar(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == ARTIFACT_TYPE,
            )
        )
        if existing is not None:
            if (existing.metadata_ or {}).get("cache_key") != cache_key:
                raise ValueError("Existing V1.2 patch has different inputs.")
            return existing

        source_document = self._load_json(source_path)
        shots_document = self._load_json(shots_path)
        input_document = {
            "source_ranking_artifact_id": source.artifact_id,
            "source_ranking_sha256": source_sha,
            "source_ranking": source_document,
            "reviewed_shots": shots_document,
            "shortlist_size": shortlist_size,
        }
        Draft202012Validator(
            self._load_json(self.package_root / "input_schema.json")
        ).validate(input_document)
        output = EventCandidateRankingV12ShortlistPatch(
            self.package_root
        ).run(
            source_ranking=source_document,
            reviewed_shots=shots_document,
            source_ranking_artifact_id=source.artifact_id,
            source_ranking_sha256=source_sha,
            shortlist_size=shortlist_size,
        )
        Draft202012Validator(
            self._load_json(self.package_root / "output_schema.json")
        ).validate(output)
        ranking_id = generate_prefixed_id("ecrankv12")
        output_root = (
            source_path.parent
            / "v1_2_shortlist_patches"
            / ranking_id
        ).resolve()
        if not output_root.is_relative_to(source_path.parent):
            raise ValueError("V1.2 output escapes the source ranking root.")
        output_root.mkdir(parents=True, exist_ok=False)
        output_path = output_root / "event_candidate_ranking_v1_2.json"
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=None,
            artifact_type=ARTIFACT_TYPE,
            file_path=output_path.relative_to(
                self.storage.project_root
            ).as_posix(),
            mime_type="application/json",
            metadata_={
                "ranking_id": ranking_id,
                "revision_id": (source.metadata_ or {}).get("revision_id"),
                "scene_id": (source.metadata_ or {}).get("scene_id"),
                "event_id": (source.metadata_ or {}).get("event_id"),
                "source_ranking_artifact_id": source.artifact_id,
                "source_ranking_sha256": source_sha,
                "reviewed_shots_artifact_id": shots.artifact_id,
                "reviewed_shots_sha256": shots_sha,
                "cache_key": cache_key,
                "sha256": sha256_file(output_path),
                "status": output["ranking_status"],
                "automatic_target_confirmation": False,
                "owner_id": user.user_id,
            },
        )
        self.db.commit()
        return artifact

