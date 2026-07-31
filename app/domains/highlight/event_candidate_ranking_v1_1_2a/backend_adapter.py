from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from sqlalchemy.orm import Session

from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    ImmutableCandidateInput,
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.contract import (
    ResolvedEventContext,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.backend_adapter import (
    EventCandidateRankingV112BackendAdapter,
)
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .service import EventCandidateRankingV112aEngine


class StaleDiscoveryInputError(ValueError):
    code = "STALE_DISCOVERY_INPUT"

    def __init__(self, detail: str) -> None:
        super().__init__(f"{self.code}: {detail}")


def cache_fingerprint_v112a(material: dict[str, Any]) -> str:
    required = (
        "ranking_source_manifest_sha256",
        "ranking_policy_sha256",
        "safety_policy_sha256",
        "contract_policy_sha256",
        "input_schema_sha256",
        "candidate_feature_schema_sha256",
        "ranking_output_schema_sha256",
        "current_scene_candidates_sha256",
        "stored_discovery_scene_candidates_sha256",
        "manifest_declared_scene_candidates_sha256",
        "scene_candidate_manifest_sha256",
        "detections_sha256",
        "shot_boundaries_sha256",
        "source_video_sha256",
        "discovery_id",
        "scene_id",
        "scene_start_sec",
        "scene_end_sec",
        "shortlist_size",
        "event_id",
        "event_time_sec",
        "event_label",
    )
    missing = [key for key in required if key not in material]
    if missing:
        raise ValueError(
            "V1.1.2a cache material is incomplete: "
            + ", ".join(missing)
        )
    canonical = json.dumps(
        {key: material[key] for key in required},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Compatibility name for callers migrating from the V1.1.2 adapter. The
# implementation lives only in the additive V1.1.2a package.
cache_fingerprint_v112 = cache_fingerprint_v112a


def resolve_frozen_discovery_root(
    storage: LocalStorage,
    *,
    revision: HighlightRevision,
    freeze_material: dict[str, Any],
) -> Path:
    current = (revision.options or {}).get("scene_target_selection") or {}
    if current.get("discovery_id") != freeze_material.get("discovery_id"):
        raise StaleDiscoveryInputError("current discovery_id changed")
    if current.get("scene_id") != freeze_material.get("scene_id"):
        raise StaleDiscoveryInputError("current discovery scene changed")
    frozen_root = storage.resolve_path(
        freeze_material["discovery_artifact_root"]
    )
    current_root = storage.resolve_path(current.get("artifact_root") or "")
    if current_root != frozen_root:
        raise StaleDiscoveryInputError("current discovery root changed")
    if not frozen_root.is_dir():
        raise StaleDiscoveryInputError("frozen discovery root is missing")
    manifest_path = (
        frozen_root / "scene_candidate_manifest.json"
    ).resolve()
    if (
        not manifest_path.is_relative_to(frozen_root)
        or not manifest_path.is_file()
        or sha256_file(manifest_path)
        != freeze_material.get("scene_candidate_manifest_sha256")
    ):
        raise StaleDiscoveryInputError(
            "frozen candidate manifest is missing or changed"
        )
    return frozen_root


class EventCandidateRankingV112aBackendAdapter:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.v112 = EventCandidateRankingV112BackendAdapter(db)
        self.package_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_2a"
        )
        self.v11_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1"
        )
        self.v111_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_1"
        )
        self.v112_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_2"
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def prepare(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
        shortlist_size: int,
    ) -> tuple[ResolvedEventContext, dict[str, Any]]:
        resolved, material = self.v112.prepare(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        revision = self.db.get(HighlightRevision, revision_id)
        if revision is None:
            raise ValueError("Highlight revision was not found.")
        discovery = (revision.options or {}).get(
            "scene_target_selection"
        ) or {}
        discovery_id = str(discovery.get("discovery_id") or "")
        if not discovery_id:
            raise ValueError("Discovery snapshot has no discovery_id.")
        artifact_root = self.storage.resolve_path(
            discovery["artifact_root"]
        )
        if not artifact_root.is_dir():
            raise ValueError("Discovery snapshot root is missing.")
        material.update(
            {
                "discovery_id": discovery_id,
                "discovery_artifact_root": artifact_root.relative_to(
                    self.storage.project_root
                ).as_posix(),
                "scene_id": scene_id,
                "scene_start_sec": resolved.scene_start_sec,
                "scene_end_sec": resolved.scene_end_sec,
                "shortlist_size": shortlist_size,
                "ranking_source_manifest_sha256": sha256_file(
                    self.package_root / "manifest.json"
                ),
                "contract_policy_sha256": sha256_file(
                    self.package_root / "contract_policy.json"
                ),
                "input_schema_sha256": sha256_file(
                    self.package_root / "input_schema.json"
                ),
                "candidate_feature_schema_sha256": sha256_file(
                    self.package_root / "candidate_feature_schema.json"
                ),
                "ranking_output_schema_sha256": sha256_file(
                    self.package_root / "ranking_output_schema.json"
                ),
                "v1_1_2_manifest_sha256": sha256_file(
                    self.v112_root / "manifest.json"
                ),
            }
        )
        input_document = {
            **material["input_document"],
            "detections_sha256": material["detections_sha256"],
            "scene_candidate_manifest_sha256": material[
                "scene_candidate_manifest_sha256"
            ],
            "discovery_id": discovery_id,
            "discovery_artifact_root": material[
                "discovery_artifact_root"
            ],
            "shortlist_size": shortlist_size,
        }
        Draft202012Validator(
            self._load_json(self.package_root / "input_schema.json")
        ).validate(input_document)
        material["input_document"] = input_document
        material["cache_fingerprint"] = cache_fingerprint_v112a(material)
        revision.options = {
            **(revision.options or {}),
            "scene_target_selection": {
                **discovery,
                "event_candidate_ranking_v1_1_2a_input": {
                    key: value
                    for key, value in material.items()
                    if key != "storage_root"
                },
            },
        }
        self.db.flush()
        return resolved, material

    def run(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        shortlist_size: int,
        resolved_event: dict[str, Any],
        freeze_material: dict[str, Any],
    ) -> dict[str, Any]:
        if shortlist_size != freeze_material.get("shortlist_size"):
            raise ValueError("Frozen shortlist size changed.")
        revision = self.db.get(HighlightRevision, revision_id)
        if revision is None:
            raise ValueError("Highlight revision was not found.")
        frozen_root = resolve_frozen_discovery_root(
            self.storage,
            revision=revision,
            freeze_material=freeze_material,
        )
        immutable = ImmutableCandidateInput.load(
            discovery_root=Path(freeze_material["storage_root"]),
            **{
                key: freeze_material[key]
                for key in (
                    "scene_candidates_relative_path",
                    "scene_candidates_sha256",
                    "detections_relative_path",
                    "detections_sha256",
                    "source_video_relative_path",
                    "source_video_sha256",
                    "shot_boundaries_relative_path",
                    "shot_boundaries_sha256",
                    "candidate_manifest_sha256",
                    "video_width",
                    "video_height",
                    "video_fps",
                    "video_frame_count",
                )
            },
        )
        immutable_sha_contract = {
            "current_scene_candidates_sha256": freeze_material[
                "current_scene_candidates_sha256"
            ],
            "stored_discovery_scene_candidates_sha256": freeze_material[
                "stored_discovery_scene_candidates_sha256"
            ],
            "manifest_declared_scene_candidates_sha256": freeze_material[
                "manifest_declared_scene_candidates_sha256"
            ],
        }
        snapshot = {
            key: str(freeze_material[key])
            for key in (
                "discovery_id",
                "discovery_artifact_root",
                "scene_id",
                "scene_candidate_manifest_sha256",
            )
        }
        document = EventCandidateRankingV112aEngine(
            self.package_root
        ).run(
            immutable_input=immutable,
            resolved_event=ResolvedEventContext(**resolved_event),
            shortlist_size=shortlist_size,
            immutable_sha_contract=immutable_sha_contract,
            scene_video_duration_contract=freeze_material[
                "duration_contract"
            ],
            discovery_snapshot=snapshot,
        )
        candidate_validator = Draft202012Validator(
            self._load_json(
                self.package_root / "candidate_feature_schema.json"
            )
        )
        for candidate in document["all_candidates"]:
            candidate_validator.validate(candidate)
        Draft202012Validator(
            self._load_json(
                self.package_root / "ranking_output_schema.json"
            )
        ).validate(document)
        ranking_id = generate_prefixed_id("ecrankv112a")
        output_root = (
            frozen_root / "rankings" / f"v1_1_2a_{ranking_id}"
        ).resolve()
        if not output_root.is_relative_to(frozen_root):
            raise ValueError("V1.1.2a output escapes frozen discovery root.")
        output_root.mkdir(parents=True, exist_ok=False)
        output_path = (
            output_root / "event_candidate_ranking_v1_1_2a.json"
        )
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=None,
            artifact_type="EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW",
            file_path=output_path.relative_to(
                self.storage.project_root
            ).as_posix(),
            mime_type="application/json",
            metadata_={
                "ranking_id": ranking_id,
                "ranking_package": document["package"],
                "ranking_schema_version": document["schema_version"],
                "ranking_policy_sha256": document["freeze"][
                    "ranking_policy_sha256"
                ],
                "revision_id": revision_id,
                "discovery_id": freeze_material["discovery_id"],
                "scene_id": resolved_event["scene_id"],
                "event_id": resolved_event["event_id"],
                "status": document["ranking_status"],
                "automatic_target_confirmation": False,
                "sha256": sha256_file(output_path),
                "owner_id": user.user_id,
            },
        )
        self.db.commit()
        return {
            "ranking_id": ranking_id,
            "artifact_id": artifact.artifact_id,
            "ranking_status": document["ranking_status"],
            "feature_completeness_state": document[
                "feature_completeness_state"
            ],
            "automatic_target_confirmation": False,
            "shortlist_candidate_ids": [
                row["candidate_id"] for row in document["shortlist"]
            ],
            "full_gallery_fallback_candidate_ids": document[
                "full_gallery_fallback"
            ],
            "output_sha256": sha256_file(output_path),
        }
