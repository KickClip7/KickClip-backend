from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from sqlalchemy.orm import Session

from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1.backend_adapter import (
    EventCandidateRankingV11BackendAdapter,
)
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    ImmutableCandidateInput,
    sha256_file,
)
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .contract import (
    ResolvedEventContext,
    extract_manifest_candidate_sha,
    resolve_event_context_v112,
    validate_scene_video_duration,
    verify_candidate_artifact_immutability,
)
from .service import EventCandidateRankingV112Engine


def cache_fingerprint_v112(material: dict[str, Any]) -> str:
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
        "shot_boundaries_sha256",
        "source_video_sha256",
        "event_id",
        "event_time_sec",
        "event_label",
    )
    missing = [key for key in required if key not in material]
    if missing:
        raise ValueError(
            f"V1.1.2 cache material is incomplete: {', '.join(missing)}"
        )
    canonical = json.dumps(
        {key: material[key] for key in required},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EventCandidateRankingV112BackendAdapter:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.v11_adapter = EventCandidateRankingV11BackendAdapter(db)
        self.package_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1_2"
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
    ) -> tuple[ResolvedEventContext, dict[str, Any]]:
        resolved = resolve_event_context_v112(
            self.db,
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        revision = self.db.get(HighlightRevision, revision_id)
        discovery = (revision.options or {}).get("scene_target_selection") or {}
        if discovery.get("scene_id") != scene_id:
            raise ValueError("Scene discovery metadata does not match the scene.")
        artifact_root = self.storage.resolve_path(discovery["artifact_root"])
        stored_candidate_relative = discovery.get(
            "scene_candidates_relative_path"
        )
        stored_candidate_sha = discovery.get("scene_candidates_sha256")
        if not stored_candidate_relative or not stored_candidate_sha:
            raise ValueError(
                "Discovery-time scene candidate SHA metadata is missing."
            )
        candidate_path = self.storage.resolve_path(
            stored_candidate_relative
        )
        if (
            not candidate_path.is_relative_to(artifact_root)
            or candidate_path.name != "scene_candidates.json"
            or not candidate_path.is_file()
        ):
            raise ValueError("Stored scene candidate path is invalid.")
        current_candidate_sha = sha256_file(candidate_path)
        manifest_path = (artifact_root / "scene_candidate_manifest.json").resolve()
        if (
            not manifest_path.is_relative_to(artifact_root)
            or not manifest_path.is_file()
        ):
            raise ValueError("Scene candidate manifest is missing.")
        current_manifest_sha = sha256_file(manifest_path)
        if (
            current_manifest_sha
            != discovery.get("scene_candidate_manifest_sha256")
        ):
            raise ValueError("Scene candidate manifest changed after discovery.")
        declared_candidate_sha = extract_manifest_candidate_sha(
            self._load_json(manifest_path)
        )
        verify_candidate_artifact_immutability(
            current_sha256=current_candidate_sha,
            stored_discovery_sha256=stored_candidate_sha,
            manifest_declared_sha256=declared_candidate_sha,
        )
        detections = self.v11_adapter._artifact(
            discovery["detections_artifact_id"], project
        )
        boundaries = self.v11_adapter._artifact(
            discovery["shot_boundaries_artifact_id"], project
        )
        video = self.v11_adapter._media(
            discovery["scene_video_asset_id"], project
        )
        detections_path = self.storage.resolve_path(detections.file_path)
        boundaries_path = self.storage.resolve_path(boundaries.file_path)
        video_path = self.storage.resolve_path(video.file_path)
        width, height, fps, frame_count = self.v11_adapter._probe_video(
            video_path, video
        )
        inputs = discovery.get("discovery_inputs") or {}
        detections_sha = sha256_file(detections_path)
        boundaries_sha = sha256_file(boundaries_path)
        video_sha = video.sha256 or sha256_file(video_path)
        for key, actual in {
            "detections_sha256": detections_sha,
            "shot_boundaries_sha256": boundaries_sha,
            "video_sha256": video_sha,
        }.items():
            if inputs.get(key) != actual:
                raise ValueError(f"Immutable discovery input changed: {key}.")
        contract_policy = self._load_json(
            self.package_root / "contract_policy.json"
        )
        duration_contract = validate_scene_video_duration(
            resolved,
            video_fps=fps,
            video_frame_count=frame_count,
            tolerance_sec=float(
                contract_policy["scene_video_duration_tolerance_sec"]
            ),
        )
        input_document = {
            "event_id": resolved.event_id,
            "scene_id": resolved.scene_id,
            "revision_id": resolved.revision_id,
            "project_id": resolved.project_id,
            "match_id": resolved.match_id,
            "event_label": resolved.event_label,
            "event_time_sec": resolved.event_time_sec,
            "scene_start_sec": resolved.scene_start_sec,
            "scene_end_sec": resolved.scene_end_sec,
            "scene_candidates_sha256": current_candidate_sha,
            "stored_discovery_scene_candidates_sha256": stored_candidate_sha,
            "manifest_declared_scene_candidates_sha256": declared_candidate_sha,
            "source_video_sha256": video_sha,
            "shot_boundaries_sha256": boundaries_sha,
        }
        Draft202012Validator(
            self._load_json(self.package_root / "input_schema.json")
        ).validate(input_document)
        material = {
            "storage_root": self.storage.storage_root.as_posix(),
            "scene_candidates_relative_path": candidate_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "scene_candidates_sha256": current_candidate_sha,
            "detections_relative_path": detections_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "detections_sha256": detections_sha,
            "source_video_relative_path": video_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "source_video_sha256": video_sha,
            "shot_boundaries_relative_path": boundaries_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "shot_boundaries_sha256": boundaries_sha,
            "candidate_manifest_sha256": current_manifest_sha,
            "video_width": width,
            "video_height": height,
            "video_fps": fps,
            "video_frame_count": frame_count,
            "event_id": resolved.event_id,
            "event_label": resolved.event_label,
            "event_time_sec": resolved.event_time_sec,
            "current_scene_candidates_sha256": current_candidate_sha,
            "stored_discovery_scene_candidates_sha256": stored_candidate_sha,
            "manifest_declared_scene_candidates_sha256": declared_candidate_sha,
            "scene_candidate_manifest_sha256": current_manifest_sha,
            "ranking_source_manifest_sha256": sha256_file(
                self.package_root / "manifest.json"
            ),
            "ranking_policy_sha256": sha256_file(
                self.v11_root / "event_candidate_ranking_policy.json"
            ),
            "safety_policy_sha256": sha256_file(
                self.v111_root / "safety_policy.json"
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
            "duration_contract": duration_contract,
            "input_document": input_document,
        }
        material["cache_fingerprint"] = cache_fingerprint_v112(material)
        revision.options = {
            **(revision.options or {}),
            "scene_target_selection": {
                **discovery,
                "event_candidate_ranking_v1_1_2_input": {
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
        document = EventCandidateRankingV112Engine(
            self.package_root
        ).run(
            immutable_input=immutable,
            resolved_event=ResolvedEventContext(**resolved_event),
            shortlist_size=shortlist_size,
            immutable_sha_contract=immutable_sha_contract,
            scene_video_duration_contract=freeze_material[
                "duration_contract"
            ],
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
        ranking_id = generate_prefixed_id("ecrankv112")
        revision = self.db.get(HighlightRevision, revision_id)
        discovery_root = self.storage.resolve_path(
            (revision.options or {})["scene_target_selection"][
                "artifact_root"
            ]
        )
        output_root = (
            discovery_root / "rankings" / f"v1_1_2_{ranking_id}"
        ).resolve()
        if not output_root.is_relative_to(discovery_root):
            raise ValueError("V1.1.2 output escapes discovery root.")
        output_root.mkdir(parents=True, exist_ok=False)
        output_path = output_root / "event_candidate_ranking_v1_1_2.json"
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=None,
            artifact_type="EVENT_CANDIDATE_RANKING_V1_1_2_SHADOW",
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
