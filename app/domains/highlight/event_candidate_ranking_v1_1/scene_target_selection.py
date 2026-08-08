from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.model import (
    HighlightRevision,
    ScenePlayerCandidate,
    SceneTargetSelection,
)
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.runtime_contract import (
    IMAGE_SUFFIXES,
    JSON_SUFFIXES,
    REVIEW_ARTIFACT_SUFFIXES,
    ConfiguredSceneRuntime,
    assert_expected_mime,
    load_runtime_json,
    prepare_selection_workspace,
    project_relative,
    redact_runtime_paths,
    sha256_file,
    validated_runtime_file,
)
from app.domains.highlight.schema import SceneTargetSelectionRead
from app.domains.highlight.event_candidate_ranking_v1_1_2.contract import (
    extract_manifest_candidate_sha,
    verify_candidate_artifact_immutability,
)
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.schema import TrackingJobCreateRequest
from app.domains.tracking.service import TrackingJobService
from app.storage.local_storage import LocalStorage


class SceneTargetSelectionService:
    def __init__(
        self,
        db: Session,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repository = HighlightRepository(db)
        self.artifacts = ArtifactRepository(db)
        self.storage = LocalStorage()
        configured_root = (
            self.settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
            or self.settings.TRACKING_PROJECT_ROOT
        )
        self.runtime_root = (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else (self.storage.project_root / ".tracking-runtime").resolve()
        )
        self.package_root = (
            self.runtime_root
            / "target_centric_tracking_scene_target_selection_v1"
        )
        self.runner = self.package_root / "run_scene_target_selection.py"

    def _revision(
        self,
        *,
        project: Project,
        revision_id: str,
    ) -> HighlightRevision:
        revision = self.repository.get_revision(revision_id)
        if revision is None or revision.project_id != project.project_id:
            raise ValueError("Highlight revision not found.")
        return revision

    @staticmethod
    def _assert_scene(revision: HighlightRevision, scene_id: str) -> None:
        if scene_id not in revision.selected_scene_ids:
            raise ValueError("Scene is not selected in this revision.")

    @staticmethod
    def _assert_artifact_scope(
        artifact: Artifact,
        *,
        project: Project,
    ) -> None:
        if artifact.match_id != project.match_id:
            raise ValueError("Artifact does not belong to this Match.")
        if artifact.project_id not in {None, project.project_id}:
            raise ValueError("Artifact does not belong to this Project.")

    def _artifact_path(self, artifact: Artifact) -> Path:
        path = self.storage.resolve_path(artifact.file_path)
        if not path.is_file():
            raise ValueError("Required server-owned artifact is missing.")
        return path

    @staticmethod
    def _shot_scene_context(
        detections_artifact: Artifact,
    ) -> dict[str, dict[str, Any]]:
        metadata = detections_artifact.metadata_ or {}
        router = metadata.get("shot_scene_router")
        if not isinstance(router, dict):
            return {}
        raw_modes = router.get("shot_modes")
        if not isinstance(raw_modes, dict):
            return {}
        result: dict[str, dict[str, Any]] = {}
        for shot_id, raw in raw_modes.items():
            if not isinstance(raw, dict):
                continue
            mode = str(raw.get("mode") or "MIXED").upper()
            if mode not in {"WIDE", "CLOSEUP", "MIXED"}:
                mode = "MIXED"
            result[str(shot_id)] = {
                "mode": mode,
                "reason": str(raw.get("reason") or ""),
                "evidence_frame_count": int(raw.get("evidence_frame_count") or 0),
                "closeup_vote_ratio": raw.get("closeup_vote_ratio"),
                "role_state": (
                    "ROLE_UNVERIFIED"
                    if mode in {"CLOSEUP", "MIXED"}
                    else "ROLE_SUPPORTED"
                ),
            }
        return result

    def _immutable_selection_file(
        self,
        selection: SceneTargetSelection,
        *,
        path_attribute: str,
        sha_attribute: str,
    ) -> Path:
        relative = getattr(selection, path_attribute)
        expected_sha = getattr(selection, sha_attribute)
        if not relative or not expected_sha:
            raise ValueError(f"Selection is missing {path_attribute}.")
        root = self.storage.resolve_path(selection.selection_artifact_root)
        path = self.storage.resolve_path(relative)
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.suffix.lower() != ".json"
        ):
            raise ValueError("Immutable selection artifact path is invalid.")
        if sha256_file(path) != expected_sha:
            raise ValueError("Immutable selection artifact hash mismatch.")
        return path

    def _output_root(
        self,
        *,
        project: Project,
        revision: HighlightRevision,
        scene_id: str,
        discovery_id: str,
    ) -> Path:
        root = (
            self.storage.storage_root
            / "matches"
            / project.match_id
            / "projects"
            / project.project_id
            / "highlight"
            / revision.revision_id
            / "scene-target-selection"
            / scene_id
            / "discoveries"
            / discovery_id
        ).resolve()
        if not root.is_relative_to(self.storage.storage_root):
            raise ValueError("Scene target selection path escapes storage.")
        return root

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _discovery_fingerprint(
        self,
        *,
        scene_video: MediaAsset,
        video_path: Path,
        shot_boundaries_path: Path,
        detections_path: Path,
    ) -> tuple[str, dict[str, str]]:
        package_manifest = (
            self.package_root
            / "scene_target_selection_frozen_manifest.json"
        )
        policy = self.package_root / "scene_target_selection_policy.json"
        rfdetr = (
            self.runtime_root
            / "weights/rfdetr/checkpoint_best_regular.pth"
        )
        inputs = {
            "video_sha256": (
                scene_video.sha256 or self._sha256(video_path)
            ),
            "shot_boundaries_sha256": self._sha256(
                shot_boundaries_path
            ),
            "detections_sha256": self._sha256(detections_path),
            "rfdetr_checkpoint_sha256": self._sha256(rfdetr),
            "candidate_policy_sha256": self._sha256(policy),
            "candidate_package_manifest_sha256": self._sha256(
                package_manifest
            ),
            "schema_version": "kickclip.scene_player_candidates.v1",
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                inputs, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return f"discovery_{fingerprint[:20]}", inputs

    def _run(self, arguments: list[str]) -> None:
        ConfiguredSceneRuntime(
            self.settings,
            default_script=self.runner,
        ).run(arguments)

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _ensure_manifest_candidate_sha(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
        candidates_path: Path,
    ) -> tuple[dict[str, Any], str, bool]:
        """Keep the strict SHA contract while repairing legacy runtime output.

        Older frozen scene-selection runners emitted a manifest without the
        scene_candidates.json digest.  We do not weaken verification: the
        backend computes the digest from the just-produced immutable file,
        records it atomically in the manifest, and still rejects any declared
        digest that differs from the current file.
        """

        current_sha256 = sha256_file(candidates_path)
        try:
            declared_sha256 = extract_manifest_candidate_sha(manifest)
        except ValueError:
            repaired = dict(manifest)
            repaired["scene_candidates_sha256"] = current_sha256
            files = repaired.get("files")
            files = dict(files) if isinstance(files, dict) else {}
            existing = files.get("scene_candidates.json")
            entry = dict(existing) if isinstance(existing, dict) else {}
            entry.update(
                {
                    "path": "scene_candidates.json",
                    "sha256": current_sha256,
                }
            )
            files["scene_candidates.json"] = entry
            repaired["files"] = files
            repaired["manifest_compatibility_patch"] = (
                "BACKEND_DECLARED_SCENE_CANDIDATES_SHA256_V1"
            )
            self._write_json_atomic(manifest_path, repaired)
            return repaired, current_sha256, True

        if declared_sha256 != current_sha256:
            raise ValueError(
                "Scene candidate manifest SHA-256 differs from "
                "scene_candidates.json."
            )
        return manifest, declared_sha256, False

    def _register_runtime_artifact(
        self,
        *,
        project: Project,
        path: Path,
        artifact_type: str,
        mime_type: str,
        metadata: dict[str, Any],
    ) -> str:
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=None,
            artifact_type=artifact_type,
            file_path=project_relative(path, self.storage.project_root),
            mime_type=mime_type,
            metadata_={
                **metadata,
                "sha256": sha256_file(path),
            },
        )
        return artifact.artifact_id

    def _register_portable_paths(
        self,
        *,
        project: Project,
        output: Path,
        value: Any,
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        artifact_ids: dict[str, str] = {}

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    if isinstance(child, str) and (
                        key.endswith("_artifact")
                        or key in {
                            "contact_sheet",
                            "tracklet_review_video",
                            "review_video",
                            "thumbnail",
                        }
                    ):
                        try:
                            path = validated_runtime_file(
                                output,
                                child,
                                allowed_suffixes=REVIEW_ARTIFACT_SUFFIXES,
                            )
                        except (FileNotFoundError, ValueError):
                            raise ValueError(
                                f"Invalid runtime review artifact: {key}"
                            ) from None
                        prefix = (
                            "image/"
                            if path.suffix.lower() in IMAGE_SUFFIXES
                            else "video/"
                        )
                        assert_expected_mime(path, prefix)
                        portable = Path(child).as_posix()
                        if portable not in artifact_ids:
                            artifact_ids[portable] = (
                                self._register_runtime_artifact(
                                    project=project,
                                    path=path,
                                    artifact_type=(
                                        "SCENE_TARGET_SELECTION_REVIEW"
                                    ),
                                    mime_type=(
                                        "image/jpeg"
                                        if prefix == "image/"
                                        else "video/mp4"
                                    ),
                                    metadata=metadata,
                                )
                            )
                    else:
                        visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
        return artifact_ids

    def discover(
        self,
        *,
        project: Project,
        revision_id: str,
        scene_id: str,
        scene_video: MediaAsset,
        shot_boundaries_artifact: Artifact,
        detections_artifact: Artifact,
    ) -> dict[str, Any]:
        revision = self._revision(project=project, revision_id=revision_id)
        self._assert_scene(revision, scene_id)
        if scene_video.match_id != project.match_id:
            raise ValueError("Scene video does not belong to this Match.")
        self._assert_artifact_scope(
            shot_boundaries_artifact, project=project
        )
        self._assert_artifact_scope(detections_artifact, project=project)
        video_path = self.storage.resolve_path(scene_video.file_path)
        boundaries_path = self._artifact_path(
            shot_boundaries_artifact
        )
        detections_path = self._artifact_path(detections_artifact)
        shot_scene_context = self._shot_scene_context(detections_artifact)
        discovery_id, discovery_inputs = self._discovery_fingerprint(
            scene_video=scene_video,
            video_path=video_path,
            shot_boundaries_path=boundaries_path,
            detections_path=detections_path,
        )
        output = self._output_root(
            project=project,
            revision=revision,
            scene_id=scene_id,
            discovery_id=discovery_id,
        )
        runtime_reused = (
            (output / "scene_candidate_manifest.json").is_file()
            and (output / "scene_candidates.json").is_file()
        )
        if not runtime_reused:
            self._run(
                [
                "discover",
                "--project-root",
                str(self.runtime_root),
                "--scene-id",
                scene_id,
                "--discovery-id",
                discovery_id,
                "--video",
                str(video_path),
                "--detections-csv",
                str(detections_path),
                "--shot-boundaries",
                str(boundaries_path),
                "--output-root",
                str(output),
                ]
            )
        candidates, candidates_path = load_runtime_json(
            output, "scene_candidates.json"
        )
        manifest, manifest_path = load_runtime_json(
            output,
            "scene_candidate_manifest.json",
        )
        manifest, manifest_candidate_sha256, manifest_sha_repaired = (
            self._ensure_manifest_candidate_sha(
                manifest=manifest,
                manifest_path=manifest_path,
                candidates_path=candidates_path,
            )
        )
        scene_candidates_sha256 = sha256_file(candidates_path)
        verify_candidate_artifact_immutability(
            current_sha256=scene_candidates_sha256,
            stored_discovery_sha256=scene_candidates_sha256,
            manifest_declared_sha256=manifest_candidate_sha256,
        )
        existing_ids = {
            row.candidate_id
            for row in self.repository.list_candidates(
                revision.revision_id, scene_id
            )
        }
        for row in candidates["candidates"]:
            if row["candidate_id"] in existing_ids:
                continue
            representative_path = validated_runtime_file(
                output,
                row["representative_observation"]["thumbnail_artifact"],
                allowed_suffixes=IMAGE_SUFFIXES,
            )
            assert_expected_mime(representative_path, "image/")
            thumbnail = self.artifacts.create(
                match_id=project.match_id,
                project_id=project.project_id,
                analysis_job_id=None,
                artifact_type="SCENE_TARGET_CANDIDATE_REPRESENTATIVE",
                file_path=representative_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                mime_type="image/jpeg",
                metadata_={
                    "revision_id": revision.revision_id,
                    "scene_id": scene_id,
                    "candidate_id": row["candidate_id"],
                    "shot_id": row["shot_id"],
                },
            )
            artifact_ids = {"representative": thumbnail.artifact_id}
            for key, mime_type, artifact_type in (
                (
                    "contact_sheet",
                    "image/jpeg",
                    "SCENE_TARGET_CANDIDATE_CONTACT_SHEET",
                ),
                (
                    "tracklet_review_video",
                    "video/mp4",
                    "SCENE_TARGET_CANDIDATE_TRACKLET_REVIEW",
                ),
            ):
                portable_path = row["artifacts"].get(key)
                if not portable_path:
                    continue
                media_path = validated_runtime_file(
                    output,
                    portable_path,
                    allowed_suffixes=(
                        IMAGE_SUFFIXES
                        if mime_type.startswith("image/")
                        else REVIEW_ARTIFACT_SUFFIXES - IMAGE_SUFFIXES
                    ),
                )
                assert_expected_mime(
                    media_path,
                    "image/" if mime_type.startswith("image/") else "video/",
                )
                media_artifact = self.artifacts.create(
                    match_id=project.match_id,
                    project_id=project.project_id,
                    analysis_job_id=None,
                    artifact_type=artifact_type,
                    file_path=media_path.relative_to(
                        self.storage.project_root
                    ).as_posix(),
                    mime_type=mime_type,
                    metadata_={
                        "revision_id": revision.revision_id,
                        "scene_id": scene_id,
                        "candidate_id": row["candidate_id"],
                        "shot_id": row["shot_id"],
                    },
                )
                artifact_ids[key] = media_artifact.artifact_id
            initialization = row["tracking_initialization_observation"]
            product_scene_context = shot_scene_context.get(
                str(row["shot_id"]),
                {
                    "mode": "MIXED",
                    "reason": "SHOT_SCENE_CONTEXT_MISSING",
                    "evidence_frame_count": 0,
                    "closeup_vote_ratio": None,
                    "role_state": "ROLE_UNVERIFIED",
                },
            )
            self.repository.create_candidate(
                candidate_id=row["candidate_id"],
                revision_id=revision.revision_id,
                scene_id=scene_id,
                anchor_time_sec=float(initialization["time_sec"]),
                anchor_source_time_sec=float(initialization["time_sec"]),
                anchor_frame_index=int(initialization["frame_index"]),
                bbox_xyxy=initialization["bbox_xyxy"],
                thumbnail_artifact_id=thumbnail.artifact_id,
                track_length_frames=int(row["observation_count"]),
                trackability_score=float(
                    row["quality"]["trackability_score"]
                ),
                status="AVAILABLE",
                metadata_={
                    "scene_wide_v1": True,
                    "discovery_id": discovery_id,
                    "shot_id": row["shot_id"],
                    "shot_index": row["shot_index"],
                    "first_frame": row["first_frame"],
                    "last_frame": row["last_frame"],
                    "representative_observation": row[
                        "representative_observation"
                    ],
                    "tracking_initialization_observation": initialization,
                    "quality": row["quality"],
                    "product_scene_context": product_scene_context,
                    "product_role_state": product_scene_context["role_state"],
                    "artifacts": row["artifacts"],
                    "artifact_ids": artifact_ids,
                    "artifact_root": str(
                        output.relative_to(self.storage.project_root)
                    ),
                    "candidate_cache_key": manifest[
                        "candidate_cache_key"
                    ],
                },
            )
        revision.options = {
            **(revision.options or {}),
            "scene_target_selection": {
                "status": "WAITING_TARGET_SELECTION",
                "scene_id": scene_id,
                "scene_video_asset_id": scene_video.asset_id,
                "shot_boundaries_artifact_id": (
                    shot_boundaries_artifact.artifact_id
                ),
                "detections_artifact_id": detections_artifact.artifact_id,
                "artifact_root": str(
                    output.relative_to(self.storage.project_root)
                ),
                "candidate_cache_key": manifest["candidate_cache_key"],
                "scene_candidate_manifest_sha256": sha256_file(
                    manifest_path
                ),
                "scene_candidates_relative_path": str(
                    candidates_path.relative_to(self.storage.project_root)
                ).replace("\\", "/"),
                "scene_candidates_sha256": scene_candidates_sha256,
                "manifest_declared_scene_candidates_sha256": (
                    manifest_candidate_sha256
                ),
                "scene_candidate_manifest_sha_repaired": (
                    manifest_sha_repaired
                ),
                "discovery_id": discovery_id,
                "discovery_inputs": discovery_inputs,
                "runtime_reused": runtime_reused,
                "candidate_count": len(candidates["candidates"]),
                "shot_count": candidates["shot_boundary"]["shot_count"],
                "shot_scene_context": shot_scene_context,
                "shot_scene_router_policy": (
                    ((detections_artifact.metadata_ or {}).get("shot_scene_router") or {}).get(
                        "policy_version"
                    )
                ),
            },
        }
        revision.status = "PLAYER_SELECTION_REQUIRED"
        revision.pending_action = "SELECT_PLAYER"
        self.db.commit()
        return manifest

    def candidate_rows(
        self,
        *,
        project: Project,
        revision_id: str,
        scene_id: str,
    ) -> tuple[HighlightRevision, list[dict[str, Any]]]:
        revision = self._revision(project=project, revision_id=revision_id)
        self._assert_scene(revision, scene_id)
        current_discovery_id = (
            ((revision.options or {}).get("scene_target_selection") or {})
            .get("discovery_id")
        )
        rows = []
        for candidate in self.repository.list_candidates(
            revision.revision_id, scene_id
        ):
            metadata = candidate.metadata_ or {}
            if not metadata.get("scene_wide_v1"):
                continue
            if metadata.get("discovery_id") != current_discovery_id:
                continue
            # Defense-in-depth for the WIDE-only tracking product policy.
            # Fresh V8 detections already exclude CLOSEUP/MIXED rows before
            # scene-candidate grouping, but stale/legacy database candidates
            # must never leak back into the selectable gallery.
            product_scene_context = metadata.get("product_scene_context") or {}
            if str(product_scene_context.get("mode") or "MIXED").upper() != "WIDE":
                continue
            artifact_ids = metadata.get("artifact_ids") or {}
            representative = dict(
                metadata["representative_observation"]
            )
            representative.pop("thumbnail_artifact", None)
            if artifact_ids.get("representative"):
                artifact_id = artifact_ids["representative"]
                representative["thumbnail"] = {
                    "artifact_id": artifact_id,
                    "url": (
                        f"/api/v1/artifacts/{artifact_id}/download"
                    ),
                }
            public_artifacts = {
                key: {
                    "artifact_id": artifact_id,
                    "url": (
                        f"/api/v1/artifacts/{artifact_id}/download"
                    ),
                }
                for key, artifact_id in artifact_ids.items()
                if key != "representative"
            }
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "scene_id": scene_id,
                    "shot_id": metadata["shot_id"],
                    "shot_index": metadata["shot_index"],
                    "first_frame": metadata["first_frame"],
                    "last_frame": metadata["last_frame"],
                    "observation_count": candidate.track_length_frames,
                    "representative_observation": representative,
                    "tracking_initialization_observation": metadata[
                        "tracking_initialization_observation"
                    ],
                    "quality": metadata["quality"],
                    "product_scene_context": metadata.get("product_scene_context") or {
                        "mode": "MIXED",
                        "role_state": "ROLE_UNVERIFIED",
                    },
                    "product_role_state": str(
                        metadata.get("product_role_state") or "ROLE_UNVERIFIED"
                    ),
                    "artifacts": public_artifacts,
                    "artifact_ids": artifact_ids,
                    "gallery_visibility": (
                        "VISIBLE"
                        if candidate.trackability_score >= 0.25
                        else "HIDDEN_LOW_QUALITY"
                    ),
                }
            )
        rows.sort(key=lambda row: (row["shot_index"], row["first_frame"]))
        return revision, rows

    def create_selection(
        self,
        *,
        project: Project,
        revision_id: str,
        scene_id: str,
        candidate_id: str,
        scene_video: MediaAsset,
        user: User,
    ) -> SceneTargetSelection:
        revision = self._revision(project=project, revision_id=revision_id)
        self._assert_scene(revision, scene_id)
        discovery = (revision.options or {}).get(
            "scene_target_selection"
        ) or {}
        if discovery.get("scene_video_asset_id") != scene_video.asset_id:
            raise ValueError(
                "Scene video does not match the discovery input."
            )
        candidate = self.repository.get_candidate(
            revision_id=revision.revision_id,
            candidate_id=candidate_id,
        )
        if candidate is None or candidate.scene_id != scene_id:
            raise ValueError("Candidate does not belong to this scene.")
        metadata = candidate.metadata_ or {}
        if not metadata.get("scene_wide_v1"):
            raise ValueError("Candidate is not a scene-wide selection candidate.")
        if metadata.get("discovery_id") != discovery.get("discovery_id"):
            raise ValueError("Candidate belongs to an obsolete discovery cache.")
        product_scene_context = metadata.get("product_scene_context") or {}
        scene_mode = str(product_scene_context.get("mode") or "MIXED").upper()
        if scene_mode != "WIDE":
            raise ValueError(
                "KickClip target tracking accepts WIDE shots only; "
                f"candidate scene mode is {scene_mode}."
            )
        discovery_output = self.storage.resolve_path(metadata["artifact_root"])
        next_revision = self.repository.next_target_selection_revision(
            revision_id=revision.revision_id,
            scene_id=scene_id,
        )
        workspace = prepare_selection_workspace(
            discovery_output,
            revision=next_revision,
        )
        self._run(
            [
                "select",
                "--output-root",
                str(workspace.staging_root),
                "--video",
                str(self.storage.resolve_path(scene_video.file_path)),
                "--candidate-id",
                candidate_id,
                "--reviewer",
                user.user_id,
                "--selection-revision",
                str(next_revision),
            ]
        )
        selection_artifact, _ = load_runtime_json(
            workspace.staging_root,
            "target_selection.json",
        )
        reference_artifact, _ = load_runtime_json(
            workspace.staging_root,
            "target_reference_set.json",
        )
        if int(selection_artifact["target_selection_revision"]) != next_revision:
            raise RuntimeError("Selection revision allocation disagrees with database.")
        selection_id = str(selection_artifact["selection_id"])
        if not selection_id.replace("_", "").replace("-", "").isalnum():
            raise RuntimeError("Runtime returned an unsafe selection identifier.")
        output = workspace.finalize(selection_id)
        selection_json_path = validated_runtime_file(
            output,
            "target_selection.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        reference_json_path = validated_runtime_file(
            output,
            "target_reference_set.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        path_metadata = {
            "revision_id": revision.revision_id,
            "scene_id": scene_id,
            "selection_id": selection_id,
            "selection_revision": next_revision,
        }
        artifact_ids_by_path = self._register_portable_paths(
            project=project,
            output=output,
            value=reference_artifact,
            metadata=path_metadata,
        )
        artifact_ids_by_path["target_selection.json"] = (
            self._register_runtime_artifact(
                project=project,
                path=selection_json_path,
                artifact_type="SCENE_TARGET_SELECTION_JSON",
                mime_type="application/json",
                metadata=path_metadata,
            )
        )
        artifact_ids_by_path["target_reference_set.json"] = (
            self._register_runtime_artifact(
                project=project,
                path=reference_json_path,
                artifact_type="SCENE_TARGET_REFERENCE_SET_JSON",
                mime_type="application/json",
                metadata=path_metadata,
            )
        )
        output_relative = project_relative(output, self.storage.project_root)
        selection = self.repository.create_target_selection(
            selection_id=selection_id,
            owner_id=user.user_id,
            match_id=project.match_id,
            project_id=project.project_id,
            revision_id=revision.revision_id,
            scene_id=scene_id,
            selection_revision=next_revision,
            selected_candidate_id=candidate_id,
            status="EARLIER_DISCOVERY_REQUIRED",
            artifact_root=output_relative,
            selection_artifact_root=output_relative,
            target_selection_path=project_relative(
                selection_json_path,
                self.storage.project_root,
            ),
            target_selection_sha256=sha256_file(selection_json_path),
            target_reference_set_path=project_relative(
                reference_json_path,
                self.storage.project_root,
            ),
            target_reference_set_sha256=sha256_file(reference_json_path),
            earlier_proposals_path=None,
            earlier_proposals_sha256=None,
            earlier_decision_path=None,
            earlier_decision_sha256=None,
            selection_artifact=selection_artifact,
            reference_set_artifact=reference_artifact,
            earlier_proposals_artifact={},
            earlier_decision_artifact={},
            candidate_cache_key=metadata["candidate_cache_key"],
            metadata_={
                "scene_video_asset_id": scene_video.asset_id,
                "selection_source": "USER_SELECTED_SCENE_WIDE_CANDIDATE",
                "discovery_id": metadata["discovery_id"],
                "discovery_artifact_root": metadata["artifact_root"],
                "artifact_ids_by_path": artifact_ids_by_path,
            },
        )
        for reference in reference_artifact["references"]:
            self.repository.create_target_reference(
                selection_id=selection.selection_id,
                reference_id=reference["reference_id"],
                source_candidate_id=reference["source_candidate_id"],
                frame_index=reference["frame_index"],
                bbox_xyxy=reference["bbox_xyxy"],
                crop_artifact=reference["crop_artifact"],
                quality=reference["quality"],
                metadata_={
                    "selection_reason": reference["selection_reason"]
                },
            )
        self.db.commit()
        self.db.refresh(selection)
        return selection

    def discover_earlier(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
    ) -> SceneTargetSelection:
        self._assert_selection_owner(selection, user)
        output = self.storage.resolve_path(selection.artifact_root)
        existing_rows = self.repository.list_earlier_anchor_proposals(
            selection.selection_id
        )
        if selection.earlier_proposals_artifact and existing_rows:
            return selection
        self._run(
            [
                "propose-earlier",
                "--project-root",
                str(self.runtime_root),
                "--output-root",
                str(output),
                "--device",
                self.settings.TRACKING_DEVICE,
            ]
        )
        proposals, proposals_path = load_runtime_json(
            output,
            "earlier_candidate_proposals.json",
        )
        artifact_ids_by_path = dict(
            (selection.metadata_ or {}).get("artifact_ids_by_path") or {}
        )
        artifact_ids_by_path.update(
            self._register_portable_paths(
                project=selection.revision.project,
                output=output,
                value=proposals,
                metadata={
                    "revision_id": selection.revision_id,
                    "scene_id": selection.scene_id,
                    "selection_id": selection.selection_id,
                    "selection_revision": selection.selection_revision,
                },
            )
        )
        artifact_ids_by_path["earlier_candidate_proposals.json"] = (
            self._register_runtime_artifact(
                project=selection.revision.project,
                path=proposals_path,
                artifact_type="SCENE_TARGET_EARLIER_PROPOSALS_JSON",
                mime_type="application/json",
                metadata={
                    "selection_id": selection.selection_id,
                    "selection_revision": selection.selection_revision,
                },
            )
        )
        selection.earlier_proposals_artifact = proposals
        selection.earlier_proposals_path = project_relative(
            proposals_path,
            self.storage.project_root,
        )
        selection.earlier_proposals_sha256 = sha256_file(proposals_path)
        selection.status = proposals["state"]
        for row in proposals["proposals"]:
            self.repository.upsert_earlier_anchor_proposal(
                selection_id=selection.selection_id,
                candidate_id=row["candidate_id"],
                source_revision_id=selection.revision_id,
                source_discovery_id=str(
                    (selection.metadata_ or {}).get("discovery_id") or ""
                ),
                retrieval_rank=row["retrieval_rank"],
                retrieval_score=row.get("retrieval_score"),
                prototype_similarity=row.get("prototype_similarity"),
                decision_state="PENDING_USER_CONFIRMATION",
                artifacts=row.get("artifacts") or {},
                metadata_={
                    "automatic_confirmation_allowed": False,
                    "selection_revision": selection.selection_revision,
                    "discovery_id": (
                        (selection.metadata_ or {}).get("discovery_id")
                    ),
                },
            )
        selection.metadata_ = {
            **(selection.metadata_ or {}),
            "artifact_ids_by_path": artifact_ids_by_path,
        }
        self.db.commit()
        self.db.refresh(selection)
        return selection

    def decide_earlier(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
        decision: str,
        candidate_id: str | None,
    ) -> SceneTargetSelection:
        self._assert_selection_owner(selection, user)
        output = self.storage.resolve_path(selection.artifact_root)
        action_key = f"{decision}:{candidate_id or ''}"
        completed_action = (selection.metadata_ or {}).get(
            "earlier_decision_action_key"
        )
        if completed_action:
            if completed_action == action_key:
                return selection
            raise ValueError("Earlier anchor decision was already processed.")
        if decision == "candidate":
            proposal = self.repository.get_earlier_anchor_proposal(
                selection_id=selection.selection_id,
                candidate_id=str(candidate_id),
            )
            if proposal is None:
                raise ValueError(
                    "Candidate is not in the presented earlier proposal allowlist."
                )
            if proposal.decision_state != "PENDING_USER_CONFIRMATION":
                raise ValueError("Earlier proposal is no longer pending.")
            if (
                proposal.retrieval_rank
                > self.settings.SCENE_TARGET_SELECTION_MAX_CONFIRMABLE_EARLIER_RANK
            ):
                raise ValueError(
                    "Earlier proposal rank exceeds the confirmable allowlist."
                )
            proposal_scope = proposal.metadata_ or {}
            if (
                int(proposal_scope.get("selection_revision", -1))
                != selection.selection_revision
                or proposal_scope.get("discovery_id")
                != (selection.metadata_ or {}).get("discovery_id")
            ):
                raise ValueError(
                    "Earlier proposal does not match this selection revision."
                )
            arguments = [
                "confirm-earlier",
                "--output-root",
                str(output),
                "--candidate-id",
                str(candidate_id),
                "--reviewer",
                user.user_id,
            ]
        elif decision == "reject_all":
            arguments = [
                "confirm-earlier",
                "--output-root",
                str(output),
                "--reviewer",
                user.user_id,
                "--reject-all",
            ]
        else:
            arguments = [
                "start-selected-shot",
                "--output-root",
                str(output),
                "--reviewer",
                user.user_id,
            ]
        self._run(arguments)
        artifact, decision_path = load_runtime_json(
            output,
            "earlier_anchor_decision.json",
        )
        selection.earlier_decision_artifact = artifact
        selection.earlier_decision_path = project_relative(
            decision_path,
            self.storage.project_root,
        )
        selection.earlier_decision_sha256 = sha256_file(decision_path)
        selection.status = artifact["state"]
        for proposal in self.repository.list_earlier_anchor_proposals(
            selection.selection_id
        ):
            proposal.decision_state = (
                "USER_CONFIRMED"
                if proposal.candidate_id == artifact.get(
                    "confirmed_candidate_id"
                )
                else "USER_REJECTED"
            )
        artifact_ids_by_path = dict(
            (selection.metadata_ or {}).get("artifact_ids_by_path") or {}
        )
        artifact_ids_by_path["earlier_anchor_decision.json"] = (
            self._register_runtime_artifact(
                project=selection.revision.project,
                path=decision_path,
                artifact_type="SCENE_TARGET_EARLIER_DECISION_JSON",
                mime_type="application/json",
                metadata={
                    "selection_id": selection.selection_id,
                    "selection_revision": selection.selection_revision,
                },
            )
        )
        selection.metadata_ = {
            **(selection.metadata_ or {}),
            "earlier_decision_action_key": action_key,
            "artifact_ids_by_path": artifact_ids_by_path,
        }
        self.db.commit()
        self.db.refresh(selection)
        return selection

    def create_tracking_job(
        self,
        *,
        selection: SceneTargetSelection,
        project: Project,
        user: User,
        scene_video: MediaAsset,
        shot_boundaries_artifact: Artifact,
    ):
        self._assert_selection_owner(selection, user)
        if selection.project_id != project.project_id:
            raise ValueError("Target selection does not belong to this Project.")
        expected_asset_id = (selection.metadata_ or {}).get(
            "scene_video_asset_id"
        )
        if expected_asset_id != scene_video.asset_id:
            raise ValueError(
                "Scene video does not match the immutable target selection."
            )
        if not selection.earlier_decision_artifact:
            raise ValueError("Earlier anchor decision is required.")
        self._assert_artifact_scope(
            shot_boundaries_artifact, project=project
        )
        output = self.storage.resolve_path(selection.artifact_root)
        self._run(
            [
                "prepare-tracking",
                "--output-root",
                str(output),
                "--video",
                str(self.storage.resolve_path(scene_video.file_path)),
                "--shot-boundaries",
                str(self._artifact_path(shot_boundaries_artifact)),
            ]
        )
        launch, launch_path = load_runtime_json(
            output,
            "tracking_launch_manifest.json",
        )
        launch_sha256 = sha256_file(launch_path)
        shot_boundaries_path = self._artifact_path(shot_boundaries_artifact)
        shot_boundaries_sha256 = sha256_file(shot_boundaries_path)
        target_selection_path = self._immutable_selection_file(
            selection,
            path_attribute="target_selection_path",
            sha_attribute="target_selection_sha256",
        )
        reference_set_path = self._immutable_selection_file(
            selection,
            path_attribute="target_reference_set_path",
            sha_attribute="target_reference_set_sha256",
        )
        decision_path = self._immutable_selection_file(
            selection,
            path_attribute="earlier_decision_path",
            sha_attribute="earlier_decision_sha256",
        )
        response = TrackingJobService(self.db).create_job(
            user=user,
            asset=scene_video,
            project=project,
            payload=TrackingJobCreateRequest(
                media_asset_id=scene_video.asset_id,
                initial_bbox_xyxy=launch["anchor_bbox_xyxy"],
                project_id=project.project_id,
                match_id=project.match_id,
                reacquisition_mode="assisted",
            ),
            cache_discriminator=launch["tracking_cache_key"],
            runtime_context={
                "scene_target_selection": {
                    "selection_id": selection.selection_id,
                    "selection_revision": selection.selection_revision,
                    "selected_candidate_id": selection.selected_candidate_id,
                    "tracking_cache_key": launch["tracking_cache_key"],
                    "selection_artifact_root": str(output),
                    "r2_production_manifest_sha256": (
                        self.settings.TRACKING_R2_MANIFEST_SHA256
                    ),
                    "tracking_launch_manifest_path": str(launch_path),
                    "tracking_launch_manifest_sha256": launch_sha256,
                    "shot_boundaries_path": str(shot_boundaries_path),
                    "shot_boundaries_sha256": shot_boundaries_sha256,
                    "target_selection_path": str(target_selection_path),
                    "target_selection_sha256": (
                        selection.target_selection_sha256
                    ),
                    "target_reference_set_path": str(reference_set_path),
                    "target_reference_set_sha256": (
                        selection.target_reference_set_sha256
                    ),
                    "earlier_anchor_decision_path": str(decision_path),
                    "earlier_anchor_decision_sha256": (
                        selection.earlier_decision_sha256
                    ),
                }
            },
        )
        selection.tracking_job_id = response.job_id
        selection.tracking_cache_key = launch["tracking_cache_key"]
        selection.status = "TRACKING_QUEUED"
        self.db.commit()
        return response

    @staticmethod
    def _assert_selection_owner(
        selection: SceneTargetSelection,
        user: User,
    ) -> None:
        if (
            selection.owner_id != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Target selection not found.")

    @staticmethod
    def read(selection: SceneTargetSelection) -> SceneTargetSelectionRead:
        artifact_ids = dict(
            (selection.metadata_ or {}).get("artifact_ids_by_path") or {}
        )
        return SceneTargetSelectionRead(
            selection_id=selection.selection_id,
            selection_revision=selection.selection_revision,
            project_id=selection.project_id,
            revision_id=selection.revision_id,
            scene_id=selection.scene_id,
            selected_candidate_id=selection.selected_candidate_id,
            status=selection.status,
            target_selection=redact_runtime_paths(
                selection.selection_artifact or {},
                artifact_ids_by_path=artifact_ids,
            ),
            target_reference_set=redact_runtime_paths(
                selection.reference_set_artifact or {},
                artifact_ids_by_path=artifact_ids,
            ),
            earlier_candidate_proposals=(
                redact_runtime_paths(
                    selection.earlier_proposals_artifact or {},
                    artifact_ids_by_path=artifact_ids,
                )
            ),
            earlier_anchor_decision=(
                redact_runtime_paths(
                    selection.earlier_decision_artifact or {},
                    artifact_ids_by_path=artifact_ids,
                )
            ),
            tracking_job_id=selection.tracking_job_id,
            tracking_cache_key=selection.tracking_cache_key,
        )
