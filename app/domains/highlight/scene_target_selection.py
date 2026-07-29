from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
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
from app.domains.highlight.schema import SceneTargetSelectionRead
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.schema import TrackingJobCreateRequest
from app.domains.tracking.service import TrackingJobService
from app.storage.local_storage import LocalStorage


R2_MANIFEST_SHA256 = (
    "751338f51c4f7c08bb24576e5afea7d82b9cbbdefbb651ff1a3c1d5c45ffcc86"
)


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
        self.runtime_root = (
            self.storage.project_root / ".tracking-runtime"
        ).resolve()
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
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self.package_root)
            if not existing
            else f"{self.package_root}{os.pathsep}{existing}"
        )
        completed = subprocess.run(
            [sys.executable, str(self.runner), *arguments],
            cwd=str(self.storage.project_root),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Scene target selection runtime failed: "
                + (completed.stderr[-3000:] or completed.stdout[-3000:])
            )

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
        candidates = json.loads(
            (output / "scene_candidates.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (output / "scene_candidate_manifest.json").read_text(
                encoding="utf-8"
            )
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
            representative_path = output / row[
                "representative_observation"
            ]["thumbnail_artifact"]
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
                media_path = output / portable_path
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
                "discovery_id": discovery_id,
                "discovery_inputs": discovery_inputs,
                "runtime_reused": runtime_reused,
                "candidate_count": len(candidates["candidates"]),
                "shot_count": candidates["shot_boundary"]["shot_count"],
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
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "scene_id": scene_id,
                    "shot_id": metadata["shot_id"],
                    "shot_index": metadata["shot_index"],
                    "first_frame": metadata["first_frame"],
                    "last_frame": metadata["last_frame"],
                    "observation_count": candidate.track_length_frames,
                    "representative_observation": metadata[
                        "representative_observation"
                    ],
                    "tracking_initialization_observation": metadata[
                        "tracking_initialization_observation"
                    ],
                    "quality": metadata["quality"],
                    "artifacts": metadata["artifacts"],
                    "artifact_ids": metadata.get("artifact_ids") or {},
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
        output = self.storage.resolve_path(metadata["artifact_root"])
        next_revision = self.repository.next_target_selection_revision(
            revision_id=revision.revision_id,
            scene_id=scene_id,
        )
        self._run(
            [
                "select",
                "--output-root",
                str(output),
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
        selection_artifact = json.loads(
            (output / "target_selection.json").read_text(encoding="utf-8")
        )
        reference_artifact = json.loads(
            (output / "target_reference_set.json").read_text(encoding="utf-8")
        )
        if int(selection_artifact["target_selection_revision"]) != next_revision:
            raise RuntimeError("Selection revision allocation disagrees with database.")
        selection = self.repository.create_target_selection(
            selection_id=selection_artifact["selection_id"],
            owner_id=user.user_id,
            match_id=project.match_id,
            project_id=project.project_id,
            revision_id=revision.revision_id,
            scene_id=scene_id,
            selection_revision=next_revision,
            selected_candidate_id=candidate_id,
            status="EARLIER_DISCOVERY_REQUIRED",
            artifact_root=metadata["artifact_root"],
            selection_artifact=selection_artifact,
            reference_set_artifact=reference_artifact,
            earlier_proposals_artifact={},
            earlier_decision_artifact={},
            candidate_cache_key=metadata["candidate_cache_key"],
            metadata_={
                "scene_video_asset_id": scene_video.asset_id,
                "selection_source": "USER_SELECTED_SCENE_WIDE_CANDIDATE",
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
        proposals = json.loads(
            (output / "earlier_candidate_proposals.json").read_text(
                encoding="utf-8"
            )
        )
        selection.earlier_proposals_artifact = proposals
        selection.status = proposals["state"]
        for row in proposals["proposals"]:
            self.repository.create_earlier_anchor_proposal(
                selection_id=selection.selection_id,
                candidate_id=row["candidate_id"],
                retrieval_rank=row["retrieval_rank"],
                retrieval_score=row.get("retrieval_score"),
                prototype_similarity=row.get("prototype_similarity"),
                decision_state="PENDING_USER_CONFIRMATION",
                artifacts=row.get("artifacts") or {},
                metadata_={"automatic_confirmation_allowed": False},
            )
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
        if decision == "candidate":
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
        artifact = json.loads(
            (output / "earlier_anchor_decision.json").read_text(
                encoding="utf-8"
            )
        )
        selection.earlier_decision_artifact = artifact
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
        launch = json.loads(
            (output / "tracking_launch_manifest.json").read_text(
                encoding="utf-8"
            )
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
                    "r2_production_manifest_sha256": R2_MANIFEST_SHA256,
                    "tracking_launch_manifest_path": str(
                        output / "tracking_launch_manifest.json"
                    ),
                    "shot_boundaries_path": str(
                        self._artifact_path(shot_boundaries_artifact)
                    ),
                    "target_selection_path": str(
                        output / "target_selection.json"
                    ),
                    "target_reference_set_path": str(
                        output / "target_reference_set.json"
                    ),
                    "earlier_anchor_decision_path": str(
                        output / "earlier_anchor_decision.json"
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
        return SceneTargetSelectionRead(
            selection_id=selection.selection_id,
            selection_revision=selection.selection_revision,
            project_id=selection.project_id,
            revision_id=selection.revision_id,
            scene_id=selection.scene_id,
            selected_candidate_id=selection.selected_candidate_id,
            status=selection.status,
            target_selection=selection.selection_artifact or {},
            target_reference_set=selection.reference_set_artifact or {},
            earlier_candidate_proposals=(
                selection.earlier_proposals_artifact or {}
            ),
            earlier_anchor_decision=(
                selection.earlier_decision_artifact or {}
            ),
            tracking_job_id=selection.tracking_job_id,
            tracking_cache_key=selection.tracking_cache_key,
        )
