from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.model import SceneTargetSelection
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.runtime_contract import (
    JSON_SUFFIXES,
    ConfiguredSceneRuntime,
    load_runtime_json,
    prepare_selection_workspace,
    project_relative,
    sha256_file,
    validated_runtime_file,
)
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.schema import TrackingJobCreateRequest
from app.domains.tracking.service import TrackingJobService
from app.storage.local_storage import LocalStorage


class SceneTargetReviewabilityService:
    """Additive human-review layer; frozen V1/R3 adapters remain untouched."""

    def __init__(
        self,
        db: Session,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repository = HighlightRepository(db)
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
            / "target_centric_tracking_scene_target_selection_r2"
        )
        self.runner = self.package_root / "run_scene_target_selection_r2.py"

    def owned(
        self,
        selection_id: str,
        user: User,
        *,
        for_update: bool = False,
    ) -> SceneTargetSelection:
        selection = (
            self.repository.get_target_selection_for_update(selection_id)
            if for_update
            else self.repository.get_target_selection(selection_id)
        )
        if selection is None or (
            selection.owner_id != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Target selection not found.")
        return selection

    def _run(self, arguments: list[str]) -> dict[str, Any]:
        result = ConfiguredSceneRuntime(
            self.settings,
            default_script=self.runner,
            configured_script=(
                self.settings.SCENE_TARGET_REVIEWABILITY_SCRIPT_PATH
            ),
            script_setting_name="SCENE_TARGET_REVIEWABILITY_SCRIPT_PATH",
        ).run(
            [
                "--project-root",
                str(self.runtime_root),
                *arguments,
            ],
            expect_json_stdout=True,
        )
        assert result is not None
        return result

    def _output(self, selection: SceneTargetSelection) -> Path:
        output = self.storage.resolve_path(selection.artifact_root)
        if not output.is_relative_to(self.storage.storage_root):
            raise ValueError("Target selection artifact root is invalid.")
        return output

    def _immutable_selection_file(
        self,
        selection: SceneTargetSelection,
        *,
        path_attribute: str,
        sha_attribute: str,
    ) -> Path:
        root = self._output(selection)
        relative = getattr(selection, path_attribute)
        expected = getattr(selection, sha_attribute)
        if not relative or not expected:
            raise ValueError(f"Selection is missing {path_attribute}.")
        path = self.storage.resolve_path(relative)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Immutable selection artifact path is invalid.")
        if sha256_file(path) != expected:
            raise ValueError("Immutable selection artifact hash mismatch.")
        return path

    def _artifact_path(
        self,
        artifact: Artifact,
        *,
        project: Project,
    ) -> Path:
        if artifact.match_id != project.match_id:
            raise ValueError("Artifact does not belong to this Match.")
        if artifact.project_id not in {None, project.project_id}:
            raise ValueError("Artifact does not belong to this Project.")
        path = self.storage.resolve_path(artifact.file_path)
        if not path.is_file():
            raise ValueError("Required artifact is missing.")
        return path

    def _video_path(
        self,
        selection: SceneTargetSelection,
        video: MediaAsset,
    ) -> Path:
        expected = (selection.metadata_ or {}).get(
            "scene_video_asset_id"
        )
        if expected != video.asset_id or video.match_id != selection.match_id:
            raise ValueError("Scene video does not match target selection.")
        path = self.storage.resolve_path(video.file_path)
        if not path.is_file():
            raise ValueError("Scene video is missing.")
        return path

    def prepare_selected_review(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
        project: Project,
        video: MediaAsset,
        boundaries: Artifact,
    ) -> dict[str, Any]:
        self._assert_project(selection, project)
        result = self._run(
            [
                "prepare-selected-review",
                "--output-root",
                str(self._output(selection)),
                "--video",
                str(self._video_path(selection, video)),
                "--shot-boundaries",
                str(self._artifact_path(boundaries, project=project)),
            ]
        )
        selection.status = result["state"]
        selection.metadata_ = {
            **(selection.metadata_ or {}),
            "selected_target_identity_confirmation": result,
            "reviewability_shot_boundaries_artifact_id": (
                boundaries.artifact_id
            ),
        }
        self.db.commit()
        return result

    def decide_selected_identity(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
        decision: str,
        identity_basis: str,
    ) -> dict[str, Any]:
        result = self._run(
            [
                "decide-selected-target",
                "--output-root",
                str(self._output(selection)),
                "--decision",
                decision,
                "--reviewer",
                user.user_id,
                "--identity-basis",
                identity_basis,
            ]
        )
        selection.status = result["state"]
        selection.metadata_ = {
            **(selection.metadata_ or {}),
            "selected_target_identity_confirmation": result,
        }
        self.db.commit()
        return result

    def prepare_earlier_review(
        self,
        *,
        selection: SceneTargetSelection,
        project: Project,
        video: MediaAsset,
        boundaries: Artifact,
    ) -> dict[str, Any]:
        self._assert_selected_confirmed(selection)
        result = self._run(
            [
                "prepare-earlier-review",
                "--output-root",
                str(self._output(selection)),
                "--video",
                str(self._video_path(selection, video)),
                "--shot-boundaries",
                str(self._artifact_path(boundaries, project=project)),
            ]
        )
        selection.status = result["state"]
        selection.metadata_ = {
            **(selection.metadata_ or {}),
            "earlier_candidate_reviewability": result,
        }
        self.db.commit()
        return result

    def render_review_ui(
        self,
        *,
        selection: SceneTargetSelection,
        project: Project,
        video: MediaAsset,
        boundaries: Artifact,
    ) -> dict[str, Any]:
        return self._run(
            [
                "render-target-review-ui",
                "--output-root",
                str(self._output(selection)),
                "--video",
                str(self._video_path(selection, video)),
                "--shot-boundaries",
                str(self._artifact_path(boundaries, project=project)),
            ]
        )

    def validate_manual_anchor(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
        project: Project,
        video: MediaAsset,
        boundaries: Artifact,
        global_frame: int,
        click_xy: list[float] | None,
        drawn_bbox: list[float] | None,
        identity_basis: str,
    ) -> dict[str, Any]:
        self._assert_selected_confirmed(selection)
        arguments = [
            "validate-manual-anchor",
            "--output-root",
            str(self._output(selection)),
            "--video",
            str(self._video_path(selection, video)),
            "--shot-boundaries",
            str(self._artifact_path(boundaries, project=project)),
            "--global-frame",
            str(global_frame),
            "--reviewer",
            user.user_id,
            "--identity-basis",
            identity_basis,
        ]
        if click_xy is not None:
            arguments.extend(["--click-xy", *(str(value) for value in click_xy)])
        else:
            arguments.extend(
                ["--drawn-bbox", *(str(value) for value in drawn_bbox or [])]
            )
        result = self._run(arguments)
        selection.status = result["state"]
        selection.metadata_ = {
            **(selection.metadata_ or {}),
            "manual_anchor_draft": result,
        }
        self.db.commit()
        return result

    def confirm_manual_anchor(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
    ) -> SceneTargetSelection:
        self._assert_selected_confirmed(selection)
        expected_revision = self.repository.next_target_selection_revision(
            revision_id=selection.revision_id,
            scene_id=selection.scene_id,
        )
        discovery_root_value = (selection.metadata_ or {}).get(
            "discovery_artifact_root"
        )
        if not discovery_root_value:
            raise ValueError("Selection is missing its discovery artifact root.")
        workspace = prepare_selection_workspace(
            self.storage.resolve_path(discovery_root_value),
            source_root=self._output(selection),
            revision=expected_revision,
        )
        result = self._run(
            [
                "confirm-manual-anchor",
                "--output-root",
                str(workspace.staging_root),
                "--reviewer",
                user.user_id,
            ]
        )
        selection_artifact = result["target_selection"]
        decision = result["decision"]
        if int(selection_artifact["target_selection_revision"]) != expected_revision:
            raise RuntimeError(
                "Manual selection revision disagrees with database."
            )
        versioned_selection_path = validated_runtime_file(
            workspace.staging_root,
            f"target_selection_r{expected_revision:04d}.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        versioned_reference_path = validated_runtime_file(
            workspace.staging_root,
            f"target_reference_set_r{expected_revision:04d}.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        decision_path = validated_runtime_file(
            workspace.staging_root,
            "reviewability_r2/manual_anchor/manual_earlier_anchor_decision.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        shutil.copy2(
            versioned_selection_path,
            workspace.staging_root / "target_selection.json",
        )
        shutil.copy2(
            versioned_reference_path,
            workspace.staging_root / "target_reference_set.json",
        )
        reference_artifact, _ = load_runtime_json(
            workspace.staging_root,
            "target_reference_set.json",
        )
        selection_id = str(selection_artifact["selection_id"])
        if not selection_id.replace("_", "").replace("-", "").isalnum():
            raise RuntimeError("Runtime returned an unsafe selection identifier.")
        output = workspace.finalize(selection_id)
        target_selection_path = validated_runtime_file(
            output,
            "target_selection.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        target_reference_set_path = validated_runtime_file(
            output,
            "target_reference_set.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        final_decision_path = validated_runtime_file(
            output,
            decision_path.relative_to(workspace.staging_root),
            allowed_suffixes=JSON_SUFFIXES,
        )
        output_relative = project_relative(output, self.storage.project_root)
        from app.domains.highlight.scene_target_selection import (
            SceneTargetSelectionService,
        )

        artifact_service = SceneTargetSelectionService(
            self.db,
            settings=self.settings,
        )
        artifact_ids_by_path = dict(
            (selection.metadata_ or {}).get("artifact_ids_by_path") or {}
        )
        artifact_metadata = {
            "revision_id": selection.revision_id,
            "scene_id": selection.scene_id,
            "selection_id": selection_id,
            "selection_revision": expected_revision,
        }
        artifact_ids_by_path.update(
            artifact_service._register_portable_paths(
                project=selection.revision.project,
                output=output,
                value=reference_artifact,
                metadata=artifact_metadata,
            )
        )
        for portable, path, artifact_type in (
            (
                "target_selection.json",
                target_selection_path,
                "SCENE_TARGET_SELECTION_JSON",
            ),
            (
                "target_reference_set.json",
                target_reference_set_path,
                "SCENE_TARGET_REFERENCE_SET_JSON",
            ),
            (
                final_decision_path.relative_to(output).as_posix(),
                final_decision_path,
                "SCENE_TARGET_EARLIER_DECISION_JSON",
            ),
        ):
            artifact_ids_by_path[portable] = (
                artifact_service._register_runtime_artifact(
                    project=selection.revision.project,
                    path=path,
                    artifact_type=artifact_type,
                    mime_type="application/json",
                    metadata=artifact_metadata,
                )
            )
        created = self.repository.create_target_selection(
            selection_id=selection_id,
            owner_id=selection.owner_id,
            match_id=selection.match_id,
            project_id=selection.project_id,
            revision_id=selection.revision_id,
            scene_id=selection.scene_id,
            selection_revision=expected_revision,
            selected_candidate_id=selection.selected_candidate_id,
            status="USER_MANUALLY_SELECTED_EARLIER_ANCHOR",
            artifact_root=output_relative,
            selection_artifact_root=output_relative,
            target_selection_path=project_relative(
                target_selection_path,
                self.storage.project_root,
            ),
            target_selection_sha256=sha256_file(target_selection_path),
            target_reference_set_path=project_relative(
                target_reference_set_path,
                self.storage.project_root,
            ),
            target_reference_set_sha256=sha256_file(
                target_reference_set_path
            ),
            earlier_proposals_path=selection.earlier_proposals_path,
            earlier_proposals_sha256=selection.earlier_proposals_sha256,
            earlier_decision_path=project_relative(
                final_decision_path,
                self.storage.project_root,
            ),
            earlier_decision_sha256=sha256_file(final_decision_path),
            selection_artifact=selection_artifact,
            reference_set_artifact=reference_artifact,
            earlier_proposals_artifact=selection.earlier_proposals_artifact,
            earlier_decision_artifact=decision,
            candidate_cache_key=selection.candidate_cache_key,
            metadata_={
                **(selection.metadata_ or {}),
                "previous_selection_id": selection.selection_id,
                "manual_anchor_detection_validated": True,
                "discovery_id": (
                    selection.metadata_ or {}
                ).get("discovery_id"),
                "discovery_artifact_root": discovery_root_value,
                "artifact_ids_by_path": artifact_ids_by_path,
                "scene_video_asset_id": (
                    selection.metadata_ or {}
                ).get("scene_video_asset_id"),
            },
        )
        for reference in reference_artifact.get("references") or []:
            self.repository.create_target_reference(
                selection_id=created.selection_id,
                reference_id=reference["reference_id"],
                source_candidate_id=reference["source_candidate_id"],
                frame_index=reference["frame_index"],
                bbox_xyxy=reference["bbox_xyxy"],
                crop_artifact=reference["crop_artifact"],
                quality=reference["quality"],
                metadata_={
                    "selection_reason": reference["selection_reason"],
                    "copied_from_selection_id": selection.selection_id,
                },
            )
        self.db.commit()
        self.db.refresh(created)
        return created

    def create_manual_tracking_job(
        self,
        *,
        selection: SceneTargetSelection,
        user: User,
        project: Project,
        video: MediaAsset,
        boundaries: Artifact,
    ):
        self._assert_project(selection, project)
        if selection.status != "USER_MANUALLY_SELECTED_EARLIER_ANCHOR":
            raise ValueError("A confirmed manual anchor revision is required.")
        output = self._output(selection)
        result = self._run(
            [
                "prepare-manual-tracking",
                "--output-root",
                str(output),
                "--video",
                str(self._video_path(selection, video)),
                "--shot-boundaries",
                str(self._artifact_path(boundaries, project=project)),
            ]
        )
        revision = selection.selection_revision
        launch_path = validated_runtime_file(
            output,
            "reviewability_r2/manual_anchor/tracking_launch_manifest.json",
            allowed_suffixes=JSON_SUFFIXES,
        )
        selection_path = self._immutable_selection_file(
            selection,
            path_attribute="target_selection_path",
            sha_attribute="target_selection_sha256",
        )
        references_path = self._immutable_selection_file(
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
            asset=video,
            project=project,
            payload=TrackingJobCreateRequest(
                media_asset_id=video.asset_id,
                initial_bbox_xyxy=result["anchor_bbox_xyxy"],
                project_id=project.project_id,
                match_id=project.match_id,
                reacquisition_mode="assisted",
            ),
            cache_discriminator=result["tracking_cache_key"],
            runtime_context={
                "scene_target_selection": {
                    "selection_id": selection.selection_id,
                    "selection_revision": revision,
                    "selected_candidate_id": selection.selected_candidate_id,
                    "manual_anchor_candidate_id": result[
                        "anchor_candidate_id"
                    ],
                    "tracking_cache_key": result["tracking_cache_key"],
                    "selection_artifact_root": str(output),
                    "r2_production_manifest_sha256": (
                        self.settings.TRACKING_R2_MANIFEST_SHA256
                    ),
                    "tracking_launch_manifest_path": str(launch_path),
                    "shot_boundaries_path": str(
                        self._artifact_path(boundaries, project=project)
                    ),
                    "target_selection_path": str(selection_path),
                    "target_selection_sha256": (
                        selection.target_selection_sha256
                    ),
                    "target_reference_set_path": str(references_path),
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
        selection.tracking_cache_key = result["tracking_cache_key"]
        selection.status = "TRACKING_QUEUED"
        self.db.commit()
        return response

    @staticmethod
    def _assert_project(
        selection: SceneTargetSelection,
        project: Project,
    ) -> None:
        if selection.project_id != project.project_id:
            raise ValueError("Target selection does not belong to Project.")

    @staticmethod
    def _assert_selected_confirmed(
        selection: SceneTargetSelection,
    ) -> None:
        confirmation = (selection.metadata_ or {}).get(
            "selected_target_identity_confirmation"
        ) or {}
        if confirmation.get("state") != "TARGET_IDENTITY_HUMAN_CONFIRMED":
            raise ValueError(
                "Selected target identity must be human-confirmed first."
            )

