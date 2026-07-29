from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.model import SceneTargetSelection
from app.domains.highlight.repository import HighlightRepository
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.schema import TrackingJobCreateRequest
from app.domains.tracking.service import TrackingJobService
from app.storage.local_storage import LocalStorage


R2_MANIFEST_SHA256 = (
    "751338f51c4f7c08bb24576e5afea7d82b9cbbdefbb651ff1a3c1d5c45ffcc86"
)


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
        self.runtime_root = (
            self.storage.project_root / ".tracking-runtime"
        ).resolve()
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
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self.package_root)
            if not existing
            else f"{self.package_root}{os.pathsep}{existing}"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(self.runner),
                "--project-root",
                str(self.runtime_root),
                *arguments,
            ],
            cwd=str(self.storage.project_root),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Scene target reviewability runtime failed: "
                + (completed.stderr[-3000:] or completed.stdout[-3000:])
            )
        return json.loads(completed.stdout)

    def _output(self, selection: SceneTargetSelection) -> Path:
        output = self.storage.resolve_path(selection.artifact_root)
        if not output.is_relative_to(self.storage.storage_root):
            raise ValueError("Target selection artifact root is invalid.")
        return output

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
        result = self._run(
            [
                "confirm-manual-anchor",
                "--output-root",
                str(self._output(selection)),
                "--reviewer",
                user.user_id,
            ]
        )
        selection_artifact = result["target_selection"]
        decision = result["decision"]
        expected_revision = self.repository.next_target_selection_revision(
            revision_id=selection.revision_id,
            scene_id=selection.scene_id,
        )
        if int(selection_artifact["target_selection_revision"]) != expected_revision:
            raise RuntimeError(
                "Manual selection revision disagrees with database."
            )
        reference_artifact = json.loads(
            (
                self._output(selection)
                / f"target_reference_set_r{expected_revision:04d}.json"
            ).read_text(encoding="utf-8")
        )
        created = self.repository.create_target_selection(
            selection_id=selection_artifact["selection_id"],
            owner_id=selection.owner_id,
            match_id=selection.match_id,
            project_id=selection.project_id,
            revision_id=selection.revision_id,
            scene_id=selection.scene_id,
            selection_revision=expected_revision,
            selected_candidate_id=selection.selected_candidate_id,
            status="USER_MANUALLY_SELECTED_EARLIER_ANCHOR",
            artifact_root=selection.artifact_root,
            selection_artifact=selection_artifact,
            reference_set_artifact=reference_artifact,
            earlier_proposals_artifact=selection.earlier_proposals_artifact,
            earlier_decision_artifact=decision,
            candidate_cache_key=selection.candidate_cache_key,
            metadata_={
                **(selection.metadata_ or {}),
                "previous_selection_id": selection.selection_id,
                "manual_anchor_detection_validated": True,
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
        launch_path = (
            output
            / "reviewability_r2"
            / "manual_anchor"
            / "tracking_launch_manifest.json"
        )
        selection_path = output / f"target_selection_r{revision:04d}.json"
        references_path = output / f"target_reference_set_r{revision:04d}.json"
        decision_path = (
            output
            / "reviewability_r2"
            / "manual_anchor"
            / "manual_earlier_anchor_decision.json"
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
                    "r2_production_manifest_sha256": R2_MANIFEST_SHA256,
                    "tracking_launch_manifest_path": str(launch_path),
                    "shot_boundaries_path": str(
                        self._artifact_path(boundaries, project=project)
                    ),
                    "target_selection_path": str(selection_path),
                    "target_reference_set_path": str(references_path),
                    "earlier_anchor_decision_path": str(decision_path),
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

