from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    ImmutableCandidateInput,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.backend_adapter import (
    EventCandidateRankingV112aBackendAdapter,
)
from app.domains.highlight.event_candidate_ranking_v1_2.backend_adapter import (
    EventCandidateRankingV12BackendAdapter,
)
from app.domains.highlight.model import HighlightRevision, SceneAITask
from app.domains.highlight.scene_target_selection import (
    SceneTargetSelectionService,
)
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .artifacts import canonical_sha256, sha256_file, write_json_atomic
from .candidate_grouping import (
    CANDIDATE_GROUPING_POLICY_VERSION,
    build_candidate_grouping,
)
from .errors import CandidatePreparationError
from .media_cache import SharedFrameCache
from .review_bundle import (
    MEDIA_MATERIALIZATION_POLICY,
    build_candidate_review_bundle,
)
from .reviewed_input_recovery import (
    ReviewedInputRecoveryError,
    find_recoverable_reviewed_input_bundle,
)
from .service import (
    BUNDLE_ARTIFACT_TYPE,
    GROUPING_ARTIFACT_TYPE,
    public_candidate_id,
)
from .work_metrics import CandidatePreparationWorkMetrics

SOURCE_ARTIFACT_TYPE = "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW"
REVIEWED_SHOTS_ARTIFACT_TYPE = "REVIEWED_SHOT_BOUNDARIES"
AUTO_SHOTS_ARTIFACT_TYPE = "AUTO_SHOT_BOUNDARIES"
PREPARATION_TASK_TYPE = "EVENT_CANDIDATE_RECOMMENDATION_PREPARE_R1C"
USABLE_RANKING_STATES = {
    "AVAILABLE",
    "COMPLETED",
    "PROVISIONAL_SHADOW_ONLY",
    "READY",
    "SUCCESS",
}
USABLE_REVIEW_STATES = {
    "AVAILABLE",
    "COMPLETED",
    "READY",
    "REVIEWED_PASS",
    "SUCCESS",
}
_integration_installed = False


class _CandidatePipelineSceneTargetSelectionService(SceneTargetSelectionService):
    """Use a Windows-safe immutable root for generated candidate media."""

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
            / "candidate_discoveries"
            / project.project_id
            / revision.revision_id
            / scene_id
            / discovery_id
        ).resolve()
        if not root.is_relative_to(self.storage.storage_root):
            raise ValueError("Scene target selection path escapes storage.")
        return root


class EventCandidateRecommendationPreparationService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.media = MediaAssetRepository(db)

    @staticmethod
    def _metadata_matches(
        artifact: Artifact,
        expected: dict[str, Any],
    ) -> bool:
        metadata = artifact.metadata_ or {}
        return all(metadata.get(key) == value for key, value in expected.items())

    @staticmethod
    def _windows_access_path(path: Path) -> Path:
        value = str(path)
        if value.startswith("\\\\?\\") or value.startswith("//?/"):
            return path
        return Path("\\\\?\\" + value) if len(value) >= 248 else path

    @staticmethod
    def _logical_windows_path(path: Path) -> Path:
        value = str(path)
        if value.startswith("\\\\?\\") or value.startswith("//?/"):
            value = value[4:]
        return Path(value).resolve()

    def _verified_artifact_path(
        self,
        artifact: Artifact,
        *,
        code: str,
    ) -> Path:
        try:
            path = self.storage.resolve_path(artifact.file_path)
        except ValueError as exc:
            raise CandidatePreparationError(
                code, "Artifact path is outside immutable storage."
            ) from exc
        path = self._windows_access_path(path)
        if not path.is_file():
            raise CandidatePreparationError(code, "Artifact file is missing.")
        expected = str((artifact.metadata_ or {}).get("sha256") or "")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise CandidatePreparationError(code, "Artifact SHA-256 validation failed.")
        return path

    def _latest_source(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> tuple[Artifact, Path]:
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == SOURCE_ARTIFACT_TYPE,
            )
        ).all()
        expected = {
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
        }
        matches = [
            row
            for row in rows
            if row.match_id == project.match_id
            and self._metadata_matches(row, expected)
            and str((row.metadata_ or {}).get("status") or "") in USABLE_RANKING_STATES
        ]
        if not matches:
            raise CandidatePreparationError(
                "V1_1_2A_RANKING_NOT_READY",
                "A completed V1.1.2a ranking for this event and scene is not ready.",
            )
        matches.sort(key=lambda row: row.created_at, reverse=True)
        source = matches[0]
        return source, self._verified_artifact_path(
            source, code="V1_1_2A_RANKING_NOT_READY"
        )

    def _has_source(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
        reviewed_shot_boundaries_sha256: str | None = None,
    ) -> bool:
        expected = {
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
        }
        return any(
            row.match_id == project.match_id
            and self._metadata_matches(row, expected)
            and (
                reviewed_shot_boundaries_sha256 is None
                or str(
                    ((row.metadata_ or {}).get("freeze_material") or {}).get(
                        "shot_boundaries_sha256"
                    )
                    or ""
                )
                == reviewed_shot_boundaries_sha256
            )
            for row in self.db.scalars(
                select(Artifact).where(
                    Artifact.project_id == project.project_id,
                    Artifact.artifact_type == SOURCE_ARTIFACT_TYPE,
                )
            ).all()
        )

    def _copy_immutable(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(self._windows_access_path(destination)) != sha256_file(
                self._windows_access_path(source)
            ):
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Existing revision input differs from the reviewed immutable input.",
                )
            return
        shutil.copyfile(
            self._windows_access_path(source),
            self._windows_access_path(destination),
        )

    def _materialize_candidate_inputs(
        self,
        *,
        project: Project,
        revision: HighlightRevision,
        event: TimelineEvent,
        scene_id: str,
    ) -> tuple[MediaAsset, Artifact, Artifact]:
        """Resolve exact automatic-or-reviewed candidate discovery inputs.

        Fresh revisions consume their materialized automatic artifact directly.
        The donor/recovery branch below is retained only for R12-R14 reviewed
        bundles and still requires the same Match, event, source asset, scene
        interval and scene-video digest.
        """
        existing = (revision.options or {}).get("candidate_pipeline_inputs") or {}
        if existing.get("scene_id") == scene_id:
            scene_video = self.db.get(
                MediaAsset, str(existing.get("scene_video_asset_id") or "")
            )
            reviewed = self.db.get(
                Artifact,
                str(
                    existing.get("shot_boundaries_artifact_id")
                    or existing.get("reviewed_shots_artifact_id")
                    or ""
                ),
            )
            detections = self.db.get(
                Artifact, str(existing.get("detections_artifact_id") or "")
            )
            if (
                scene_video is not None
                and reviewed is not None
                and detections is not None
                and reviewed.artifact_type
                in {AUTO_SHOTS_ARTIFACT_TYPE, REVIEWED_SHOTS_ARTIFACT_TYPE}
            ):
                self._verified_artifact_path(
                    reviewed, code="SHOT_BOUNDARY_DISCOVERY_INPUT_NOT_READY"
                )
                self._verified_artifact_path(
                    detections, code="INPUT_PROVENANCE_MISMATCH"
                )
                video_path = self._windows_access_path(
                    self.storage.resolve_path(scene_video.file_path)
                )
                if (
                    video_path.is_file()
                    and scene_video.sha256
                    and sha256_file(video_path) == scene_video.sha256
                    and existing.get("source_video_sha256") == scene_video.sha256
                ):
                    source_asset = (
                        revision.action_spotting_job.media_asset
                        if revision.action_spotting_job is not None
                        else None
                    )
                    source_fps = float(
                        source_asset.fps if source_asset is not None else 0.0
                    )
                    if source_fps <= 0:
                        raise CandidatePreparationError(
                            "FRAME_MAPPING_NOT_READY",
                            "The immutable source FPS required for frame mapping is missing.",
                        )
                    revision.options = {
                        **(revision.options or {}),
                        "candidate_pipeline_inputs": {
                            **existing,
                            "source_start_frame": round(
                                float(event.start_sec) * source_fps
                            ),
                            "source_frame_mapping_basis": (
                                "timeline_event_start_sec_x_source_asset_fps"
                            ),
                            "candidate_source_to_video_frame_offset": 0,
                            "candidate_video_frame_mapping_basis": (
                                "scene_candidate_frame_index_is_scene_video_frame_index"
                            ),
                        },
                    }
                    self.db.commit()
                    return scene_video, reviewed, detections
                raise CandidatePreparationError(
                    "SOURCE_VIDEO_NOT_READY",
                    "The materialized scene video is missing or its SHA-256 changed.",
                )
        job = revision.action_spotting_job
        source_asset = job.media_asset if job is not None else None
        if source_asset is None or source_asset.match_id != project.match_id:
            raise CandidatePreparationError(
                "SOURCE_VIDEO_NOT_READY",
                "The selected revision has no immutable Action Spotting source video.",
            )
        source_path = self.storage.resolve_path(source_asset.file_path)
        if (
            not source_path.is_file()
            or not source_asset.sha256
            or sha256_file(self._windows_access_path(source_path))
            != source_asset.sha256
        ):
            raise CandidatePreparationError(
                "SOURCE_VIDEO_NOT_READY",
                "The Action Spotting source video is missing or its SHA-256 changed.",
            )
        source_fps = float(source_asset.fps or 0.0)
        if source_fps <= 0:
            raise CandidatePreparationError(
                "FRAME_MAPPING_NOT_READY",
                "The immutable source FPS required for frame mapping is missing.",
            )

        reviewed_rows = self.db.scalars(
            select(Artifact).where(
                Artifact.match_id == project.match_id,
                Artifact.artifact_type == REVIEWED_SHOTS_ARTIFACT_TYPE,
            )
        ).all()
        donors: list[tuple[Artifact, MediaAsset, Artifact]] = []
        for reviewed_candidate in reviewed_rows:
            metadata = reviewed_candidate.metadata_ or {}
            if metadata.get("scene_id") != scene_id or metadata.get("event_id") not in {
                None,
                event.timeline_event_id,
            }:
                continue
            donor_revision = self.db.get(
                HighlightRevision, str(metadata.get("revision_id") or "")
            )
            if donor_revision is None:
                continue

            # Prefer the generic candidate-pipeline provenance.  The legacy
            # representative_goal block is retained only as a compatibility
            # fallback for old seeded databases.
            candidate_inputs = (donor_revision.options or {}).get(
                "candidate_pipeline_inputs"
            ) or {}
            representative_goal = (donor_revision.options or {}).get(
                "representative_goal"
            ) or {}
            if (
                candidate_inputs.get("scene_id") == scene_id
                and metadata.get("action_spotting_source_video_sha256")
                == source_asset.sha256
                and abs(
                    float(metadata.get("event_source_start_sec", -1))
                    - float(event.start_sec)
                )
                <= 1e-6
                and abs(
                    float(metadata.get("event_source_end_sec", -1))
                    - float(event.end_sec)
                )
                <= 1e-6
            ):
                scene_asset_id = candidate_inputs.get("scene_video_asset_id")
                detections_artifact_id = candidate_inputs.get("detections_artifact_id")
            elif (
                representative_goal.get("event_id") == event.timeline_event_id
                and representative_goal.get("source_job_id")
                == revision.action_spotting_job_id
                and representative_goal.get("source_asset_id") == source_asset.asset_id
                and abs(
                    float(representative_goal.get("scene_start_sec", -1))
                    - float(event.start_sec)
                )
                <= 1e-6
                and abs(
                    float(representative_goal.get("scene_end_sec", -1))
                    - float(event.end_sec)
                )
                <= 1e-6
            ):
                scene_asset_id = representative_goal.get("scene_video_asset_id")
                detections_artifact_id = representative_goal.get(
                    "detections_artifact_id"
                )
            else:
                continue
            scene_asset = self.db.get(MediaAsset, str(scene_asset_id or ""))
            detection_artifact = self.db.get(
                Artifact, str(detections_artifact_id or "")
            )
            if (
                scene_asset is None
                or detection_artifact is None
                or scene_asset.match_id != project.match_id
                or detection_artifact.match_id != project.match_id
                or (detection_artifact.metadata_ or {}).get("scene_id") != scene_id
            ):
                continue
            donors.append((reviewed_candidate, scene_asset, detection_artifact))

        donor_reviewed: Artifact | None = None
        donor_detections: Artifact | None = None
        materialized_from_artifact_id: str | None = None
        materialized_from_detections_artifact_id: str | None = None
        recovery_metadata: dict[str, Any] = {}
        if donors:
            donors.sort(key=lambda row: row[0].created_at, reverse=True)
            donor_reviewed, donor_video, donor_detections = donors[0]
            donor_video_path = self._windows_access_path(
                self.storage.resolve_path(donor_video.file_path)
            )
            reviewed_path = self._verified_artifact_path(
                donor_reviewed, code="REVIEWED_SHOT_BOUNDARIES_NOT_READY"
            )
            try:
                detections_path = self._windows_access_path(
                    self.storage.resolve_path(donor_detections.file_path)
                )
            except ValueError as exc:
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Frozen detection path is outside immutable storage.",
                ) from exc
            detection_metadata = donor_detections.metadata_ or {}
            expected_detection_sha = str(
                detection_metadata.get("sha256")
                or detection_metadata.get("detections_sha256")
                or ""
            )
            if (
                not detections_path.is_file()
                or len(expected_detection_sha) != 64
                or sha256_file(detections_path) != expected_detection_sha
            ):
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Frozen detection file is missing or its SHA-256 changed.",
                )
            video_sha = (
                sha256_file(donor_video_path) if donor_video_path.is_file() else ""
            )
            boundaries = self._load_json(reviewed_path)
            if (
                not donor_video.sha256
                or video_sha != donor_video.sha256
                or (boundaries.get("video") or {}).get("sha256") != video_sha
                or detection_metadata.get("source_video_sha256") != video_sha
                or detection_metadata.get("detections_sha256")
                != sha256_file(detections_path)
            ):
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Reviewed inputs do not share the same immutable scene-video SHA-256.",
                )
            duration_sec = donor_video.duration_sec
            scene_fps = donor_video.fps
            scene_width = donor_video.width
            scene_height = donor_video.height
            reviewed_metadata = dict(donor_reviewed.metadata_ or {})
            materialized_from_artifact_id = donor_reviewed.artifact_id
            materialized_from_detections_artifact_id = donor_detections.artifact_id
            recovery_metadata = {
                "input_resolution_mode": "DATABASE_ARTIFACT",
                "automatic_review_approval": False,
            }
        else:
            try:
                recovered = find_recoverable_reviewed_input_bundle(
                    storage_root=self.storage.storage_root,
                    scene_id=scene_id,
                    source_video_path=self._windows_access_path(source_path),
                    source_video_sha256=source_asset.sha256,
                    source_start_sec=float(event.start_sec),
                    source_end_sec=float(event.end_sec),
                    source_fps=source_fps,
                )
            except ReviewedInputRecoveryError as exc:
                raise CandidatePreparationError(exc.code, str(exc)) from exc
            donor_video_path = self._windows_access_path(recovered.scene_video_path)
            reviewed_path = self._windows_access_path(recovered.reviewed_shots_path)
            detections_path = self._windows_access_path(recovered.detections_path)
            video_sha = recovered.scene_video_sha256
            duration_sec = recovered.duration_sec
            scene_fps = recovered.fps
            scene_width = recovered.width
            scene_height = recovered.height
            reviewed_metadata = {
                "review_state": "REVIEWED_PASS",
                "status": "RECOVERED_VERIFIED",
                "scene_id": scene_id,
                "source_video_sha256": video_sha,
            }
            detection_metadata = {
                "status": "RECOVERED_VERIFIED",
                "scene_id": scene_id,
                "source_video_sha256": video_sha,
                "detections_sha256": recovered.detections_sha256,
            }
            recovery_metadata = {
                "input_resolution_mode": "VERIFIED_STORAGE_RECOVERY",
                "recovered_from_storage_root": recovered.source_root.relative_to(
                    self.storage.project_root
                ).as_posix(),
                "source_match_similarity": recovered.source_match_similarity,
                "reviewed_shot_count": recovered.shot_count,
                "automatic_review_approval": False,
            }

        root = (
            self.storage.storage_root
            / "candidate_pipeline_inputs"
            / project.project_id
            / revision.revision_id
            / scene_id
        ).resolve()
        video_path = root / "scene.mp4"
        shots_path = root / "reviewed_shots.json"
        frozen_path = root / "detections.csv"
        self._copy_immutable(donor_video_path, video_path)
        self._copy_immutable(reviewed_path, shots_path)
        self._copy_immutable(detections_path, frozen_path)
        scene_video = self.media.create(
            match_id=project.match_id,
            asset_type="HIGHLIGHT_SCENE_CLIP",
            file_path=video_path.relative_to(self.storage.project_root).as_posix(),
            original_filename=video_path.name,
            mime_type="video/mp4",
            duration_sec=duration_sec,
            fps=scene_fps,
            width=scene_width,
            height=scene_height,
            size_bytes=video_path.stat().st_size,
            sha256=video_sha,
        )
        reviewed_artifact_metadata = {
            **reviewed_metadata,
            "revision_id": revision.revision_id,
            "event_id": event.timeline_event_id,
            "scene_id": scene_id,
            "sha256": sha256_file(self._windows_access_path(shots_path)),
            "source_video_sha256": video_sha,
            "action_spotting_source_video_sha256": source_asset.sha256,
            "event_source_start_sec": float(event.start_sec),
            "event_source_end_sec": float(event.end_sec),
            **recovery_metadata,
        }
        if materialized_from_artifact_id:
            reviewed_artifact_metadata["materialized_from_artifact_id"] = (
                materialized_from_artifact_id
            )
        reviewed = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type=REVIEWED_SHOTS_ARTIFACT_TYPE,
            file_path=shots_path.relative_to(self.storage.project_root).as_posix(),
            mime_type="application/json",
            metadata_=reviewed_artifact_metadata,
        )
        detection_artifact_metadata = {
            **detection_metadata,
            "revision_id": revision.revision_id,
            "event_id": event.timeline_event_id,
            "scene_id": scene_id,
            "sha256": sha256_file(self._windows_access_path(frozen_path)),
            "source_video_sha256": video_sha,
            "action_spotting_source_video_sha256": source_asset.sha256,
            "event_source_start_sec": float(event.start_sec),
            "event_source_end_sec": float(event.end_sec),
            "detections_sha256": sha256_file(self._windows_access_path(frozen_path)),
            **recovery_metadata,
        }
        if materialized_from_detections_artifact_id:
            detection_artifact_metadata["materialized_from_artifact_id"] = (
                materialized_from_detections_artifact_id
            )
        detections = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=revision.action_spotting_job_id,
            artifact_type="FROZEN_SCENE_DETECTIONS",
            file_path=frozen_path.relative_to(self.storage.project_root).as_posix(),
            mime_type="text/csv",
            metadata_=detection_artifact_metadata,
        )
        revision.options = {
            **(revision.options or {}),
            "candidate_pipeline_inputs": {
                "status": "MATERIALIZED",
                "scene_id": scene_id,
                "scene_video_asset_id": scene_video.asset_id,
                "reviewed_shots_artifact_id": reviewed.artifact_id,
                "shot_boundaries_artifact_id": reviewed.artifact_id,
                "shot_boundaries_sha256": reviewed_artifact_metadata["sha256"],
                "boundary_origin": "HUMAN_REVIEWED",
                "human_reviewed": True,
                "detections_artifact_id": detections.artifact_id,
                "source_video_sha256": video_sha,
                "action_spotting_source_video_sha256": source_asset.sha256,
                "event_source_start_sec": float(event.start_sec),
                "event_source_end_sec": float(event.end_sec),
                "source_start_frame": round(float(event.start_sec) * source_fps),
                "source_frame_mapping_basis": (
                    "timeline_event_start_sec_x_source_asset_fps"
                ),
                "candidate_source_to_video_frame_offset": 0,
                "candidate_video_frame_mapping_basis": (
                    "scene_candidate_frame_index_is_scene_video_frame_index"
                ),
                "automatic_target_confirmation": False,
                **recovery_metadata,
            },
        }
        self.db.commit()
        return scene_video, reviewed, detections

    def _ensure_upstream(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        shortlist_size: int,
    ) -> None:
        revision = self.db.get(HighlightRevision, revision_id)
        mapping = (
            (revision.options or {}).get("candidate_pipeline_inputs") or {}
            if revision is not None
            else {}
        )
        reviewed = self.db.get(
            Artifact,
            str(
                mapping.get("shot_boundaries_artifact_id")
                or mapping.get("reviewed_shots_artifact_id")
                or ""
            ),
        )
        reviewed_sha = (
            str((reviewed.metadata_ or {}).get("sha256") or "")
            if reviewed is not None
            else ""
        )
        if self._has_source(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            reviewed_shot_boundaries_sha256=reviewed_sha or None,
        ):
            event = self.db.get(TimelineEvent, event_id)
            if (
                revision is not None
                and event is not None
                and (
                    mapping.get("source_start_frame") is None
                    or mapping.get("candidate_source_to_video_frame_offset") is None
                )
                and mapping.get("scene_id") == scene_id
            ):
                self._materialize_candidate_inputs(
                    project=project,
                    revision=revision,
                    event=event,
                    scene_id=scene_id,
                )
            return
        event = self.db.get(TimelineEvent, event_id)
        if (
            revision is None
            or revision.project_id != project.project_id
            or event is None
            or event.match_id != project.match_id
            or scene_id not in (revision.selected_scene_ids or [])
            or event.source_job_id != revision.action_spotting_job_id
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The selected revision, event, scene and Action Spotting source do not match.",
            )
        discovery = (revision.options or {}).get("scene_target_selection") or {}
        if (
            discovery.get("scene_id") != scene_id
            or not discovery.get("discovery_id")
            or (discovery.get("discovery_inputs") or {}).get("shot_boundaries_sha256")
            != reviewed_sha
        ):
            scene_video, reviewed, detections = self._materialize_candidate_inputs(
                project=project,
                revision=revision,
                event=event,
                scene_id=scene_id,
            )
            if not self._materialize_discovery_snapshot(
                project=project,
                revision=revision,
                scene_id=scene_id,
                scene_video=scene_video,
                reviewed=reviewed,
                detections=detections,
            ):
                settings = get_settings()
                discovery_root = (
                    settings.SCENE_DISCOVERY_PROJECT_ROOT
                    or settings.SCENE_TARGET_SELECTION_PROJECT_ROOT
                )
                discovery_settings = settings.model_copy(
                    update={
                        "SCENE_TARGET_SELECTION_PROJECT_ROOT": discovery_root,
                        "SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE": (
                            settings.SCENE_DISCOVERY_PYTHON_EXECUTABLE
                            or settings.SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE
                        ),
                        "SCENE_TARGET_SELECTION_SCRIPT_PATH": (
                            settings.SCENE_DISCOVERY_SCRIPT_PATH
                            or settings.SCENE_TARGET_SELECTION_SCRIPT_PATH
                        ),
                        "SCENE_TARGET_SELECTION_PROCESS_TIMEOUT_SECONDS": (
                            settings.SCENE_DISCOVERY_PROCESS_TIMEOUT_SECONDS
                        ),
                    }
                )
                _CandidatePipelineSceneTargetSelectionService(
                    self.db, settings=discovery_settings
                ).discover(
                    project=project,
                    revision_id=revision_id,
                    scene_id=scene_id,
                    scene_video=scene_video,
                    shot_boundaries_artifact=reviewed,
                    detections_artifact=detections,
                )
        adapter = EventCandidateRankingV112aBackendAdapter(self.db)
        resolved, freeze_material = adapter.prepare(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            shortlist_size=shortlist_size,
        )
        result = adapter.run(
            project=project,
            user=user,
            revision_id=revision_id,
            shortlist_size=shortlist_size,
            resolved_event=resolved.to_dict(),
            freeze_material=freeze_material,
        )
        artifact = self.db.get(Artifact, result["artifact_id"])
        if artifact is None:
            raise CandidatePreparationError(
                "V1_1_2A_RANKING_NOT_READY",
                "V1.1.2a completed without registering its immutable artifact.",
            )
        artifact.metadata_ = {
            **(artifact.metadata_ or {}),
            "freeze_material": freeze_material,
        }
        self.db.commit()

    def _materialize_discovery_snapshot(
        self,
        *,
        project: Project,
        revision: HighlightRevision,
        scene_id: str,
        scene_video: MediaAsset,
        reviewed: Artifact,
        detections: Artifact,
    ) -> bool:
        donor_reviewed_id = str(
            (reviewed.metadata_ or {}).get("materialized_from_artifact_id") or ""
        )
        donor_reviewed = self.db.get(Artifact, donor_reviewed_id)
        donor_revision = self.db.get(
            HighlightRevision,
            str((donor_reviewed.metadata_ or {}).get("revision_id") or "")
            if donor_reviewed is not None
            else "",
        )
        donor = (
            (donor_revision.options or {}).get("scene_target_selection") or {}
            if donor_revision is not None
            else {}
        )
        inputs = donor.get("discovery_inputs") or {}
        expected = {
            "video_sha256": scene_video.sha256,
            "shot_boundaries_sha256": (reviewed.metadata_ or {}).get("sha256"),
            "detections_sha256": (detections.metadata_ or {}).get("sha256"),
        }
        if (
            donor.get("scene_id") != scene_id
            or not donor.get("discovery_id")
            or any(inputs.get(key) != value for key, value in expected.items())
        ):
            return False
        try:
            donor_root = self.storage.resolve_path(donor["artifact_root"])
            candidates_source = self._windows_access_path(
                donor_root / "scene_candidates.json"
            )
            manifest_source = self._windows_access_path(
                donor_root / "scene_candidate_manifest.json"
            )
        except (KeyError, ValueError):
            return False
        if (
            not candidates_source.is_file()
            or not manifest_source.is_file()
            or sha256_file(candidates_source) != donor.get("scene_candidates_sha256")
            or sha256_file(manifest_source)
            != donor.get("scene_candidate_manifest_sha256")
        ):
            return False
        target_root = (
            self.storage.storage_root
            / "candidate_snapshots"
            / project.project_id
            / revision.revision_id
            / scene_id
            / str(donor["discovery_id"])
        ).resolve()
        candidates_target = target_root / "scene_candidates.json"
        manifest_target = target_root / "scene_candidate_manifest.json"
        self._copy_immutable(candidates_source, candidates_target)
        self._copy_immutable(manifest_source, manifest_target)
        revision.options = {
            **(revision.options or {}),
            "scene_target_selection": {
                **donor,
                "status": "WAITING_TARGET_SELECTION",
                "scene_id": scene_id,
                "scene_video_asset_id": scene_video.asset_id,
                "shot_boundaries_artifact_id": reviewed.artifact_id,
                "detections_artifact_id": detections.artifact_id,
                "artifact_root": target_root.relative_to(
                    self.storage.project_root
                ).as_posix(),
                "scene_candidates_relative_path": candidates_target.relative_to(
                    self.storage.project_root
                ).as_posix(),
                "runtime_reused": True,
                "materialized_from_revision_id": donor_revision.revision_id,
                "automatic_target_confirmation": False,
            },
        }
        revision.status = "PLAYER_SELECTION_REQUIRED"
        revision.pending_action = "SELECT_PLAYER"
        self.db.commit()
        return True

    def _latest_shot_boundaries(
        self,
        *,
        project: Project,
        revision_id: str,
        scene_id: str,
        expected_artifact_id: str | None = None,
    ) -> tuple[Artifact, Path]:
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type.in_(
                    {AUTO_SHOTS_ARTIFACT_TYPE, REVIEWED_SHOTS_ARTIFACT_TYPE}
                ),
            )
        ).all()
        expected = {"revision_id": revision_id, "scene_id": scene_id}
        matches = []
        for row in rows:
            metadata = row.metadata_ or {}
            state = str(metadata.get("review_state") or metadata.get("status") or "")
            usable_state = (
                row.artifact_type == AUTO_SHOTS_ARTIFACT_TYPE
                and state == "STRUCTURALLY_VALID"
                and metadata.get("boundary_origin") == "AUTO_DETECTED"
                and metadata.get("human_reviewed") is False
                and metadata.get("automatic_target_confirmation") is False
            ) or (
                row.artifact_type == REVIEWED_SHOTS_ARTIFACT_TYPE
                and state in USABLE_REVIEW_STATES
            )
            if (
                row.match_id == project.match_id
                and self._metadata_matches(row, expected)
                and usable_state
                and (
                    expected_artifact_id is None
                    or row.artifact_id == expected_artifact_id
                )
            ):
                matches.append(row)
        if not matches:
            raise CandidatePreparationError(
                "SHOT_BOUNDARY_DISCOVERY_INPUT_NOT_READY",
                "Structurally valid shot boundaries for this revision and scene are not ready.",
            )
        matches.sort(key=lambda row: row.created_at, reverse=True)
        shots = matches[0]
        return shots, self._verified_artifact_path(
            shots, code="SHOT_BOUNDARY_DISCOVERY_INPUT_NOT_READY"
        )

    def _latest_reviewed_shots(
        self,
        *,
        project: Project,
        revision_id: str,
        scene_id: str,
        expected_artifact_id: str | None = None,
    ) -> tuple[Artifact, Path]:
        """Compatibility alias for R12-R14 callers and tests."""
        try:
            return self._latest_shot_boundaries(
                project=project,
                revision_id=revision_id,
                scene_id=scene_id,
                expected_artifact_id=expected_artifact_id,
            )
        except CandidatePreparationError as exc:
            if exc.code != "SHOT_BOUNDARY_DISCOVERY_INPUT_NOT_READY":
                raise
            raise CandidatePreparationError(
                "REVIEWED_SHOT_BOUNDARIES_NOT_READY", str(exc)
            ) from exc

    def _freeze_material(self, source: Artifact) -> dict[str, Any]:
        direct = (source.metadata_ or {}).get("freeze_material")
        if isinstance(direct, dict):
            return dict(direct)
        tasks = self.db.scalars(
            select(SceneAITask).where(
                SceneAITask.project_id == source.project_id,
                SceneAITask.task_type == "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW",
                SceneAITask.status == "COMPLETED",
            )
        ).all()
        matches = [
            task
            for task in tasks
            if (task.result or {}).get("artifact_id") == source.artifact_id
            and isinstance((task.payload or {}).get("freeze_material"), dict)
        ]
        if not matches:
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The V1.1.2a immutable discovery contract is unavailable.",
            )
        matches.sort(key=lambda task: task.completed_at, reverse=True)
        return dict(matches[0].payload["freeze_material"])

    def _immutable_input(
        self,
        source: Artifact,
    ) -> tuple[ImmutableCandidateInput, dict[str, Any]]:
        material = self._freeze_material(source)
        required = (
            "storage_root",
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
            "discovery_id",
            "discovery_artifact_root",
        )
        missing = [key for key in required if material.get(key) in (None, "")]
        if missing:
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The V1.1.2a freeze contract is incomplete: " + ", ".join(missing),
            )
        root = Path(str(material["storage_root"])).resolve()
        if self._logical_windows_path(root) != self.storage.storage_root:
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The V1.1.2a storage root no longer matches runtime storage.",
            )
        source_video = (root / str(material["source_video_relative_path"])).resolve()
        if (
            not source_video.is_relative_to(root)
            or not source_video.is_file()
            or sha256_file(source_video) != str(material["source_video_sha256"])
        ):
            raise CandidatePreparationError(
                "SOURCE_VIDEO_NOT_READY",
                "The immutable source video is missing or its SHA-256 changed.",
            )
        try:
            immutable = ImmutableCandidateInput.load(
                discovery_root=root,
                **{
                    key: material[key]
                    for key in required
                    if key
                    not in {
                        "storage_root",
                        "discovery_id",
                        "discovery_artifact_root",
                    }
                },
            )
        except ValueError as exc:
            message = str(exc)
            code = (
                "SOURCE_VIDEO_NOT_READY"
                if "video" in message.lower()
                else "INPUT_PROVENANCE_MISMATCH"
            )
            raise CandidatePreparationError(code, message) from exc
        manifest_path = self._windows_access_path(
            (
                self.storage.resolve_path(material["discovery_artifact_root"])
                / "scene_candidate_manifest.json"
            ).resolve()
        )
        if not manifest_path.is_file() or sha256_file(manifest_path) != str(
            material["candidate_manifest_sha256"]
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The frozen scene candidate manifest is missing or changed.",
            )
        return immutable, material

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH", "Expected a JSON object artifact."
            )
        return value

    @staticmethod
    def _bundle_candidate(
        raw: dict[str, Any],
        *,
        discovery_id: str,
        frame_offset_contract: int | None = None,
        video_frame_offset_contract: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        source_id = str(raw.get("candidate_id") or "")
        shot_id = str(raw.get("shot_id") or "")
        tracklet_id = str(raw.get("local_tracklet_id") or raw.get("tracklet_id") or "")
        observations = list(raw.get("observations") or [])
        if not source_id or not shot_id or not tracklet_id or not observations:
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "A shortlisted discovery candidate has incomplete identity data.",
            )
        mapped: list[dict[str, Any]] = []
        offsets: set[int] = set()
        for observation in observations:
            global_value = next(
                (
                    observation[key]
                    for key in (
                        "global_frame",
                        "global_frame_index",
                        "source_frame_index",
                    )
                    if observation.get(key) is not None
                ),
                None,
            )
            local_value = next(
                (
                    observation[key]
                    for key in (
                        "scene_local_frame",
                        "scene_local_frame_index",
                        "frame_index",
                    )
                    if observation.get(key) is not None
                ),
                None,
            )
            if local_value is None:
                raise CandidatePreparationError(
                    "FRAME_MAPPING_NOT_READY",
                    "Candidate observations do not contain an explicit frame mapping.",
                )
            local_frame = int(local_value)
            if global_value is None:
                if frame_offset_contract is None:
                    raise CandidatePreparationError(
                        "FRAME_MAPPING_NOT_READY",
                        "The immutable discovery contract has no source frame offset.",
                    )
                global_frame = local_frame + frame_offset_contract
            else:
                global_frame = int(global_value)
            offsets.add(global_frame - local_frame)
            mapped.append(
                {
                    "frame_index": local_frame,
                    "bbox_xyxy": observation.get("bbox_xyxy")
                    or observation.get("bbox"),
                    "confidence": (
                        observation.get("detector_confidence")
                        if observation.get("detector_confidence") is not None
                        else observation.get("confidence", 0.0)
                    ),
                }
            )
        if len(offsets) != 1:
            raise CandidatePreparationError(
                "FRAME_MAPPING_NOT_READY",
                "Candidate observation frame offsets are inconsistent.",
            )
        resolved_video_offset = (
            int(video_frame_offset_contract)
            if video_frame_offset_contract is not None
            else offsets.pop()
        )
        return (
            {
                "candidate_id": public_candidate_id(source_id),
                "source_candidate_id": source_id,
                "shot_id": shot_id,
                "local_tracklet_id": tracklet_id,
                "tracklet_id": tracklet_id,
                "observations": mapped,
                "discovery_id": discovery_id,
            },
            resolved_video_offset,
        )

    def _existing_bundle(
        self,
        *,
        project: Project,
        contract: dict[str, str],
    ) -> Artifact | None:
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == BUNDLE_ARTIFACT_TYPE,
            )
        ).all()
        matches = [row for row in rows if self._metadata_matches(row, contract)]
        if not matches:
            return None
        matches.sort(key=lambda row: row.created_at, reverse=True)
        existing = matches[0]
        path = self._verified_artifact_path(
            existing, code="REVIEW_BUNDLE_ARTIFACT_INVALID"
        )
        manifest = self._load_json(path)
        if any(manifest.get(key) != value for key, value in contract.items()):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "Existing candidate review bundle provenance does not match.",
            )
        return existing

    def _verify_bundle_directory(
        self,
        root: Path,
        *,
        contract: dict[str, str],
    ) -> tuple[dict[str, Any], str]:
        manifest_path = (root / "manifest.json").resolve()
        if (
            not manifest_path.is_relative_to(root.resolve())
            or not manifest_path.is_file()
        ):
            raise CandidatePreparationError(
                "REVIEW_BUNDLE_ARTIFACT_INVALID",
                "Candidate review bundle manifest is missing.",
            )
        manifest = self._load_json(manifest_path)
        if manifest.get("automatic_target_confirmation") is not False or any(
            manifest.get(key) != value for key, value in contract.items()
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "Candidate review bundle provenance does not match.",
            )
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise CandidatePreparationError(
                "REVIEW_BUNDLE_ARTIFACT_INVALID",
                "Candidate review bundle file manifest is missing.",
            )
        for record in files.values():
            if not isinstance(record, dict):
                raise CandidatePreparationError(
                    "REVIEW_BUNDLE_ARTIFACT_INVALID",
                    "Candidate review bundle file record is invalid.",
                )
            path = (root / str(record.get("path") or "")).resolve()
            expected = str(record.get("sha256") or "")
            if (
                not path.is_relative_to(root.resolve())
                or not path.is_file()
                or len(expected) != 64
                or sha256_file(path) != expected
            ):
                raise CandidatePreparationError(
                    "REVIEW_BUNDLE_ARTIFACT_INVALID",
                    "Candidate review bundle media SHA-256 validation failed.",
                )
        return manifest, sha256_file(manifest_path)

    def prepare(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        event_id: str,
        scene_id: str,
        shortlist_size: int,
        shot_boundaries_artifact_id: str | None = None,
        shot_boundaries_sha256: str | None = None,
        detections_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        revision = self.db.get(HighlightRevision, revision_id)
        mapping = (
            (revision.options or {}).get("candidate_pipeline_inputs") or {}
            if revision is not None
            else {}
        )
        if shot_boundaries_artifact_id is not None and (
            (
                mapping.get("shot_boundaries_artifact_id")
                or mapping.get("reviewed_shots_artifact_id")
            )
            != shot_boundaries_artifact_id
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The candidate shot-boundary artifact changed after task creation.",
            )
        if detections_artifact_id is not None and (
            mapping.get("detections_artifact_id") != detections_artifact_id
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The sampled detection artifact changed after task creation.",
            )
        self._ensure_upstream(
            project=project,
            user=user,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            shortlist_size=shortlist_size,
        )
        source, _ = self._latest_source(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        reviewed, reviewed_path = self._latest_shot_boundaries(
            project=project,
            revision_id=revision_id,
            scene_id=scene_id,
            expected_artifact_id=shot_boundaries_artifact_id,
        )
        immutable, material = self._immutable_input(source)
        reviewed_sha = sha256_file(reviewed_path)
        shot_boundary_artifact_type = reviewed.artifact_type
        human_reviewed = shot_boundary_artifact_type == REVIEWED_SHOTS_ARTIFACT_TYPE
        boundary_origin = "HUMAN_REVIEWED" if human_reviewed else "AUTO_DETECTED"
        if (
            shot_boundaries_sha256 is not None
            and reviewed_sha != shot_boundaries_sha256
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "The candidate shot-boundary SHA-256 changed after task creation.",
            )
        ranking_artifact = EventCandidateRankingV12BackendAdapter(self.db).run(
            project=project,
            user=user,
            source_ranking_artifact_id=source.artifact_id,
            shot_boundaries_artifact_id=reviewed.artifact_id,
            shortlist_size=shortlist_size,
        )
        ranking_metadata = ranking_artifact.metadata_ or {}
        required_metadata = {
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "source_ranking_artifact_id": source.artifact_id,
            "reviewed_shots_artifact_id": reviewed.artifact_id,
            "automatic_target_confirmation": False,
        }
        if (
            any(
                ranking_metadata.get(key) != value
                for key, value in required_metadata.items()
            )
            or len(str(ranking_metadata.get("cache_key") or "")) != 64
            or len(str(ranking_metadata.get("sha256") or "")) != 64
        ):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "Generated V1.2 ranking provenance is incomplete.",
            )
        ranking_path = self._verified_artifact_path(
            ranking_artifact, code="V1_2_RANKING_ARTIFACT_INVALID"
        )
        ranking = self._load_json(ranking_path)
        if ranking.get("automatic_target_confirmation") is not False:
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "V1.2 attempted automatic target confirmation.",
            )
        source_document = self._load_json(immutable.scene_candidates_path)
        source_candidates = {
            str(row.get("candidate_id") or ""): row
            for row in source_document.get("candidates") or []
        }
        source_ranking_id = str((source.metadata_ or {}).get("ranking_id") or "")
        patch_id = str(ranking_metadata.get("ranking_id") or "")
        if not source_ranking_id or not patch_id:
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH", "Ranking identifiers are missing."
            )
        bundle_count = 0

        work_metrics = CandidatePreparationWorkMetrics()

        shared_frame_cache = SharedFrameCache(
            self.storage.storage_root / "shared_frame_cache_r1"
        )

        shortlist_rows = list(ranking.get("shortlist") or [])
        work_metrics.increment(
            "shortlisted_candidate_count",
            len(shortlist_rows),
        )

        revision = self.db.get(
            HighlightRevision,
            revision_id,
        )
        mapping = (
            (revision.options or {}).get("candidate_pipeline_inputs") or {}
            if revision is not None
            else {}
        )
        frame_offset_contract = mapping.get("source_start_frame")
        video_frame_offset_contract = mapping.get(
            "candidate_source_to_video_frame_offset"
        )

        prepared_by_candidate_id: dict[str, dict[str, Any]] = {}
        grouping_inputs: list[dict[str, Any]] = []
        for shortlist_rank, shortlist_row in enumerate(
            shortlist_rows,
            start=1,
        ):
            source_candidate_id = str(shortlist_row.get("candidate_id") or "")
            raw = source_candidates.get(source_candidate_id)
            if raw is None:
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    (
                        "Shortlisted candidate is absent from frozen "
                        f"discovery: {source_candidate_id}"
                    ),
                )
            candidate, frame_offset = self._bundle_candidate(
                raw,
                discovery_id=str(material["discovery_id"]),
                frame_offset_contract=(
                    int(frame_offset_contract)
                    if frame_offset_contract is not None
                    else None
                ),
                video_frame_offset_contract=(
                    int(video_frame_offset_contract)
                    if video_frame_offset_contract is not None
                    else None
                ),
            )
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in prepared_by_candidate_id:
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Public candidate IDs are not unique.",
                )
            global_rank = int(
                shortlist_row.get(
                    "original_global_rank",
                    shortlist_row.get("rank"),
                )
            )
            prepared_by_candidate_id[candidate_id] = {
                "candidate": candidate,
                "frame_offset": frame_offset,
                "shortlist_row": shortlist_row,
                "shortlist_rank": shortlist_rank,
                "global_rank": global_rank,
            }
            grouping_inputs.append(
                {
                    **candidate,
                    "frame_offset": frame_offset,
                    "shortlist_rank": shortlist_rank,
                    "global_rank": global_rank,
                }
            )

        with work_metrics.stage("candidate_fragment_grouping"):
            grouping_document = build_candidate_grouping(
                video_path=immutable.source_video_path,
                source_video_sha256=immutable.source_video_sha256,
                shortlist_patch_id=patch_id,
                ranking_id=source_ranking_id,
                candidates=grouping_inputs,
                shared_frame_cache=shared_frame_cache,
                metrics=work_metrics,
            )

        grouping_document = {
            **grouping_document,
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "candidate_manifest_sha256": (immutable.candidate_manifest_sha256),
            "shot_boundaries_artifact_id": reviewed.artifact_id,
            "shot_boundaries_sha256": reviewed_sha,
            "shot_boundary_artifact_type": shot_boundary_artifact_type,
            "boundary_origin": boundary_origin,
            "human_reviewed": human_reviewed,
            "media_materialization_policy": (MEDIA_MATERIALIZATION_POLICY),
        }
        grouping_fingerprint = canonical_sha256(grouping_document)
        grouping_root = (
            self.storage.storage_root
            / "event_candidate_groupings_r1"
            / patch_id
            / grouping_fingerprint[:24]
        ).resolve()
        if not grouping_root.is_relative_to(self.storage.storage_root):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                "Candidate grouping path escapes immutable storage.",
            )
        grouping_path = grouping_root / "candidate_grouping.json"
        if grouping_path.exists():
            existing_grouping_document = self._load_json(grouping_path)
            if canonical_sha256(existing_grouping_document) != grouping_fingerprint:
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Existing candidate grouping content differs.",
                )
            grouping_sha256 = sha256_file(grouping_path)
        else:
            grouping_sha256 = write_json_atomic(
                grouping_path,
                grouping_document,
            )

        grouping_artifact_contract = {
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "shortlist_patch_id": patch_id,
            "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
            "source_video_sha256": immutable.source_video_sha256,
            "candidate_manifest_sha256": (immutable.candidate_manifest_sha256),
            "shot_boundaries_artifact_id": reviewed.artifact_id,
            "shot_boundaries_sha256": reviewed_sha,
            "shot_boundary_artifact_type": shot_boundary_artifact_type,
            "boundary_origin": boundary_origin,
            "human_reviewed": human_reviewed,
            "sha256": grouping_sha256,
        }
        grouping_rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == GROUPING_ARTIFACT_TYPE,
            )
        ).all()
        existing_grouping_artifact = next(
            (
                row
                for row in grouping_rows
                if row.match_id == project.match_id
                and self._metadata_matches(
                    row,
                    grouping_artifact_contract,
                )
            ),
            None,
        )
        if existing_grouping_artifact is None:
            self.artifacts.create(
                match_id=project.match_id,
                project_id=project.project_id,
                analysis_job_id=None,
                artifact_type=GROUPING_ARTIFACT_TYPE,
                file_path=grouping_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                mime_type="application/json",
                metadata_={
                    **grouping_artifact_contract,
                    "ranking_id": source_ranking_id,
                    "source_candidate_count": grouping_document[
                        "source_candidate_count"
                    ],
                    "display_candidate_count": grouping_document[
                        "display_candidate_count"
                    ],
                    "status": "READY",
                    "owner_id": user.user_id,
                    "automatic_target_confirmation": False,
                },
            )
            self.db.commit()
            work_metrics.increment("candidate_grouping_artifact_records_created")
        else:
            work_metrics.increment("candidate_grouping_artifact_cache_hits")

        for group in grouping_document["groups"]:
            representative_candidate_id = str(group["representative_candidate_id"])
            prepared = prepared_by_candidate_id.get(representative_candidate_id)
            if prepared is None:
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Candidate group representative is not shortlisted.",
                )

            member_candidate_ids = [
                str(value) for value in group["member_candidate_ids"]
            ]
            member_source_candidate_ids = [
                str(value) for value in group["member_source_candidate_ids"]
            ]
            group_member_fingerprint = canonical_sha256(
                {
                    "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
                    "member_candidate_ids": member_candidate_ids,
                }
            )
            candidate = {
                **prepared["candidate"],
                "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
                "candidate_grouping_sha256": grouping_sha256,
                "candidate_group_id": group["candidate_group_id"],
                "group_member_candidate_ids": member_candidate_ids,
                "group_member_source_candidate_ids": (member_source_candidate_ids),
                "group_member_fingerprint": group_member_fingerprint,
                "grouping_reason_codes": list(group.get("grouping_reason_codes") or []),
                "grouping_evidence": list(group.get("evidence") or []),
                "possible_fragment_duplicate": bool(
                    group.get("possible_fragment_duplicate", False)
                ),
            }
            frame_offset = int(prepared["frame_offset"])
            contract = {
                "shortlist_patch_id": patch_id,
                "candidate_id": representative_candidate_id,
                "source_video_sha256": immutable.source_video_sha256,
                "candidate_manifest_sha256": (immutable.candidate_manifest_sha256),
                "shot_boundaries_artifact_id": reviewed.artifact_id,
                "shot_boundaries_sha256": reviewed_sha,
                "shot_boundary_artifact_type": shot_boundary_artifact_type,
                "boundary_origin": boundary_origin,
                "human_reviewed": human_reviewed,
                "media_materialization_policy": (MEDIA_MATERIALIZATION_POLICY),
                "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
                "candidate_grouping_sha256": grouping_sha256,
                "candidate_group_id": str(group["candidate_group_id"]),
                "group_member_fingerprint": group_member_fingerprint,
            }
            existing = self._existing_bundle(
                project=project,
                contract=contract,
            )
            if existing is not None:
                work_metrics.increment("review_bundle_cache_hits")
                bundle_count += 1
                continue
            bundle_key = canonical_sha256(contract)
            output_root = (
                self.storage.storage_root
                / "event_candidate_review_bundles_r1"
                / patch_id
                / representative_candidate_id
                / bundle_key[:24]
            ).resolve()
            if not output_root.is_relative_to(self.storage.storage_root):
                raise CandidatePreparationError(
                    "INPUT_PROVENANCE_MISMATCH",
                    "Candidate review bundle path escapes immutable storage.",
                )
            if output_root.exists():
                work_metrics.increment("review_bundle_directory_reuse_count")
                with work_metrics.stage("review_bundle_integrity_verification"):
                    manifest, manifest_sha = self._verify_bundle_directory(
                        output_root,
                        contract=contract,
                    )
            else:
                staging_root = (
                    output_root.parent / f".tmp_{generate_prefixed_id('bld')}"
                ).resolve()
                try:
                    with work_metrics.stage("review_bundle_build"):
                        build_candidate_review_bundle(
                            video_path=immutable.source_video_path,
                            candidate=candidate,
                            output_root=staging_root,
                            ranking_id=source_ranking_id,
                            shortlist_patch_id=patch_id,
                            candidate_manifest_sha256=(
                                immutable.candidate_manifest_sha256
                            ),
                            source_video_sha256=(immutable.source_video_sha256),
                            shot_boundaries_artifact_id=reviewed.artifact_id,
                            shot_boundaries_sha256=reviewed_sha,
                            shot_boundary_artifact_type=(shot_boundary_artifact_type),
                            boundary_origin=boundary_origin,
                            human_reviewed=human_reviewed,
                            frame_offset=frame_offset,
                            shared_frame_cache=shared_frame_cache,
                            metrics=work_metrics,
                        )
                    with work_metrics.stage("review_bundle_integrity_verification"):
                        manifest, manifest_sha = self._verify_bundle_directory(
                            staging_root,
                            contract=contract,
                        )
                    staging_root.rename(output_root)
                except Exception:
                    if staging_root.exists():
                        shutil.rmtree(
                            self._windows_access_path(staging_root),
                            ignore_errors=True,
                        )
                    raise
            self.artifacts.create(
                match_id=project.match_id,
                project_id=project.project_id,
                analysis_job_id=None,
                artifact_type=BUNDLE_ARTIFACT_TYPE,
                file_path=(output_root / "manifest.json")
                .relative_to(self.storage.project_root)
                .as_posix(),
                mime_type="application/json",
                metadata_={
                    **contract,
                    "ranking_id": source_ranking_id,
                    "revision_id": revision_id,
                    "event_id": event_id,
                    "scene_id": scene_id,
                    "shot_id": candidate["shot_id"],
                    "tracklet_id": candidate["tracklet_id"],
                    "source_candidate_id": candidate["source_candidate_id"],
                    "group_member_candidate_ids": member_candidate_ids,
                    "group_member_source_candidate_ids": (member_source_candidate_ids),
                    "possible_fragment_duplicate": bool(
                        group.get("possible_fragment_duplicate", False)
                    ),
                    "status": "READY",
                    "sha256": manifest_sha,
                    "owner_id": user.user_id,
                    "automatic_target_confirmation": False,
                },
            )
            self.db.commit()
            work_metrics.increment("review_bundle_artifact_records_created")
            bundle_count += 1

        source_candidate_count = int(grouping_document["source_candidate_count"])
        candidate_count = int(grouping_document["display_candidate_count"])
        if bundle_count != candidate_count:
            raise CandidatePreparationError(
                "REVIEW_BUNDLES_INCOMPLETE",
                (
                    "Not all candidate group representatives have "
                    "immutable review bundles."
                ),
            )
        work_metrics.increment(
            "review_bundle_count",
            bundle_count,
        )

        metrics_path = (
            self.storage.storage_root
            / "candidate_preparation_metrics_r1"
            / patch_id
            / revision_id
            / event_id
            / f"{scene_id}.json"
        ).resolve()

        if not metrics_path.is_relative_to(self.storage.storage_root):
            raise CandidatePreparationError(
                "INPUT_PROVENANCE_MISMATCH",
                ("Candidate preparation metrics path escapes immutable storage."),
            )

        metrics_snapshot = work_metrics.snapshot()

        metrics_sha256 = write_json_atomic(
            metrics_path,
            metrics_snapshot,
        )

        recommendations_url = (
            f"/api/v1/projects/{project.project_id}/highlight/revisions/"
            f"{revision_id}/events/{event_id}/candidate-recommendations"
            f"?scene_id={scene_id}"
        )
        return {
            "ready": True,
            "ranking_version": "v1.2",
            "ranking_artifact_id": (ranking_artifact.artifact_id),
            "ranking_id": source_ranking_id,
            "shortlist_patch_id": patch_id,
            "source_candidate_count": source_candidate_count,
            "candidate_count": candidate_count,
            "display_candidate_count": candidate_count,
            "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
            "candidate_grouping_path": (
                grouping_path.relative_to(self.storage.project_root).as_posix()
            ),
            "candidate_grouping_sha256": grouping_sha256,
            "review_bundle_count": bundle_count,
            "work_metrics_path": (
                metrics_path.relative_to(self.storage.project_root).as_posix()
            ),
            "work_metrics_sha256": metrics_sha256,
            "work_metrics": metrics_snapshot,
            "recommendations_url": recommendations_url,
            "automatic_target_confirmation": False,
        }


def install_candidate_preparation_integration() -> None:
    """Add the preparation task without modifying frozen ranking runtimes."""
    global _integration_installed
    if _integration_installed:
        return
    from app.domains.highlight import scene_ai_task as scene_task_module

    original_dispatch = scene_task_module.SceneAITaskExecutor._dispatch

    def dispatch(self, db: Session, task: SceneAITask) -> dict[str, Any]:
        if task.task_type != PREPARATION_TASK_TYPE:
            return original_dispatch(self, db, task)
        payload = dict(task.payload or {})
        user = db.get(User, task.owner_id)
        project = db.get(Project, task.project_id)
        if user is None or project is None:
            raise ValueError("Scene AI task ownership context is missing.")
        return EventCandidateRecommendationPreparationService(db).prepare(
            project=project,
            user=user,
            revision_id=payload["revision_id"],
            event_id=payload["event_id"],
            scene_id=payload["scene_id"],
            shortlist_size=int(payload["shortlist_size"]),
            shot_boundaries_artifact_id=payload.get("shot_boundaries_artifact_id"),
            shot_boundaries_sha256=payload.get("shot_boundaries_sha256"),
            detections_artifact_id=payload.get("detections_artifact_id"),
        )

    scene_task_module.SceneAITaskExecutor._dispatch = dispatch
    _integration_installed = True
