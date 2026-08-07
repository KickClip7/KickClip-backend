from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1 import ANCHOR_MODE, RANKING_VERSION
from app.domains.highlight.model import HighlightRevision
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.execution import (
    R1_EXECUTION_KIND,
    R1PipelineStage,
    R1ProcessingStatus,
)
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.r1_executor import get_r1_tracking_executor
from app.domains.tracking.status import TERMINAL_STATUSES, TrackingBackendStatus
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .artifacts import (
    canonical_sha256,
    sha256_file,
    verify_manifest,
    write_json_atomic,
)
from .candidate_grouping import (
    CANDIDATE_GROUPING_POLICY_VERSION,
    CANDIDATE_GROUPING_SCHEMA_VERSION,
)
from .errors import (
    CandidateRecommendationNotPrepared,
    CandidateSelectionProvenanceMismatch,
    CandidateTrackingConfigurationInvalid,
    CandidateTrackingInputInvalid,
    CandidateTrackingRuntimeContractInvalid,
    InsufficientReviewableTargetReference,
    TrackletIdentityInconsistent,
)
from .lazy_media import (
    LazyCandidateMediaError,
    LazyCandidateMediaMaterializer,
)
from .model import (
    EventCandidateAmbiguityR1,
    EventCandidateMemoryRevisionR1,
    EventCandidateOutboxR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
    EventCandidateSelectionR1,
)
from .orchestrator import R1PipelineOrchestrator, recover_r1_pipelines
from .r3_adapter import R1R3AdapterError, R1R3InputAdapter
from .schema import (
    CandidateMediaRead,
    CandidateReviewDecisionRequest,
    CandidateReviewState,
    EventCandidateRecommendationRead,
    EventCandidateRecommendationResponse,
    EventCandidateSelectionRead,
    EventCandidateTrackingCreateResponse,
)

RANKING_ARTIFACT_TYPE = "EVENT_CANDIDATE_RANKING_V1_2_SHADOW_SHORTLIST_PATCH"
BUNDLE_ARTIFACT_TYPE = "EVENT_CANDIDATE_REVIEW_BUNDLE_R1_MANIFEST"
GROUPING_ARTIFACT_TYPE = "EVENT_CANDIDATE_GROUPING_R1"
SELECTION_ARTIFACT_TYPE = "EVENT_CANDIDATE_SELECTION_R1"
PROVENANCE_ARTIFACT_TYPE = "CANDIDATE_SELECTION_INTEGRATION_PROVENANCE_R1"
PUBLIC_ID_PATTERN = re.compile(r"(shot_\d{4}_track_\d{4})$")
AUTO_SHOT_BOUNDARIES = "AUTO_SHOT_BOUNDARIES"
REVIEWED_SHOT_BOUNDARIES = "REVIEWED_SHOT_BOUNDARIES"
SUPPORTED_TRACKING_BOUNDARY_ARTIFACT_TYPES = {
    AUTO_SHOT_BOUNDARIES,
    REVIEWED_SHOT_BOUNDARIES,
}


@dataclass(frozen=True)
class ShotBoundaryProvenance:
    artifact: Artifact
    path: Path
    sha256: str
    artifact_type: str
    boundary_origin: str
    human_reviewed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact.artifact_id,
            "artifact_type": self.artifact_type,
            "sha256": self.sha256,
            "boundary_origin": self.boundary_origin,
            "human_reviewed": self.human_reviewed,
            "automatic_target_confirmation": False,
        }


def public_candidate_id(candidate_id: str) -> str:
    match = PUBLIC_ID_PATTERN.search(candidate_id)
    return match.group(1) if match else candidate_id


class CandidateHandoffR1Service:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.lazy_media = LazyCandidateMediaMaterializer(
            storage_root=self.storage.storage_root
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object artifact.")
        return value

    def _artifact_path(self, artifact: Artifact) -> Path:
        path = self.storage.resolve_path(artifact.file_path)
        if len(str(path)) >= 248 and not str(path).startswith("\\\\?\\"):
            path = Path("\\\\?\\" + str(path))
        expected = str((artifact.metadata_ or {}).get("sha256") or "")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"Artifact SHA-256 mismatch: {artifact.artifact_id}")
        return path

    def _tracking_output_root(self, configured_path: str) -> Path:
        if not configured_path.strip():
            raise CandidateTrackingConfigurationInvalid(
                "TRACKING_OUTPUT_ROOT_MISSING",
                "TRACKING_OUTPUT_ROOT is required for the R1 runtime.",
            )
        raw_path = Path(configured_path).expanduser()
        if not raw_path.is_absolute():
            raise CandidateTrackingConfigurationInvalid(
                "TRACKING_OUTPUT_ROOT_NOT_ABSOLUTE",
                "TRACKING_OUTPUT_ROOT must be absolute.",
            )
        output_root = raw_path.resolve()
        if not output_root.is_relative_to(self.storage.storage_root):
            raise CandidateTrackingConfigurationInvalid(
                "TRACKING_OUTPUT_ROOT_OUTSIDE_STORAGE",
                "TRACKING_OUTPUT_ROOT must be inside STORAGE_ROOT so immutable "
                "R1 artifacts remain addressable by the backend.",
            )
        return output_root

    @staticmethod
    def _validate_boundary_rows(
        document: Mapping[str, Any],
        *,
        human_reviewed: bool,
    ) -> None:
        video = document.get("video") or {}
        if not isinstance(video, Mapping):
            raise CandidateSelectionProvenanceMismatch(
                "Shot-boundary video provenance is invalid."
            )
        try:
            frame_count = int(
                video.get("frame_count") or document.get("frame_count") or 0
            )
        except (TypeError, ValueError) as exc:
            raise CandidateSelectionProvenanceMismatch(
                "Shot-boundary frame count is invalid."
            ) from exc
        rows = document.get("shots")
        if frame_count <= 0 or not isinstance(rows, list) or not rows:
            raise CandidateSelectionProvenanceMismatch(
                "Shot boundaries do not contain complete frame coverage."
            )
        expected_start = 0
        seen_ids: set[str] = set()
        for expected_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise CandidateSelectionProvenanceMismatch(
                    "Shot-boundary rows must be objects."
                )
            try:
                shot_index = int(row.get("shot_index", -1))
                start = int(row.get("start_frame", -1))
                end = int(row.get("end_frame_inclusive", row.get("end_frame", -1)))
            except (TypeError, ValueError) as exc:
                raise CandidateSelectionProvenanceMismatch(
                    "Shot-boundary frame values are invalid."
                ) from exc
            shot_id = str(row.get("shot_id") or "")
            if (
                shot_index != expected_index
                or not shot_id
                or shot_id in seen_ids
                or start != expected_start
                or end < start
                or end >= frame_count
            ):
                raise CandidateSelectionProvenanceMismatch(
                    "Shot boundaries are not contiguous full-frame coverage."
                )
            if human_reviewed:
                review_state = str(
                    row.get("review_state")
                    or row.get("review_status")
                    or row.get("status")
                    or ""
                )
                if review_state not in {"PASS", "REVIEWED_PASS"}:
                    raise CandidateSelectionProvenanceMismatch(
                        "Human-reviewed shot boundaries contain an unreviewed shot."
                    )
            seen_ids.add(shot_id)
            expected_start = end + 1
        if expected_start != frame_count:
            raise CandidateSelectionProvenanceMismatch(
                "Shot boundaries do not cover the complete scene."
            )

    def _resolve_boundary_provenance(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
        source_video_sha256: str,
        manifest: Mapping[str, Any],
        selection_metadata: Mapping[str, Any] | None = None,
        legacy_selection_sha256: str | None = None,
    ) -> ShotBoundaryProvenance:
        selection_metadata = selection_metadata or {}
        nested = manifest.get("shot_boundaries") or {}
        if not isinstance(nested, Mapping):
            raise CandidateSelectionProvenanceMismatch(
                "Candidate shot-boundary provenance must be an object."
            )
        declared_sha_values = {
            str(value)
            for value in (
                nested.get("sha256"),
                manifest.get("shot_boundaries_sha256"),
                manifest.get("reviewed_shot_boundaries_sha256"),
                selection_metadata.get("shot_boundaries_sha256"),
                legacy_selection_sha256,
            )
            if value
        }
        if len(declared_sha_values) != 1:
            raise CandidateSelectionProvenanceMismatch(
                "Candidate shot-boundary SHA-256 provenance is missing or inconsistent."
            )
        expected_sha = next(iter(declared_sha_values))
        if len(expected_sha) != 64:
            raise CandidateSelectionProvenanceMismatch(
                "Candidate shot-boundary SHA-256 provenance is invalid."
            )

        declared_type = str(
            nested.get("artifact_type")
            or manifest.get("shot_boundary_artifact_type")
            or selection_metadata.get("shot_boundary_artifact_type")
            or ""
        )
        declared_artifact_id = str(
            nested.get("artifact_id")
            or manifest.get("shot_boundaries_artifact_id")
            or selection_metadata.get("shot_boundaries_artifact_id")
            or ""
        )
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type.in_(SUPPORTED_TRACKING_BOUNDARY_ARTIFACT_TYPES),
            )
        ).all()
        matches = []
        for artifact in rows:
            metadata = artifact.metadata_ or {}
            if (
                artifact.match_id == project.match_id
                and metadata.get("revision_id") == revision_id
                and metadata.get("scene_id") == scene_id
                and metadata.get("event_id") in {None, "", event_id}
                and metadata.get("sha256") == expected_sha
                and (not declared_type or artifact.artifact_type == declared_type)
                and (
                    not declared_artifact_id
                    or artifact.artifact_id == declared_artifact_id
                )
            ):
                matches.append(artifact)
        if len(matches) != 1:
            raise CandidateSelectionProvenanceMismatch(
                "The immutable candidate shot-boundary artifact is missing or ambiguous."
            )
        artifact = matches[0]
        metadata = artifact.metadata_ or {}
        path = self._artifact_path(artifact)
        document = self._load_json(path)
        artifact_type = artifact.artifact_type

        if document.get("automatic_target_confirmation") is not False:
            raise CandidateSelectionProvenanceMismatch(
                "Shot boundaries attempted automatic target confirmation."
            )
        if str((document.get("video") or {}).get("sha256") or "") != str(
            source_video_sha256
        ):
            raise CandidateSelectionProvenanceMismatch(
                "Shot boundaries target a different immutable scene video."
            )

        if artifact_type == AUTO_SHOT_BOUNDARIES:
            structural = document.get("structural_validation") or {}
            try:
                gap_count = int(structural.get("gap_count", -1))
                overlap_count = int(structural.get("overlap_count", -1))
            except (AttributeError, TypeError, ValueError) as exc:
                raise CandidateSelectionProvenanceMismatch(
                    "Automatic shot-boundary structural metrics are invalid."
                ) from exc
            if (
                metadata.get("status") != "STRUCTURALLY_VALID"
                or metadata.get("boundary_origin") != "AUTO_DETECTED"
                or metadata.get("human_reviewed") is not False
                or metadata.get("automatic_target_confirmation") is not False
                or document.get("artifact_type") != AUTO_SHOT_BOUNDARIES
                or document.get("boundary_origin") != "AUTO_DETECTED"
                or document.get("human_reviewed") is not False
                or not isinstance(structural, Mapping)
                or structural.get("status") != "PASS"
                or structural.get("complete_event_window_coverage") is not True
                or gap_count != 0
                or overlap_count != 0
            ):
                raise CandidateSelectionProvenanceMismatch(
                    "Automatic shot boundaries failed the structural tracking gate."
                )
            boundary_origin = "AUTO_DETECTED"
            human_reviewed = False
        else:
            if (
                document.get("automatic_confirmation") is not False
                or metadata.get("automatic_confirmation") is True
            ):
                raise CandidateSelectionProvenanceMismatch(
                    "Reviewed shot boundaries lack an explicit human confirmation."
                )
            boundary_origin = "HUMAN_REVIEWED"
            human_reviewed = True

        declared_origin = str(
            nested.get("boundary_origin")
            or manifest.get("boundary_origin")
            or selection_metadata.get("boundary_origin")
            or ""
        )
        declared_human_reviewed = (
            nested.get("human_reviewed")
            if "human_reviewed" in nested
            else manifest.get("human_reviewed")
            if "human_reviewed" in manifest
            else selection_metadata.get("human_reviewed")
        )
        if declared_origin and declared_origin != boundary_origin:
            raise CandidateSelectionProvenanceMismatch(
                "Candidate boundary-origin provenance does not match its artifact."
            )
        if (
            declared_human_reviewed is not None
            and declared_human_reviewed is not human_reviewed
        ):
            raise CandidateSelectionProvenanceMismatch(
                "Candidate human-review provenance does not match its artifact."
            )
        self._validate_boundary_rows(document, human_reviewed=human_reviewed)
        return ShotBoundaryProvenance(
            artifact=artifact,
            path=path,
            sha256=expected_sha,
            artifact_type=artifact_type,
            boundary_origin=boundary_origin,
            human_reviewed=human_reviewed,
        )

    @staticmethod
    def _catalog_candidate_is_safe(candidate: Mapping[str, Any]) -> bool:
        status = str(candidate.get("status") or "PENDING").upper()
        if status in {
            "EXCLUDED",
            "CONFIRMED",
            "UNREVIEWABLE_LOW_RESOLUTION",
            "EXCLUDED_BY_USER_NONE_OF_THESE",
            "EXCLUDED_BY_USER_NON_PLAYER_ROLE",
        }:
            return False
        for key in (
            "safety_gate_passed",
            "identity_pure",
            "identity_observability_gate_passed",
        ):
            if candidate.get(key) is False:
                return False
        for key in (
            "identity_purity",
            "identity_observability",
            "negative_review_gate",
            "role_confusion",
        ):
            gate = candidate.get(key)
            if isinstance(gate, Mapping) and gate.get("passed") is False:
                return False
        purity = str(
            candidate.get("purity_status")
            or candidate.get("identity_purity_status")
            or ""
        ).upper()
        return "MIXED" not in purity and "IMPURE" not in purity

    def _normalize_catalog_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        row = dict(candidate)
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or not self._catalog_candidate_is_safe(row):
            return None
        required_paths = (
            "manifest_path",
            "full_frame_context_path",
            "shot_clip_path",
            "reference_gallery_path",
        )
        for key in required_paths:
            value = row.get(key)
            if not value:
                return None
            path = Path(str(value))
            if not path.is_absolute():
                path = self.storage.resolve_path(str(value))
            path = path.resolve()
            if not path.is_file():
                return None
            sha_key = key.replace("_path", "_sha256")
            expected = str(row.get(sha_key) or "")
            actual = sha256_file(path)
            if expected and expected != actual:
                raise ValueError(
                    f"Review catalog artifact SHA mismatch: {candidate_id}/{key}"
                )
            row[key] = path.relative_to(self.storage.project_root).as_posix()
            row[sha_key] = actual
        row["candidate_id"] = candidate_id
        row["status"] = "PENDING"
        row["automatic_target_confirmation"] = False
        return row

    def _next_review_catalog_batch(
        self,
        *,
        job: TrackingJob,
        pipeline: EventCandidatePipelineR1,
        ambiguity: EventCandidateAmbiguityR1,
    ) -> EventCandidateAmbiguityR1 | None:
        state_path = Path(job.pipeline_state_path).resolve()
        if not state_path.is_file():
            return None
        state = self._load_json(state_path)
        result = dict(
            (state.get("shot_search_results") or {}).get(ambiguity.shot_id) or {}
        )
        catalog = [
            dict(item)
            for item in result.get("review_catalog_candidates") or []
            if isinstance(item, Mapping)
        ]
        if not catalog:
            runtime = dict(state.get("runtime") or {})
            catalog = [
                dict(item)
                for item in runtime.get("phase4b_review_catalog_candidates") or []
                if isinstance(item, Mapping)
            ]

        reviewed_ids = {
            str(value)
            for value in self.db.scalars(
                select(EventCandidateReviewDecisionR1.candidate_id).where(
                    EventCandidateReviewDecisionR1.tracking_job_id
                    == job.tracking_job_id,
                    EventCandidateReviewDecisionR1.candidate_id.is_not(None),
                )
            ).all()
            if value
        }
        exposed_ids: set[str] = set()
        for row in self.db.scalars(
            select(EventCandidateAmbiguityR1).where(
                EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id,
                EventCandidateAmbiguityR1.shot_id == ambiguity.shot_id,
            )
        ).all():
            # candidate_ids is intentionally only the active pointer set.  The
            # immutable candidates payload is the exposure history and must be
            # used to prevent a just-rejected (or earlier-generation) candidate
            # from being offered again.
            exposed_ids.update(
                str(candidate.get("candidate_id"))
                for candidate in (row.candidates or [])
                if candidate.get("candidate_id")
            )
            exposed_ids.update(
                str(candidate_id)
                for candidate_id in (row.candidate_ids or [])
                if candidate_id
            )
        excluded_ids = reviewed_ids | exposed_ids
        batch: list[dict[str, Any]] = []
        for candidate in catalog:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in excluded_ids:
                continue
            normalized = self._normalize_catalog_candidate(candidate)
            if normalized is None:
                continue
            batch.append(normalized)
            if len(batch) == 3:
                break
        if not batch:
            return None

        previous_rows = self.db.scalars(
            select(EventCandidateAmbiguityR1).where(
                EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id,
                EventCandidateAmbiguityR1.shot_id == ambiguity.shot_id,
            )
        ).all()
        generation = max([int(row.generation or 0) for row in previous_rows] or [0]) + 1
        safe_shot_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", ambiguity.shot_id)
        base_id = f"ambiguity_phase4b_{safe_shot_id}_g{generation:03d}"
        next_id = base_id
        suffix = 1
        existing_ids = {row.ambiguity_id for row in previous_rows}
        while next_id in existing_ids:
            suffix += 1
            next_id = f"{base_id}_{suffix}"

        artifact = Path(job.output_directory) / "ambiguities" / f"{next_id}.json"
        artifact_doc = {
            "schema_version": "kickclip.runtime_cross_shot_ambiguity.r1_4",
            "ambiguity_id": next_id,
            "tracking_job_id": job.tracking_job_id,
            "shot_id": ambiguity.shot_id,
            "status": "WAITING",
            "generation": generation,
            "candidate_ids": [item["candidate_id"] for item in batch],
            "candidates": batch,
            "source_review_catalog_reused": True,
            "automatic_target_confirmation": False,
        }
        artifact_sha = write_json_atomic(artifact, artifact_doc)
        first = batch[0]
        next_ambiguity = EventCandidateAmbiguityR1(
            tracking_job_id=job.tracking_job_id,
            ambiguity_id=next_id,
            shot_id=ambiguity.shot_id,
            status="WAITING",
            candidate_ids=artifact_doc["candidate_ids"],
            candidates=batch,
            full_frame_context_path=str(first["full_frame_context_path"]),
            full_frame_context_sha256=str(first["full_frame_context_sha256"]),
            shot_clip_path=str(first["shot_clip_path"]),
            shot_clip_sha256=str(first["shot_clip_sha256"]),
            artifact_path=artifact.relative_to(self.storage.project_root).as_posix(),
            artifact_sha256=artifact_sha,
            generation=generation,
        )
        self.db.add(next_ambiguity)
        self.db.flush()
        # The exhausted generation remains an immutable human-review result.
        # Only the newly-created generation may be WAITING.
        if ambiguity.status == "WAITING":
            ambiguity.status = "ALL_CANDIDATES_REJECTED"
        pipeline.pending_ambiguity_id = next_id
        pipeline.current_shot_id = ambiguity.shot_id
        pipeline.next_shot_id = ambiguity.shot_id
        return next_ambiguity

    def _prepared_artifact_path(
        self,
        artifact: Artifact,
        *,
        kind: str,
    ) -> Path:
        try:
            path = self.storage.resolve_path(artifact.file_path)
        except ValueError as exc:
            raise CandidateRecommendationNotPrepared(
                f"{kind}_PROVENANCE_MISMATCH"
            ) from exc
        if len(str(path)) >= 248 and not str(path).startswith("\\\\?\\"):
            path = Path("\\\\?\\" + str(path))
        if not path.is_file():
            raise CandidateRecommendationNotPrepared(f"{kind}_FILE_MISSING")
        expected = str((artifact.metadata_ or {}).get("sha256") or "")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise CandidateRecommendationNotPrepared(f"{kind}_SHA_MISMATCH")
        return path

    def _ranking_artifact(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> Artifact:
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == RANKING_ARTIFACT_TYPE,
            )
        ).all()
        matches = [
            row
            for row in rows
            if (row.metadata_ or {}).get("revision_id") == revision_id
            and (row.metadata_ or {}).get("event_id") == event_id
            and (row.metadata_ or {}).get("scene_id") == scene_id
        ]
        if not matches:
            raise CandidateRecommendationNotPrepared("V1_2_RANKING_MISSING")
        matches.sort(key=lambda row: row.created_at, reverse=True)
        return matches[0]

    def _ranking_context(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> tuple[Artifact, dict[str, Any], str, str]:
        artifact = self._ranking_artifact(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        try:
            document = self._load_json(
                self._prepared_artifact_path(artifact, kind="V1_2_RANKING")
            )
        except CandidateRecommendationNotPrepared:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            raise CandidateRecommendationNotPrepared(
                "V1_2_RANKING_PROVENANCE_MISMATCH"
            ) from exc
        if (
            document.get("schema_version") != "kickclip.event_candidate_ranking.v1_2"
            or document.get("automatic_target_confirmation") is not False
        ):
            raise CandidateRecommendationNotPrepared("V1_2_RANKING_PROVENANCE_MISMATCH")
        source_artifact_id = str(
            (artifact.metadata_ or {}).get("source_ranking_artifact_id") or ""
        )
        source = self.db.get(Artifact, source_artifact_id)
        if source is None:
            raise CandidateRecommendationNotPrepared("V1_2_RANKING_PROVENANCE_MISMATCH")
        source_metadata = source.metadata_ or {}
        if (
            source.project_id != project.project_id
            or source.match_id != project.match_id
            or source_metadata.get("revision_id") != revision_id
            or source_metadata.get("event_id") != event_id
            or source_metadata.get("scene_id") != scene_id
        ):
            raise CandidateRecommendationNotPrepared("INPUT_PROVENANCE_MISMATCH")
        self._prepared_artifact_path(source, kind="SOURCE_RANKING")
        reviewed_id = str(
            (artifact.metadata_ or {}).get("shot_boundaries_artifact_id")
            or (artifact.metadata_ or {}).get("reviewed_shots_artifact_id")
            or ""
        )
        reviewed = self.db.get(Artifact, reviewed_id)
        reviewed_metadata = reviewed.metadata_ if reviewed is not None else {}
        if (
            reviewed is None
            or reviewed.project_id != project.project_id
            or reviewed.match_id != project.match_id
            or (reviewed_metadata or {}).get("revision_id") != revision_id
            or (reviewed_metadata or {}).get("scene_id") != scene_id
        ):
            raise CandidateRecommendationNotPrepared("INPUT_PROVENANCE_MISMATCH")
        self._prepared_artifact_path(reviewed, kind="REVIEWED_SHOTS")
        ranking_id = str(source_metadata.get("ranking_id") or "")
        shortlist_patch_id = str((artifact.metadata_ or {}).get("ranking_id") or "")
        if not ranking_id or not shortlist_patch_id:
            raise CandidateRecommendationNotPrepared("V1_2_RANKING_PROVENANCE_MISMATCH")
        return artifact, document, ranking_id, shortlist_patch_id

    def _grouping_context(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
        ranking: dict[str, Any],
        ranking_id: str,
        shortlist_patch_id: str,
    ) -> tuple[Artifact, dict[str, Any]]:
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == GROUPING_ARTIFACT_TYPE,
            )
        ).all()
        matches = [
            row
            for row in rows
            if row.match_id == project.match_id
            and (row.metadata_ or {}).get("revision_id") == revision_id
            and (row.metadata_ or {}).get("event_id") == event_id
            and (row.metadata_ or {}).get("scene_id") == scene_id
            and (row.metadata_ or {}).get("shortlist_patch_id") == shortlist_patch_id
            and (row.metadata_ or {}).get("candidate_grouping_policy")
            == CANDIDATE_GROUPING_POLICY_VERSION
            and (row.metadata_ or {}).get("status") == "READY"
        ]
        if not matches:
            raise CandidateRecommendationNotPrepared("CANDIDATE_GROUPING_MISSING")
        matches.sort(key=lambda row: row.created_at, reverse=True)
        artifact = matches[0]
        try:
            document = self._load_json(
                self._prepared_artifact_path(
                    artifact,
                    kind="CANDIDATE_GROUPING",
                )
            )
        except CandidateRecommendationNotPrepared:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            raise CandidateRecommendationNotPrepared(
                "CANDIDATE_GROUPING_PROVENANCE_MISMATCH"
            ) from exc

        groups = document.get("groups")
        if (
            document.get("schema_version") != CANDIDATE_GROUPING_SCHEMA_VERSION
            or document.get("policy_version") != CANDIDATE_GROUPING_POLICY_VERSION
            or document.get("ranking_id") != ranking_id
            or document.get("shortlist_patch_id") != shortlist_patch_id
            or document.get("revision_id") != revision_id
            or document.get("event_id") != event_id
            or document.get("scene_id") != scene_id
            or document.get("automatic_target_confirmation") is not False
            or document.get("grouping_is_identity_confirmation") is not False
            or not isinstance(groups, list)
            or not groups
        ):
            raise CandidateRecommendationNotPrepared(
                "CANDIDATE_GROUPING_PROVENANCE_MISMATCH"
            )

        expected_candidate_ids = {
            public_candidate_id(str(row["candidate_id"]))
            for row in ranking.get("shortlist") or []
        }
        seen_members: set[str] = set()
        representatives: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                raise CandidateRecommendationNotPrepared(
                    "CANDIDATE_GROUPING_PROVENANCE_MISMATCH"
                )
            representative = str(group.get("representative_candidate_id") or "")
            members = [str(value) for value in group.get("member_candidate_ids") or []]
            if (
                not representative
                or representative not in members
                or representative in representatives
                or not members
                or len(members) != len(set(members))
                or seen_members.intersection(members)
            ):
                raise CandidateRecommendationNotPrepared(
                    "CANDIDATE_GROUPING_PROVENANCE_MISMATCH"
                )
            representatives.add(representative)
            seen_members.update(members)

        if (
            seen_members != expected_candidate_ids
            or int(document.get("source_candidate_count", -1))
            != len(expected_candidate_ids)
            or int(document.get("display_candidate_count", -1)) != len(groups)
        ):
            raise CandidateRecommendationNotPrepared(
                "CANDIDATE_GROUPING_PROVENANCE_MISMATCH"
            )
        return artifact, document

    def _bundle_artifact(
        self,
        *,
        project_id: str,
        shortlist_patch_id: str,
        candidate_id: str,
    ) -> Artifact:
        rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project_id,
                Artifact.artifact_type == BUNDLE_ARTIFACT_TYPE,
            )
        ).all()
        matches = [
            row
            for row in rows
            if (row.metadata_ or {}).get("shortlist_patch_id") == shortlist_patch_id
            and (row.metadata_ or {}).get("candidate_id") == candidate_id
            and (row.metadata_ or {}).get("status") != "INCOMPLETE"
        ]
        matches.sort(key=lambda row: row.created_at, reverse=True)
        match = matches[0] if matches else None
        if match is None:
            raise CandidateRecommendationNotPrepared(
                f"REVIEW_BUNDLE_MISSING:{candidate_id}"
            )
        return match

    def list_recommendations(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
    ) -> EventCandidateRecommendationResponse:
        revision = self.db.get(HighlightRevision, revision_id)
        pipeline_inputs = (
            (revision.options or {}).get("candidate_pipeline_inputs") or {}
            if revision is not None
            else {}
        )
        source_video_asset_id = str(pipeline_inputs.get("scene_video_asset_id") or "")
        if revision is not None and not source_video_asset_id:
            raise CandidateRecommendationNotPrepared("SOURCE_VIDEO_ASSET_MISSING")
        _, ranking, ranking_id, patch_id = self._ranking_context(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        grouping_artifact, grouping = self._grouping_context(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            ranking=ranking,
            ranking_id=ranking_id,
            shortlist_patch_id=patch_id,
        )
        grouping_sha256 = str((grouping_artifact.metadata_ or {}).get("sha256") or "")
        ranking_by_candidate_id = {
            public_candidate_id(str(row["candidate_id"])): row
            for row in ranking["shortlist"]
        }

        candidates: list[EventCandidateRecommendationRead] = []
        groups = sorted(
            grouping["groups"],
            key=lambda row: int(row["representative_shortlist_rank"]),
        )
        for group in groups:
            candidate_id = str(group["representative_candidate_id"])
            row = ranking_by_candidate_id.get(candidate_id)
            if row is None:
                raise CandidateRecommendationNotPrepared(
                    "CANDIDATE_GROUPING_PROVENANCE_MISMATCH"
                )
            bundle_artifact = self._bundle_artifact(
                project_id=project.project_id,
                shortlist_patch_id=patch_id,
                candidate_id=candidate_id,
            )
            try:
                manifest = self._load_json(
                    self._prepared_artifact_path(
                        bundle_artifact,
                        kind="REVIEW_BUNDLE",
                    )
                )
            except CandidateRecommendationNotPrepared:
                raise
            except (ValueError, json.JSONDecodeError) as exc:
                raise CandidateRecommendationNotPrepared(
                    f"REVIEW_BUNDLE_PROVENANCE_MISMATCH:{candidate_id}"
                ) from exc

            member_candidate_ids = [
                str(value) for value in group["member_candidate_ids"]
            ]
            if (
                manifest.get("candidate_id") != candidate_id
                or manifest.get("candidate_media_id") != candidate_id
                or manifest.get("ranking_id") != ranking_id
                or manifest.get("shortlist_patch_id") != patch_id
                or manifest.get("candidate_grouping_policy")
                != CANDIDATE_GROUPING_POLICY_VERSION
                or manifest.get("candidate_grouping_sha256") != grouping_sha256
                or manifest.get("candidate_group_id") != group["candidate_group_id"]
                or list(manifest.get("group_member_candidate_ids") or [])
                != member_candidate_ids
                or manifest.get("grouping_is_identity_confirmation") is not False
            ):
                raise CandidateRecommendationNotPrepared(
                    f"REVIEW_BUNDLE_PROVENANCE_MISMATCH:{candidate_id}"
                )

            quality = manifest["quality"]
            base = (
                f"/api/v1/event-candidate-recommendations/{patch_id}"
                f"/candidates/{candidate_id}/media"
            )
            project_query = f"?project_id={project.project_id}"
            grouping_reason_codes = list(group.get("grouping_reason_codes") or [])
            risk_codes = list(
                quality.get("purity_diagnostics", {}).get(
                    "reason_codes",
                    [],
                )
            )
            if len(member_candidate_ids) > 1:
                risk_codes.append("POSSIBLE_FRAGMENT_DUPLICATE_GROUP")

            reason_codes = list(row.get("shortlist_patch_reason_codes") or [])
            if len(member_candidate_ids) > 1:
                reason_codes.append("FRAGMENT_GROUP_REPRESENTATIVE")

            candidates.append(
                EventCandidateRecommendationRead(
                    ranking_version=RANKING_VERSION,
                    ranking_id=ranking_id,
                    shortlist_patch_id=patch_id,
                    candidate_id=candidate_id,
                    shortlist_rank=int(group["representative_shortlist_rank"]),
                    global_rank=int(row.get("original_global_rank", row.get("rank"))),
                    shot_id=str(row["shot_id"]),
                    tracklet_id=str(manifest["tracklet_id"]),
                    reviewability=str(quality["reviewability"]),
                    best_frame=int(quality["selected_best_frame"]),
                    media=CandidateMediaRead(
                        full_frame_context_url=(
                            f"{base}/full_frame_context{project_query}"
                        ),
                        best_crop_native_url=(
                            f"{base}/best_crop_native{project_query}"
                        ),
                        best_crop_display_url=(
                            f"{base}/best_crop_display{project_query}"
                        ),
                        first_middle_last_url=(
                            f"{base}/first_middle_last{project_query}"
                        ),
                        tracklet_video_url=(f"{base}/tracklet_video{project_query}"),
                        reference_gallery_url=(
                            f"{base}/reference_gallery{project_query}"
                        ),
                    ),
                    reason_codes=reason_codes,
                    risk_codes=sorted(set(risk_codes)),
                    candidate_group_id=str(group["candidate_group_id"]),
                    group_member_candidate_ids=member_candidate_ids,
                    grouped_candidate_count=len(member_candidate_ids),
                    grouping_reason_codes=grouping_reason_codes,
                    possible_fragment_duplicate=(len(member_candidate_ids) > 1),
                    automatic_target_confirmation=False,
                )
            )
        return EventCandidateRecommendationResponse(
            ranking_version=RANKING_VERSION,
            ranking_id=ranking_id,
            shortlist_patch_id=patch_id,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            source_video_asset_id=source_video_asset_id,
            source_candidate_count=int(grouping["source_candidate_count"]),
            display_candidate_count=int(grouping["display_candidate_count"]),
            candidate_grouping_policy=(CANDIDATE_GROUPING_POLICY_VERSION),
            candidates=candidates,
            automatic_target_confirmation=False,
            production_recommendation_ui="BLOCKED",
        )

    def _lazy_source_video_path(
        self,
        *,
        project: Project,
        source_video_sha256: str,
    ) -> Path:
        if len(source_video_sha256) != 64:
            raise ValueError("Candidate source video SHA-256 is invalid.")
        rows = self.db.scalars(
            select(MediaAsset).where(
                MediaAsset.match_id == project.match_id,
                MediaAsset.sha256 == source_video_sha256,
            )
        ).all()
        rows.sort(
            key=lambda row: (
                row.asset_type != "HIGHLIGHT_SCENE_CLIP",
                row.created_at,
            )
        )
        for asset in rows:
            try:
                path = self.storage.resolve_path(asset.file_path)
            except ValueError:
                continue
            if len(str(path)) >= 248 and not str(path).startswith("\\\\?\\"):
                path = Path("\\\\?\\" + str(path))
            if path.is_file() and sha256_file(path) == source_video_sha256:
                return path
        raise ValueError(
            "The immutable source video required for lazy candidate media "
            "is missing or changed."
        )

    def resolve_media(
        self,
        *,
        project: Project,
        shortlist_patch_id: str,
        candidate_id: str,
        media_name: str,
    ) -> tuple[Path, str]:
        artifact = self._bundle_artifact(
            project_id=project.project_id,
            shortlist_patch_id=shortlist_patch_id,
            candidate_id=candidate_id,
        )
        manifest_path = self._artifact_path(artifact)
        manifest = self._load_json(manifest_path)
        root = manifest_path.parent.resolve()

        record = (manifest.get("files") or {}).get(media_name)
        if isinstance(record, Mapping):
            path = (root / str(record.get("path") or "")).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or sha256_file(path) != record.get("sha256")
            ):
                raise ValueError("Candidate media integrity validation failed.")
            mime = "video/mp4" if path.suffix.lower() == ".mp4" else "image/jpeg"
            return path, mime

        lazy_spec = (manifest.get("lazy_media") or {}).get(media_name)
        if not isinstance(lazy_spec, Mapping):
            raise ValueError("Candidate media is not allowlisted.")

        source_video_path: Path | None = None
        if str(lazy_spec.get("kind") or "") in {
            "FIRST_MIDDLE_LAST",
            "TRACKLET_VIDEO",
        }:
            source_video_path = self._lazy_source_video_path(
                project=project,
                source_video_sha256=str(manifest.get("source_video_sha256") or ""),
            )
        manifest_sha256 = str((artifact.metadata_ or {}).get("sha256") or "")
        try:
            return self.lazy_media.materialize(
                bundle_root=root,
                bundle_manifest_sha256=manifest_sha256,
                manifest=manifest,
                media_name=media_name,
                source_video_path=source_video_path,
            )
        except LazyCandidateMediaError as exc:
            raise ValueError(str(exc)) from exc

    def create_selection(
        self,
        *,
        project: Project,
        revision_id: str,
        event_id: str,
        scene_id: str,
        ranking_id: str,
        shortlist_patch_id: str,
        candidate_id: str,
        source_video: MediaAsset,
        user: User,
    ) -> EventCandidateSelectionR1:
        _, ranking, actual_ranking_id, actual_patch_id = self._ranking_context(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
        )
        if ranking_id != actual_ranking_id or shortlist_patch_id != actual_patch_id:
            raise CandidateSelectionProvenanceMismatch(
                "The submitted ranking identity is not the served V1.2 ranking."
            )
        grouping_artifact, grouping = self._grouping_context(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            ranking=ranking,
            ranking_id=actual_ranking_id,
            shortlist_patch_id=actual_patch_id,
        )
        selected_group = next(
            (
                group
                for group in grouping["groups"]
                if str(group["representative_candidate_id"]) == candidate_id
            ),
            None,
        )
        if selected_group is None:
            raise CandidateSelectionProvenanceMismatch(
                "The selected candidate is not a served group representative."
            )
        grouping_sha256 = str((grouping_artifact.metadata_ or {}).get("sha256") or "")
        ranking_row = next(
            (
                row
                for row in ranking["shortlist"]
                if public_candidate_id(str(row["candidate_id"])) == candidate_id
            ),
            None,
        )
        if ranking_row is None:
            raise CandidateSelectionProvenanceMismatch(
                "The selected candidate is not in the served V1.2 shortlist."
            )
        bundle_artifact = self._bundle_artifact(
            project_id=project.project_id,
            shortlist_patch_id=shortlist_patch_id,
            candidate_id=candidate_id,
        )
        manifest_path = self._artifact_path(bundle_artifact)
        manifest = self._load_json(manifest_path)
        manifest_sha256 = sha256_file(manifest_path)
        if verify_manifest(manifest_path.parent, manifest):
            raise ValueError("Candidate review bundle integrity validation failed.")
        if manifest.get("source_video_sha256") != source_video.sha256:
            raise CandidateSelectionProvenanceMismatch(
                "The selected candidate bundle targets a different source video."
            )
        boundary = self._resolve_boundary_provenance(
            project=project,
            revision_id=revision_id,
            event_id=event_id,
            scene_id=scene_id,
            source_video_sha256=str(source_video.sha256),
            manifest=manifest,
        )
        expected_group_members = [
            str(value) for value in selected_group["member_candidate_ids"]
        ]
        if (
            manifest.get("candidate_grouping_policy")
            != CANDIDATE_GROUPING_POLICY_VERSION
            or manifest.get("candidate_grouping_sha256") != grouping_sha256
            or manifest.get("candidate_group_id")
            != selected_group["candidate_group_id"]
            or list(manifest.get("group_member_candidate_ids") or [])
            != expected_group_members
            or manifest.get("grouping_is_identity_confirmation") is not False
        ):
            raise CandidateSelectionProvenanceMismatch(
                "The selected candidate grouping provenance does not match."
            )
        identities = {
            public_candidate_id(str(ranking_row["candidate_id"])),
            str(manifest.get("candidate_id") or ""),
            str(manifest.get("candidate_media_id") or ""),
            candidate_id,
        }
        if identities != {candidate_id}:
            raise CandidateSelectionProvenanceMismatch(
                "Ranking, card media, and user-selected candidate IDs differ."
            )

        existing_rows = self.db.scalars(
            select(EventCandidateSelectionR1).where(
                EventCandidateSelectionR1.project_id == project.project_id,
                EventCandidateSelectionR1.revision_id == revision_id,
                EventCandidateSelectionR1.event_id == event_id,
                EventCandidateSelectionR1.scene_id == scene_id,
                EventCandidateSelectionR1.owner_id == user.user_id,
            )
        ).all()
        existing_rows.sort(key=lambda row: row.created_at, reverse=True)
        for existing in existing_rows:
            if (
                existing.ranking_id != ranking_id
                or existing.shortlist_patch_id != shortlist_patch_id
                or existing.candidate_id != candidate_id
                or existing.source_video_sha256 != source_video.sha256
                or existing.candidate_media_bundle_sha256 != manifest_sha256
                or existing.reviewed_shot_boundaries_sha256 != boundary.sha256
                or (existing.metadata_ or {}).get("candidate_grouping_sha256")
                != grouping_sha256
                or (existing.metadata_ or {}).get("candidate_group_id")
                != selected_group["candidate_group_id"]
            ):
                continue
            try:
                selection_path = self.storage.resolve_path(
                    existing.selection_artifact_path
                )
            except ValueError:
                continue
            if (
                selection_path.is_file()
                and sha256_file(selection_path) == existing.selection_artifact_sha256
            ):
                return existing

        selection_id = generate_prefixed_id("ecselr1")
        selected_at = datetime.now(timezone.utc)
        root = (manifest_path.parents[1] / "selections" / selection_id).resolve()
        root.mkdir(parents=True, exist_ok=False)
        document = {
            "schema_version": (
                "kickclip.event_candidate_selection_tracking_handoff.r1_1"
            ),
            "immutable": True,
            "selection_id": selection_id,
            "project_id": project.project_id,
            "revision_id": revision_id,
            "event_id": event_id,
            "scene_id": scene_id,
            "ranking_id": ranking_id,
            "shortlist_patch_id": shortlist_patch_id,
            "discovery_id": str(manifest["discovery_id"]),
            "candidate_id": candidate_id,
            "candidate_group_id": str(selected_group["candidate_group_id"]),
            "group_member_candidate_ids": expected_group_members,
            "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
            "candidate_grouping_sha256": grouping_sha256,
            "grouping_is_identity_confirmation": False,
            "shot_id": str(manifest["shot_id"]),
            "tracklet_id": str(manifest["tracklet_id"]),
            "user_id": user.user_id,
            "selected_at": selected_at.isoformat(),
            "candidate_manifest_sha256": str(manifest["candidate_manifest_sha256"]),
            "candidate_media_bundle_sha256": manifest_sha256,
            "source_video_sha256": str(source_video.sha256),
            "shot_boundaries": boundary.as_dict(),
            "shot_boundaries_sha256": boundary.sha256,
            "shot_boundary_artifact_type": boundary.artifact_type,
            "boundary_origin": boundary.boundary_origin,
            "human_reviewed": boundary.human_reviewed,
            "reviewed_shot_boundaries_sha256": (
                boundary.sha256 if boundary.human_reviewed else None
            ),
            "automatic_target_confirmation": False,
        }
        selection_path = root / "selection_record.json"
        selection_sha = write_json_atomic(selection_path, document)
        try:
            row = EventCandidateSelectionR1(
                selection_id=selection_id,
                project_id=project.project_id,
                owner_id=user.user_id,
                revision_id=revision_id,
                event_id=event_id,
                scene_id=scene_id,
                ranking_id=ranking_id,
                shortlist_patch_id=shortlist_patch_id,
                discovery_id=str(manifest["discovery_id"]),
                candidate_id=candidate_id,
                shot_id=str(manifest["shot_id"]),
                tracklet_id=str(manifest["tracklet_id"]),
                selected_at=selected_at,
                candidate_manifest_sha256=str(manifest["candidate_manifest_sha256"]),
                candidate_media_bundle_sha256=manifest_sha256,
                source_video_sha256=str(source_video.sha256),
                # Backward-compatible storage column. Generic provenance is
                # explicit in metadata and the immutable selection record.
                reviewed_shot_boundaries_sha256=boundary.sha256,
                selection_artifact_path=selection_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                selection_artifact_sha256=selection_sha,
                media_bundle_manifest_path=manifest_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                metadata_={
                    "automatic_target_confirmation": False,
                    "shot_boundaries_artifact_id": boundary.artifact.artifact_id,
                    "shot_boundaries_sha256": boundary.sha256,
                    "shot_boundary_artifact_type": boundary.artifact_type,
                    "boundary_origin": boundary.boundary_origin,
                    "human_reviewed": boundary.human_reviewed,
                    "frozen_source_candidate_id": ranking_row["candidate_id"],
                    "candidate_group_id": str(selected_group["candidate_group_id"]),
                    "group_member_candidate_ids": expected_group_members,
                    "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
                    "candidate_grouping_sha256": grouping_sha256,
                    "grouping_is_identity_confirmation": False,
                },
            )
            self.db.add(row)
            self.artifacts.create(
                match_id=project.match_id,
                project_id=project.project_id,
                analysis_job_id=None,
                artifact_type=SELECTION_ARTIFACT_TYPE,
                file_path=row.selection_artifact_path,
                mime_type="application/json",
                metadata_={
                    "selection_id": selection_id,
                    "sha256": selection_sha,
                    "candidate_id": candidate_id,
                    "shot_boundaries_artifact_id": boundary.artifact.artifact_id,
                    "shot_boundaries_sha256": boundary.sha256,
                    "shot_boundary_artifact_type": boundary.artifact_type,
                    "boundary_origin": boundary.boundary_origin,
                    "human_reviewed": boundary.human_reviewed,
                    "candidate_group_id": str(selected_group["candidate_group_id"]),
                    "group_member_candidate_ids": expected_group_members,
                    "candidate_grouping_policy": (CANDIDATE_GROUPING_POLICY_VERSION),
                    "candidate_grouping_sha256": grouping_sha256,
                    "shortlist_patch_id": shortlist_patch_id,
                    "owner_id": user.user_id,
                },
            )
            self.db.commit()
            self.db.refresh(row)
            return row
        except Exception:
            self.db.rollback()
            raise

    @staticmethod
    def selection_read(
        selection: EventCandidateSelectionR1,
    ) -> EventCandidateSelectionRead:
        metadata = selection.metadata_ or {}
        human_reviewed = metadata.get("human_reviewed")
        return EventCandidateSelectionRead(
            selection_id=selection.selection_id,
            project_id=selection.project_id,
            revision_id=selection.revision_id,
            event_id=selection.event_id,
            scene_id=selection.scene_id,
            ranking_id=selection.ranking_id,
            shortlist_patch_id=selection.shortlist_patch_id,
            discovery_id=selection.discovery_id,
            candidate_id=selection.candidate_id,
            shot_id=selection.shot_id,
            tracklet_id=selection.tracklet_id,
            user_id=selection.owner_id,
            selected_at=selection.selected_at,
            candidate_manifest_sha256=selection.candidate_manifest_sha256,
            candidate_media_bundle_sha256=(selection.candidate_media_bundle_sha256),
            source_video_sha256=selection.source_video_sha256,
            shot_boundaries_artifact_id=metadata.get("shot_boundaries_artifact_id"),
            shot_boundaries_sha256=str(
                metadata.get("shot_boundaries_sha256")
                or selection.reviewed_shot_boundaries_sha256
            ),
            shot_boundary_artifact_type=metadata.get("shot_boundary_artifact_type"),
            boundary_origin=metadata.get("boundary_origin"),
            human_reviewed=human_reviewed,
            reviewed_shot_boundaries_sha256=(
                selection.reviewed_shot_boundaries_sha256
                if human_reviewed is True
                else None
            ),
            selection_artifact_sha256=selection.selection_artifact_sha256,
            tracking_job_id=selection.tracking_job_id,
        )

    def create_tracking_handoff(
        self,
        *,
        selection: EventCandidateSelectionR1,
        source_video: MediaAsset,
        project: Project,
        user: User,
    ) -> EventCandidateTrackingCreateResponse:
        if selection.owner_id != user.user_id and not user.developer_mode_enabled:
            raise ValueError("Selection is not accessible.")
        if source_video.sha256 != selection.source_video_sha256:
            raise CandidateSelectionProvenanceMismatch(
                "Tracking source video differs from immutable selection."
            )
        manifest_path = self.storage.resolve_path(selection.media_bundle_manifest_path)
        if sha256_file(manifest_path) != selection.candidate_media_bundle_sha256:
            raise CandidateSelectionProvenanceMismatch(
                "Candidate media bundle changed after selection."
            )
        manifest = self._load_json(manifest_path)
        boundary = self._resolve_boundary_provenance(
            project=project,
            revision_id=selection.revision_id,
            event_id=selection.event_id,
            scene_id=selection.scene_id,
            source_video_sha256=selection.source_video_sha256,
            manifest=manifest,
            selection_metadata=selection.metadata_,
            legacy_selection_sha256=(selection.reviewed_shot_boundaries_sha256),
        )
        quality = dict(manifest.get("quality") or {})
        references = list(manifest.get("reference_gallery") or [])
        if len(references) < 3 or quality.get("reviewability") == "UNREVIEWABLE":
            raise InsufficientReviewableTargetReference(
                "Candidate lacks three reviewable, diverse native references."
            )
        if not quality.get("identity_pure"):
            raise TrackletIdentityInconsistent(
                "Selected candidate tracklet is not identity-pure."
            )
        anchor = dict(manifest.get("best_observation") or {})
        if (
            int(anchor.get("frame", -1)) < 0
            or not isinstance(anchor.get("bbox_xyxy"), list)
            or len(anchor["bbox_xyxy"]) != 4
        ):
            raise InsufficientReviewableTargetReference(
                "Candidate best-observation anchor is invalid."
            )
        retry_of: TrackingJob | None = None
        retry_generation = 1
        if selection.tracking_job_id:
            existing_job = self.db.get(TrackingJob, selection.tracking_job_id)
            if existing_job is None:
                raise CandidateSelectionProvenanceMismatch(
                    "Selection references a missing tracking job."
                )
            handoff_metadata = dict(
                (existing_job.runtime_metadata or {}).get("event_candidate_handoff_r1")
                or {}
            )
            provenance_sha = str(handoff_metadata.get("provenance_sha256") or "")
            provenance_path = (
                Path(existing_job.output_directory) / "integration_provenance.json"
            )
            if (
                not provenance_sha
                or not provenance_path.is_file()
                or sha256_file(provenance_path) != provenance_sha
            ):
                raise CandidateSelectionProvenanceMismatch(
                    "Existing tracking handoff provenance is missing or changed."
                )
            if existing_job.status in TERMINAL_STATUSES:
                retry_of = existing_job
                previous_pipeline = self.db.scalar(
                    select(EventCandidatePipelineR1).where(
                        EventCandidatePipelineR1.tracking_job_id
                        == existing_job.tracking_job_id
                    )
                )
                retry_generation = max(
                    2, int(previous_pipeline.generation if previous_pipeline else 1) + 1
                )
            else:
                return EventCandidateTrackingCreateResponse(
                    tracking_job_id=existing_job.tracking_job_id,
                    selection_id=selection.selection_id,
                    selected_candidate_id=selection.candidate_id,
                    selected_shot_id=selection.shot_id,
                    selected_reference_frame=int(anchor["frame"]),
                    selected_bbox=[float(value) for value in anchor["bbox_xyxy"]],
                    anchor_mode=ANCHOR_MODE,
                    provenance_artifact_sha256=provenance_sha,
                    status=existing_job.status,
                )
        payload = {
            "selected_candidate_id": selection.candidate_id,
            "selected_shot_id": selection.shot_id,
            "selected_reference_frame": int(anchor["frame"]),
            "selected_bbox": [float(value) for value in anchor["bbox_xyxy"]],
            "reference_gallery": [
                {
                    "path": str((manifest_path.parent / value["path"]).resolve()),
                    "sha256": value["crop_sha256"],
                    "frame_id": int(value["frame"]),
                    "scale_class": value["scale_class"],
                }
                for value in references
            ],
            "shot_boundaries": boundary.as_dict(),
            "anchor_mode": ANCHOR_MODE,
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
        }
        provenance = {
            "schema_version": "kickclip.selection_provenance.r1_3",
            "selection_id": selection.selection_id,
            "ranking_version": RANKING_VERSION,
            "ranking_id": selection.ranking_id,
            "shortlist_patch_id": selection.shortlist_patch_id,
            "discovery_id": selection.discovery_id,
            "ranking_candidate_id": selection.candidate_id,
            "candidate_media_id": manifest["candidate_media_id"],
            "user_selected_candidate_id": selection.candidate_id,
            "tracking_task_candidate_id": payload["selected_candidate_id"],
            "tracking_anchor_candidate_id": anchor["candidate_id"],
            "tracking_anchor_frame": payload["selected_reference_frame"],
            "tracking_anchor_bbox": payload["selected_bbox"],
            "tracking_anchor_shot": payload["selected_shot_id"],
            "shot_boundaries": boundary.as_dict(),
            "anchor_mode": ANCHOR_MODE,
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
        }
        candidate_ids = {
            str(provenance[key])
            for key in (
                "ranking_candidate_id",
                "candidate_media_id",
                "user_selected_candidate_id",
                "tracking_task_candidate_id",
                "tracking_anchor_candidate_id",
            )
        }
        if candidate_ids != {selection.candidate_id}:
            raise CandidateSelectionProvenanceMismatch(
                "CANDIDATE_SELECTION_PROVENANCE_MISMATCH"
            )

        video_path = self.storage.resolve_path(source_video.file_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise CandidateTrackingInputInvalid(
                "SOURCE_VIDEO_UNREADABLE",
                "Tracking source video cannot be opened.",
            )
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        job_id = generate_prefixed_id("trk")
        test_name = f"event_candidate_handoff_r1_{job_id}"
        settings = get_settings()
        output_root = self._tracking_output_root(settings.TRACKING_OUTPUT_ROOT)
        output = output_root / test_name
        output.mkdir(parents=True, exist_ok=False)
        payload_sha = write_json_atomic(output / "tracking_payload.json", payload)
        provenance_sha = write_json_atomic(
            output / "integration_provenance.json", provenance
        )
        bootstrap = {
            "schema_version": "kickclip.bootstrap_reference_timeline.r1",
            "tracking_job_id": job_id,
            "selected_candidate_id": selection.candidate_id,
            "selected_shot_id": selection.shot_id,
            "anchor_frame": int(anchor["frame"]),
            "source": "IMMUTABLE_SELECTION_EVIDENCE_ONLY",
            "tracking_success": False,
            "public_runtime_timeline": False,
            "observations": [
                {
                    **dict(item),
                    "state": "BOOTSTRAP_REFERENCE",
                }
                for item in (manifest.get("observations") or [])
            ],
        }
        bootstrap_sha = write_json_atomic(
            output / "bootstrap_reference_timeline.json", bootstrap
        )
        state = {
            "schema_version": "kickclip.event_candidate_handoff_state.r1_2",
            "pipeline_version": "EVENT_CANDIDATE_HANDOFF_R1_REAL_V1_V2_ADAPTER",
            "test_name": test_name,
            "status": "RUNNING",
            "decision": "QUEUED_FOR_REAL_V1_V2_RUNTIME",
            "execution_kind": R1_EXECUTION_KIND,
            "pipeline_stage": R1PipelineStage.SELECTED_SHOT_TRACKING.value,
            "processing_status": R1ProcessingStatus.READY.value,
            "selected_candidate_id": selection.candidate_id,
            "pending_action": None,
            "ambiguities": [],
            "runtime": {
                "synthetic_tracking_used": False,
                "observation_copy_used_as_success": False,
                "frame_zero_fallback_used": False,
                "automatic_target_confirmation": False,
            },
        }
        state_path = output / "pipeline_state.json"
        state_sha = write_json_atomic(state_path, state)
        job = TrackingJob(
            tracking_job_id=job_id,
            owner_id=user.user_id,
            match_id=project.match_id,
            project_id=project.project_id,
            media_asset_id=source_video.asset_id,
            test_name=test_name,
            initial_bbox=payload["selected_bbox"],
            bbox_format="xyxy_pixels",
            device=settings.TRACKING_DEVICE,
            reacquisition_mode=settings.TRACKING_REACQUISITION_MODE,
            status=TrackingBackendStatus.QUEUED.value,
            pipeline_status="RUNNING",
            pipeline_decision="QUEUED_FOR_REAL_V1_V2_RUNTIME",
            current_stage=R1PipelineStage.SELECTED_SHOT_TRACKING.value,
            execution_kind=R1_EXECUTION_KIND,
            pipeline_stage=R1PipelineStage.SELECTED_SHOT_TRACKING.value,
            processing_status=R1ProcessingStatus.READY.value,
            current_shot_id=selection.shot_id,
            pending_action_type=None,
            pending_ambiguity_id=None,
            output_directory=str(output),
            pipeline_state_path=str(state_path),
            timeline_path=None,
            queued_action={"kind": "new"},
            artifact_index={},
            runtime_metadata={
                "event_candidate_handoff_r1": {
                    "selection_id": selection.selection_id,
                    "selected_candidate_id": selection.candidate_id,
                    "candidate_manifest_path": str(manifest_path),
                    "candidate_manifest_sha256": selection.candidate_media_bundle_sha256,
                    "shot_boundaries": boundary.as_dict(),
                    "payload_sha256": payload_sha,
                    "provenance_sha256": provenance_sha,
                    "bootstrap_reference_timeline_path": str(
                        output / "bootstrap_reference_timeline.json"
                    ),
                    "bootstrap_reference_timeline_sha256": bootstrap_sha,
                    "bootstrap_observations_are_tracking_success": False,
                    "source_video": {
                        "path": str(video_path),
                        "sha256": source_video.sha256,
                        "frame_count": frame_count,
                        "fps": fps,
                        "width": width,
                        "height": height,
                    },
                    "retry": (
                        {
                            "previous_job_id": retry_of.tracking_job_id,
                            "previous_pipeline_state_sha256": sha256_file(
                                Path(retry_of.pipeline_state_path)
                            ),
                            "new_generation": retry_generation,
                            "retry_reason": (
                                "RETRY_AFTER_TERMINAL_JOB_WITH_PHASE1_COMPATIBILITY:"
                                f"{retry_of.status}"
                            ),
                        }
                        if retry_of is not None
                        else None
                    ),
                }
            },
            schema_version="kickclip.event_candidate_handoff_state.r1_2",
            pipeline_version="EVENT_CANDIDATE_HANDOFF_R1_REAL_V1_V2_ADAPTER",
            video_sha256=source_video.sha256,
            frozen_manifest_present=False,
            run_attempt=0,
        )
        try:
            self.db.add(job)
            self.db.flush()
            selection.tracking_job_id = job_id
            adapter = R1R3InputAdapter(settings=settings, storage=self.storage)
            try:
                adapter_result = adapter.build(
                    job=job,
                    selection=selection,
                    candidate_manifest_path=manifest_path,
                    source_video_path=video_path,
                    shot_boundaries_path=boundary.path,
                    shot_boundaries_provenance=boundary.as_dict(),
                )
            except R1R3AdapterError as exc:
                raise CandidateTrackingRuntimeContractInvalid(
                    "R1_R3_ADAPTER_CONTRACT_INVALID",
                    str(exc),
                ) from exc
            adapter.attach_to_job(job, adapter_result)
            R1PipelineOrchestrator(self.db).initialize(
                job=job,
                selection=selection,
                state_sha=state_sha,
                generation=retry_generation,
            )
            self.artifacts.create(
                match_id=project.match_id,
                project_id=project.project_id,
                analysis_job_id=None,
                artifact_type=PROVENANCE_ARTIFACT_TYPE,
                file_path=(output / "integration_provenance.json")
                .relative_to(self.storage.project_root)
                .as_posix(),
                mime_type="application/json",
                metadata_={
                    "tracking_job_id": job_id,
                    "selection_id": selection.selection_id,
                    "sha256": provenance_sha,
                    "shot_boundaries_artifact_id": (boundary.artifact.artifact_id),
                    "shot_boundaries_sha256": boundary.sha256,
                    "shot_boundary_artifact_type": boundary.artifact_type,
                    "boundary_origin": boundary.boundary_origin,
                    "human_reviewed": boundary.human_reviewed,
                },
            )
            # Register the immutable R1 selection as a pending highlight
            # candidate in the same transaction as the tracking handoff. This
            # keeps the server authoritative even if the browser closes before
            # the tracking job reaches a terminal state.
            from app.domains.highlight.service import HighlightWorkflowService

            HighlightWorkflowService(self.db).include_tracking_job_candidate(
                project=project,
                job=job,
                commit=False,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        get_r1_tracking_executor().submit(job_id)
        return EventCandidateTrackingCreateResponse(
            tracking_job_id=job_id,
            selection_id=selection.selection_id,
            selected_candidate_id=selection.candidate_id,
            selected_shot_id=selection.shot_id,
            selected_reference_frame=int(anchor["frame"]),
            selected_bbox=payload["selected_bbox"],
            anchor_mode=ANCHOR_MODE,
            provenance_artifact_sha256=provenance_sha,
            status=TrackingBackendStatus.QUEUED.value,
        )

    def record_review_decision(
        self,
        *,
        job: TrackingJob,
        ambiguity_id: str,
        request: CandidateReviewDecisionRequest,
        user: User,
    ) -> EventCandidateReviewDecisionR1:
        if job.owner_id != user.user_id and not user.developer_mode_enabled:
            raise ValueError("Tracking job is not accessible.")
        if job.execution_kind != R1_EXECUTION_KIND:
            raise ValueError("UNROUTABLE_TRACKING_JOB")
        idempotency_key = request.idempotency_key or canonical_sha256(
            {
                "job_id": job.tracking_job_id,
                "ambiguity_id": ambiguity_id,
                "candidate_id": request.candidate_id,
                "state": request.state.value,
                "reviewer": user.user_id,
            }
        )
        existing = self.db.scalar(
            select(EventCandidateReviewDecisionR1).where(
                EventCandidateReviewDecisionR1.idempotency_key == idempotency_key
            )
        )
        if existing is not None:
            return existing
        job = self.db.scalar(
            select(TrackingJob)
            .where(TrackingJob.tracking_job_id == job.tracking_job_id)
            .with_for_update()
        )
        pipeline = self.db.scalar(
            select(EventCandidatePipelineR1)
            .where(EventCandidatePipelineR1.tracking_job_id == job.tracking_job_id)
            .with_for_update()
        )
        ambiguity = self.db.scalar(
            select(EventCandidateAmbiguityR1)
            .where(
                EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id,
                EventCandidateAmbiguityR1.ambiguity_id == ambiguity_id,
            )
            .with_for_update()
        )
        if job is None or pipeline is None or ambiguity is None:
            raise ValueError("R1 job, pipeline, or ambiguity is missing.")
        if (
            pipeline.pending_ambiguity_id != ambiguity_id
            or job.status != TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
            or ambiguity.status != "WAITING"
        ):
            raise ValueError("Tracking job is not waiting for this ambiguity.")
        candidate_ids = list(ambiguity.candidate_ids or [])
        full_shot_states = {
            CandidateReviewState.TARGET_ABSENT,
            CandidateReviewState.NONE_OF_THESE,
            CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE,
        }
        if request.state not in full_shot_states:
            if request.candidate_id not in candidate_ids:
                raise ValueError("Candidate is not in the pending ambiguity.")
        if request.state in full_shot_states and (
            request.full_frame_context_sha256 != ambiguity.full_frame_context_sha256
            or request.shot_clip_sha256 != ambiguity.shot_clip_sha256
        ):
            raise ValueError(
                f"{request.state.value} requires the exact full-frame and full-shot evidence."
            )

        decision_id = generate_prefixed_id("ecdecr1")
        result_state = (
            "UNRESOLVED_LOW_RESOLUTION"
            if request.state == CandidateReviewState.UNREVIEWABLE_LOW_RESOLUTION
            else (
                (
                    "SEARCH_EXHAUSTED_NONE_OF_THESE"
                    if request.state == CandidateReviewState.NONE_OF_THESE
                    else "SEARCH_EXHAUSTED_NON_PLAYER_ROLE"
                )
                if request.state
                in {
                    CandidateReviewState.NONE_OF_THESE,
                    CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE,
                }
                else request.state.value
            )
        )
        document = {
            "schema_version": "kickclip.candidate_review_decision.r1_4",
            "decision_id": decision_id,
            "tracking_job_id": job.tracking_job_id,
            "ambiguity_id": ambiguity_id,
            "state": request.state.value,
            "resulting_tracking_state": result_state,
            "candidate_id": request.candidate_id,
            "rejected_candidate_ids": (
                candidate_ids
                if request.state
                in {
                    CandidateReviewState.NONE_OF_THESE,
                    CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE,
                }
                else (
                    [request.candidate_id]
                    if request.state
                    in {
                        CandidateReviewState.DIFFERENT_PLAYER,
                        CandidateReviewState.UNREVIEWABLE_LOW_RESOLUTION,
                    }
                    and request.candidate_id
                    else []
                )
            ),
            "reviewer": user.user_id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "full_frame_context_sha256": request.full_frame_context_sha256,
            "shot_clip_sha256": request.shot_clip_sha256,
            "note": request.note,
            "automatic_target_confirmation": False,
            "idempotency_key": idempotency_key,
        }
        path = Path(job.output_directory) / "review_decisions" / f"{decision_id}.json"
        sha = write_json_atomic(path, document)
        row = EventCandidateReviewDecisionR1(
            decision_id=decision_id,
            tracking_job_id=job.tracking_job_id,
            reviewer_id=user.user_id,
            ambiguity_id=ambiguity_id,
            candidate_id=request.candidate_id,
            decision_state=request.state.value,
            confirmation_artifact_path=path.relative_to(
                self.storage.project_root
            ).as_posix(),
            decision_artifact_sha256=sha,
            note=request.note,
            idempotency_key=idempotency_key,
            metadata_={
                "resulting_tracking_state": result_state,
                "rejected_candidate_ids": (
                    candidate_ids
                    if request.state
                    in {
                        CandidateReviewState.NONE_OF_THESE,
                        CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE,
                    }
                    else (
                        [request.candidate_id]
                        if request.state
                        in {
                            CandidateReviewState.DIFFERENT_PLAYER,
                            CandidateReviewState.UNREVIEWABLE_LOW_RESOLUTION,
                        }
                        and request.candidate_id
                        else []
                    )
                ),
            },
        )
        outbox = EventCandidateOutboxR1(
            tracking_job_id=job.tracking_job_id,
            idempotency_key=idempotency_key,
            event_type="APPLY_CANDIDATE_REVIEW_DECISION",
            status="PENDING",
            payload=document,
            attempts=0,
        )
        orchestrator = R1PipelineOrchestrator(self.db)
        selection = self.db.get(EventCandidateSelectionR1, pipeline.selection_id)
        if selection is None:
            raise ValueError("R1 selection is missing.")

        updated: list[dict[str, Any]] = []
        for raw in ambiguity.candidates or []:
            candidate = dict(raw)
            if request.state in {
                CandidateReviewState.NONE_OF_THESE,
                CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE,
            }:
                candidate["status"] = (
                    "EXCLUDED_BY_USER_NONE_OF_THESE"
                    if request.state == CandidateReviewState.NONE_OF_THESE
                    else "EXCLUDED_BY_USER_NON_PLAYER_ROLE"
                )
            elif candidate.get("candidate_id") == request.candidate_id:
                candidate["status"] = {
                    CandidateReviewState.SAME_PLAYER: "CONFIRMED",
                    CandidateReviewState.DIFFERENT_PLAYER: "EXCLUDED",
                    CandidateReviewState.UNREVIEWABLE_LOW_RESOLUTION: "UNREVIEWABLE_LOW_RESOLUTION",
                    CandidateReviewState.TARGET_ABSENT: candidate.get(
                        "status", "PENDING"
                    ),
                    CandidateReviewState.NONE_OF_THESE: "EXCLUDED_BY_USER_NONE_OF_THESE",
                    CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE: "EXCLUDED_BY_USER_NON_PLAYER_ROLE",
                }[request.state]
            updated.append(candidate)
        ambiguity.candidates = updated
        remaining = [
            item
            for item in updated
            if str(item.get("status") or "").upper() == "PENDING"
        ]
        # candidate_ids is the ordered active-candidate pointer, never an audit
        # history. Persist it before selecting the continuation branch so a
        # rejected final candidate cannot remain addressable after this commit.
        ambiguity.candidate_ids = [str(item["candidate_id"]) for item in remaining]
        pipeline.latest_decision_id = decision_id
        job.latest_decision_id = decision_id
        pipeline.current_shot_id = ambiguity.shot_id
        summary = dict(pipeline.summary or {})
        summary["confirmation_count"] = int(summary.get("confirmation_count") or 0) + 1
        if request.candidate_id:
            reviewed_candidate_ids = list(summary.get("reviewed_candidate_ids") or [])
            if request.candidate_id not in reviewed_candidate_ids:
                reviewed_candidate_ids.append(request.candidate_id)
            summary["reviewed_candidate_ids"] = reviewed_candidate_ids
        summary["reviewed_candidate_count"] = len(
            summary.get("reviewed_candidate_ids") or []
        )
        submit_runtime = False
        continuation = "TRACKING_RESUMED"
        next_candidate_id: str | None = None
        next_ambiguity_id: str | None = None
        remaining_candidate_count = 0

        if request.state == CandidateReviewState.SAME_PLAYER:
            memory_row, memory_document = orchestrator.build_memory(
                job=job,
                candidate_id=str(request.candidate_id),
                reviewer_id=user.user_id,
                decision_id=decision_id,
            )
            self.db.add(memory_row)
            self.db.flush()
            pipeline.current_memory_revision_id = memory_row.memory_revision_id
            job.current_memory_revision_id = memory_row.memory_revision_id
            scene = dict(
                (job.runtime_metadata or {}).get("scene_target_selection") or {}
            )
            memory_path = self.storage.resolve_path(memory_row.artifact_path)
            scene.update(
                {
                    "current_target_memory_path": str(memory_path),
                    "current_target_memory_sha256": memory_row.artifact_sha256,
                    "candidate_scoring_generation": int(
                        memory_document["candidate_scoring_generation"]
                    ),
                }
            )
            metadata = dict(job.runtime_metadata or {})
            metadata["scene_target_selection"] = scene
            job.runtime_metadata = metadata
            job.queued_action = {
                "kind": "ambiguity",
                "ambiguity_id": ambiguity_id,
                "candidate_id": request.candidate_id,
                "decision": "confirmed",
                "reviewer": user.user_id,
                "note": request.note,
            }
            ambiguity.status = "RESOLVED"
            pipeline.pending_ambiguity_id = None
            pipeline.last_completed_ambiguity_id = ambiguity_id
            pipeline.pipeline_stage = R1PipelineStage.BUILDING_TARGET_MEMORY.value
            summary["memory_revision_count"] = (
                int(summary.get("memory_revision_count") or 0) + 1
            )
            summary["last_memory_sha256"] = memory_row.artifact_sha256
            summary["last_memory_reference_count"] = len(memory_row.reference_frame_ids)
            submit_runtime = True
        elif request.state == CandidateReviewState.NONE_OF_THESE:
            rejected_candidate_ids = [
                str(candidate_id) for candidate_id in candidate_ids if candidate_id
            ]
            if not rejected_candidate_ids:
                raise ValueError(
                    "NONE_OF_THESE requires a non-empty pending candidate set."
                )
            job.queued_action = {
                "kind": "none_of_these",
                "ambiguity_id": ambiguity_id,
                "candidate_ids": rejected_candidate_ids,
                "decision": "none_of_these",
                "reviewer": user.user_id,
                "note": request.note,
            }
            ambiguity.status = "ALL_CANDIDATES_REJECTED"
            pipeline.pending_ambiguity_id = None
            pipeline.last_completed_ambiguity_id = ambiguity_id
            excluded = list(summary.get("excluded_candidate_ids") or [])
            for candidate_id in rejected_candidate_ids:
                if candidate_id not in excluded:
                    excluded.append(candidate_id)
            summary["excluded_candidate_ids"] = excluded
            none_shots = list(summary.get("none_of_these_shot_ids") or [])
            if ambiguity.shot_id not in none_shots:
                none_shots.append(ambiguity.shot_id)
            summary["none_of_these_shot_ids"] = none_shots
            submit_runtime = True
        elif request.state == CandidateReviewState.NONE_OF_THESE_NON_PLAYER_ROLE:
            rejected_candidate_ids = [
                str(candidate_id) for candidate_id in candidate_ids if candidate_id
            ]
            if not rejected_candidate_ids:
                raise ValueError(
                    "NONE_OF_THESE_NON_PLAYER_ROLE requires a non-empty pending candidate set."
                )
            job.queued_action = {
                "kind": "non_player_role",
                "ambiguity_id": ambiguity_id,
                "candidate_ids": rejected_candidate_ids,
                "decision": "non_player_role",
                "reviewer": user.user_id,
                "note": request.note,
            }
            ambiguity.status = "ALL_CANDIDATES_NON_PLAYER_ROLE"
            pipeline.pending_ambiguity_id = None
            pipeline.last_completed_ambiguity_id = ambiguity_id
            role_excluded = list(summary.get("non_player_role_candidate_ids") or [])
            for candidate_id in rejected_candidate_ids:
                if candidate_id not in role_excluded:
                    role_excluded.append(candidate_id)
            summary["non_player_role_candidate_ids"] = role_excluded
            role_shots = list(summary.get("non_player_role_shot_ids") or [])
            if ambiguity.shot_id not in role_shots:
                role_shots.append(ambiguity.shot_id)
            summary["non_player_role_shot_ids"] = role_shots
            submit_runtime = True
        elif request.state == CandidateReviewState.TARGET_ABSENT:
            job.queued_action = {
                "kind": "ambiguity",
                "ambiguity_id": ambiguity_id,
                "decision": "absent",
                "reviewer": user.user_id,
                "note": request.note,
            }
            ambiguity.status = "TARGET_ABSENT"
            pipeline.pending_ambiguity_id = None
            pipeline.last_completed_ambiguity_id = ambiguity_id
            summary.setdefault("shot_absent_ids", []).append(ambiguity.shot_id)
            submit_runtime = True
        elif remaining:
            # Candidate-local rejection/unreviewable: keep the same dynamic ambiguity.
            ambiguity.candidates = remaining
            ambiguity.status = "WAITING"
            pipeline.pending_ambiguity_id = ambiguity_id
            pipeline.pipeline_stage = (
                R1PipelineStage.WAITING_CROSS_SHOT_CONFIRMATION.value
            )
            pipeline.processing_status = R1ProcessingStatus.WAITING.value
            job.status = TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
            job.pipeline_status = "NEEDS_CONFIRMATION"
            job.pending_action_type = "CROSS_SHOT_CONFIRMATION"
            job.pending_ambiguity_id = ambiguity_id
            outbox.status = "COMPLETED"
            outbox.completed_at = datetime.now(timezone.utc)
            outbox.artifact_path = row.confirmation_artifact_path
            outbox.artifact_sha256 = sha
            continuation = "REVIEW_NEXT_CANDIDATE"
            remaining_candidate_count = len(remaining)
            next_candidate_id = str(remaining[0]["candidate_id"])
            next_ambiguity_id = ambiguity_id
        else:
            ambiguity.status = (
                "ALL_CANDIDATES_REJECTED"
                if request.state == CandidateReviewState.DIFFERENT_PLAYER
                else "UNRESOLVED_LOW_RESOLUTION"
            )
            next_batch = self._next_review_catalog_batch(
                job=job,
                pipeline=pipeline,
                ambiguity=ambiguity,
            )
            if next_batch is not None:
                job.queued_action = None
                job.status = TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
                job.pipeline_status = "NEEDS_CONFIRMATION"
                job.pending_action_type = "CROSS_SHOT_CONFIRMATION"
                job.pending_ambiguity_id = next_batch.ambiguity_id
                pipeline.pending_ambiguity_id = next_batch.ambiguity_id
                pipeline.pipeline_stage = (
                    R1PipelineStage.WAITING_CROSS_SHOT_CONFIRMATION.value
                )
                pipeline.processing_status = R1ProcessingStatus.WAITING.value
                outbox.status = "COMPLETED"
                outbox.completed_at = datetime.now(timezone.utc)
                outbox.artifact_path = row.confirmation_artifact_path
                outbox.artifact_sha256 = sha
                continuation = "REVIEW_NEXT_BATCH"
                # This response describes the exhausted generation. The new
                # generation is loaded from next_ambiguity_id as server truth.
                remaining_candidate_count = 0
                next_candidate_id = None
                next_ambiguity_id = next_batch.ambiguity_id
                summary["candidate_batch_generation"] = int(next_batch.generation)
                summary["remaining_candidate_count"] = remaining_candidate_count
            else:
                kind = (
                    "candidate_rejected"
                    if request.state == CandidateReviewState.DIFFERENT_PLAYER
                    else "candidate_unreviewable"
                )
                job.queued_action = {
                    "kind": kind,
                    "ambiguity_id": ambiguity_id,
                    "candidate_id": request.candidate_id,
                    "decision_artifact_path": row.confirmation_artifact_path,
                    "decision_artifact_sha256": sha,
                    "reviewer": user.user_id,
                    "note": request.note,
                }
                pipeline.pending_ambiguity_id = None
                pipeline.last_completed_ambiguity_id = ambiguity_id
                if kind == "candidate_rejected":
                    excluded = list(summary.get("excluded_candidate_ids") or [])
                    if request.candidate_id not in excluded:
                        excluded.append(request.candidate_id)
                    summary["excluded_candidate_ids"] = excluded
                else:
                    unresolved = list(
                        summary.get("unresolved_low_resolution_shot_ids") or []
                    )
                    if ambiguity.shot_id not in unresolved:
                        unresolved.append(ambiguity.shot_id)
                    summary["unresolved_low_resolution_shot_ids"] = unresolved
                submit_runtime = True
                continuation = "SEARCH_NEXT_SHOT"

        summary["remaining_candidate_count"] = remaining_candidate_count
        row.metadata_ = {
            **dict(row.metadata_ or {}),
            "continuation": continuation,
            "remaining_candidate_count": remaining_candidate_count,
            "next_candidate_id": next_candidate_id,
            "next_ambiguity_id": next_ambiguity_id,
            "automatic_target_confirmation": False,
        }
        pipeline.summary = summary
        if submit_runtime:
            pipeline.pipeline_stage = R1PipelineStage.APPLYING_HUMAN_DECISION.value
            pipeline.processing_status = R1ProcessingStatus.READY.value
            job.status = TrackingBackendStatus.QUEUED.value
            job.pipeline_status = "RUNNING"
            job.pipeline_decision = f"R1_RESUME_{request.state.value}"
            job.pending_action_type = None
            job.pending_ambiguity_id = None
        orchestrator.synchronize(
            job=job,
            selection=selection,
            pipeline=pipeline,
            ambiguity=ambiguity if not submit_runtime else None,
        )
        try:
            self.db.add(row)
            self.db.add(outbox)
            self.db.commit()
            self.db.refresh(row)
        except Exception:
            self.db.rollback()
            raise
        if submit_runtime:
            get_r1_tracking_executor().submit(job.tracking_job_id)
        return row

    def recover_rejected_candidate_resume(
        self,
        *,
        tracking_job_id: str,
        submit_runtime: bool = True,
    ) -> TrackingJob:
        """Requeue one provenance-complete DIFFERENT_PLAYER decision without recreating it."""
        job = self.db.scalar(
            select(TrackingJob)
            .where(TrackingJob.tracking_job_id == tracking_job_id)
            .with_for_update()
        )
        pipeline = self.db.scalar(
            select(EventCandidatePipelineR1)
            .where(EventCandidatePipelineR1.tracking_job_id == tracking_job_id)
            .with_for_update()
        )
        if job is None or pipeline is None or not job.latest_decision_id:
            raise ValueError(
                "R14 recovery requires an R1 job, pipeline, and latest decision."
            )
        decision = self.db.get(EventCandidateReviewDecisionR1, job.latest_decision_id)
        if (
            decision is None
            or decision.decision_state != CandidateReviewState.DIFFERENT_PLAYER.value
        ):
            raise ValueError(
                "R14 recovery requires a durable DIFFERENT_PLAYER decision."
            )
        ambiguity = self.db.scalar(
            select(EventCandidateAmbiguityR1)
            .where(
                EventCandidateAmbiguityR1.tracking_job_id == tracking_job_id,
                EventCandidateAmbiguityR1.ambiguity_id == decision.ambiguity_id,
            )
            .with_for_update()
        )
        if ambiguity is None or not decision.candidate_id:
            raise ValueError("R14 recovery ambiguity provenance is missing.")
        candidates = [dict(item) for item in (ambiguity.candidates or [])]
        rejected = next(
            (
                item
                for item in candidates
                if str(item.get("candidate_id") or "") == decision.candidate_id
            ),
            None,
        )
        if rejected is None or str(rejected.get("status") or "").upper() != "EXCLUDED":
            raise ValueError("R14 recovery candidate is not durably EXCLUDED.")
        active = [
            item
            for item in candidates
            if str(item.get("status") or "").upper() == "PENDING"
        ]
        if active:
            raise ValueError(
                "R14 recovery is only valid after the active batch is exhausted."
            )
        artifact = self.storage.resolve_path(decision.confirmation_artifact_path)
        if (
            not artifact.is_file()
            or sha256_file(artifact) != decision.decision_artifact_sha256
        ):
            raise ValueError("R14 recovery decision artifact is missing or changed.")
        outbox = self.db.scalar(
            select(EventCandidateOutboxR1)
            .where(
                EventCandidateOutboxR1.tracking_job_id == tracking_job_id,
                EventCandidateOutboxR1.idempotency_key == decision.idempotency_key,
            )
            .with_for_update()
        )
        if outbox is None:
            raise ValueError("R14 recovery requires the original decision outbox row.")

        runtime_state = (
            self._load_json(Path(job.pipeline_state_path))
            if Path(job.pipeline_state_path).is_file()
            else {}
        )
        runtime = dict(runtime_state.get("runtime") or {})
        negative_ids = {
            str(value)
            for value in dict(
                runtime.get("phase4b_identity_negative_memory") or {}
            ).get("rejected_candidate_ids")
            or []
            if value
        }
        rejection_already_applied = (
            decision.candidate_id in negative_ids
            and runtime_state.get("pending_action") is None
            and runtime.get("phase4a_initial_memory_review_status") == "PASS"
            and runtime.get("phase4b_cross_shot_scoring_authorized") is True
            and str(runtime_state.get("status") or "").upper()
            in {
                "RUNNING",
                "COMPLETE_WITH_UNRESOLVED_GAPS",
                "COMPLETED_WITH_UNRESOLVED_GAPS",
            }
        )

        ambiguity.candidate_ids = []
        ambiguity.status = "ALL_CANDIDATES_REJECTED"
        pipeline.pending_ambiguity_id = None
        pipeline.last_completed_ambiguity_id = ambiguity.ambiguity_id
        pipeline.latest_decision_id = decision.decision_id
        pipeline.pipeline_stage = (
            R1PipelineStage.SEARCHING_NEXT_SHOT.value
            if rejection_already_applied
            else R1PipelineStage.APPLYING_HUMAN_DECISION.value
        )
        pipeline.processing_status = R1ProcessingStatus.READY.value
        pipeline.completed_at = None
        pipeline.failure_code = None
        job.queued_action = (
            {"kind": "recovery_resume"}
            if rejection_already_applied
            else {
                "kind": "candidate_rejected",
                "ambiguity_id": ambiguity.ambiguity_id,
                "candidate_id": decision.candidate_id,
                "decision_artifact_path": decision.confirmation_artifact_path,
                "decision_artifact_sha256": decision.decision_artifact_sha256,
                "reviewer": decision.reviewer_id,
                "note": decision.note,
            }
        )
        job.status = TrackingBackendStatus.QUEUED.value
        job.pipeline_status = "RUNNING"
        job.pipeline_decision = (
            "R1_RECOVER_NEXT_SHOT"
            if rejection_already_applied
            else "R1_RECOVER_DIFFERENT_PLAYER"
        )
        job.pending_action_type = None
        job.pending_ambiguity_id = None
        job.process_pid = None
        job.process_return_code = None
        job.error_type = None
        job.error_message = None
        job.completed_at = None
        job.finished_at = None
        outbox.status = "COMPLETED" if rejection_already_applied else "PENDING"
        if not rejection_already_applied:
            outbox.completed_at = None
        outbox.error_message = None

        selection = self.db.get(EventCandidateSelectionR1, pipeline.selection_id)
        if selection is None:
            raise ValueError("R14 recovery selection provenance is missing.")
        R1PipelineOrchestrator(self.db).synchronize(
            job=job,
            selection=selection,
            pipeline=pipeline,
            ambiguity=None,
        )
        self.db.commit()
        self.db.refresh(job)
        if submit_runtime:
            get_r1_tracking_executor().submit(tracking_job_id)
        return job


def recover_candidate_handoff_state(db: Session) -> dict[str, Any]:
    """Recover from authoritative R1 DB state; never scan legacy pointers."""
    return recover_r1_pipelines(db)
