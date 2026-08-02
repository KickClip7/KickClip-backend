from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1 import ANCHOR_MODE, RANKING_VERSION
from app.core.config import get_settings
from app.domains.media.model import MediaAsset
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.execution import R1_EXECUTION_KIND, R1PipelineStage, R1ProcessingStatus
from app.domains.tracking.status import TERMINAL_STATUSES, TrackingBackendStatus
from app.domains.tracking.r1_executor import get_r1_tracking_executor
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
from .r3_adapter import R1R3InputAdapter
from .schema import (
    CandidateMediaRead,
    CandidateReviewDecisionRequest,
    CandidateReviewState,
    EventCandidateRecommendationRead,
    EventCandidateRecommendationResponse,
    EventCandidateSelectionRead,
    EventCandidateTrackingCreateResponse,
)


RANKING_ARTIFACT_TYPE = (
    "EVENT_CANDIDATE_RANKING_V1_2_SHADOW_SHORTLIST_PATCH"
)
BUNDLE_ARTIFACT_TYPE = "EVENT_CANDIDATE_REVIEW_BUNDLE_R1_MANIFEST"
GROUPING_ARTIFACT_TYPE = "EVENT_CANDIDATE_GROUPING_R1"
SELECTION_ARTIFACT_TYPE = "EVENT_CANDIDATE_SELECTION_R1"
PROVENANCE_ARTIFACT_TYPE = "CANDIDATE_SELECTION_INTEGRATION_PROVENANCE_R1"
PUBLIC_ID_PATTERN = re.compile(r"(shot_\d{4}_track_\d{4})$")


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
            document.get("schema_version")
            != "kickclip.event_candidate_ranking.v1_2"
            or document.get("automatic_target_confirmation") is not False
        ):
            raise CandidateRecommendationNotPrepared(
                "V1_2_RANKING_PROVENANCE_MISMATCH"
            )
        source_artifact_id = str(
            (artifact.metadata_ or {}).get("source_ranking_artifact_id") or ""
        )
        source = self.db.get(Artifact, source_artifact_id)
        if source is None:
            raise CandidateRecommendationNotPrepared(
                "V1_2_RANKING_PROVENANCE_MISMATCH"
            )
        source_metadata = source.metadata_ or {}
        if (
            source.project_id != project.project_id
            or source.match_id != project.match_id
            or source_metadata.get("revision_id") != revision_id
            or source_metadata.get("event_id") != event_id
            or source_metadata.get("scene_id") != scene_id
        ):
            raise CandidateRecommendationNotPrepared(
                "INPUT_PROVENANCE_MISMATCH"
            )
        self._prepared_artifact_path(source, kind="SOURCE_RANKING")
        reviewed_id = str(
            (artifact.metadata_ or {}).get("reviewed_shots_artifact_id") or ""
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
            raise CandidateRecommendationNotPrepared(
                "INPUT_PROVENANCE_MISMATCH"
            )
        self._prepared_artifact_path(reviewed, kind="REVIEWED_SHOTS")
        ranking_id = str(source_metadata.get("ranking_id") or "")
        shortlist_patch_id = str(
            (artifact.metadata_ or {}).get("ranking_id") or ""
        )
        if not ranking_id or not shortlist_patch_id:
            raise CandidateRecommendationNotPrepared(
                "V1_2_RANKING_PROVENANCE_MISMATCH"
            )
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
            and (row.metadata_ or {}).get("shortlist_patch_id")
            == shortlist_patch_id
            and (row.metadata_ or {}).get("candidate_grouping_policy")
            == CANDIDATE_GROUPING_POLICY_VERSION
            and (row.metadata_ or {}).get("status") == "READY"
        ]
        if not matches:
            raise CandidateRecommendationNotPrepared(
                "CANDIDATE_GROUPING_MISSING"
            )
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
            document.get("schema_version")
            != CANDIDATE_GROUPING_SCHEMA_VERSION
            or document.get("policy_version")
            != CANDIDATE_GROUPING_POLICY_VERSION
            or document.get("ranking_id") != ranking_id
            or document.get("shortlist_patch_id")
            != shortlist_patch_id
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
            representative = str(
                group.get("representative_candidate_id") or ""
            )
            members = [
                str(value)
                for value in group.get("member_candidate_ids") or []
            ]
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
            or int(document.get("display_candidate_count", -1))
            != len(groups)
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
            if (row.metadata_ or {}).get("shortlist_patch_id")
            == shortlist_patch_id
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
        source_video_asset_id = str(
            pipeline_inputs.get("scene_video_asset_id") or ""
        )
        if revision is not None and not source_video_asset_id:
            raise CandidateRecommendationNotPrepared(
                "SOURCE_VIDEO_ASSET_MISSING"
            )
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
        grouping_sha256 = str(
            (grouping_artifact.metadata_ or {}).get("sha256") or ""
        )
        ranking_by_candidate_id = {
            public_candidate_id(str(row["candidate_id"])): row
            for row in ranking["shortlist"]
        }

        candidates: list[EventCandidateRecommendationRead] = []
        groups = sorted(
            grouping["groups"],
            key=lambda row: int(
                row["representative_shortlist_rank"]
            ),
        )
        for group in groups:
            candidate_id = str(
                group["representative_candidate_id"]
            )
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
                str(value)
                for value in group["member_candidate_ids"]
            ]
            if (
                manifest.get("candidate_id") != candidate_id
                or manifest.get("candidate_media_id") != candidate_id
                or manifest.get("ranking_id") != ranking_id
                or manifest.get("shortlist_patch_id") != patch_id
                or manifest.get("candidate_grouping_policy")
                != CANDIDATE_GROUPING_POLICY_VERSION
                or manifest.get("candidate_grouping_sha256")
                != grouping_sha256
                or manifest.get("candidate_group_id")
                != group["candidate_group_id"]
                or list(manifest.get("group_member_candidate_ids") or [])
                != member_candidate_ids
                or manifest.get("grouping_is_identity_confirmation")
                is not False
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
            grouping_reason_codes = list(
                group.get("grouping_reason_codes") or []
            )
            risk_codes = list(
                quality.get("purity_diagnostics", {}).get(
                    "reason_codes",
                    [],
                )
            )
            if len(member_candidate_ids) > 1:
                risk_codes.append("POSSIBLE_FRAGMENT_DUPLICATE_GROUP")

            reason_codes = list(
                row.get("shortlist_patch_reason_codes") or []
            )
            if len(member_candidate_ids) > 1:
                reason_codes.append("FRAGMENT_GROUP_REPRESENTATIVE")

            candidates.append(
                EventCandidateRecommendationRead(
                    ranking_version=RANKING_VERSION,
                    ranking_id=ranking_id,
                    shortlist_patch_id=patch_id,
                    candidate_id=candidate_id,
                    shortlist_rank=int(
                        group["representative_shortlist_rank"]
                    ),
                    global_rank=int(
                        row.get("original_global_rank", row.get("rank"))
                    ),
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
                        tracklet_video_url=(
                            f"{base}/tracklet_video{project_query}"
                        ),
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
                    possible_fragment_duplicate=(
                        len(member_candidate_ids) > 1
                    ),
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
            source_candidate_count=int(
                grouping["source_candidate_count"]
            ),
            display_candidate_count=int(
                grouping["display_candidate_count"]
            ),
            candidate_grouping_policy=(
                CANDIDATE_GROUPING_POLICY_VERSION
            ),
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
                raise ValueError(
                    "Candidate media integrity validation failed."
                )
            mime = (
                "video/mp4"
                if path.suffix.lower() == ".mp4"
                else "image/jpeg"
            )
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
                source_video_sha256=str(
                    manifest.get("source_video_sha256") or ""
                ),
            )
        manifest_sha256 = str(
            (artifact.metadata_ or {}).get("sha256") or ""
        )
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
        if (
            ranking_id != actual_ranking_id
            or shortlist_patch_id != actual_patch_id
        ):
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
                if str(group["representative_candidate_id"])
                == candidate_id
            ),
            None,
        )
        if selected_group is None:
            raise CandidateSelectionProvenanceMismatch(
                "The selected candidate is not a served group representative."
            )
        grouping_sha256 = str(
            (grouping_artifact.metadata_ or {}).get("sha256") or ""
        )
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
        expected_group_members = [
            str(value)
            for value in selected_group["member_candidate_ids"]
        ]
        if (
            manifest.get("candidate_grouping_policy")
            != CANDIDATE_GROUPING_POLICY_VERSION
            or manifest.get("candidate_grouping_sha256")
            != grouping_sha256
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
                or existing.candidate_media_bundle_sha256
                != manifest_sha256
                or (existing.metadata_ or {}).get(
                    "candidate_grouping_sha256"
                )
                != grouping_sha256
                or (existing.metadata_ or {}).get(
                    "candidate_group_id"
                )
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
                and sha256_file(selection_path)
                == existing.selection_artifact_sha256
            ):
                return existing

        selection_id = generate_prefixed_id("ecselr1")
        selected_at = datetime.now(timezone.utc)
        root = (
            manifest_path.parents[1] / "selections" / selection_id
        ).resolve()
        root.mkdir(parents=True, exist_ok=False)
        document = {
            "schema_version": (
                "kickclip.event_candidate_selection_tracking_handoff.r1"
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
            "candidate_group_id": str(
                selected_group["candidate_group_id"]
            ),
            "group_member_candidate_ids": expected_group_members,
            "candidate_grouping_policy": (
                CANDIDATE_GROUPING_POLICY_VERSION
            ),
            "candidate_grouping_sha256": grouping_sha256,
            "grouping_is_identity_confirmation": False,
            "shot_id": str(manifest["shot_id"]),
            "tracklet_id": str(manifest["tracklet_id"]),
            "user_id": user.user_id,
            "selected_at": selected_at.isoformat(),
            "candidate_manifest_sha256": str(
                manifest["candidate_manifest_sha256"]
            ),
            "candidate_media_bundle_sha256": manifest_sha256,
            "source_video_sha256": str(source_video.sha256),
            "reviewed_shot_boundaries_sha256": str(
                manifest["reviewed_shot_boundaries_sha256"]
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
                candidate_manifest_sha256=str(
                    manifest["candidate_manifest_sha256"]
                ),
                candidate_media_bundle_sha256=manifest_sha256,
                source_video_sha256=str(source_video.sha256),
                reviewed_shot_boundaries_sha256=str(
                    manifest["reviewed_shot_boundaries_sha256"]
                ),
                selection_artifact_path=selection_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                selection_artifact_sha256=selection_sha,
                media_bundle_manifest_path=manifest_path.relative_to(
                    self.storage.project_root
                ).as_posix(),
                metadata_={
                    "automatic_target_confirmation": False,
                    "frozen_source_candidate_id": ranking_row["candidate_id"],
                    "candidate_group_id": str(
                        selected_group["candidate_group_id"]
                    ),
                    "group_member_candidate_ids": expected_group_members,
                    "candidate_grouping_policy": (
                        CANDIDATE_GROUPING_POLICY_VERSION
                    ),
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
                    "candidate_group_id": str(
                        selected_group["candidate_group_id"]
                    ),
                    "group_member_candidate_ids": expected_group_members,
                    "candidate_grouping_policy": (
                        CANDIDATE_GROUPING_POLICY_VERSION
                    ),
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
            candidate_media_bundle_sha256=(
                selection.candidate_media_bundle_sha256
            ),
            source_video_sha256=selection.source_video_sha256,
            reviewed_shot_boundaries_sha256=(
                selection.reviewed_shot_boundaries_sha256
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
        reviewed_rows = self.db.scalars(
            select(Artifact).where(
                Artifact.project_id == project.project_id,
                Artifact.artifact_type == "REVIEWED_SHOT_BOUNDARIES",
            )
        ).all()
        reviewed_matches = [
            artifact
            for artifact in reviewed_rows
            if (artifact.metadata_ or {}).get("revision_id")
            == selection.revision_id
            and (artifact.metadata_ or {}).get("scene_id")
            == selection.scene_id
            and (artifact.metadata_ or {}).get("sha256")
            == selection.reviewed_shot_boundaries_sha256
        ]
        reviewed_matches.sort(
            key=lambda artifact: artifact.created_at, reverse=True
        )
        if not reviewed_matches:
            raise CandidateSelectionProvenanceMismatch(
                "The immutable reviewed shot-boundary artifact is missing."
            )
        reviewed_shot_boundaries_path = self._artifact_path(
            reviewed_matches[0]
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
                (existing_job.runtime_metadata or {}).get(
                    "event_candidate_handoff_r1"
                )
                or {}
            )
            provenance_sha = str(
                handoff_metadata.get("provenance_sha256") or ""
            )
            provenance_path = (
                Path(existing_job.output_directory)
                / "integration_provenance.json"
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
                    selected_bbox=[
                        float(value) for value in anchor["bbox_xyxy"]
                    ],
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
            "reviewed_shot_boundaries": {
                "sha256": selection.reviewed_shot_boundaries_sha256,
                "source": "IMMUTABLE_HUMAN_REVIEWED_SHOT_BOUNDARIES",
            },
            "anchor_mode": ANCHOR_MODE,
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
        }
        provenance = {
            "schema_version": "kickclip.selection_provenance.r1_2",
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
            raise ValueError("Tracking source video cannot be opened.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        job_id = generate_prefixed_id("trk")
        test_name = f"event_candidate_handoff_r1_{job_id}"
        settings = get_settings()
        if not settings.TRACKING_OUTPUT_ROOT:
            raise ValueError("TRACKING_OUTPUT_ROOT is required for the R1 runtime.")
        output_root = Path(settings.TRACKING_OUTPUT_ROOT).expanduser().resolve()
        if not output_root.is_absolute():
            raise ValueError("TRACKING_OUTPUT_ROOT must be absolute.")
        if not output_root.is_relative_to(self.storage.storage_root):
            raise ValueError(
                "TRACKING_OUTPUT_ROOT must be inside STORAGE_ROOT so immutable "
                "R1 artifacts remain addressable by the backend."
            )
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
            adapter_result = adapter.build(
                job=job,
                selection=selection,
                candidate_manifest_path=manifest_path,
                source_video_path=video_path,
                reviewed_shot_boundaries_path=reviewed_shot_boundaries_path,
            )
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
                file_path=(output / "integration_provenance.json").relative_to(
                    self.storage.project_root
                ).as_posix(),
                mime_type="application/json",
                metadata_={
                    "tracking_job_id": job_id,
                    "selection_id": selection.selection_id,
                    "sha256": provenance_sha,
                },
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
        if request.state != CandidateReviewState.TARGET_ABSENT:
            if request.candidate_id not in candidate_ids:
                raise ValueError("Candidate is not in the pending ambiguity.")
        if request.state == CandidateReviewState.TARGET_ABSENT and (
            request.full_frame_context_sha256 != ambiguity.full_frame_context_sha256
            or request.shot_clip_sha256 != ambiguity.shot_clip_sha256
        ):
            raise ValueError(
                "TARGET_ABSENT requires the exact full-frame and full-shot evidence."
            )

        decision_id = generate_prefixed_id("ecdecr1")
        result_state = (
            "UNRESOLVED_LOW_RESOLUTION"
            if request.state == CandidateReviewState.UNREVIEWABLE_LOW_RESOLUTION
            else request.state.value
        )
        document = {
            "schema_version": "kickclip.candidate_review_decision.r1_2",
            "decision_id": decision_id,
            "tracking_job_id": job.tracking_job_id,
            "ambiguity_id": ambiguity_id,
            "state": request.state.value,
            "resulting_tracking_state": result_state,
            "candidate_id": request.candidate_id,
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
            metadata_={"resulting_tracking_state": result_state},
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
            if candidate.get("candidate_id") == request.candidate_id:
                candidate["status"] = {
                    CandidateReviewState.SAME_PLAYER: "CONFIRMED",
                    CandidateReviewState.DIFFERENT_PLAYER: "EXCLUDED",
                    CandidateReviewState.UNREVIEWABLE_LOW_RESOLUTION: "UNREVIEWABLE_LOW_RESOLUTION",
                    CandidateReviewState.TARGET_ABSENT: candidate.get("status", "PENDING"),
                }[request.state]
            updated.append(candidate)
        ambiguity.candidates = updated
        remaining = [item for item in updated if item.get("status") == "PENDING"]
        pipeline.latest_decision_id = decision_id
        job.latest_decision_id = decision_id
        pipeline.current_shot_id = ambiguity.shot_id
        summary = dict(pipeline.summary or {})
        summary["confirmation_count"] = int(summary.get("confirmation_count") or 0) + 1
        submit_runtime = False

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
            scene = dict((job.runtime_metadata or {}).get("scene_target_selection") or {})
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
            summary["memory_revision_count"] = int(summary.get("memory_revision_count") or 0) + 1
            summary["last_memory_sha256"] = memory_row.artifact_sha256
            summary["last_memory_reference_count"] = len(memory_row.reference_frame_ids)
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
            ambiguity.status = "WAITING"
            pipeline.pending_ambiguity_id = ambiguity_id
            pipeline.pipeline_stage = R1PipelineStage.WAITING_CROSS_SHOT_CONFIRMATION.value
            pipeline.processing_status = R1ProcessingStatus.WAITING.value
            job.status = TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
            job.pipeline_status = "NEEDS_CONFIRMATION"
            job.pending_action_type = "CROSS_SHOT_CONFIRMATION"
            job.pending_ambiguity_id = ambiguity_id
            outbox.status = "COMPLETED"
            outbox.completed_at = datetime.now(timezone.utc)
            outbox.artifact_path = row.confirmation_artifact_path
            outbox.artifact_sha256 = sha
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
                "reviewer": user.user_id,
                "note": request.note,
            }
            ambiguity.status = (
                "ALL_CANDIDATES_REJECTED"
                if kind == "candidate_rejected"
                else "UNRESOLVED_LOW_RESOLUTION"
            )
            pipeline.pending_ambiguity_id = None
            pipeline.last_completed_ambiguity_id = ambiguity_id
            if kind == "candidate_rejected":
                summary.setdefault("excluded_candidate_ids", []).append(
                    request.candidate_id
                )
            else:
                summary.setdefault("unresolved_low_resolution_shot_ids", []).append(
                    ambiguity.shot_id
                )
            submit_runtime = True

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

def recover_candidate_handoff_state(db: Session) -> dict[str, Any]:
    """Recover from authoritative R1 DB state; never scan legacy pointers."""
    return recover_r1_pipelines(db)
