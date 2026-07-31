from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.model import (
    EventCandidateRanking,
    ScenePlayerCandidate,
)
from app.domains.highlight.event_candidate_verifier import (
    EventCandidateRankingVerifier,
)
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.runtime_contract import (
    project_relative,
    sha256_file,
)
from app.domains.highlight.schema import (
    EventCandidateRankingRead,
    EventCandidateRankingRequest,
    EventCandidateEvaluationRead,
    EventCandidateLabelRequest,
    EventCandidateScoreRead,
)
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id


PACKAGE_NAME = "target_centric_tracking_event_candidate_ranking_v1"
STATUS = "PROVISIONAL_SHADOW_ONLY"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


class EventCandidateRankingService:
    """Shadow-only event relevance policy; never confirms a target."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = HighlightRepository(db)
        self.artifacts = ArtifactRepository(db)
        self.storage = LocalStorage()
        self.package_root = (
            self.storage.project_root
            / "configs"
            / "models"
            / "event_candidate_ranking"
            / PACKAGE_NAME
        ).resolve()
        self.policy_path = self.package_root / "event_candidate_ranking_policy.json"
        self.feature_schema_path = (
            self.package_root / "event_candidate_feature_schema.json"
        )
        self.manifest_path = self.package_root / "manifest.json"
        self.verifier = EventCandidateRankingVerifier(self.package_root)

    def rank(
        self,
        *,
        project: Project,
        revision_id: str,
        payload: EventCandidateRankingRequest,
        user: User,
    ) -> EventCandidateRanking:
        revision = self.repository.get_revision(revision_id)
        verification = self.verifier.check()
        if not verification.available:
            raise RuntimeError(verification.message)
        if revision is None or revision.project_id != project.project_id:
            raise ValueError("Highlight revision not found.")
        if payload.scene_id not in revision.selected_scene_ids:
            raise ValueError("Scene is not selected in this revision.")
        scene = self.db.get(TimelineEvent, payload.scene_id)
        event = self.db.get(TimelineEvent, payload.event_id)
        if (
            scene is None
            or event is None
            or scene.match_id != project.match_id
            or event.match_id != project.match_id
        ):
            raise ValueError("Event context does not belong to this Match.")
        self._assert_event_contract(scene, event, payload)
        discovery = (
            (revision.options or {}).get("scene_target_selection") or {}
        )
        expected_manifest_sha = discovery.get(
            "scene_candidate_manifest_sha256"
        )
        expected_boundaries_sha = (
            discovery.get("discovery_inputs") or {}
        ).get("shot_boundaries_sha256")
        if payload.scene_candidate_manifest_sha256 != expected_manifest_sha:
            raise ValueError("Scene candidate manifest SHA-256 mismatch.")
        if payload.shot_boundaries_sha256 != expected_boundaries_sha:
            raise ValueError("Shot boundaries SHA-256 mismatch.")

        policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        candidates = [
            row
            for row in self.repository.list_candidates(
                revision_id,
                payload.scene_id,
            )
            if (row.metadata_ or {}).get("discovery_id")
            == discovery.get("discovery_id")
        ]
        if not candidates:
            raise ValueError("Immutable scene candidate set is empty.")
        rows = self._score_candidates(
            candidates,
            payload=payload,
            policy=policy,
        )
        ranking_id = generate_prefixed_id("ecrank")
        output_root = (
            self.storage.resolve_path(discovery["artifact_root"])
            / "rankings"
            / ranking_id
        ).resolve()
        discovery_root = self.storage.resolve_path(discovery["artifact_root"])
        if not output_root.is_relative_to(discovery_root):
            raise ValueError("Event ranking output escapes discovery root.")
        output_root.mkdir(parents=True, exist_ok=False)
        output_path = output_root / "event_candidate_ranking.json"
        document = {
            "schema_version": "kickclip.event_candidate_ranking.v1",
            "package": PACKAGE_NAME,
            "status": STATUS,
            "time_coordinate_system": "SOURCE_VIDEO_SECONDS",
            "candidate_time_coordinate_system": "SCENE_LOCAL_SECONDS",
            "event_context": payload.model_dump(exclude={"shortlist_size"}),
            "policy_sha256": sha256_file(self.policy_path),
            "feature_schema_sha256": sha256_file(self.feature_schema_path),
            "automatic_target_confirmation": False,
            "shortlist_size": payload.shortlist_size,
            "shortlist": rows[: payload.shortlist_size],
            "all_candidates": rows,
        }
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=None,
            artifact_type="EVENT_CANDIDATE_RANKING_JSON",
            file_path=project_relative(
                output_path,
                self.storage.project_root,
            ),
            mime_type="application/json",
            metadata_={
                "ranking_id": ranking_id,
                "status": STATUS,
                "sha256": sha256_file(output_path),
            },
        )
        ranking = self.repository.create_event_candidate_ranking(
            ranking_id=ranking_id,
            owner_id=user.user_id,
            match_id=project.match_id,
            project_id=project.project_id,
            revision_id=revision_id,
            scene_id=payload.scene_id,
            event_id=payload.event_id,
            event_label=payload.event_label,
            event_time_sec=payload.event_time_sec,
            event_confidence=payload.event_confidence,
            scene_start_sec=payload.scene_start_sec,
            scene_end_sec=payload.scene_end_sec,
            scene_candidate_manifest_sha256=(
                payload.scene_candidate_manifest_sha256
            ),
            shot_boundaries_sha256=payload.shot_boundaries_sha256,
            policy_sha256=sha256_file(self.policy_path),
            feature_schema_sha256=sha256_file(self.feature_schema_path),
            status=STATUS,
            shortlist_size=payload.shortlist_size,
            output_path=project_relative(
                output_path,
                self.storage.project_root,
            ),
            output_sha256=sha256_file(output_path),
            metadata_={
                "artifact_id": artifact.artifact_id,
                "manifest_sha256": sha256_file(self.manifest_path),
                "automatic_target_confirmation": False,
                "candidate_count": len(rows),
            },
        )
        for row in rows:
            self.repository.create_event_candidate_score(
                ranking_id=ranking_id,
                candidate_id=row["candidate_id"],
                rank=row["rank"],
                raw_features=row["raw_features"],
                event_relevance_score=row["event_relevance_score"],
                trackability_score=row["trackability_score"],
                recommendation_score=row["recommendation_score"],
                reason_codes=row["reason_codes"],
                risk_codes=row["risk_codes"],
            )
        self.db.commit()
        self.db.refresh(ranking)
        return ranking

    def read(
        self,
        ranking: EventCandidateRanking,
        *,
        user: User,
    ) -> EventCandidateRankingRead:
        if (
            ranking.owner_id != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Event candidate ranking not found.")
        scores = self.repository.list_event_candidate_scores(
            ranking.ranking_id
        )
        score_rows = [
            EventCandidateScoreRead(
                candidate_id=row.candidate_id,
                rank=row.rank,
                raw_features=row.raw_features,
                event_relevance_score=row.event_relevance_score,
                trackability_score=row.trackability_score,
                recommendation_score=row.recommendation_score,
                reason_codes=list(row.reason_codes),
                risk_codes=list(row.risk_codes),
                artifact_ids=self._candidate_artifact_ids(
                    ranking.revision_id,
                    row.candidate_id,
                ),
            )
            for row in scores
        ]
        return EventCandidateRankingRead(
            ranking_id=ranking.ranking_id,
            status=ranking.status,
            time_coordinate_system="SOURCE_VIDEO_SECONDS",
            event_id=ranking.event_id,
            event_label=ranking.event_label,
            event_time_sec=ranking.event_time_sec,
            event_confidence=ranking.event_confidence,
            scene_id=ranking.scene_id,
            scene_start_sec=ranking.scene_start_sec,
            scene_end_sec=ranking.scene_end_sec,
            shortlist_size=ranking.shortlist_size,
            shortlist=score_rows[: ranking.shortlist_size],
            all_candidates=score_rows,
            automatic_target_confirmation=False,
            shadow_only=ranking.status == STATUS,
        )

    def label(
        self,
        ranking: EventCandidateRanking,
        *,
        user: User,
        payload: EventCandidateLabelRequest,
    ) -> None:
        self._assert_owner(ranking, user)
        score_ids = {
            row.candidate_id
            for row in self.repository.list_event_candidate_scores(
                ranking.ranking_id
            )
        }
        if payload.candidate_id not in score_ids:
            raise ValueError("Candidate is not part of this ranking.")
        self.repository.upsert_event_candidate_label(
            ranking_id=ranking.ranking_id,
            candidate_id=payload.candidate_id,
            reviewer_id=user.user_id,
            role=payload.role,
            metadata_={
                "note": payload.note,
                "offline_ground_truth_only": True,
                "excluded_from_production_features": True,
            },
        )
        self.db.commit()

    def evaluate(
        self,
        ranking: EventCandidateRanking,
        *,
        user: User,
    ) -> EventCandidateEvaluationRead:
        self._assert_owner(ranking, user)
        scores = self.repository.list_event_candidate_scores(
            ranking.ranking_id
        )
        labels = self.repository.list_event_candidate_labels(
            ranking.ranking_id
        )
        roles_by_candidate: dict[str, set[str]] = {}
        for label in labels:
            roles_by_candidate.setdefault(label.candidate_id, set()).add(
                label.role
            )
        primary_ranks = [
            score.rank
            for score in scores
            if "PRIMARY_EVENT_ACTOR"
            in roles_by_candidate.get(score.candidate_id, set())
        ]
        if not primary_ranks:
            return EventCandidateEvaluationRead(
                ranking_id=ranking.ranking_id,
                status="NOT_RUN",
                candidate_generation_actor_coverage=None,
                primary_actor_recall_at_1=None,
                primary_actor_recall_at_3=None,
                primary_actor_recall_at_5=None,
                mrr=None,
                broadcast_non_actor_top_1_rate=None,
                labeled_candidate_count=len(labels),
            )
        best_rank = min(primary_ranks)
        top_roles = (
            roles_by_candidate.get(scores[0].candidate_id, set())
            if scores
            else set()
        )
        return EventCandidateEvaluationRead(
            ranking_id=ranking.ranking_id,
            status="MEASURED",
            candidate_generation_actor_coverage=1.0,
            primary_actor_recall_at_1=1.0 if best_rank <= 1 else 0.0,
            primary_actor_recall_at_3=1.0 if best_rank <= 3 else 0.0,
            primary_actor_recall_at_5=1.0 if best_rank <= 5 else 0.0,
            mrr=round(1.0 / best_rank, 6),
            broadcast_non_actor_top_1_rate=(
                1.0
                if "BROADCAST_CLOSEUP_NON_ACTOR" in top_roles
                else 0.0
            ),
            labeled_candidate_count=len(labels),
        )

    @staticmethod
    def _assert_owner(
        ranking: EventCandidateRanking,
        user: User,
    ) -> None:
        if (
            ranking.owner_id != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Event candidate ranking not found.")

    @staticmethod
    def _assert_event_contract(
        scene: TimelineEvent,
        event: TimelineEvent,
        payload: EventCandidateRankingRequest,
    ) -> None:
        if event.label.casefold() != payload.event_label.casefold():
            raise ValueError("event_label does not match the durable event.")
        if abs(event.timestamp_sec - payload.event_time_sec) > 0.001:
            raise ValueError(
                "event_time_sec must be source-video seconds from the durable event."
            )
        if (
            abs(scene.start_sec - payload.scene_start_sec) > 0.001
            or abs(scene.end_sec - payload.scene_end_sec) > 0.001
        ):
            raise ValueError("Scene source-video time range mismatch.")
        if (
            event.confidence is not None
            and payload.event_confidence is not None
            and abs(event.confidence - payload.event_confidence) > 0.001
        ):
            raise ValueError("event_confidence does not match the durable event.")

    def _score_candidates(
        self,
        candidates: list[ScenePlayerCandidate],
        *,
        payload: EventCandidateRankingRequest,
        policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        max_observations = max(
            max(1, candidate.track_length_frames)
            for candidate in candidates
        )
        rows = [
            self._features_and_score(
                candidate,
                payload=payload,
                policy=policy,
                max_observations=max_observations,
            )
            for candidate in candidates
        ]
        rows.sort(
            key=lambda row: (
                -row["recommendation_score"],
                -row["event_relevance_score"],
                -row["trackability_score"],
                row["candidate_id"],
            )
        )
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def _features_and_score(
        self,
        candidate: ScenePlayerCandidate,
        *,
        payload: EventCandidateRankingRequest,
        policy: dict[str, Any],
        max_observations: int,
    ) -> dict[str, Any]:
        metadata = candidate.metadata_ or {}
        quality = dict(metadata.get("quality") or {})
        representative = dict(
            metadata.get("representative_observation") or {}
        )
        local_event_time = (
            payload.event_time_sec - payload.scene_start_sec
        )
        observation_time = _number(
            representative,
            "time_sec",
            "scene_time_sec",
        )
        if observation_time is None:
            observation_time = float(candidate.anchor_time_sec)
        relative_time = observation_time - local_event_time
        decay = float(policy["temporal_decay_seconds"])
        temporal = math.exp(-abs(relative_time) / decay)
        bbox_size = _number(
            quality,
            "bbox_area_ratio",
            "normalized_bbox_area",
        )
        sharpness = _number(
            quality,
            "sharpness_score",
            "sharpness",
        )
        center = _number(
            quality,
            "center_proximity",
            "screen_center_proximity",
        )
        ball_coverage = _number(quality, "ball_coverage")
        ball_proximity = _number(
            quality,
            "ball_proximity",
            "normalized_ball_proximity",
        )
        motion = _number(quality, "motion_proxy", "motion_score")
        trackability = _clamp(candidate.trackability_score)
        repeated_focus = _clamp(
            candidate.track_length_frames / max_observations
        )
        pre_presence = 1.0 if relative_time <= 0 else 0.0
        post_presence = 1.0 if relative_time >= 0 else 0.0
        closeup = (
            _clamp(bbox_size)
            if relative_time >= 0 and bbox_size is not None
            else 0.0
        )
        missing_visual = sum(
            value is None for value in (bbox_size, sharpness, center)
        )
        risk = _clamp(
            max(
                1.0 - trackability,
                0.25 if ball_proximity is None else 0.0,
                missing_visual / 10.0,
            )
        )
        features: dict[str, float | None] = {
            "event_relative_time_sec": round(relative_time, 6),
            "temporal_proximity": round(temporal, 6),
            "pre_event_presence": pre_presence,
            "post_event_presence": post_presence,
            "bbox_size": (
                round(_clamp(bbox_size), 6)
                if bbox_size is not None
                else None
            ),
            "sharpness": (
                round(_clamp(sharpness), 6)
                if sharpness is not None
                else None
            ),
            "center_proximity": (
                round(_clamp(center), 6)
                if center is not None
                else None
            ),
            "post_event_closeup": round(closeup, 6),
            "repeated_broadcast_focus": round(repeated_focus, 6),
            "ball_coverage": (
                round(_clamp(ball_coverage), 6)
                if ball_coverage is not None
                else None
            ),
            "ball_proximity": (
                round(_clamp(ball_proximity), 6)
                if ball_proximity is not None
                else None
            ),
            "motion_proxy": (
                round(_clamp(motion), 6)
                if motion is not None
                else None
            ),
            "trackability": round(trackability, 6),
            "risk": round(risk, 6),
        }
        weights = policy["event_relevance_weights"]
        weighted = 0.0
        available_weight = 0.0
        for name, weight in weights.items():
            value = features.get(name)
            if value is not None:
                weighted += float(value) * float(weight)
                available_weight += float(weight)
        relevance = weighted / available_weight if available_weight else 0.0
        recommendation_weights = policy["recommendation_weights"]
        recommendation = (
            relevance * float(recommendation_weights["event_relevance"])
            + trackability * float(recommendation_weights["trackability"])
            - risk * float(recommendation_weights["risk_penalty"])
        )
        reasons: list[str] = []
        if temporal >= 0.65:
            reasons.append("NEAR_EVENT_TIME")
        if closeup >= 0.12:
            reasons.append("POST_EVENT_CLOSEUP")
        if repeated_focus >= 0.5:
            reasons.append("REPEATED_BROADCAST_FOCUS")
        if ball_proximity is not None and ball_proximity >= 0.5:
            reasons.append("NEAR_BALL")
        if trackability >= 0.6:
            reasons.append("TRACKABLE")
        if not reasons:
            reasons.append("WEAK_EVENT_EVIDENCE")
        risks = ["PROVISIONAL_SHADOW_ONLY"]
        if trackability < 0.25:
            risks.append("LOW_TRACKABILITY")
        if missing_visual:
            risks.append("MISSING_VISUAL_QUALITY_FEATURES")
        if ball_proximity is None or ball_coverage is None:
            risks.append("MISSING_BALL_FEATURES")
        return {
            "candidate_id": candidate.candidate_id,
            "raw_features": features,
            "event_relevance_score": round(_clamp(relevance), 6),
            "trackability_score": round(trackability, 6),
            "recommendation_score": round(
                _clamp(recommendation),
                6,
            ),
            "reason_codes": reasons,
            "risk_codes": risks,
            "artifact_ids": dict(metadata.get("artifact_ids") or {}),
        }

    def _candidate_artifact_ids(
        self,
        revision_id: str,
        candidate_id: str,
    ) -> dict[str, str]:
        candidate = self.repository.get_candidate(
            revision_id=revision_id,
            candidate_id=candidate_id,
        )
        return (
            dict((candidate.metadata_ or {}).get("artifact_ids") or {})
            if candidate is not None
            else {}
        )
