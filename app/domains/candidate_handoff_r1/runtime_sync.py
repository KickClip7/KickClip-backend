from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domains.candidate_handoff_r1.artifacts import sha256_file, write_json_atomic
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateMemoryRevisionR1,
    EventCandidateOutboxR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
    EventCandidateSelectionR1,
)
from app.domains.candidate_handoff_r1.orchestrator import R1PipelineOrchestrator
from app.domains.tracking.execution import R1PipelineStage, R1ProcessingStatus
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.state_mapper import TrackingStateMapping
from app.domains.tracking.status import TrackingBackendStatus
from app.storage.local_storage import LocalStorage


BBOX_STATES = {"ACTIVE", "REACQUIRED"}
NULL_BBOX_STATES = {"LOST", "SEARCHING", "AMBIGUOUS", "ABSENT", "TERMINATED"}
VISIBLE_TERMINAL_SHOT_STATES = {
    "ACCEPTED",
    "TRACKED",
    "ACTIVE",
    "REACQUIRED",
    "ABSENT_CONFIRMED",
    "TARGET_ABSENT",
    "SEARCH_EXHAUSTED_NO_REVIEWABLE_CANDIDATE",
    "SEARCH_EXHAUSTED_NONE_OF_THESE",
    "SEARCH_EXHAUSTED_NON_PLAYER_ROLE",
    "SEARCH_EXHAUSTED_UNREVIEWABLE_GROUP_OCCLUSION",
}
UNRESOLVED_SHOT_MARKERS = {
    "UNRESOLVED_LOW_RESOLUTION",
    "AMBIGUOUS_REVIEW_REQUIRED",
    "SEARCHING",
    "UNRESOLVED",
}


class R1RuntimeSyncError(ValueError):
    pass


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: object) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def timeline_metrics(path: Path | None) -> dict[str, Any]:
    metrics = {
        "frame_count": 0,
        "active_or_reacquired_bbox_frames": 0,
        "searching_frames": 0,
        "lost_frames": 0,
        "absent_frames": 0,
        "ambiguous_frames": 0,
        "silent_wrong_player_switches": None,
        "timeline_valid": False,
        "timeline_sha256": None,
    }
    if path is None or not path.is_file():
        return metrics
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping) or not isinstance(value.get("frames"), list):
        return metrics
    seen: set[int] = set()
    for row in _rows(value.get("frames")):
        index = int(row.get("frame_index", -1))
        if index < 0 or index in seen:
            raise R1RuntimeSyncError("Runtime target timeline contains invalid frame indices.")
        seen.add(index)
        state = str(row.get("state") or "")
        bbox = row.get("bbox_xyxy")
        if state in BBOX_STATES:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise R1RuntimeSyncError(f"{state} frame requires a real bbox.")
            metrics["active_or_reacquired_bbox_frames"] += 1
        elif state in NULL_BBOX_STATES and bbox is not None:
            raise R1RuntimeSyncError(f"{state} frame must have a null bbox.")
        if state == "SEARCHING":
            metrics["searching_frames"] += 1
        elif state == "LOST":
            metrics["lost_frames"] += 1
        elif state == "ABSENT":
            metrics["absent_frames"] += 1
        elif state == "AMBIGUOUS":
            metrics["ambiguous_frames"] += 1
    metrics["frame_count"] = len(seen)
    provenance = _object(value.get("provenance"))
    metrics["silent_wrong_player_switches"] = provenance.get(
        "silent_wrong_player_switches",
        value.get("silent_wrong_player_switches"),
    )
    metrics["timeline_valid"] = True
    metrics["timeline_sha256"] = sha256_file(path)
    return metrics


def completion_state_from_runtime(
    state: Mapping[str, Any],
    *,
    timeline_valid: bool,
    preview_generated: bool,
    processing_outbox_count: int,
) -> TrackingBackendStatus:
    status = str(state.get("status") or "")
    pending = state.get("pending_action")
    shots = _rows(state.get("shots"))
    unresolved = [
        shot
        for shot in shots
        if str(shot.get("status") or "").upper() in UNRESOLVED_SHOT_MARKERS
        or "LOW_RESOLUTION" in str(shot.get("status") or "").upper()
    ]
    nonterminal = [
        shot
        for shot in shots
        if str(shot.get("status") or "").upper()
        not in VISIBLE_TERMINAL_SHOT_STATES | UNRESOLVED_SHOT_MARKERS
        and not str(shot.get("status") or "").upper().startswith("TERMINATED")
    ]
    if status in {"FAILED", "FATAL", "ERROR"}:
        return TrackingBackendStatus.FAILED
    if status in {"COMPLETE_WITH_SAFE_BLOCK", "BLOCKED"}:
        return TrackingBackendStatus.COMPLETED_SAFE_BLOCK
    # A durable review request takes precedence over unresolved-shot summary
    # markers. Otherwise a valid pending ambiguity can be misclassified as a
    # terminal COMPLETED_WITH_UNRESOLVED_GAPS job.
    if status in {"NEEDS_CONFIRMATION", "WAITING_CROSS_SHOT_CONFIRMATION"}:
        if isinstance(pending, Mapping) and pending.get("type") == "MEMORY_REVIEW":
            return TrackingBackendStatus.WAITING_MEMORY_REVIEW
        return TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION
    if status == "COMPLETE_WITH_UNRESOLVED_GAPS" or unresolved:
        return TrackingBackendStatus.COMPLETED_WITH_UNRESOLVED_GAPS
    if status == "COMPLETE":
        if pending or processing_outbox_count or nonterminal or not timeline_valid or not preview_generated:
            return TrackingBackendStatus.COMPLETED_SAFE_BLOCK
        return TrackingBackendStatus.COMPLETED
    return TrackingBackendStatus.RUNNING


def extract_pending_ambiguity(state: Mapping[str, Any]) -> dict[str, Any] | None:
    pending = _object(state.get("pending_action"))
    if str(pending.get("type") or "") != "CROSS_SHOT_CONFIRMATION":
        return None
    ambiguity_id = str(pending.get("ambiguity_id") or "")
    if not ambiguity_id:
        raise R1RuntimeSyncError("Runtime pending ambiguity has no ambiguity_id.")
    ambiguity = next(
        (
            item
            for item in _rows(state.get("ambiguities"))
            if str(item.get("ambiguity_id") or "") == ambiguity_id
        ),
        None,
    )
    if ambiguity is None:
        raise R1RuntimeSyncError("Runtime pending ambiguity is absent from state.")
    candidates = _rows(
        ambiguity.get("review_candidates")
        or ambiguity.get("candidates")
    )
    if not candidates:
        raise R1RuntimeSyncError("EMPTY_PENDING_AMBIGUITY_FORBIDDEN")
    return {**ambiguity, "candidates": candidates, "pending": pending}


class R1RuntimeStateSynchronizer:
    """Synchronize real subprocess state into the R1 DB/API contract."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.storage.project_root).as_posix()

    def _evidence(self, value: object, *, label: str) -> tuple[str, str]:
        path = Path(str(value or "")).resolve()
        if not path.is_file():
            raise R1RuntimeSyncError(f"Runtime {label} evidence is missing: {path}")
        return self._relative(path), sha256_file(path)

    def _sync_ambiguity(
        self,
        *,
        job: TrackingJob,
        pipeline: EventCandidatePipelineR1,
        payload: Mapping[str, Any],
    ) -> EventCandidateAmbiguityR1:
        ambiguity_id = str(payload["ambiguity_id"])
        candidates: list[dict[str, Any]] = []
        for raw in _rows(payload.get("candidates")):
            candidate_id = str(raw.get("candidate_id") or "")
            if not candidate_id:
                raise R1RuntimeSyncError("Runtime ambiguity candidate has no candidate_id.")
            manifest_path, manifest_sha = self._evidence(
                raw.get("manifest_path"), label="candidate manifest"
            )
            full_path, full_sha = self._evidence(
                raw.get("full_frame_context_path") or raw.get("best_frame_path"),
                label="full-frame context",
            )
            shot_path, shot_sha = self._evidence(
                raw.get("shot_clip_path"), label="full-shot clip"
            )
            gallery_path, gallery_sha = self._evidence(
                raw.get("reference_gallery_path") or raw.get("contact_sheet"),
                label="reference gallery",
            )
            score_evidence = _object(raw.get("score_evidence"))
            candidates.append(
                {
                    **raw,
                    "candidate_id": candidate_id,
                    "status": str(raw.get("status") or "PENDING"),
                    "manifest_path": manifest_path,
                    "manifest_sha256": str(raw.get("manifest_sha256") or manifest_sha),
                    "full_frame_context_path": full_path,
                    "full_frame_context_sha256": full_sha,
                    "shot_clip_path": shot_path,
                    "shot_clip_sha256": shot_sha,
                    "reference_gallery_path": gallery_path,
                    "reference_gallery_sha256": gallery_sha,
                    "score_evidence": {
                        **score_evidence,
                        "memory_revision_id": score_evidence.get("memory_revision_id")
                        or pipeline.current_memory_revision_id,
                        "memory_source_reference_count": score_evidence.get(
                            "memory_source_reference_count"
                        ),
                        "candidate_scoring_generation": score_evidence.get(
                            "candidate_scoring_generation"
                        )
                        or ((job.runtime_metadata or {}).get("scene_target_selection") or {}).get(
                            "candidate_scoring_generation"
                        ),
                    },
                }
            )
        reviewed_candidate_ids = {
            str(value)
            for value in self.db.scalars(
                select(EventCandidateReviewDecisionR1.candidate_id).where(
                    EventCandidateReviewDecisionR1.tracking_job_id
                    == job.tracking_job_id,
                    EventCandidateReviewDecisionR1.ambiguity_id == ambiguity_id,
                    EventCandidateReviewDecisionR1.candidate_id.is_not(None),
                )
            ).all()
            if value
        }
        if reviewed_candidate_ids:
            candidates = [
                item
                for item in candidates
                if str(item.get("candidate_id") or "") not in reviewed_candidate_ids
            ]
        if not candidates:
            raise R1RuntimeSyncError("EMPTY_PENDING_AMBIGUITY_FORBIDDEN")

        phase4b_evidence = [
            _object(item.get("score_evidence"))
            for item in candidates
            if _object(item.get("score_evidence")).get("phase4b_policy")
        ]
        if phase4b_evidence:
            if len(phase4b_evidence) != len(candidates):
                raise R1RuntimeSyncError(
                    "Phase 4-B ambiguity mixes incompatible score evidence."
                )
            memory_ids = {
                str(item.get("memory_revision_id") or "")
                for item in phase4b_evidence
            }
            memory_shas = {
                str(item.get("memory_revision_sha256") or "")
                for item in phase4b_evidence
            }
            generations = {
                int(item.get("candidate_scoring_generation") or 0)
                for item in phase4b_evidence
            }
            if (
                len(memory_ids) != 1
                or "" in memory_ids
                or len(memory_shas) != 1
                or "" in memory_shas
                or len(generations) != 1
                or min(generations) < 2
                or any(
                    item.get("backend_memory_used_by_phase4b_scoring") is not True
                    or item.get("automatic_target_confirmation") is not False
                    for item in phase4b_evidence
                )
            ):
                raise R1RuntimeSyncError(
                    "Phase 4-B score evidence is incomplete or inconsistent."
                )
            memory_id = next(iter(memory_ids))
            memory_sha = next(iter(memory_shas))
            generation = next(iter(generations))
            if pipeline.current_memory_revision_id != memory_id:
                raise R1RuntimeSyncError(
                    "Phase 4-B scoring did not use the current target memory revision."
                )
            memory_row = self.db.get(EventCandidateMemoryRevisionR1, memory_id)
            if memory_row is None or memory_row.artifact_sha256 != memory_sha:
                raise R1RuntimeSyncError(
                    "Phase 4-B target memory SHA provenance mismatch."
                )
            metadata = dict(job.runtime_metadata or {})
            scene = dict(metadata.get("scene_target_selection") or {})
            scene["candidate_scoring_generation"] = generation
            scene["current_target_memory_path"] = str(
                self.storage.resolve_path(memory_row.artifact_path)
            )
            scene["current_target_memory_sha256"] = memory_sha

            identity_rows = [
                _object(item.get("identity_negative_memory"))
                for item in phase4b_evidence
            ]
            identity_rows = [
                item for item in identity_rows if item.get("revision_id")
            ]
            if identity_rows:
                revision_ids = {str(item.get("revision_id") or "") for item in identity_rows}
                manifest_shas = {
                    str(item.get("manifest_sha256") or "") for item in identity_rows
                }
                embedding_shas = {
                    str(item.get("embeddings_sha256") or "") for item in identity_rows
                }
                if (
                    len(identity_rows) != len(phase4b_evidence)
                    or len(revision_ids) != 1
                    or "" in revision_ids
                    or len(manifest_shas) != 1
                    or "" in manifest_shas
                    or len(embedding_shas) != 1
                    or "" in embedding_shas
                    or any(
                        item.get("user_confirmed_only") is not True
                        or item.get("automatic_target_confirmation") is not False
                        for item in identity_rows
                    )
                ):
                    raise R1RuntimeSyncError(
                        "Phase 4-B identity-negative memory provenance is inconsistent."
                    )
                identity = identity_rows[0]
                manifest_path = Path(str(identity.get("manifest_path") or "")).resolve()
                embeddings_path = Path(str(identity.get("embeddings_path") or "")).resolve()
                if (
                    not manifest_path.is_file()
                    or sha256_file(manifest_path) != next(iter(manifest_shas))
                    or not embeddings_path.is_file()
                    or sha256_file(embeddings_path) != next(iter(embedding_shas))
                ):
                    raise R1RuntimeSyncError(
                        "Phase 4-B identity-negative memory artifact changed or is missing."
                    )
                scene["current_identity_negative_memory_revision_id"] = next(
                    iter(revision_ids)
                )
                scene["current_identity_negative_memory_path"] = str(manifest_path)
                scene["current_identity_negative_memory_sha256"] = next(
                    iter(manifest_shas)
                )
                scene["current_identity_negative_embeddings_path"] = str(
                    embeddings_path
                )
                scene["current_identity_negative_embeddings_sha256"] = next(
                    iter(embedding_shas)
                )
                scene["current_identity_negative_embedding_count"] = int(
                    identity.get("embedding_count") or 0
                )

            metadata["scene_target_selection"] = scene
            job.runtime_metadata = metadata

        first = candidates[0]
        artifact = Path(job.output_directory) / "ambiguities" / f"{ambiguity_id}.json"
        artifact_doc = {
            "schema_version": "kickclip.runtime_cross_shot_ambiguity.r1",
            "ambiguity_id": ambiguity_id,
            "tracking_job_id": job.tracking_job_id,
            "shot_id": str(payload.get("shot_id") or first.get("shot_id") or ""),
            "status": "WAITING",
            "recommended_candidate": payload.get("recommended_candidate"),
            "candidates": candidates,
            "automatic_target_confirmation": False,
        }
        artifact_sha = write_json_atomic(artifact, artifact_doc)
        row = self.db.scalar(
            select(EventCandidateAmbiguityR1).where(
                EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id,
                EventCandidateAmbiguityR1.ambiguity_id == ambiguity_id,
            )
        )
        values = {
            "shot_id": artifact_doc["shot_id"],
            "status": "WAITING",
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "candidates": candidates,
            "full_frame_context_path": first["full_frame_context_path"],
            "full_frame_context_sha256": first["full_frame_context_sha256"],
            "shot_clip_path": first["shot_clip_path"],
            "shot_clip_sha256": first["shot_clip_sha256"],
            "artifact_path": self._relative(artifact),
            "artifact_sha256": artifact_sha,
            "generation": pipeline.generation + 1,
        }
        if row is None:
            row = EventCandidateAmbiguityR1(
                tracking_job_id=job.tracking_job_id,
                ambiguity_id=ambiguity_id,
                **values,
            )
            self.db.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        return row

    def _sync_initial_memory(
        self,
        *,
        job: TrackingJob,
        pipeline: EventCandidatePipelineR1,
        selection: EventCandidateSelectionR1,
        state: Mapping[str, Any],
    ) -> None:
        pending = _object(state.get("pending_action"))
        if pending.get("type") != "MEMORY_REVIEW":
            return
        runtime = _object(state.get("runtime"))
        path = Path(str(runtime.get("memory_revision_path") or "")).resolve()
        expected_sha = str(runtime.get("memory_revision_sha256") or "")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise R1RuntimeSyncError("Initial immutable target memory is missing or changed.")
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, Mapping):
            raise R1RuntimeSyncError("Initial target memory is not a JSON object.")
        revision_id = str(document.get("memory_revision_id") or "")
        if not revision_id or document.get("selection_id") != selection.selection_id:
            raise R1RuntimeSyncError("Initial target memory provenance mismatch.")
        row = self.db.get(EventCandidateMemoryRevisionR1, revision_id)
        references = _rows(document.get("references"))
        if row is None:
            if not references or int(document.get("active_or_reacquired_frame_count") or 0) < 1:
                raise R1RuntimeSyncError("Initial target memory has no real ACTIVE evidence.")
            banks: dict[str, list[dict[str, Any]]] = {}
            for reference in references:
                banks.setdefault(str(reference.get("scale_class") or "unknown"), []).append(reference)
            row = EventCandidateMemoryRevisionR1(
                memory_revision_id=revision_id,
                tracking_job_id=job.tracking_job_id,
                source_candidate_id=selection.candidate_id,
                source_shot_id=selection.shot_id,
                source_tracklet_id=selection.tracklet_id,
                reviewer_id=selection.owner_id,
                confirmation_decision_id=f"initial_memory_review:{job.tracking_job_id}",
                previous_memory_revision_id=None,
                previous_memory_sha256=None,
                new_memory_sha256=expected_sha,
                reference_frame_ids=[int(item["frame_id"]) for item in references],
                references=references,
                scale_banks=banks,
                artifact_path=self._relative(path),
                artifact_sha256=expected_sha,
            )
            self.db.add(row)
            self.db.flush()
        elif row.tracking_job_id != job.tracking_job_id or row.artifact_sha256 != expected_sha:
            raise R1RuntimeSyncError("Initial target memory row provenance mismatch.")
        job.current_memory_revision_id = revision_id
        pipeline.current_memory_revision_id = revision_id
        metadata = dict(job.runtime_metadata or {})
        scene = dict(metadata.get("scene_target_selection") or {})
        scene.update(
            {
                "current_target_memory_path": str(path),
                "current_target_memory_sha256": expected_sha,
                "candidate_scoring_generation": max(
                    1, int(scene.get("candidate_scoring_generation") or 1)
                ),
            }
        )
        metadata["scene_target_selection"] = scene
        job.runtime_metadata = metadata

    def sync(
        self,
        *,
        job: TrackingJob,
        state: Mapping[str, Any],
        mapping: TrackingStateMapping,
    ) -> None:
        pipeline = self.db.scalar(
            select(EventCandidatePipelineR1).where(
                EventCandidatePipelineR1.tracking_job_id == job.tracking_job_id
            )
        )
        if pipeline is None:
            raise R1RuntimeSyncError("R1 pipeline row is missing.")
        selection = self.db.get(EventCandidateSelectionR1, pipeline.selection_id)
        if selection is None:
            raise R1RuntimeSyncError("R1 selection row is missing.")

        self._sync_initial_memory(
            job=job,
            pipeline=pipeline,
            selection=selection,
            state=state,
        )

        pending = extract_pending_ambiguity(state)
        ambiguity = None
        if pending is not None:
            incoming_id = str(pending.get("ambiguity_id") or "")
            current_pending = None
            if pipeline.pending_ambiguity_id and pipeline.pending_ambiguity_id != incoming_id:
                current_pending = self.db.scalar(
                    select(EventCandidateAmbiguityR1).where(
                        EventCandidateAmbiguityR1.tracking_job_id
                        == job.tracking_job_id,
                        EventCandidateAmbiguityR1.ambiguity_id
                        == pipeline.pending_ambiguity_id,
                        EventCandidateAmbiguityR1.status == "WAITING",
                    )
                )
            # A DB-created R14 catalog generation is newer than the stale runtime
            # pending action. Human decisions must not be rolled back to WAITING.
            ambiguity = current_pending or self._sync_ambiguity(
                job=job,
                pipeline=pipeline,
                payload=pending,
            )
            pipeline.pending_ambiguity_id = ambiguity.ambiguity_id
            pipeline.next_shot_id = ambiguity.shot_id
            pipeline.current_shot_id = ambiguity.shot_id
            pipeline.pipeline_stage = R1PipelineStage.WAITING_CROSS_SHOT_CONFIRMATION.value
            pipeline.processing_status = R1ProcessingStatus.WAITING.value
            pipeline.completed_at = None
            pipeline.failure_code = None
            job.completed_at = None
            job.finished_at = None
            job.failure_code = None
        else:
            pipeline.pending_ambiguity_id = None
            pipeline.next_shot_id = None

        # The subprocess result is the durable acknowledgement for the queued
        # review action. Complete those outbox rows before evaluating terminal gates.
        self.db.execute(
            update(EventCandidateOutboxR1)
            .where(
                EventCandidateOutboxR1.tracking_job_id == job.tracking_job_id,
                EventCandidateOutboxR1.status == "PENDING",
            )
            .values(
                status="COMPLETED",
                completed_at=datetime.now(timezone.utc),
                artifact_path=self._relative(Path(job.pipeline_state_path)),
                artifact_sha256=sha256_file(Path(job.pipeline_state_path)),
            )
        )
        outbox_count = 0
        timeline_path = Path(job.timeline_path).resolve() if job.timeline_path else None
        metrics = timeline_metrics(timeline_path)
        preview_generated = bool(
            job.tracking_preview_path
            and Path(job.tracking_preview_path).is_file()
            and job.target_centered_preview_path
            and Path(job.target_centered_preview_path).is_file()
        )
        completion = completion_state_from_runtime(
            state,
            timeline_valid=bool(metrics["timeline_valid"]),
            preview_generated=preview_generated,
            processing_outbox_count=outbox_count,
        )
        job.status = completion.value if completion.value != TrackingBackendStatus.RUNNING.value else mapping.backend_status.value
        shots = _rows(state.get("shots"))
        unresolved = [
            item for item in shots
            if str(item.get("status") or "").upper() in UNRESOLVED_SHOT_MARKERS
            or "LOW_RESOLUTION" in str(item.get("status") or "").upper()
        ]
        terminal = [
            item for item in shots
            if str(item.get("status") or "").upper() in VISIBLE_TERMINAL_SHOT_STATES
            or str(item.get("status") or "").upper().startswith("TERMINATED")
        ]
        summary = dict(pipeline.summary or {})
        summary.update(
            {
                **metrics,
                "reviewed_shots_total": len(shots),
                "reviewed_shots_terminal": len(terminal),
                "unresolved_shot_count": len(unresolved),
                "preview_generated": preview_generated,
                "target_centered_frames_with_real_bbox": metrics[
                    "active_or_reacquired_bbox_frames"
                ],
                "runtime_status": state.get("status"),
                "runtime_decision": state.get("decision"),
                "automatic_target_confirmation": False,
                "phase4b_identity_negative_memory": _object(
                    _object(state.get("runtime")).get("phase4b_identity_negative_memory")
                ),
                "phase4b_persistent_role_negative_memory": _object(
                    _object(state.get("runtime")).get(
                        "phase4b_persistent_role_negative_memory"
                    )
                ),
            }
        )
        pipeline.summary = summary
        pipeline.state_artifact_path = self._relative(Path(job.pipeline_state_path))
        pipeline.state_artifact_sha256 = sha256_file(Path(job.pipeline_state_path))
        if job.current_memory_revision_id:
            pipeline.current_memory_revision_id = job.current_memory_revision_id
        if completion == TrackingBackendStatus.COMPLETED:
            pipeline.pipeline_stage = R1PipelineStage.COMPLETED.value
            pipeline.processing_status = R1ProcessingStatus.COMPLETED.value
            pipeline.completed_at = datetime.now(timezone.utc)
        elif completion == TrackingBackendStatus.COMPLETED_WITH_UNRESOLVED_GAPS:
            pipeline.pipeline_stage = R1PipelineStage.COMPLETED_WITH_UNRESOLVED_GAPS.value
            pipeline.processing_status = R1ProcessingStatus.COMPLETED_WITH_UNRESOLVED_GAPS.value
            pipeline.completed_at = datetime.now(timezone.utc)
        elif completion == TrackingBackendStatus.COMPLETED_SAFE_BLOCK:
            pipeline.pipeline_stage = R1PipelineStage.COMPLETED_SAFE_BLOCK.value
            pipeline.processing_status = R1ProcessingStatus.COMPLETED_SAFE_BLOCK.value
            pipeline.completed_at = datetime.now(timezone.utc)
            pipeline.failure_code = str(state.get("failure_code") or state.get("decision") or "SAFE_BLOCK")
        elif completion == TrackingBackendStatus.FAILED:
            pipeline.pipeline_stage = R1PipelineStage.FAILED.value
            pipeline.processing_status = R1ProcessingStatus.FAILED.value
            pipeline.failure_code = str(state.get("failure_code") or "RUNTIME_FAILED")
        elif completion == TrackingBackendStatus.WAITING_MEMORY_REVIEW:
            pipeline.pipeline_stage = R1PipelineStage.WAITING_MEMORY_REVIEW.value
            pipeline.processing_status = R1ProcessingStatus.WAITING.value
            pipeline.completed_at = None
            pipeline.failure_code = None
            job.completed_at = None
            job.finished_at = None
            job.failure_code = None
        elif ambiguity is None:
            pipeline.pipeline_stage = R1PipelineStage.SEARCHING_NEXT_SHOT.value
            pipeline.processing_status = R1ProcessingStatus.PROCESSING.value

        R1PipelineOrchestrator(self.db).synchronize(
            job=job,
            selection=selection,
            pipeline=pipeline,
            ambiguity=ambiguity,
        )
