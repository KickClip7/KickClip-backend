from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.candidate_handoff_r1.artifacts import (
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateHandoffPointerR1,
    EventCandidateMemoryRevisionR1,
    EventCandidatePipelineR1,
    EventCandidateSelectionR1,
)
from app.domains.tracking.execution import (
    R1_EXECUTION_KIND,
    R1PipelineStage,
    R1ProcessingStatus,
)
from app.domains.tracking.model import TrackingJob
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id


class R1PipelineOrchestrator:
    """Persist R1 DB/pointer state without simulating tracking or candidate search."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected object artifact: {path}")
        return value

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.storage.project_root).as_posix()

    def _candidate_manifest_path(self, job: TrackingJob) -> Path:
        metadata = (job.runtime_metadata or {}).get("event_candidate_handoff_r1")
        if not isinstance(metadata, dict) or not metadata.get("candidate_manifest_path"):
            raise ValueError("Immutable candidate manifest path is missing from R1 runtime metadata.")
        return Path(str(metadata["candidate_manifest_path"])).resolve()

    def _manifest(self, job: TrackingJob, candidate_id: str) -> tuple[Path, dict[str, Any]]:
        initial = self._candidate_manifest_path(job)
        if candidate_id == str((job.runtime_metadata or {}).get("event_candidate_handoff_r1", {}).get("selected_candidate_id") or ""):
            path = initial
        else:
            # Runtime ambiguities must provide a concrete immutable candidate manifest.
            ambiguity = self.db.scalar(
                select(EventCandidateAmbiguityR1)
                .where(EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id)
                .order_by(EventCandidateAmbiguityR1.generation.desc())
            )
            candidate = next(
                (
                    item
                    for item in (ambiguity.candidates if ambiguity else [])
                    if str(item.get("candidate_id")) == candidate_id
                ),
                None,
            )
            if not candidate or not candidate.get("manifest_path"):
                raise ValueError("Runtime candidate manifest provenance is missing.")
            path = self.storage.resolve_path(str(candidate["manifest_path"]))
        document = self._load(path)
        if str(document.get("candidate_id")) != candidate_id:
            raise ValueError("CANDIDATE_MEDIA_MAPPING_MISMATCH")
        expected = str(
            next(
                (
                    item.get("manifest_sha256")
                    for row in self.db.scalars(
                        select(EventCandidateAmbiguityR1).where(
                            EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id
                        )
                    )
                    for item in (row.candidates or [])
                    if str(item.get("candidate_id")) == candidate_id
                ),
                "",
            )
        )
        if expected and sha256_file(path) != expected:
            raise ValueError("CANDIDATE_MEDIA_MAPPING_MISMATCH")
        return path, document

    def initialize(
        self,
        *,
        job: TrackingJob,
        selection: EventCandidateSelectionR1,
        state_sha: str,
        generation: int = 1,
    ) -> EventCandidatePipelineR1:
        pointer_path = Path(job.output_directory) / "current_state_pointer.json"
        pipeline = EventCandidatePipelineR1(
            tracking_job_id=job.tracking_job_id,
            selection_id=selection.selection_id,
            generation=max(1, generation),
            execution_kind=R1_EXECUTION_KIND,
            pipeline_stage=R1PipelineStage.SELECTED_SHOT_TRACKING.value,
            processing_status=R1ProcessingStatus.READY.value,
            current_shot_id=selection.shot_id,
            pending_ambiguity_id=None,
            next_shot_id=None,
            state_artifact_path=self._relative(Path(job.pipeline_state_path)),
            state_artifact_sha256=state_sha,
            pointer_snapshot_path=self._relative(pointer_path),
            pointer_snapshot_sha256="0" * 64,
            summary={
                "reviewed_shots_source": "IMMUTABLE_REVIEWED_SHOT_BOUNDARIES",
                "candidate_source": "RUNTIME_DYNAMIC_SCENE_DISCOVERY",
                "bootstrap_observations_are_tracking_success": False,
                "automatic_target_confirmation": False,
                "confirmation_count": 0,
                "memory_revision_count": 0,
                "silent_wrong_player_switches": 0,
            },
        )
        self.db.add(pipeline)
        self.db.flush()
        self.synchronize(job=job, selection=selection, pipeline=pipeline, ambiguity=None)
        return pipeline

    def build_memory(
        self,
        *,
        job: TrackingJob,
        candidate_id: str,
        reviewer_id: str,
        decision_id: str,
    ) -> tuple[EventCandidateMemoryRevisionR1, dict[str, Any]]:
        manifest_path, manifest = self._manifest(job, candidate_id)
        quality = dict(manifest.get("quality") or {})
        if not quality.get("identity_pure"):
            raise ValueError("INVALID_TARGET_MEMORY_REFERENCE: tracklet is not identity-pure")
        references: list[dict[str, Any]] = []
        banks: dict[str, list[dict[str, Any]]] = {
            "wide": [],
            "medium": [],
            "close-up": [],
        }
        for ref in manifest.get("reference_gallery") or []:
            path = (manifest_path.parent / str(ref["path"])).resolve()
            digest = sha256_file(path)
            if digest != ref.get("crop_sha256"):
                raise ValueError("INVALID_TARGET_MEMORY_REFERENCE: crop SHA mismatch")
            item = {
                "frame_id": int(ref["frame"]),
                "path": self._relative(path),
                "crop_sha256": digest,
                "scale_category": str(ref.get("scale_class") or "unknown"),
                "reference_quality": str(quality.get("reviewability") or "UNKNOWN"),
            }
            references.append(item)
            banks.setdefault(item["scale_category"], []).append(item)
        if not references:
            raise ValueError("INVALID_TARGET_MEMORY_REFERENCE: no immutable reference crops")
        previous = self.db.scalar(
            select(EventCandidateMemoryRevisionR1)
            .where(EventCandidateMemoryRevisionR1.tracking_job_id == job.tracking_job_id)
            .order_by(EventCandidateMemoryRevisionR1.created_at.desc())
        )
        revision_id = generate_prefixed_id("ecmem")
        document = {
            "schema_version": "kickclip.target_memory_revision.r1_2",
            "immutable": True,
            "memory_revision_id": revision_id,
            "previous_memory_revision_id": previous.memory_revision_id if previous else None,
            "previous_memory_sha256": previous.new_memory_sha256 if previous else None,
            "source_candidate_id": candidate_id,
            "source_shot_id": str(manifest["shot_id"]),
            "source_tracklet_id": str(manifest["tracklet_id"]),
            "reviewer": reviewer_id,
            "confirmation_decision_id": decision_id,
            "reference_frame_ids": [item["frame_id"] for item in references],
            "references": references,
            "reference_count": len(references),
            "scale_banks": banks,
            "scale_banks_used": [key for key, value in banks.items() if value],
            "candidate_scoring_generation": int(
                ((job.runtime_metadata or {}).get("scene_target_selection") or {}).get(
                    "candidate_scoring_generation", 1
                )
            ) + 1,
            "cross_scale_absolute_comparison": False,
            "automatic_target_confirmation": False,
        }
        document["new_memory_sha256"] = canonical_sha256(document)
        path = Path(job.output_directory) / "target_memory_revisions" / f"{revision_id}.json"
        artifact_sha = write_json_atomic(path, document)
        row = EventCandidateMemoryRevisionR1(
            memory_revision_id=revision_id,
            tracking_job_id=job.tracking_job_id,
            source_candidate_id=candidate_id,
            source_shot_id=str(manifest["shot_id"]),
            source_tracklet_id=str(manifest["tracklet_id"]),
            reviewer_id=reviewer_id,
            confirmation_decision_id=decision_id,
            previous_memory_revision_id=previous.memory_revision_id if previous else None,
            previous_memory_sha256=previous.new_memory_sha256 if previous else None,
            new_memory_sha256=str(document["new_memory_sha256"]),
            reference_frame_ids=document["reference_frame_ids"],
            references=references,
            scale_banks=banks,
            artifact_path=self._relative(path),
            artifact_sha256=artifact_sha,
        )
        return row, document

    def synchronize(
        self,
        *,
        job: TrackingJob,
        selection: EventCandidateSelectionR1,
        pipeline: EventCandidatePipelineR1,
        ambiguity: EventCandidateAmbiguityR1 | None,
    ) -> None:
        """Write a DB-derived pointer snapshot, never overwrite runtime pipeline_state.json."""
        pipeline.generation = max(1, int(pipeline.generation or 0) + 1)
        snapshot = {
            "schema_version": "kickclip.handoff_pointer.r1_2",
            "authoritative_source": "event_candidate_pipelines_r1",
            "selection_id": selection.selection_id,
            "current_tracking_job_id": job.tracking_job_id,
            "current_stage": pipeline.pipeline_stage,
            "processing_status": pipeline.processing_status,
            "pending_ambiguity_id": pipeline.pending_ambiguity_id,
            "current_memory_revision_id": pipeline.current_memory_revision_id,
            "latest_decision_id": pipeline.latest_decision_id,
            "generation": pipeline.generation,
            "runtime_pipeline_state_path": job.pipeline_state_path,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "automatic_target_confirmation": False,
        }
        snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
        pointer_path = self.storage.resolve_path(pipeline.pointer_snapshot_path)
        pointer_file_sha = write_json_atomic(pointer_path, snapshot)
        pipeline.pointer_snapshot_sha256 = pointer_file_sha

        pointer = self.db.scalar(
            select(EventCandidateHandoffPointerR1).where(
                EventCandidateHandoffPointerR1.project_id == selection.project_id,
                EventCandidateHandoffPointerR1.revision_id == selection.revision_id,
                EventCandidateHandoffPointerR1.event_id == selection.event_id,
                EventCandidateHandoffPointerR1.scene_id == selection.scene_id,
            )
        )
        if pointer is None:
            pointer = EventCandidateHandoffPointerR1(
                project_id=selection.project_id,
                revision_id=selection.revision_id,
                event_id=selection.event_id,
                scene_id=selection.scene_id,
                state_artifact_path=pipeline.pointer_snapshot_path,
                state_artifact_sha256=pointer_file_sha,
                task_state=pipeline.pipeline_stage,
                generation=0,
                metadata_={},
            )
            self.db.add(pointer)
        pointer.current_selection_id = selection.selection_id
        pointer.current_tracking_job_id = job.tracking_job_id
        pointer.current_ambiguity_id = pipeline.pending_ambiguity_id
        pointer.current_stage = pipeline.pipeline_stage
        pointer.current_memory_revision_id = pipeline.current_memory_revision_id
        pointer.latest_decision_id = pipeline.latest_decision_id
        pointer.generation = pipeline.generation
        pointer.state_artifact_path = pipeline.pointer_snapshot_path
        pointer.state_artifact_sha256 = pointer_file_sha
        pointer.snapshot_sha256 = str(snapshot["snapshot_sha256"])
        pointer.task_state = pipeline.pipeline_stage
        pointer.metadata_ = {
            "authoritative_source": "event_candidate_pipelines_r1",
            "runtime_state_is_source_of_tracking_truth": True,
            "automatic_target_confirmation": False,
        }

        job.execution_kind = R1_EXECUTION_KIND
        job.pipeline_stage = pipeline.pipeline_stage
        job.current_stage = pipeline.pipeline_stage
        job.processing_status = pipeline.processing_status
        job.current_shot_id = pipeline.current_shot_id
        job.pending_ambiguity_id = pipeline.pending_ambiguity_id
        job.last_completed_ambiguity_id = pipeline.last_completed_ambiguity_id
        job.latest_decision_id = pipeline.latest_decision_id
        job.current_memory_revision_id = pipeline.current_memory_revision_id
        job.next_shot_id = pipeline.next_shot_id
        job.completed_at = pipeline.completed_at
        job.failure_code = pipeline.failure_code
        metadata = dict(job.runtime_metadata or {})
        r1 = dict(metadata.get("event_candidate_handoff_r1") or {})
        r1.update(
            {
                "pipeline_id": pipeline.pipeline_id,
                "generation": pipeline.generation,
                "pointer_snapshot_sha256": pointer_file_sha,
            }
        )
        metadata["event_candidate_handoff_r1"] = r1
        job.runtime_metadata = metadata


def recover_r1_pipelines(db: Session) -> dict[str, Any]:
    """Validate existing R1 DB pointers; runtime executor handles process recovery."""
    recovered = 0
    failures: list[str] = []
    orchestrator = R1PipelineOrchestrator(db)
    pipelines = db.scalars(
        select(EventCandidatePipelineR1)
        .where(EventCandidatePipelineR1.execution_kind == R1_EXECUTION_KIND)
        .order_by(EventCandidatePipelineR1.updated_at.asc())
    ).all()
    for pipeline in pipelines:
        job = db.get(TrackingJob, pipeline.tracking_job_id)
        selection = db.get(EventCandidateSelectionR1, pipeline.selection_id)
        if job is None or selection is None:
            failures.append(pipeline.tracking_job_id)
            continue
        ambiguity = None
        if pipeline.pending_ambiguity_id:
            ambiguity = db.scalar(
                select(EventCandidateAmbiguityR1).where(
                    EventCandidateAmbiguityR1.tracking_job_id == job.tracking_job_id,
                    EventCandidateAmbiguityR1.ambiguity_id == pipeline.pending_ambiguity_id,
                )
            )
            if ambiguity is None or not ambiguity.candidates:
                pipeline.processing_status = R1ProcessingStatus.RECOVERY_REQUIRED.value
                pipeline.failure_code = "EMPTY_PENDING_AMBIGUITY_FORBIDDEN"
                failures.append(job.tracking_job_id)
                continue
        orchestrator.synchronize(
            job=job,
            selection=selection,
            pipeline=pipeline,
            ambiguity=ambiguity,
        )
        recovered += 1
    db.commit()
    return {"recovered": recovered, "failures": failures}
