from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.api.v1 import event_candidate_handoff_r1 as candidate_api
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateHandoffPointerR1,
    EventCandidateOutboxR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
    EventCandidateSelectionR1,
)
from app.domains.candidate_handoff_r1.schema import CandidateReviewDecisionRequest
from app.domains.candidate_handoff_r1.service import CandidateHandoffR1Service
from app.domains.tracking.execution import R1_EXECUTION_KIND
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.status import TrackingBackendStatus


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(root: Path, candidate_id: str) -> dict:
    candidate_root = root / candidate_id
    candidate_root.mkdir(parents=True)
    paths = {}
    for name in (
        "manifest",
        "full_frame_context",
        "shot_clip",
        "reference_gallery",
    ):
        suffix = ".json" if name == "manifest" else ".bin"
        path = candidate_root / f"{name}{suffix}"
        if name == "manifest":
            path.write_text(
                json.dumps({"candidate_id": candidate_id, "quality": {"identity_pure": True}}),
                encoding="utf-8",
            )
        else:
            path.write_bytes(f"{candidate_id}:{name}".encode())
        paths[f"{name}_path"] = path.resolve().relative_to(Path.cwd()).as_posix()
        paths[f"{name}_sha256"] = _sha(path)
    return {
        "candidate_id": candidate_id,
        "shot_id": "shot_0004",
        "status": "PENDING",
        "identity_purity": {"passed": True},
        "identity_observability": {"passed": True},
        "negative_review_gate": {"passed": True},
        "role_confusion": {"passed": True},
        **paths,
    }


def test_r14_reviews_current_candidates_then_catalog_without_runtime(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    db = SessionLocal()
    root = Path("storage") / f"pytest_r14_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        candidates = [_candidate(root, f"candidate_{letter}") for letter in "abcde"]
        state_path = root / "pipeline_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "status": "NEEDS_CONFIRMATION",
                    "pending_action": {
                        "type": "CROSS_SHOT_CONFIRMATION",
                        "ambiguity_id": "ambiguity_phase4b_shot_0004_g002",
                    },
                    "shot_search_results": {
                        "shot_0004": {"review_catalog_candidates": candidates}
                    },
                }
            ),
            encoding="utf-8",
        )
        user = User(
            user_id="user-r14",
            email="r14@example.test",
            password_hash="hash",
            display_name="R14",
            developer_mode_enabled=True,
        )
        job = TrackingJob(
            tracking_job_id="trk-r14",
            owner_id=user.user_id,
            match_id="match-r14",
            project_id="project-r14",
            media_asset_id="media-r14",
            test_name=f"r14-{uuid.uuid4().hex}",
            initial_bbox=[1, 2, 3, 4],
            device="cpu",
            status=TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value,
            execution_kind=R1_EXECUTION_KIND,
            processing_status="WAITING",
            current_shot_id="shot_0004",
            pipeline_status="NEEDS_CONFIRMATION",
            pending_action_type="CROSS_SHOT_CONFIRMATION",
            pending_ambiguity_id="ambiguity_phase4b_shot_0004_g002",
            output_directory=str(root.resolve()),
            pipeline_state_path=str(state_path.resolve()),
            queued_action={},
            artifact_index={},
            runtime_metadata={},
        )
        selection = EventCandidateSelectionR1(
            selection_id="selection-r14",
            project_id="project-r14",
            owner_id=user.user_id,
            revision_id="revision-r14",
            event_id="event-r14",
            scene_id="scene-r14",
            ranking_id="ranking-r14",
            shortlist_patch_id="patch-r14",
            discovery_id="discovery-r14",
            candidate_id="initial-target",
            shot_id="shot_0001",
            tracklet_id="track-initial",
            selected_at=datetime.now(timezone.utc),
            candidate_manifest_sha256="1" * 64,
            candidate_media_bundle_sha256="2" * 64,
            source_video_sha256="3" * 64,
            reviewed_shot_boundaries_sha256="4" * 64,
            selection_artifact_path=(root / "selection.json").resolve().relative_to(Path.cwd()).as_posix(),
            selection_artifact_sha256="5" * 64,
            media_bundle_manifest_path=(root / "bundle.json").resolve().relative_to(Path.cwd()).as_posix(),
            tracking_job_id=job.tracking_job_id,
        )
        pointer_path = root / "pointer.json"
        pipeline = EventCandidatePipelineR1(
            tracking_job_id=job.tracking_job_id,
            selection_id=selection.selection_id,
            generation=2,
            execution_kind=R1_EXECUTION_KIND,
            pipeline_stage="WAITING_CROSS_SHOT_CONFIRMATION",
            processing_status="WAITING",
            current_shot_id="shot_0004",
            pending_ambiguity_id="ambiguity_phase4b_shot_0004_g002",
            state_artifact_path=state_path.resolve().relative_to(Path.cwd()).as_posix(),
            state_artifact_sha256=_sha(state_path),
            pointer_snapshot_path=pointer_path.resolve().relative_to(Path.cwd()).as_posix(),
            pointer_snapshot_sha256="0" * 64,
            summary={},
        )
        ambiguity = EventCandidateAmbiguityR1(
            tracking_job_id=job.tracking_job_id,
            ambiguity_id="ambiguity_phase4b_shot_0004_g002",
            shot_id="shot_0004",
            status="WAITING",
            candidate_ids=[row["candidate_id"] for row in candidates[:3]],
            candidates=candidates[:3],
            full_frame_context_path=candidates[0]["full_frame_context_path"],
            full_frame_context_sha256=candidates[0]["full_frame_context_sha256"],
            shot_clip_path=candidates[0]["shot_clip_path"],
            shot_clip_sha256=candidates[0]["shot_clip_sha256"],
            artifact_path=state_path.resolve().relative_to(Path.cwd()).as_posix(),
            artifact_sha256=_sha(state_path),
            generation=2,
        )
        db.add_all([user, job, selection, pipeline, ambiguity])
        db.commit()

        submitted = []
        monkeypatch.setattr(
            "app.domains.candidate_handoff_r1.service.get_r1_tracking_executor",
            lambda: type("Executor", (), {"submit": lambda self, job_id: submitted.append(job_id)})(),
        )
        service = CandidateHandoffR1Service(db)

        monkeypatch.setattr(
            candidate_api,
            "require_tracking_job_access",
            lambda *_args, **_kwargs: job,
        )
        gallery_response = candidate_api.get_ambiguity_candidate_media(
            job.tracking_job_id,
            ambiguity.ambiguity_id,
            "candidate_a",
            "reference-gallery",
            db,
            user,
        )
        assert Path(gallery_response.path).read_bytes() == b"candidate_a:reference_gallery"

        def reject(ambiguity_id: str, candidate_id: str, key: str):
            return service.record_review_decision(
                job=job,
                ambiguity_id=ambiguity_id,
                request=CandidateReviewDecisionRequest(
                    state="DIFFERENT_PLAYER",
                    candidate_id=candidate_id,
                    idempotency_key=key,
                ),
                user=user,
            )

        first = reject(ambiguity.ambiguity_id, "candidate_a", "r14-key-a")
        assert first.metadata_["continuation"] == "REVIEW_NEXT_CANDIDATE"
        assert first.metadata_["remaining_candidate_count"] == 2
        assert ambiguity.candidate_ids == ["candidate_b", "candidate_c"]
        assert submitted == []

        retry = reject(ambiguity.ambiguity_id, "candidate_a", "r14-key-a")
        assert retry.decision_id == first.decision_id
        assert db.scalar(select(func.count(EventCandidateOutboxR1.outbox_id))) == 1
        with pytest.raises(ValueError, match="Candidate is not in the pending ambiguity"):
            reject(ambiguity.ambiguity_id, "candidate_a", "r14-key-a-stale")

        second = reject(ambiguity.ambiguity_id, "candidate_b", "r14-key-b")
        assert second.metadata_["remaining_candidate_count"] == 1
        assert ambiguity.candidate_ids == ["candidate_c"]
        third = reject(ambiguity.ambiguity_id, "candidate_c", "r14-key-c")
        assert third.metadata_["continuation"] == "REVIEW_NEXT_BATCH"
        assert third.metadata_["remaining_candidate_count"] == 0
        assert third.metadata_["next_candidate_id"] is None
        next_id = str(third.metadata_["next_ambiguity_id"])
        next_ambiguity = db.scalar(
            select(EventCandidateAmbiguityR1).where(
                EventCandidateAmbiguityR1.ambiguity_id == next_id
            )
        )
        assert ambiguity.status == "ALL_CANDIDATES_REJECTED"
        assert ambiguity.candidate_ids == []
        assert sum(
            str(item.get("status") or "").upper() == "PENDING"
            for item in ambiguity.candidates
        ) == 0
        assert next_ambiguity.candidate_ids == ["candidate_d", "candidate_e"]
        assert job.status == TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
        assert job.pending_ambiguity_id == next_id
        assert pipeline.pending_ambiguity_id == next_id
        assert submitted == []

        reject(next_id, "candidate_d", "r14-key-d")
        final = reject(next_id, "candidate_e", "r14-key-e")
        assert final.metadata_["continuation"] == "SEARCH_NEXT_SHOT"
        assert final.metadata_["remaining_candidate_count"] == 0
        assert final.metadata_["next_candidate_id"] is None
        assert final.metadata_["next_ambiguity_id"] is None
        assert next_ambiguity.status == "ALL_CANDIDATES_REJECTED"
        assert next_ambiguity.candidate_ids == []
        assert pipeline.pending_ambiguity_id is None
        assert job.pending_ambiguity_id is None
        assert job.status == TrackingBackendStatus.QUEUED.value
        assert submitted == [job.tracking_job_id]
        assert job.queued_action["kind"] == "candidate_rejected"
        assert db.scalar(select(func.count(EventCandidateReviewDecisionR1.decision_id))) == 5
        assert db.scalar(select(func.count(EventCandidateOutboxR1.outbox_id))) == 5

        # Reproduce the historical split pointer caused by a failed subprocess,
        # then recover from the existing decision/outbox without recreating either.
        job.status = TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
        job.pending_ambiguity_id = next_id
        job.queued_action = {}
        pipeline.pending_ambiguity_id = None
        db.commit()
        recovered = service.recover_rejected_candidate_resume(
            tracking_job_id=job.tracking_job_id,
            submit_runtime=False,
        )
        assert recovered.status == TrackingBackendStatus.QUEUED.value
        assert recovered.pending_ambiguity_id is None
        assert recovered.queued_action["kind"] == "candidate_rejected"
        assert pipeline.pending_ambiguity_id is None
        assert db.scalar(select(func.count(EventCandidateReviewDecisionR1.decision_id))) == 5
        assert db.scalar(select(func.count(EventCandidateOutboxR1.outbox_id))) == 5

        # If the runtime already banked the negative and only the following
        # shot failed, recovery resumes that state without replaying the human
        # decision or creating another negative-memory revision.
        runtime_state = json.loads(state_path.read_text(encoding="utf-8"))
        runtime_state["status"] = "RUNNING"
        runtime_state["pending_action"] = None
        runtime_state["runtime"] = {
            "phase4a_initial_memory_review_status": "PASS",
            "phase4b_cross_shot_scoring_authorized": True,
            "phase4b_identity_negative_memory": {
                "rejected_candidate_ids": ["candidate_e"]
            },
            "phase4b_next_shot_id": "shot_0005",
        }
        state_path.write_text(json.dumps(runtime_state), encoding="utf-8")
        resumed = service.recover_rejected_candidate_resume(
            tracking_job_id=job.tracking_job_id,
            submit_runtime=False,
        )
        assert resumed.queued_action == {"kind": "recovery_resume"}
        assert pipeline.pipeline_stage == "SEARCHING_NEXT_SHOT"
        final_outbox = db.scalar(
            select(EventCandidateOutboxR1).where(
                EventCandidateOutboxR1.idempotency_key == "r14-key-e"
            )
        )
        assert final_outbox.status == "COMPLETED"
        assert db.scalar(select(func.count(EventCandidateReviewDecisionR1.decision_id))) == 5
        assert db.scalar(select(func.count(EventCandidateOutboxR1.outbox_id))) == 5
    finally:
        db.close()
        engine.dispose()
        shutil.rmtree(root, ignore_errors=True)
