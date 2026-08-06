from __future__ import annotations

from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateOutboxR1,
    EventCandidatePipelineR1,
)
from app.domains.tracking.model import TrackingJob


JOB_ID = "trk_82bc437e94cb"
AMBIGUITY_ID = "ambiguity_phase4b_shot_0010_g020"
DECISION_ID = "ecdecr1_b6fb20b6446b"
MEMORY_REVISION_ID = "ecmem_5339a6e1eecf"


def main() -> int:
    errors: list[str] = []
    with SessionLocal() as db:
        job = db.get(TrackingJob, JOB_ID)
        pipeline = db.scalar(
            select(EventCandidatePipelineR1).where(
                EventCandidatePipelineR1.tracking_job_id == JOB_ID
            )
        )
        ambiguity = db.scalar(
            select(EventCandidateAmbiguityR1).where(
                EventCandidateAmbiguityR1.tracking_job_id == JOB_ID,
                EventCandidateAmbiguityR1.ambiguity_id == AMBIGUITY_ID,
            )
        )
        outbox_rows = db.scalars(
            select(EventCandidateOutboxR1).where(
                EventCandidateOutboxR1.tracking_job_id == JOB_ID
            )
        ).all()

        if job is None:
            errors.append("JOB_MISSING")
        if pipeline is None:
            errors.append("PIPELINE_MISSING")
        if ambiguity is None:
            errors.append("AMBIGUITY_MISSING")
        if errors:
            print("status=FAIL")
            print(f"errors={errors}")
            return 2

        assert job is not None
        assert pipeline is not None
        assert ambiguity is not None

        if job.status != "COMPLETED_WITH_UNRESOLVED_GAPS":
            errors.append(f"JOB_STATUS:{job.status}")
        if job.pipeline_status != "COMPLETE_WITH_UNRESOLVED_GAPS":
            errors.append(f"JOB_PIPELINE_STATUS:{job.pipeline_status}")
        if job.pending_ambiguity_id is not None:
            errors.append("JOB_PENDING_AMBIGUITY_NOT_CLEARED")
        if job.pending_action_type is not None:
            errors.append("JOB_PENDING_ACTION_NOT_CLEARED")
        if job.current_memory_revision_id != MEMORY_REVISION_ID:
            errors.append("JOB_MEMORY_REVISION_MISMATCH")
        if job.latest_decision_id != DECISION_ID:
            errors.append("JOB_LATEST_DECISION_MISMATCH")

        if pipeline.pipeline_stage != "COMPLETED_WITH_UNRESOLVED_GAPS":
            errors.append(
                f"PIPELINE_STAGE:{pipeline.pipeline_stage}"
            )
        if pipeline.processing_status != "COMPLETED_WITH_UNRESOLVED_GAPS":
            errors.append(
                f"PIPELINE_PROCESSING_STATUS:{pipeline.processing_status}"
            )
        if pipeline.pending_ambiguity_id is not None:
            errors.append("PIPELINE_PENDING_AMBIGUITY_NOT_CLEARED")
        if pipeline.current_memory_revision_id != MEMORY_REVISION_ID:
            errors.append("PIPELINE_MEMORY_REVISION_MISMATCH")
        if pipeline.latest_decision_id != DECISION_ID:
            errors.append("PIPELINE_LATEST_DECISION_MISMATCH")

        if ambiguity.status != "RESOLVED":
            errors.append(f"AMBIGUITY_STATUS:{ambiguity.status}")

        decision_outbox = [
            row
            for row in outbox_rows
            if str((row.payload or {}).get("decision_id") or "")
            == DECISION_ID
        ]
        if len(decision_outbox) != 1:
            errors.append("DECISION_OUTBOX_COUNT_MISMATCH")
        elif decision_outbox[0].status != "COMPLETED":
            errors.append(
                f"DECISION_OUTBOX_STATUS:{decision_outbox[0].status}"
            )

        print(f"status={'PASS' if not errors else 'FAIL'}")
        print(f"tracking_job_id={job.tracking_job_id}")
        print(f"job_status={job.status}")
        print(f"pipeline_status={job.pipeline_status}")
        print(f"pipeline_stage={pipeline.pipeline_stage}")
        print(f"processing_status={pipeline.processing_status}")
        print(f"ambiguity_status={ambiguity.status}")
        print(f"current_memory_revision_id={job.current_memory_revision_id}")
        print(f"latest_decision_id={job.latest_decision_id}")
        print(
            "decision_outbox_status="
            f"{decision_outbox[0].status if decision_outbox else None}"
        )
        print(f"errors={errors}")
        return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
