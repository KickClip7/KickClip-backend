from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db.session import SessionLocal
from app.domains.candidate_handoff_r1.runtime_sync import R1RuntimeStateSynchronizer
from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.execution import R1_EXECUTION_KIND
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.state_mapper import map_pipeline_state, read_pipeline_state
from app.domains.tracking.sync import apply_pipeline_result


ALLOWED_DECISIONS = {
    "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION",
    "SAFE_BLOCK_PHASE4B_NO_REVIEWABLE_CANDIDATES",
    "SAFE_BLOCK_PHASE4B_ALL_REMAINING_SHOTS_EXHAUSTED",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize an already-generated Phase 4-B pipeline_state.json into "
            "the existing R1 tracking job DB rows without rerunning inference."
        )
    )
    parser.add_argument("--tracking-job-id", required=True)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        job = db.get(TrackingJob, args.tracking_job_id)
        if job is None:
            raise ValueError(f"Tracking job does not exist: {args.tracking_job_id}")
        if job.execution_kind != R1_EXECUTION_KIND:
            raise ValueError("UNROUTABLE_TRACKING_JOB")
        state_path = Path(job.pipeline_state_path).resolve()
        state = read_pipeline_state(state_path)
        if state is None:
            raise FileNotFoundError(state_path)
        decision = str(state.get("decision") or "")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(
                "Pipeline state is not a Phase 4-B assisted-confirmation/safe-block state: "
                + decision
            )
        runtime = state.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("Phase 4-B runtime metadata is missing.")
        report_path = Path(str(runtime.get("phase4b_report_path") or "")).resolve()
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict) or report.get("status") != "PASS":
            raise ValueError("Phase 4-B report is not PASS.")
        if runtime.get("backend_memory_used_by_phase4b_scoring") is not True:
            raise ValueError("Phase 4-B memory-use provenance is missing.")
        if runtime.get("automatic_target_confirmation") is not False:
            raise ValueError("AUTOMATIC_TARGET_CONFIRMATION_FORBIDDEN")

        mapping = map_pipeline_state(
            state,
            process_return_code=0,
            process_ended=True,
        )
        artifacts = TrackingArtifactService()
        apply_pipeline_result(
            job,
            state=state,
            mapping=mapping,
            process_return_code=0,
            process_pid=None,
            artifacts=artifacts,
        )
        R1RuntimeStateSynchronizer(db).sync(
            job=job,
            state=state,
            mapping=mapping,
        )
        db.commit()
        db.refresh(job)
        print(f"status=PASS")
        print(f"tracking_job_id={job.tracking_job_id}")
        print(f"backend_status={job.status}")
        print(f"pipeline_status={job.pipeline_status}")
        print(f"pipeline_decision={job.pipeline_decision}")
        print(f"pending_action_type={job.pending_action_type}")
        print(f"pending_ambiguity_id={job.pending_ambiguity_id}")
        print(f"current_memory_revision_id={job.current_memory_revision_id}")
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
