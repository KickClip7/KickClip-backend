from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateMemoryRevisionR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
)
from app.domains.tracking.execution import R1PipelineStage, R1ProcessingStatus
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.r1_executor import get_r1_tracking_executor
from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.verifier import (
    get_scene_target_tracking_verifier,
)


JOB_ID = "trk_82bc437e94cb"
AMBIGUITY_ID = "ambiguity_phase4b_shot_0010_g020"
CANDIDATE_ID = "shot_0010_track_0047"
DECISION_ID = "ecdecr1_b6fb20b6446b"
MEMORY_REVISION_ID = "ecmem_5339a6e1eecf"
EXPECTED_MEMORY_SHA256 = (
    "8493d8f12cb09084e93a484195ed7b6bc764838949ac23eecca964eecd8bdae7"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_tail(path: Path, *, max_lines: int = 120) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _wait_for_phase4c_result(
    *,
    backend_root: Path,
    output_dir: Path,
    timeout_seconds: int = 600,
) -> None:
    report_path = (
        output_dir
        / "phase4c_post_confirmation"
        / "phase4c_post_confirmation_report.json"
    )
    log_dir = (
        backend_root
        / "storage"
        / "tracking_runtime"
        / "_backend_process_logs"
        / output_dir.name
    )
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    deadline = time.monotonic() + timeout_seconds
    last_status = None
    while time.monotonic() < deadline:
        if report_path.is_file():
            report = _load_json(report_path)
            if (
                report.get("status") == "PASS"
                and report.get("candidate_id") == CANDIDATE_ID
                and report.get("decision_id") == DECISION_ID
                and report.get("memory_revision_id")
                == MEMORY_REVISION_ID
            ):
                print("=== PHASE 4-C FINALIZED ===")
                print("status=PASS")
                print(f"report_path={report_path}")
                print(f"report_sha256={_sha256(report_path)}")
                print(
                    "real_detector_observation_count="
                    f"{report.get('real_detector_observation_count')}"
                )
                print(
                    "active_or_reacquired_bbox_frames="
                    f"{report.get('active_or_reacquired_bbox_frames')}"
                )
                print(
                    "automatic_target_confirmation="
                    f"{report.get('automatic_target_confirmation')}"
                )
                return

        with SessionLocal() as db:
            current = db.scalar(
                select(TrackingJob).where(
                    TrackingJob.tracking_job_id == JOB_ID
                )
            )
            if current is None:
                raise RuntimeError(
                    f"Tracking job disappeared while waiting: {JOB_ID}"
                )
            status = str(current.status or "")
            if status != last_status:
                print(f"runtime_status={status}")
                last_status = status
            if status == TrackingBackendStatus.FAILED.value:
                stdout_tail = _read_tail(stdout_path)
                stderr_tail = _read_tail(stderr_path)
                raise RuntimeError(
                    "Phase 4-C runtime failed.\n"
                    f"stdout_tail:\n{stdout_tail}\n"
                    f"stderr_tail:\n{stderr_tail}"
                )

        time.sleep(1.0)

    stdout_tail = _read_tail(stdout_path)
    stderr_tail = _read_tail(stderr_path)
    raise RuntimeError(
        "Phase 4-C report was not created within "
        f"{timeout_seconds} seconds.\n"
        f"stdout_tail:\n{stdout_tail}\n"
        f"stderr_tail:\n{stderr_tail}"
    )


def main() -> int:
    installation = (
        get_scene_target_tracking_verifier().check(force=True)
    )
    print("=== RUNTIME PREFLIGHT ===")
    print(f"installation_code={installation.code}")
    print(f"installation_available={installation.available}")
    if not installation.available:
        raise RuntimeError(
            "Scene-target tracking runtime preflight failed: "
            f"{installation.code}: {installation.message}"
        )

    # IMPORTANT:
    # Start the executor before this job is moved back to QUEUED.
    # A fresh standalone executor runs startup reconciliation inside start().
    # If the job is queued first, that reconciliation reads the old durable
    # CROSS_SHOT_CONFIRMATION state and can restore WAITING before the worker
    # calls claim_queued(), producing a false runtime_submit=PASS with no
    # subprocess and no Phase 4-C report.
    executor = get_r1_tracking_executor()
    executor.start()
    print("executor_started_before_queue=True")

    with SessionLocal() as db:
        job = db.scalar(
            select(TrackingJob)
            .where(TrackingJob.tracking_job_id == JOB_ID)
            .with_for_update()
        )
        pipeline = db.scalar(
            select(EventCandidatePipelineR1)
            .where(EventCandidatePipelineR1.tracking_job_id == JOB_ID)
            .with_for_update()
        )
        ambiguity = db.scalar(
            select(EventCandidateAmbiguityR1)
            .where(
                EventCandidateAmbiguityR1.tracking_job_id == JOB_ID,
                EventCandidateAmbiguityR1.ambiguity_id == AMBIGUITY_ID,
            )
            .with_for_update()
        )
        decision = db.get(EventCandidateReviewDecisionR1, DECISION_ID)
        memory = db.get(EventCandidateMemoryRevisionR1, MEMORY_REVISION_ID)

        if any(
            value is None
            for value in (job, pipeline, ambiguity, decision, memory)
        ):
            raise RuntimeError(
                "Required job/pipeline/ambiguity/decision/memory row is missing."
            )

        output_dir = Path(str(job.output_directory)).resolve()
        report_path = (
            output_dir
            / "phase4c_post_confirmation"
            / "phase4c_post_confirmation_report.json"
        )
        if report_path.is_file():
            report = _load_json(report_path)
            if (
                report.get("status") == "PASS"
                and report.get("candidate_id") == CANDIDATE_ID
                and report.get("decision_id") == DECISION_ID
                and report.get("memory_revision_id") == MEMORY_REVISION_ID
            ):
                print("status=ALREADY_FINALIZED")
                print(f"report_path={report_path}")
                print(f"report_sha256={_sha256(report_path)}")
                return 0

        if decision.decision_state != "SAME_PLAYER":
            raise RuntimeError(
                f"Unexpected decision state: {decision.decision_state}"
            )
        if decision.candidate_id != CANDIDATE_ID:
            raise RuntimeError("Decision candidate mismatch.")
        if decision.ambiguity_id != AMBIGUITY_ID:
            raise RuntimeError("Decision ambiguity mismatch.")
        if ambiguity.status != "RESOLVED":
            raise RuntimeError(
                f"Ambiguity is not resolved: {ambiguity.status}"
            )
        if pipeline.current_memory_revision_id != MEMORY_REVISION_ID:
            raise RuntimeError("Pipeline memory revision mismatch.")
        if job.current_memory_revision_id != MEMORY_REVISION_ID:
            raise RuntimeError("Job memory revision mismatch.")

        memory_path = Path(str(memory.artifact_path))
        if not memory_path.is_absolute():
            memory_path = (
                Path(__file__).resolve().parents[1] / memory_path
            ).resolve()
        if not memory_path.is_file():
            raise RuntimeError(f"Memory artifact is missing: {memory_path}")
        actual_memory_sha = _sha256(memory_path)
        if (
            memory.artifact_sha256 != EXPECTED_MEMORY_SHA256
            or actual_memory_sha != EXPECTED_MEMORY_SHA256
        ):
            raise RuntimeError(
                "Confirmed target memory SHA-256 mismatch."
            )
        memory_document = _load_json(memory_path)
        if (
            memory_document.get("source_candidate_id") != CANDIDATE_ID
            or memory_document.get("confirmation_decision_id") != DECISION_ID
            or memory_document.get("automatic_target_confirmation") is not False
        ):
            raise RuntimeError("Confirmed target memory provenance mismatch.")

        state_path = Path(str(job.pipeline_state_path)).resolve()
        state = _load_json(state_path)
        pending = _object(state.get("pending_action"))
        if (
            pending.get("type") != "CROSS_SHOT_CONFIRMATION"
            or pending.get("ambiguity_id") != AMBIGUITY_ID
            or CANDIDATE_ID
            not in [str(value) for value in pending.get("candidate_ids") or []]
            or pending.get("automatic_target_confirmation") is not False
        ):
            raise RuntimeError(
                "Durable runtime state no longer contains the reviewed ambiguity."
            )

        job.queued_action = {
            "kind": "ambiguity",
            "ambiguity_id": AMBIGUITY_ID,
            "candidate_id": CANDIDATE_ID,
            "decision": "confirmed",
            "reviewer": decision.reviewer_id,
            "note": decision.note,
        }
        job.status = TrackingBackendStatus.QUEUED.value
        job.pipeline_status = "RUNNING"
        job.pipeline_decision = "R1_RESUME_SAME_PLAYER_PHASE4C"
        job.pipeline_stage = R1PipelineStage.APPLYING_HUMAN_DECISION.value
        job.current_stage = R1PipelineStage.APPLYING_HUMAN_DECISION.value
        job.processing_status = R1ProcessingStatus.READY.value
        job.pending_action_type = None
        job.pending_ambiguity_id = None
        job.completed_at = None
        job.finished_at = None
        job.failure_code = None

        pipeline.pipeline_stage = R1PipelineStage.APPLYING_HUMAN_DECISION.value
        pipeline.processing_status = R1ProcessingStatus.READY.value
        pipeline.pending_ambiguity_id = None
        pipeline.last_completed_ambiguity_id = AMBIGUITY_ID
        pipeline.latest_decision_id = DECISION_ID
        pipeline.current_memory_revision_id = MEMORY_REVISION_ID
        pipeline.current_shot_id = "shot_0010"
        pipeline.next_shot_id = "shot_0010"
        pipeline.completed_at = None
        pipeline.failure_code = None

        db.commit()

    print("status=QUEUED")
    print(f"tracking_job_id={JOB_ID}")
    print(f"ambiguity_id={AMBIGUITY_ID}")
    print(f"candidate_id={CANDIDATE_ID}")
    print(f"decision_id={DECISION_ID}")
    print(f"memory_revision_id={MEMORY_REVISION_ID}")
    print("automatic_target_confirmation=False")
    submitted = executor.submit(JOB_ID)
    if not submitted:
        raise RuntimeError(
            "R1 runtime executor did not accept the queued job."
        )
    print("runtime_submit=PASS")

    backend_root = Path(__file__).resolve().parents[1]
    _wait_for_phase4c_result(
        backend_root=backend_root,
        output_dir=output_dir,
        timeout_seconds=600,
    )
    executor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
