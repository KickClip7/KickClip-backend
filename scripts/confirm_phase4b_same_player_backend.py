from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.auth.model import User
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateMemoryRevisionR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
)
from app.domains.candidate_handoff_r1.schema import (
    CandidateReviewDecisionRequest,
    CandidateReviewState,
)
from app.domains.candidate_handoff_r1.service import CandidateHandoffR1Service
from app.domains.tracking.execution import R1PipelineStage, R1ProcessingStatus
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.status import TrackingBackendStatus


JOB_ID = "trk_82bc437e94cb"
AMBIGUITY_ID = "ambiguity_phase4b_shot_0010_g020"
CANDIDATE_ID = "shot_0010_track_0047"

IDEMPOTENCY_KEY = (
    "phase4b-r12-shot0010-track0047-same-player-g020-v1"
)

EXPECTED_PHASE4B_POLICY = (
    "APPROVED_MEMORY_FROZEN_B0_B1_B2_ASSISTED_REVIEW_"
    "R12_GROUP_CONFIDENCE_REVIEW_CATALOG"
)

WAITING_STATUS = (
    TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
)
WAITING_STAGE = (
    R1PipelineStage.WAITING_CROSS_SHOT_CONFIRMATION.value
)
WAITING_PROCESSING = R1ProcessingStatus.WAITING.value

# 현재 sync 버그로 잘못 저장될 수 있는 상태만 제한적으로 복구한다.
REPAIRABLE_TERMINAL_STATUSES = {
    TrackingBackendStatus.COMPLETED_WITH_UNRESOLVED_GAPS.value,
    TrackingBackendStatus.COMPLETED_SAFE_BLOCK.value,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read runtime state: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Runtime state must be a JSON object: {path}")
    return value


def _as_object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _rows(value: object) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in _as_list(value)
        if isinstance(item, Mapping)
    ]


def _select_shot_attempt(
    report: Mapping[str, Any],
    shot_id: str,
) -> dict[str, Any]:
    attempts = _rows(report.get("attempts"))
    if not attempts:
        attempts = [dict(report)]

    selected = next(
        (
            attempt
            for attempt in attempts
            if str(attempt.get("shot_id") or "") == shot_id
        ),
        None,
    )
    if selected is None:
        available = [
            str(attempt.get("shot_id") or "")
            for attempt in attempts
        ]
        raise RuntimeError(
            "Phase 4-B shot attempt is missing: "
            f"expected={shot_id}, available={available}"
        )
    return selected


def _validate_runtime_pending(job: TrackingJob) -> dict[str, Any]:
    state_path = Path(str(job.pipeline_state_path or "")).resolve()
    if not state_path.is_file():
        raise RuntimeError(
            f"pipeline_state.json is missing: {state_path}"
        )

    state = _load_json(state_path)
    pending = _as_object(state.get("pending_action"))
    runtime = _as_object(state.get("runtime"))

    if pending.get("type") != "CROSS_SHOT_CONFIRMATION":
        raise RuntimeError(
            "Runtime is not waiting for CROSS_SHOT_CONFIRMATION: "
            f"type={pending.get('type')!r}"
        )
    if pending.get("ambiguity_id") != AMBIGUITY_ID:
        raise RuntimeError(
            "Runtime ambiguity mismatch: "
            f"expected={AMBIGUITY_ID}, "
            f"actual={pending.get('ambiguity_id')}"
        )
    if pending.get("shot_id") != "shot_0010":
        raise RuntimeError(
            "Runtime shot mismatch: "
            f"actual={pending.get('shot_id')}"
        )

    candidate_ids = [
        str(value) for value in _as_list(pending.get("candidate_ids"))
    ]
    if CANDIDATE_ID not in candidate_ids:
        raise RuntimeError(
            "Confirmed candidate is not present in runtime pending action: "
            f"{CANDIDATE_ID}"
        )
    if pending.get("automatic_target_confirmation") is not False:
        raise RuntimeError(
            "Runtime pending action must preserve "
            "automatic_target_confirmation=false."
        )

    report_path = (
        state_path.parent / "phase4b_first_cross_shot_report.json"
    )
    if not report_path.is_file():
        raise RuntimeError(
            f"Phase 4-B report is missing: {report_path}"
        )
    report = _load_json(report_path)
    attempt = _select_shot_attempt(report, "shot_0010")

    # R12 report contract uses the key "policy", not "phase4b_policy".
    report_policy = str(report.get("policy") or "")
    attempt_policy = str(attempt.get("policy") or "")

    if report_policy != EXPECTED_PHASE4B_POLICY:
        raise RuntimeError(
            "Unexpected Phase 4-B report policy: "
            f"{report_policy!r}; keys={sorted(report.keys())}"
        )
    if attempt_policy != EXPECTED_PHASE4B_POLICY:
        raise RuntimeError(
            "Unexpected Phase 4-B attempt policy: "
            f"{attempt_policy!r}; shot_id={attempt.get('shot_id')!r}"
        )

    report_candidate_ids = [
        str(item.get("candidate_id") or "")
        for item in _rows(attempt.get("review_candidates"))
    ]
    if CANDIDATE_ID not in report_candidate_ids:
        raise RuntimeError(
            "Confirmed candidate is not present in the R12 review set: "
            f"{CANDIDATE_ID}; review_candidate_ids="
            f"{report_candidate_ids}"
        )
    if attempt.get("automatic_target_confirmation") is not False:
        raise RuntimeError(
            "Phase 4-B attempt must preserve "
            "automatic_target_confirmation=false."
        )
    if attempt.get("candidate_link_created") is not False:
        raise RuntimeError(
            "Phase 4-B attempt already created a candidate link before "
            "human review."
        )

    generation = int(
        runtime.get("candidate_scoring_generation") or 0
    )
    attempt_generation = int(
        attempt.get("candidate_scoring_generation") or generation
    )
    if generation < 20:
        raise RuntimeError(
            "Unexpected runtime candidate scoring generation: "
            f"{generation}"
        )
    if attempt_generation != generation:
        raise RuntimeError(
            "Runtime/report candidate scoring generation mismatch: "
            f"runtime={generation}, attempt={attempt_generation}"
        )

    return {
        "state_path": str(state_path),
        "report_path": str(report_path),
        "candidate_ids": candidate_ids,
        "candidate_scoring_generation": generation,
        "phase4b_policy": attempt_policy,
        "review_candidate_ids": report_candidate_ids,
    }


def _repair_waiting_state_if_safe(
    *,
    db,
    job: TrackingJob,
    pipeline: EventCandidatePipelineR1,
    ambiguity: EventCandidateAmbiguityR1,
) -> bool:
    """Repair only the contradictory DB status produced by runtime sync.

    The repair is allowed only when the immutable runtime state, DB pipeline,
    DB ambiguity, and candidate set all point to the same pending review.
    """
    runtime_check = _validate_runtime_pending(job)
    candidate_ids = list(ambiguity.candidate_ids or [])

    if pipeline.pending_ambiguity_id != AMBIGUITY_ID:
        raise RuntimeError(
            "Pipeline pending ambiguity mismatch: "
            f"{pipeline.pending_ambiguity_id}"
        )
    if ambiguity.status != "WAITING":
        raise RuntimeError(
            f"Ambiguity is not WAITING: {ambiguity.status}"
        )
    if CANDIDATE_ID not in candidate_ids:
        raise RuntimeError(
            "Confirmed candidate is not in the DB ambiguity candidate set."
        )

    if job.status == WAITING_STATUS:
        print("db_waiting_state_repair=NOT_REQUIRED")
        return False

    if job.status not in REPAIRABLE_TERMINAL_STATUSES:
        raise RuntimeError(
            "Refusing to repair an unexpected job status: "
            f"{job.status}"
        )

    print("=== SAFE WAITING-STATE RECONCILIATION ===")
    print(f"previous_job_status={job.status}")
    print(f"previous_pipeline_stage={pipeline.pipeline_stage}")
    print(
        "runtime_candidate_scoring_generation="
        f"{runtime_check['candidate_scoring_generation']}"
    )
    print(
        "runtime_phase4b_policy="
        f"{runtime_check['phase4b_policy']}"
    )

    # record_review_decision()의 공식 precondition과 DB pointer를 일치시킨다.
    job.status = WAITING_STATUS
    if hasattr(job, "pipeline_status"):
        job.pipeline_status = "NEEDS_CONFIRMATION"
    if hasattr(job, "pending_action_type"):
        job.pending_action_type = "CROSS_SHOT_CONFIRMATION"
    job.pending_ambiguity_id = AMBIGUITY_ID
    job.pipeline_stage = WAITING_STAGE
    job.current_stage = WAITING_STAGE
    job.processing_status = WAITING_PROCESSING
    job.completed_at = None
    if hasattr(job, "finished_at"):
        job.finished_at = None
    job.failure_code = None

    pipeline.pipeline_stage = WAITING_STAGE
    pipeline.processing_status = WAITING_PROCESSING
    pipeline.pending_ambiguity_id = AMBIGUITY_ID
    pipeline.current_shot_id = ambiguity.shot_id
    pipeline.next_shot_id = ambiguity.shot_id
    pipeline.completed_at = None
    pipeline.failure_code = None

    db.commit()
    print("db_waiting_state_repair=APPLIED")
    print(f"repaired_job_status={WAITING_STATUS}")
    print(f"repaired_pipeline_stage={WAITING_STAGE}")
    return True


def main() -> int:
    with SessionLocal() as db:
        # 이미 등록됐다면 DB 상태를 건드리지 않고 종료한다.
        existing = db.scalar(
            select(EventCandidateReviewDecisionR1).where(
                EventCandidateReviewDecisionR1.idempotency_key
                == IDEMPOTENCY_KEY
            )
        )
        if existing is not None:
            print("status=ALREADY_REGISTERED")
            print(f"decision_id={existing.decision_id}")
            print(f"decision_state={existing.decision_state}")
            print(f"candidate_id={existing.candidate_id}")
            print(f"ambiguity_id={existing.ambiguity_id}")
            print(
                "confirmation_artifact_path="
                f"{existing.confirmation_artifact_path}"
            )
            return 0

        job = db.scalar(
            select(TrackingJob)
            .where(TrackingJob.tracking_job_id == JOB_ID)
            .with_for_update()
        )
        if job is None:
            raise RuntimeError(f"Tracking job not found: {JOB_ID}")

        owner = db.get(User, job.owner_id)
        if owner is None:
            raise RuntimeError(
                f"Tracking job owner not found: owner_id={job.owner_id}"
            )

        pipeline = db.scalar(
            select(EventCandidatePipelineR1)
            .where(
                EventCandidatePipelineR1.tracking_job_id == JOB_ID
            )
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
        if pipeline is None or ambiguity is None:
            raise RuntimeError(
                "R1 pipeline or pending ambiguity is missing. "
                "Run scripts.sync_phase4b_existing_job first."
            )

        candidate_ids = list(ambiguity.candidate_ids or [])

        print("=== PRECHECK ===")
        print(f"tracking_job_id={job.tracking_job_id}")
        print(f"job_status={job.status}")
        print(f"pipeline_stage={pipeline.pipeline_stage}")
        print(
            "pipeline_pending_ambiguity_id="
            f"{pipeline.pending_ambiguity_id}"
        )
        print(f"ambiguity_status={ambiguity.status}")
        print(
            "candidate_ids="
            + json.dumps(candidate_ids, ensure_ascii=False)
        )

        _repair_waiting_state_if_safe(
            db=db,
            job=job,
            pipeline=pipeline,
            ambiguity=ambiguity,
        )

        # commit 후 최신 row를 다시 읽는다.
        db.expire_all()
        job = db.get(TrackingJob, JOB_ID)
        owner = db.get(User, job.owner_id) if job is not None else None
        if job is None or owner is None:
            raise RuntimeError("Job or owner disappeared after reconciliation.")

        request = CandidateReviewDecisionRequest(
            state=CandidateReviewState.SAME_PLAYER,
            candidate_id=CANDIDATE_ID,
            note=(
                "User visually verified that every saved reference frame "
                "for shot_0010_track_0047 contains the selected player."
            ),
            idempotency_key=IDEMPOTENCY_KEY,
        )

        decision = CandidateHandoffR1Service(
            db
        ).record_review_decision(
            job=job,
            ambiguity_id=AMBIGUITY_ID,
            request=request,
            user=owner,
        )

        db.expire_all()

        updated_job = db.get(TrackingJob, JOB_ID)
        updated_pipeline = db.scalar(
            select(EventCandidatePipelineR1).where(
                EventCandidatePipelineR1.tracking_job_id == JOB_ID
            )
        )
        updated_ambiguity = db.scalar(
            select(EventCandidateAmbiguityR1).where(
                EventCandidateAmbiguityR1.tracking_job_id == JOB_ID,
                EventCandidateAmbiguityR1.ambiguity_id == AMBIGUITY_ID,
            )
        )

        memory = None
        if (
            updated_job is not None
            and updated_job.current_memory_revision_id
        ):
            memory = db.get(
                EventCandidateMemoryRevisionR1,
                updated_job.current_memory_revision_id,
            )

        print("=== SAME_PLAYER REGISTERED ===")
        print("status=PASS")
        print(f"decision_id={decision.decision_id}")
        print(f"decision_state={decision.decision_state}")
        print(f"candidate_id={decision.candidate_id}")
        print(f"ambiguity_id={decision.ambiguity_id}")
        print(
            "confirmation_artifact_path="
            f"{decision.confirmation_artifact_path}"
        )

        if updated_ambiguity is not None:
            print(f"ambiguity_status={updated_ambiguity.status}")

        if updated_pipeline is not None:
            print(
                f"pipeline_stage={updated_pipeline.pipeline_stage}"
            )
            print(
                "pipeline_pending_ambiguity_id="
                f"{updated_pipeline.pending_ambiguity_id}"
            )
            print(
                "pipeline_memory_revision_id="
                f"{updated_pipeline.current_memory_revision_id}"
            )

        if updated_job is not None:
            print(f"job_status={updated_job.status}")
            print(
                f"pipeline_status={updated_job.pipeline_status}"
            )
            print(
                f"pipeline_decision={updated_job.pipeline_decision}"
            )
            print(
                "job_pending_ambiguity_id="
                f"{updated_job.pending_ambiguity_id}"
            )
            print(
                "current_memory_revision_id="
                f"{updated_job.current_memory_revision_id}"
            )
            print(
                "queued_action="
                + json.dumps(
                    updated_job.queued_action or {},
                    ensure_ascii=False,
                )
            )

        if memory is not None:
            print(f"memory_artifact_path={memory.artifact_path}")
            print(
                f"memory_artifact_sha256={memory.artifact_sha256}"
            )
            print(
                "memory_reference_frame_ids="
                + json.dumps(memory.reference_frame_ids or [])
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
