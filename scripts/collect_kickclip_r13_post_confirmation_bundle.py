from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.candidate_handoff_r1.model import (
    EventCandidateAmbiguityR1,
    EventCandidateMemoryRevisionR1,
    EventCandidateOutboxR1,
    EventCandidatePipelineR1,
    EventCandidateReviewDecisionR1,
)
from app.domains.tracking.model import TrackingJob


JOB_ID = "trk_82bc437e94cb"
TEST_NAME = "event_candidate_handoff_r1_trk_82bc437e94cb"
AMBIGUITY_ID = "ambiguity_phase4b_shot_0010_g020"
DECISION_ID = "ecdecr1_b6fb20b6446b"
MEMORY_REVISION_ID = "ecmem_5339a6e1eecf"
CANDIDATE_ID = "shot_0010_track_0047"


def json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    mapper = sa_inspect(row).mapper
    result: dict[str, Any] = {}
    for attr in mapper.column_attrs:
        result[attr.key] = json_safe(getattr(row, attr.key))
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copy_file(
    *,
    source: Path,
    bundle_root: Path,
    relative: Path,
    required: bool = False,
) -> bool:
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return False
    target = bundle_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_tree_filtered(
    *,
    source: Path,
    bundle_root: Path,
    relative: Path,
    allowed_suffixes: set[str],
) -> int:
    if not source.is_dir():
        return 0
    count = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        rel = relative / path.relative_to(source)
        copy_file(source=path, bundle_root=bundle_root, relative=rel)
        count += 1
    return count


def main() -> int:
    backend = Path(__file__).resolve().parents[1]
    desktop = Path.home() / "Desktop"
    work = desktop / "kickclip_r13_post_confirmation_bundle"
    zip_path = desktop / "kickclip_r13_post_confirmation_bundle.zip"

    if work.exists():
        shutil.rmtree(work)
    if zip_path.exists():
        zip_path.unlink()
    work.mkdir(parents=True)

    with SessionLocal() as db:
        job = db.scalar(
            select(TrackingJob).where(
                TrackingJob.tracking_job_id == JOB_ID
            )
        )
        if job is None:
            raise RuntimeError(f"Tracking job not found: {JOB_ID}")

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
        decisions = db.scalars(
            select(EventCandidateReviewDecisionR1).where(
                EventCandidateReviewDecisionR1.tracking_job_id == JOB_ID
            )
        ).all()
        memories = db.scalars(
            select(EventCandidateMemoryRevisionR1).where(
                EventCandidateMemoryRevisionR1.tracking_job_id == JOB_ID
            )
        ).all()
        outbox_rows = db.scalars(
            select(EventCandidateOutboxR1).where(
                EventCandidateOutboxR1.tracking_job_id == JOB_ID
            )
        ).all()

        db_snapshot = {
            "job": row_to_dict(job),
            "pipeline": row_to_dict(pipeline),
            "ambiguity": row_to_dict(ambiguity),
            "review_decisions": [row_to_dict(row) for row in decisions],
            "memory_revisions": [row_to_dict(row) for row in memories],
            "outbox": [row_to_dict(row) for row in outbox_rows],
            "expected": {
                "job_id": JOB_ID,
                "ambiguity_id": AMBIGUITY_ID,
                "decision_id": DECISION_ID,
                "memory_revision_id": MEMORY_REVISION_ID,
                "candidate_id": CANDIDATE_ID,
            },
        }

        db_out = work / "db_snapshot.json"
        db_out.write_text(
            json.dumps(db_snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        job_root = Path(str(job.output_directory)).resolve()

    source_files = [
        "app/domains/candidate_handoff_r1/runtime/r1_v1_v2_adapter_cli.py",
        "app/domains/candidate_handoff_r1/runtime/r1_v1_v2_adapter_manifest.json",
        "app/domains/candidate_handoff_r1/service.py",
        "app/domains/candidate_handoff_r1/runtime_sync.py",
        "app/domains/candidate_handoff_r1/orchestrator.py",
        "app/domains/candidate_handoff_r1/r3_adapter.py",
        "app/domains/candidate_handoff_r1/model.py",
        "app/domains/tracking/r1_executor.py",
        "app/domains/tracking/process_runner.py",
        "app/domains/tracking/state_mapper.py",
        "app/domains/tracking/execution.py",
        "app/domains/tracking/status.py",
        "app/domains/tracking/model.py",
        "scripts/confirm_phase4b_same_player_backend.py",
    ]
    for relative in source_files:
        copy_file(
            source=backend / relative,
            bundle_root=work,
            relative=Path("source") / relative,
        )

    required_job_files = [
        "pipeline_state.json",
        "pipeline_summary.json",
        "phase4b_first_cross_shot_report.json",
        "target_memory_revisions/ecmem_5339a6e1eecf.json",
        "review_decisions/ecdecr1_b6fb20b6446b.json",
    ]
    optional_job_files = [
        "target_timeline.json",
        "phase3c_selected_shot_timeline.json",
        "phase3c_bidirectional_timeline_merge_report.json",
        "phase3b_bidirectional_phase1_report.json",
        "phase4a_initial_target_memory_report.json",
        "handoff_pointer.json",
    ]

    for relative in required_job_files:
        copy_file(
            source=job_root / relative,
            bundle_root=work,
            relative=Path("job") / relative,
            required=True,
        )
    for relative in optional_job_files:
        copy_file(
            source=job_root / relative,
            bundle_root=work,
            relative=Path("job") / relative,
        )

    copy_tree_filtered(
        source=job_root / "r3_inputs",
        bundle_root=work,
        relative=Path("job/r3_inputs"),
        allowed_suffixes={".json", ".csv", ".txt"},
    )

    shot_root = job_root / "phase4b_cross_shot" / "shot_0010"
    for name in [
        "phase4b_attempt_report.json",
        "ranked_candidates.json",
        "ranked_candidates.csv",
        "ranked_candidates_all.csv",
        "review_catalog.csv",
        "safe_gate.json",
        "candidate_assignments.csv",
    ]:
        copy_file(
            source=shot_root / name,
            bundle_root=work,
            relative=Path("job/phase4b_cross_shot/shot_0010") / name,
        )

    candidate_root = shot_root / "candidates" / CANDIDATE_ID
    copy_tree_filtered(
        source=candidate_root,
        bundle_root=work,
        relative=Path(
            "job/phase4b_cross_shot/shot_0010/candidates"
        )
        / CANDIDATE_ID,
        allowed_suffixes={
            ".json",
            ".csv",
            ".txt",
            ".jpg",
            ".jpeg",
            ".png",
        },
    )

    log_root = (
        backend
        / "storage"
        / "tracking_runtime"
        / "_backend_process_logs"
        / TEST_NAME
    )
    copy_tree_filtered(
        source=log_root,
        bundle_root=work,
        relative=Path("logs"),
        allowed_suffixes={".log", ".txt", ".json"},
    )

    inventory: dict[str, Any] = {
        "schema_version": "kickclip.r13_post_confirmation_bundle.v1",
        "backend_root": str(backend),
        "job_root": str(job_root),
        "job_id": JOB_ID,
        "candidate_id": CANDIDATE_ID,
        "decision_id": DECISION_ID,
        "memory_revision_id": MEMORY_REVISION_ID,
        "files": {},
    }
    for path in sorted(work.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(work).as_posix()
        inventory["files"][rel] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    (work / "INVENTORY.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=path.relative_to(work).as_posix(),
                )

    print("status=PASS")
    print(f"bundle={zip_path}")
    print(f"bundle_sha256={sha256_file(zip_path)}")
    print(f"file_count={sum(1 for p in work.rglob('*') if p.is_file())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
