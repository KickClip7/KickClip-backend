from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal
from app.domains.candidate_handoff_r1.service import CandidateHandoffR1Service


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Requeue a durable R14 DIFFERENT_PLAYER decision without recreating it."
    )
    parser.add_argument("tracking_job_id")
    parser.add_argument(
        "--submit-runtime",
        action="store_true",
        help="Submit the provenance-validated recovery to the tracking executor.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        job = CandidateHandoffR1Service(db).recover_rejected_candidate_resume(
            tracking_job_id=args.tracking_job_id,
            submit_runtime=args.submit_runtime,
        )
        print(
            json.dumps(
                {
                    "tracking_job_id": job.tracking_job_id,
                    "status": job.status,
                    "pending_ambiguity_id": job.pending_ambiguity_id,
                    "latest_decision_id": job.latest_decision_id,
                    "queued_action_kind": (job.queued_action or {}).get("kind"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
