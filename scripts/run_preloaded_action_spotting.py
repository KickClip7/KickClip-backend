"""Run real Champion Action Spotting for every preloaded local match."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.runtime.job_runner import JobRunner
from app.db.session import SessionLocal
from app.domains.action_spotting.schema import ActionSpottingJobRequest
from app.domains.analysis.model import AnalysisJob
from app.domains.highlight.action_cache import ActionSpottingCacheService
from app.domains.match.model import Match
from app.domains.timeline.repository import TimelineEventRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--match-id",
        action="append",
        help="Run only this preloaded Match ID. Repeat to select more than one.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with SessionLocal() as db:
        matches = list(
            db.scalars(
                select(Match)
                .where(Match.metadata_.is_not(None))
                .order_by(Match.created_at.asc())
            ).all()
        )
        match_ids = [
            match.match_id
            for match in matches
            if (match.metadata_ or {}).get("kickclip_preloaded_source")
        ]
    if args.match_id:
        requested = set(args.match_id)
        match_ids = [match_id for match_id in match_ids if match_id in requested]
        missing = requested.difference(match_ids)
        if missing:
            raise ValueError(
                f"Requested preloaded Match IDs do not exist: {sorted(missing)}"
            )
    if not match_ids:
        raise ValueError("No preloaded matches were found.")

    request_options = ActionSpottingJobRequest(
        run_feature_extraction=True,
        feature_extraction_mode="auto",
        device=args.device,
    ).to_job_options()
    job_ids: list[str] = []
    for match_id in match_ids:
        with SessionLocal() as db:
            job, reused = ActionSpottingCacheService(db).get_or_create(
                match_id=match_id,
                request_options=request_options,
            )
            job_ids.append(job.analysis_job_id)
            print(
                json.dumps(
                    {
                        "phase": "queued",
                        "match_id": match_id,
                        "analysis_job_id": job.analysis_job_id,
                        "status": job.status,
                        "cache_reused": reused,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        JobRunner().run(job_ids[-1])
        with SessionLocal() as db:
            completed = db.get(AnalysisJob, job_ids[-1])
            events = TimelineEventRepository(db).list_by_source_job(job_ids[-1])
            print(
                json.dumps(
                    {
                        "phase": "finished",
                        "match_id": match_id,
                        "analysis_job_id": job_ids[-1],
                        "status": completed.status if completed else "MISSING",
                        "progress": completed.progress if completed else None,
                        "event_count": len(events),
                        "error": completed.error_message if completed else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    failed = False
    summary = []
    with SessionLocal() as db:
        for match_id, job_id in zip(match_ids, job_ids):
            job = db.get(AnalysisJob, job_id)
            events = TimelineEventRepository(db).list_by_source_job(job_id)
            row = {
                "match_id": match_id,
                "analysis_job_id": job_id,
                "status": job.status if job else "MISSING",
                "progress": job.progress if job else None,
                "event_count": len(events),
                "error": job.error_message if job else None,
            }
            summary.append(row)
            failed = failed or job is None or job.status != "COMPLETED"
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
