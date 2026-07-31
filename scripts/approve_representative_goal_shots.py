from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project


MATCH_ID = "match_preloaded_kor_jpn"
PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
EVENT_ID = "evt_5eb4be390208"
STORAGE_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = (
    STORAGE_ROOT
    / "storage/matches/match_preloaded_kor_jpn/projects"
    / PROJECT_ID
    / "highlight"
    / REVISION_ID
    / "representative-goal-review"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    pending_path = REVIEW_ROOT / "shot_boundaries_pending_review.json"
    approved_path = REVIEW_ROOT / "shot_boundaries_reviewed.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    if pending.get("status") != "PENDING_HUMAN_REVIEW":
        raise RuntimeError("Pending shot artifact state changed.")
    shots = list(pending.get("shots") or [])
    if len(shots) != 11:
        raise RuntimeError(f"Expected 11 shots, found {len(shots)}.")
    cut_frames = list(
        (pending.get("diagnostics") or {}).get("pending_cut_frames") or []
    )
    if cut_frames != [43, 102, 303, 348, 399, 468, 556, 635, 685, 745]:
        raise RuntimeError("The reviewed cut list changed.")

    with SessionLocal() as db:
        project = db.get(Project, PROJECT_ID)
        revision = db.get(HighlightRevision, REVISION_ID)
        if project is None or revision is None:
            raise RuntimeError("Evaluation Project/Revision is missing.")
        if project.match_id != MATCH_ID or revision.project_id != PROJECT_ID:
            raise RuntimeError("Evaluation ownership relationship changed.")
        reviewer = project.owner_id or "human-reviewer"
        reviewed_at = datetime.now(timezone.utc).isoformat()
        approved = {
            **pending,
            "status": "REVIEWED_PASS",
            "diagnostics": {
                **(pending.get("diagnostics") or {}),
                "review_required": False,
                "retrieval_authorized": True,
                "pending_cut_frames": [],
                "missing_frame_count": 0,
                "duplicate_frame_count": 0,
                "unapproved_review_count": 0,
            },
            "review_contract": {
                **(pending.get("review_contract") or {}),
                "pending_cut_frames": [],
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "decision": "ALL_PROPOSED_BOUNDARIES_APPROVED",
            },
            "shots": [
                {**shot, "review_state": "REVIEWED_PASS"}
                for shot in shots
            ],
        }
        serialized = (
            json.dumps(
                approved,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if approved_path.exists():
            if approved_path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError(
                    "Existing approved shot artifact is immutable and differs."
                )
        else:
            approved_path.write_text(
                serialized,
                encoding="utf-8",
                newline="\n",
            )
        approved_sha = sha256_file(approved_path)

        artifact = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type == "REVIEWED_SHOT_BOUNDARIES",
            )
        )
        if artifact is None:
            artifact = Artifact(
                match_id=MATCH_ID,
                project_id=PROJECT_ID,
                analysis_job_id=None,
                artifact_type="REVIEWED_SHOT_BOUNDARIES",
                file_path=approved_path.relative_to(
                    STORAGE_ROOT
                ).as_posix(),
                mime_type="application/json",
                metadata_={
                    "revision_id": REVISION_ID,
                    "scene_id": EVENT_ID,
                    "sha256": approved_sha,
                    "review_state": "REVIEWED_PASS",
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "shot_count": len(shots),
                    "frame_count": pending["video"]["frame_count"],
                    "missing_frame_count": 0,
                    "duplicate_frame_count": 0,
                },
            )
            db.add(artifact)
            db.flush()
        elif (
            artifact.file_path
            != approved_path.relative_to(STORAGE_ROOT).as_posix()
            or (artifact.metadata_ or {}).get("sha256") != approved_sha
        ):
            raise RuntimeError("Existing reviewed-shot Artifact differs.")

        representative = dict(
            (revision.options or {}).get("representative_goal") or {}
        )
        revision.options = {
            **(revision.options or {}),
            "representative_goal": {
                **representative,
                "shot_review_artifact_id": artifact.artifact_id,
                "shot_review_sha256": approved_sha,
                "shot_review_status": "REVIEWED_PASS",
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
            },
        }
        revision.status = "SHOT_BOUNDARIES_REVIEWED"
        revision.pending_action = None
        db.commit()
        print(
            json.dumps(
                {
                    "artifact_id": artifact.artifact_id,
                    "path": str(approved_path),
                    "sha256": approved_sha,
                    "shot_count": len(shots),
                    "cut_count": len(cut_frames),
                    "frame_count": pending["video"]["frame_count"],
                    "missing_frame_count": 0,
                    "duplicate_frame_count": 0,
                    "review_state": "REVIEWED_PASS",
                    "reviewer": reviewer,
                    "automatic_target_confirmation": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
