from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.domains.action_spotting.errors import ActionSpottingError
from app.domains.analysis.model import AnalysisJob
from app.domains.artifact.repository import ArtifactRepository
from app.storage.local_storage import LocalStorage


def attach_action_spotting_failure(
    db: Session,
    job: AnalysisJob,
    error: ActionSpottingError,
) -> str:
    """Persist private diagnostics and attach only the safe contract to the job."""

    storage = LocalStorage()
    directory = storage.storage_root / "action_spotting" / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{job.analysis_job_id}.json"
    payload = {
        "analysis_job_id": job.analysis_job_id,
        "match_id": job.match_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **error.to_internal_dict(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

    relative_path = output_path.relative_to(storage.project_root).as_posix()
    artifact = ArtifactRepository(db).create(
        match_id=job.match_id,
        analysis_job_id=job.analysis_job_id,
        artifact_type="ACTION_SPOTTING_DIAGNOSTICS",
        file_path=relative_path,
        mime_type="application/json",
        metadata_={
            "error_code": error.code,
            "retryable": error.retryable,
        },
    )
    options = dict(job.options or {})
    options["action_spotting_error"] = {
        **error.to_public_dict(),
        "diagnostics_artifact_id": artifact.artifact_id,
    }
    options["action_spotting_state"] = error.code
    history = list(options.get("action_spotting_state_history") or [])
    history.append(error.code)
    options["action_spotting_state_history"] = history
    job.options = options
    db.flush()
    return artifact.artifact_id
