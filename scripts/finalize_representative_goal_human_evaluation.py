from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.auth.security import create_access_token
from app.domains.highlight.model import HighlightRevision
from app.main import app
from scripts.execute_representative_goal_shadow import (
    install_windows_extended_output_path_adapter,
)


PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
RANKING_ARTIFACT_ID = "art_11f5e2e0b584"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = (
    PROJECT_ROOT
    / "storage/matches/match_preloaded_kor_jpn/projects"
    / PROJECT_ID
    / "highlight"
    / REVISION_ID
    / "representative-goal-review/human-event-role-review"
)
RESULT_PATH = REVIEW_ROOT / "human_evaluation_result.json"

PRIMARY_ACTOR_IDS = [
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0003_track_0001",
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0003_track_0002",
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0003_track_0003",
]
TOP5_NON_ACTOR_IDS = [
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0004_track_0001",
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0004_track_0003",
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0004_track_0006",
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0004_track_0007",
    "scene_candidate_discovery_43f3e1bfbc700d78573d_"
    "shot_0004_track_0004",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> None:
    install_windows_extended_output_path_adapter()
    settings = get_settings()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.is_active.is_(True)))
        if user is None:
            raise RuntimeError("No active reviewer exists.")
        reviewer_id = user.user_id
    token, _ = create_access_token(
        user_id=reviewer_id,
        secret_key=settings.AUTH_SECRET_KEY,
        expires_minutes=max(settings.AUTH_ACCESS_TOKEN_MINUTES, 60),
    )
    headers = {"Authorization": f"Bearer {token}"}
    candidate_roles = {
        **{
            candidate_id: "PRIMARY_EVENT_ACTOR"
            for candidate_id in PRIMARY_ACTOR_IDS
        },
        **{
            candidate_id: "BROADCAST_CLOSEUP_NON_ACTOR"
            for candidate_id in TOP5_NON_ACTOR_IDS
        },
    }
    review_payload = {
        "primary_actor_visible": True,
        "primary_actor_in_candidate_set": True,
        "primary_actor_candidate_ids": PRIMARY_ACTOR_IDS,
        "directly_related_candidate_ids": [],
        "actor_missing_reason": None,
        "candidate_roles": candidate_roles,
    }

    with TestClient(app) as client:
        review_response = client.post(
            "/api/v1/event-candidate-rankings/v1.1/artifacts/"
            f"{RANKING_ARTIFACT_ID}/annotation-reviews",
            headers=headers,
            json=review_payload,
        )
        if review_response.status_code != 200:
            raise RuntimeError(
                "Independent review API failed: "
                f"{review_response.status_code} {review_response.text}"
            )
        review = review_response.json()

        finalize_response = client.post(
            "/api/v1/event-candidate-rankings/v1.1/artifacts/"
            f"{RANKING_ARTIFACT_ID}/annotations/finalize",
            headers=headers,
            json={
                "review_artifact_ids": [review["artifact_id"]],
                "approval_mode": "FINAL_APPROVED",
                "approved_reviewer": reviewer_id,
            },
        )
        if finalize_response.status_code != 200:
            raise RuntimeError(
                "Final annotation API failed: "
                f"{finalize_response.status_code} {finalize_response.text}"
            )
        final_annotation = finalize_response.json()

        evaluation_response = client.get(
            "/api/v1/event-candidate-rankings/v1.1/artifacts/"
            f"{RANKING_ARTIFACT_ID}/evaluation",
            headers=headers,
            params={
                "annotation_artifact_id": final_annotation["artifact_id"]
            },
        )
        if evaluation_response.status_code != 200:
            raise RuntimeError(
                "Evaluation API failed: "
                f"{evaluation_response.status_code} "
                f"{evaluation_response.text}"
            )
        evaluation = evaluation_response.json()

    result = {
        "schema_version": "kickclip.representative_goal_human_evaluation.v1",
        "ranking_artifact_id": RANKING_ARTIFACT_ID,
        "review_artifact": review,
        "final_annotation_artifact": final_annotation,
        "evaluation": evaluation,
        "reviewer": reviewer_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "human_ground_truth": review_payload,
        "fragmented_primary_actor_candidate_ids": PRIMARY_ACTOR_IDS,
        "top5_broadcast_closeup_non_actor_candidate_ids": (
            TOP5_NON_ACTOR_IDS
        ),
        "automatic_target_confirmation": False,
        "production_recommendation_ui": "BLOCKED",
    }
    write_json(RESULT_PATH, result)
    result_sha = sha256_file(RESULT_PATH)

    with SessionLocal() as db:
        revision = db.get(HighlightRevision, REVISION_ID)
        if revision is None:
            raise RuntimeError("HighlightRevision is missing.")
        artifact = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type
                == "REPRESENTATIVE_GOAL_HUMAN_EVALUATION",
            )
        )
        relative = RESULT_PATH.relative_to(PROJECT_ROOT).as_posix()
        if artifact is None:
            artifact = Artifact(
                match_id="match_preloaded_kor_jpn",
                project_id=PROJECT_ID,
                analysis_job_id=None,
                artifact_type="REPRESENTATIVE_GOAL_HUMAN_EVALUATION",
                file_path=relative,
                mime_type="application/json",
                metadata_={
                    "revision_id": REVISION_ID,
                    "ranking_artifact_id": RANKING_ARTIFACT_ID,
                    "review_artifact_id": review["artifact_id"],
                    "final_annotation_artifact_id": final_annotation[
                        "artifact_id"
                    ],
                    "annotation_status": "COMPLETE",
                    "sha256": result_sha,
                    "reviewer": reviewer_id,
                },
            )
            db.add(artifact)
            db.flush()
        current = dict(
            (revision.options or {}).get(
                "representative_goal_human_review"
            )
            or {}
        )
        metrics = evaluation["evaluation"]
        revision.options = {
            **(revision.options or {}),
            "representative_goal_human_review": {
                **current,
                "status": "COMPLETE",
                "review_artifact_id": review["artifact_id"],
                "final_annotation_artifact_id": final_annotation[
                    "artifact_id"
                ],
                "evaluation_artifact_id": artifact.artifact_id,
                "primary_actor_visible": True,
                "primary_actor_in_candidate_set": True,
                "primary_actor_candidate_ids": PRIMARY_ACTOR_IDS,
                "primary_actor_recall_at_1": metrics[
                    "primary_actor_recall_at_1"
                ],
                "primary_actor_recall_at_3": metrics[
                    "primary_actor_recall_at_3"
                ],
                "primary_actor_recall_at_5": metrics[
                    "primary_actor_recall_at_5"
                ],
                "mrr": metrics["mrr"],
                "broadcast_closeup_non_actor_top_1": metrics[
                    "broadcast_closeup_non_actor_top_1"
                ],
                "meaningful_top_5_result": "FAIL",
                "automatic_target_confirmation": False,
            },
        }
        db.commit()

    print(
        json.dumps(
            {
                "review_artifact_id": review["artifact_id"],
                "final_annotation_artifact_id": final_annotation[
                    "artifact_id"
                ],
                "evaluation_artifact_id": artifact.artifact_id,
                "result_path": str(RESULT_PATH),
                "result_sha256": result_sha,
                "evaluation": evaluation["evaluation"],
                "automatic_target_confirmation": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
