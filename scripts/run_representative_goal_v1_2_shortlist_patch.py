from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_2.backend_adapter import (
    EventCandidateRankingV12BackendAdapter,
)
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project
from app.storage.local_storage import LocalStorage
from scripts.execute_representative_goal_shadow import (
    install_windows_extended_output_path_adapter,
)


PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
SOURCE_RANKING_ARTIFACT_ID = "art_11f5e2e0b584"
REVIEWED_SHOTS_ARTIFACT_ID = "art_d5b4cd14e23a"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    PROJECT_ROOT
    / "storage/matches/match_preloaded_kor_jpn/projects"
    / PROJECT_ID
    / "highlight"
    / REVISION_ID
    / "representative-goal-review/event-candidate-ranking-v1-2"
)
RESULT_PATH = RESULT_ROOT / "counterfactual_evaluation.json"

EXPECTED_FROZEN_HASHES = {
    "v1_1_2a_manifest": (
        "d1d9bc507f268b7aecf319fe25c5f967bd2d58179939459f752b26b8044ddafd"
    ),
    "source_ranking": (
        "958d085f642c8e50c0ae7bf1c4a63eb5d77df7920c7d1591bc08d2d3016a5197"
    ),
    "final_annotation": (
        "f247e5eb7bb2af68eb4b7dad89ea72a392b2351a38f51e4721fde23b741813fc"
    ),
    "human_evaluation": (
        "50067723c3c586727050f839e14198d1dc50ce9808adc2805683b68bcf5741d3"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> None:
    install_windows_extended_output_path_adapter()
    storage = LocalStorage()
    manifest_path = (
        storage.project_root
        / "configs/models/event_candidate_ranking/"
        "target_centric_tracking_event_candidate_ranking_v1_1_2a/"
        "manifest.json"
    )
    human_evaluation_path = (
        storage.project_root
        / "storage/matches/match_preloaded_kor_jpn/projects/"
        f"{PROJECT_ID}/highlight/{REVISION_ID}/representative-goal-review/"
        "human-event-role-review/human_evaluation_result.json"
    )

    with SessionLocal() as db:
        project = db.get(Project, PROJECT_ID)
        revision = db.get(HighlightRevision, REVISION_ID)
        user = db.scalar(select(User).where(User.is_active.is_(True)))
        source = db.get(Artifact, SOURCE_RANKING_ARTIFACT_ID)
        final_annotation = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type
                == "EVENT_ACTOR_ANNOTATION_V1_1_FINAL",
            )
        )
        if (
            project is None
            or revision is None
            or user is None
            or source is None
            or final_annotation is None
        ):
            raise RuntimeError("Representative V1.2 scope is incomplete.")
        source_path = storage.resolve_path(source.file_path)
        annotation_path = storage.resolve_path(final_annotation.file_path)
        frozen_before = {
            "v1_1_2a_manifest": sha256_file(manifest_path),
            "source_ranking": sha256_file(source_path),
            "final_annotation": sha256_file(annotation_path),
            "human_evaluation": sha256_file(human_evaluation_path),
        }
        if frozen_before != EXPECTED_FROZEN_HASHES:
            raise RuntimeError(
                f"Frozen V1.1.2a baseline changed: {frozen_before}"
            )

        # No human annotation is loaded before this shortlist-only call.
        output_artifact = EventCandidateRankingV12BackendAdapter(db).run(
            project=project,
            user=user,
            source_ranking_artifact_id=SOURCE_RANKING_ARTIFACT_ID,
            reviewed_shots_artifact_id=REVIEWED_SHOTS_ARTIFACT_ID,
            shortlist_size=5,
        )
        output_path = storage.resolve_path(output_artifact.file_path)
        output = load_json(output_path)

        # Human labels enter only after the V1.2 output is immutable.
        annotation = load_json(annotation_path)
        primary_actor_ids = set(
            annotation["primary_actor_candidate_ids"]
        )
        shortlist_ids = [
            row["candidate_id"] for row in output["shortlist"]
        ]
        actor_shortlist_ranks = [
            index
            for index, candidate_id in enumerate(shortlist_ids, start=1)
            if candidate_id in primary_actor_ids
        ]
        actor_shortlist_rank = (
            min(actor_shortlist_ranks) if actor_shortlist_ranks else None
        )
        original_actor_ranks = [
            int(row["rank"])
            for row in output["all_candidates"]
            if row["candidate_id"] in primary_actor_ids
        ]
        shot_0004_count = sum(
            row["shot_id"] == "shot_0004"
            for row in output["shortlist"]
        )
        evaluation = {
            "schema_version": (
                "kickclip.event_candidate_ranking.v1_2."
                "counterfactual_evaluation"
            ),
            "scope": "SINGLE_EVENT_COUNTERFACTUAL",
            "ranking_artifact_id": output_artifact.artifact_id,
            "source_ranking_artifact_id": SOURCE_RANKING_ARTIFACT_ID,
            "final_annotation_artifact_id": final_annotation.artifact_id,
            "human_labels_used_for_shortlist": False,
            "human_labels_used_after_output_for_evaluation": True,
            "primary_actor_candidate_ids": sorted(primary_actor_ids),
            "original_primary_actor_ranks": sorted(original_actor_ranks),
            "original_primary_actor_best_rank": min(original_actor_ranks),
            "primary_actor_in_shortlist": bool(actor_shortlist_ranks),
            "primary_actor_shortlist_rank": actor_shortlist_rank,
            "primary_actor_recall_at_1": float(
                actor_shortlist_rank is not None
                and actor_shortlist_rank <= 1
            ),
            "primary_actor_recall_at_3": float(
                actor_shortlist_rank is not None
                and actor_shortlist_rank <= 3
            ),
            "primary_actor_recall_at_5": float(
                actor_shortlist_rank is not None
                and actor_shortlist_rank <= 5
            ),
            "shot_0004_candidate_count": shot_0004_count,
            "max_candidates_per_shot": max(
                output["shot_counts"].values()
            ),
            "original_global_ranks_unchanged": output[
                "global_ranking_invariant"
            ]["original_global_ranks_unchanged"],
            "original_scores_unchanged": output[
                "global_ranking_invariant"
            ]["original_scores_unchanged"],
            "new_top5_candidate_ids": shortlist_ids,
            "new_top5": [
                {
                    "shortlist_rank": row["shortlist_rank"],
                    "original_global_rank": row["original_global_rank"],
                    "candidate_id": row["candidate_id"],
                    "shot_id": row["shot_id"],
                    "recommendation_score": row[
                        "recommendation_score"
                    ],
                    "reason_codes": row[
                        "shortlist_patch_reason_codes"
                    ],
                }
                for row in output["shortlist"]
            ],
            "automatic_target_confirmation": False,
            "production_recommendation_ui": "BLOCKED",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        if RESULT_PATH.exists():
            existing = load_json(RESULT_PATH)
            comparable = dict(existing)
            comparable.pop("evaluated_at", None)
            current = dict(evaluation)
            current.pop("evaluated_at", None)
            if comparable != current:
                raise RuntimeError(
                    "Existing V1.2 counterfactual evaluation differs."
                )
        else:
            RESULT_PATH.write_text(
                json.dumps(
                    evaluation,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        result_sha = sha256_file(RESULT_PATH)
        evaluation_artifact = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type
                == "EVENT_CANDIDATE_RANKING_V1_2_COUNTERFACTUAL_EVALUATION",
            )
        )
        if evaluation_artifact is None:
            evaluation_artifact = Artifact(
                match_id=project.match_id,
                project_id=project.project_id,
                analysis_job_id=None,
                artifact_type=(
                    "EVENT_CANDIDATE_RANKING_V1_2_"
                    "COUNTERFACTUAL_EVALUATION"
                ),
                file_path=RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                mime_type="application/json",
                metadata_={
                    "ranking_artifact_id": output_artifact.artifact_id,
                    "source_ranking_artifact_id": source.artifact_id,
                    "final_annotation_artifact_id": (
                        final_annotation.artifact_id
                    ),
                    "sha256": result_sha,
                    "primary_actor_recall_at_5": evaluation[
                        "primary_actor_recall_at_5"
                    ],
                    "automatic_target_confirmation": False,
                },
            )
            db.add(evaluation_artifact)
            db.flush()
        current = dict(
            (revision.options or {}).get(
                "event_candidate_ranking_v1_2"
            )
            or {}
        )
        revision.options = {
            **(revision.options or {}),
            "event_candidate_ranking_v1_2": {
                **current,
                "status": "PROVISIONAL_SHADOW_ONLY",
                "ranking_artifact_id": output_artifact.artifact_id,
                "counterfactual_evaluation_artifact_id": (
                    evaluation_artifact.artifact_id
                ),
                "primary_actor_recall_at_5": evaluation[
                    "primary_actor_recall_at_5"
                ],
                "automatic_target_confirmation": False,
                "production_recommendation_ui": "BLOCKED",
            },
        }
        db.commit()

        frozen_after = {
            "v1_1_2a_manifest": sha256_file(manifest_path),
            "source_ranking": sha256_file(source_path),
            "final_annotation": sha256_file(annotation_path),
            "human_evaluation": sha256_file(human_evaluation_path),
        }
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen V1.1.2a material changed during V1.2.")
        print(
            json.dumps(
                {
                    "ranking_artifact_id": output_artifact.artifact_id,
                    "ranking_output_sha256": sha256_file(output_path),
                    "counterfactual_evaluation_artifact_id": (
                        evaluation_artifact.artifact_id
                    ),
                    "counterfactual_evaluation_sha256": result_sha,
                    "frozen_before": frozen_before,
                    "frozen_after": frozen_after,
                    **evaluation,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
