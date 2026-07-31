from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.highlight.model import HighlightRevision
from app.domains.project.model import Project


PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    PROJECT_ROOT
    / "storage/matches/match_preloaded_kor_jpn/projects"
    / PROJECT_ID
    / "highlight"
    / REVISION_ID
    / "representative-goal-review/event-candidate-ranking-v1-2"
)
DIAGNOSTIC_CSV = (
    RESULT_ROOT / "full-class-diagnostic/rfdetr_full_class_frames_303_398.csv"
)
DIAGNOSTIC_JSON = (
    RESULT_ROOT / "full-class-diagnostic/rfdetr_full_class_frames_303_398.json"
)
EVIDENCE_CONTRACT = RESULT_ROOT / "ranking_evidence_artifact_contract.json"
SCHEMA = (
    PROJECT_ROOT
    / "configs/models/event_candidate_ranking/"
    "target_centric_tracking_event_candidate_ranking_v1_2/"
    "ranking_evidence_artifact_schema.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def get_or_create_artifact(
    *,
    db: Any,
    project: Project,
    artifact_type: str,
    path: Path,
    mime_type: str,
    metadata: dict[str, Any],
) -> Artifact:
    expected_sha = sha256_file(path)
    existing = db.scalar(
        select(Artifact).where(
            Artifact.project_id == PROJECT_ID,
            Artifact.artifact_type == artifact_type,
        )
    )
    if existing is not None:
        if (
            existing.file_path != relative(path)
            or (existing.metadata_ or {}).get("sha256") != expected_sha
        ):
            raise RuntimeError(f"Existing {artifact_type} differs.")
        return existing
    artifact = Artifact(
        match_id=project.match_id,
        project_id=project.project_id,
        analysis_job_id=None,
        artifact_type=artifact_type,
        file_path=relative(path),
        mime_type=mime_type,
        metadata_={**metadata, "sha256": expected_sha},
    )
    db.add(artifact)
    db.flush()
    return artifact


def main() -> None:
    if not DIAGNOSTIC_CSV.is_file() or not DIAGNOSTIC_JSON.is_file():
        raise RuntimeError("Full-class diagnostic output is missing.")
    diagnostic = load_json(DIAGNOSTIC_JSON)
    if diagnostic.get("score_input") is not False:
        raise RuntimeError("Diagnostic must not be a score input.")
    if diagnostic.get("shortlist_input") is not False:
        raise RuntimeError("Diagnostic must not be a shortlist input.")
    if (
        diagnostic["detections"]["csv_sha256"]
        != sha256_file(DIAGNOSTIC_CSV)
    ):
        raise RuntimeError("Diagnostic CSV SHA-256 mismatch.")

    with SessionLocal() as db:
        project = db.get(Project, PROJECT_ID)
        revision = db.get(HighlightRevision, REVISION_ID)
        player_detections = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type == "FROZEN_SCENE_DETECTIONS",
            )
        )
        ranking = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type
                == "EVENT_CANDIDATE_RANKING_V1_2_SHADOW_SHORTLIST_PATCH",
            )
        )
        if (
            project is None
            or revision is None
            or player_detections is None
            or ranking is None
        ):
            raise RuntimeError("Representative V1.2 scope is incomplete.")
        player_path = PROJECT_ROOT / player_detections.file_path
        player_sha = sha256_file(player_path)

        csv_artifact = get_or_create_artifact(
            db=db,
            project=project,
            artifact_type="RFDETR_FULL_CLASS_EVENT_WINDOW_DETECTIONS_V1",
            path=DIAGNOSTIC_CSV,
            mime_type="text/csv",
            metadata={
                "revision_id": REVISION_ID,
                "ranking_artifact_id": ranking.artifact_id,
                "frame_start": 303,
                "frame_end_inclusive": 398,
                "diagnostic_only": True,
                "score_input": False,
                "shortlist_input": False,
            },
        )
        report_artifact = get_or_create_artifact(
            db=db,
            project=project,
            artifact_type="RFDETR_FULL_CLASS_EVENT_WINDOW_DIAGNOSTIC_V1",
            path=DIAGNOSTIC_JSON,
            mime_type="application/json",
            metadata={
                "revision_id": REVISION_ID,
                "ranking_artifact_id": ranking.artifact_id,
                "detections_artifact_id": csv_artifact.artifact_id,
                "frame_start": 303,
                "frame_end_inclusive": 398,
                "diagnostic_only": True,
                "score_input": False,
                "shortlist_input": False,
            },
        )
        evidence = {
            "schema_version": "kickclip.ranking_evidence.v1",
            "player_discovery_detections": {
                "artifact_id": player_detections.artifact_id,
                "sha256": player_sha,
                "classes": ["player", "goalkeeper"],
                "immutable": True,
            },
            "ranking_object_detections": {
                "artifact_id": csv_artifact.artifact_id,
                "sha256": sha256_file(DIAGNOSTIC_CSV),
                "classes": [
                    "player",
                    "goalkeeper",
                    "referee",
                    "staff",
                    "ball",
                ],
                "frame_range": {
                    "start_frame": 303,
                    "end_frame_inclusive": 398,
                },
                "diagnostic_only": True,
            },
            "connected_to_scoring": False,
        }
        Draft202012Validator(load_json(SCHEMA)).validate(evidence)
        serialized = (
            json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if EVIDENCE_CONTRACT.exists():
            if EVIDENCE_CONTRACT.read_text(encoding="utf-8") != serialized:
                raise RuntimeError("Existing evidence contract differs.")
        else:
            EVIDENCE_CONTRACT.write_text(
                serialized,
                encoding="utf-8",
                newline="\n",
            )
        evidence_artifact = get_or_create_artifact(
            db=db,
            project=project,
            artifact_type="EVENT_CANDIDATE_RANKING_V1_2_EVIDENCE_CONTRACT",
            path=EVIDENCE_CONTRACT,
            mime_type="application/json",
            metadata={
                "revision_id": REVISION_ID,
                "ranking_artifact_id": ranking.artifact_id,
                "diagnostic_artifact_id": report_artifact.artifact_id,
                "connected_to_scoring": False,
            },
        )
        current = dict(
            (revision.options or {}).get("event_candidate_ranking_v1_2")
            or {}
        )
        revision.options = {
            **(revision.options or {}),
            "event_candidate_ranking_v1_2": {
                **current,
                "full_class_diagnostic_artifact_id": (
                    report_artifact.artifact_id
                ),
                "full_class_detections_artifact_id": csv_artifact.artifact_id,
                "ranking_evidence_contract_artifact_id": (
                    evidence_artifact.artifact_id
                ),
                "diagnostic_connected_to_scoring": False,
                "automatic_target_confirmation": False,
                "production_recommendation_ui": "BLOCKED",
            },
        }
        db.commit()
        print(
            json.dumps(
                {
                    "diagnostic_artifact_id": report_artifact.artifact_id,
                    "detections_artifact_id": csv_artifact.artifact_id,
                    "evidence_contract_artifact_id": (
                        evidence_artifact.artifact_id
                    ),
                    "diagnostic_sha256": sha256_file(DIAGNOSTIC_JSON),
                    "detections_sha256": sha256_file(DIAGNOSTIC_CSV),
                    "evidence_contract_sha256": sha256_file(
                        EVIDENCE_CONTRACT
                    ),
                    "connected_to_scoring": False,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
