from __future__ import annotations

import hashlib
import json
import time
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
from app.domains.highlight import scene_target_selection as scene_selection_module
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.main import app
from app.storage.local_storage import LocalStorage


MATCH_ID = "match_preloaded_kor_jpn"
PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
EVENT_ID = "evt_5eb4be390208"
SCENE_ID = EVENT_ID
SCENE_ASSET_ID = "asset_eval_goal_evt_5eb4be390208_scene"
STORAGE_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = (
    STORAGE_ROOT
    / "storage/matches/match_preloaded_kor_jpn/projects"
    / PROJECT_ID
    / "highlight"
    / REVISION_ID
    / "representative-goal-review"
)
STATE_PATH = REVIEW_ROOT / "representative_goal_shadow_run_state.json"
REPORT_PATH = REVIEW_ROOT / "representative_goal_shadow_run_report.json"


def install_windows_extended_output_path_adapter() -> None:
    """Use Win32 long paths in memory while persisted paths stay portable."""

    project_root = STORAGE_ROOT.resolve()
    extended_project_root = Path("\\\\?\\" + str(project_root))
    extended_storage_root = extended_project_root / "storage"

    def initialize_long_path_storage(self) -> None:
        self.project_root = extended_project_root
        self.storage_root = extended_storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)

    LocalStorage.__init__ = initialize_long_path_storage

    original = scene_selection_module.SceneTargetSelectionService._run

    def run_with_extended_output_path(self, arguments: list[str]) -> None:
        adjusted = list(arguments)
        if adjusted and adjusted[0] == "discover":
            try:
                index = adjusted.index("--output-root") + 1
            except (ValueError, IndexError):
                index = -1
            if index >= 0:
                output = str(Path(adjusted[index]).resolve())
                if not output.startswith("\\\\?\\"):
                    adjusted[index] = "\\\\?\\" + output
        original(self, adjusted)

    scene_selection_module.SceneTargetSelectionService._run = (
        run_with_extended_output_path
    )


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


def register_inputs() -> tuple[str, str, str]:
    reviewed_path = REVIEW_ROOT / "shot_boundaries_reviewed.json"
    detections_path = REVIEW_ROOT / "detections.csv"
    detection_metadata_path = REVIEW_ROOT / "detections_metadata.json"
    detection_metadata = json.loads(
        detection_metadata_path.read_text(encoding="utf-8")
    )
    if sha256_file(detections_path) != detection_metadata["detections_sha256"]:
        raise RuntimeError("Frozen detections SHA-256 mismatch.")
    with SessionLocal() as db:
        project = db.get(Project, PROJECT_ID)
        revision = db.get(HighlightRevision, REVISION_ID)
        scene_asset = db.get(MediaAsset, SCENE_ASSET_ID)
        if project is None or revision is None or scene_asset is None:
            raise RuntimeError("Representative evaluation scope is incomplete.")
        user = db.scalar(select(User).where(User.is_active.is_(True)))
        if user is None:
            raise RuntimeError("No active development user exists.")
        if project.owner_id not in (None, user.user_id):
            raise RuntimeError("Evaluation Project has an unexpected owner.")
        project.owner_id = user.user_id
        reviewed_artifact = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type == "REVIEWED_SHOT_BOUNDARIES",
            )
        )
        if reviewed_artifact is None:
            raise RuntimeError("Reviewed shot-boundary artifact is missing.")
        if (reviewed_artifact.metadata_ or {}).get(
            "sha256"
        ) != sha256_file(reviewed_path):
            raise RuntimeError("Reviewed shot-boundary SHA-256 mismatch.")

        detections_artifact = db.scalar(
            select(Artifact).where(
                Artifact.project_id == PROJECT_ID,
                Artifact.artifact_type == "FROZEN_SCENE_DETECTIONS",
            )
        )
        relative = detections_path.relative_to(STORAGE_ROOT).as_posix()
        if detections_artifact is None:
            detections_artifact = Artifact(
                match_id=MATCH_ID,
                project_id=PROJECT_ID,
                analysis_job_id=None,
                artifact_type="FROZEN_SCENE_DETECTIONS",
                file_path=relative,
                mime_type="text/csv",
                metadata_={
                    "revision_id": REVISION_ID,
                    "scene_id": SCENE_ID,
                    **detection_metadata,
                },
            )
            db.add(detections_artifact)
            db.flush()
        elif (
            detections_artifact.file_path != relative
            or (detections_artifact.metadata_ or {}).get("detections_sha256")
            != detection_metadata["detections_sha256"]
        ):
            raise RuntimeError("Existing detections Artifact differs.")

        representative = dict(
            (revision.options or {}).get("representative_goal") or {}
        )
        revision.options = {
            **(revision.options or {}),
            "representative_goal": {
                **representative,
                "detections_artifact_id": detections_artifact.artifact_id,
                "detections_sha256": detection_metadata[
                    "detections_sha256"
                ],
                "detections_metadata_relative_path": (
                    detection_metadata_path.relative_to(
                        STORAGE_ROOT
                    ).as_posix()
                ),
            },
        }
        db.commit()
        return (
            user.user_id,
            reviewed_artifact.artifact_id,
            detections_artifact.artifact_id,
        )


def wait_for_task(
    client: TestClient,
    *,
    task: dict[str, Any],
    headers: dict[str, str],
    state: dict[str, Any],
    state_key: str,
    timeout_sec: int,
) -> dict[str, Any]:
    task_id = task["task_id"]
    history = state.setdefault(f"{state_key}_status_history", [])
    last_status = None
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/v1/scene-ai-tasks/{task_id}",
            headers=headers,
        )
        response.raise_for_status()
        current = response.json()
        status = current["status"]
        if status != last_status:
            history.append(
                {
                    "status": status,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            state[state_key] = current
            write_json(STATE_PATH, state)
            print(
                json.dumps(
                    {"task": state_key, "task_id": task_id, "status": status},
                    sort_keys=True,
                ),
                flush=True,
            )
            last_status = status
        if status == "COMPLETED":
            return current
        if status == "FAILED":
            raise RuntimeError(
                f"{state_key} failed: {current.get('error_message')}"
            )
        time.sleep(2.0)
    raise TimeoutError(f"{state_key} did not finish in {timeout_sec}s.")


def main() -> None:
    install_windows_extended_output_path_adapter()
    user_id, shot_artifact_id, detections_artifact_id = register_inputs()
    settings = get_settings()
    token, _ = create_access_token(
        user_id=user_id,
        secret_key=settings.AUTH_SECRET_KEY,
        expires_minutes=max(settings.AUTH_ACCESS_TOKEN_MINUTES, 360),
    )
    headers = {"Authorization": f"Bearer {token}"}
    state: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "revision_id": REVISION_ID,
        "event_id": EVENT_ID,
        "scene_id": SCENE_ID,
        "scene_video_asset_id": SCENE_ASSET_ID,
        "reviewed_shot_boundaries_artifact_id": shot_artifact_id,
        "detections_artifact_id": detections_artifact_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "automatic_target_confirmation": False,
    }
    write_json(STATE_PATH, state)

    with TestClient(app) as client:
        discovery_response = client.post(
            f"/api/v1/projects/{PROJECT_ID}/highlight/revisions/"
            f"{REVISION_ID}/player-candidates/discover",
            headers=headers,
            json={
                "scene_id": SCENE_ID,
                "scene_video_asset_id": SCENE_ASSET_ID,
                "shot_boundaries_artifact_id": shot_artifact_id,
                "detections_artifact_id": detections_artifact_id,
            },
        )
        if discovery_response.status_code != 202:
            raise RuntimeError(
                "Discovery endpoint rejected the request: "
                f"{discovery_response.status_code} "
                f"{discovery_response.text}"
            )
        discovery_task = discovery_response.json()
        if discovery_task["status"] == "FAILED":
            retry = client.post(
                f"/api/v1/scene-ai-tasks/{discovery_task['task_id']}/retry",
                headers=headers,
            )
            if retry.status_code != 202:
                raise RuntimeError(
                    "Discovery retry endpoint rejected the request: "
                    f"{retry.status_code} {retry.text}"
                )
            discovery_task = retry.json()
        state["discovery_task_initial"] = discovery_task
        write_json(STATE_PATH, state)
        discovery_completed = wait_for_task(
            client,
            task=discovery_task,
            headers=headers,
            state=state,
            state_key="discovery_task",
            timeout_sec=21_600,
        )

        ranking_response = client.post(
            f"/api/v1/projects/{PROJECT_ID}/highlight/revisions/"
            f"{REVISION_ID}/event-candidate-rankings/v1.1.2a",
            headers=headers,
            json={
                "event_id": EVENT_ID,
                "scene_id": SCENE_ID,
                "shortlist_size": 5,
            },
        )
        if ranking_response.status_code != 202:
            raise RuntimeError(
                "V1.1.2a endpoint rejected the request: "
                f"{ranking_response.status_code} {ranking_response.text}"
            )
        ranking_task = ranking_response.json()
        state["ranking_task_initial"] = ranking_task
        write_json(STATE_PATH, state)
        ranking_completed = wait_for_task(
            client,
            task=ranking_task,
            headers=headers,
            state=state,
            state_key="ranking_task",
            timeout_sec=7_200,
        )

    artifact_id = ranking_completed["result"]["artifact_id"]
    with SessionLocal() as db:
        revision = db.get(HighlightRevision, REVISION_ID)
        artifact = db.get(Artifact, artifact_id)
        if revision is None or artifact is None:
            raise RuntimeError("Completed ranking output was not persisted.")
        discovery = dict(
            (revision.options or {}).get("scene_target_selection") or {}
        )
        output_path = Path(
            "\\\\?\\" + str((STORAGE_ROOT / artifact.file_path).resolve())
        )
        ranking = json.loads(output_path.read_text(encoding="utf-8"))
        if (artifact.metadata_ or {}).get("sha256") != sha256_file(
            output_path
        ):
            raise RuntimeError("Ranking artifact SHA-256 mismatch.")
        report = {
            "project_id": PROJECT_ID,
            "revision_id": REVISION_ID,
            "event_id": EVENT_ID,
            "scene_id": SCENE_ID,
            "discovery_id": discovery.get("discovery_id"),
            "source_video_sha256": discovery.get("discovery_inputs", {}).get(
                "scene_video_sha256"
            ),
            "reviewed_shot_sha256": discovery.get(
                "discovery_inputs", {}
            ).get("shot_boundaries_sha256"),
            "detections_sha256": discovery.get(
                "discovery_inputs", {}
            ).get("detections_sha256"),
            "scene_candidates_sha256": discovery.get(
                "scene_candidates_sha256"
            ),
            "candidate_manifest_sha256": discovery.get(
                "scene_candidate_manifest_sha256"
            ),
            "candidate_count": discovery.get("candidate_count"),
            "reviewed_shot_count": discovery.get("shot_count"),
            "discovery_task": discovery_completed,
            "ranking_task": ranking_completed,
            "ranking_artifact_id": artifact_id,
            "ranking_artifact_relative_path": artifact.file_path,
            "ranking_artifact_sha256": sha256_file(output_path),
            "ranking": ranking,
            "automatic_target_confirmation": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(REPORT_PATH, report)
        state["completed_at"] = report["completed_at"]
        state["report_path"] = str(REPORT_PATH)
        write_json(STATE_PATH, state)
        print(
            json.dumps(
                {
                    "report": str(REPORT_PATH),
                    "discovery_id": report["discovery_id"],
                    "candidate_count": report["candidate_count"],
                    "ranking_artifact_id": artifact_id,
                    "ranking_status": ranking["ranking_status"],
                    "feature_completeness_state": ranking[
                        "feature_completeness_state"
                    ],
                    "top_5": [
                        row["candidate_id"]
                        for row in ranking["shortlist"]
                    ],
                    "automatic_target_confirmation": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
