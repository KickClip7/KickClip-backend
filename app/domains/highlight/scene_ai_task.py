from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking import (
    EventCandidateRankingService,
)
from app.domains.highlight.event_candidate_ranking_v1_1.backend_adapter import (
    EventCandidateRankingV11BackendAdapter,
)
from app.domains.highlight.event_candidate_ranking_v1_1_1.backend_adapter import (
    EventCandidateRankingV111BackendAdapter,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2.backend_adapter import (
    EventCandidateRankingV112BackendAdapter,
)
from app.domains.highlight.model import SceneAITask
from app.domains.highlight.scene_target_selection import (
    SceneTargetSelectionService,
)
from app.domains.highlight.scene_target_reviewability import (
    SceneTargetReviewabilityService,
)
from app.domains.highlight.schema import (
    EventCandidateRankingRequest,
    SceneAITaskRead,
)
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.tracking.executor import get_tracking_executor
from app.utils.id_generator import generate_prefixed_id


logger = logging.getLogger(__name__)

DISCOVERY = "SCENE_CANDIDATE_DISCOVERY"
REFERENCE_BUILD = "TARGET_REFERENCE_BUILD"
EARLIER_DISCOVERY = "EARLIER_CANDIDATE_DISCOVERY"
EARLIER_DECISION = "EARLIER_ANCHOR_DECISION"
TRACKING_PREPARATION = "SCENE_TRACKING_PREPARATION"
EVENT_RANKING = "EVENT_CANDIDATE_RANKING"
EVENT_RANKING_V1_1 = "EVENT_CANDIDATE_RANKING_V1_1_SHADOW"
EVENT_RANKING_V1_1_1 = "EVENT_CANDIDATE_RANKING_V1_1_1_SHADOW"
EVENT_RANKING_V1_1_2 = "EVENT_CANDIDATE_RANKING_V1_1_2_SHADOW"
REVIEW_SELECTED_PREPARE = "SELECTED_TARGET_REVIEW_PREPARE"
REVIEW_SELECTED_DECISION = "SELECTED_TARGET_REVIEW_DECISION"
REVIEW_EARLIER_PREPARE = "EARLIER_TARGET_REVIEW_PREPARE"
REVIEW_UI_RENDER = "TARGET_REVIEW_UI_RENDER"
MANUAL_ANCHOR_VALIDATE = "MANUAL_ANCHOR_VALIDATE"
MANUAL_ANCHOR_CONFIRM = "MANUAL_ANCHOR_CONFIRM"
MANUAL_TRACKING_PREPARATION = "MANUAL_TRACKING_PREPARATION"


class SceneAITaskService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def enqueue(
        self,
        *,
        user: User,
        project: Project,
        task_type: str,
        payload: dict[str, Any],
    ) -> tuple[SceneAITask, bool]:
        canonical = json.dumps(
            {
                "owner_id": user.user_id,
                "project_id": project.project_id,
                "task_type": task_type,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        existing = self.db.scalar(
            select(SceneAITask).where(
                SceneAITask.idempotency_key == key
            )
        )
        if existing is not None:
            return existing, True
        task = SceneAITask(
            task_id=generate_prefixed_id("saitask"),
            owner_id=user.user_id,
            match_id=project.match_id,
            project_id=project.project_id,
            task_type=task_type,
            status="QUEUED",
            idempotency_key=key,
            payload=payload,
            result={},
            attempt_count=0,
            max_attempts=3,
        )
        self.db.add(task)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            raced = self.db.scalar(
                select(SceneAITask).where(
                    SceneAITask.idempotency_key == key
                )
            )
            if raced is None:
                raise
            return raced, True
        self.db.refresh(task)
        return task, False

    def owned(self, task_id: str, user: User) -> SceneAITask:
        task = self.db.get(SceneAITask, task_id)
        if task is None or (
            task.owner_id != user.user_id
            and not user.developer_mode_enabled
        ):
            raise ValueError("Scene AI task not found.")
        return task

    def retry(self, task: SceneAITask) -> SceneAITask:
        if task.status != "FAILED":
            raise ValueError("Only failed Scene AI tasks can be retried.")
        if task.attempt_count >= task.max_attempts:
            raise ValueError("Scene AI task retry limit was reached.")
        task.status = "QUEUED"
        task.error_message = None
        task.completed_at = None
        self.db.commit()
        return task

    @staticmethod
    def read(task: SceneAITask) -> SceneAITaskRead:
        return SceneAITaskRead(
            task_id=task.task_id,
            task_type=task.task_type,
            status=task.status,
            attempt_count=task.attempt_count,
            max_attempts=task.max_attempts,
            result=dict(task.result or {}),
            error_message=task.error_message,
            status_url=f"/api/v1/scene-ai-tasks/{task.task_id}",
            retry_url=(
                f"/api/v1/scene-ai-tasks/{task.task_id}/retry"
                if task.status == "FAILED"
                and task.attempt_count < task.max_attempts
                else None
            ),
        )


class SceneAITaskExecutor:
    """DB-claimed executor with startup recovery and bounded retries."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._pool: ThreadPoolExecutor | None = None
        self._started = False
        self._stopping = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stopping.clear()
            self._pool = ThreadPoolExecutor(
                max_workers=(
                    self.settings.SCENE_TARGET_SELECTION_MAX_CONCURRENT_JOBS
                ),
                thread_name_prefix="kickclip-scene-ai",
            )
        for task_id in self._recoverable():
            self.submit(task_id)

    def submit(self, task_id: str) -> bool:
        if not self._started:
            self.start()
        with self._lock:
            if (
                self._pool is None
                or self._stopping.is_set()
                or task_id in self._active
            ):
                return False
            self._active.add(task_id)
            pool = self._pool
        pool.submit(self._execute_and_release, task_id)
        return True

    def shutdown(self) -> None:
        self._stopping.set()
        with self._lock:
            pool = self._pool
            self._pool = None
            self._started = False
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _execute_and_release(self, task_id: str) -> None:
        try:
            self._execute(task_id)
        finally:
            with self._lock:
                self._active.discard(task_id)

    def _execute(self, task_id: str) -> None:
        db = SessionLocal()
        try:
            claimed = db.execute(
                update(SceneAITask)
                .where(
                    SceneAITask.task_id == task_id,
                    SceneAITask.status == "QUEUED",
                    SceneAITask.attempt_count < SceneAITask.max_attempts,
                )
                .values(
                    status="RUNNING",
                    attempt_count=SceneAITask.attempt_count + 1,
                    started_at=datetime.now(timezone.utc),
                    completed_at=None,
                    error_message=None,
                )
            )
            if int(claimed.rowcount or 0) != 1:
                db.rollback()
                return
            db.commit()
            task = db.get(SceneAITask, task_id)
            if task is None:
                return
            task.result = self._dispatch(db, task)
            task.status = "COMPLETED"
            task.completed_at = datetime.now(timezone.utc)
            task.error_message = None
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.exception("Scene AI task failed: %s", task_id)
            task = db.get(SceneAITask, task_id)
            if task is not None:
                task.status = "FAILED"
                task.completed_at = datetime.now(timezone.utc)
                task.error_message = str(exc)[:3000]
                task.process_pid = None
                db.commit()
        finally:
            db.close()

    def _dispatch(
        self,
        db: Session,
        task: SceneAITask,
    ) -> dict[str, Any]:
        payload = dict(task.payload)
        user = db.get(User, task.owner_id)
        project = db.get(Project, task.project_id)
        if user is None or project is None:
            raise ValueError("Scene AI task ownership context is missing.")
        service = SceneTargetSelectionService(db, settings=self.settings)
        if task.task_type == DISCOVERY:
            service.discover(
                project=project,
                revision_id=payload["revision_id"],
                scene_id=payload["scene_id"],
                scene_video=self._required(
                    db, MediaAsset, payload["scene_video_asset_id"]
                ),
                shot_boundaries_artifact=self._required(
                    db,
                    Artifact,
                    payload["shot_boundaries_artifact_id"],
                ),
                detections_artifact=self._required(
                    db,
                    Artifact,
                    payload["detections_artifact_id"],
                ),
            )
            return {
                "revision_id": payload["revision_id"],
                "scene_id": payload["scene_id"],
                "candidates_url": (
                    f"/api/v1/projects/{project.project_id}/highlight/"
                    f"revisions/{payload['revision_id']}/player-candidates"
                    f"?scene_id={payload['scene_id']}"
                ),
            }
        if task.task_type == REFERENCE_BUILD:
            selection = service.create_selection(
                project=project,
                revision_id=payload["revision_id"],
                scene_id=payload["scene_id"],
                candidate_id=payload["candidate_id"],
                scene_video=self._required(
                    db, MediaAsset, payload["scene_video_asset_id"]
                ),
                user=user,
            )
            return {
                "selection_id": selection.selection_id,
                "selection_url": (
                    f"/api/v1/target-selections/{selection.selection_id}"
                ),
            }
        if task.task_type == EARLIER_DISCOVERY:
            selection = self._selection(service, payload, user)
            service.discover_earlier(selection=selection, user=user)
            return self._selection_result(selection)
        if task.task_type == EARLIER_DECISION:
            selection = self._selection(service, payload, user)
            service.decide_earlier(
                selection=selection,
                user=user,
                decision=payload["decision"],
                candidate_id=payload.get("candidate_id"),
            )
            return self._selection_result(selection)
        if task.task_type == TRACKING_PREPARATION:
            selection = self._selection(service, payload, user)
            response = service.create_tracking_job(
                selection=selection,
                project=project,
                user=user,
                scene_video=self._required(
                    db, MediaAsset, payload["scene_video_asset_id"]
                ),
                shot_boundaries_artifact=self._required(
                    db,
                    Artifact,
                    payload["shot_boundaries_artifact_id"],
                ),
            )
            get_tracking_executor().submit(response.job_id)
            return {
                "tracking_job_id": response.job_id,
                "tracking_job_url": (
                    f"/api/v1/tracking/jobs/{response.job_id}"
                ),
            }
        if task.task_type == EVENT_RANKING:
            ranking = EventCandidateRankingService(db).rank(
                project=project,
                revision_id=payload["revision_id"],
                payload=EventCandidateRankingRequest.model_validate(
                    payload["ranking"]
                ),
                user=user,
            )
            return {
                "ranking_id": ranking.ranking_id,
                "ranking_url": (
                    f"/api/v1/event-candidate-rankings/{ranking.ranking_id}"
                ),
            }
        if task.task_type == EVENT_RANKING_V1_1:
            return EventCandidateRankingV11BackendAdapter(db).run(
                project=project,
                user=user,
                revision_id=payload["revision_id"],
                ranking=payload["ranking"],
                freeze_material=payload["freeze_material"],
            )
        if task.task_type == EVENT_RANKING_V1_1_1:
            return EventCandidateRankingV111BackendAdapter(db).run(
                project=project,
                user=user,
                revision_id=payload["revision_id"],
                shortlist_size=payload["shortlist_size"],
                resolved_event=payload["resolved_event"],
                freeze_material=payload["freeze_material"],
            )
        if task.task_type == EVENT_RANKING_V1_1_2:
            return EventCandidateRankingV112BackendAdapter(db).run(
                project=project,
                user=user,
                revision_id=payload["revision_id"],
                shortlist_size=payload["shortlist_size"],
                resolved_event=payload["resolved_event"],
                freeze_material=payload["freeze_material"],
            )
        review = SceneTargetReviewabilityService(
            db,
            settings=self.settings,
        )
        if task.task_type in {
            REVIEW_SELECTED_PREPARE,
            REVIEW_EARLIER_PREPARE,
            REVIEW_UI_RENDER,
            MANUAL_ANCHOR_VALIDATE,
            MANUAL_TRACKING_PREPARATION,
        }:
            selection = review.owned(
                payload["selection_id"],
                user,
                for_update=True,
            )
            video = self._required(
                db,
                MediaAsset,
                payload["scene_video_asset_id"],
            )
            boundaries = self._required(
                db,
                Artifact,
                payload["shot_boundaries_artifact_id"],
            )
            if task.task_type == REVIEW_SELECTED_PREPARE:
                return review.prepare_selected_review(
                    selection=selection,
                    user=user,
                    project=project,
                    video=video,
                    boundaries=boundaries,
                )
            if task.task_type == REVIEW_EARLIER_PREPARE:
                return review.prepare_earlier_review(
                    selection=selection,
                    project=project,
                    video=video,
                    boundaries=boundaries,
                )
            if task.task_type == REVIEW_UI_RENDER:
                return review.render_review_ui(
                    selection=selection,
                    project=project,
                    video=video,
                    boundaries=boundaries,
                )
            if task.task_type == MANUAL_ANCHOR_VALIDATE:
                return review.validate_manual_anchor(
                    selection=selection,
                    user=user,
                    project=project,
                    video=video,
                    boundaries=boundaries,
                    global_frame=int(payload["global_frame"]),
                    click_xy=payload.get("click_xy"),
                    drawn_bbox=payload.get("drawn_bbox_xyxy"),
                    identity_basis=payload["identity_basis"],
                )
            response = review.create_manual_tracking_job(
                selection=selection,
                user=user,
                project=project,
                video=video,
                boundaries=boundaries,
            )
            get_tracking_executor().submit(response.job_id)
            return {
                "tracking_job_id": response.job_id,
                "tracking_job_url": (
                    f"/api/v1/tracking/jobs/{response.job_id}"
                ),
            }
        if task.task_type == REVIEW_SELECTED_DECISION:
            selection = review.owned(
                payload["selection_id"],
                user,
                for_update=True,
            )
            return review.decide_selected_identity(
                selection=selection,
                user=user,
                decision=payload["decision"],
                identity_basis=payload["identity_basis"],
            )
        if task.task_type == MANUAL_ANCHOR_CONFIRM:
            selection = review.owned(
                payload["selection_id"],
                user,
                for_update=True,
            )
            created = review.confirm_manual_anchor(
                selection=selection,
                user=user,
            )
            return {
                "selection_id": created.selection_id,
                "previous_selection_id": selection.selection_id,
                "selection_url": (
                    f"/api/v1/target-selections/{created.selection_id}"
                ),
            }
        raise ValueError(f"Unsupported Scene AI task type: {task.task_type}")

    @staticmethod
    def _required(db: Session, model, identifier: str):
        value = db.get(model, identifier)
        if value is None:
            raise ValueError(f"Scene AI task resource is missing: {identifier}")
        return value

    @staticmethod
    def _selection(
        service: SceneTargetSelectionService,
        payload: dict[str, Any],
        user: User,
    ):
        selection = service.repository.get_target_selection_for_update(
            payload["selection_id"]
        )
        if selection is None:
            raise ValueError("Target selection not found.")
        service._assert_selection_owner(selection, user)
        return selection

    @staticmethod
    def _selection_result(selection) -> dict[str, Any]:
        return {
            "selection_id": selection.selection_id,
            "selection_url": (
                f"/api/v1/target-selections/{selection.selection_id}"
            ),
        }

    def _recoverable(self) -> list[str]:
        db = SessionLocal()
        try:
            running = list(
                db.scalars(
                    select(SceneAITask).where(
                        SceneAITask.status == "RUNNING"
                    )
                ).all()
            )
            for task in running:
                task.status = "QUEUED"
                task.process_pid = None
                task.error_message = (
                    "Recovered after backend restart; retrying."
                )
            queued = list(
                db.scalars(
                    select(SceneAITask).where(
                        SceneAITask.status == "QUEUED",
                        SceneAITask.attempt_count
                        < SceneAITask.max_attempts,
                    )
                ).all()
            )
            db.commit()
            return [task.task_id for task in queued]
        except Exception:
            db.rollback()
            logger.exception("Scene AI startup recovery failed.")
            return []
        finally:
            db.close()


_executor: SceneAITaskExecutor | None = None
_executor_lock = threading.Lock()


def get_scene_ai_task_executor() -> SceneAITaskExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = SceneAITaskExecutor()
        return _executor
