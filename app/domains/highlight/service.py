from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.agent.rule_based_planner import LABEL_PRIORITY, RuleBasedClipPlanner
from app.domains.analysis.model import AnalysisJob
from app.domains.auth.model import User
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.highlight.action_cache import ActionSpottingCacheService
from app.domains.highlight.candidate_discovery import (
    PlayerCandidateDiscoveryService,
)
from app.domains.highlight.model import (
    HighlightRevision,
    PlayerFocusSubject,
    ScenePlayerCandidate,
    SceneTrackingBinding,
)
from app.domains.highlight.player_detector import discovery_failure_payload
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.request_parser import HighlightRequestParser
from app.domains.highlight.scene_clips import SceneClipService
from app.domains.highlight.schema import (
    HighlightAnalyzeResponse,
    HighlightClipPlanResponse,
    HighlightFocusMode,
    HighlightRequest,
    HighlightRevisionRead,
    HighlightSceneRead,
    HighlightScopeType,
    HighlightStatusResponse,
    PlayerCandidateRead,
    PlayerFocusSubjectRead,
    SceneTrackingRead,
)
from app.domains.highlight.timeline_mapping import TrackingTimelineMapper
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.player.repository import PlayerRepository
from app.domains.project.model import Project
from app.domains.render.schema import RenderCreateRequest
from app.domains.render.service import RenderJobService
from app.domains.render.tracking_transform import TrackingTransformBuilder
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.tracking.errors import TrackingError
from app.domains.tracking.executor import get_tracking_executor
from app.domains.tracking.schema import TrackingJobCreateRequest
from app.domains.tracking.service import TrackingJobService
from app.domains.tracking.status import WAITING_STATUSES, TrackingBackendStatus
from app.domains.tracking.timeline import TrackingTimelineService


class HighlightWorkflowService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.repository = HighlightRepository(db)
        self.timeline_repository = TimelineEventRepository(db)
        self.media_repository = MediaAssetRepository(db)
        self.clip_plan_repository = ClipPlanRepository(db)
        self.player_repository = PlayerRepository(db)
        self.request_parser = HighlightRequestParser()
        self.cache = ActionSpottingCacheService(db)

    # ------------------------------------------------------------------
    # Analysis and immutable scene-selection revisions
    # ------------------------------------------------------------------
    def analyze(
        self,
        *,
        project: Project,
        user_request: str,
        structured_request: HighlightRequest | None = None,
    ) -> tuple[HighlightRevision, bool, AnalysisJob]:
        current = self.repository.get_current_revision(project.project_id)
        parsed = structured_request or self.request_parser.parse(
            user_request,
            match_metadata=project.match.metadata_ or {},
            current_revision_exists=current is not None,
        )
        action_job, reused = self.cache.get_or_create(
            match_id=project.match_id,
            request_options={
                "run_feature_extraction": True,
                "highlight_workflow": True,
            },
        )
        revision = self.repository.create_revision(
            project_id=project.project_id,
            revision_number=self.repository.next_revision_number(project.project_id),
            parent_revision_id=current.revision_id if current else None,
            action_spotting_job_id=action_job.analysis_job_id,
            user_request=user_request,
            structured_request=parsed.model_dump(mode="json"),
            selected_scene_ids=[],
            scene_selection=[],
            focus_mode=parsed.focus_mode.value,
            status="ACTION_SPOTTING_RUNNING",
            pending_action="SELECT_SCENES",
            options={
                "action_spotting_reused": reused,
                "action_spotting_cache_key": action_job.cache_key,
            },
        )
        self.db.commit()
        self.db.refresh(revision)
        if action_job.status == "COMPLETED":
            revision = self.reconcile_revision(revision)
            if (
                parsed.focus_mode == HighlightFocusMode.PLAYER
                and revision.selected_scene_ids
            ):
                revision.status = "PLAYER_DISCOVERY_RUNNING"
                revision.pending_action = "SELECT_PLAYER"
                self.db.commit()
                self.db.refresh(revision)
        return revision, reused, action_job

    def reconcile_revision(
        self,
        revision: HighlightRevision,
    ) -> HighlightRevision:
        if revision.action_spotting_job_id and not revision.scene_selection:
            job = revision.action_spotting_job
            if job is None:
                job = self.db.get(AnalysisJob, revision.action_spotting_job_id)
            if job is None:
                revision.status = "FAILED"
                revision.error_message = "Action Spotting job no longer exists."
            elif job.status == "COMPLETED":
                self._select_initial_scenes(revision, job)
            elif job.status == "FAILED":
                revision.status = "FAILED"
                revision.pending_action = None
                revision.error_message = job.error_message
            else:
                revision.status = "ACTION_SPOTTING_RUNNING"
                revision.pending_action = "SELECT_SCENES"

        if revision.focus_mode == HighlightFocusMode.PLAYER.value:
            self._reconcile_tracking(revision)

        if revision.render_job_id:
            render = revision.render_job
            if render is None:
                render = RenderJobService(self.db).get_render_job(
                    revision.render_job_id
                )
            if render is not None:
                if render.status == "completed":
                    revision.status = "COMPLETED"
                    revision.pending_action = "COMPLETED"
                elif render.status == "failed":
                    revision.status = "FAILED"
                    revision.pending_action = None
                    revision.error_message = render.error_message
                elif render.status in {"queued", "running"}:
                    revision.status = "RENDERING"

        self.db.commit()
        self.db.refresh(revision)
        return revision

    def select_scenes(
        self,
        *,
        project: Project,
        scene_ids: list[str],
        selection_source: str,
        render_strategies: dict[str, str],
    ) -> HighlightRevision:
        current = self.require_current_revision(project.project_id)
        current = self.reconcile_revision(current)
        available = {
            event.timeline_event_id
            for event in self._action_scenes(current)
        }
        unknown = set(scene_ids).difference(available)
        if unknown:
            raise ValueError("One or more scenes do not belong to this workflow.")
        selection = []
        selected = set(scene_ids)
        previous_strategies = {
            row["scene_id"]: row.get("render_strategy", "FULL_FRAME")
            for row in current.scene_selection or []
        }
        for scene_id in available:
            included = scene_id in selected
            selection.append(
                {
                    "scene_id": scene_id,
                    "state": (
                        f"{selection_source}_SELECTED"
                        if included
                        else "USER_EXCLUDED"
                    ),
                    "render_strategy": render_strategies.get(
                        scene_id,
                        previous_strategies.get(scene_id, "FULL_FRAME"),
                    )
                    if included
                    else "EXCLUDE",
                }
            )
        revision = self._fork_revision(
            current,
            selected_scene_ids=scene_ids,
            scene_selection=selection,
            focus_mode=current.focus_mode,
            focus_subject_id=current.focus_subject_id,
            status="SCENES_SELECTED",
            pending_action="READY_TO_PLAN",
            clip_plan_id=None,
            render_job_id=None,
        )
        self.db.commit()
        self.db.refresh(revision)
        return revision

    # ------------------------------------------------------------------
    # Candidate discovery and scene-local user confirmation
    # ------------------------------------------------------------------
    def start_player_focus(
        self,
        *,
        project: Project,
        user_request: str,
        source_revision_id: str | None,
    ) -> HighlightRevision:
        source = (
            self.require_revision(project.project_id, source_revision_id)
            if source_revision_id
            else self.require_current_revision(project.project_id)
        )
        source = self.reconcile_revision(source)
        if not source.selected_scene_ids:
            raise ValueError("Player focus requires selected highlight scenes.")
        parsed = self.request_parser.parse(
            user_request,
            match_metadata=project.match.metadata_ or {},
            current_revision_exists=True,
        )
        parsed.scope.type = HighlightScopeType.SELECTED_SCENES
        parsed.focus_mode = HighlightFocusMode.PLAYER
        revision = self._fork_revision(
            source,
            user_request=user_request,
            structured_request=parsed.model_dump(mode="json"),
            focus_mode=HighlightFocusMode.PLAYER.value,
            focus_subject_id=None,
            clip_plan_id=None,
            render_job_id=None,
            status="PLAYER_DISCOVERY_RUNNING",
            pending_action="SELECT_PLAYER",
        )
        self.db.commit()
        self.db.refresh(revision)
        return revision

    def run_candidate_discovery(self, revision_id: str) -> None:
        revision = self.repository.get_revision(revision_id)
        if revision is None:
            return
        try:
            source_asset = self._revision_source_asset(revision)
            scenes = self._selected_scenes(revision)
            PlayerCandidateDiscoveryService(self.db).discover(
                revision=revision,
                source_asset=source_asset,
                scenes=scenes,
            )
            revision.status = "PLAYER_SELECTION_REQUIRED"
            revision.pending_action = "SELECT_PLAYER"
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            failed = self.repository.get_revision(revision_id)
            if failed is not None:
                failed.status = "FAILED"
                failed.pending_action = None
                failed.error_message = str(exc)
                failed.options = {
                    **(failed.options or {}),
                    "candidate_discovery": discovery_failure_payload(
                        exc,
                        settings=self.settings,
                    ),
                }
                self.db.commit()

    def select_focus_subject(
        self,
        *,
        project: Project,
        revision_id: str | None,
        display_name: str,
        anchor_scene_id: str,
        candidate_id: str,
    ) -> tuple[HighlightRevision, PlayerFocusSubject, SceneTrackingBinding]:
        revision = (
            self.require_revision(project.project_id, revision_id)
            if revision_id
            else self.require_current_revision(project.project_id)
        )
        if revision.focus_mode != HighlightFocusMode.PLAYER.value:
            raise ValueError("Revision is not in PLAYER focus mode.")
        candidate = self._validated_candidate(
            revision=revision,
            scene_id=anchor_scene_id,
            candidate_id=candidate_id,
        )
        if revision.focus_subject_id:
            raise ValueError("Revision already has a focus subject.")
        subject = self.repository.create_focus_subject(
            project_id=project.project_id,
            display_name=display_name,
            identity_source="USER_DEFINED",
            anchor_scene_id=anchor_scene_id,
            anchor_candidate_id=candidate_id,
            metadata_={
                "scope": "PROJECT_EDIT_SUBJECT",
                "is_match_global_identity": False,
            },
        )
        revision.focus_subject_id = subject.focus_subject_id
        revision.status = "TRACKING_RUNNING"
        revision.pending_action = "CONFIRM_PLAYER_CANDIDATE"
        binding = self._create_candidate_binding(
            revision=revision,
            subject=subject,
            candidate=candidate,
        )
        self.db.commit()
        self.db.refresh(revision)
        self.db.refresh(subject)
        self.db.refresh(binding)
        return revision, subject, binding

    def confirm_scene_player(
        self,
        *,
        project: Project,
        scene_id: str,
        revision_id: str | None,
        decision: str,
        candidate_id: str | None,
        render_strategy: str | None,
    ) -> tuple[HighlightRevision, SceneTrackingBinding]:
        revision = (
            self.require_revision(project.project_id, revision_id)
            if revision_id
            else self.require_current_revision(project.project_id)
        )
        if scene_id not in revision.selected_scene_ids:
            raise ValueError("Scene is not selected in this revision.")
        if revision.focus_subject_id is None:
            raise ValueError("Select an anchor player before confirming scenes.")
        existing = self.repository.get_binding(
            revision_id=revision.revision_id,
            scene_id=scene_id,
        )
        if existing is not None:
            return revision, existing
        subject = self.repository.get_focus_subject(revision.focus_subject_id)
        if subject is None:
            raise ValueError("Focus subject no longer exists.")

        if decision == "absent":
            binding = self.repository.create_binding(
                revision_id=revision.revision_id,
                scene_id=scene_id,
                focus_subject_id=subject.focus_subject_id,
                selected_candidate_id=None,
                confirmation_source="USER",
                target_presence_status="ABSENT",
                render_strategy=render_strategy or "EXCLUDE",
                status="COMPLETED",
                timeline_summary={
                    "target_absent_segments": [
                        {
                            "start_time_sec": self._scene(scene_id).start_sec,
                            "end_time_sec": self._scene(scene_id).end_sec,
                            "states": ["ABSENT"],
                        }
                    ],
                    "crop_segments": [],
                },
                metadata_={"decision": "USER_CONFIRMED_ABSENT"},
            )
        else:
            if not candidate_id:
                raise ValueError("candidate_id is required.")
            candidate = self._validated_candidate(
                revision=revision,
                scene_id=scene_id,
                candidate_id=candidate_id,
            )
            binding = self._create_candidate_binding(
                revision=revision,
                subject=subject,
                candidate=candidate,
            )
        revision.status = "TRACKING_RUNNING"
        revision.pending_action = "CONFIRM_PLAYER_CANDIDATE"
        self.db.commit()
        self.db.refresh(binding)
        return revision, binding

    def start_tracking_for_binding(
        self,
        *,
        binding_id: str,
        user: User,
    ) -> SceneTrackingBinding:
        binding = self.db.get(SceneTrackingBinding, binding_id)
        if binding is None:
            raise ValueError("Scene tracking binding not found.")
        if binding.tracking_job_id:
            return binding
        revision = self.repository.get_revision(binding.revision_id)
        if revision is None:
            raise ValueError("Highlight revision not found.")
        candidate = self.repository.get_candidate(
            revision_id=revision.revision_id,
            candidate_id=binding.selected_candidate_id or "",
        )
        if candidate is None or candidate.scene_id != binding.scene_id:
            raise ValueError("Confirmed player candidate is invalid.")
        scene = self._scene(binding.scene_id)
        source_asset = self._revision_source_asset(revision)
        project = revision.project
        try:
            extracted = SceneClipService(self.db).extract_for_candidate(
                source_asset=source_asset,
                project_id=revision.project_id,
                revision=revision,
                scene=scene,
                candidate=candidate,
            )
            response = TrackingJobService(self.db).create_job(
                user=user,
                asset=extracted.asset,
                project=project,
                payload=TrackingJobCreateRequest(
                    media_asset_id=extracted.asset.asset_id,
                    initial_bbox_xyxy=candidate.bbox_xyxy,
                    project_id=revision.project_id,
                    match_id=scene.match_id,
                    reacquisition_mode="assisted",
                ),
            )
            binding.scene_clip_asset_id = extracted.asset.asset_id
            binding.tracking_job_id = response.job_id
            binding.source_start_time_sec = extracted.source_start_time_sec
            binding.source_end_time_sec = extracted.source_end_time_sec
            binding.source_start_frame = extracted.source_start_frame
            binding.source_end_frame = extracted.source_end_frame
            binding.source_fps = extracted.source_fps
            binding.clip_fps = extracted.clip_fps
            binding.clip_frame_count = extracted.clip_frame_count
            binding.status = "TRACKING_QUEUED"
            binding.target_presence_status = "SEARCHING"
            binding.metadata_ = {
                **(binding.metadata_ or {}),
                "clip_extraction": {
                    "command": extracted.command,
                    "mapping": "anchor_rebased_local_zero",
                },
            }
            self.db.commit()
            get_tracking_executor().submit(response.job_id)
            return binding
        except Exception as exc:
            self.db.rollback()
            failed = self.db.get(SceneTrackingBinding, binding_id)
            if failed is not None:
                failed.status = "FAILED"
                failed.error_message = str(exc)
                self.db.commit()
            raise

    # ------------------------------------------------------------------
    # Tracking-aware planning and rendering
    # ------------------------------------------------------------------
    def create_clip_plan(
        self,
        *,
        project: Project,
        revision_id: str | None,
        allow_absent_full_frame: bool,
    ) -> HighlightClipPlanResponse:
        revision = (
            self.require_revision(project.project_id, revision_id)
            if revision_id
            else self.require_current_revision(project.project_id)
        )
        revision = self.reconcile_revision(revision)
        events = self._selected_scenes(revision)
        if not events:
            raise ValueError("Revision has no selected scenes.")
        request = HighlightRequest.model_validate(revision.structured_request)
        planned = RuleBasedClipPlanner().build_plan(
            prompt=revision.user_request,
            events=events,
            players=self.player_repository.list_by_match(project.match_id),
            target_duration_sec=request.desired_duration_sec,
            selected_player_id=None,
            options={"max_items": 12},
        )
        plan = self.clip_plan_repository.create_plan(
            project_id=project.project_id,
            mode=(
                "TRACKING_AWARE_AGENT"
                if revision.focus_mode == HighlightFocusMode.PLAYER.value
                else "AGENT_GENERATED"
            ),
            summary=planned.summary,
            target_duration_sec=planned.target_duration_sec,
            actual_duration_sec=0,
            created_by="agent",
            options={
                "highlight_revision_id": revision.revision_id,
                "prompt": revision.user_request,
                "ratio": request.aspect_ratio,
                "captions_enabled": request.subtitle_style is not None,
                "focus_mode": revision.focus_mode,
                "focus_subject_id": revision.focus_subject_id,
                "render_provenance": {
                    "clip_plan_revision": revision.revision_id,
                    "focus_subject_id": revision.focus_subject_id,
                    "crop_policy_version": "target_centered_crop_v1",
                },
            },
        )
        event_by_id = {event.timeline_event_id: event for event in events}
        bindings = {
            row.scene_id: row
            for row in self.repository.list_bindings(revision.revision_id)
        }
        item_count = 0
        total_duration = 0.0
        included_scene_ids: set[str] = set()
        included_tracking_job_ids: set[str] = set()
        for planned_item in planned.items:
            event = event_by_id[planned_item.timeline_event_id]
            if revision.focus_mode == HighlightFocusMode.PLAYER.value:
                created = self._create_tracking_plan_items(
                    clip_plan_id=plan.clip_plan_id,
                    order_start=item_count,
                    event=event,
                    binding=bindings.get(event.timeline_event_id),
                    allow_absent_full_frame=allow_absent_full_frame,
                    desired_remaining=(
                        max(
                            0.0,
                            float(planned.target_duration_sec) - total_duration,
                        )
                        if planned.target_duration_sec is not None
                        else None
                    ),
                )
                item_count += len(created)
                total_duration += sum(row["duration_sec"] for row in created)
                if created:
                    included_scene_ids.add(event.timeline_event_id)
                    binding = bindings.get(event.timeline_event_id)
                    if binding and binding.tracking_job_id:
                        included_tracking_job_ids.add(binding.tracking_job_id)
            else:
                duration = float(planned_item.end_sec) - float(planned_item.start_sec)
                self.clip_plan_repository.create_item(
                    clip_plan_id=plan.clip_plan_id,
                    timeline_event_id=event.timeline_event_id,
                    start_sec=planned_item.start_sec,
                    end_sec=planned_item.end_sec,
                    duration_sec=duration,
                    order_index=item_count,
                    reason=planned_item.reason,
                    metadata_={
                        "scene_id": event.timeline_event_id,
                        "source_start_time_sec": planned_item.start_sec,
                        "source_end_time_sec": planned_item.end_sec,
                        "focus_mode": "NONE",
                        "render_strategy": "FULL_FRAME",
                    },
                )
                item_count += 1
                total_duration += duration
                included_scene_ids.add(event.timeline_event_id)

        if item_count == 0:
            self.db.rollback()
            if revision.focus_mode == HighlightFocusMode.PLAYER.value:
                raise ValueError("NO_TARGET_SCENES")
            raise ValueError("No selected scenes can be planned.")
        plan.actual_duration_sec = round(total_duration, 3)
        plan.options = {
            **(plan.options or {}),
            "render_provenance": {
                **((plan.options or {}).get("render_provenance") or {}),
                "source_scene_ids": sorted(included_scene_ids),
                "tracking_job_ids": sorted(included_tracking_job_ids),
                "timeline_schema_version": "kickclip.target_centric_e2e.v1"
                if included_tracking_job_ids
                else None,
            },
        }
        revision.scene_selection = [
            {
                **row,
                "state": (
                    "INCLUDED_IN_CLIP_PLAN"
                    if row["scene_id"] in included_scene_ids
                    else row.get("state", "CANDIDATE")
                ),
            }
            for row in revision.scene_selection or []
        ]
        revision.clip_plan_id = plan.clip_plan_id
        revision.status = "CLIP_PLAN_READY"
        revision.pending_action = "READY_TO_RENDER"
        self.db.commit()
        self.db.refresh(revision)
        return HighlightClipPlanResponse(
            revision=self.revision_read(revision),
            clip_plan_id=plan.clip_plan_id,
            total_duration_sec=round(total_duration, 3),
            item_count=item_count,
        )

    def create_render(
        self,
        *,
        project: Project,
        revision_id: str | None,
        options: dict[str, Any],
    ):
        revision = (
            self.require_revision(project.project_id, revision_id)
            if revision_id
            else self.require_current_revision(project.project_id)
        )
        if not revision.clip_plan_id:
            raise ValueError("Create a ClipPlan before rendering.")
        request = HighlightRequest.model_validate(revision.structured_request)
        render = RenderJobService(self.db).create_render_job(
            RenderCreateRequest(
                clip_plan_id=revision.clip_plan_id,
                options={
                    "ratio": request.aspect_ratio,
                    **options,
                },
            )
        )
        revision = self.repository.get_revision(revision.revision_id)
        if revision is None:
            raise ValueError("Highlight revision not found.")
        revision.render_job_id = render.render_job_id
        revision.status = "RENDERING"
        revision.pending_action = None
        self.db.commit()
        self.db.refresh(revision)
        return revision, render

    # ------------------------------------------------------------------
    # Read models and status
    # ------------------------------------------------------------------
    def scenes(self, project_id: str) -> tuple[HighlightRevision, list[HighlightSceneRead]]:
        revision = self.reconcile_revision(
            self.require_current_revision(project_id)
        )
        selection = {
            row["scene_id"]: row
            for row in revision.scene_selection or []
        }
        media_asset_id = (
            revision.action_spotting_job.media_asset_id
            if revision.action_spotting_job
            else None
        )
        rows = []
        for event in self._action_scenes(revision):
            state = selection.get(
                event.timeline_event_id,
                {
                    "state": "CANDIDATE",
                    "render_strategy": "EXCLUDE",
                },
            )
            rows.append(
                HighlightSceneRead(
                    scene_id=event.timeline_event_id,
                    match_id=event.match_id,
                    media_asset_id=media_asset_id,
                    start_time_sec=event.start_sec,
                    end_time_sec=event.end_sec,
                    representative_time_sec=event.timestamp_sec,
                    primary_label=event.label,
                    confidence=event.confidence,
                    source_predictions=(event.metadata_ or {}).get(
                        "source_predictions",
                        [],
                    ),
                    provenance=(event.metadata_ or {}).get(
                        "scene_provenance",
                        {},
                    ),
                    selection_state=state["state"],
                    render_strategy=state.get("render_strategy", "EXCLUDE"),
                )
            )
        return revision, rows

    def candidates(
        self,
        project_id: str,
    ) -> tuple[HighlightRevision, list[PlayerCandidateRead]]:
        revision = self.reconcile_revision(
            self.require_current_revision(project_id)
        )
        rows = []
        for index, candidate in enumerate(
            self.repository.list_candidates(revision.revision_id),
            start=1,
        ):
            rows.append(
                PlayerCandidateRead(
                    candidate_id=candidate.candidate_id,
                    scene_id=candidate.scene_id,
                    display_label=(candidate.metadata_ or {}).get(
                        "display_label",
                        f"선수 후보 {index}",
                    ),
                    anchor_time_sec=candidate.anchor_time_sec,
                    anchor_source_time_sec=candidate.anchor_source_time_sec,
                    anchor_frame_index=candidate.anchor_frame_index,
                    bbox_xyxy=candidate.bbox_xyxy,
                    thumbnail_artifact_id=candidate.thumbnail_artifact_id,
                    thumbnail_url=(
                        f"/api/v1/artifacts/{candidate.thumbnail_artifact_id}/download"
                        if candidate.thumbnail_artifact_id
                        else None
                    ),
                    track_length_frames=candidate.track_length_frames,
                    trackability_score=candidate.trackability_score,
                    status=candidate.status,
                    detector_provenance=(candidate.metadata_ or {}).get(
                        "detector_provenance",
                        {},
                    ),
                )
            )
        return revision, rows

    def status(self, project_id: str) -> HighlightStatusResponse:
        revision = self.reconcile_revision(
            self.require_current_revision(project_id)
        )
        subject = revision.focus_subject
        bindings = self.repository.list_bindings(revision.revision_id)
        return HighlightStatusResponse(
            revision=self.revision_read(revision),
            action_spotting_status=(
                revision.action_spotting_job.status
                if revision.action_spotting_job
                else None
            ),
            pending_action=revision.pending_action,
            focus_subject=(
                PlayerFocusSubjectRead(
                    focus_subject_id=subject.focus_subject_id,
                    display_name=subject.display_name,
                    identity_source=subject.identity_source,
                    anchor_scene_id=subject.anchor_scene_id,
                    anchor_candidate_id=subject.anchor_candidate_id,
                )
                if subject
                else None
            ),
            scene_tracking=[
                SceneTrackingRead(
                    scene_id=binding.scene_id,
                    binding_id=binding.binding_id,
                    candidate_id=binding.selected_candidate_id,
                    tracking_job_id=binding.tracking_job_id,
                    status=binding.status,
                    tracking_status=(
                        binding.tracking_job.status
                        if binding.tracking_job
                        else None
                    ),
                    target_presence_status=binding.target_presence_status,
                    render_strategy=binding.render_strategy,
                    timeline_summary=binding.timeline_summary or {},
                    error_message=binding.error_message,
                )
                for binding in bindings
            ],
        )

    @staticmethod
    def revision_read(revision: HighlightRevision) -> HighlightRevisionRead:
        return HighlightRevisionRead(
            revision_id=revision.revision_id,
            project_id=revision.project_id,
            revision_number=revision.revision_number,
            parent_revision_id=revision.parent_revision_id,
            action_spotting_job_id=revision.action_spotting_job_id,
            user_request=revision.user_request,
            request=HighlightRequest.model_validate(revision.structured_request),
            selected_scene_ids=revision.selected_scene_ids or [],
            focus_mode=revision.focus_mode,
            focus_subject_id=revision.focus_subject_id,
            clip_plan_id=revision.clip_plan_id,
            render_job_id=revision.render_job_id,
            status=revision.status,
            pending_action=revision.pending_action,
            error_message=revision.error_message,
            candidate_discovery=(revision.options or {}).get(
                "candidate_discovery",
                {},
            ),
            created_at=revision.created_at,
            updated_at=revision.updated_at,
        )

    def require_current_revision(self, project_id: str) -> HighlightRevision:
        revision = self.repository.get_current_revision(project_id)
        if revision is None:
            raise ValueError("Highlight workflow has not been started.")
        return revision

    def require_revision(
        self,
        project_id: str,
        revision_id: str,
    ) -> HighlightRevision:
        revision = self.repository.get_revision(revision_id)
        if revision is None or revision.project_id != project_id:
            raise ValueError("Highlight revision not found.")
        return revision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_initial_scenes(
        self,
        revision: HighlightRevision,
        job: AnalysisJob,
    ) -> None:
        request = HighlightRequest.model_validate(revision.structured_request)
        events = self.timeline_repository.list_by_source_job(job.analysis_job_id)
        eligible = self._filter_scope(events, request)
        if request.event_labels:
            allowed = set(request.event_labels)
            eligible = [row for row in eligible if row.label in allowed]
        ranked = sorted(
            eligible,
            key=lambda row: (
                float(row.highlight_score or 0.0),
                float(row.confidence or 0.0),
                float(LABEL_PRIORITY.get(row.label, 0)),
            ),
            reverse=True,
        )
        selected: list[TimelineEvent] = []
        duration = 0.0
        for event in ranked:
            event_duration = max(0.0, float(event.end_sec) - float(event.start_sec))
            if selected and duration + event_duration > request.desired_duration_sec * 1.15:
                continue
            selected.append(event)
            duration += event_duration
            if duration >= request.desired_duration_sec:
                break

        selected_ids = {row.timeline_event_id for row in selected}
        revision.selected_scene_ids = [
            row.timeline_event_id for row in selected
        ]
        revision.scene_selection = [
            {
                "scene_id": event.timeline_event_id,
                "state": (
                    "AGENT_SELECTED"
                    if event.timeline_event_id in selected_ids
                    else "CANDIDATE"
                ),
                "render_strategy": (
                    "FULL_FRAME"
                    if event.timeline_event_id in selected_ids
                    else "EXCLUDE"
                ),
            }
            for event in events
        ]
        if selected:
            revision.status = "SCENES_SELECTED"
            revision.pending_action = "READY_TO_PLAN"
        else:
            revision.status = "SCENES_READY"
            revision.pending_action = "SELECT_SCENES"

    @staticmethod
    def _filter_scope(
        events: list[TimelineEvent],
        request: HighlightRequest,
    ) -> list[TimelineEvent]:
        scope = request.scope
        if scope.start_time_sec is not None or scope.end_time_sec is not None:
            start = float(scope.start_time_sec or 0.0)
            end = (
                float(scope.end_time_sec)
                if scope.end_time_sec is not None
                else float("inf")
            )
            return [
                event
                for event in events
                if event.timestamp_sec >= start and event.timestamp_sec <= end
            ]
        if scope.type == HighlightScopeType.FIRST_HALF:
            return [event for event in events if event.half == 1]
        elif scope.type == HighlightScopeType.SECOND_HALF:
            return [event for event in events if event.half == 2]
        if scope.requires_period_boundary:
            return []
        return events

    def _fork_revision(
        self,
        source: HighlightRevision,
        **overrides,
    ) -> HighlightRevision:
        values = {
            "project_id": source.project_id,
            "revision_number": self.repository.next_revision_number(source.project_id),
            "parent_revision_id": source.revision_id,
            "action_spotting_job_id": source.action_spotting_job_id,
            "user_request": source.user_request,
            "structured_request": dict(source.structured_request or {}),
            "selected_scene_ids": list(source.selected_scene_ids or []),
            "scene_selection": [dict(row) for row in source.scene_selection or []],
            "focus_mode": source.focus_mode,
            "focus_subject_id": source.focus_subject_id,
            "clip_plan_id": source.clip_plan_id,
            "render_job_id": source.render_job_id,
            "status": source.status,
            "pending_action": source.pending_action,
            "options": dict(source.options or {}),
        }
        values.update(overrides)
        return self.repository.create_revision(**values)

    def _action_scenes(self, revision: HighlightRevision) -> list[TimelineEvent]:
        if revision.action_spotting_job_id is None:
            return []
        return self.timeline_repository.list_by_source_job(
            revision.action_spotting_job_id
        )

    def _selected_scenes(self, revision: HighlightRevision) -> list[TimelineEvent]:
        selected = set(revision.selected_scene_ids or [])
        return [
            scene
            for scene in self._action_scenes(revision)
            if scene.timeline_event_id in selected
        ]

    def _revision_source_asset(self, revision: HighlightRevision) -> MediaAsset:
        job = revision.action_spotting_job
        if job and job.media_asset_id:
            asset = self.media_repository.get_by_id(job.media_asset_id)
            if asset is not None:
                return asset
        return self.cache.source_asset(revision.project.match_id)

    def _validated_candidate(
        self,
        *,
        revision: HighlightRevision,
        scene_id: str,
        candidate_id: str,
    ) -> ScenePlayerCandidate:
        if scene_id not in revision.selected_scene_ids:
            raise ValueError("Candidate scene is not selected in this revision.")
        candidate = self.repository.get_candidate(
            revision_id=revision.revision_id,
            candidate_id=candidate_id,
        )
        if candidate is None or candidate.scene_id != scene_id:
            raise ValueError("Player candidate does not belong to this scene.")
        if candidate.status != "AVAILABLE":
            raise ValueError("Player candidate is not available.")
        return candidate

    def _create_candidate_binding(
        self,
        *,
        revision: HighlightRevision,
        subject: PlayerFocusSubject,
        candidate: ScenePlayerCandidate,
    ) -> SceneTrackingBinding:
        existing = self.repository.get_binding(
            revision_id=revision.revision_id,
            scene_id=candidate.scene_id,
        )
        if existing is not None:
            return existing
        candidate.status = "USER_SELECTED"
        return self.repository.create_binding(
            revision_id=revision.revision_id,
            scene_id=candidate.scene_id,
            focus_subject_id=subject.focus_subject_id,
            selected_candidate_id=candidate.candidate_id,
            confirmation_source="USER",
            target_presence_status="SEARCHING",
            render_strategy="TARGET_CENTERED",
            status="QUEUED_FOR_EXTRACTION",
            metadata_={"identity_decision": "USER_CONFIRMED_SCENE_LOCAL_CANDIDATE"},
        )

    def _reconcile_tracking(self, revision: HighlightRevision) -> None:
        if revision.status == "PLAYER_DISCOVERY_RUNNING":
            return
        bindings = self.repository.list_bindings(revision.revision_id)
        waiting = False
        running = False
        failed = False
        for binding in bindings:
            job = binding.tracking_job
            if job is None:
                if binding.status in {"QUEUED_FOR_EXTRACTION", "TRACKING_QUEUED"}:
                    running = True
                continue
            if job.status in WAITING_STATUSES:
                waiting = True
                binding.status = "TRACKING_CONFIRMATION_REQUIRED"
                binding.target_presence_status = "AMBIGUOUS"
            elif job.status in {
                TrackingBackendStatus.QUEUED.value,
                TrackingBackendStatus.RUNNING.value,
            }:
                running = True
                binding.status = "TRACKING_RUNNING"
            elif job.status == TrackingBackendStatus.COMPLETED.value:
                try:
                    timeline = TrackingTimelineService().read(job)
                    summary = TrackingTimelineMapper.summarize(binding, timeline)
                    binding.timeline_summary = summary
                    binding.status = "COMPLETED"
                    binding.target_presence_status = (
                        "PRESENT"
                        if summary.get("crop_segments")
                        else "ABSENT"
                    )
                except TrackingError as exc:
                    binding.status = "FAILED"
                    binding.error_message = str(exc)
                    failed = True
            elif job.status == TrackingBackendStatus.COMPLETED_SAFE_BLOCK.value:
                binding.status = "COMPLETED_SAFE_BLOCK"
                binding.target_presence_status = "SEARCHING"
            elif job.status in {
                TrackingBackendStatus.FAILED.value,
                TrackingBackendStatus.CANCELLED.value,
            }:
                binding.status = "FAILED"
                binding.error_message = job.error_message
                failed = True

        selected = set(revision.selected_scene_ids or [])
        bound = {row.scene_id for row in bindings}
        missing = selected.difference(bound)
        crop_ready = any(
            bool((row.timeline_summary or {}).get("crop_segments"))
            for row in bindings
        )
        if waiting:
            revision.status = "TRACKING_CONFIRMATION_REQUIRED"
            revision.pending_action = (
                "REVIEW_TRACKING_MEMORY"
                if any(
                    row.tracking_job
                    and row.tracking_job.status
                    == TrackingBackendStatus.WAITING_MEMORY_REVIEW.value
                    for row in bindings
                )
                else "REVIEW_TRACKING_SEGMENT"
            )
        elif running:
            revision.status = "TRACKING_RUNNING"
            revision.pending_action = (
                "CONFIRM_PLAYER_CANDIDATE" if missing else None
            )
        elif missing:
            revision.status = "PLAYER_SELECTION_REQUIRED"
            revision.pending_action = "CONFIRM_PLAYER_CANDIDATE"
        elif crop_ready:
            revision.status = "SCENES_SELECTED"
            revision.pending_action = "READY_TO_PLAN"
        elif bindings and not failed:
            revision.status = "NO_TARGET_SCENES"
            revision.pending_action = "SELECT_PLAYER"
        elif failed:
            revision.status = "TRACKING_CONFIRMATION_REQUIRED"
            revision.pending_action = "REVIEW_TRACKING_SEGMENT"

    def _create_tracking_plan_items(
        self,
        *,
        clip_plan_id: str,
        order_start: int,
        event: TimelineEvent,
        binding: SceneTrackingBinding | None,
        allow_absent_full_frame: bool,
        desired_remaining: float | None,
    ) -> list[dict[str, float]]:
        if binding is None:
            return []
        if binding.render_strategy == "FULL_FRAME":
            if not allow_absent_full_frame:
                return []
            duration = float(event.end_sec) - float(event.start_sec)
            self.clip_plan_repository.create_item(
                clip_plan_id=clip_plan_id,
                timeline_event_id=event.timeline_event_id,
                start_sec=event.start_sec,
                end_sec=event.end_sec,
                duration_sec=duration,
                order_index=order_start,
                reason="User explicitly kept target-absent scene full-frame",
                metadata_={
                    "scene_id": event.timeline_event_id,
                    "focus_mode": "PLAYER",
                    "focus_subject_id": binding.focus_subject_id,
                    "render_strategy": "FULL_FRAME",
                    "target_presence_status": binding.target_presence_status,
                },
            )
            return [{"duration_sec": duration}]
        if (
            binding.render_strategy != "TARGET_CENTERED"
            or binding.tracking_job is None
            or binding.status != "COMPLETED"
        ):
            return []

        timeline = TrackingTimelineService().read(binding.tracking_job)
        summary = binding.timeline_summary or TrackingTimelineMapper.summarize(
            binding,
            timeline,
        )
        source_width = int(timeline["video"]["width"])
        source_height = int(timeline["video"]["height"])
        request_aspect = HighlightRequest.model_validate(
            binding.revision.structured_request
        ).aspect_ratio
        if request_aspect == "9:16":
            output_width, output_height = 1080, 1920
        elif request_aspect == "1:1":
            output_width, output_height = 1080, 1080
        else:
            output_width, output_height = 1920, 1080

        created: list[dict[str, float]] = []
        for segment in summary.get("crop_segments") or []:
            start = max(float(segment["start_time_sec"]), float(event.start_sec))
            end = min(float(segment["end_time_sec"]), float(event.end_sec))
            if desired_remaining is not None:
                end = min(end, start + max(0.0, desired_remaining))
            if end - start < 0.1:
                continue
            keyframes = TrackingTimelineMapper.crop_keyframes(
                binding,
                timeline,
                source_start_sec=start,
                source_end_sec=end,
            )
            if not keyframes:
                continue
            transform = TrackingTransformBuilder().build(
                keyframes=keyframes,
                source_width=source_width,
                source_height=source_height,
                output_width=output_width,
                output_height=output_height,
            )
            duration = end - start
            self.clip_plan_repository.create_item(
                clip_plan_id=clip_plan_id,
                timeline_event_id=event.timeline_event_id,
                start_sec=start,
                end_sec=end,
                duration_sec=duration,
                order_index=order_start + len(created),
                reason="User-confirmed target is trackable in this scene range",
                metadata_={
                    "scene_id": event.timeline_event_id,
                    "source_start_time_sec": start,
                    "source_end_time_sec": end,
                    "focus_mode": "PLAYER",
                    "focus_subject_id": binding.focus_subject_id,
                    "tracking_job_id": binding.tracking_job_id,
                    "timeline_version": timeline.get("schema_version"),
                    "render_strategy": "TARGET_CENTERED",
                    "target_presence_status": "PRESENT",
                    "tracking_transform": transform,
                },
            )
            created.append({"duration_sec": duration})
            if desired_remaining is not None:
                desired_remaining -= duration
                if desired_remaining <= 0:
                    break
        return created

    def _scene(self, scene_id: str) -> TimelineEvent:
        scene = self.timeline_repository.get_by_id(scene_id)
        if scene is None:
            raise ValueError("Highlight scene not found.")
        return scene
