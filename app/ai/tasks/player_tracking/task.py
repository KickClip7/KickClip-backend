from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ai.runtime.base_task import BaseAITask
from app.ai.runtime.task_context import TaskContext
from app.ai.tasks.player_tracking.detector import create_player_detector
from app.ai.tasks.player_tracking.postprocessor import PlayerTrackingPostprocessor
from app.ai.tasks.player_tracking.tracker import create_player_tracker
from app.ai.tasks.player_tracking.types import PlayerTrackingInput
from app.domains.artifact.repository import ArtifactRepository
from app.domains.media.repository import MediaAssetRepository
from app.domains.player.schema import PlayerCreate, PlayerTrackCreate
from app.domains.player.service import PlayerService
from app.domains.timeline.repository import TimelineEventRepository
from app.storage.local_storage import LocalStorage
from app.storage.workspace import get_match_player_tracks_subdir


class PlayerTrackingTask(BaseAITask):
    task_type = "PLAYER_TRACKING"

    def __init__(self, mode: str = "dummy") -> None:
        self.storage = LocalStorage()
        self.mode = mode
        self.detector = create_player_detector(mode)
        self.tracker = create_player_tracker(mode)
        self.postprocessor = PlayerTrackingPostprocessor()

    def run(self, context: TaskContext) -> dict[str, Any]:
        context.mark_step_running("player_tracking")

        task_input = self._build_task_input(context)

        detected_players = self.detector.detect(task_input)
        track_seeds = self.tracker.build_tracks(
            task_input=task_input,
            detected_players=detected_players,
        )

        player_rows, raw_to_canonical_id = self.postprocessor.normalize_players(
            match_id=context.job.match_id,
            detected_players=detected_players,
        )

        # TimelineEvent.player_ids도 canonical player_id로 갱신한다.
        self._rewrite_timeline_event_player_ids(
            events=task_input.events,
            raw_to_canonical_id=raw_to_canonical_id,
        )

        created_at = datetime.now(timezone.utc).isoformat()
        result = self.postprocessor.build_artifact_payload(
            task_type=self.task_type,
            mode=self.mode,
            analysis_job_id=context.job.analysis_job_id,
            match_id=context.job.match_id,
            source_video_asset_id=task_input.source_video_asset_id,
            players=player_rows,
            track_seeds=track_seeds,
            raw_to_canonical_id=raw_to_canonical_id,
            diagnostics={
                "detector": getattr(self.detector, "name", type(self.detector).__name__),
                "tracker": getattr(self.tracker, "name", type(self.tracker).__name__),
                "postprocessor": self.postprocessor.name,
                "source_video_path": task_input.source_video_path.as_posix() if task_input.source_video_path else None,
                "event_count": len(task_input.events),
            },
            created_at=created_at,
        )

        context.mark_step_completed("player_tracking")

        scoring_step = context.get_step("scoring")
        if scoring_step is not None:
            context.mark_step_running("scoring")
            context.mark_step_completed("scoring")

        return result

    def save_artifacts(self, context: TaskContext, result: dict[str, Any]) -> None:
        artifact_path = self._save_player_tracks_json(context, result)

        artifact_repo = ArtifactRepository(context.db)
        artifact = artifact_repo.create(
            match_id=context.job.match_id,
            analysis_job_id=context.job.analysis_job_id,
            artifact_type="PLAYER_TRACKS",
            file_path=artifact_path,
            mime_type="application/json",
            metadata_={
                "schema_version": result["schema_version"],
                "task_type": self.task_type,
                "mode": result["mode"],
                "num_players": result["num_players"],
                "num_tracks": result["num_tracks"],
                "source_video_asset_id": result.get("source_video_asset_id"),
            },
        )
        context.db.flush()

        track_rows = self.postprocessor.normalize_tracks(
            match_id=context.job.match_id,
            source_job_id=context.job.analysis_job_id,
            track_artifact_id=artifact.artifact_id,
            track_seeds=[
                self.postprocessor.track_seed_from_dict(item)
                for item in result["tracks"]
            ],
            id_map=result["raw_to_canonical_id"],
        )

        player_service = PlayerService(context.db)
        player_service.replace_players_and_tracks_for_match(
            match_id=context.job.match_id,
            players=[PlayerCreate(**player) for player in result["players"]],
            tracks=[PlayerTrackCreate(**track) for track in track_rows],
            commit=False,
        )

        context.db.commit()

    def _build_task_input(self, context: TaskContext) -> PlayerTrackingInput:
        timeline_repo = TimelineEventRepository(context.db)
        events = timeline_repo.list_current_by_match(match_id=context.job.match_id)

        source_video_asset_id: str | None = None
        source_video_path: Path | None = None

        media_repo = MediaAssetRepository(context.db)
        assets = media_repo.list_by_match(context.job.match_id)
        raw_video = next((asset for asset in assets if asset.asset_type == "RAW_VIDEO"), None)
        if raw_video is not None:
            source_video_asset_id = raw_video.asset_id
            source_video_path = self.storage.project_root / raw_video.file_path

        match = context.job.match
        return PlayerTrackingInput(
            match_id=context.job.match_id,
            analysis_job_id=context.job.analysis_job_id,
            source_video_asset_id=source_video_asset_id,
            source_video_path=source_video_path,
            match_duration_sec=match.duration_sec,
            home_team=match.home_team,
            away_team=match.away_team,
            events=events,
            options=context.job.options or {},
        )

    def _save_player_tracks_json(
        self,
        context: TaskContext,
        result: dict[str, Any],
    ) -> str:
        output_dir = (
            self.storage.storage_root
            / get_match_player_tracks_subdir(context.job.match_id)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"player_tracks_{context.job.analysis_job_id}.json"
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        return self._to_project_relative_path(output_path)

    def _rewrite_timeline_event_player_ids(
        self,
        events,
        raw_to_canonical_id: dict[str, str],
    ) -> None:
        for event in events:
            if not event.player_ids:
                continue

            event.player_ids = [
                raw_to_canonical_id.get(player_id, player_id)
                for player_id in event.player_ids
            ]

    def _to_project_relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.storage.project_root).as_posix()
        except ValueError:
            return path.as_posix()
