from __future__ import annotations

import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.ai.tasks.highlight_spotting.postprocessor import HighlightPostprocessor
from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.agent.rule_based_planner import RuleBasedClipPlanner
from app.domains.analysis.model import AnalysisJob
from app.domains.artifact.model import Artifact
from app.domains.auth.model import User
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.highlight.action_cache import ActionSpottingCacheService
from app.domains.highlight.model import SceneTrackingBinding
from app.domains.highlight.repository import HighlightRepository
from app.domains.highlight.request_parser import HighlightRequestParser
from app.domains.highlight.schema import HighlightRequest, HighlightScope
from app.domains.highlight.service import HighlightWorkflowService
from app.domains.highlight.timeline_mapping import TrackingTimelineMapper
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.render.ffmpeg_renderer import FFmpegRenderer
from app.domains.render.template_builder import RenderTemplateBuilder
from app.domains.render.tracking_transform import TrackingTransformBuilder
from app.domains.timeline.model import TimelineEvent
from app.domains.tracking.model import TrackingJob


class HighlightSceneNormalizationTests(unittest.TestCase):
    def test_v9_events_use_the_product_clip_windows(self) -> None:
        processor = HighlightPostprocessor(default_threshold=0.0)
        windows = {
            "Goal": (15, 30, 45),
            "Shot": (8, 14, 22),
            "Penalty": (25, 40, 65),
            "Card": (15, 25, 40),
            "Corner": (10, 18, 28),
        }

        for label, (before_sec, after_sec, total_sec) in windows.items():
            with self.subTest(label=label):
                scene = processor.process(
                    [
                        {
                            "timestamp_sec": 100,
                            "label": label,
                            "confidence": 1.0,
                            # Model-provided ranges must not override product policy.
                            "start_sec": 99,
                            "end_sec": 101,
                        }
                    ],
                    match_duration_sec=200,
                )[0]
                self.assertEqual(scene["start_sec"], 100 - before_sec)
                self.assertEqual(scene["end_sec"], 100 + after_sec)
                self.assertEqual(scene["duration_sec"], total_sec)

    def test_product_window_is_only_shortened_at_video_boundary(self) -> None:
        processor = HighlightPostprocessor(default_threshold=0.0)
        start_scene = processor.process(
            [{"timestamp_sec": 5, "label": "Goal", "confidence": 1.0}],
            match_duration_sec=100,
        )[0]
        end_scene = processor.process(
            [{"timestamp_sec": 90, "label": "Goal", "confidence": 1.0}],
            match_duration_sec=100,
        )[0]

        self.assertEqual(
            (start_scene["start_sec"], start_scene["end_sec"]),
            (0.0, 35.0),
        )
        self.assertEqual(
            (end_scene["start_sec"], end_scene["end_sec"]),
            (75.0, 100.0),
        )
        self.assertTrue(
            start_scene["metadata"]["scene_provenance"]["boundary_clamped"]
        )
        self.assertTrue(
            end_scene["metadata"]["scene_provenance"]["boundary_clamped"]
        )

    def test_shot_and_goal_merge_without_losing_point_predictions(self) -> None:
        processor = HighlightPostprocessor(
            default_threshold=0.3,
            max_candidates=20,
            nms_window_sec=8,
            min_duration_sec=4,
            max_duration_sec=12,
            pre_event_sec=4,
            post_event_sec=4,
            event_windows={},
            merge_overlapping_scenes=True,
            class_priority={"goal": 100, "shot": 80},
        )
        scenes = processor.process(
            [
                {"time_sec": 2616, "label": "Shot", "confidence": 0.57},
                {"time_sec": 2617, "label": "Goal", "confidence": 0.60},
            ],
            match_duration_sec=2620,
        )
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["label"], "goal")
        self.assertEqual(scenes[0]["end_sec"], 2620)
        self.assertEqual(
            [row["label"] for row in scenes[0]["metadata"]["source_predictions"]],
            ["Shot", "Goal"],
        )
        self.assertTrue(
            scenes[0]["metadata"]["scene_provenance"]["merged"]
        )

    def test_same_label_nms_preserves_suppressed_prediction_provenance(self) -> None:
        processor = HighlightPostprocessor(
            default_threshold=0.3,
            nms_window_sec=10,
        )
        scenes = processor.process(
            [
                {"timestamp_sec": 10, "label": "Shot", "confidence": 0.9},
                {"timestamp_sec": 11, "label": "Shot", "confidence": 0.8},
            ],
            match_duration_sec=30,
        )
        self.assertEqual(len(scenes), 1)
        self.assertEqual(
            len(scenes[0]["metadata"]["source_predictions"]),
            2,
        )

    def test_overlapping_scenes_that_cannot_merge_are_split_without_duplication(self) -> None:
        processor = HighlightPostprocessor(
            default_threshold=0.3,
            nms_window_sec=1,
            max_duration_sec=8,
            pre_event_sec=4,
            post_event_sec=4,
            event_windows={},
            merge_overlapping_scenes=True,
            class_priority={"goal": 100, "foul": 50},
        )
        scenes = processor.process(
            [
                {"time_sec": 10, "label": "Goal", "confidence": 0.9},
                {"time_sec": 17, "label": "Foul", "confidence": 0.8},
            ],
            match_duration_sec=30,
        )
        self.assertEqual(len(scenes), 2)
        self.assertLessEqual(scenes[0]["end_sec"], scenes[1]["start_sec"])
        self.assertTrue(
            scenes[0]["metadata"]["scene_provenance"]["overlap_resolved"]
        )


class HighlightRequestParserTests(unittest.TestCase):
    def test_v9_penalty_keyword_is_parsed(self) -> None:
        parsed = HighlightRequestParser().parse("페널티 장면만 모아줘")
        self.assertEqual(parsed.event_labels, ["penalty"])

    def test_first_half_never_assumes_zero_to_2700(self) -> None:
        parsed = HighlightRequestParser().parse("전반전 하이라이트만 뽑아줘")
        self.assertEqual(parsed.scope.type.value, "FIRST_HALF")
        self.assertIsNone(parsed.scope.start_time_sec)
        self.assertIsNone(parsed.scope.end_time_sec)
        self.assertTrue(parsed.scope.requires_period_boundary)

    def test_period_metadata_and_relative_second_half_minute_are_structured(self) -> None:
        parsed = HighlightRequestParser().parse(
            "후반 20분부터 슈팅 하이라이트",
            match_metadata={
                "period_boundaries": {
                    "second_half": {
                        "start_time_sec": 3500,
                        "end_time_sec": 6500,
                    }
                }
            },
        )
        self.assertEqual(parsed.scope.start_time_sec, 4700)
        self.assertEqual(parsed.scope.end_time_sec, 6500)
        self.assertEqual(parsed.scope.relative_start_sec, 1200)
        self.assertEqual(parsed.event_labels, ["shot"])

    def test_current_scene_reference_and_player_focus(self) -> None:
        parsed = HighlightRequestParser().parse(
            "이 장면들에서 A선수를 줌해줘",
            current_revision_exists=True,
        )
        self.assertEqual(parsed.scope.type.value, "SELECTED_SCENES")
        self.assertEqual(parsed.focus_mode.value, "PLAYER")

    def test_period_relative_range_filters_by_resolved_source_time(self) -> None:
        events = [
            TimelineEvent(
                timeline_event_id="evt_early",
                match_id="match_1",
                event_type="shot",
                label="shot",
                half=2,
                timestamp_sec=4600,
                start_sec=4595,
                end_sec=4605,
                duration_sec=10,
            ),
            TimelineEvent(
                timeline_event_id="evt_late",
                match_id="match_1",
                event_type="shot",
                label="shot",
                half=2,
                timestamp_sec=4800,
                start_sec=4795,
                end_sec=4805,
                duration_sec=10,
            ),
        ]
        request = HighlightRequestParser().parse(
            "후반 20분부터",
            match_metadata={
                "period_boundaries": {
                    "second_half": [3500, 6500],
                }
            },
        )
        filtered = HighlightWorkflowService._filter_scope(events, request)
        self.assertEqual(
            [event.timeline_event_id for event in filtered],
            ["evt_late"],
        )


class PlayerFallbackAndTransformTests(unittest.TestCase):
    def test_player_filter_does_not_fallback_to_match_wide_events(self) -> None:
        event = TimelineEvent(
            timeline_event_id="evt_1",
            match_id="match_1",
            event_type="goal",
            label="goal",
            timestamp_sec=10,
            start_sec=8,
            end_sec=12,
            duration_sec=4,
            player_ids=["player_other"],
        )
        plan = RuleBasedClipPlanner().build_plan(
            prompt="A 선수 하이라이트",
            events=[event],
            players=[],
            selected_player_id="player_a",
        )
        self.assertEqual(plan.items, [])

    def test_source_mapping_uses_local_timestamp_and_never_crops_ambiguous(self) -> None:
        binding = SceneTrackingBinding(
            binding_id="bind_1",
            revision_id="rev_1",
            scene_id="evt_1",
            focus_subject_id="focus_1",
            source_start_time_sec=100.0,
            source_fps=25.0,
            confirmation_source="USER",
            target_presence_status="PRESENT",
            render_strategy="TARGET_CENTERED",
            status="COMPLETED",
        )
        timeline = {
            "schema_version": "kickclip.target_centric_e2e.v1",
            "pipeline_version": "1",
            "frames": [
                {
                    "frame_index": 0,
                    "time_seconds": 0.0,
                    "state": "ACTIVE",
                    "bbox_xyxy": [100, 100, 180, 300],
                    "tracking_confidence": 0.9,
                },
                {
                    "frame_index": 1,
                    "time_seconds": 0.04,
                    "state": "AMBIGUOUS",
                    "bbox_xyxy": None,
                    "tracking_confidence": 0.0,
                },
            ],
        }
        summary = TrackingTimelineMapper.summarize(binding, timeline)
        self.assertEqual(summary["crop_segments"][0]["start_time_sec"], 100.0)
        self.assertEqual(len(summary["ambiguity_segments"]), 1)
        keyframes = TrackingTimelineMapper.crop_keyframes(
            binding,
            timeline,
            source_start_sec=100,
            source_end_sec=101,
        )
        self.assertEqual(len(keyframes), 1)
        self.assertEqual(keyframes[0]["state"], "ACTIVE")

    def test_tracking_transform_is_clamped_and_renderer_requires_it(self) -> None:
        transform = TrackingTransformBuilder().build(
            keyframes=[
                {
                    "relative_time_sec": 0,
                    "state": "ACTIVE",
                    "bbox_xyxy": [1800, 800, 1910, 1070],
                },
                {
                    "relative_time_sec": 1,
                    "state": "ACTIVE_LOW_CONFIDENCE",
                    "bbox_xyxy": [1810, 790, 1915, 1060],
                },
            ],
            source_width=1920,
            source_height=1080,
            output_width=1080,
            output_height=1920,
        )
        self.assertGreaterEqual(transform["keyframes"][0]["x"], 0)
        self.assertGreaterEqual(transform["keyframes"][0]["y"], 0)
        filter_value = TrackingTransformBuilder.ffmpeg_filter(
            transform=transform,
            output_width=1080,
            output_height=1920,
        )
        self.assertIn("crop=", filter_value)
        template = RenderTemplateBuilder().build(ratio="9:16", quality="1080p")
        with self.assertRaises(ValueError):
            FFmpegRenderer._build_segment_command(
                source_video_path=__import__("pathlib").Path("source.mp4"),
                output_path=__import__("pathlib").Path("output.mp4"),
                start_sec=0,
                duration_sec=1,
                template=template,
                item_metadata={"render_strategy": "TARGET_CENTERED"},
            )


class HighlightRevisionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = self.SessionLocal()
        self.user = User(
            user_id="usr_1",
            email="highlight@example.com",
            password_hash="!",
            display_name="Highlight",
        )
        self.match = Match(
            match_id="match_1",
            owner_id=self.user.user_id,
            duration_sec=500,
            metadata_={
                "period_boundaries": {
                    "first_half": [100, 250],
                    "second_half": [300, 500],
                }
            },
        )
        self.project = Project(
            project_id="proj_1",
            match_id=self.match.match_id,
            owner_id=self.user.user_id,
            title="One workflow",
        )
        self.asset = MediaAsset(
            asset_id="asset_1",
            match_id=self.match.match_id,
            asset_type="RAW_VIDEO",
            file_path="storage/fake.mp4",
            sha256="a" * 64,
            fps=25,
            width=1920,
            height=1080,
        )
        self.job = AnalysisJob(
            analysis_job_id="job_1",
            match_id=self.match.match_id,
            media_asset_id=self.asset.asset_id,
            job_type="HIGHLIGHT_SPOTTING",
            status="COMPLETED",
            progress=100,
            options={},
        )
        self.events = [
            TimelineEvent(
                timeline_event_id="evt_first",
                match_id=self.match.match_id,
                source_job_id=self.job.analysis_job_id,
                event_type="goal",
                label="goal",
                timestamp_sec=150,
                start_sec=145,
                end_sec=155,
                duration_sec=10,
                confidence=0.9,
                highlight_score=9,
                player_ids=[],
                metadata_={"source_predictions": []},
            ),
            TimelineEvent(
                timeline_event_id="evt_second",
                match_id=self.match.match_id,
                source_job_id=self.job.analysis_job_id,
                event_type="shot",
                label="shot",
                timestamp_sec=350,
                start_sec=345,
                end_sec=355,
                duration_sec=10,
                confidence=0.8,
                highlight_score=8,
                player_ids=[],
                metadata_={"source_predictions": []},
            ),
        ]
        self.db.add_all(
            [
                self.user,
                self.match,
                self.project,
                self.asset,
                self.job,
                *self.events,
            ]
        )
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_action_cache_is_shared_for_same_asset_model_and_policy(self) -> None:
        # The existing completed job is not cache-keyed, so the first call
        # creates the reusable run and the second returns the same row.
        cache = ActionSpottingCacheService(self.db)
        first, reused_first = cache.get_or_create(
            match_id=self.match.match_id,
            request_options={"dummy_mode": True},
        )
        second, reused_second = cache.get_or_create(
            match_id=self.match.match_id,
            request_options={"dummy_mode": True},
        )
        self.assertFalse(reused_first)
        self.assertTrue(reused_second)
        self.assertEqual(first.analysis_job_id, second.analysis_job_id)

    def test_revisions_preserve_parent_and_player_focus_uses_selected_set(self) -> None:
        repository = HighlightRepository(self.db)
        revision = repository.create_revision(
            project_id=self.project.project_id,
            revision_number=1,
            action_spotting_job_id=self.job.analysis_job_id,
            user_request="전반전 하이라이트",
            structured_request=HighlightRequest(
                scope=HighlightScope(
                    type="FIRST_HALF",
                    start_time_sec=100,
                    end_time_sec=250,
                    boundary_source="MATCH_METADATA",
                )
            ).model_dump(mode="json"),
            selected_scene_ids=[],
            scene_selection=[],
            focus_mode="NONE",
            status="ACTION_SPOTTING_RUNNING",
            pending_action="SELECT_SCENES",
        )
        self.db.commit()
        service = HighlightWorkflowService(self.db)
        revision = service.reconcile_revision(revision)
        self.assertEqual(revision.selected_scene_ids, ["evt_first"])

        revision2 = service.select_scenes(
            project=self.project,
            scene_ids=["evt_first"],
            selection_source="USER",
            render_strategies={},
        )
        self.assertEqual(revision2.parent_revision_id, revision.revision_id)
        revision3 = service.start_player_focus(
            project=self.project,
            user_request="이 장면들에서 A선수를 줌해줘",
            source_revision_id=revision2.revision_id,
        )
        self.assertEqual(revision3.parent_revision_id, revision2.revision_id)
        self.assertEqual(revision3.selected_scene_ids, ["evt_first"])
        self.assertEqual(revision3.focus_mode, "PLAYER")

        seen_scene_ids: list[str] = []

        def fake_discover(*, revision, source_asset, scenes):
            seen_scene_ids.extend(scene.timeline_event_id for scene in scenes)
            return []

        with patch(
            "app.domains.highlight.service.PlayerCandidateDiscoveryService.discover",
            side_effect=fake_discover,
        ):
            service.run_candidate_discovery(revision3.revision_id)
        self.assertEqual(seen_scene_ids, ["evt_first"])

    def test_candidate_discovery_failure_is_not_rewritten_as_selection_required(
        self,
    ) -> None:
        repository = HighlightRepository(self.db)
        revision = repository.create_revision(
            project_id=self.project.project_id,
            revision_number=1,
            action_spotting_job_id=self.job.analysis_job_id,
            user_request="선수 후보를 찾아줘",
            structured_request=HighlightRequest(
                focus_mode="PLAYER",
            ).model_dump(mode="json"),
            selected_scene_ids=["evt_first"],
            scene_selection=[
                {
                    "scene_id": "evt_first",
                    "state": "USER_SELECTED",
                    "render_strategy": "TARGET_CENTERED",
                }
            ],
            focus_mode="PLAYER",
            status="PLAYER_DISCOVERY_RUNNING",
            pending_action="SELECT_PLAYER",
        )
        self.db.commit()
        service = HighlightWorkflowService(self.db)

        with patch(
            "app.domains.highlight.service.PlayerCandidateDiscoveryService.discover",
            side_effect=RuntimeError("checkpoint missing"),
        ):
            service.run_candidate_discovery(revision.revision_id)

        failed = repository.get_revision(revision.revision_id)
        self.assertIsNotNone(failed)
        failed = service.reconcile_revision(failed)
        self.assertEqual(failed.status, "FAILED")
        self.assertEqual(failed.error_message, "checkpoint missing")
        with self.assertRaisesRegex(ValueError, "checkpoint missing"):
            service.candidates(self.project.project_id)

    def test_selected_player_tracking_is_persisted_in_plan_and_reused_by_render(
        self,
    ) -> None:
        repository = HighlightRepository(self.db)
        revision = repository.create_revision(
            project_id=self.project.project_id,
            revision_number=1,
            action_spotting_job_id=self.job.analysis_job_id,
            user_request="이 장면에서 선택한 선수를 중심으로 보여줘",
            structured_request=HighlightRequest(
                focus_mode="PLAYER",
                aspect_ratio="9:16",
            ).model_dump(mode="json"),
            selected_scene_ids=["evt_first"],
            scene_selection=[
                {
                    "scene_id": "evt_first",
                    "state": "USER_SELECTED",
                    "render_strategy": "TARGET_CENTERED",
                }
            ],
            focus_mode="PLAYER",
            status="PLAYER_SELECTION_REQUIRED",
            pending_action="SELECT_PLAYER",
        )
        thumbnail = Artifact(
            artifact_id="art_candidate",
            match_id=self.match.match_id,
            project_id=self.project.project_id,
            artifact_type="PLAYER_CANDIDATE_THUMBNAIL",
            file_path="storage/candidate.jpg",
            mime_type="image/jpeg",
        )
        self.db.add(thumbnail)
        candidate = repository.create_candidate(
            candidate_id="candidate_1",
            revision_id=revision.revision_id,
            scene_id="evt_first",
            anchor_time_sec=2,
            anchor_source_time_sec=147,
            anchor_frame_index=3675,
            bbox_xyxy=[100, 100, 260, 500],
            thumbnail_artifact_id=thumbnail.artifact_id,
            track_length_frames=5,
            trackability_score=0.93,
            status="AVAILABLE",
            metadata_={
                "display_label": "선수 후보 1",
                "confidence": 0.94,
                "class_name": "player",
            },
        )
        self.db.commit()
        service = HighlightWorkflowService(self.db)

        _, candidates = service.candidates(
            self.project.project_id,
            user_id=self.user.user_id,
        )
        self.assertEqual(candidates[0].player_id, candidate.candidate_id)
        self.assertEqual(candidates[0].track_id, candidate.candidate_id)
        self.assertEqual(candidates[0].confidence, 0.94)
        self.assertIn("?token=", candidates[0].representative_image_url)

        _, subject, binding = service.select_focus_subject(
            project=self.project,
            revision_id=revision.revision_id,
            display_name=None,
            anchor_scene_id=None,
            candidate_id=candidate.candidate_id,
        )
        tracking_job = TrackingJob(
            tracking_job_id="trk_completed",
            owner_id=self.user.user_id,
            match_id=self.match.match_id,
            project_id=self.project.project_id,
            media_asset_id=self.asset.asset_id,
            test_name="tracking_highlight_integration",
            initial_bbox=candidate.bbox_xyxy,
            bbox_format="xyxy_pixels",
            device="cpu",
            reacquisition_mode="assisted",
            status="COMPLETED",
            current_stage="FINALIZE",
            output_directory="/tmp/tracking_highlight_integration",
            pipeline_state_path="/tmp/tracking_highlight_integration/state.json",
        )
        self.db.add(tracking_job)
        self.db.flush()
        binding.tracking_job_id = tracking_job.tracking_job_id
        binding.source_start_time_sec = 145
        binding.source_end_time_sec = 155
        binding.source_fps = 25
        binding.status = "TRACKING_RUNNING"
        self.db.commit()

        timeline = {
            "schema_version": "kickclip.target_centric_e2e.v1",
            "pipeline_version": "1",
            "video": {
                "width": 1920,
                "height": 1080,
                "fps": 25,
                "frame_count": 250,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "time_seconds": 0,
                    "state": "ACTIVE",
                    "bbox_xyxy": [100, 100, 260, 500],
                    "tracking_confidence": 0.95,
                },
                {
                    "frame_index": 100,
                    "time_seconds": 4,
                    "state": "ACTIVE",
                    "bbox_xyxy": [400, 120, 560, 520],
                    "tracking_confidence": 0.92,
                },
            ],
        }
        with patch(
            "app.domains.highlight.service.TrackingTimelineService.read",
            return_value=timeline,
        ):
            status = service.status(self.project.project_id)
            self.assertEqual(status.scene_tracking[0].progress, 100)
            self.assertEqual(status.scene_tracking[0].status, "COMPLETED")
            response = service.create_clip_plan(
                project=self.project,
                revision_id=revision.revision_id,
                allow_absent_full_frame=False,
            )

        plan = ClipPlanRepository(self.db).get_by_id(response.clip_plan_id)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.options["focus_subject_id"], subject.focus_subject_id)
        self.assertEqual(
            plan.options["render_provenance"]["tracking_job_ids"],
            [tracking_job.tracking_job_id],
        )
        metadata = plan.items[0].metadata_
        self.assertEqual(metadata["render_strategy"], "TARGET_CENTERED")
        self.assertEqual(metadata["tracking_job_id"], tracking_job.tracking_job_id)
        self.assertIn("tracking_transform", metadata)

        first_revision, first_render = service.create_render(
            project=self.project,
            revision_id=revision.revision_id,
            options={},
        )
        _, second_render = service.create_render(
            project=self.project,
            revision_id=revision.revision_id,
            options={},
        )
        self.assertEqual(first_render.render_job_id, second_render.render_job_id)
        self.assertTrue(second_render.reused)
        self.assertEqual(first_revision.render_job_id, first_render.render_job_id)

        command = FFmpegRenderer._build_segment_command(
            source_video_path=__import__("pathlib").Path("source.mp4"),
            output_path=__import__("pathlib").Path("output.mp4"),
            start_sec=plan.items[0].start_sec,
            duration_sec=plan.items[0].duration_sec,
            template=RenderTemplateBuilder().build(
                ratio="9:16",
                quality="1080p",
            ),
            item_metadata=metadata,
        )
        self.assertIn("crop=", command[command.index("-vf") + 1])


if __name__ == "__main__":
    unittest.main()
