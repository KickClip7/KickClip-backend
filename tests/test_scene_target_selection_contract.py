from __future__ import annotations

import hashlib
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking import (
    EventCandidateRankingService,
)
from app.domains.highlight.model import (
    EarlierAnchorProposal,
    HighlightRevision,
    ScenePlayerCandidate,
    SceneTargetSelection,
)
from app.domains.highlight.scene_target_selection import (
    SceneTargetSelectionService,
)
from app.domains.highlight.scene_ai_task import (
    DISCOVERY,
    SceneAITaskService,
)
from app.domains.highlight.runtime_contract import (
    prepare_selection_workspace,
    redact_runtime_paths,
    validated_runtime_file,
)
from app.domains.highlight.schema import (
    EventCandidateLabelRequest,
    EventCandidateRankingRequest,
)
from app.domains.match.model import Match
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent
from app.domains.tracking.verifier import (
    SceneTargetTrackingInstallationVerifier,
    TrackingInstallationStatus,
)
from app.domains.tracking.errors import TrackingValidationError
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.process_runner import TrackingProcessRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class SelectionArtifactIsolationTests(unittest.TestCase):
    def test_later_revision_cannot_overwrite_earlier_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            discovery = Path(temporary).resolve()
            (discovery / "candidates").mkdir()
            (discovery / "candidates" / "candidate.json").write_text(
                '{"candidate_id":"candidate_a"}',
                encoding="utf-8",
            )

            first = prepare_selection_workspace(discovery, revision=1)
            (first.staging_root / "target_selection.json").write_text(
                '{"revision":1}',
                encoding="utf-8",
            )
            first_root = first.finalize("tsel_first")

            second = prepare_selection_workspace(discovery, revision=2)
            (second.staging_root / "target_selection.json").write_text(
                '{"revision":2}',
                encoding="utf-8",
            )
            second_root = second.finalize("tsel_second")

            self.assertEqual(
                json.loads(
                    (first_root / "target_selection.json").read_text(
                        encoding="utf-8"
                    )
                )["revision"],
                1,
            )
            self.assertEqual(
                json.loads(
                    (second_root / "target_selection.json").read_text(
                        encoding="utf-8"
                    )
                )["revision"],
                2,
            )
            self.assertNotEqual(first_root, second_root)

    def test_runtime_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root.parent / "outside-selection.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                with self.assertRaises(ValueError):
                    validated_runtime_file(
                        root,
                        "../outside-selection.json",
                        allowed_suffixes={".json"},
                    )
            finally:
                outside.unlink(missing_ok=True)

    def test_runtime_paths_are_redacted_to_artifact_ids(self) -> None:
        result = redact_runtime_paths(
            {
                "crop_artifact": "references/crop.jpg",
                "internal_path": "C:/runtime/private/file.json",
            },
            artifact_ids_by_path={
                "references/crop.jpg": "art_crop",
            },
        )
        self.assertEqual(
            result["crop_artifact"]["artifact_id"],
            "art_crop",
        )
        self.assertNotIn("internal_path", result)


class _GenericVerified:
    def check(self):
        return TrackingInstallationStatus(
            enabled=True,
            available=True,
            checked_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
            code="TRACKING_AVAILABLE",
            message="ok",
        )


class SceneR3VerifierTests(unittest.TestCase):
    def test_r3_verifier_checks_hashes_and_strict_smoke_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            r3_script = root / "run_r3.py"
            r3_script.write_text("pass\n", encoding="utf-8")
            selection_manifest = root / "selection-manifest.json"
            r2_manifest = root / "r2-manifest.json"
            r3_manifest = root / "r3-manifest.json"
            for path in (selection_manifest, r2_manifest, r3_manifest):
                path.write_text('{"frozen":true}', encoding="utf-8")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            settings = Settings.model_construct(
                TRACKING_ENABLED=True,
                TRACKING_PROJECT_ROOT=str(root),
                TRACKING_PYTHON_EXECUTABLE=sys.executable,
                TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH=str(r3_script),
                TRACKING_SCENE_SELECTION_VERIFY_SCRIPT_PATH=str(
                    FIXTURES / "fake_r3_scene_selection_verifier.py"
                ),
                SCENE_TARGET_SELECTION_PROJECT_ROOT=str(root),
                SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE=sys.executable,
                SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH=str(
                    FIXTURES / "fake_scene_selection_verifier.py"
                ),
                SCENE_TARGET_SELECTION_MANIFEST_PATH=str(selection_manifest),
                SCENE_TARGET_SELECTION_MANIFEST_SHA256=digest(
                    selection_manifest
                ),
                TRACKING_R2_MANIFEST_PATH=str(r2_manifest),
                TRACKING_R2_MANIFEST_SHA256=digest(r2_manifest),
                TRACKING_R3_MANIFEST_PATH=str(r3_manifest),
                TRACKING_R3_MANIFEST_SHA256=digest(r3_manifest),
                TRACKING_VERIFY_TIMEOUT_SECONDS=10,
            )
            status = SceneTargetTrackingInstallationVerifier(
                settings,
                generic=_GenericVerified(),
            ).check()
            self.assertTrue(status.available)
            self.assertTrue(
                status.components["FULL_TARGET_SELECTION_E2E_VERIFIED"]
            )

            settings.TRACKING_R3_MANIFEST_SHA256 = "0" * 64
            failed = SceneTargetTrackingInstallationVerifier(
                settings,
                generic=_GenericVerified(),
            ).check()
            self.assertFalse(failed.available)
            self.assertEqual(
                failed.code,
                "SCENE_TARGET_MANIFEST_HASH_MISMATCH",
            )


class EventCandidateRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "storage"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()
        self.engine.dispose()

    def test_shadow_shortlist_and_human_evaluation_are_separate(self) -> None:
        db = self.Session()
        try:
            user = User(
                user_id="usr_rank",
                email="rank@example.com",
                password_hash="hash",
                display_name="Ranker",
            )
            match = Match(match_id="match_rank", owner_id=user.user_id)
            project = Project(
                project_id="proj_rank",
                match_id=match.match_id,
                owner_id=user.user_id,
                title="Ranking",
            )
            scene = TimelineEvent(
                timeline_event_id="evt_goal",
                match_id=match.match_id,
                event_type="HIGHLIGHT",
                label="Goal",
                timestamp_sec=12.4,
                start_sec=2.4,
                end_sec=28.4,
                duration_sec=26.0,
                confidence=0.81,
            )
            discovery_root = Path(self.temporary.name).resolve()
            manifest_sha = "a" * 64
            boundaries_sha = "b" * 64
            revision = HighlightRevision(
                revision_id="hrev_rank",
                project_id=project.project_id,
                revision_number=1,
                user_request="goal actor",
                selected_scene_ids=[scene.timeline_event_id],
                scene_selection=[],
                options={
                    "scene_target_selection": {
                        "discovery_id": "discovery_rank",
                        "artifact_root": discovery_root.relative_to(
                            PROJECT_ROOT
                        ).as_posix(),
                        "scene_candidate_manifest_sha256": manifest_sha,
                        "discovery_inputs": {
                            "shot_boundaries_sha256": boundaries_sha,
                        },
                    }
                },
            )
            db.add_all([user, match, project, scene, revision])
            for index, offset in enumerate((-0.2, 3.5, 8.0), start=1):
                db.add(
                    ScenePlayerCandidate(
                        candidate_id=f"candidate_{index}",
                        revision_id=revision.revision_id,
                        scene_id=scene.timeline_event_id,
                        anchor_time_sec=10.0 + offset,
                        anchor_source_time_sec=10.0 + offset,
                        anchor_frame_index=index,
                        bbox_xyxy=[0, 0, 10, 20],
                        track_length_frames=10 - index,
                        trackability_score=0.8 - index * 0.1,
                        metadata_={
                            "discovery_id": "discovery_rank",
                            "quality": {
                                "bbox_area_ratio": 0.2 / index,
                                "sharpness_score": 0.9 / index,
                                "center_proximity": 0.8 / index,
                            },
                            "representative_observation": {
                                "time_sec": 10.0 + offset,
                            },
                            "artifact_ids": {},
                        },
                    )
                )
            db.commit()

            service = EventCandidateRankingService(db)
            ranking = service.rank(
                project=project,
                revision_id=revision.revision_id,
                payload=EventCandidateRankingRequest(
                    event_id=scene.timeline_event_id,
                    event_label="Goal",
                    event_time_sec=12.4,
                    event_confidence=0.81,
                    scene_id=scene.timeline_event_id,
                    scene_start_sec=2.4,
                    scene_end_sec=28.4,
                    scene_candidate_manifest_sha256=manifest_sha,
                    shot_boundaries_sha256=boundaries_sha,
                    shortlist_size=3,
                ),
                user=user,
            )
            result = service.read(ranking, user=user)
            self.assertEqual(result.status, "PROVISIONAL_SHADOW_ONLY")
            self.assertFalse(result.automatic_target_confirmation)
            self.assertEqual(len(result.shortlist), 3)
            self.assertEqual(len(result.all_candidates), 3)
            self.assertIsNone(
                service.evaluate(ranking, user=user).primary_actor_recall_at_3
            )

            actor_id = result.all_candidates[1].candidate_id
            service.label(
                ranking,
                user=user,
                payload=EventCandidateLabelRequest(
                    candidate_id=actor_id,
                    role="PRIMARY_EVENT_ACTOR",
                ),
            )
            measured = service.evaluate(ranking, user=user)
            self.assertEqual(measured.status, "MEASURED")
            self.assertEqual(
                measured.primary_actor_recall_at_3,
                1.0,
            )
            self.assertEqual(
                measured.primary_actor_recall_at_5,
                1.0,
            )
        finally:
            db.close()


class EarlierProposalContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def _selection(self, db: Session):
        user = User(
            user_id="usr_proposal",
            email="proposal@example.com",
            password_hash="hash",
            display_name="Proposal",
        )
        match = Match(match_id="match_proposal", owner_id=user.user_id)
        project = Project(
            project_id="proj_proposal",
            match_id=match.match_id,
            owner_id=user.user_id,
            title="Proposal",
        )
        scene = TimelineEvent(
            timeline_event_id="evt_proposal",
            match_id=match.match_id,
            event_type="HIGHLIGHT",
            label="Goal",
            timestamp_sec=10,
            start_sec=0,
            end_sec=20,
            duration_sec=20,
        )
        revision = HighlightRevision(
            revision_id="hrev_proposal",
            project_id=project.project_id,
            revision_number=1,
            user_request="proposal",
            selected_scene_ids=[scene.timeline_event_id],
            scene_selection=[],
        )
        candidate = ScenePlayerCandidate(
            candidate_id="candidate_selected",
            revision_id=revision.revision_id,
            scene_id=scene.timeline_event_id,
            anchor_time_sec=1,
            anchor_source_time_sec=1,
            anchor_frame_index=1,
            bbox_xyxy=[0, 0, 1, 1],
            trackability_score=0.8,
        )
        selection = SceneTargetSelection(
            selection_id="tsel_proposal",
            owner_id=user.user_id,
            match_id=match.match_id,
            project_id=project.project_id,
            revision_id=revision.revision_id,
            scene_id=scene.timeline_event_id,
            selection_revision=1,
            selected_candidate_id=candidate.candidate_id,
            artifact_root="storage",
            selection_artifact_root="storage",
            target_selection_path="storage/target_selection.json",
            target_selection_sha256="1" * 64,
            target_reference_set_path="storage/target_reference_set.json",
            target_reference_set_sha256="2" * 64,
            earlier_proposals_path="storage/earlier_candidate_proposals.json",
            earlier_proposals_sha256="3" * 64,
            selection_artifact={},
            reference_set_artifact={},
            earlier_proposals_artifact={
                "state": "WAITING_EARLIER_ANCHOR_CONFIRMATION",
                "proposals": [{"candidate_id": "earlier_1"}],
            },
            earlier_decision_artifact={},
            candidate_cache_key="c" * 64,
            metadata_={
                "discovery_id": "discovery_proposal",
            },
        )
        proposal = EarlierAnchorProposal(
            selection_id=selection.selection_id,
            candidate_id="earlier_1",
            source_revision_id=revision.revision_id,
            source_discovery_id="discovery_proposal",
            retrieval_rank=6,
            decision_state="PENDING_USER_CONFIRMATION",
        )
        db.add_all(
            [
                user,
                match,
                project,
                scene,
                revision,
                candidate,
                selection,
                proposal,
            ]
        )
        db.commit()
        return user, selection

    def test_discover_earlier_reuses_existing_rows(self) -> None:
        db = self.Session()
        try:
            user, selection = self._selection(db)
            settings = Settings.model_construct(
                SCENE_TARGET_SELECTION_MAX_CONFIRMABLE_EARLIER_RANK=5,
                SCENE_TARGET_SELECTION_PROJECT_ROOT="",
                TRACKING_PROJECT_ROOT="",
            )
            service = SceneTargetSelectionService(db, settings=settings)
            with patch.object(service, "_run") as runtime:
                returned = service.discover_earlier(
                    selection=selection,
                    user=user,
                )
            self.assertIs(returned, selection)
            runtime.assert_not_called()
        finally:
            db.close()

    def test_decision_rejects_candidate_outside_confirmable_rank(self) -> None:
        db = self.Session()
        try:
            user, selection = self._selection(db)
            settings = Settings.model_construct(
                SCENE_TARGET_SELECTION_MAX_CONFIRMABLE_EARLIER_RANK=5,
                SCENE_TARGET_SELECTION_PROJECT_ROOT="",
                TRACKING_PROJECT_ROOT="",
            )
            service = SceneTargetSelectionService(db, settings=settings)
            with patch.object(service, "_run") as runtime:
                with self.assertRaisesRegex(ValueError, "rank exceeds"):
                    service.decide_earlier(
                        selection=selection,
                        user=user,
                        decision="candidate",
                        candidate_id="earlier_1",
                    )
            runtime.assert_not_called()
        finally:
            db.close()


class R3DispatchContractTests(unittest.TestCase):
    def test_dispatch_revalidates_selection_root_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "runs"
            output.mkdir()
            r3 = root / "run_r3.py"
            r3.write_text("pass\n", encoding="utf-8")
            selection_root = root / "selection"
            selection_root.mkdir()
            paths = {}
            for name in (
                "tracking_launch_manifest.json",
                "target_selection.json",
                "target_reference_set.json",
                "earlier_anchor_decision.json",
            ):
                path = selection_root / name
                path.write_text("{}", encoding="utf-8")
                paths[name] = path
            boundaries = root / "boundaries.json"
            boundaries.write_text("{}", encoding="utf-8")
            context = {
                "selection_artifact_root": str(selection_root),
                "tracking_launch_manifest_path": str(
                    paths["tracking_launch_manifest.json"]
                ),
                "shot_boundaries_path": str(boundaries),
                "target_selection_path": str(paths["target_selection.json"]),
                "target_selection_sha256": hashlib.sha256(
                    paths["target_selection.json"].read_bytes()
                ).hexdigest(),
                "target_reference_set_path": str(
                    paths["target_reference_set.json"]
                ),
                "target_reference_set_sha256": hashlib.sha256(
                    paths["target_reference_set.json"].read_bytes()
                ).hexdigest(),
                "earlier_anchor_decision_path": str(
                    paths["earlier_anchor_decision.json"]
                ),
                "earlier_anchor_decision_sha256": hashlib.sha256(
                    paths["earlier_anchor_decision.json"].read_bytes()
                ).hexdigest(),
            }
            settings = Settings.model_construct(
                TRACKING_PROJECT_ROOT=str(root),
                TRACKING_PYTHON_EXECUTABLE=sys.executable,
                TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH=str(r3),
                TRACKING_OUTPUT_ROOT=str(output),
                TRACKING_PREVIEW_ENABLED=True,
                TRACKING_DEVICE="cpu",
            )
            job = TrackingJob(
                tracking_job_id="trk_r3",
                owner_id="usr",
                match_id="match",
                media_asset_id="asset",
                test_name="tracking_r3",
                initial_bbox=[0, 0, 10, 10],
                bbox_format="xyxy_pixels",
                device="cpu",
                reacquisition_mode="assisted",
                status="QUEUED",
                output_directory=str(output / "tracking_r3"),
                pipeline_state_path=str(
                    output / "tracking_r3" / "pipeline_state.json"
                ),
                runtime_metadata={"scene_target_selection": context},
            )
            runner = TrackingProcessRunner(settings)
            command = runner.build_new_command(
                job,
                video_path=root / "video.mp4",
            )
            self.assertEqual(Path(command[1]), r3)

            paths["target_selection.json"].write_text(
                '{"tampered":true}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TrackingValidationError,
                "hash mismatch",
            ):
                runner.build_new_command(
                    job,
                    video_path=root / "video.mp4",
                )


class SceneAITaskDurabilityTests(unittest.TestCase):
    def test_enqueue_is_idempotent_and_failed_task_is_retryable(self) -> None:
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(
            bind=engine,
            class_=Session,
            expire_on_commit=False,
        )
        db = session_factory()
        try:
            user = User(
                user_id="usr_task",
                email="task@example.com",
                password_hash="hash",
                display_name="Task",
            )
            match = Match(match_id="match_task", owner_id=user.user_id)
            project = Project(
                project_id="proj_task",
                match_id=match.match_id,
                owner_id=user.user_id,
                title="Task",
            )
            db.add_all([user, match, project])
            db.commit()
            service = SceneAITaskService(db)
            payload = {
                "revision_id": "hrev_task",
                "scene_id": "evt_task",
            }
            first, first_reused = service.enqueue(
                user=user,
                project=project,
                task_type=DISCOVERY,
                payload=payload,
            )
            second, second_reused = service.enqueue(
                user=user,
                project=project,
                task_type=DISCOVERY,
                payload=payload,
            )
            self.assertFalse(first_reused)
            self.assertTrue(second_reused)
            self.assertEqual(first.task_id, second.task_id)

            first.status = "FAILED"
            first.attempt_count = 1
            db.commit()
            retried = service.retry(first)
            self.assertEqual(retried.status, "QUEUED")
            self.assertIsNone(retried.error_message)
        finally:
            db.close()
            engine.dispose()


class SceneMigrationChainTests(unittest.TestCase):
    def test_required_migrations_form_one_chain_through_scene_ai_head(self) -> None:
        versions = PROJECT_ROOT / "alembic" / "versions"
        expected = [
            ("20260728_0012", "20260727_0011"),
            ("20260730_0013", "20260728_0012"),
            ("20260730_0014", "20260730_0013"),
            ("20260730_0015", "20260730_0014"),
            ("20260730_0016", "20260730_0015"),
        ]
        files = {path.name: path for path in versions.glob("*.py")}
        for revision, parent in expected:
            path = next(
                path
                for name, path in files.items()
                if name.startswith(revision)
            )
            tree = ast.parse(path.read_text(encoding="utf-8"))
            values = {}
            for node in tree.body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id in {"revision", "down_revision"}
                ):
                    values[node.target.id] = ast.literal_eval(node.value)
            self.assertEqual(values["revision"], revision)
            self.assertEqual(values["down_revision"], parent)
