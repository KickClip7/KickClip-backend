from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db import models as _models  # noqa: F401 - register relationship targets
from app.db.base import Base
from app.domains.auth.model import User
from app.domains.match.model import Match
from app.domains.media.model import MediaAsset
from app.domains.tracking.artifacts import TrackingArtifactService
from app.domains.tracking.errors import (
    TrackingArtifactNotFoundError,
    TrackingContractError,
    TrackingProcessTimeoutError,
    TrackingValidationError,
)
from app.domains.tracking.executor import classify_runtime_failure
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.process_runner import TrackingProcessRunner
from app.domains.tracking.schema import TrackingAmbiguityConfirmationRequest
from app.domains.tracking.service import TrackingJobService
from app.domains.tracking.state_mapper import (
    map_pipeline_state,
    read_pipeline_state,
)
from app.domains.tracking.status import TrackingBackendStatus
from app.domains.tracking.timeline import TrackingTimelineService
from app.domains.tracking.validation import (
    build_action_key,
    generate_tracking_test_name,
    pending_review_candidate_ids,
    validate_bbox_xyxy,
)
from app.domains.tracking.verifier import TrackingInstallationVerifier


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_settings(root: Path, *, timeout: float = 30) -> Settings:
    return Settings.model_construct(
        TRACKING_ENABLED=True,
        TRACKING_PROJECT_ROOT=str(root),
        TRACKING_E2E_SCRIPT_PATH=str(FIXTURES / "fake_tracking_runner.py"),
        TRACKING_VERIFY_SCRIPT_PATH=str(FIXTURES / "fake_tracking_verifier.py"),
        TRACKING_PYTHON_EXECUTABLE=sys.executable,
        TRACKING_OUTPUT_ROOT=str(root / "runs" / "target_centric_tracking_e2e_v1"),
        TRACKING_DEVICE="cpu",
        TRACKING_REACQUISITION_MODE="assisted",
        TRACKING_MAX_CONCURRENT_JOBS=1,
        TRACKING_PROCESS_TIMEOUT_SECONDS=timeout,
        TRACKING_VERIFY_TIMEOUT_SECONDS=10,
        TRACKING_PREVIEW_ENABLED=True,
    )


def fixture_job(settings: Settings, test_name: str) -> TrackingJob:
    root = Path(settings.TRACKING_OUTPUT_ROOT)
    return TrackingJob(
        tracking_job_id=f"trk_{test_name}",
        owner_id="usr_fixture",
        match_id="match_fixture",
        media_asset_id="asset_fixture",
        test_name=test_name,
        initial_bbox=[10, 10, 30, 50],
        bbox_format="xyxy_pixels",
        device="cpu",
        reacquisition_mode="assisted",
        status="QUEUED",
        output_directory=str(root / test_name),
        pipeline_state_path=str(root / test_name / "pipeline_state.json"),
        queued_action={"kind": "new"},
        artifact_index={},
        runtime_metadata={},
    )


class TrackingValidationTests(unittest.TestCase):
    def test_bbox_validation(self) -> None:
        self.assertEqual(
            validate_bbox_xyxy([1, 2, 10, 20], width=100, height=50),
            [1.0, 2.0, 10.0, 20.0],
        )
        with self.assertRaises(TrackingValidationError):
            validate_bbox_xyxy([10, 2, 1, 20], width=100, height=50)
        with self.assertRaises(TrackingValidationError):
            validate_bbox_xyxy([1, 2, 101, 20], width=100, height=50)

    def test_server_test_name_is_runner_safe(self) -> None:
        self.assertEqual(
            generate_tracking_test_name("trk_ab-12_cd"),
            "tracking_trkab12cd",
        )

    def test_candidate_validation_uses_presented_review_candidates(self) -> None:
        state = {
            "pending_action": {
                "type": "CROSS_SHOT_CONFIRMATION",
                "ambiguity_id": "ambiguity_0001",
            },
            "ambiguities": [
                {
                    "ambiguity_id": "ambiguity_0001",
                    "status": "PENDING",
                    "review_candidates": [
                        {"candidate_id": "shot_0001_track_0001"}
                    ],
                }
            ],
        }
        self.assertEqual(
            pending_review_candidate_ids(state, "ambiguity_0001"),
            {"shot_0001_track_0001"},
        )
        self.assertEqual(
            pending_review_candidate_ids(state, "ambiguity_9999"),
            set(),
        )

    def test_action_key_is_stable_for_idempotency(self) -> None:
        self.assertEqual(
            build_action_key("ambiguity", "a1", "candidate", "c1"),
            build_action_key("ambiguity", "a1", "candidate", "c1"),
        )

    def test_installation_verifier_disables_missing_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            settings.TRACKING_E2E_SCRIPT_PATH = str(root / "missing_runner.py")
            settings.TRACKING_VERIFY_SCRIPT_PATH = str(root / "missing_verifier.py")
            result = TrackingInstallationVerifier(settings).check()
            self.assertFalse(result.available)
            self.assertEqual(result.code, "TRACKING_RUNTIME_MISSING")


class TrackingStateMappingTests(unittest.TestCase):
    def test_waiting_mappings_ignore_expected_nonzero_exit(self) -> None:
        memory = {
            "status": "NEEDS_CONFIRMATION",
            "decision": "PAUSE",
            "pending_action": {
                "type": "MEMORY_REVIEW",
                "review_stage": "MEMORY",
            },
        }
        mapped = map_pipeline_state(memory, process_return_code=3)
        self.assertEqual(
            mapped.backend_status,
            TrackingBackendStatus.WAITING_MEMORY_REVIEW,
        )

    def test_complete_with_safe_block_is_not_failed(self) -> None:
        mapped = map_pipeline_state(
            {
                "status": "COMPLETE_WITH_SAFE_BLOCK",
                "decision": "SAFE_BLOCK",
                "pending_action": None,
            },
            process_return_code=0,
        )
        self.assertEqual(
            mapped.backend_status,
            TrackingBackendStatus.COMPLETED_SAFE_BLOCK,
        )

    def test_missing_state_and_fatal_exit_fails(self) -> None:
        mapped = map_pipeline_state(None, process_return_code=2)
        self.assertEqual(mapped.backend_status, TrackingBackendStatus.FAILED)

    def test_reconcile_waiting_state_after_restart(self) -> None:
        mapped = map_pipeline_state(
            {
                "status": "NEEDS_CONFIRMATION",
                "pending_action": {
                    "type": "PHASE1_INTERNAL_REVIEW",
                    "review_stage": "STAGE2D",
                },
            },
            process_return_code=3,
            process_ended=False,
        )
        self.assertEqual(
            mapped.backend_status,
            TrackingBackendStatus.WAITING_SEGMENT_REVIEW,
        )

    def test_fatal_diagnostics_distinguish_cuda_and_model_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            stdout.write_text("", encoding="utf-8")
            stderr.write_text(
                "RuntimeError: CUDA requested but unavailable",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_runtime_failure(stdout, stderr)[0],
                "CUDA_INITIALIZATION_FAILED",
            )
            stderr.write_text(
                "Frozen checkpoint is missing or changed",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_runtime_failure(stdout, stderr)[0],
                "MODEL_INITIALIZATION_FAILED",
            )


class TrackingProcessIntegrationTests(unittest.TestCase):
    def test_new_command_uses_explicit_external_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            video = root / "video.mp4"
            job = fixture_job(settings, "tracking_command")
            command = TrackingProcessRunner(settings).build_new_command(
                job,
                video_path=video,
            )
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1], str(FIXTURES / "fake_tracking_runner.py"))
            self.assertEqual(
                command[command.index("--project-root") + 1],
                str(root),
            )
            self.assertEqual(
                command[command.index("--output-root") + 1],
                settings.TRACKING_OUTPUT_ROOT,
            )
            self.assertEqual(
                command[command.index("--reacquisition-mode") + 1],
                "assisted",
            )
            self.assertNotIn("--resume", command)

    def test_memory_to_candidate_to_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            job = fixture_job(settings, "tracking_flow")
            runner = TrackingProcessRunner(settings)

            first = runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            state = read_pipeline_state(Path(job.pipeline_state_path))
            self.assertEqual(first.return_code, 3)
            self.assertEqual(
                map_pipeline_state(
                    state,
                    process_return_code=first.return_code,
                ).backend_status,
                TrackingBackendStatus.WAITING_MEMORY_REVIEW,
            )

            artifacts = TrackingArtifactService(settings)
            index = artifacts.collect(job, state)
            self.assertTrue(index["memory_contact_sheet"]["exists"])
            copied = (
                Path(job.output_directory)
                / index["memory_contact_sheet"]["relative_path"]
            )
            self.assertTrue(copied.is_relative_to(Path(job.output_directory)))

            job.queued_action = {
                "kind": "review",
                "stage": "MEMORY",
                "decision": "approve",
                "reviewer": "usr_fixture",
            }
            second = runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            state = read_pipeline_state(Path(job.pipeline_state_path))
            self.assertEqual(second.return_code, 3)
            self.assertEqual(
                map_pipeline_state(
                    state,
                    process_return_code=second.return_code,
                ).backend_status,
                TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION,
            )

            job.queued_action = {
                "kind": "ambiguity",
                "ambiguity_id": "ambiguity_0001",
                "decision": "candidate",
                "candidate_id": "shot_0001_track_0001",
                "reviewer": "usr_fixture",
            }
            third = runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            state = read_pipeline_state(Path(job.pipeline_state_path))
            self.assertEqual(third.return_code, 0)
            self.assertEqual(
                map_pipeline_state(
                    state,
                    process_return_code=third.return_code,
                ).backend_status,
                TrackingBackendStatus.COMPLETED,
            )

    def test_confirm_absent_completes_without_bbox_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            job = fixture_job(settings, "tracking_absent")
            runner = TrackingProcessRunner(settings)
            runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            job.queued_action = {
                "kind": "review",
                "stage": "MEMORY",
                "decision": "approve",
            }
            runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            job.queued_action = {
                "kind": "ambiguity",
                "ambiguity_id": "ambiguity_0001",
                "decision": "absent",
            }
            result = runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            self.assertEqual(result.return_code, 0)
            state = read_pipeline_state(Path(job.pipeline_state_path))
            self.assertEqual(state["status"], "COMPLETE")

    def test_safe_block_and_fatal_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            runner = TrackingProcessRunner(settings)

            safe = fixture_job(settings, "tracking_safe_block")
            safe_result = runner.run(
                runner.command_for_job(safe, video_path=video),
                test_name=safe.test_name,
            )
            safe_state = read_pipeline_state(Path(safe.pipeline_state_path))
            self.assertEqual(
                map_pipeline_state(
                    safe_state,
                    process_return_code=safe_result.return_code,
                ).backend_status,
                TrackingBackendStatus.COMPLETED_SAFE_BLOCK,
            )

            fatal = fixture_job(settings, "tracking_fatal")
            fatal_result = runner.run(
                runner.command_for_job(fatal, video_path=video),
                test_name=fatal.test_name,
            )
            self.assertEqual(fatal_result.return_code, 2)
            self.assertIsNone(read_pipeline_state(Path(fatal.pipeline_state_path)))
            self.assertEqual(
                map_pipeline_state(
                    None,
                    process_return_code=fatal_result.return_code,
                ).backend_status,
                TrackingBackendStatus.FAILED,
            )

    def test_timeout_terminates_fake_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root, timeout=0.1)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            job = fixture_job(settings, "tracking_timeout")
            runner = TrackingProcessRunner(settings)
            with self.assertRaises(TrackingProcessTimeoutError):
                runner.run(
                    runner.command_for_job(job, video_path=video),
                    test_name=job.test_name,
                )


class TrackingArtifactAndTimelineTests(unittest.TestCase):
    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            job = fixture_job(settings, "tracking_security")
            Path(job.output_directory).mkdir(parents=True)
            outside = root / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            job.artifact_index = {
                "report": {
                    "relative_path": "../../secret.txt",
                    "mime_type": "text/plain",
                }
            }
            with self.assertRaises(TrackingArtifactNotFoundError):
                TrackingArtifactService(settings).resolve(job, "report")

    def test_timeline_range_and_absolute_path_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            job = fixture_job(settings, "tracking_timeline")
            runner = TrackingProcessRunner(settings)
            runner.run(
                runner.command_for_job(job, video_path=video),
                test_name=job.test_name,
            )
            state = read_pipeline_state(Path(job.pipeline_state_path))
            job.artifact_index = TrackingArtifactService(settings).collect(job, state)
            payload = TrackingTimelineService(
                TrackingArtifactService(settings)
            ).read(job, start_frame=1, end_frame=1)
            self.assertEqual([frame["frame_index"] for frame in payload["frames"]], [1])
            self.assertEqual(
                payload["video"]["path"],
                "media_asset:asset_fixture",
            )
            self.assertIsNone(payload["provenance"]["phase1_manifest"]["path"])

    def test_uncertain_frame_with_bbox_is_rejected(self) -> None:
        payload = {
            "schema_version": "kickclip.target_centric_e2e.v1",
            "frames": [
                {
                    "frame_index": 0,
                    "state": "AMBIGUOUS",
                    "bbox_xyxy": [1, 2, 3, 4],
                }
            ],
        }
        with self.assertRaises(TrackingContractError):
            TrackingTimelineService._validate(payload)


class TrackingServiceIdempotencyTests(unittest.TestCase):
    def test_identical_confirmation_is_applied_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = fixture_settings(root)
            engine = create_engine("sqlite+pysqlite:///:memory:")
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
            db: Session = SessionLocal()
            try:
                user = User(
                    user_id="usr_fixture",
                    email="tracking@example.com",
                    password_hash="!",
                    display_name="Tracking",
                )
                match = Match(
                    match_id="match_fixture",
                    owner_id=user.user_id,
                )
                asset = MediaAsset(
                    asset_id="asset_fixture",
                    match_id=match.match_id,
                    asset_type="RAW_VIDEO",
                    file_path="storage/video.mp4",
                )
                db.add_all([user, match, asset])
                db.flush()

                job = fixture_job(settings, "tracking_idempotent")
                job.status = (
                    TrackingBackendStatus.WAITING_CROSS_SHOT_CONFIRMATION.value
                )
                job.pending_action_type = "CROSS_SHOT_CONFIRMATION"
                job.pending_ambiguity_id = "ambiguity_0001"
                state = {
                    "test_name": job.test_name,
                    "status": "NEEDS_CONFIRMATION",
                    "pending_action": {
                        "type": "CROSS_SHOT_CONFIRMATION",
                        "ambiguity_id": "ambiguity_0001",
                    },
                    "ambiguities": [
                        {
                            "ambiguity_id": "ambiguity_0001",
                            "status": "PENDING",
                            "review_candidates": [
                                {
                                    "candidate_id": (
                                        "shot_0001_track_0001"
                                    )
                                }
                            ],
                        }
                    ],
                }
                state_path = Path(job.pipeline_state_path)
                state_path.parent.mkdir(parents=True)
                state_path.write_text(json.dumps(state), encoding="utf-8")
                db.add(job)
                db.commit()

                payload = TrackingAmbiguityConfirmationRequest(
                    decision="candidate",
                    candidate_id="shot_0001_track_0001",
                )
                service = TrackingJobService(db, settings=settings)
                first = service.queue_ambiguity_confirmation(
                    tracking_job_id=job.tracking_job_id,
                    ambiguity_id="ambiguity_0001",
                    user=user,
                    payload=payload,
                )
                first_key = first.last_action_key
                first_action = dict(first.queued_action)
                second = service.queue_ambiguity_confirmation(
                    tracking_job_id=job.tracking_job_id,
                    ambiguity_id="ambiguity_0001",
                    user=user,
                    payload=payload,
                )
                self.assertEqual(second.last_action_key, first_key)
                self.assertEqual(second.queued_action, first_action)
                self.assertEqual(second.status, TrackingBackendStatus.QUEUED.value)
            finally:
                db.close()
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
