from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import (
    event_candidate_ranking_v1_1_2a as api_module,
    highlights as highlights_module,
)
from app.db import models as _models  # noqa: F401
from app.db.base import Base
from app.db.session import get_db
from app.core.config import Settings
from app.domains.artifact.model import Artifact
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.highlight.event_candidate_ranking_v1_1.contract import (
    sha256_file,
)
from app.domains.highlight.event_candidate_ranking_v1_1.schema import (
    EventAnnotationFinalizeRequest,
    EventAnnotationReviewRequest,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a.annotation_service import (
    EventAnnotationCompatibilityV112aService,
)
from app.domains.highlight.event_candidate_ranking_v1_1_2a_integration import (
    EVENT_RANKING_V1_1_2A,
    IntegratedSceneTargetTrackingInstallationVerifier,
    install_v112a_integration,
)
from app.domains.highlight import (
    event_candidate_ranking_v1_1_2a_integration as integration_module,
)
from app.domains.highlight.model import SceneAITask
from app.domains.highlight.scene_ai_task import SceneAITaskExecutor
from app.domains.highlight import scene_ai_task as scene_task_module
from app.domains.match.model import Match
from app.domains.project.model import Project
from app.domains.tracking.verifier import TrackingInstallationStatus


class _TempStorage:
    def __init__(self, root: Path) -> None:
        self.project_root = root.resolve()
        self.storage_root = self.project_root

    def resolve_path(self, value) -> Path:
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )


def _seed_owner(db: Session) -> tuple[User, Project]:
    user = User(
        user_id="user",
        email="user@example.com",
        password_hash="x",
        display_name="User",
        role="USER",
        is_active=True,
        developer_mode_enabled=False,
        event_weights={},
    )
    match = Match(match_id="match", owner_id="user", metadata_={})
    project = Project(
        project_id="project",
        match_id="match",
        owner_id="user",
        title="Project",
        status="DRAFT",
    )
    db.add_all([user, match, project])
    db.commit()
    return user, project


def _ranking_document(version: str, policy_sha: str) -> dict:
    return {
        "schema_version": f"kickclip.event_candidate_ranking.{version}",
        "package": (
            "target_centric_tracking_event_candidate_ranking_"
            + version
        ),
        "freeze": {
            "source_manifest_sha256": "a" * 64,
            "candidate_feature_schema_sha256": "b" * 64,
            "ranking_policy_sha256": policy_sha,
        },
        "all_candidates": [
            {"candidate_id": "candidate_a"},
            {"candidate_id": "candidate_b"},
        ],
        "shortlist": [{"candidate_id": "candidate_a"}],
    }


def _add_ranking(
    db: Session,
    root: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    version: str,
    policy_sha: str,
) -> Artifact:
    path = root / f"{artifact_id}.json"
    path.write_text(
        json.dumps(_ranking_document(version, policy_sha)),
        encoding="utf-8",
    )
    artifact = Artifact(
        artifact_id=artifact_id,
        match_id="match",
        project_id="project",
        analysis_job_id=None,
        artifact_type=artifact_type,
        file_path=path.name,
        mime_type="application/json",
        metadata_={
            "sha256": sha256_file(path),
            "ranking_id": artifact_id,
            "owner_id": "user",
        },
    )
    db.add(artifact)
    db.flush()
    return artifact


def test_v1_1_2a_annotation_e2e_and_v112_review_mixing_blocked(
    tmp_path: Path,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user, _ = _seed_owner(db)
        v112a = _add_ranking(
            db,
            tmp_path,
            artifact_id="ranking_v112a",
            artifact_type=(
                "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW"
            ),
            version="v1_1_2a",
            policy_sha="c" * 64,
        )
        v112 = _add_ranking(
            db,
            tmp_path,
            artifact_id="ranking_v112",
            artifact_type=(
                "EVENT_CANDIDATE_RANKING_V1_1_2_SHADOW"
            ),
            version="v1_1_2",
            policy_sha="c" * 64,
        )
        db.commit()
        service = EventAnnotationCompatibilityV112aService(db)
        service.storage = _TempStorage(tmp_path)
        payload = EventAnnotationReviewRequest(
            primary_actor_visible=True,
            primary_actor_in_candidate_set=True,
            primary_actor_candidate_ids=["candidate_a"],
            directly_related_candidate_ids=["candidate_b"],
            candidate_roles={
                "candidate_a": "PRIMARY_EVENT_ACTOR"
            },
        )
        review_v112a = service.submit_review(
            ranking_artifact_id=v112a.artifact_id,
            user=user,
            payload=payload,
        )
        review_v112 = service.submit_review(
            ranking_artifact_id=v112.artifact_id,
            user=user,
            payload=payload,
        )
        with pytest.raises(
            ValueError,
            match="ranking version does not match",
        ):
            service.finalize(
                ranking_artifact_id=v112a.artifact_id,
                user=user,
                payload=EventAnnotationFinalizeRequest(
                    review_artifact_ids=[review_v112.artifact_id],
                    approval_mode="FINAL_APPROVED",
                    approved_reviewer="user",
                ),
            )
        final = service.finalize(
            ranking_artifact_id=v112a.artifact_id,
            user=user,
            payload=EventAnnotationFinalizeRequest(
                review_artifact_ids=[review_v112a.artifact_id],
                approval_mode="FINAL_APPROVED",
                approved_reviewer="user",
            ),
        )
        evaluation = service.evaluate(
            ranking_artifact_id=v112a.artifact_id,
            annotation_artifact_id=final.artifact_id,
            user=user,
        )
        assert evaluation["primary_actor_recall_at_1"] == 1.0
        assert evaluation["ranking_identity"][
            "ranking_package"
        ].endswith("v1_1_2a")
        assert evaluation["cross_version_metrics_mixed"] is False


def test_existing_annotation_api_uses_v112a_compatibility_service() -> None:
    install_v112a_integration(highlights_module)
    assert (
        highlights_module.EventAnnotationCompatibilityService
        is EventAnnotationCompatibilityV112aService
    )


def test_post_queue_dispatch_creates_v112a_artifact_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    with SessionFactory() as db:
        user, _ = _seed_owner(db)
        user_id = user.user_id

    class _Resolved:
        def to_dict(self):
            return {
                "event_id": "event",
                "event_label": "Goal",
                "canonical_event_label": "goal",
                "event_time_sec": 5.0,
                "event_confidence": 0.9,
                "event_source_job_id": "job",
                "event_source_artifact_id": "artifact",
                "scene_id": "scene",
                "scene_start_sec": 0.0,
                "scene_end_sec": 10.0,
                "match_id": "match",
                "project_id": "project",
                "revision_id": "revision",
            }

    freeze = {
        "discovery_id": "discovery",
        "shortlist_size": 5,
    }
    monkeypatch.setattr(
        api_module.EventCandidateRankingV112aBackendAdapter,
        "prepare",
        lambda self, **kwargs: (_Resolved(), freeze),
    )

    class _SubmitRecorder:
        task_ids: list[str] = []

        def submit(self, task_id: str) -> bool:
            self.task_ids.append(task_id)
            return True

    recorder = _SubmitRecorder()
    monkeypatch.setattr(
        api_module,
        "get_scene_ai_task_executor",
        lambda: recorder,
    )
    application = FastAPI()
    application.include_router(api_module.router, prefix="/api/v1")

    def _db_override():
        db = SessionFactory()
        try:
            yield db
        finally:
            db.close()

    def _user_override():
        with SessionFactory() as db:
            return db.get(User, user_id)

    application.dependency_overrides[get_db] = _db_override
    application.dependency_overrides[
        get_current_user
    ] = _user_override
    response = TestClient(application).post(
        "/api/v1/projects/project/highlight/revisions/revision/"
        "event-candidate-rankings/v1.1.2a",
        json={
            "event_id": "event",
            "scene_id": "scene",
            "shortlist_size": 5,
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "QUEUED"
    assert body["task_type"] == EVENT_RANKING_V1_1_2A
    assert recorder.task_ids == [body["task_id"]]
    with SessionFactory() as db:
        task = db.get(SceneAITask, body["task_id"])
        assert task is not None
        assert task.payload == {
            "revision_id": "revision",
            "shortlist_size": 5,
            "resolved_event": _Resolved().to_dict(),
            "freeze_material": freeze,
        }

    def _fake_run(adapter, **kwargs):
        artifact = Artifact(
            artifact_id="artifact_v112a",
            match_id="match",
            project_id="project",
            analysis_job_id=None,
            artifact_type=(
                "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW"
            ),
            file_path="storage/fake-v112a.json",
            mime_type="application/json",
            metadata_={
                "automatic_target_confirmation": False,
                "owner_id": "user",
                "sha256": "d" * 64,
            },
        )
        adapter.db.add(artifact)
        return {
            "artifact_id": artifact.artifact_id,
            "automatic_target_confirmation": False,
        }

    monkeypatch.setattr(
        integration_module.EventCandidateRankingV112aBackendAdapter,
        "run",
        _fake_run,
    )
    monkeypatch.setattr(
        scene_task_module,
        "SessionLocal",
        SessionFactory,
    )
    executor = SceneAITaskExecutor()
    executor._execute(body["task_id"])
    with SessionFactory() as db:
        completed = db.get(SceneAITask, body["task_id"])
        artifact = db.get(Artifact, "artifact_v112a")
        assert completed.status == "COMPLETED"
        assert completed.result["artifact_id"] == "artifact_v112a"
        assert artifact.artifact_type == (
            "EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW"
        )
        assert artifact.metadata_[
            "automatic_target_confirmation"
        ] is False


def test_global_verifier_requires_v112a_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_status = TrackingInstallationStatus(
        enabled=True,
        available=True,
        checked_at=datetime.now(timezone.utc),
        code="BASE",
        message="base",
        components={
            "SCENE_DISCOVERY_RUNTIME_VERIFIED": True,
            "R3_TRACKING_RUNTIME_VERIFIED": False,
            "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED": True,
            "EVENT_RANKING_SAFETY_RUNTIME_VERIFIED": True,
            "EVENT_RANKING_CONTRACT_RUNTIME_VERIFIED": True,
            "EVENT_RANKING_RUNTIME_VERIFIED": True,
            "FULL_EVENT_RECOMMENDATION_E2E_VERIFIED": False,
        },
    )
    monkeypatch.setattr(
        integration_module._base_scene_verifier,
        "_run_check",
        lambda self: base_status,
    )

    class _CompatibilityVerifier:
        def __init__(self, package_root):
            self.package_root = package_root

        def check(self):
            return SimpleNamespace(
                event_ranking_compatibility_runtime_verified=True
            )

    monkeypatch.setattr(
        integration_module,
        "EventCandidateRankingV112aVerifier",
        _CompatibilityVerifier,
    )
    monkeypatch.setattr(
        IntegratedSceneTargetTrackingInstallationVerifier,
        "_check_scene_discovery_runtime",
        lambda self: (True, "ok"),
    )
    verifier = IntegratedSceneTargetTrackingInstallationVerifier.__new__(
        IntegratedSceneTargetTrackingInstallationVerifier
    )
    result = verifier._run_check()
    assert result.components[
        "EVENT_RANKING_COMPATIBILITY_RUNTIME_VERIFIED"
    ] is True
    assert result.components[
        "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"
    ] is True
    assert result.components["EVENT_RANKING_RUNTIME_VERIFIED"] is True
    assert result.components["R3_TRACKING_RUNTIME_VERIFIED"] is False
    assert result.components[
        "FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"
    ] is False

    class _FailedCompatibilityVerifier(_CompatibilityVerifier):
        def check(self):
            return SimpleNamespace(
                event_ranking_compatibility_runtime_verified=False
            )

    monkeypatch.setattr(
        integration_module,
        "EventCandidateRankingV112aVerifier",
        _FailedCompatibilityVerifier,
    )
    failed = verifier._run_check()
    assert failed.components[
        "EVENT_RANKING_COMPATIBILITY_RUNTIME_VERIFIED"
    ] is False
    assert failed.components[
        "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"
    ] is False
    assert failed.components["EVENT_RANKING_RUNTIME_VERIFIED"] is False
    assert failed.components[
        "FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"
    ] is False


def test_scene_discovery_and_event_ranking_do_not_require_r3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path.resolve()
    package = (
        runtime_root
        / "target_centric_tracking_scene_target_selection_v1"
    )
    package.mkdir()
    runner = package / "run_scene_target_selection.py"
    runner.write_text("pass\n", encoding="utf-8")
    manifest = package / "scene_target_selection_frozen_manifest.json"
    manifest.write_text('{"frozen":true}', encoding="utf-8")
    checkpoint = (
        runtime_root
        / "weights"
        / "rfdetr"
        / "checkpoint_best_regular.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-rfdetr")
    verifier_script = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "fake_scene_selection_verifier.py"
    )

    class _NoR3:
        def check(self):
            return TrackingInstallationStatus(
                enabled=False,
                available=False,
                checked_at=datetime.now(timezone.utc),
                code="TRACKING_DISABLED",
                message="R3 tracking is not installed.",
            )

    class _CompatibilityVerifier:
        def __init__(self, package_root):
            self.package_root = package_root

        def check(self):
            return SimpleNamespace(
                event_ranking_compatibility_runtime_verified=True
            )

    monkeypatch.setattr(
        integration_module,
        "EventCandidateRankingV112aVerifier",
        _CompatibilityVerifier,
    )
    settings = Settings.model_construct(
        TRACKING_ENABLED=False,
        TRACKING_PROJECT_ROOT="",
        TRACKING_PYTHON_EXECUTABLE="",
        TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH="",
        TRACKING_SCENE_SELECTION_VERIFY_SCRIPT_PATH="",
        SCENE_TARGET_SELECTION_PROJECT_ROOT=str(runtime_root),
        SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE=sys.executable,
        SCENE_TARGET_SELECTION_SCRIPT_PATH=str(runner),
        SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH=str(
            verifier_script
        ),
        SCENE_TARGET_SELECTION_MANIFEST_PATH=str(manifest),
        SCENE_TARGET_SELECTION_MANIFEST_SHA256=hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        TRACKING_R2_MANIFEST_PATH="",
        TRACKING_R2_MANIFEST_SHA256="",
        TRACKING_R3_MANIFEST_PATH="",
        TRACKING_R3_MANIFEST_SHA256="",
        TRACKING_VERIFY_TIMEOUT_SECONDS=10,
    )
    status = IntegratedSceneTargetTrackingInstallationVerifier(
        settings,
        generic=_NoR3(),
    ).check()
    assert status.available is False
    assert status.components["SCENE_DISCOVERY_RUNTIME_VERIFIED"] is True
    assert status.components[
        "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"
    ] is True
    assert status.components[
        "EVENT_RANKING_SAFETY_RUNTIME_VERIFIED"
    ] is True
    assert status.components[
        "EVENT_RANKING_CONTRACT_RUNTIME_VERIFIED"
    ] is True
    assert status.components["R3_TRACKING_RUNTIME_VERIFIED"] is False
    assert status.components[
        "FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"
    ] is False


def test_compat_r1_runtime_states_are_reported_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path.resolve()
    package = (
        runtime_root
        / "target_centric_tracking_scene_discovery_compat_r1"
    )
    package.mkdir()
    runner = package / "compat_runtime.py"
    runner.write_text("pass\n", encoding="utf-8")
    manifest = package / "scene_discovery_compat_r1_manifest.json"
    manifest.write_text('{"runtime_mode":"COMPAT_R1"}', encoding="utf-8")
    checkpoint = (
        runtime_root
        / "weights"
        / "rfdetr"
        / "checkpoint_best_regular.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-rfdetr")
    verifier_script = package / "verify_compat_runtime.py"
    verifier_script.write_text(
        "import json\n"
        "print(json.dumps({"
        "'scene_discovery_frozen_runtime_verified': False,"
        "'scene_discovery_compat_runtime_verified': True,"
        "'scene_discovery_runtime_verified': True,"
        "'scene_discovery_runtime_mode': 'COMPAT_R1'}))\n",
        encoding="utf-8",
    )

    class _NoR3:
        def check(self):
            return TrackingInstallationStatus(
                enabled=False,
                available=False,
                checked_at=datetime.now(timezone.utc),
                code="TRACKING_DISABLED",
                message="R3 tracking is not installed.",
            )

    class _CompatibilityVerifier:
        def __init__(self, package_root):
            self.package_root = package_root

        def check(self):
            return SimpleNamespace(
                event_ranking_compatibility_runtime_verified=True
            )

    monkeypatch.setattr(
        integration_module,
        "EventCandidateRankingV112aVerifier",
        _CompatibilityVerifier,
    )
    settings = Settings.model_construct(
        TRACKING_ENABLED=False,
        TRACKING_PROJECT_ROOT="",
        TRACKING_PYTHON_EXECUTABLE="",
        SCENE_DISCOVERY_PROJECT_ROOT=str(runtime_root),
        SCENE_DISCOVERY_PYTHON_EXECUTABLE=sys.executable,
        SCENE_DISCOVERY_SCRIPT_PATH=str(runner),
        SCENE_DISCOVERY_VERIFY_SCRIPT_PATH=str(verifier_script),
        SCENE_DISCOVERY_MANIFEST_PATH=str(manifest),
        SCENE_DISCOVERY_MANIFEST_SHA256=hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        SCENE_TARGET_SELECTION_PROJECT_ROOT="",
        SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE="",
        SCENE_TARGET_SELECTION_SCRIPT_PATH="",
        SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH="",
        SCENE_TARGET_SELECTION_MANIFEST_PATH="",
        SCENE_TARGET_SELECTION_MANIFEST_SHA256="",
        TRACKING_R2_MANIFEST_PATH="",
        TRACKING_R2_MANIFEST_SHA256="",
        TRACKING_R3_MANIFEST_PATH="",
        TRACKING_R3_MANIFEST_SHA256="",
        TRACKING_VERIFY_TIMEOUT_SECONDS=10,
    )
    status = IntegratedSceneTargetTrackingInstallationVerifier(
        settings,
        generic=_NoR3(),
    ).check()
    assert status.components[
        "SCENE_DISCOVERY_FROZEN_RUNTIME_VERIFIED"
    ] is False
    assert status.components[
        "SCENE_DISCOVERY_COMPAT_RUNTIME_VERIFIED"
    ] is True
    assert status.components["SCENE_DISCOVERY_RUNTIME_VERIFIED"] is True
    assert status.components[
        "SCENE_DISCOVERY_RUNTIME_MODE_COMPAT_R1"
    ] is True
    assert status.components["R3_TRACKING_RUNTIME_VERIFIED"] is False
    assert status.components[
        "EVENT_RANKING_SHADOW_RUNTIME_VERIFIED"
    ] is True
    assert status.components[
        "FULL_EVENT_RECOMMENDATION_E2E_VERIFIED"
    ] is False


def test_scene_discovery_dispatch_uses_compat_runtime_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    script = tmp_path / "compat_runtime.py"
    script.write_text("pass\n", encoding="utf-8")

    class _Runtime:
        def __init__(self, settings, **kwargs):
            assert settings.SCENE_TARGET_SELECTION_SCRIPT_PATH == str(script)
            assert kwargs["script_setting_name"] == (
                "SCENE_DISCOVERY_SCRIPT_PATH"
            )

        def run(self, arguments):
            calls.append(arguments)

    monkeypatch.setattr(
        integration_module, "ConfiguredSceneRuntime", _Runtime
    )
    service = SimpleNamespace(
        settings=Settings.model_construct(
            SCENE_DISCOVERY_PROJECT_ROOT=str(tmp_path),
            SCENE_DISCOVERY_PYTHON_EXECUTABLE=sys.executable,
            SCENE_DISCOVERY_SCRIPT_PATH=str(script),
            SCENE_DISCOVERY_PROCESS_TIMEOUT_SECONDS=60,
            SCENE_TARGET_SELECTION_PROJECT_ROOT="",
            SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE="",
            SCENE_TARGET_SELECTION_SCRIPT_PATH="",
        )
    )
    integration_module._integrated_scene_selection_run(
        service,
        ["discover", "--scene-id", "scene"],
    )
    assert calls == [["discover", "--scene-id", "scene"]]
