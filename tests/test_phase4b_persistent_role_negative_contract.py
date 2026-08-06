from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    PROJECT_ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)
CANDIDATE_SCHEMA_PATH = (
    PROJECT_ROOT / "app" / "domains" / "candidate_handoff_r1" / "schema.py"
)
TRACKING_SCHEMA_PATH = PROJECT_ROOT / "app" / "domains" / "tracking" / "schema.py"
TRACKING_SERVICE_PATH = PROJECT_ROOT / "app" / "domains" / "tracking" / "service.py"
CANDIDATE_SERVICE_PATH = (
    PROJECT_ROOT / "app" / "domains" / "candidate_handoff_r1" / "service.py"
)
PROCESS_RUNNER_PATH = PROJECT_ROOT / "app" / "domains" / "tracking" / "process_runner.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("kickclip_persistent_role_adapter_test", ADAPTER_PATH)
CANDIDATE_SCHEMA = _load("kickclip_persistent_role_schema_test", CANDIDATE_SCHEMA_PATH)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest):
    base = PROJECT_ROOT / "storage" / "pytest_phase4b_role_tmp"
    base.mkdir(parents=True, exist_ok=True)
    safe = "".join(
        value if value.isalnum() or value in {"-", "_"} else "_"
        for value in request.node.name
    )
    path = base / f"{safe}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_matrix(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, rows.astype(np.float32), allow_pickle=False)


def _candidate(tmp_path: Path, candidate_id: str, rows: np.ndarray) -> dict:
    candidate_dir = tmp_path / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True)
    embeddings_path = candidate_dir / "candidate_embeddings.npy"
    prototype_path = candidate_dir / "candidate_prototype.npy"
    _write_matrix(embeddings_path, rows)
    with prototype_path.open("wb") as stream:
        np.save(stream, rows.mean(axis=0).astype(np.float32), allow_pickle=False)
    manifest = {
        "schema_version": "kickclip.runtime_candidate_manifest.r1_1",
        "candidate_id": candidate_id,
        "shot_id": "shot_0006",
        "candidate_embeddings": {
            "path": str(embeddings_path),
            "sha256": _sha(embeddings_path),
            "prototype_path": str(prototype_path),
            "prototype_sha256": _sha(prototype_path),
        },
        "automatic_target_confirmation": False,
    }
    manifest_path = candidate_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "candidate_id": candidate_id,
        "shot_id": "shot_0006",
        "status": "PENDING",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
    }


def _historical_role_memory(tmp_path: Path) -> dict:
    memory_dir = tmp_path / "shot_0004" / "memory_contract"
    embeddings_path = memory_dir / "same_shot_role_negative_embeddings.npy"
    rows = np.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.9, 0.1]],
        dtype=np.float32,
    )
    _write_matrix(embeddings_path, rows)
    manifest = {
        "schema_version": "kickclip.phase4b_same_shot_role_negative_memory.v1",
        "policy": ADAPTER.PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
        "shot_id": "shot_0004",
        "negative_memory_available": True,
        "selected_count": 2,
        "references": [
            {
                "detection_id": "staff_1",
                "class_id": 3,
                "class_name": "staff",
                "path": "staff_1.jpg",
                "sha256": "a" * 64,
            },
            {
                "detection_id": "referee_1",
                "class_id": 2,
                "class_name": "referee",
                "path": "referee_1.jpg",
                "sha256": "b" * 64,
            },
        ],
        "automatic_target_confirmation": False,
    }
    manifest_path = memory_dir / "same_shot_role_negative_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "policy": ADAPTER.PHASE4B_NEGATIVE_ROLE_MEMORY_POLICY,
        "negative_memory_available": True,
        "selected_count": 2,
        "embeddings_path": str(embeddings_path),
        "embeddings_sha256": _sha(embeddings_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
    }


def _state(tmp_path: Path, candidate: dict) -> dict:
    identity_marker = {
        "policy": ADAPTER.PHASE4B_IDENTITY_NEGATIVE_MEMORY_POLICY,
        "revision_id": "identity_should_not_change",
        "embedding_count": 24,
    }
    return {
        "status": "NEEDS_CONFIRMATION",
        "decision": "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION",
        "pending_action": {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": "ambiguity_phase4b_shot_0006_g006",
            "shot_id": "shot_0006",
            "candidate_ids": [candidate["candidate_id"]],
        },
        "ambiguities": [
            {
                "ambiguity_id": "ambiguity_phase4b_shot_0006_g006",
                "shot_id": "shot_0006",
                "status": "PENDING",
                "review_candidates": [candidate],
            }
        ],
        "shots": [
            {"shot_id": "shot_0004", "status": "SEARCH_EXHAUSTED_NO_REVIEWABLE_CANDIDATE"},
            {"shot_id": "shot_0006", "status": "AMBIGUOUS_REVIEW_REQUIRED"},
            {"shot_id": "shot_0007", "status": "SEARCHING_MEMORY_READY"},
        ],
        "shot_search_results": {
            "shot_0004": {
                "status": "SAFE_REJECTED_SEARCHING",
                "negative_role_memory": _historical_role_memory(tmp_path),
            },
            "shot_0006": {"status": "AMBIGUOUS"},
        },
        "confirmations": [],
        "runtime": {
            "candidate_scoring_generation": 6,
            "phase4b_identity_negative_memory": identity_marker,
            "memory_revision_path": str(tmp_path / "target_memory.json"),
            "memory_revision_sha256": "c" * 64,
        },
    }


def test_schema_accepts_non_player_role_full_shot_decision() -> None:
    request = CANDIDATE_SCHEMA.CandidateReviewDecisionRequest(
        state="NONE_OF_THESE_NON_PLAYER_ROLE",
        candidate_id=None,
        full_frame_context_sha256="a" * 64,
        shot_clip_sha256="b" * 64,
    )
    assert request.state.value == "NONE_OF_THESE_NON_PLAYER_ROLE"


def test_schema_rejects_candidate_id_for_non_player_role() -> None:
    with pytest.raises(Exception):
        CANDIDATE_SCHEMA.CandidateReviewDecisionRequest(
            state="NONE_OF_THESE_NON_PLAYER_ROLE",
            candidate_id="shot_0006_track_0001",
            full_frame_context_sha256="a" * 64,
            shot_clip_sha256="b" * 64,
        )


def test_non_player_role_builds_persistent_memory_and_bootstraps_history(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        tmp_path,
        "shot_0006_track_0001",
        np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    state = _state(tmp_path, candidate)
    identity_before = dict(state["runtime"]["phase4b_identity_negative_memory"])

    memory = ADAPTER._phase4b_apply_non_player_role_review(
        output_dir=tmp_path,
        state=state,
        ambiguity_id="ambiguity_phase4b_shot_0006_g006",
        reviewer="USER",
        note="The surfaced candidate is coaching staff.",
    )

    assert state["status"] == "RUNNING"
    assert state["decision"] == ADAPTER.PHASE4B_CONTINUE_DECISION
    assert state["pending_action"] is None
    assert state["shots"][1]["status"] == ADAPTER.PHASE4B_NON_PLAYER_ROLE_SHOT_STATUS
    assert state["runtime"]["phase4b_next_shot_id"] == "shot_0007"
    assert state["runtime"]["phase4b_identity_negative_memory"] == identity_before
    assert state["confirmations"][-1]["decision"] == "NONE_OF_THESE_NON_PLAYER_ROLE"
    assert memory["user_confirmed_non_player_role"] is True
    assert memory["automatic_target_confirmation"] is False
    assert Path(memory["manifest_path"]).is_file()
    assert Path(memory["embeddings_path"]).is_file()

    manifest = json.loads(Path(memory["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == ADAPTER.PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_SCHEMA
    assert manifest["policy"] == ADAPTER.PHASE4B_PERSISTENT_ROLE_NEGATIVE_MEMORY_POLICY
    assert manifest["decision"] == "NONE_OF_THESE_NON_PLAYER_ROLE"
    assert manifest["target_positive_memory_modified"] is False
    assert manifest["identity_negative_memory_modified"] is False
    assert manifest["historical_detector_source_count"] == 1
    assert manifest["new_candidate_embedding_count"] == 2
    assert any(
        row.get("source_kind") == "USER_CONFIRMED_NON_PLAYER_ROLE_CANDIDATE"
        for row in manifest["references"]
    )
    assert any(
        row.get("source_kind") == "DETECTOR_LABELED_STAFF_REFEREE"
        for row in manifest["references"]
    )


def test_non_player_role_accumulates_previous_persistent_memory(tmp_path: Path) -> None:
    first_candidate = _candidate(
        tmp_path,
        "shot_0006_track_0001",
        np.asarray([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32),
    )
    # Remove historical source to isolate previous-revision accumulation.
    state = _state(tmp_path, first_candidate)
    state["shot_search_results"]["shot_0004"]["negative_role_memory"][
        "negative_memory_available"
    ] = False
    first = ADAPTER._phase4b_apply_non_player_role_review(
        output_dir=tmp_path,
        state=state,
        ambiguity_id="ambiguity_phase4b_shot_0006_g006",
        reviewer="USER",
        note="first",
    )

    second_candidate = _candidate(
        tmp_path,
        "shot_0007_track_0001",
        np.asarray([[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]], dtype=np.float32),
    )
    state["pending_action"] = {
        "type": "CROSS_SHOT_CONFIRMATION",
        "ambiguity_id": "ambiguity_phase4b_shot_0007_g007",
        "shot_id": "shot_0007",
        "candidate_ids": [second_candidate["candidate_id"]],
    }
    state["ambiguities"].append(
        {
            "ambiguity_id": "ambiguity_phase4b_shot_0007_g007",
            "shot_id": "shot_0007",
            "status": "PENDING",
            "review_candidates": [second_candidate],
        }
    )
    state["shots"][2]["status"] = "AMBIGUOUS_REVIEW_REQUIRED"
    state["shots"].append({"shot_id": "shot_0008", "status": "SEARCHING_MEMORY_READY"})
    second = ADAPTER._phase4b_apply_non_player_role_review(
        output_dir=tmp_path,
        state=state,
        ambiguity_id="ambiguity_phase4b_shot_0007_g007",
        reviewer="USER",
        note="second",
    )
    manifest = json.loads(Path(second["manifest_path"]).read_text(encoding="utf-8"))
    assert second["revision_id"] != first["revision_id"]
    assert second["embedding_count"] >= first["embedding_count"]
    assert manifest["previous_revision_id"] == first["revision_id"]


def test_runtime_sources_wire_non_player_role_to_adapter() -> None:
    tracking_schema = TRACKING_SCHEMA_PATH.read_text(encoding="utf-8")
    tracking_service = TRACKING_SERVICE_PATH.read_text(encoding="utf-8")
    candidate_service = CANDIDATE_SERVICE_PATH.read_text(encoding="utf-8")
    process_runner = PROCESS_RUNNER_PATH.read_text(encoding="utf-8")
    adapter = ADAPTER_PATH.read_text(encoding="utf-8")

    assert 'NON_PLAYER_ROLE = "non_player_role"' in tracking_schema
    assert 'NONE_OF_THESE_NON_PLAYER_ROLE = "NONE_OF_THESE_NON_PLAYER_ROLE"' in CANDIDATE_SCHEMA_PATH.read_text(encoding="utf-8")
    assert '"kind": "non_player_role"' in candidate_service
    assert '"non_player_role": "non_player_role"' in tracking_service
    assert 'elif kind == "non_player_role":' in process_runner
    assert '"--reject-all-candidates-as-non-player-role"' in process_runner
    assert 'args.resume and args.reject_all_candidates_as_non_player_role' in adapter
    assert "USER_CONFIRMED_NON_PLAYER_PLUS_DETECTOR_ROLE_NEGATIVES_R1" in adapter
