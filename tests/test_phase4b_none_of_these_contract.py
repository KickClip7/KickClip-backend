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


ADAPTER = _load("kickclip_none_of_these_adapter_test", ADAPTER_PATH)
CANDIDATE_SCHEMA = _load("kickclip_none_of_these_schema_test", CANDIDATE_SCHEMA_PATH)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest):
    """Avoid Windows TEMP permission failures by using project-local storage."""
    base = PROJECT_ROOT / "storage" / "pytest_phase4b_none_tmp"
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


def _candidate(tmp_path: Path, candidate_id: str, rows: np.ndarray) -> dict:
    candidate_dir = tmp_path / candidate_id
    candidate_dir.mkdir(parents=True)
    embeddings_path = candidate_dir / "candidate_embeddings.npy"
    prototype_path = candidate_dir / "candidate_prototype.npy"
    with embeddings_path.open("wb") as stream:
        np.save(stream, rows.astype(np.float32), allow_pickle=False)
    prototype = rows.mean(axis=0).astype(np.float32)
    with prototype_path.open("wb") as stream:
        np.save(stream, prototype, allow_pickle=False)
    manifest = {
        "schema_version": "kickclip.runtime_candidate_manifest.r1_1",
        "candidate_id": candidate_id,
        "shot_id": "shot_0005",
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
        "shot_id": "shot_0005",
        "status": "PENDING",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
    }


def _state(candidates: list[dict]) -> dict:
    return {
        "status": "NEEDS_CONFIRMATION",
        "decision": "PAUSE_FOR_PHASE4B_CROSS_SHOT_CONFIRMATION",
        "pending_action": {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": "ambiguity_phase4b_shot_0005_g005",
            "shot_id": "shot_0005",
            "candidate_ids": [row["candidate_id"] for row in candidates],
        },
        "ambiguities": [
            {
                "ambiguity_id": "ambiguity_phase4b_shot_0005_g005",
                "shot_id": "shot_0005",
                "status": "PENDING",
                "review_candidates": candidates,
            }
        ],
        "shots": [
            {"shot_id": "shot_0005", "status": "AMBIGUOUS_REVIEW_REQUIRED"},
            {"shot_id": "shot_0006", "status": "SEARCHING_MEMORY_READY"},
        ],
        "shot_search_results": {"shot_0005": {"status": "AMBIGUOUS"}},
        "confirmations": [],
        "runtime": {"candidate_scoring_generation": 5},
    }


def test_candidate_review_schema_accepts_none_of_these_without_candidate() -> None:
    request = CANDIDATE_SCHEMA.CandidateReviewDecisionRequest(
        state="NONE_OF_THESE",
        candidate_id=None,
        full_frame_context_sha256="a" * 64,
        shot_clip_sha256="b" * 64,
    )
    assert request.state.value == "NONE_OF_THESE"
    assert request.candidate_id is None


def test_candidate_review_schema_rejects_candidate_for_none_of_these() -> None:
    with pytest.raises(Exception):
        CANDIDATE_SCHEMA.CandidateReviewDecisionRequest(
            state="NONE_OF_THESE",
            candidate_id="shot_0005_track_0001",
            full_frame_context_sha256="a" * 64,
            shot_clip_sha256="b" * 64,
        )


def test_no_preview_detector_accepts_only_the_ffmpeg_warning(tmp_path: Path) -> None:
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    audit_path = detector_dir / "audit.json"
    manifest_path = detector_dir / "input_manifest.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "PASS_WITH_WARNINGS",
                "findings": [
                    {"severity": "WARNING", "code": "FFMPEG_NOT_FOUND"}
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({"status": "PASS_WITH_WARNINGS"}),
        encoding="utf-8",
    )

    ADAPTER._phase4b_accept_no_preview_ffmpeg_warning(detector_dir)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert manifest["status"] == "PASS"
    assert audit["findings"][0]["code"] == "FFMPEG_NOT_FOUND"
    assert (
        audit["adapter_warning_compatibility"]["policy"]
        == "NO_PREVIEW_FFMPEG_WARNING_COMPATIBILITY_R1"
    )


def test_phase4b_authorization_recovers_only_from_matching_pass_report(
    tmp_path: Path,
) -> None:
    memory_sha = "a" * 64
    state = {"runtime": {"memory_revision_sha256": memory_sha}}
    (tmp_path / "phase4b_first_cross_shot_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "policy": ADAPTER.PHASE4B_POLICY,
                "memory_revision_sha256": memory_sha,
                "automatic_target_confirmation": False,
            }
        ),
        encoding="utf-8",
    )

    ADAPTER._phase4b_restore_authorization_from_report(
        output_dir=tmp_path,
        state=state,
    )

    assert state["runtime"]["phase4a_initial_memory_review_status"] == "PASS"
    assert state["runtime"]["phase4b_cross_shot_scoring_authorized"] is True


def test_none_of_these_builds_user_confirmed_identity_negative_memory(
    tmp_path: Path,
) -> None:
    candidate_1 = _candidate(
        tmp_path,
        "shot_0005_track_0001",
        np.asarray([[1, 0, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32),
    )
    candidate_2 = _candidate(
        tmp_path,
        "shot_0005_track_0002",
        np.asarray([[0, 1, 0, 0], [0, 0.9, 0.1, 0]], dtype=np.float32),
    )
    state = _state([candidate_1, candidate_2])

    memory = ADAPTER._phase4b_apply_none_of_these_review(
        output_dir=tmp_path,
        state=state,
        ambiguity_id="ambiguity_phase4b_shot_0005_g005",
        reviewer="USER",
        note="Both candidates are different people.",
    )

    assert state["status"] == "RUNNING"
    assert state["decision"] == ADAPTER.PHASE4B_CONTINUE_DECISION
    assert state["pending_action"] is None
    assert state["shots"][0]["status"] == ADAPTER.PHASE4B_NONE_OF_THESE_SHOT_STATUS
    assert state["runtime"]["phase4b_next_shot_id"] == "shot_0006"
    assert state["confirmations"][-1]["decision"] == "NONE_OF_THESE"
    assert memory["user_confirmed_only"] is True
    assert memory["automatic_target_confirmation"] is False
    assert Path(memory["manifest_path"]).is_file()
    assert Path(memory["embeddings_path"]).is_file()
    assert _sha(Path(memory["manifest_path"])) == memory["manifest_sha256"]
    assert _sha(Path(memory["embeddings_path"])) == memory["embeddings_sha256"]

    manifest = json.loads(Path(memory["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["decision"] == "NONE_OF_THESE"
    assert manifest["target_positive_memory_modified"] is False
    assert manifest["rejected_candidate_ids"] == [
        "shot_0005_track_0001",
        "shot_0005_track_0002",
    ]


def test_none_of_these_accumulates_previous_identity_negatives(tmp_path: Path) -> None:
    first = _candidate(
        tmp_path,
        "shot_0005_track_0001",
        np.asarray([[1, 0, 0], [0.8, 0.2, 0]], dtype=np.float32),
    )
    state = _state([first])
    first_memory = ADAPTER._phase4b_apply_none_of_these_review(
        output_dir=tmp_path,
        state=state,
        ambiguity_id="ambiguity_phase4b_shot_0005_g005",
        reviewer="USER",
        note="first",
    )

    second = _candidate(
        tmp_path,
        "shot_0006_track_0001",
        np.asarray([[0, 1, 0], [0, 0.8, 0.2]], dtype=np.float32),
    )
    state["pending_action"] = {
        "type": "CROSS_SHOT_CONFIRMATION",
        "ambiguity_id": "ambiguity_phase4b_shot_0006_g006",
        "shot_id": "shot_0006",
        "candidate_ids": [second["candidate_id"]],
    }
    state["ambiguities"].append(
        {
            "ambiguity_id": "ambiguity_phase4b_shot_0006_g006",
            "shot_id": "shot_0006",
            "status": "PENDING",
            "review_candidates": [second],
        }
    )
    state["shots"][1]["status"] = "AMBIGUOUS_REVIEW_REQUIRED"
    state["shots"].append(
        {"shot_id": "shot_0007", "status": "SEARCHING_MEMORY_READY"}
    )
    second_memory = ADAPTER._phase4b_apply_none_of_these_review(
        output_dir=tmp_path,
        state=state,
        ambiguity_id="ambiguity_phase4b_shot_0006_g006",
        reviewer="USER",
        note="second",
    )
    assert second_memory["revision_id"] != first_memory["revision_id"]
    assert second_memory["embedding_count"] >= first_memory["embedding_count"]
    second_manifest = json.loads(
        Path(second_memory["manifest_path"]).read_text(encoding="utf-8")
    )
    assert second_manifest["previous_revision_id"] == first_memory["revision_id"]


def test_r14_unreviewable_resume_only_banks_different_player_candidates(
    tmp_path: Path,
) -> None:
    different = _candidate(
        tmp_path,
        "shot_0005_track_0001",
        np.asarray([[1, 0, 0], [0.9, 0.1, 0]], dtype=np.float32),
    )
    unreviewable = _candidate(
        tmp_path,
        "shot_0005_track_0002",
        np.asarray([[0, 1, 0], [0, 0.9, 0.1]], dtype=np.float32),
    )
    state = _state([different, unreviewable])
    positive_memory = tmp_path / "positive.json"
    positive_memory.write_text(json.dumps({"revision_id": "positive"}), encoding="utf-8")
    positive_sha = _sha(positive_memory)
    state["runtime"].update(
        {
            "memory_revision_path": str(positive_memory),
            "memory_revision_sha256": positive_sha,
        }
    )
    state["shots"][1]["status"] = "SEARCHING_NO_MEMORY"
    ambiguity_id = "ambiguity_phase4b_shot_0005_g005"
    ambiguity_dir = tmp_path / "ambiguities"
    decision_dir = tmp_path / "review_decisions"
    ambiguity_dir.mkdir()
    decision_dir.mkdir()
    (ambiguity_dir / f"{ambiguity_id}.json").write_text(
        json.dumps(
            {
                "ambiguity_id": ambiguity_id,
                "shot_id": "shot_0005",
                "candidates": [different, unreviewable],
            }
        ),
        encoding="utf-8",
    )
    first_decision = decision_dir / "decision-a.json"
    first_decision.write_text(
        json.dumps(
            {
                "ambiguity_id": ambiguity_id,
                "candidate_id": different["candidate_id"],
                "state": "DIFFERENT_PLAYER",
            }
        ),
        encoding="utf-8",
    )
    current_decision = decision_dir / "decision-b.json"
    current_decision.write_text(
        json.dumps(
            {
                "ambiguity_id": ambiguity_id,
                "candidate_id": unreviewable["candidate_id"],
                "state": "UNREVIEWABLE_LOW_RESOLUTION",
            }
        ),
        encoding="utf-8",
    )

    memory = ADAPTER._phase4b_resume_rejected_candidate(
        output_dir=tmp_path,
        state=state,
        candidate_id=unreviewable["candidate_id"],
        decision_state="UNREVIEWABLE_LOW_RESOLUTION",
        decision_path=current_decision,
        decision_sha256=_sha(current_decision),
        reviewer="USER",
        note="too small",
    )

    assert memory is not None
    assert memory["rejected_candidate_ids"] == [different["candidate_id"]]
    assert unreviewable["candidate_id"] not in memory["rejected_candidate_ids"]
    assert state["runtime"]["memory_revision_sha256"] == positive_sha
    assert state["status"] == "RUNNING"
    assert state["runtime"]["phase4b_next_shot_id"] == "shot_0006"


def test_r14_last_rejected_candidate_completes_with_unresolved_gaps(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        tmp_path,
        "shot_0005_track_0001",
        np.asarray([[1, 0], [0.9, 0.1]], dtype=np.float32),
    )
    candidate["manifest_path"] = (
        f"storage/tracking_runtime/{tmp_path.name}/"
        f"{candidate['candidate_id']}/candidate_manifest.json"
    )
    state = _state([candidate])
    state["shots"] = state["shots"][:1]
    ambiguity_id = "ambiguity_phase4b_shot_0005_g005"
    (tmp_path / "ambiguities").mkdir()
    (tmp_path / "review_decisions").mkdir()
    (tmp_path / "ambiguities" / f"{ambiguity_id}.json").write_text(
        json.dumps(
            {
                "ambiguity_id": ambiguity_id,
                "shot_id": "shot_0005",
                "candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "review_decisions" / "decision-a.json"
    decision.write_text(
        json.dumps(
            {
                "ambiguity_id": ambiguity_id,
                "candidate_id": candidate["candidate_id"],
                "state": "DIFFERENT_PLAYER",
            }
        ),
        encoding="utf-8",
    )
    ADAPTER._phase4b_resume_rejected_candidate(
        output_dir=tmp_path,
        state=state,
        candidate_id=candidate["candidate_id"],
        decision_state="DIFFERENT_PLAYER",
        decision_path=decision,
        decision_sha256=_sha(decision),
        reviewer="USER",
        note="different",
    )
    assert state["status"] == "COMPLETE_WITH_UNRESOLVED_GAPS"
    assert state["pending_action"] is None


def test_none_of_these_rejects_manifest_sha_mismatch(tmp_path: Path) -> None:
    candidate = _candidate(
        tmp_path,
        "shot_0005_track_0001",
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    candidate["manifest_sha256"] = "0" * 64
    state = _state([candidate])
    with pytest.raises(ValueError):
        ADAPTER._phase4b_apply_none_of_these_review(
            output_dir=tmp_path,
            state=state,
            ambiguity_id="ambiguity_phase4b_shot_0005_g005",
            reviewer="USER",
            note="bad hash",
        )


def test_runtime_sources_wire_none_of_these_to_adapter() -> None:
    tracking_schema = TRACKING_SCHEMA_PATH.read_text(encoding="utf-8")
    tracking_service = TRACKING_SERVICE_PATH.read_text(encoding="utf-8")
    candidate_service = CANDIDATE_SERVICE_PATH.read_text(encoding="utf-8")
    process_runner = PROCESS_RUNNER_PATH.read_text(encoding="utf-8")
    adapter = ADAPTER_PATH.read_text(encoding="utf-8")

    assert 'NONE_OF_THESE = "none_of_these"' in tracking_schema
    assert '"kind": "none_of_these"' in candidate_service
    assert 'action_kind = {' in tracking_service
    assert 'elif kind == "none_of_these":' in process_runner
    assert '"--reject-all-candidates"' in process_runner
    assert 'args.resume and args.reject_all_candidates' in adapter
    assert "USER_REJECTED_CROSS_SHOT_CANDIDATES_R1" in adapter
