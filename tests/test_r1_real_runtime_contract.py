from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.domains.candidate_handoff_r1.r3_adapter import R1R3InputAdapter
from app.domains.candidate_handoff_r1.runtime.phase1_compatibility_executor import (
    classify_stage0_findings,
)
from app.domains.candidate_handoff_r1.runtime_sync import (
    R1RuntimeSyncError,
    completion_state_from_runtime,
    extract_pending_ambiguity,
    timeline_metrics,
)
from app.domains.tracking.process_runner import TrackingProcessRunner
from app.domains.tracking.status import TrackingBackendStatus


FORBIDDEN = (
    "R1_SEARCH_QUEUE",
    "shot_0004_track_0001",
    "shot_0002_track_0034",
    "shot_0003_track_0003",
    "shot_0004_track_0003",
    "target_centric_tracking_v2_production_r3",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return digest(path)


def test_production_sources_have_no_scripted_queue_or_fixture_literals() -> None:
    root = Path(__file__).parents[1] / "app"
    production = [
        root / "domains" / "candidate_handoff_r1",
        root / "domains" / "tracking",
    ]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in production
        for path in directory.rglob("*.py")
    )
    for literal in FORBIDDEN:
        assert literal not in text


def test_stage0_warning_allowlist_is_fail_closed() -> None:
    allowed, blocked = classify_stage0_findings(
        {
            "findings": [
                {"severity": "WARNING", "code": "DURATION_OUTSIDE_RANGE"},
                {"severity": "WARNING", "code": "V6_OUTPUT_NOT_FOUND"},
                {"severity": "WARNING", "code": "V7_OUTPUT_NOT_FOUND"},
            ]
        },
        requested_device="cpu",
    )
    assert allowed == [
        "DURATION_OUTSIDE_RANGE",
        "V6_OUTPUT_NOT_FOUND",
        "V7_OUTPUT_NOT_FOUND",
    ]
    assert blocked == []

    _, blocked = classify_stage0_findings(
        {
            "findings": [
                {"severity": "WARNING", "code": "UNKNOWN_WARNING"},
                {"severity": "WARNING", "code": "RFDETR_HASH_MISMATCH"},
                {"severity": "WARNING", "code": "BBOX_OUTSIDE_FRAME"},
            ]
        },
        requested_device="cpu",
    )
    assert blocked == [
        "BBOX_OUTSIDE_FRAME",
        "RFDETR_HASH_MISMATCH",
        "UNKNOWN_WARNING",
    ]


def test_cuda_warning_is_blocked_when_cuda_was_explicitly_requested() -> None:
    allowed, blocked = classify_stage0_findings(
        {"findings": [{"severity": "WARNING", "code": "CUDA_NOT_AVAILABLE"}]},
        requested_device="cuda",
    )
    assert allowed == []
    assert blocked == ["CUDA_NOT_AVAILABLE"]


class FakeStorage:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.storage_root = project_root

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def test_r1_adapter_preserves_provenance_and_forbids_frame_zero_fallback(tmp_path: Path) -> None:
    discovery = tmp_path / "discovery"
    bundle = discovery / "candidate_bundles" / "dynamic_candidate_alpha"
    bundle.mkdir(parents=True)
    refs = []
    for index, scale in enumerate(("wide", "medium", "close-up"), start=1):
        crop = bundle / f"ref_{index}.jpg"
        crop.write_bytes(f"crop-{index}".encode())
        refs.append(
            {
                "frame": 340 + index,
                "path": crop.name,
                "crop_sha256": digest(crop),
                "scale_class": scale,
            }
        )
    manifest = {
        "candidate_id": "dynamic_candidate_alpha",
        "candidate_media_id": "dynamic_candidate_alpha",
        "shot_id": "shot_dynamic",
        "tracklet_id": "track_dynamic",
        "best_observation": {
            "candidate_id": "dynamic_candidate_alpha",
            "frame": 342,
            "bbox_xyxy": [1, 2, 30, 40],
        },
        "quality": {"identity_pure": True, "reviewability": "REVIEWABLE"},
        "reference_gallery": refs,
    }
    manifest_path = bundle / "manifest.json"
    manifest_sha = write_json(manifest_path, manifest)
    boundaries_path = discovery / "reviewed_shot_boundaries.json"
    boundaries_sha = write_json(
        boundaries_path,
        {
            "shots": [
                {
                    "shot_id": "shot_dynamic",
                    "start_frame": 300,
                    "end_frame_inclusive": 344,
                }
            ]
        },
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    selection_record = tmp_path / "selection.json"
    selection_record_sha = write_json(selection_record, {"immutable": True})
    scene = tmp_path / "scene_manifest.json"
    r2 = tmp_path / "r2_manifest.json"
    r3 = tmp_path / "r3_manifest.json"
    settings = SimpleNamespace(
        SCENE_TARGET_SELECTION_MANIFEST_PATH=str(scene),
        SCENE_TARGET_SELECTION_MANIFEST_SHA256=write_json(scene, {"scene": True}),
        TRACKING_R2_MANIFEST_PATH=str(r2),
        TRACKING_R2_MANIFEST_SHA256=write_json(r2, {"r2": True}),
        TRACKING_R3_MANIFEST_PATH=str(r3),
        TRACKING_R3_MANIFEST_SHA256=write_json(r3, {"adapter": True}),
    )
    selection = SimpleNamespace(
        selection_id="selection_dynamic",
        ranking_id="ranking_dynamic",
        shortlist_patch_id="patch_v1_2",
        discovery_id="discovery_dynamic",
        candidate_id="dynamic_candidate_alpha",
        shot_id="shot_dynamic",
        tracklet_id="track_dynamic",
        candidate_media_bundle_sha256=manifest_sha,
        source_video_sha256=digest(video),
        reviewed_shot_boundaries_sha256=boundaries_sha,
        selection_artifact_path=str(selection_record),
        selection_artifact_sha256=selection_record_sha,
    )
    output = tmp_path / "output"
    output.mkdir()
    job = SimpleNamespace(
        output_directory=str(output),
        tracking_job_id="job_dynamic",
        test_name="test_dynamic",
        runtime_metadata={},
    )
    adapter = R1R3InputAdapter(settings=settings, storage=FakeStorage(tmp_path))
    result = adapter.build(
        job=job,
        selection=selection,
        candidate_manifest_path=manifest_path,
        source_video_path=video,
        reviewed_shot_boundaries_path=boundaries_path,
    )
    target = json.loads(result.target_selection_path.read_text())
    anchor = json.loads(result.earlier_anchor_decision_path.read_text())
    assert target["best_anchor_frame"] == 342
    assert target["selected_shot_start_frame"] == 300
    assert target["selected_candidate_id"] == "dynamic_candidate_alpha"
    assert anchor["frame_zero_fallback_used"] is False
    assert target["automatic_target_confirmation"] is False
    adapter.attach_to_job(job, result)
    assert job.runtime_metadata["scene_target_selection"]["target_selection_sha256"] == result.target_selection_sha256


def _runner_settings(tmp_path: Path, script: Path) -> SimpleNamespace:
    python = Path(sys.executable).resolve()
    return SimpleNamespace(
        TRACKING_PROJECT_ROOT=str(tmp_path),
        TRACKING_PYTHON_EXECUTABLE=str(python),
        TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH=str(script),
        TRACKING_E2E_SCRIPT_PATH=str(script),
        TRACKING_OUTPUT_ROOT=str(tmp_path / "out"),
        TRACKING_PREVIEW_ENABLED=True,
        TRACKING_PROCESS_TIMEOUT_SECONDS=30,
    )


def test_process_runner_requires_explicit_adapter_path(tmp_path: Path) -> None:
    settings = _runner_settings(tmp_path, tmp_path / "missing.py")
    settings.TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH = ""
    runner = TrackingProcessRunner(settings)
    job = SimpleNamespace(runtime_metadata={"scene_target_selection": {}})
    with pytest.raises(Exception, match="no production_r3"):
        runner._runner_script(job)


def test_resume_commands_cover_confirm_absent_reject_and_unreviewable(tmp_path: Path) -> None:
    script = tmp_path / "adapter.py"
    script.write_text("print('adapter')\n")
    out = tmp_path / "out"
    out.mkdir()
    immutable = tmp_path / "immutable"
    immutable.mkdir()
    files = {}
    for key in (
        "tracking_launch_manifest_path",
        "shot_boundaries_path",
        "target_selection_path",
        "target_reference_set_path",
        "earlier_anchor_decision_path",
    ):
        path = immutable / f"{key}.json"
        path.write_text("{}\n")
        files[key] = str(path)
    scene = {
        "selection_artifact_root": str(immutable),
        **files,
        "tracking_launch_manifest_sha256": digest(Path(files["tracking_launch_manifest_path"])),
        "shot_boundaries_sha256": digest(Path(files["shot_boundaries_path"])),
        "target_selection_sha256": digest(Path(files["target_selection_path"])),
        "target_reference_set_sha256": digest(Path(files["target_reference_set_path"])),
        "earlier_anchor_decision_sha256": digest(Path(files["earlier_anchor_decision_path"])),
        "candidate_scoring_generation": 2,
    }
    job = SimpleNamespace(
        runtime_metadata={"scene_target_selection": scene},
        test_name="contract",
        device="cpu",
        reacquisition_mode="assisted",
    )
    runner = TrackingProcessRunner(_runner_settings(tmp_path, script))
    confirmed = runner.build_resume_command(
        job,
        {"kind": "ambiguity", "ambiguity_id": "amb", "candidate_id": "dyn", "decision": "confirmed"},
    )
    absent = runner.build_resume_command(
        job,
        {"kind": "ambiguity", "ambiguity_id": "amb", "decision": "absent"},
    )
    rejected = runner.build_resume_command(
        job,
        {"kind": "candidate_rejected", "ambiguity_id": "amb", "candidate_id": "dyn"},
    )
    unreviewable = runner.build_resume_command(
        job,
        {"kind": "candidate_unreviewable", "ambiguity_id": "amb", "candidate_id": "dyn"},
    )
    assert "--confirmed-candidate" in confirmed
    assert "--confirm-absent" in absent
    assert "--rejected-candidate" in rejected
    assert "--unreviewable-candidate" in unreviewable
    assert "--candidate-scoring-generation" in confirmed


def test_rejected_candidate_decision_artifact_resolves_from_backend_storage(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "adapter.py"
    script.write_text("print('adapter')\n")
    immutable = tmp_path / "immutable"
    immutable.mkdir()
    paths = {}
    for key in (
        "tracking_launch_manifest_path",
        "shot_boundaries_path",
        "target_selection_path",
        "target_reference_set_path",
        "earlier_anchor_decision_path",
    ):
        path = immutable / f"{key}.json"
        path.write_text("{}\n")
        paths[key] = str(path)
    scene = {
        "selection_artifact_root": str(immutable),
        **paths,
        **{
            key.replace("_path", "_sha256"): digest(Path(value))
            for key, value in paths.items()
        },
        "candidate_scoring_generation": 2,
    }
    decision = tmp_path / "review-decision.json"
    decision.write_text('{"state":"DIFFERENT_PLAYER"}\n')
    monkeypatch.setattr(
        "app.domains.tracking.process_runner.LocalStorage",
        lambda: type(
            "Storage",
            (),
            {"resolve_path": lambda self, value: decision},
        )(),
    )
    job = SimpleNamespace(
        runtime_metadata={"scene_target_selection": scene},
        test_name="rejection_resume",
        device="cpu",
        reacquisition_mode="assisted",
    )
    command = TrackingProcessRunner(_runner_settings(tmp_path, script)).build_resume_command(
        job,
        {
            "kind": "candidate_rejected",
            "ambiguity_id": "amb",
            "candidate_id": "candidate-final",
            "decision_artifact_path": "storage/review-decision.json",
            "decision_artifact_sha256": digest(decision),
        },
    )
    assert command[command.index("--review-decision-artifact") + 1] == str(
        decision.resolve()
    )


def test_empty_dynamic_ambiguity_is_forbidden() -> None:
    with pytest.raises(R1RuntimeSyncError, match="EMPTY_PENDING_AMBIGUITY_FORBIDDEN"):
        extract_pending_ambiguity(
            {
                "pending_action": {
                    "type": "CROSS_SHOT_CONFIRMATION",
                    "ambiguity_id": "amb",
                },
                "ambiguities": [
                    {"ambiguity_id": "amb", "review_candidates": []}
                ],
            }
        )


def test_completion_gate_distinguishes_unresolved_and_safe_block() -> None:
    assert completion_state_from_runtime(
        {
            "status": "NEEDS_CONFIRMATION",
            "pending_action": {"type": "MEMORY_REVIEW"},
            "shots": [{"status": "INITIAL_TARGET_TRACKING_ONLY"}],
        },
        timeline_valid=True,
        preview_generated=False,
        processing_outbox_count=0,
    ) == TrackingBackendStatus.WAITING_MEMORY_REVIEW
    assert completion_state_from_runtime(
        {"status": "COMPLETE", "pending_action": None, "shots": [{"status": "ACCEPTED"}]},
        timeline_valid=True,
        preview_generated=True,
        processing_outbox_count=0,
    ) == TrackingBackendStatus.COMPLETED
    assert completion_state_from_runtime(
        {"status": "COMPLETE", "pending_action": None, "shots": [{"status": "UNRESOLVED_LOW_RESOLUTION"}]},
        timeline_valid=True,
        preview_generated=True,
        processing_outbox_count=0,
    ) == TrackingBackendStatus.COMPLETED_WITH_UNRESOLVED_GAPS
    assert completion_state_from_runtime(
        {"status": "COMPLETE", "pending_action": None, "shots": [{"status": "ACCEPTED"}]},
        timeline_valid=False,
        preview_generated=False,
        processing_outbox_count=1,
    ) == TrackingBackendStatus.COMPLETED_SAFE_BLOCK


def test_timeline_requires_real_bbox_for_active_and_null_for_searching(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    write_json(
        valid,
        {
            "frames": [
                {"frame_index": 0, "state": "ACTIVE", "bbox_xyxy": [1, 1, 2, 2]},
                {"frame_index": 1, "state": "SEARCHING", "bbox_xyxy": None},
                {"frame_index": 2, "state": "REACQUIRED", "bbox_xyxy": [2, 2, 3, 3]},
            ],
            "provenance": {"silent_wrong_player_switches": 0},
        },
    )
    metrics = timeline_metrics(valid)
    assert metrics["active_or_reacquired_bbox_frames"] == 2
    invalid = tmp_path / "invalid.json"
    write_json(
        invalid,
        {"frames": [{"frame_index": 0, "state": "SEARCHING", "bbox_xyxy": [1, 1, 2, 2]}]},
    )
    with pytest.raises(R1RuntimeSyncError):
        timeline_metrics(invalid)


def test_r3_subprocess_contract_confirmed_candidate_adds_frames(tmp_path: Path) -> None:
    """R3_SUBPROCESS_CONTRACT_TEST only; this is not a frozen AI performance PASS."""
    script = Path(__file__).parent / "fixtures" / "fake_r3_contract_cli.py"
    root = tmp_path / "runtime"
    root.mkdir()
    output = tmp_path / "out"
    base = [
        sys.executable,
        str(script),
        "--project-root",
        str(root),
        "--test-name",
        "contract_job",
        "--output-root",
        str(output),
        "--device",
        "cpu",
        "--reacquisition-mode",
        "assisted",
    ]
    initial = subprocess.run(
        base + ["--video", str(tmp_path / "video.mp4"), "--initial-bbox", "1", "1", "2", "2"],
        check=False,
    )
    assert initial.returncode == 3
    state_path = output / "contract_job" / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    pending = extract_pending_ambiguity(state)
    candidate_id = pending["candidates"][0]["candidate_id"]
    before = timeline_metrics(output / "contract_job" / "target_timeline.json")
    resumed = subprocess.run(
        base
        + [
            "--resume",
            "--ambiguity-id",
            pending["ambiguity_id"],
            "--confirmed-candidate",
            candidate_id,
        ],
        check=False,
    )
    assert resumed.returncode == 0
    after = timeline_metrics(output / "contract_job" / "target_timeline.json")
    assert before["active_or_reacquired_bbox_frames"] == 45
    assert after["active_or_reacquired_bbox_frames"] == 65
    final = json.loads(state_path.read_text())
    assert final["status"] == "COMPLETE"
    assert final["runtime"]["full_event_recommendation_e2e"] == "NOT_RUN"


def test_executor_source_separates_legacy_and_r1_claims() -> None:
    root = Path(__file__).parents[1] / "app" / "domains" / "tracking"
    legacy = (root / "executor.py").read_text()
    r1 = (root / "r1_executor.py").read_text()
    repository = (root / "repository.py").read_text()
    assert "execution_kind: str = LEGACY_EXECUTION_KIND" in legacy
    assert "execution_kind=R1_EXECUTION_KIND" in r1
    assert "TrackingJob.execution_kind == execution_kind" in repository


def test_observation_copy_is_bootstrap_only_not_public_tracking_success() -> None:
    service = (
        Path(__file__).parents[1]
        / "app"
        / "domains"
        / "candidate_handoff_r1"
        / "service.py"
    ).read_text()
    assert "PROVISIONAL_ACTIVE" not in service
    assert "INTERPOLATED_WITHIN_PURE_TRACKLET" not in service
    assert '"tracking_success": False' in service
    assert 'timeline_path=None' in service


def test_memory_revision_is_passed_to_runtime_command(tmp_path: Path) -> None:
    script = tmp_path / "adapter.py"
    script.write_text("print('adapter')\n")
    immutable = tmp_path / "immutable"
    immutable.mkdir()
    paths = {}
    for key in (
        "tracking_launch_manifest_path",
        "shot_boundaries_path",
        "target_selection_path",
        "target_reference_set_path",
        "earlier_anchor_decision_path",
    ):
        path = immutable / f"{key}.json"
        path.write_text("{}\n")
        paths[key] = str(path)
    memory = immutable / "memory.json"
    memory.write_text('{"references": [1,2,3]}\n')
    scene = {
        "selection_artifact_root": str(immutable),
        **paths,
        "tracking_launch_manifest_sha256": digest(Path(paths["tracking_launch_manifest_path"])),
        "shot_boundaries_sha256": digest(Path(paths["shot_boundaries_path"])),
        "target_selection_sha256": digest(Path(paths["target_selection_path"])),
        "target_reference_set_sha256": digest(Path(paths["target_reference_set_path"])),
        "earlier_anchor_decision_sha256": digest(Path(paths["earlier_anchor_decision_path"])),
        "current_target_memory_path": str(memory),
        "current_target_memory_sha256": digest(memory),
        "candidate_scoring_generation": 4,
    }
    job = SimpleNamespace(
        runtime_metadata={"scene_target_selection": scene},
        test_name="memory_contract",
        device="cpu",
        reacquisition_mode="assisted",
    )
    command = TrackingProcessRunner(_runner_settings(tmp_path, script)).build_resume_command(
        job,
        {"kind": "ambiguity", "ambiguity_id": "amb", "candidate_id": "dynamic", "decision": "confirmed"},
    )
    assert command[command.index("--target-memory-revision") + 1] == str(memory)
    assert command[command.index("--target-memory-sha256") + 1] == digest(memory)
    assert command[command.index("--candidate-scoring-generation") + 1] == "4"


def test_different_player_does_not_mark_whole_shot_absent() -> None:
    service = (
        Path(__file__).parents[1]
        / "app"
        / "domains"
        / "candidate_handoff_r1"
        / "service.py"
    ).read_text()
    different_block = service.split(
        "elif request.state == CandidateReviewState.TARGET_ABSENT:", 1
    )[1]
    assert '"candidate_rejected"' in different_block
    assert "CANDIDATE_REJECTED_SHOT_NOT_MARKED_ABSENT" not in service
    assert "summary.setdefault(\"shot_absent_ids\"" in service
    target_absent_section = service.split(
        "elif request.state == CandidateReviewState.TARGET_ABSENT:", 1
    )[1].split("elif remaining:", 1)[0]
    assert "shot_absent_ids" in target_absent_section


def test_unreviewable_is_not_converted_to_target_absent() -> None:
    service = (
        Path(__file__).parents[1]
        / "app"
        / "domains"
        / "candidate_handoff_r1"
        / "service.py"
    ).read_text()
    assert '"UNRESOLVED_LOW_RESOLUTION"' in service
    assert '"candidate_unreviewable"' in service
    assert "result_state = \"AMBIGUOUS\"" not in service


def test_adapter_cli_materializes_selection_anchor_view_without_frame_zero_fallback(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    research = tmp_path / "research"
    required = [
        research / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        research / "target_centric_tracking_e2e_v1" / "verify_e2e_installation.py",
        research / "target_centric_tracking_v1" / "run_phase1_frozen_pipeline.py",
        research / "target_centric_tracking_v1" / "phase1_frozen_manifest.json",
        research / "target_centric_tracking_v1" / "stage0_audit_inputs.py",
        research / "target_centric_tracking_v1" / "stage1_generate_rfdetr_detections.py",
        research / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
        research / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        research / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        research / "target_centric_tracking_v2" / "stage3b2_make_safe_cross_shot_decision.py",
        research / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        research / "target_centric_tracking_v2" / "stage3c0_run_short_clip_phase1_compatibility.py",
        research / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "phase1_frozen_manifest.json":
            write_json(path, {"models": {}})
        elif path.name == "stage3b3_confirm_user_selected_cross_shot_anchor.py":
            path.write_text(
                "def choose_anchor(rows):\n"
                "    row = rows[0]\n"
                "    return row, [row]\n",
                encoding="utf-8",
            )
        elif path.name == "run_target_centric_pipeline.py":
            path.write_text(
                "import argparse,json\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser(); p.add_argument('--project-root'); p.add_argument('--video'); "
                "p.add_argument('--test-name',required=True); p.add_argument('--initial-bbox',nargs=4); "
                "p.add_argument('--device'); p.add_argument('--reacquisition-mode'); p.add_argument('--output-root',required=True); "
                "p.add_argument('--backend-safe-weights-runner'); "
                "p.add_argument('--cut-frames',nargs='*'); p.add_argument('--reviewer'); p.add_argument('--review-note'); "
                "p.add_argument('--overwrite',action='store_true'); p.add_argument('--no-preview',action='store_true'); a=p.parse_args(); "
                "o=Path(a.output_root)/a.test_name; o.mkdir(parents=True,exist_ok=True); "
                "json.dump({'status':'COMPLETE','decision':'DUMMY_REAL_SOURCE_CONTRACT','video':{'path':a.video},"
                "'shots':[{'shot_id':'shot_0000','status':'TARGET_CONFIRMED_AND_TRACKED'}],"
                "'ambiguities':[],'pending_action':None},open(o/'pipeline_state.json','w')); "
                "json.dump({'video':{},'frames':[{'frame_index':i,'state':'ACTIVE','bbox_xyxy':[1,1,2,2]} for i in range(5)]},open(o/'target_timeline.json','w')); "
                "json.dump({'status':'COMPLETE'},open(o/'pipeline_summary.json','w')); "
                "json.dump({'video':a.video,'cut_frames':a.cut_frames},open(o/'invocation.json','w'))\n",
                encoding="utf-8",
            )
        else:
            path.write_text("# contract source\n", encoding="utf-8")

    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
    writer.release()

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    target_selection = artifacts / "target_selection.json"
    target_reference = artifacts / "target_reference.json"
    anchor = artifacts / "anchor.json"
    boundaries = artifacts / "boundaries.json"
    write_json(
        target_selection,
        {
            "best_anchor_frame": 5,
            "selected_shot_start_frame": 2,
            "selected_shot_end_frame_inclusive": 6,
            "shot_id": "selected_dynamic_shot",
            "automatic_target_confirmation": False,
        },
    )
    write_json(target_reference, {"references": [1, 2, 3]})
    write_json(anchor, {"frame_zero_fallback_used": False})
    write_json(
        boundaries,
        {
            "shots": [
                {"shot_id": "selected_dynamic_shot", "start_frame": 2, "end_frame_inclusive": 6},
                {"shot_id": "next_dynamic_shot", "start_frame": 7, "end_frame_inclusive": 9},
            ]
        },
    )
    launch = artifacts / "launch.json"
    write_json(
        launch,
        {
            "source_video": {"path": str(video), "sha256": digest(video)},
            "target_selection": {"path": str(target_selection), "sha256": digest(target_selection)},
            "target_reference_set": {"path": str(target_reference), "sha256": digest(target_reference)},
            "earlier_anchor_decision": {"path": str(anchor), "sha256": digest(anchor)},
            "shot_boundaries": {"path": str(boundaries), "sha256": digest(boundaries)},
        },
    )
    output = tmp_path / "out"
    adapter = (
        Path(__file__).parents[1]
        / "app"
        / "domains"
        / "candidate_handoff_r1"
        / "runtime"
        / "r1_v1_v2_adapter_cli.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(adapter),
            "--project-root", str(research),
            "--video", str(video),
            "--test-name", "anchor_adapter",
            "--initial-bbox", "1", "2", "3", "4",
            "--device", "cpu",
            "--reacquisition-mode", "assisted",
            "--output-root", str(output),
            "--tracking-launch-manifest", str(launch),
            "--shot-boundaries", str(boundaries),
            "--target-selection", str(target_selection),
            "--target-reference-set", str(target_reference),
            "--earlier-anchor-decision", str(anchor),
            "--candidate-scoring-generation", "1",
            "--contract-test-skip-phase1-compatibility",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    job_dir = output / "anchor_adapter"
    view = json.loads((job_dir / "_selection_anchor_view" / "selection_view.json").read_text())
    assert view["source_offset_frame"] == 5
    assert view["unresolved_prefix_frame_count"] == 3
    assert view["frame_zero_fallback_used"] is False
    assert view["actual_anchor_materialized_as_view_frame_zero"] is True
    assert view["same_shot_pre_cut_source"]["source_start_frame"] == 5
    assert view["same_shot_pre_cut_source"]["source_end_frame_inclusive"] == 6
    assert view["same_shot_pre_cut_source"]["frame_count"] == 2
    assert view["cross_shot_search_source"]["source_start_frame"] == 7
    invocation = json.loads((job_dir / "_provided_e2e_output" / "anchor_adapter" / "invocation.json").read_text())
    assert Path(invocation["video"]).name == "selection_anchor_suffix.mp4"
    state = json.loads((job_dir / "pipeline_state.json").read_text())
    assert state["status"] == "COMPLETE_WITH_UNRESOLVED_GAPS"
    assert state["runtime"]["selection_aware_video_adapter_used"] is True
    timeline = json.loads((job_dir / "target_timeline.json").read_text())
    assert len(timeline["frames"]) == 10
    assert timeline["frames"][5]["state"] == "ACTIVE"
    assert timeline["frames"][0]["state"] == "SEARCHING"


def test_r3_subprocess_contract_absent_resume_advances_without_bbox_growth(tmp_path: Path) -> None:
    script = Path(__file__).parent / "fixtures" / "fake_r3_contract_cli.py"
    output = tmp_path / "out"
    base = [
        sys.executable,
        str(script),
        "--project-root",
        str(tmp_path),
        "--test-name",
        "absent_contract",
        "--output-root",
        str(output),
        "--device",
        "cpu",
        "--reacquisition-mode",
        "assisted",
    ]
    subprocess.run(base + ["--video", str(tmp_path / "v.mp4"), "--initial-bbox", "1", "1", "2", "2"], check=False)
    state_path = output / "absent_contract" / "pipeline_state.json"
    pending = extract_pending_ambiguity(json.loads(state_path.read_text()))
    before = timeline_metrics(output / "absent_contract" / "target_timeline.json")
    completed = subprocess.run(
        base + ["--resume", "--ambiguity-id", pending["ambiguity_id"], "--confirm-absent"],
        check=False,
    )
    assert completed.returncode == 0
    after = timeline_metrics(output / "absent_contract" / "target_timeline.json")
    state = json.loads(state_path.read_text())
    assert before["active_or_reacquired_bbox_frames"] == after["active_or_reacquired_bbox_frames"]
    assert state["shots"][1]["status"] == "ABSENT_CONFIRMED"


def test_r3_subprocess_contract_unreviewable_is_unresolved_not_absent(tmp_path: Path) -> None:
    script = Path(__file__).parent / "fixtures" / "fake_r3_contract_cli.py"
    output = tmp_path / "out"
    base = [sys.executable, str(script), "--project-root", str(tmp_path), "--test-name", "lowres_contract", "--output-root", str(output), "--device", "cpu", "--reacquisition-mode", "assisted"]
    subprocess.run(base + ["--video", str(tmp_path / "v.mp4"), "--initial-bbox", "1", "1", "2", "2"], check=False)
    state_path = output / "lowres_contract" / "pipeline_state.json"
    pending = extract_pending_ambiguity(json.loads(state_path.read_text()))
    candidate = pending["candidates"][0]["candidate_id"]
    completed = subprocess.run(base + ["--resume", "--ambiguity-id", pending["ambiguity_id"], "--unreviewable-candidate", candidate], check=False)
    assert completed.returncode == 0
    state = json.loads(state_path.read_text())
    assert state["status"] == "COMPLETE_WITH_UNRESOLVED_GAPS"
    assert state["shots"][1]["status"] == "UNRESOLVED_LOW_RESOLUTION"
    assert state["shots"][1]["status"] != "ABSENT_CONFIRMED"


def test_recovery_filters_execution_kind_and_preserves_db_generation() -> None:
    repository = (
        Path(__file__).parents[1]
        / "app"
        / "domains"
        / "tracking"
        / "repository.py"
    ).read_text()
    orchestrator = (
        Path(__file__).parents[1]
        / "app"
        / "domains"
        / "candidate_handoff_r1"
        / "orchestrator.py"
    ).read_text()
    assert "list_recoverable" in repository
    assert "TrackingJob.execution_kind == execution_kind" in repository
    assert "event_candidate_pipelines_r1" in orchestrator
    assert "never overwrite runtime pipeline_state.json" in orchestrator
    assert "pipeline.generation = max(1" in orchestrator


def test_confirmed_resume_safe_blocks_when_new_memory_cannot_drive_scoring(tmp_path: Path) -> None:
    research = tmp_path / "research"
    required = [
        research / "target_centric_tracking_e2e_v1" / "run_target_centric_pipeline.py",
        research / "target_centric_tracking_e2e_v1" / "verify_e2e_installation.py",
        research / "target_centric_tracking_v1" / "run_phase1_frozen_pipeline.py",
        research / "target_centric_tracking_v1" / "phase1_frozen_manifest.json",
        research / "target_centric_tracking_v1" / "stage0_audit_inputs.py",
        research / "target_centric_tracking_v1" / "stage1_generate_rfdetr_detections.py",
        research / "target_centric_tracking_v1" / "stage2_run_conservative_target_association.py",
        research / "target_centric_tracking_v2" / "stage3b0_build_postcut_candidate_tracklets.py",
        research / "target_centric_tracking_v2" / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
        research / "target_centric_tracking_v2" / "stage3b2_make_safe_cross_shot_decision.py",
        research / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py",
        research / "target_centric_tracking_v2" / "stage3c0_run_short_clip_phase1_compatibility.py",
        research / "global_ID_tracking_upgrade_v6" / "stage2b1_extract_frozen_tracking_reid_embeddings_v6.py",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"models": {}}) if path.name == "phase1_frozen_manifest.json" else path.write_text("# source\n")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    selection = artifacts / "selection.json"
    references = artifacts / "references.json"
    anchor = artifacts / "anchor.json"
    boundaries = artifacts / "boundaries.json"
    memory = artifacts / "memory.json"
    write_json(selection, {"automatic_target_confirmation": False})
    write_json(references, {"references": [1, 2, 3]})
    write_json(anchor, {"frame_zero_fallback_used": False})
    write_json(boundaries, {"shots": [{"shot_id": "dynamic", "start_frame": 0, "end_frame_inclusive": 1}]})
    write_json(memory, {"revision_id": "memory_0002", "references": [1, 2, 3, 4]})
    launch = artifacts / "launch.json"
    write_json(
        launch,
        {
            "target_selection": {"path": str(selection), "sha256": digest(selection)},
            "target_reference_set": {"path": str(references), "sha256": digest(references)},
            "earlier_anchor_decision": {"path": str(anchor), "sha256": digest(anchor)},
            "shot_boundaries": {"path": str(boundaries), "sha256": digest(boundaries)},
        },
    )
    adapter = Path(__file__).parents[1] / "app" / "domains" / "candidate_handoff_r1" / "runtime" / "r1_v1_v2_adapter_cli.py"
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable, str(adapter),
            "--project-root", str(research),
            "--test-name", "memory_gate",
            "--output-root", str(output),
            "--tracking-launch-manifest", str(launch),
            "--shot-boundaries", str(boundaries),
            "--target-selection", str(selection),
            "--target-reference-set", str(references),
            "--earlier-anchor-decision", str(anchor),
            "--target-memory-revision", str(memory),
            "--target-memory-sha256", digest(memory),
            "--candidate-scoring-generation", "2",
            "--resume", "--ambiguity-id", "dynamic_ambiguity",
            "--confirmed-candidate", "dynamic_candidate",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    state = json.loads((output / "memory_gate" / "pipeline_state.json").read_text())
    assert state["failure_code"] == "RUNTIME_MEMORY_UPDATE_UNSUPPORTED"
    assert state["runtime"]["backend_memory_used_by_provided_e2e_scoring"] is False
    assert state["runtime"]["candidate_scoring_generation"] == 2


def test_runtime_candidate_evidence_uses_actual_assignments_and_original_frames(tmp_path: Path) -> None:
    import cv2
    import numpy as np
    from app.domains.candidate_handoff_r1.runtime import r1_v1_v2_adapter_cli as cli

    root = tmp_path / "research"
    anchor_helper = root / "target_centric_tracking_v2" / "stage3b3_confirm_user_selected_cross_shot_anchor.py"
    anchor_helper.parent.mkdir(parents=True)
    anchor_helper.write_text(
        "def choose_anchor(rows):\n"
        "    row=max(rows,key=lambda item: float(item['confidence']))\n"
        "    return row, rows\n",
        encoding="utf-8",
    )
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((24, 32, 3), index * 8, dtype=np.uint8))
    writer.release()
    raw_dir = tmp_path / "raw"
    strip = raw_dir / "work" / "shots" / "shot_0001" / "candidate_strips" / "dynamic_track.jpg"
    strip.parent.mkdir(parents=True)
    assert cv2.imwrite(str(strip), np.zeros((20, 20, 3), dtype=np.uint8))
    assignments = raw_dir / "assignments.csv"
    assignments.write_text(
        "candidate_id,frame_index,detection_id,confidence,x1,y1,x2,y2\n"
        "dynamic_track,2,det1,0.7,1,2,10,20\n"
        "dynamic_track,3,det2,0.9,2,3,11,21\n",
        encoding="utf-8",
    )
    view = {
        "source_video": {**cli._video_metadata(video)},
        "source_offset_frame": 5,
        "shot_map": [
            {
                "raw_shot_id": "shot_0001",
                "raw_shot_index": 1,
                "source_shot_id": "reviewed_next_shot",
                "source_start_frame": 7,
                "source_end_frame_inclusive": 11,
                "processed_source_start_frame": 7,
                "local_start_frame": 2,
                "local_end_frame_inclusive": 6,
            }
        ],
    }
    memory = tmp_path / "memory.json"
    write_json(memory, {"revision_id": "memory_2", "references": [1, 2, 3, 4]})
    evidence = cli._candidate_evidence(
        root=root,
        output_dir=tmp_path / "normalized",
        raw_dir=raw_dir,
        raw_state={"memory": {"target_embeddings": "actual_frozen_gallery.npy"}},
        view=view,
        ambiguity={
            "ambiguity_id": "ambiguity_dynamic",
            "shot_id": "shot_0001",
            "assignments": str(assignments),
            "contact_sheet": str(strip),
        },
        candidate={
            "candidate_id": "dynamic_track",
            "start_frame": 2,
            "end_frame_inclusive": 3,
            "retrieval_score": 0.75,
        },
        memory_path=memory,
        memory_sha=digest(memory),
        generation=2,
    )
    assert evidence["shot_id"] == "reviewed_next_shot"
    assert evidence["best_frame"] == 8
    assert Path(evidence["manifest_path"]).is_file()
    manifest = json.loads(Path(evidence["manifest_path"]).read_text())
    assert [row["frame_index"] for row in manifest["observations"]] == [7, 8]
    assert manifest["score_evidence"]["backend_memory_reference_count"] == 4
    assert manifest["score_evidence"]["backend_memory_used_by_provided_e2e_scoring"] is False
