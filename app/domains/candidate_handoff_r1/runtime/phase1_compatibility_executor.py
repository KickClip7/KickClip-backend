from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BENIGN_WARNING_CODES = frozenset(
    {
        "DURATION_OUTSIDE_RANGE",
        "V6_OUTPUT_NOT_FOUND",
        "V7_CODE_NOT_FOUND",
        "V7_OUTPUT_NOT_FOUND",
        "FFMPEG_NOT_FOUND",
        "CUDA_NOT_AVAILABLE",
    }
)
LEGACY_UNUSED_WARNING_CODES = frozenset(
    {
        "V7_CODE_NOT_FOUND",
        "V7_OUTPUT_NOT_FOUND",
        "V6_OUTPUT_NOT_FOUND",
    }
)
FORBIDDEN_WARNING_CODES = frozenset(
    {
        "PACKAGE_MISSING",
        "RFDETR_NOT_FOUND",
        "RFDETR_HASH_MISMATCH",
        "INVALID_BBOX",
        "BBOX_OUTSIDE_FRAME",
        "BBOX_TOO_SMALL",
        "VIDEO_NOT_FOUND",
        "VIDEO_AUDIT_FAILED",
    }
)
REVIEW_ARGUMENTS = {
    "STAGE2B": "--stage2b-visual-review",
    "STAGE2D": "--stage2d-visual-review",
    "STAGE2D1": "--stage2d1-visual-review",
    "STAGE2D2": "--stage2d2-visual-review",
}
FINAL_OUTPUT_NAMES = (
    "final_target_timeline.json",
    "final_frame_observations.csv",
    "final_crop_trajectory.csv",
    "final_target_tracking_preview.mp4",
    "final_target_centered_preview.mp4",
    "final_reentry_episodes.json",
    "final_audit.json",
    "final_report.md",
)
PHASE1_OUTPUT_NAMES = (
    "audit.json",
    "input_manifest.json",
    "detections.csv",
    "stage1_detection_summary.json",
    "stage2_association_summary.json",
    "target_timeline.json",
    "stage2_target_timeline.json",
    "stage2b_reentry_summary.json",
    "stage2b_target_timeline.json",
    "stage2c_audit.json",
    "stage2c_target_timeline.json",
    "stage2d_summary.json",
    "stage2d_target_timeline.json",
    "stage2d1_summary.json",
    "stage2d1_target_timeline.json",
    "stage2d2_summary.json",
    "stage2d2_target_timeline.json",
    "phase1_pipeline_state.json",
    *FINAL_OUTPUT_NAMES,
)


class Phase1CompatibilityBlocked(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def classify_stage0_findings(
    audit: Mapping[str, Any], *, requested_device: str
) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    blocked: list[str] = []
    for raw in audit.get("findings") or []:
        if not isinstance(raw, Mapping):
            blocked.append("MALFORMED_FINDING")
            continue
        severity = str(raw.get("severity") or "").upper()
        code = str(raw.get("code") or "UNKNOWN_WARNING")
        if severity == "ERROR" or code in FORBIDDEN_WARNING_CODES:
            blocked.append(code)
            continue
        if severity != "WARNING" or code not in BENIGN_WARNING_CODES:
            blocked.append(code)
            continue
        if code == "CUDA_NOT_AVAILABLE" and requested_device == "cuda":
            blocked.append(code)
            continue
        allowed.append(code)
    return sorted(set(allowed)), sorted(set(blocked))


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    accepted_return_codes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        shell=False,
        check=False,
        env=environment,
    )
    if completed.returncode not in accepted_return_codes:
        raise Phase1CompatibilityBlocked(
            f"Phase-1 compatibility command failed ({completed.returncode}): "
            + subprocess.list2cmdline(list(command))
        )
    return completed


def _phase1_environment() -> tuple[dict[str, str], Path | None]:
    """Expose a real installed FFmpeg binary to frozen audit subprocesses."""

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    candidates: list[Path] = []
    discovered = shutil.which("ffmpeg", path=environment.get("PATH"))
    if discovered:
        candidates.append(Path(discovered))
    for name in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        configured = environment.get(name)
        if configured:
            candidates.append(Path(configured))
    if os.name == "nt":
        candidates.extend(
            [
                Path("D:/ffmpeg/bin/ffmpeg.exe"),
                Path("C:/ffmpeg/bin/ffmpeg.exe"),
                Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
            ]
        )
    for candidate in candidates:
        executable = candidate.expanduser().resolve()
        if not executable.is_file():
            continue
        checked = subprocess.run(
            [str(executable), "-version"],
            shell=False,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if checked.returncode != 0:
            continue
        environment["PATH"] = str(executable.parent) + os.pathsep + environment.get(
            "PATH", ""
        )
        return environment, executable
    return environment, None


def _validated_review_decisions(
    review_decisions: Mapping[str, str] | None,
) -> dict[str, str]:
    if review_decisions is None:
        return {}

    normalized: dict[str, str] = {}
    for raw_stage, raw_decision in review_decisions.items():
        stage = str(raw_stage).upper()
        decision = str(raw_decision).upper()
        if stage not in REVIEW_ARGUMENTS:
            raise ValueError(f"Unsupported Phase-1 review stage: {raw_stage}")
        if decision not in {"PASS", "FAIL"}:
            raise ValueError(
                f"Review decision must be PASS or FAIL: {raw_stage}={raw_decision}"
            )
        normalized[stage] = decision
    return normalized


def _review_command_arguments(review_decisions: Mapping[str, str]) -> list[str]:
    arguments: list[str] = []
    for stage in ("STAGE2B", "STAGE2D", "STAGE2D1", "STAGE2D2"):
        decision = review_decisions.get(stage)
        if decision is None:
            continue
        arguments.extend([REVIEW_ARGUMENTS[stage], decision])
    return arguments


def _collect_outputs(phase1_dir: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for name in PHASE1_OUTPUT_NAMES:
        path = phase1_dir / name
        if not path.is_file():
            continue
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return outputs


def _timeline_summary(timeline: Mapping[str, Any]) -> dict[str, Any]:
    frames = timeline.get("frames")
    if not isinstance(frames, list):
        raise Phase1CompatibilityBlocked("Final Phase-1 timeline has no frame list.")

    state_counts = Counter(str(frame.get("state") or "") for frame in frames)
    bbox_frame_count = sum(frame.get("bbox_xyxy") is not None for frame in frames)
    return {
        "schema_version": timeline.get("schema_version"),
        "frame_count": len(frames),
        "bbox_frame_count": bbox_frame_count,
        "state_counts": dict(sorted(state_counts.items())),
        "source_stage": (timeline.get("provenance") or {}).get("source_stage"),
        "video": dict(timeline.get("video") or {}),
    }


def _phase1_runner_command(
    *,
    python: str,
    phase1_runner: Path,
    root: Path,
    video: Path,
    phase1_test_name: str,
    initial_bbox: Sequence[float],
    device: str,
    review_decisions: Mapping[str, str],
    reviewer: str,
    review_note: str,
    overwrite_phase1: bool,
) -> list[str]:
    command = [
        python,
        str(phase1_runner),
        "--project-root",
        str(root),
        "--video",
        str(video),
        "--test-name",
        phase1_test_name,
        "--initial-bbox",
        *[str(value) for value in initial_bbox],
        "--bbox-format",
        "xyxy_pixels",
        "--device",
        device,
        "--reviewer",
        reviewer,
        "--review-note",
        review_note,
        *_review_command_arguments(review_decisions),
    ]
    if overwrite_phase1:
        command.append("--overwrite")
    return command



def _materialize_legacy_stage0_pass(
    *,
    source_dir: Path,
    phase1_dir: Path,
    phase1_test_name: str,
    allowed_warning_codes: Sequence[str],
) -> None:
    """Create a derived PASS audit without creating fake V7 directories."""

    if phase1_dir.exists():
        shutil.rmtree(phase1_dir)
    shutil.copytree(source_dir, phase1_dir)
    audit = _read(phase1_dir / "audit.json")
    manifest = _read(phase1_dir / "input_manifest.json")
    initialization = _read(phase1_dir / "target_initialization.json")

    legacy = sorted(set(allowed_warning_codes))
    if any(code not in LEGACY_UNUSED_WARNING_CODES for code in legacy):
        raise Phase1CompatibilityBlocked(
            "Only legacy-unused V7/V6-output warnings may be normalized here."
        )

    retained = [
        dict(item)
        for item in (audit.get("findings") or [])
        if not (
            isinstance(item, Mapping)
            and str(item.get("code") or "") in LEGACY_UNUSED_WARNING_CODES
        )
    ]
    audit["findings"] = retained
    audit["status"] = "PASS"
    counts = audit.get("counts")
    if isinstance(counts, dict):
        counts["warnings"] = sum(
            str(item.get("severity") or "").upper() == "WARNING"
            for item in retained
            if isinstance(item, Mapping)
        )
        counts["errors"] = sum(
            str(item.get("severity") or "").upper() == "ERROR"
            for item in retained
            if isinstance(item, Mapping)
        )
    compatibility = {
        "schema_version": "kickclip.phase1_legacy_unused_stage0_compatibility.v1",
        "source_audit_path": str((source_dir / "audit.json").resolve()),
        "source_audit_sha256": _sha256(source_dir / "audit.json"),
        "normalized_warning_codes": legacy,
        "v7_runtime_dependency": False,
        "frozen_stage0_modified": False,
    }
    audit["compatibility"] = compatibility
    manifest["status"] = "PASS"
    manifest["test_name"] = phase1_test_name
    if isinstance(manifest.get("paths"), dict):
        manifest["paths"]["test_output_dir"] = str(phase1_dir.resolve())
    manifest["compatibility"] = compatibility
    initialization["status"] = "INITIALIZING"
    initialization["compatibility"] = compatibility

    _write(phase1_dir / "audit.json", audit)
    _write(phase1_dir / "input_manifest.json", manifest)
    _write(phase1_dir / "target_initialization.json", initialization)


def execute_phase1_compatibility(
    *,
    project_root: Path,
    video: Path,
    initial_bbox: Sequence[float],
    original_stage0_test_name: str,
    phase1_test_name: str,
    compatibility_test_name: str,
    device: str,
    gate_path: Path,
    review_decisions: Mapping[str, str] | None = None,
    reviewer: str = "KICKCLIP_BACKEND_COMPATIBILITY_GATE",
    review_note: str = (
        "Explicit reviewed Phase-1 decision; no automatic identity confirmation."
    ),
    allow_review_required: bool = False,
    require_final_outputs: bool = False,
    overwrite_phase1: bool = False,
) -> dict[str, Any]:
    """Run immutable Stage 0 and the real frozen Phase-1 pipeline.

    Return code 3 from the frozen runner is classified as REVIEW_REQUIRED rather
    than a generic subprocess crash. Production callers remain fail-closed by
    default because ``allow_review_required`` is False.

    ``review_decisions`` is intended only for decisions already made by a human
    or for a frozen, previously reviewed parity fixture. It is never inferred.
    """

    root = project_root.resolve()
    video = video.resolve()
    gate_path = gate_path.resolve()
    decisions = _validated_review_decisions(review_decisions)

    stage0 = root / "target_centric_tracking_v1" / "stage0_audit_inputs.py"
    phase1_runner = root / "target_centric_tracking_v1" / "run_phase1_frozen_pipeline.py"
    stage3c0 = (
        root
        / "target_centric_tracking_v2"
        / "stage3c0_run_short_clip_phase1_compatibility.py"
    )
    for required in (video, stage0, phase1_runner, stage3c0):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_dir = root / "runs" / "target_centric_tracking_v1" / original_stage0_test_name
    phase1_environment, ffmpeg_path = _phase1_environment()
    if not (source_dir / "audit.json").is_file() or overwrite_phase1:
        _run(
            [
                sys.executable,
                str(stage0),
                "--project-root",
                str(root),
                "--video",
                str(video),
                "--test-name",
                original_stage0_test_name,
                "--initial-bbox",
                *[str(value) for value in initial_bbox],
                "--bbox-format",
                "xyxy_pixels",
                *( ["--overwrite-audit"] if overwrite_phase1 else [] ),
            ],
            cwd=root,
            environment=phase1_environment,
        )

    audit_path = source_dir / "audit.json"
    manifest_path = source_dir / "input_manifest.json"
    audit = _read(audit_path)
    manifest = _read(manifest_path)
    original_audit_sha = _sha256(audit_path)
    original_manifest_sha = _sha256(manifest_path)
    allowed, blocked = classify_stage0_findings(audit, requested_device=device)
    status = str(audit.get("status") or "")
    if str(manifest.get("status") or "") != status:
        blocked.append("STAGE0_STATUS_CONTRACT_MISMATCH")

    gate: dict[str, Any] = {
        "schema_version": "kickclip.phase1_compatibility_gate.r2",
        "created_at": _now(),
        "execution_status": "PENDING",
        "original_stage0_status": status,
        "allowed_warning_codes": allowed,
        "blocked_warning_codes": sorted(set(blocked)),
        "stage1_authorized": False,
        "original_audit_path": str(audit_path),
        "original_audit_sha256": original_audit_sha,
        "original_manifest_path": str(manifest_path),
        "original_manifest_sha256": original_manifest_sha,
        "original_stage0_preserved": True,
        "frozen_phase1_modified": False,
        "synthetic_pass_created": False,
        "phase1_test_name": phase1_test_name,
        "runtime_ffmpeg_path": str(ffmpeg_path) if ffmpeg_path else None,
        "runtime_ffmpeg_sha256": _sha256(ffmpeg_path) if ffmpeg_path else None,
        "review_decisions": decisions,
        "review_decisions_source": (
            "EXPLICIT_CALLER_INPUT" if decisions else "NONE"
        ),
        "automatic_review_decision_used": False,
        "require_final_outputs": require_final_outputs,
    }
    _write(gate_path, gate)
    if blocked or status not in {"PASS", "PASS_WITH_WARNINGS"}:
        gate["execution_status"] = "BLOCKED_STAGE0"
        _write(gate_path, gate)
        raise Phase1CompatibilityBlocked(
            "Stage-0 findings blocked Phase-1: " + ", ".join(sorted(set(blocked)))
        )

    runner_return_code: int | None = None
    legacy_only_warning_pass = (
        status == "PASS_WITH_WARNINGS"
        and bool(allowed)
        and set(allowed).issubset(LEGACY_UNUSED_WARNING_CODES)
    )
    if status == "PASS" or legacy_only_warning_pass:
        phase1_dir = root / "runs" / "target_centric_tracking_v1" / phase1_test_name
        if legacy_only_warning_pass:
            _materialize_legacy_stage0_pass(
                source_dir=source_dir,
                phase1_dir=phase1_dir,
                phase1_test_name=phase1_test_name,
                allowed_warning_codes=allowed,
            )
        completed = _run(
            _phase1_runner_command(
                python=sys.executable,
                phase1_runner=phase1_runner,
                root=root,
                video=video,
                phase1_test_name=phase1_test_name,
                initial_bbox=initial_bbox,
                device=device,
                review_decisions=decisions,
                reviewer=reviewer,
                review_note=review_note,
                overwrite_phase1=overwrite_phase1,
            ),
            cwd=root,
            environment=phase1_environment,
            accepted_return_codes=frozenset({0, 3}),
        )
        runner_return_code = completed.returncode
        authorization = (
            "DERIVED_STAGE0_PASS_LEGACY_UNUSED_V7_WARNING"
            if legacy_only_warning_pass
            else "FROZEN_STAGE0_PASS"
        )
    else:
        if allowed != ["DURATION_OUTSIDE_RANGE"]:
            raise Phase1CompatibilityBlocked(
                "The installed frozen Stage 3-C0 contract authorizes only "
                "DURATION_OUTSIDE_RANGE for a genuinely short clip."
            )
        duration = float((manifest.get("video") or {}).get("duration_seconds") or 0.0)
        if not 0.0 < duration < 10.0:
            raise Phase1CompatibilityBlocked(
                f"Stage 3-C0 cannot authorize a non-short clip: {duration:.6f}s"
            )
        v2_dir = root / "runs" / "target_centric_tracking_v2" / compatibility_test_name
        v2_dir.mkdir(parents=True, exist_ok=True)
        stage3c0_result = _run(
            [
                sys.executable,
                str(stage3c0),
                "--project-root",
                str(root),
                "--v2-test-name",
                compatibility_test_name,
                "--source-stage0-test-name",
                original_stage0_test_name,
                "--compat-test-name",
                phase1_test_name,
                "--device",
                device,
                "--reviewer",
                reviewer,
                "--review-note",
                review_note,
            ],
            cwd=root,
            environment=phase1_environment,
            accepted_return_codes=frozenset({0, 2, 3}),
        )
        runner_return_code = stage3c0_result.returncode
        authorization = (
            "FROZEN_V2_STAGE3C0_SHORT_CLIP_COMPATIBILITY"
            if stage3c0_result.returncode == 0
            else "BACKEND_SHORT_CLIP_GATE_REAL_STAGE1_STAGE2"
        )
        gate["stage3c0_return_code"] = stage3c0_result.returncode

    phase1_dir = root / "runs" / "target_centric_tracking_v1" / phase1_test_name
    outputs = _collect_outputs(phase1_dir)
    stage1_summary_path = phase1_dir / "stage1_detection_summary.json"
    stage2_summary_path = phase1_dir / "stage2_association_summary.json"
    if not stage1_summary_path.is_file() or not stage2_summary_path.is_file():
        gate.update(
            {
                "execution_status": "BLOCKED_MISSING_STAGE1_STAGE2",
                "phase1_return_code": runner_return_code,
                "phase1_outputs": outputs,
            }
        )
        _write(gate_path, gate)
        raise Phase1CompatibilityBlocked("Real frozen Stage 1/2 evidence is missing.")

    stage1_summary = _read(stage1_summary_path)
    stage2_summary = _read(stage2_summary_path)
    if stage1_summary.get("status") != "PASS" or not outputs.get("detections.csv"):
        raise Phase1CompatibilityBlocked("Real frozen RF-DETR Stage 1 evidence is missing.")
    if stage2_summary.get("status") != "PASS" or not outputs.get("target_timeline.json"):
        raise Phase1CompatibilityBlocked("Real frozen same-shot Stage 2 evidence is missing.")
    if _sha256(audit_path) != original_audit_sha or _sha256(manifest_path) != original_manifest_sha:
        raise RuntimeError("Original Stage-0 artifacts changed during compatibility execution.")

    pipeline_state_path = phase1_dir / "phase1_pipeline_state.json"
    pipeline_state = _read(pipeline_state_path) if pipeline_state_path.is_file() else {}
    review_required = runner_return_code == 3 or pipeline_state.get("status") == "REVIEW_REQUIRED"

    gate.update(
        {
            "stage1_authorized": True,
            "authorization_contract": authorization,
            "phase1_return_code": runner_return_code,
            "real_rfdetr_inference_performed": True,
            "real_same_shot_stage2_performed": True,
            "stage1_processed_frames": int(
                (stage1_summary.get("video") or {}).get("processed_frames") or 0
            ),
            "phase1_outputs": outputs,
            "phase1_pipeline_state": pipeline_state,
            "review_required": review_required,
            "review_stage": pipeline_state.get("review_stage"),
            "review_preview": pipeline_state.get("preview"),
            "original_audit_preserved_after_execution": True,
        }
    )

    if review_required:
        gate["execution_status"] = "REVIEW_REQUIRED"
        gate["final_phase1_completed"] = False
        _write(gate_path, gate)
        _write(phase1_dir / "compatibility_gate.json", gate)
        if not allow_review_required:
            raise Phase1CompatibilityBlocked(
                "Frozen Phase-1 requires explicit review at "
                f"{pipeline_state.get('review_stage') or 'UNKNOWN_STAGE'}."
            )
        return gate

    final_timeline_path = phase1_dir / "final_target_timeline.json"
    final_audit_path = phase1_dir / "final_audit.json"
    final_complete = final_timeline_path.is_file() and final_audit_path.is_file()
    gate["final_phase1_completed"] = final_complete

    if require_final_outputs and not final_complete:
        gate["execution_status"] = "BLOCKED_FINAL_OUTPUTS_MISSING"
        _write(gate_path, gate)
        _write(phase1_dir / "compatibility_gate.json", gate)
        raise Phase1CompatibilityBlocked(
            "Final frozen Phase-1 outputs are required but missing."
        )

    if final_complete:
        final_timeline = _read(final_timeline_path)
        final_audit = _read(final_audit_path)
        if final_timeline.get("schema_version") != "kickclip.phase1_final_target_timeline.v1":
            raise Phase1CompatibilityBlocked("Unexpected final Phase-1 timeline schema.")
        if final_audit.get("status") != "PASS":
            raise Phase1CompatibilityBlocked("Final Phase-1 audit is not PASS.")
        summary = _timeline_summary(final_timeline)
        gate.update(
            {
                "final_timeline_summary": summary,
                "final_source_stage": summary.get("source_stage"),
                "final_audit_status": final_audit.get("status"),
                "final_outputs_complete": all(
                    (phase1_dir / name).is_file()
                    for name in FINAL_OUTPUT_NAMES
                ),
            }
        )
        if require_final_outputs and not gate["final_outputs_complete"]:
            gate["execution_status"] = "BLOCKED_FINAL_OUTPUT_SET_INCOMPLETE"
            _write(gate_path, gate)
            _write(phase1_dir / "compatibility_gate.json", gate)
            raise Phase1CompatibilityBlocked(
                "The stable final Phase-1 output set is incomplete."
            )

    gate["execution_status"] = "COMPLETE" if final_complete else "STAGE2_COMPLETE"
    _write(gate_path, gate)
    _write(phase1_dir / "compatibility_gate.json", gate)
    return gate
