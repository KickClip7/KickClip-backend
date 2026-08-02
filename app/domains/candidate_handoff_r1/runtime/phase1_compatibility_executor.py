from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BENIGN_WARNING_CODES = frozenset(
    {
        "DURATION_OUTSIDE_RANGE",
        "V6_OUTPUT_NOT_FOUND",
        "V7_OUTPUT_NOT_FOUND",
        "FFMPEG_NOT_FOUND",
        "CUDA_NOT_AVAILABLE",
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
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[Any]:
    completed = subprocess.run(
        list(command), cwd=str(cwd), shell=False, check=False, env=environment
    )
    if completed.returncode != 0 and not allow_failure:
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
) -> dict[str, Any]:
    """Run immutable Stage 0 and a reviewed V2 compatibility contract.

    The original Stage-0 directory is never edited. For genuinely short clips,
    the frozen V2 Stage 3-C0 adapter creates a separate derived Phase-1 run and
    invokes the real frozen Stage 1/2 implementation.
    """

    root = project_root.resolve()
    video = video.resolve()
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

    # These are optional reusable-output roots, not frozen inputs. Creating the
    # empty roots prevents installation-layout warnings from obscuring the
    # actual clip warning; no reusable tracking result is fabricated.
    (root / "runs" / "global_ID_tracking_upgrade_v6").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "global_ID_tracking_upgrade_v7").mkdir(parents=True, exist_ok=True)

    source_dir = root / "runs" / "target_centric_tracking_v1" / original_stage0_test_name
    phase1_environment, ffmpeg_path = _phase1_environment()
    if not (source_dir / "audit.json").is_file():
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
        "schema_version": "kickclip.phase1_compatibility_gate.r1",
        "created_at": _now(),
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
    }
    _write(gate_path, gate)
    if blocked or status not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise Phase1CompatibilityBlocked(
            "Stage-0 findings blocked Phase-1: " + ", ".join(sorted(set(blocked)))
        )

    if status == "PASS":
        _run(
            [
                sys.executable,
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
            ],
            cwd=root,
            environment=phase1_environment,
        )
        authorization = "FROZEN_STAGE0_PASS"
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
                "KICKCLIP_BACKEND_COMPATIBILITY_GATE",
                "--review-note",
                "Immutable selected-shot pre-cut region; no padding or threshold change.",
            ],
            cwd=root,
            environment=phase1_environment,
            allow_failure=True,
        )
        authorization = (
            "FROZEN_V2_STAGE3C0_SHORT_CLIP_COMPATIBILITY"
            if stage3c0_result.returncode == 0
            else "BACKEND_SHORT_CLIP_GATE_REAL_STAGE1_STAGE2"
        )
        gate["stage3c0_return_code"] = stage3c0_result.returncode

    phase1_dir = root / "runs" / "target_centric_tracking_v1" / phase1_test_name
    outputs: dict[str, dict[str, Any]] = {}
    for name in (
        "detections.csv",
        "stage1_detection_summary.json",
        "stage2_association_summary.json",
        "target_timeline.json",
        "stage2_target_timeline.json",
    ):
        path = phase1_dir / name
        if path.is_file():
            outputs[name] = {"path": str(path), "sha256": _sha256(path)}
    stage1_summary = _read(phase1_dir / "stage1_detection_summary.json")
    stage2_summary = _read(phase1_dir / "stage2_association_summary.json")
    if stage1_summary.get("status") != "PASS" or not outputs.get("detections.csv"):
        raise Phase1CompatibilityBlocked("Real frozen RF-DETR Stage 1 evidence is missing.")
    if stage2_summary.get("status") != "PASS" or not outputs.get("target_timeline.json"):
        raise Phase1CompatibilityBlocked("Real frozen same-shot Stage 2 evidence is missing.")
    if _sha256(audit_path) != original_audit_sha or _sha256(manifest_path) != original_manifest_sha:
        raise RuntimeError("Original Stage-0 artifacts changed during compatibility execution.")

    gate.update(
        {
            "stage1_authorized": True,
            "authorization_contract": authorization,
            "real_rfdetr_inference_performed": True,
            "real_same_shot_stage2_performed": True,
            "stage1_processed_frames": int(
                (stage1_summary.get("video") or {}).get("processed_frames") or 0
            ),
            "phase1_outputs": outputs,
            "original_audit_preserved_after_execution": True,
        }
    )
    _write(gate_path, gate)
    _write(phase1_dir / "compatibility_gate.json", gate)
    return gate
