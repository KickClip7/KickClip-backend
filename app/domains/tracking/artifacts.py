from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.domains.tracking.errors import (
    TrackingArtifactNotFoundError,
    TrackingContractError,
)
from app.domains.tracking.model import TrackingJob
from app.domains.tracking.verifier import configured_absolute_path
from app.storage.local_storage import LocalStorage

ARTIFACT_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
STATIC_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "pipeline_manifest": (
        "pipeline_manifest.json",
        "provenance",
        "application/json",
    ),
    "pipeline_state": (
        "pipeline_state.json",
        "state",
        "application/json",
    ),
    "pipeline_summary": (
        "pipeline_summary.json",
        "summary",
        "application/json",
    ),
    "target_timeline_json": (
        "target_timeline.json",
        "timeline",
        "application/json",
    ),
    "target_timeline_csv": (
        "target_timeline.csv",
        "timeline",
        "text/csv",
    ),
    "target_segments": (
        "target_segments.json",
        "segments",
        "application/json",
    ),
    "ambiguities": ("ambiguities.json", "ambiguities", "application/json"),
    "confirmations": (
        "confirmations.json",
        "confirmations",
        "application/json",
    ),
    "report": ("report.md", "report", "text/markdown"),
    "shot_boundaries": (
        "shot_boundaries.csv",
        "shots",
        "text/csv",
    ),
    "scene_clip_validation": (
        "scene_clip_validation.json",
        "input_validation",
        "application/json",
    ),
    "initial_bbox_frame0_preview": (
        "initial_bbox_frame0_preview.jpg",
        "input_validation_preview",
        "image/jpeg",
    ),
    "initial_bbox_detection_match": (
        "initial_bbox_detection_match.json",
        "input_validation",
        "application/json",
    ),
    "tracking_preview": (
        "full_frame_tracking_preview.mp4",
        "preview",
        "video/mp4",
    ),
    "target_centered_preview": (
        "target_centered_preview.mp4",
        "preview",
        "video/mp4",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrackingArtifactService:
    """Builds a server-owned allowlist and resolves no caller-provided paths."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def collect(
        self,
        job: TrackingJob,
        state: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        root = self.job_root(job)
        index: dict[str, dict[str, Any]] = {}

        for key, (filename, kind, mime_type) in STATIC_ARTIFACTS.items():
            source = root / filename
            served = source
            source_sha256 = None
            if source.is_file() and source.suffix.lower() == ".json":
                served = root / "backend_artifacts" / "public" / f"{key}.json"
                self._write_public_json(source=source, target=served, root=root)
                source_sha256 = sha256_file(source)
            elif source.is_file() and kind == "preview":
                browser_preview = self._write_browser_preview(
                    source=source,
                    target=(
                        root
                        / "backend_artifacts"
                        / "public"
                        / f"{key}.mp4"
                    ),
                    root=root,
                )
                if browser_preview is not None:
                    served = browser_preview
            index[key] = self._entry(
                root=root,
                path=served,
                key=key,
                kind=kind,
                mime_type=mime_type,
            )
            if source_sha256 is not None:
                index[key]["source_sha256"] = source_sha256

        if state is not None:
            self._collect_pending(index=index, root=root, state=state)
            self._collect_ambiguities(index=index, root=root, state=state)
        return index

    def stage_input_validation_artifacts(
        self,
        job: TrackingJob,
        artifact_paths: Mapping[str, str],
    ) -> None:
        """Copy allowlisted bridge artifacts into the durable job output root."""

        root = self.job_root(job)
        storage_root = LocalStorage().storage_root
        root.mkdir(parents=True, exist_ok=True)
        for key in (
            "scene_clip_validation",
            "initial_bbox_frame0_preview",
            "initial_bbox_detection_match",
        ):
            raw_path = artifact_paths.get(key)
            if not raw_path:
                continue
            source = Path(raw_path).resolve()
            filename = STATIC_ARTIFACTS[key][0]
            target = (root / filename).resolve()
            if (
                not source.is_file()
                or not source.is_relative_to(storage_root)
                or not target.is_relative_to(root)
            ):
                raise TrackingContractError(
                    f"Input validation artifact is invalid: {key}."
                )
            shutil.copy2(source, target)

    def resolve(
        self,
        job: TrackingJob,
        artifact_key: str,
    ) -> tuple[Path, str, str]:
        if not ARTIFACT_KEY_PATTERN.fullmatch(artifact_key):
            raise TrackingArtifactNotFoundError()
        record = (job.artifact_index or {}).get(artifact_key)
        if not isinstance(record, Mapping) or not record.get("relative_path"):
            raise TrackingArtifactNotFoundError()
        root = self.job_root(job)
        path = (root / str(record["relative_path"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise TrackingArtifactNotFoundError()
        if (
            record.get("kind") in {"preview", "review_preview"}
            and path.suffix.lower() == ".mp4"
        ):
            browser_preview = self._write_browser_preview(
                source=path,
                target=(
                    root
                    / "backend_artifacts"
                    / "public"
                    / f"{artifact_key}.mp4"
                ),
                root=root,
            )
            if browser_preview is not None:
                path = browser_preview
        mime_type = str(
            record.get("mime_type")
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        return path, mime_type, path.name

    def job_root(self, job: TrackingJob) -> Path:
        output_root = configured_absolute_path(
            self.settings.TRACKING_OUTPUT_ROOT,
            "TRACKING_OUTPUT_ROOT",
        )
        expected = (output_root / job.test_name).resolve()
        stored = Path(job.output_directory).resolve()
        if stored == expected and stored.is_relative_to(output_root):
            return stored

        r1_metadata = (job.runtime_metadata or {}).get(
            "event_candidate_handoff_r1"
        )
        r1_root = (
            LocalStorage().storage_root / "event_candidate_handoff_r1"
        ).resolve()
        pipeline_state = Path(job.pipeline_state_path).resolve()
        if (
            isinstance(r1_metadata, Mapping)
            and stored.name == job.tracking_job_id
            and stored.is_relative_to(r1_root)
            and pipeline_state.parent == stored
        ):
            return stored

        raise TrackingContractError("Tracking job output root is invalid.")

    def _collect_pending(
        self,
        *,
        index: dict[str, dict[str, Any]],
        root: Path,
        state: Mapping[str, Any],
    ) -> None:
        pending = state.get("pending_action")
        if not isinstance(pending, Mapping):
            return
        pending_type = str(pending.get("type") or "")
        if pending_type == "MEMORY_REVIEW":
            self._add_state_file(
                index,
                root,
                key="memory_contact_sheet",
                kind="review_contact_sheet",
                raw_path=pending.get("contact_sheet"),
            )
            self._add_state_file(
                index,
                root,
                key="memory_tracking_preview",
                kind="review_preview",
                raw_path=pending.get("preview"),
            )
        elif pending_type == "CROSS_SHOT_CONFIRMATION":
            ambiguity_id = self._safe_component(
                str(pending.get("ambiguity_id") or "")
            )
            key = f"{ambiguity_id}_contact_sheet"
            self._add_state_file(
                index,
                root,
                key=key,
                kind="ambiguity_contact_sheet",
                raw_path=pending.get("contact_sheet"),
            )
        elif pending_type in {"SEGMENT_VISUAL_REVIEW", "PHASE1_INTERNAL_REVIEW"}:
            segment_id = self._safe_component(
                str(pending.get("segment_id") or "segment")
            )
            stage = self._safe_component(
                str(pending.get("review_stage") or "review").lower()
            )
            self._add_state_file(
                index,
                root,
                key=f"{segment_id}_{stage}_tracking_preview",
                kind="review_preview",
                raw_path=pending.get("preview"),
            )
            self._add_state_file(
                index,
                root,
                key=f"{segment_id}_{stage}_centered_preview",
                kind="review_preview",
                raw_path=pending.get("centered_preview"),
            )

    def _collect_ambiguities(
        self,
        *,
        index: dict[str, dict[str, Any]],
        root: Path,
        state: Mapping[str, Any],
    ) -> None:
        ambiguities = state.get("ambiguities")
        if not isinstance(ambiguities, list):
            return
        for ambiguity in ambiguities:
            if not isinstance(ambiguity, Mapping):
                continue
            ambiguity_id = self._safe_component(
                str(ambiguity.get("ambiguity_id") or "")
            )
            if not ambiguity_id:
                continue
            self._add_state_file(
                index,
                root,
                key=f"{ambiguity_id}_contact_sheet",
                kind="ambiguity_contact_sheet",
                raw_path=ambiguity.get("contact_sheet"),
            )

    def _add_state_file(
        self,
        index: dict[str, dict[str, Any]],
        root: Path,
        *,
        key: str,
        kind: str,
        raw_path: object,
    ) -> None:
        if not raw_path or not ARTIFACT_KEY_PATTERN.fullmatch(key):
            return
        raw = Path(str(raw_path))
        source = (
            raw.resolve()
            if raw.is_absolute()
            else (root / raw).resolve()
        )
        project_root = configured_absolute_path(
            self.settings.TRACKING_PROJECT_ROOT,
            "TRACKING_PROJECT_ROOT",
        )
        if (
            not source.is_file()
            or (
                not source.is_relative_to(root)
                and not source.is_relative_to(project_root)
            )
        ):
            return

        if kind == "review_preview" and source.suffix.lower() == ".mp4":
            browser_preview = self._write_browser_preview(
                source=source,
                target=(
                    root
                    / "backend_artifacts"
                    / "public"
                    / f"{key}.mp4"
                ),
                root=root,
            )
            if browser_preview is not None:
                target = browser_preview
            elif source.is_relative_to(root):
                target = source
            else:
                target = self._copy_external_artifact(
                    source=source,
                    root=root,
                    key=key,
                )
        elif source.is_relative_to(root):
            target = source
        else:
            target = self._copy_external_artifact(
                source=source,
                root=root,
                key=key,
            )

        mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        index[key] = self._entry(
            root=root,
            path=target,
            key=key,
            kind=kind,
            mime_type=mime_type,
        )

    @staticmethod
    def _copy_external_artifact(*, source: Path, root: Path, key: str) -> Path:
        suffix = source.suffix if source.suffix else ".bin"
        target = (root / "backend_artifacts" / f"{key}{suffix}").resolve()
        if not target.is_relative_to(root):
            raise TrackingContractError("Artifact path escapes the tracking job root.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            not target.is_file()
            or target.stat().st_size != source.stat().st_size
            or target.stat().st_mtime_ns < source.stat().st_mtime_ns
        ):
            shutil.copy2(source, target)
        return target

    @staticmethod
    def _write_browser_preview(
        *,
        source: Path,
        target: Path,
        root: Path,
    ) -> Path | None:
        """Transcode a runtime preview into a broadly playable H.264 MP4."""

        target = target.resolve()
        if not target.is_relative_to(root):
            raise TrackingContractError("Public artifact path escapes job root.")
        if (
            target.is_file()
            and target.stat().st_size > 0
            and target.stat().st_mtime_ns >= source.stat().st_mtime_ns
        ):
            return target

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.stem}.{uuid4().hex}.tmp{target.suffix}"
        )
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if (
                completed.returncode != 0
                or not temporary.is_file()
                or temporary.stat().st_size <= 0
            ):
                temporary.unlink(missing_ok=True)
                return None
            os.replace(temporary, target)
            return target
        except OSError:
            temporary.unlink(missing_ok=True)
            return None

    def _entry(
        self,
        *,
        root: Path,
        path: Path,
        key: str,
        kind: str,
        mime_type: str,
    ) -> dict[str, Any]:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise TrackingContractError("Artifact path escapes the tracking job root.")
        exists = resolved.is_file()
        record: dict[str, Any] = {
            "key": key,
            "kind": kind,
            "relative_path": resolved.relative_to(root).as_posix(),
            "mime_type": mime_type,
            "exists": exists,
            "size_bytes": None,
            "updated_at": None,
            "sha256": None,
        }
        if exists:
            stat = resolved.stat()
            record["size_bytes"] = stat.st_size
            record["updated_at"] = datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat()
            if stat.st_size <= 64 * 1024 * 1024 and resolved.suffix.lower() in {
                ".json",
                ".csv",
                ".md",
            }:
                record["sha256"] = sha256_file(resolved)
        return record

    @staticmethod
    def _safe_component(value: str) -> str:
        return value if ARTIFACT_KEY_PATTERN.fullmatch(value) else ""

    @staticmethod
    def _write_public_json(*, source: Path, target: Path, root: Path) -> None:
        target = target.resolve()
        if not target.is_relative_to(root):
            raise TrackingContractError("Public artifact path escapes job root.")
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackingContractError(
                f"Tracking JSON artifact is invalid: {source.name}"
            ) from exc
        redacted = _redact_absolute_paths(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(redacted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, target)


def _redact_absolute_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_absolute_paths(item) for item in value]
    if isinstance(value, str) and (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        return None
    return value
