from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.core.config import Settings, get_settings
from app.domains.candidate_handoff_r1.artifacts import sha256_file, write_json_atomic
from app.domains.candidate_handoff_r1.model import (
    EventCandidateMemoryRevisionR1,
    EventCandidateSelectionR1,
)
from app.domains.tracking.model import TrackingJob
from app.storage.local_storage import LocalStorage


class R1R3AdapterError(ValueError):
    """Raised when immutable R1 selection artifacts cannot form an R3 launch."""


@dataclass(frozen=True)
class R1R3AdapterResult:
    selection_artifact_root: Path
    tracking_launch_manifest_path: Path
    tracking_launch_manifest_sha256: str
    shot_boundaries_path: Path
    shot_boundaries_sha256: str
    target_selection_path: Path
    target_selection_sha256: str
    target_reference_set_path: Path
    target_reference_set_sha256: str
    earlier_anchor_decision_path: Path
    earlier_anchor_decision_sha256: str
    tracking_cache_key: str
    current_target_memory_path: Path | None = None
    current_target_memory_sha256: str | None = None
    candidate_scoring_generation: int = 1

    def runtime_metadata(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "selection_artifact_root": str(self.selection_artifact_root),
            "tracking_launch_manifest_path": str(self.tracking_launch_manifest_path),
            "tracking_launch_manifest_sha256": self.tracking_launch_manifest_sha256,
            "shot_boundaries_path": str(self.shot_boundaries_path),
            "shot_boundaries_sha256": self.shot_boundaries_sha256,
            "target_selection_path": str(self.target_selection_path),
            "target_selection_sha256": self.target_selection_sha256,
            "target_reference_set_path": str(self.target_reference_set_path),
            "target_reference_set_sha256": self.target_reference_set_sha256,
            "earlier_anchor_decision_path": str(self.earlier_anchor_decision_path),
            "earlier_anchor_decision_sha256": self.earlier_anchor_decision_sha256,
            "tracking_cache_key": self.tracking_cache_key,
            "candidate_scoring_generation": self.candidate_scoring_generation,
        }
        if self.current_target_memory_path is not None:
            value["current_target_memory_path"] = str(self.current_target_memory_path)
            value["current_target_memory_sha256"] = self.current_target_memory_sha256
        return value


class R1R3InputAdapter:
    """Build immutable R1 -> provided V1/V2 runtime launch artifacts.

    The adapter intentionally does not invent a production_r3 package. The configured
    subprocess entry point is a backend-owned CLI that validates these files and then
    calls the supplied frozen V1/V2 research stages.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        storage: LocalStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or LocalStorage()

    @staticmethod
    def _load_object(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise R1R3AdapterError(f"Unreadable JSON artifact: {path}") from exc
        if not isinstance(value, dict):
            raise R1R3AdapterError(f"Expected JSON object: {path}")
        return value

    @staticmethod
    def _require_sha(path: Path, expected: str, label: str) -> str:
        if not path.is_file():
            raise R1R3AdapterError(f"Missing {label}: {path}")
        actual = sha256_file(path)
        if len(expected) != 64 or actual != expected:
            raise R1R3AdapterError(f"{label} SHA-256 mismatch: {path}")
        return actual

    @staticmethod
    def _configured_manifest(path_value: str, sha_value: str, label: str) -> dict[str, str]:
        if not path_value or not sha_value:
            raise R1R3AdapterError(f"{label} manifest configuration is missing.")
        path = Path(path_value).expanduser().resolve()
        actual = R1R3InputAdapter._require_sha(path, sha_value.strip().lower(), label)
        return {"path": str(path), "sha256": actual}

    def build(
        self,
        *,
        job: TrackingJob,
        selection: EventCandidateSelectionR1,
        candidate_manifest_path: Path,
        source_video_path: Path,
        reviewed_shot_boundaries_path: Path,
        current_memory: EventCandidateMemoryRevisionR1 | None = None,
        candidate_scoring_generation: int = 1,
    ) -> R1R3AdapterResult:
        candidate_manifest_path = candidate_manifest_path.resolve()
        source_video_path = source_video_path.resolve()
        reviewed_shot_boundaries_path = reviewed_shot_boundaries_path.resolve()
        self._require_sha(
            candidate_manifest_path,
            selection.candidate_media_bundle_sha256,
            "candidate review bundle",
        )
        self._require_sha(source_video_path, selection.source_video_sha256, "source video")
        manifest = self._load_object(candidate_manifest_path)

        identities = {
            str(selection.candidate_id),
            str(manifest.get("candidate_id") or ""),
            str(manifest.get("candidate_media_id") or ""),
            str((manifest.get("best_observation") or {}).get("candidate_id") or ""),
        }
        if identities != {selection.candidate_id}:
            raise R1R3AdapterError("CANDIDATE_SELECTION_PROVENANCE_MISMATCH")
        quality = manifest.get("quality")
        if not isinstance(quality, Mapping) or quality.get("identity_pure") is not True:
            raise R1R3AdapterError("Selected candidate is not identity-pure.")

        anchor = manifest.get("best_observation")
        if not isinstance(anchor, Mapping):
            raise R1R3AdapterError("Selected candidate best observation is missing.")
        anchor_frame = int(anchor.get("frame", -1))
        anchor_bbox = anchor.get("bbox_xyxy")
        if anchor_frame < 0 or not isinstance(anchor_bbox, list) or len(anchor_bbox) != 4:
            raise R1R3AdapterError("Selected candidate anchor is invalid.")

        bundle_root = candidate_manifest_path.parent
        self._require_sha(
            reviewed_shot_boundaries_path,
            selection.reviewed_shot_boundaries_sha256,
            "reviewed shot boundaries",
        )
        boundaries = self._load_object(reviewed_shot_boundaries_path)
        shots = list(boundaries.get("shots") or boundaries.get("boundaries") or [])
        selected_shot = next(
            (row for row in shots if str(row.get("shot_id")) == selection.shot_id),
            None,
        )
        if selected_shot is None:
            raise R1R3AdapterError("Selected shot is absent from reviewed boundaries.")
        shot_start = int(selected_shot.get("start_frame", -1))
        shot_end = int(
            selected_shot.get("end_frame_inclusive", selected_shot.get("end_frame", -1))
        )
        if not shot_start <= anchor_frame <= shot_end:
            raise R1R3AdapterError("Selected anchor is outside its reviewed shot.")

        reference_rows: list[dict[str, Any]] = []
        for row in manifest.get("reference_gallery") or []:
            if not isinstance(row, Mapping):
                continue
            crop = (bundle_root / str(row.get("path") or "")).resolve()
            if not crop.is_relative_to(bundle_root):
                raise R1R3AdapterError("Reference crop escapes immutable bundle root.")
            expected = str(row.get("crop_sha256") or "")
            digest = self._require_sha(crop, expected, "reference crop")
            reference_rows.append(
                {
                    "frame_id": int(row["frame"]),
                    "path": str(crop),
                    "sha256": digest,
                    "scale": str(row.get("scale_class") or "unknown"),
                    "source_candidate_id": selection.candidate_id,
                }
            )
        if len(reference_rows) < 3:
            raise R1R3AdapterError("At least three immutable target references are required.")

        root = (Path(job.output_directory) / "r3_inputs").resolve()
        root.mkdir(parents=True, exist_ok=True)
        selection_record_path = self.storage.resolve_path(selection.selection_artifact_path)
        self._require_sha(
            selection_record_path,
            selection.selection_artifact_sha256,
            "selection record",
        )

        scene_manifest = self._configured_manifest(
            self.settings.SCENE_TARGET_SELECTION_MANIFEST_PATH,
            self.settings.SCENE_TARGET_SELECTION_MANIFEST_SHA256,
            "Scene Target Selection",
        )
        r2_manifest = self._configured_manifest(
            self.settings.TRACKING_R2_MANIFEST_PATH,
            self.settings.TRACKING_R2_MANIFEST_SHA256,
            "R2",
        )
        r3_manifest = self._configured_manifest(
            self.settings.TRACKING_R3_MANIFEST_PATH,
            self.settings.TRACKING_R3_MANIFEST_SHA256,
            "R3 adapter",
        )

        provenance = {
            "selection_id": selection.selection_id,
            "ranking_id": selection.ranking_id,
            "shortlist_patch_id": selection.shortlist_patch_id,
            "discovery_id": selection.discovery_id,
            "selected_candidate_id": selection.candidate_id,
            "shot_id": selection.shot_id,
            "tracklet_id": selection.tracklet_id,
            "best_anchor_frame": anchor_frame,
            "best_anchor_bbox_xyxy": [float(value) for value in anchor_bbox],
            "source_video_sha256": selection.source_video_sha256,
            "reviewed_shot_boundaries_sha256": selection.reviewed_shot_boundaries_sha256,
            "scene_target_selection_manifest": scene_manifest,
            "r2_manifest": r2_manifest,
            "r3_adapter_manifest": r3_manifest,
            "automatic_target_confirmation": False,
        }
        target_selection = {
            "schema_version": "kickclip.r1_target_selection.v1",
            "immutable": True,
            **provenance,
            "selected_shot_start_frame": shot_start,
            "selected_shot_end_frame_inclusive": shot_end,
            "selection_record": {
                "path": str(selection_record_path),
                "sha256": selection.selection_artifact_sha256,
            },
            "candidate_review_bundle": {
                "path": str(candidate_manifest_path),
                "sha256": selection.candidate_media_bundle_sha256,
            },
        }
        target_selection_path = root / "target_selection.json"
        target_selection_sha = write_json_atomic(target_selection_path, target_selection)

        target_reference_set = {
            "schema_version": "kickclip.r1_target_reference_set.v1",
            "immutable": True,
            "selection_id": selection.selection_id,
            "selected_candidate_id": selection.candidate_id,
            "reference_count": len(reference_rows),
            "references": reference_rows,
            "scale_banks": sorted({row["scale"] for row in reference_rows}),
            "automatic_target_confirmation": False,
        }
        target_reference_set_path = root / "target_reference_set.json"
        target_reference_set_sha = write_json_atomic(
            target_reference_set_path, target_reference_set
        )

        earlier_anchor_decision = {
            "schema_version": "kickclip.r1_earlier_anchor_decision.v1",
            "immutable": True,
            "decision": "USER_SELECTED_EVENT_CANDIDATE",
            "selection_id": selection.selection_id,
            "selected_candidate_id": selection.candidate_id,
            "selected_shot_id": selection.shot_id,
            "anchor_frame": anchor_frame,
            "anchor_bbox_xyxy": [float(value) for value in anchor_bbox],
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
        }
        earlier_anchor_decision_path = root / "earlier_anchor_decision.json"
        earlier_anchor_decision_sha = write_json_atomic(
            earlier_anchor_decision_path, earlier_anchor_decision
        )

        memory_path: Path | None = None
        memory_sha: str | None = None
        if current_memory is not None:
            memory_path = self.storage.resolve_path(current_memory.artifact_path)
            memory_sha = self._require_sha(
                memory_path, current_memory.artifact_sha256, "target memory revision"
            )

        cache_material = {
            **provenance,
            "target_selection_sha256": target_selection_sha,
            "target_reference_set_sha256": target_reference_set_sha,
            "earlier_anchor_decision_sha256": earlier_anchor_decision_sha,
            "current_target_memory_sha256": memory_sha,
            "candidate_scoring_generation": candidate_scoring_generation,
        }
        cache_key_path = root / "tracking_cache_material.json"
        tracking_cache_key = write_json_atomic(cache_key_path, cache_material)
        launch_manifest = {
            "schema_version": "kickclip.r1_v1_v2_launch_manifest.v1",
            "integration_path": "B.ADD_THIN_BACKEND_ADAPTER_TO_V1_V2_STAGES",
            "tracking_job_id": job.tracking_job_id,
            "test_name": job.test_name,
            "source_video": {
                "path": str(source_video_path),
                "sha256": selection.source_video_sha256,
            },
            "shot_boundaries": {
                "path": str(reviewed_shot_boundaries_path),
                "sha256": selection.reviewed_shot_boundaries_sha256,
            },
            "target_selection": {
                "path": str(target_selection_path),
                "sha256": target_selection_sha,
            },
            "target_reference_set": {
                "path": str(target_reference_set_path),
                "sha256": target_reference_set_sha,
            },
            "earlier_anchor_decision": {
                "path": str(earlier_anchor_decision_path),
                "sha256": earlier_anchor_decision_sha,
            },
            "current_target_memory": (
                {"path": str(memory_path), "sha256": memory_sha}
                if memory_path is not None
                else None
            ),
            "candidate_scoring_generation": candidate_scoring_generation,
            "tracking_cache_key": tracking_cache_key,
            "automatic_target_confirmation": False,
        }
        tracking_launch_manifest_path = root / "tracking_launch_manifest.json"
        tracking_launch_manifest_sha = write_json_atomic(
            tracking_launch_manifest_path, launch_manifest
        )

        return R1R3AdapterResult(
            selection_artifact_root=root,
            tracking_launch_manifest_path=tracking_launch_manifest_path,
            tracking_launch_manifest_sha256=tracking_launch_manifest_sha,
            shot_boundaries_path=reviewed_shot_boundaries_path,
            shot_boundaries_sha256=selection.reviewed_shot_boundaries_sha256,
            target_selection_path=target_selection_path,
            target_selection_sha256=target_selection_sha,
            target_reference_set_path=target_reference_set_path,
            target_reference_set_sha256=target_reference_set_sha,
            earlier_anchor_decision_path=earlier_anchor_decision_path,
            earlier_anchor_decision_sha256=earlier_anchor_decision_sha,
            tracking_cache_key=tracking_cache_key,
            current_target_memory_path=memory_path,
            current_target_memory_sha256=memory_sha,
            candidate_scoring_generation=candidate_scoring_generation,
        )

    def attach_to_job(self, job: TrackingJob, result: R1R3AdapterResult) -> None:
        metadata = dict(job.runtime_metadata or {})
        metadata["scene_target_selection"] = result.runtime_metadata()
        metadata["r1_runtime_integration"] = {
            "integration_path": "B.ADD_THIN_BACKEND_ADAPTER_TO_V1_V2_STAGES",
            "frame_zero_fallback_used": False,
            "automatic_target_confirmation": False,
        }
        job.runtime_metadata = metadata
