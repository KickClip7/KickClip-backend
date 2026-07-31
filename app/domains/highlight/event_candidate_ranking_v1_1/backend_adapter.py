from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy.orm import Session

from app.domains.artifact.model import Artifact
from app.domains.artifact.repository import ArtifactRepository
from app.domains.auth.model import User
from app.domains.highlight.model import HighlightRevision
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.storage.local_storage import LocalStorage
from app.utils.id_generator import generate_prefixed_id

from .contract import ImmutableCandidateInput, sha256_file
from .service import EventCandidateRankingV11Engine


def cache_fingerprint(material: dict[str, Any]) -> str:
    required = (
        "ranking_source_manifest_sha256",
        "policy_sha256",
        "feature_schema_sha256",
        "candidate_manifest_sha256",
        "shot_boundaries_sha256",
        "event_id",
        "event_time_sec",
        "event_label",
        "source_video_sha256",
    )
    missing = [key for key in required if key not in material]
    if missing:
        raise ValueError(
            f"V1.1 cache material is incomplete: {', '.join(missing)}"
        )
    canonical = json.dumps(
        {key: material[key] for key in required},
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EventCandidateRankingV11BackendAdapter:
    """Server-side immutable input bridge for the additive V1.1 shadow task."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = LocalStorage()
        self.artifacts = ArtifactRepository(db)
        self.package_root = (
            self.storage.project_root
            / "configs/models/event_candidate_ranking/"
            "target_centric_tracking_event_candidate_ranking_v1_1"
        ).resolve()

    def _artifact(self, artifact_id: str, project: Project) -> Artifact:
        artifact = self.db.get(Artifact, artifact_id)
        if artifact is None or artifact.match_id != project.match_id:
            raise ValueError("Immutable ranking artifact is missing or out of scope.")
        return artifact

    def _media(self, asset_id: str, project: Project) -> MediaAsset:
        media = self.db.get(MediaAsset, asset_id)
        if media is None or media.match_id != project.match_id:
            raise ValueError("Scene source video is missing or out of scope.")
        return media

    @staticmethod
    def _probe_video(path: Path, media: MediaAsset) -> tuple[int, int, float, int]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("Scene source video cannot be probed.")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        width = int(media.width or width)
        height = int(media.height or height)
        fps = float(media.fps or fps)
        if min(width, height, frame_count) <= 0 or fps <= 0:
            raise ValueError("Scene source video probe is incomplete.")
        return width, height, fps, frame_count

    def freeze_material(
        self,
        *,
        project: Project,
        revision_id: str,
        scene_id: str,
        event_id: str,
        event_label: str,
        event_time_sec: float,
    ) -> dict[str, Any]:
        revision = self.db.get(HighlightRevision, revision_id)
        if revision is None or revision.project_id != project.project_id:
            raise ValueError("Highlight revision is missing or out of scope.")
        discovery = (revision.options or {}).get("scene_target_selection") or {}
        if discovery.get("scene_id") != scene_id:
            raise ValueError("Scene discovery metadata does not match the request.")
        artifact_root = self.storage.resolve_path(discovery["artifact_root"])
        candidate_path = (artifact_root / "scene_candidates.json").resolve()
        if not candidate_path.is_relative_to(artifact_root) or not candidate_path.is_file():
            raise ValueError("Immutable scene candidate artifact is missing.")
        detections = self._artifact(discovery["detections_artifact_id"], project)
        boundaries = self._artifact(
            discovery["shot_boundaries_artifact_id"], project
        )
        video = self._media(discovery["scene_video_asset_id"], project)
        detections_path = self.storage.resolve_path(detections.file_path)
        boundaries_path = self.storage.resolve_path(boundaries.file_path)
        video_path = self.storage.resolve_path(video.file_path)
        width, height, fps, frame_count = self._probe_video(video_path, video)
        candidate_sha = sha256_file(candidate_path)
        detections_sha = sha256_file(detections_path)
        boundaries_sha = sha256_file(boundaries_path)
        video_sha = video.sha256 or sha256_file(video_path)
        inputs = discovery.get("discovery_inputs") or {}
        if inputs.get("detections_sha256") != detections_sha:
            raise ValueError("Detections artifact changed after candidate discovery.")
        if inputs.get("shot_boundaries_sha256") != boundaries_sha:
            raise ValueError("Shot-boundary artifact changed after candidate discovery.")
        if inputs.get("video_sha256") != video_sha:
            raise ValueError("Source video changed after candidate discovery.")
        material = {
            "storage_root": self.storage.storage_root.as_posix(),
            "scene_candidates_relative_path": candidate_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "scene_candidates_sha256": candidate_sha,
            "detections_relative_path": detections_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "detections_sha256": detections_sha,
            "source_video_relative_path": video_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "source_video_sha256": video_sha,
            "shot_boundaries_relative_path": boundaries_path.relative_to(
                self.storage.storage_root
            ).as_posix(),
            "shot_boundaries_sha256": boundaries_sha,
            "candidate_manifest_sha256": discovery[
                "scene_candidate_manifest_sha256"
            ],
            "candidate_package_manifest_sha256": inputs[
                "candidate_package_manifest_sha256"
            ],
            "video_width": width,
            "video_height": height,
            "video_fps": fps,
            "video_frame_count": frame_count,
            "event_id": event_id,
            "event_label": event_label,
            "event_time_sec": event_time_sec,
            "ranking_source_manifest_sha256": sha256_file(
                self.package_root / "manifest.json"
            ),
            "policy_sha256": sha256_file(
                self.package_root / "event_candidate_ranking_policy.json"
            ),
            "feature_schema_sha256": sha256_file(
                self.package_root / "event_candidate_feature_schema.json"
            ),
        }
        material["cache_fingerprint"] = cache_fingerprint(material)
        revision.options = {
            **(revision.options or {}),
            "scene_target_selection": {
                **discovery,
                "event_candidate_ranking_v1_1_input": {
                    key: material[key]
                    for key in (
                        "scene_candidates_relative_path",
                        "scene_candidates_sha256",
                        "detections_relative_path",
                        "detections_sha256",
                        "source_video_relative_path",
                        "source_video_sha256",
                        "video_width",
                        "video_height",
                        "video_fps",
                        "video_frame_count",
                        "shot_boundaries_relative_path",
                        "shot_boundaries_sha256",
                        "candidate_manifest_sha256",
                        "candidate_package_manifest_sha256",
                    )
                },
            },
        }
        self.db.flush()
        return material

    def run(
        self,
        *,
        project: Project,
        user: User,
        revision_id: str,
        ranking: dict[str, Any],
        freeze_material: dict[str, Any],
    ) -> dict[str, Any]:
        root = Path(freeze_material["storage_root"]).resolve()
        immutable_input = ImmutableCandidateInput.load(
            discovery_root=root,
            **{
                key: freeze_material[key]
                for key in (
                    "scene_candidates_relative_path",
                    "scene_candidates_sha256",
                    "detections_relative_path",
                    "detections_sha256",
                    "source_video_relative_path",
                    "source_video_sha256",
                    "shot_boundaries_relative_path",
                    "shot_boundaries_sha256",
                    "candidate_manifest_sha256",
                    "video_width",
                    "video_height",
                    "video_fps",
                    "video_frame_count",
                )
            },
        )
        document = EventCandidateRankingV11Engine(self.package_root).run(
            immutable_input=immutable_input,
            **ranking,
        )
        ranking_id = generate_prefixed_id("ecrankv11")
        discovery_root = self.storage.resolve_path(
            (
                self.db.get(HighlightRevision, revision_id).options
                or {}
            )["scene_target_selection"]["artifact_root"]
        )
        output_root = (
            discovery_root / "rankings" / f"v1_1_{ranking_id}"
        ).resolve()
        if not output_root.is_relative_to(discovery_root):
            raise ValueError("V1.1 output escapes discovery root.")
        output_root.mkdir(parents=True, exist_ok=False)
        output_path = output_root / "event_candidate_ranking_v1_1.json"
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        artifact = self.artifacts.create(
            match_id=project.match_id,
            project_id=project.project_id,
            analysis_job_id=None,
            artifact_type="EVENT_CANDIDATE_RANKING_V1_1_SHADOW",
            file_path=output_path.relative_to(
                self.storage.project_root
            ).as_posix(),
            mime_type="application/json",
            metadata_={
                "ranking_id": ranking_id,
                "revision_id": revision_id,
                "scene_id": ranking["scene_id"],
                "event_id": ranking["event_id"],
                "status": document["ranking_status"],
                "automatic_target_confirmation": False,
                "sha256": sha256_file(output_path),
                "freeze_material": {
                    key: value
                    for key, value in freeze_material.items()
                    if key != "storage_root"
                },
                "owner_id": user.user_id,
            },
        )
        self.db.commit()
        return {
            "ranking_id": ranking_id,
            "artifact_id": artifact.artifact_id,
            "ranking_status": document["ranking_status"],
            "automatic_target_confirmation": False,
            "shortlist_candidate_ids": [
                row["candidate_id"] for row in document["shortlist"]
            ],
            "full_gallery_fallback_candidate_ids": document[
                "full_gallery_fallback"
            ],
            "output_sha256": sha256_file(output_path),
        }
