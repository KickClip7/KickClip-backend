from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .artifacts import canonical_sha256, sha256_file, write_json_atomic
from .media_cache import SharedFrameCache
from .work_metrics import CandidatePreparationWorkMetrics


LAZY_MEDIA_SCHEMA_VERSION = "kickclip.candidate_lazy_media.r1"
LAZY_MEDIA_CACHE_SCHEMA_VERSION = "kickclip.candidate_lazy_media_cache.r1"
LAZY_MEDIA_NAMES = {
    "best_crop_display",
    "first_middle_last",
    "tracklet_video",
    "reference_gallery",
}
MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


class LazyCandidateMediaError(ValueError):
    pass


class LazyCandidateMediaMaterializer:
    """Materialize deferred review media without mutating the immutable bundle.

    The immutable candidate bundle stores only the generation specification.
    Generated assets live in a separate content-addressed cache keyed by the
    immutable bundle SHA-256 and the canonical lazy-media specification.
    """

    _locks_guard = threading.Lock()
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, *, storage_root: Path) -> None:
        self.storage_root = storage_root.resolve()
        self.lazy_root = (
            self.storage_root / "lazy_candidate_media_r1"
        ).resolve()
        self.frame_cache = SharedFrameCache(
            self.storage_root / "shared_frame_cache_r1"
        )
        self.lazy_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _lock_for(cls, key: str) -> threading.Lock:
        with cls._locks_guard:
            lock = cls._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._locks[key] = lock
            return lock

    @staticmethod
    def _clip_bbox(
        bbox: Iterable[float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        values = [float(value) for value in bbox]
        if len(values) != 4:
            raise LazyCandidateMediaError(
                "bbox_xyxy must contain four values."
            )
        x1, y1, x2, y2 = values
        x1i = max(0, min(width - 1, int(math.floor(x1))))
        y1i = max(0, min(height - 1, int(math.floor(y1))))
        x2i = max(x1i + 1, min(width, int(math.ceil(x2))))
        y2i = max(y1i + 1, min(height, int(math.ceil(y2))))
        return x1i, y1i, x2i, y2i

    @staticmethod
    def _label_image(
        image: np.ndarray,
        lines: list[str],
        *,
        origin: tuple[int, int] = (14, 28),
    ) -> None:
        x, y = origin
        for line in lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 27

    @classmethod
    def _contact_sheet(
        cls,
        items: list[tuple[np.ndarray, str]],
        *,
        cell_width: int = 420,
        cell_height: int = 360,
    ) -> np.ndarray:
        if not items:
            raise LazyCandidateMediaError(
                "Contact sheet has no source items."
            )

        cells: list[np.ndarray] = []
        for image, label in items:
            if image is None or image.size == 0:
                raise LazyCandidateMediaError(
                    "Contact sheet contains an empty image."
                )
            canvas = np.full(
                (cell_height, cell_width, 3),
                24,
                dtype=np.uint8,
            )
            available_height = cell_height - 44
            scale = min(
                cell_width / image.shape[1],
                available_height / image.shape[0],
            )
            resized = cv2.resize(
                image,
                (
                    max(1, int(image.shape[1] * scale)),
                    max(1, int(image.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            x = (cell_width - resized.shape[1]) // 2
            y = 34 + (available_height - resized.shape[0]) // 2
            canvas[
                y : y + resized.shape[0],
                x : x + resized.shape[1],
            ] = resized
            cls._label_image(canvas, [label], origin=(10, 25))
            cells.append(canvas)

        columns = min(3, max(1, len(cells)))
        rows = int(math.ceil(len(cells) / columns))
        blank = np.full(
            (cell_height, cell_width, 3),
            24,
            dtype=np.uint8,
        )
        while len(cells) < rows * columns:
            cells.append(blank.copy())
        return np.vstack(
            [
                np.hstack(
                    cells[
                        row * columns : (row + 1) * columns
                    ]
                )
                for row in range(rows)
            ]
        )

    @staticmethod
    def _validate_bundle_file(
        *,
        bundle_root: Path,
        record: Mapping[str, Any],
    ) -> Path:
        path = (
            bundle_root / str(record.get("path") or "")
        ).resolve()
        expected = str(record.get("sha256") or "")
        if (
            not path.is_relative_to(bundle_root)
            or not path.is_file()
            or len(expected) != 64
            or sha256_file(path) != expected
        ):
            raise LazyCandidateMediaError(
                "Lazy media source file integrity validation failed."
            )
        return path

    @staticmethod
    def _write_image_atomic(
        path: Path,
        image: np.ndarray,
        *,
        quality: int = 95,
    ) -> None:
        suffix = path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise LazyCandidateMediaError(
                f"Unsupported lazy image suffix: {suffix}"
            )

        extension = ".png" if suffix == ".png" else ".jpg"
        parameters: list[int] = []
        if extension == ".jpg":
            parameters = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

        ok, encoded = cv2.imencode(extension, image, parameters)
        if not ok:
            raise LazyCandidateMediaError(
                f"Failed to encode lazy image: {path.name}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.incomplete{path.suffix}"
        )
        payload = encoded.tobytes()
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _read_sidecar(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _cached_output(
        self,
        *,
        output_path: Path,
        sidecar_path: Path,
        bundle_manifest_sha256: str,
        specification_sha256: str,
    ) -> Path | None:
        sidecar = self._read_sidecar(sidecar_path)
        if sidecar is None or not output_path.is_file():
            return None
        expected = str(sidecar.get("output_sha256") or "")
        if (
            sidecar.get("schema_version")
            != LAZY_MEDIA_CACHE_SCHEMA_VERSION
            or sidecar.get("bundle_manifest_sha256")
            != bundle_manifest_sha256
            or sidecar.get("specification_sha256")
            != specification_sha256
            or len(expected) != 64
            or sha256_file(output_path) != expected
        ):
            return None
        return output_path

    def _materialize_best_crop_display(
        self,
        *,
        bundle_root: Path,
        manifest: Mapping[str, Any],
        specification: Mapping[str, Any],
        output_path: Path,
    ) -> None:
        source_key = str(
            specification.get("source_file_key")
            or "best_crop_native"
        )
        source_record = (manifest.get("files") or {}).get(source_key)
        if not isinstance(source_record, Mapping):
            raise LazyCandidateMediaError(
                "Native best crop is missing from the immutable bundle."
            )
        source_path = self._validate_bundle_file(
            bundle_root=bundle_root,
            record=source_record,
        )
        crop = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if crop is None or crop.size == 0:
            raise LazyCandidateMediaError(
                "Native best crop is not decodable."
            )
        factor = int(specification.get("scale_factor", 3))
        if factor < 1 or factor > 8:
            raise LazyCandidateMediaError(
                "Invalid lazy display upscale factor."
            )
        display = cv2.resize(
            crop,
            (
                crop.shape[1] * factor,
                crop.shape[0] * factor,
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )
        self._write_image_atomic(output_path, display)

    def _materialize_first_middle_last(
        self,
        *,
        source_video_path: Path,
        source_video_sha256: str,
        specification: Mapping[str, Any],
        output_path: Path,
        metrics: CandidatePreparationWorkMetrics,
    ) -> None:
        raw_items = list(specification.get("items") or [])
        if not raw_items:
            raise LazyCandidateMediaError(
                "First/middle/last specification is empty."
            )
        frame_indices = [
            int(item["frame"])
            for item in raw_items
            if isinstance(item, Mapping)
        ]
        cache = self.frame_cache.load_frames(
            video_path=source_video_path,
            video_sha256=source_video_sha256,
            frame_indices=frame_indices,
            metrics=metrics,
        )
        items: list[tuple[np.ndarray, str]] = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise LazyCandidateMediaError(
                    "Invalid first/middle/last item."
                )
            frame_index = int(raw["frame"])
            frame = cache.frames.get(frame_index)
            if frame is None:
                raise LazyCandidateMediaError(
                    f"Lazy source frame is unavailable: {frame_index}"
                )
            x1, y1, x2, y2 = self._clip_bbox(
                raw.get("bbox_xyxy") or [],
                cache.width,
                cache.height,
            )
            crop = frame[y1:y2, x1:x2].copy()
            label = str(raw.get("label") or "REFERENCE")
            items.append(
                (crop, f"frame {frame_index} / {label}")
            )
        sheet = self._contact_sheet(items)
        self._write_image_atomic(output_path, sheet)

    def _materialize_reference_gallery(
        self,
        *,
        bundle_root: Path,
        specification: Mapping[str, Any],
        output_path: Path,
    ) -> None:
        raw_items = list(specification.get("items") or [])
        if not raw_items:
            raise LazyCandidateMediaError(
                "Reference gallery specification is empty."
            )
        items: list[tuple[np.ndarray, str]] = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise LazyCandidateMediaError(
                    "Invalid reference gallery item."
                )
            path = (
                bundle_root / str(raw.get("path") or "")
            ).resolve()
            expected = str(raw.get("sha256") or "")
            if (
                not path.is_relative_to(bundle_root)
                or not path.is_file()
                or len(expected) != 64
                or sha256_file(path) != expected
            ):
                raise LazyCandidateMediaError(
                    "Reference crop integrity validation failed."
                )
            crop = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if crop is None or crop.size == 0:
                raise LazyCandidateMediaError(
                    "Reference crop is not decodable."
                )
            frame_index = int(raw.get("frame", -1))
            width = int(raw.get("width", crop.shape[1]))
            height = int(raw.get("height", crop.shape[0]))
            items.append(
                (
                    crop,
                    f"frame {frame_index} {width}x{height}",
                )
            )
        sheet = self._contact_sheet(items)
        self._write_image_atomic(output_path, sheet)

    def _materialize_tracklet_video(
        self,
        *,
        source_video_path: Path,
        source_video_sha256: str,
        specification: Mapping[str, Any],
        output_path: Path,
        metrics: CandidatePreparationWorkMetrics,
    ) -> None:
        start_frame = int(specification.get("start_frame", -1))
        end_frame = int(specification.get("end_frame", -1))
        width = int(specification.get("width", 0))
        height = int(specification.get("height", 0))
        fps = float(specification.get("output_fps", 0.0))
        max_bbox_age = int(
            specification.get("max_bbox_age_frames", 2)
        )
        candidate_id = str(
            specification.get("candidate_id") or "candidate"
        )
        observations = list(
            specification.get("observations") or []
        )
        if (
            start_frame < 0
            or end_frame < start_frame
            or width <= 0
            or height <= 0
            or fps <= 0
            or not observations
        ):
            raise LazyCandidateMediaError(
                "Tracklet video specification is invalid."
            )

        raw_frame_indices = list(
            specification.get("frame_indices") or []
        )
        frame_indices = sorted(
            {
                int(value)
                for value in raw_frame_indices
                if start_frame <= int(value) <= end_frame
            }
        )
        if not frame_indices:
            frame_indices = list(range(start_frame, end_frame + 1))
        cache = self.frame_cache.load_frames(
            video_path=source_video_path,
            video_sha256=source_video_sha256,
            frame_indices=frame_indices,
            metrics=metrics,
        )
        if cache.width != width or cache.height != height:
            raise LazyCandidateMediaError(
                "Tracklet video dimensions differ from the source video."
            )

        feature_by_frame: dict[int, list[float]] = {}
        for raw in observations:
            if not isinstance(raw, Mapping):
                continue
            frame_index = int(raw.get("frame", -1))
            bbox = raw.get("bbox_xyxy")
            if frame_index >= 0 and isinstance(bbox, list) and len(bbox) == 4:
                feature_by_frame[frame_index] = [
                    float(value) for value in bbox
                ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.incomplete.mp4"
        )
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise LazyCandidateMediaError(
                "Lazy tracklet video writer cannot be opened."
            )

        try:
            last_bbox: list[float] | None = None
            last_bbox_frame = -1
            for frame_index in frame_indices:
                source = cache.frames.get(frame_index)
                if source is None:
                    raise LazyCandidateMediaError(
                        f"Tracklet source frame is unavailable: {frame_index}"
                    )
                frame = source.copy()
                if frame_index in feature_by_frame:
                    last_bbox = feature_by_frame[frame_index]
                    last_bbox_frame = frame_index
                if (
                    last_bbox is not None
                    and frame_index - last_bbox_frame <= max_bbox_age
                ):
                    x1, y1, x2, y2 = self._clip_bbox(
                        last_bbox,
                        width,
                        height,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        4,
                    )
                self._label_image(
                    frame,
                    [
                        f"{candidate_id} frame={frame_index}",
                        "REVIEW ONLY - no automatic target confirmation",
                    ],
                )
                writer.write(frame)
                metrics.increment("lazy_video_frames_rendered")
        finally:
            writer.release()

        if not temporary.is_file() or temporary.stat().st_size <= 0:
            temporary.unlink(missing_ok=True)
            raise LazyCandidateMediaError(
                "Lazy tracklet video was not created."
            )
        os.replace(temporary, output_path)

    def materialize(
        self,
        *,
        bundle_root: Path,
        bundle_manifest_sha256: str,
        manifest: Mapping[str, Any],
        media_name: str,
        source_video_path: Path | None,
    ) -> tuple[Path, str]:
        if media_name not in LAZY_MEDIA_NAMES:
            raise LazyCandidateMediaError(
                "Candidate media is not lazily allowlisted."
            )
        if len(bundle_manifest_sha256) != 64:
            raise LazyCandidateMediaError(
                "Immutable bundle SHA-256 is invalid."
            )

        lazy = manifest.get("lazy_media") or {}
        specification = lazy.get(media_name)
        if not isinstance(specification, Mapping):
            raise LazyCandidateMediaError(
                "Candidate media is not available in this bundle."
            )
        if (
            specification.get("schema_version")
            != LAZY_MEDIA_SCHEMA_VERSION
        ):
            raise LazyCandidateMediaError(
                "Lazy media specification version is unsupported."
            )

        filename = Path(
            str(specification.get("filename") or "")
        ).name
        if not filename or filename != str(
            specification.get("filename") or ""
        ):
            raise LazyCandidateMediaError(
                "Lazy media filename is invalid."
            )
        suffix = Path(filename).suffix.lower()
        mime = MIME_BY_SUFFIX.get(suffix)
        if mime is None:
            raise LazyCandidateMediaError(
                "Lazy media suffix is unsupported."
            )

        specification_sha256 = canonical_sha256(
            {
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "media_name": media_name,
                "specification": dict(specification),
            }
        )
        output_root = (
            self.lazy_root
            / bundle_manifest_sha256[:2]
            / bundle_manifest_sha256
            / media_name
            / specification_sha256[:24]
        ).resolve()
        if not output_root.is_relative_to(self.lazy_root):
            raise LazyCandidateMediaError(
                "Lazy media output path escapes storage."
            )
        output_path = output_root / filename
        sidecar_path = output_root / "materialization.json"
        lock_key = f"{bundle_manifest_sha256}:{media_name}:{specification_sha256}"

        with self._lock_for(lock_key):
            cached = self._cached_output(
                output_path=output_path,
                sidecar_path=sidecar_path,
                bundle_manifest_sha256=bundle_manifest_sha256,
                specification_sha256=specification_sha256,
            )
            if cached is not None:
                return cached, mime

            output_root.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
            metrics = CandidatePreparationWorkMetrics(
                schema_version=(
                    "kickclip.candidate_lazy_media_work_metrics.r1"
                )
            )
            started = time.perf_counter()
            kind = str(specification.get("kind") or "")

            if kind == "BEST_CROP_DISPLAY":
                self._materialize_best_crop_display(
                    bundle_root=bundle_root,
                    manifest=manifest,
                    specification=specification,
                    output_path=output_path,
                )
            elif kind == "FIRST_MIDDLE_LAST":
                if source_video_path is None:
                    raise LazyCandidateMediaError(
                        "Source video is required for this lazy media."
                    )
                self._materialize_first_middle_last(
                    source_video_path=source_video_path,
                    source_video_sha256=str(
                        manifest.get("source_video_sha256") or ""
                    ),
                    specification=specification,
                    output_path=output_path,
                    metrics=metrics,
                )
            elif kind == "REFERENCE_GALLERY":
                self._materialize_reference_gallery(
                    bundle_root=bundle_root,
                    specification=specification,
                    output_path=output_path,
                )
            elif kind == "TRACKLET_VIDEO":
                if source_video_path is None:
                    raise LazyCandidateMediaError(
                        "Source video is required for this lazy media."
                    )
                self._materialize_tracklet_video(
                    source_video_path=source_video_path,
                    source_video_sha256=str(
                        manifest.get("source_video_sha256") or ""
                    ),
                    specification=specification,
                    output_path=output_path,
                    metrics=metrics,
                )
            else:
                raise LazyCandidateMediaError(
                    f"Unsupported lazy media kind: {kind}"
                )

            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise LazyCandidateMediaError(
                    "Lazy media materialization produced no file."
                )
            output_sha256 = sha256_file(output_path)
            metrics.add_duration(
                "lazy_media_materialization",
                time.perf_counter() - started,
            )
            metrics.increment("lazy_media_files_written")
            metrics.increment(
                "lazy_media_bytes_written",
                output_path.stat().st_size,
            )
            write_json_atomic(
                sidecar_path,
                {
                    "schema_version": LAZY_MEDIA_CACHE_SCHEMA_VERSION,
                    "bundle_manifest_sha256": bundle_manifest_sha256,
                    "specification_sha256": specification_sha256,
                    "media_name": media_name,
                    "output_path": output_path.name,
                    "output_sha256": output_sha256,
                    "size_bytes": output_path.stat().st_size,
                    "metrics": metrics.snapshot(),
                },
            )
            return output_path, mime
