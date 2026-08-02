from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .artifacts import sha256_file, write_json_atomic
from .work_metrics import CandidatePreparationWorkMetrics


FRAME_CACHE_SCHEMA_VERSION = "kickclip.shared_frame_cache.r1"
FRAME_EXTRACTION_VERSION = "opencv-bgr-jpeg-v1"


@dataclass(frozen=True)
class FrameCacheResult:
    frames: dict[int, np.ndarray]

    width: int
    height: int
    fps: float
    frame_count: int

    cache_hits: int
    cache_misses: int
    decoded_frames: int
    files_written: int
    bytes_written: int


class SharedFrameCache:
    """영상 SHA와 frame index 기반의 공용 lazy frame cache.

    요청된 frame만 저장한다.

    가까운 frame들이 함께 요청되면 한 번의 sequential decode로
    처리하여 frame마다 VideoCapture seek를 반복하지 않는다.
    """

    _locks_guard = threading.Lock()
    _video_locks: dict[str, threading.Lock] = {}

    def __init__(
        self,
        root: Path,
        *,
        jpeg_quality: int = 95,
        maximum_sequential_gap: int = 12,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                "jpeg_quality must be between 1 and 100."
            )

        if maximum_sequential_gap < 0:
            raise ValueError(
                "maximum_sequential_gap must be non-negative."
            )

        self.root = root.resolve()
        self.jpeg_quality = int(jpeg_quality)
        self.maximum_sequential_gap = int(
            maximum_sequential_gap
        )

        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _lock_for(
        cls,
        video_sha256: str,
    ) -> threading.Lock:
        with cls._locks_guard:
            lock = cls._video_locks.get(video_sha256)

            if lock is None:
                lock = threading.Lock()
                cls._video_locks[video_sha256] = lock

            return lock

    def _video_root(
        self,
        video_sha256: str,
    ) -> Path:
        if len(video_sha256) != 64:
            raise ValueError(
                "video_sha256 must be a 64-character SHA-256."
            )

        return (
            self.root
            / video_sha256[:2]
            / video_sha256
        )

    def _frame_path(
        self,
        video_sha256: str,
        frame_index: int,
    ) -> Path:
        return (
            self._video_root(video_sha256)
            / "frames"
            / f"frame_{frame_index:09d}.jpg"
        )

    @staticmethod
    def _read_cached(
        path: Path,
    ) -> np.ndarray | None:
        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            return None

        frame = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if frame is None or frame.size == 0:
            return None

        return frame

    @staticmethod
    def _group_indices(
        indices: list[int],
        maximum_gap: int,
    ) -> list[list[int]]:
        groups: list[list[int]] = []

        for frame_index in indices:
            if (
                not groups
                or frame_index - groups[-1][-1]
                > maximum_gap + 1
            ):
                groups.append([frame_index])
            else:
                groups[-1].append(frame_index)

        return groups

    def _write_frame_atomic(
        self,
        path: Path,
        frame: np.ndarray,
    ) -> int:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not encoded_ok:
            raise RuntimeError(
                f"Failed to encode cached frame: {path}"
            )

        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.incomplete"
        )
        payload = encoded.tobytes()

        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)

        return len(payload)

    def _ensure_manifest(
        self,
        *,
        video_path: Path,
        video_sha256: str,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
    ) -> None:
        root = self._video_root(video_sha256)
        manifest_path = root / "manifest.json"

        if manifest_path.is_file():
            return

        write_json_atomic(
            manifest_path,
            {
                "schema_version": (
                    FRAME_CACHE_SCHEMA_VERSION
                ),
                "frame_extraction_version": (
                    FRAME_EXTRACTION_VERSION
                ),
                "source_video_path": str(
                    video_path.resolve()
                ),
                "source_video_sha256": video_sha256,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "jpeg_quality": self.jpeg_quality,
            },
        )

    def load_frames(
        self,
        *,
        video_path: Path,
        video_sha256: str,
        frame_indices: Iterable[int],
        metrics: (
            CandidatePreparationWorkMetrics | None
        ) = None,
        verify_source_sha256: bool = False,
    ) -> FrameCacheResult:
        requested = sorted(
            {
                int(value)
                for value in frame_indices
                if int(value) >= 0
            }
        )

        if not requested:
            return FrameCacheResult(
                frames={},
                width=0,
                height=0,
                fps=0.0,
                frame_count=0,
                cache_hits=0,
                cache_misses=0,
                decoded_frames=0,
                files_written=0,
                bytes_written=0,
            )

        video_path = video_path.resolve()

        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        if (
            verify_source_sha256
            and sha256_file(video_path)
            != video_sha256
        ):
            raise ValueError(
                "Source video SHA-256 mismatch."
            )

        metadata_capture = cv2.VideoCapture(
            str(video_path)
        )

        if not metadata_capture.isOpened():
            raise ValueError(
                f"Video cannot be opened: {video_path}"
            )

        width = int(
            metadata_capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
        height = int(
            metadata_capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
        fps = float(
            metadata_capture.get(
                cv2.CAP_PROP_FPS
            )
        )
        frame_count = int(
            metadata_capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        metadata_capture.release()

        if requested[-1] >= frame_count:
            raise ValueError(
                f"Requested frame {requested[-1]} "
                f"exceeds video frame count {frame_count}."
            )

        lock = self._lock_for(video_sha256)
        started = time.perf_counter()

        with lock:
            frames: dict[int, np.ndarray] = {}
            missing: list[int] = []

            for frame_index in requested:
                path = self._frame_path(
                    video_sha256,
                    frame_index,
                )
                cached = self._read_cached(path)

                if cached is None:
                    missing.append(frame_index)
                else:
                    frames[frame_index] = cached

            cache_hits = len(requested) - len(missing)
            cache_misses = len(missing)
            decoded_frames = 0
            files_written = 0
            bytes_written = 0

            if missing:
                capture = cv2.VideoCapture(
                    str(video_path)
                )

                if not capture.isOpened():
                    raise ValueError(
                        f"Video cannot be opened: "
                        f"{video_path}"
                    )

                self._ensure_manifest(
                    video_path=video_path,
                    video_sha256=video_sha256,
                    width=width,
                    height=height,
                    fps=fps,
                    frame_count=frame_count,
                )

                try:
                    groups = self._group_indices(
                        missing,
                        self.maximum_sequential_gap,
                    )

                    for group in groups:
                        wanted = set(group)

                        capture.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            group[0],
                        )

                        for frame_index in range(
                            group[0],
                            group[-1] + 1,
                        ):
                            ok, frame = capture.read()

                            if not ok or frame is None:
                                raise ValueError(
                                    "Video frame is not "
                                    "decodable: "
                                    f"{frame_index}"
                                )

                            decoded_frames += 1

                            if frame_index not in wanted:
                                continue

                            path = self._frame_path(
                                video_sha256,
                                frame_index,
                            )

                            bytes_written += (
                                self._write_frame_atomic(
                                    path,
                                    frame,
                                )
                            )
                            files_written += 1
                            frames[frame_index] = (
                                frame.copy()
                            )
                finally:
                    capture.release()

            for frame_index in requested:
                if frame_index in frames:
                    continue

                cached = self._read_cached(
                    self._frame_path(
                        video_sha256,
                        frame_index,
                    )
                )

                if cached is None:
                    raise RuntimeError(
                        "Cached frame is missing after "
                        "materialization: "
                        f"{frame_index}"
                    )

                frames[frame_index] = cached

        if metrics is not None:
            metrics.increment(
                "frame_cache_requests",
                len(requested),
            )
            metrics.increment(
                "frame_cache_hits",
                cache_hits,
            )
            metrics.increment(
                "frame_cache_misses",
                cache_misses,
            )
            metrics.increment(
                "video_capture_open_count",
                1 + (1 if missing else 0),
            )

            metrics.increment(
                "video_metadata_capture_open_count",
                1,
            )

            metrics.increment(
                "video_decode_capture_open_count",
                1 if missing else 0,
            )
            metrics.increment(
                "video_frames_decoded",
                decoded_frames,
            )
            metrics.increment(
                "frame_cache_files_written",
                files_written,
            )
            metrics.increment(
                "frame_cache_bytes_written",
                bytes_written,
            )
            metrics.add_duration(
                "frame_cache_load",
                time.perf_counter() - started,
            )

        return FrameCacheResult(
            frames=frames,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            decoded_frames=decoded_frames,
            files_written=files_written,
            bytes_written=bytes_written,
        )