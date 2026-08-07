from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.domains.highlight.player_detector import (
    PlayerDetection,
    PlayerDetector,
)


DETECTION_CACHE_SCHEMA_VERSION = "kickclip.player_detection_cache.r1"
DETECTION_CACHE_POLICY_VERSION = "VIDEO_FRAME_DETECTOR_CONTRACT_R2_ROLE_AWARE"


@dataclass(frozen=True)
class DetectionCacheBatchResult:
    detections_by_frame: list[list[PlayerDetection]]
    detector_fingerprint: str
    cache_hits: int
    cache_misses: int
    inference_frame_count: int
    cache_write_count: int

    def as_dict(self) -> dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "schema_version": DETECTION_CACHE_SCHEMA_VERSION,
            "policy_version": DETECTION_CACHE_POLICY_VERSION,
            "detector_fingerprint": self.detector_fingerprint,
            "requested_frame_count": total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": (
                round(self.cache_hits / total, 6)
                if total > 0
                else 0.0
            ),
            "inference_frame_count": self.inference_frame_count,
            "cache_write_count": self.cache_write_count,
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(
                value.items(),
                key=lambda row: str(row[0]),
            )
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _frame_sha256(frame_bgr: Any) -> str:
    array = np.ascontiguousarray(frame_bgr)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(
        ",".join(str(value) for value in array.shape).encode("ascii")
    )
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _detector_contract(runtime_metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields that can change detector inference semantics.

    Device names and absolute checkpoint paths are intentionally excluded so
    the same immutable checkpoint/config can reuse cache entries across
    Windows, macOS and Linux hosts.
    """

    semantic_keys = (
        "backend",
        "component_runtime_version",
        "rfdetr_version",
        "model_class",
        "checkpoint_sha256",
        "confidence_threshold",
        "inference_threshold",
        "output_confidence_threshold",
        "detector_profile",
        "class_mapping",
        "candidate_class_ids",
        "output_class_ids",
        "input_color",
        "preprocessing",
        "bbox_semantics",
        "bbox_bounds_policy",
        "additional_nms",
        "strict_checkpoint_audit",
    )
    return {
        key: _json_safe(runtime_metadata.get(key))
        for key in semantic_keys
        if runtime_metadata.get(key) is not None
    }


def detector_fingerprint(runtime_metadata: dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema_version": DETECTION_CACHE_SCHEMA_VERSION,
            "policy_version": DETECTION_CACHE_POLICY_VERSION,
            "detector_contract": _detector_contract(runtime_metadata),
        }
    )


class PlayerDetectionCache:
    """Content-addressed per-video, per-frame player detection cache.

    A cache entry is accepted only when all of these match:
    - immutable source video SHA-256
    - source frame index
    - detector semantic fingerprint
    - decoded BGR frame SHA-256

    Frame content hashing prevents reuse when a caller supplies a frame that
    does not actually correspond to the claimed video/frame coordinate.
    """

    _locks_guard = threading.Lock()
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _lock_for(cls, key: str) -> threading.Lock:
        with cls._locks_guard:
            lock = cls._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._locks[key] = lock
            return lock

    def _contract_root(
        self,
        *,
        source_video_sha256: str,
        detector_fingerprint_value: str,
    ) -> Path:
        if len(source_video_sha256) != 64:
            raise ValueError(
                "source_video_sha256 must be a 64-character SHA-256."
            )
        if len(detector_fingerprint_value) != 64:
            raise ValueError(
                "detector_fingerprint must be a 64-character SHA-256."
            )
        return (
            self.root
            / source_video_sha256[:2]
            / source_video_sha256
            / detector_fingerprint_value
        )

    def _frame_path(
        self,
        *,
        source_video_sha256: str,
        detector_fingerprint_value: str,
        frame_index: int,
    ) -> Path:
        return (
            self._contract_root(
                source_video_sha256=source_video_sha256,
                detector_fingerprint_value=detector_fingerprint_value,
            )
            / "frames"
            / f"frame_{frame_index:09d}.json"
        )

    @staticmethod
    def _serialize_detection(
        detection: PlayerDetection,
    ) -> dict[str, Any]:
        return {
            "bbox_xyxy": [
                round(float(value), 6)
                for value in detection.bbox_xyxy
            ],
            "confidence": round(float(detection.confidence), 8),
            "class_id": int(detection.class_id),
            "class_name": str(detection.class_name),
        }

    @staticmethod
    def _deserialize_detection(
        payload: dict[str, Any],
    ) -> PlayerDetection:
        bbox = payload.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Cached detection bbox_xyxy is invalid.")
        values = [float(value) for value in bbox]
        confidence = float(payload["confidence"])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cached detection bbox_xyxy contains non-finite values.")
        if not math.isfinite(confidence):
            raise ValueError("Cached detection confidence is non-finite.")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("Cached detection bbox_xyxy is degenerate.")
        return PlayerDetection(
            bbox_xyxy=values,
            confidence=confidence,
            class_id=int(payload["class_id"]),
            class_name=str(payload["class_name"]),
        )

    @staticmethod
    def _read_entry(
        path: Path,
        *,
        source_video_sha256: str,
        detector_fingerprint_value: str,
        frame_index: int,
        frame_sha256: str,
    ) -> list[PlayerDetection] | None:
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            expected = {
                "schema_version": DETECTION_CACHE_SCHEMA_VERSION,
                "policy_version": DETECTION_CACHE_POLICY_VERSION,
                "source_video_sha256": source_video_sha256,
                "detector_fingerprint": detector_fingerprint_value,
                "frame_index": frame_index,
                "frame_bgr_sha256": frame_sha256,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                return None
            rows = payload.get("detections")
            if not isinstance(rows, list):
                return None
            return [
                PlayerDetectionCache._deserialize_detection(row)
                for row in rows
                if isinstance(row, dict)
            ]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.incomplete"
        )
        data = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _ensure_manifest(
        self,
        *,
        source_video_sha256: str,
        detector_fingerprint_value: str,
        runtime_metadata: dict[str, Any],
    ) -> None:
        root = self._contract_root(
            source_video_sha256=source_video_sha256,
            detector_fingerprint_value=detector_fingerprint_value,
        )
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            return
        self._write_json_atomic(
            manifest_path,
            {
                "schema_version": DETECTION_CACHE_SCHEMA_VERSION,
                "policy_version": DETECTION_CACHE_POLICY_VERSION,
                "source_video_sha256": source_video_sha256,
                "detector_fingerprint": detector_fingerprint_value,
                "detector_contract": _detector_contract(runtime_metadata),
            },
        )

    def detect_batch(
        self,
        *,
        source_video_sha256: str,
        frame_indices: Sequence[int],
        frames_bgr: Sequence[Any],
        detector: PlayerDetector,
    ) -> DetectionCacheBatchResult:
        if len(frame_indices) != len(frames_bgr):
            raise ValueError(
                "frame_indices and frames_bgr must have the same length."
            )
        if not frame_indices:
            fingerprint = detector_fingerprint(detector.runtime_metadata)
            return DetectionCacheBatchResult(
                detections_by_frame=[],
                detector_fingerprint=fingerprint,
                cache_hits=0,
                cache_misses=0,
                inference_frame_count=0,
                cache_write_count=0,
            )

        normalized_indices = [int(value) for value in frame_indices]
        if any(value < 0 for value in normalized_indices):
            raise ValueError("frame_indices must be non-negative.")

        runtime_metadata = dict(detector.runtime_metadata)
        fingerprint = detector_fingerprint(runtime_metadata)
        lock_key = f"{source_video_sha256}:{fingerprint}"
        lock = self._lock_for(lock_key)

        frame_hashes = [_frame_sha256(frame) for frame in frames_bgr]
        results: list[list[PlayerDetection] | None] = [
            None for _ in frames_bgr
        ]
        miss_positions: list[int] = []

        with lock:
            self._ensure_manifest(
                source_video_sha256=source_video_sha256,
                detector_fingerprint_value=fingerprint,
                runtime_metadata=runtime_metadata,
            )

            for position, (frame_index, frame_hash) in enumerate(
                zip(normalized_indices, frame_hashes)
            ):
                path = self._frame_path(
                    source_video_sha256=source_video_sha256,
                    detector_fingerprint_value=fingerprint,
                    frame_index=frame_index,
                )
                cached = self._read_entry(
                    path,
                    source_video_sha256=source_video_sha256,
                    detector_fingerprint_value=fingerprint,
                    frame_index=frame_index,
                    frame_sha256=frame_hash,
                )
                if cached is None:
                    miss_positions.append(position)
                else:
                    results[position] = cached

            cache_write_count = 0
            if miss_positions:
                inferred = detector.detect_batch(
                    [frames_bgr[position] for position in miss_positions]
                )
                if len(inferred) != len(miss_positions):
                    raise RuntimeError(
                        "Player detector output count does not match cache misses."
                    )
                for position, detections in zip(miss_positions, inferred):
                    frame_index = normalized_indices[position]
                    frame_hash = frame_hashes[position]
                    rows = list(detections)
                    results[position] = rows
                    path = self._frame_path(
                        source_video_sha256=source_video_sha256,
                        detector_fingerprint_value=fingerprint,
                        frame_index=frame_index,
                    )
                    self._write_json_atomic(
                        path,
                        {
                            "schema_version": DETECTION_CACHE_SCHEMA_VERSION,
                            "policy_version": DETECTION_CACHE_POLICY_VERSION,
                            "source_video_sha256": source_video_sha256,
                            "detector_fingerprint": fingerprint,
                            "frame_index": frame_index,
                            "frame_bgr_sha256": frame_hash,
                            "frame_shape": [
                                int(value)
                                for value in np.asarray(
                                    frames_bgr[position]
                                ).shape
                            ],
                            "detections": [
                                self._serialize_detection(detection)
                                for detection in rows
                            ],
                        },
                    )
                    cache_write_count += 1

        finalized: list[list[PlayerDetection]] = []
        for position, value in enumerate(results):
            if value is None:
                raise RuntimeError(
                    "Detection cache did not resolve frame position "
                    f"{position}."
                )
            finalized.append(value)

        return DetectionCacheBatchResult(
            detections_by_frame=finalized,
            detector_fingerprint=fingerprint,
            cache_hits=len(frame_indices) - len(miss_positions),
            cache_misses=len(miss_positions),
            inference_frame_count=len(miss_positions),
            cache_write_count=cache_write_count,
        )
