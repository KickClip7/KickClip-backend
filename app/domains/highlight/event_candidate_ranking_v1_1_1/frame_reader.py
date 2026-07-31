from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class BoundedVideoFrameReader:
    """Random-access decoder with a strict frame-count LRU bound."""

    def __init__(self, path: Path, *, max_cached_frames: int = 8) -> None:
        if max_cached_frames < 1:
            raise ValueError("max_cached_frames must be positive.")
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError("Source video cannot be opened.")
        self.max_cached_frames = max_cached_frames
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.peak_cached_frames = 0
        self.peak_cached_bytes = 0
        self.decode_count = 0
        self.cache_hit_count = 0
        self.next_decode_frame = 0

    def get(self, frame_index: int) -> np.ndarray | None:
        cached = self.cache.pop(frame_index, None)
        if cached is not None:
            self.cache[frame_index] = cached
            self.cache_hit_count += 1
            return cached
        if frame_index != self.next_decode_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.next_decode_frame = frame_index + 1
        self.decode_count += 1
        self.cache[frame_index] = frame
        while len(self.cache) > self.max_cached_frames:
            self.cache.popitem(last=False)
        cached_bytes = sum(item.nbytes for item in self.cache.values())
        self.peak_cached_frames = max(self.peak_cached_frames, len(self.cache))
        self.peak_cached_bytes = max(self.peak_cached_bytes, cached_bytes)
        return frame

    def stats(self) -> dict[str, int]:
        return {
            "max_cached_frames": self.max_cached_frames,
            "peak_cached_frames": self.peak_cached_frames,
            "peak_cached_bytes": self.peak_cached_bytes,
            "decode_count": self.decode_count,
            "cache_hit_count": self.cache_hit_count,
        }

    def close(self) -> None:
        self.cache.clear()
        self.capture.release()
