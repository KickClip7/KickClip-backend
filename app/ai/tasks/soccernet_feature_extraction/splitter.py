from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CHUNK_FILE_PATTERN = re.compile(r"^chunk_(\d+)\.npy$")


@dataclass(frozen=True)
class FeatureMergeResult:
    """Result of merging chunk_XXX.npy files into one feature array."""

    chunk_paths: list[Path]
    merged_path: Path
    merged_shape: tuple[int, ...]
    dtype: str
    feature_dim: int

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_paths)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "num_chunks": self.num_chunks,
            "chunk_paths": [path.as_posix() for path in self.chunk_paths],
            "merged_path": self.merged_path.as_posix(),
            "merged_shape": list(self.merged_shape),
            "dtype": self.dtype,
            "feature_dim": self.feature_dim,
        }


@dataclass(frozen=True)
class FeatureSplitResult:
    """Result of splitting merged feature into half1 and half2 arrays."""

    merged_path: Path
    half1_path: Path
    half2_path: Path
    merged_shape: tuple[int, ...]
    half1_shape: tuple[int, ...]
    half2_shape: tuple[int, ...]
    feature_fps: float
    split_strategy: str
    split_sec: float
    split_index: int
    duration_sec: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "merged_path": self.merged_path.as_posix(),
            "half1_path": self.half1_path.as_posix(),
            "half2_path": self.half2_path.as_posix(),
            "merged_shape": list(self.merged_shape),
            "half1_shape": list(self.half1_shape),
            "half2_shape": list(self.half2_shape),
            "feature_fps": self.feature_fps,
            "split_strategy": self.split_strategy,
            "split_sec": self.split_sec,
            "split_index": self.split_index,
            "duration_sec": self.duration_sec,
        }


class FeatureMergeError(RuntimeError):
    """Raised when chunk feature files cannot be merged safely."""


class FeatureSplitError(RuntimeError):
    """Raised when merged feature cannot be split safely."""


def collect_chunk_feature_paths(chunks_dir: str | Path) -> list[Path]:
    """Collect chunk_XXX.npy files sorted by the numeric XXX part.

    String sorting is intentionally avoided. This keeps ordering correct even
    if future chunk names are not zero-padded.
    """

    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        raise FeatureMergeError(f"Chunk directory does not exist: {chunks_dir.as_posix()}")
    if not chunks_dir.is_dir():
        raise FeatureMergeError(f"Chunk path is not a directory: {chunks_dir.as_posix()}")

    indexed_paths: list[tuple[int, Path]] = []
    for path in chunks_dir.iterdir():
        if not path.is_file():
            continue
        match = CHUNK_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        indexed_paths.append((int(match.group(1)), path))

    if not indexed_paths:
        raise FeatureMergeError(f"No chunk_XXX.npy files found in: {chunks_dir.as_posix()}")

    indexed_paths.sort(key=lambda item: item[0])
    expected = list(range(indexed_paths[0][0], indexed_paths[-1][0] + 1))
    actual = [index for index, _ in indexed_paths]
    if actual != expected:
        raise FeatureMergeError(
            "Missing chunk file(s). "
            f"expected indexes {expected}, got {actual} in {chunks_dir.as_posix()}"
        )

    # Service-generated chunks should start from 0. Treat a different first
    # index as a likely partial/corrupted extraction output.
    if actual[0] != 0:
        raise FeatureMergeError(
            f"First chunk index must be 0, got {actual[0]} in {chunks_dir.as_posix()}"
        )

    return [path for _, path in indexed_paths]


def merge_chunk_features(
    *,
    chunks_dir: str | Path | None = None,
    chunk_paths: Sequence[str | Path] | None = None,
    merged_path: str | Path,
    expected_dim: int = 512,
    overwrite: bool = True,
) -> FeatureMergeResult:
    """Merge chunk feature arrays into merged_feature.npy.

    All chunk arrays must be 2-D, non-empty, and share the same feature dim.
    The second dimension must match expected_dim.
    """

    if chunk_paths is None:
        if chunks_dir is None:
            raise FeatureMergeError("Either chunks_dir or chunk_paths must be provided.")
        resolved_chunk_paths = collect_chunk_feature_paths(chunks_dir)
    else:
        resolved_chunk_paths = [Path(path) for path in chunk_paths]

    if not resolved_chunk_paths:
        raise FeatureMergeError("No chunk feature paths were provided.")

    merged_path = Path(merged_path)
    if merged_path.exists() and not overwrite:
        merged_array = np.load(merged_path, mmap_mode="r")
        _validate_feature_array(
            merged_array,
            path=merged_path,
            expected_dim=expected_dim,
            allow_empty=False,
        )
        return FeatureMergeResult(
            chunk_paths=list(resolved_chunk_paths),
            merged_path=merged_path,
            merged_shape=tuple(int(v) for v in merged_array.shape),
            dtype=str(merged_array.dtype),
            feature_dim=int(merged_array.shape[1]),
        )

    arrays: list[np.ndarray] = []
    reference_dim: int | None = None
    reference_dtype: str | None = None

    for path in resolved_chunk_paths:
        path = Path(path)
        if not path.exists():
            raise FeatureMergeError(f"Chunk feature file does not exist: {path.as_posix()}")
        if not path.is_file():
            raise FeatureMergeError(f"Chunk feature path is not a file: {path.as_posix()}")

        array = np.load(path)
        _validate_feature_array(
            array,
            path=path,
            expected_dim=expected_dim,
            allow_empty=False,
        )

        feature_dim = int(array.shape[1])
        dtype = str(array.dtype)
        if reference_dim is None:
            reference_dim = feature_dim
            reference_dtype = dtype
        elif feature_dim != reference_dim:
            raise FeatureMergeError(
                "Chunk feature dim mismatch: "
                f"expected {reference_dim}, got {feature_dim} at {path.as_posix()}"
            )

        if reference_dtype is not None and dtype != reference_dtype:
            # NumPy can concatenate different dtypes through promotion, but the
            # model input should remain predictable. Fail loudly instead.
            raise FeatureMergeError(
                "Chunk feature dtype mismatch: "
                f"expected {reference_dtype}, got {dtype} at {path.as_posix()}"
            )

        arrays.append(array)

    try:
        merged = np.concatenate(arrays, axis=0)
    except Exception as exc:
        raise FeatureMergeError(f"Failed to concatenate chunk features: {exc}") from exc

    _validate_feature_array(
        merged,
        path=merged_path,
        expected_dim=expected_dim,
        allow_empty=False,
    )

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(merged_path, merged)

    return FeatureMergeResult(
        chunk_paths=list(resolved_chunk_paths),
        merged_path=merged_path,
        merged_shape=tuple(int(v) for v in merged.shape),
        dtype=str(merged.dtype),
        feature_dim=int(merged.shape[1]),
    )


def split_merged_feature(
    *,
    merged_path: str | Path,
    half1_path: str | Path,
    half2_path: str | Path,
    duration_sec: float,
    split_strategy: str = "midpoint",
    split_sec: float | None = None,
    expected_dim: int = 512,
    overwrite: bool = True,
) -> FeatureSplitResult:
    """Split merged_feature.npy into half1_feature.npy and half2_feature.npy.

    feature_fps is calculated from the actual merged row count and video
    duration. It is deliberately not hard-coded.
    """

    merged_path = Path(merged_path)
    half1_path = Path(half1_path)
    half2_path = Path(half2_path)

    if not merged_path.exists():
        raise FeatureSplitError(f"Merged feature does not exist: {merged_path.as_posix()}")

    merged = np.load(merged_path, mmap_mode="r")
    _validate_feature_array(
        merged,
        path=merged_path,
        expected_dim=expected_dim,
        allow_empty=False,
    )

    if duration_sec <= 0:
        raise FeatureSplitError(f"duration_sec must be greater than 0, got {duration_sec}")

    total_rows = int(merged.shape[0])
    feature_fps = total_rows / float(duration_sec)

    resolved_strategy = split_strategy.lower().strip()
    if resolved_strategy == "none":
        # Keep a sensible behavior for callers that choose not to semantically
        # split: physical half files are still generated for the current
        # HighlightFeatureLoader compatibility.
        resolved_strategy = "midpoint"

    if resolved_strategy == "midpoint":
        resolved_split_sec = float(duration_sec) / 2.0
    elif resolved_strategy == "manual":
        if split_sec is None:
            raise FeatureSplitError("split_sec is required when split_strategy='manual'.")
        resolved_split_sec = float(split_sec)
    else:
        raise FeatureSplitError(
            "split_strategy must be one of 'midpoint', 'manual', or 'none'; "
            f"got {split_strategy!r}"
        )

    if not 0 < resolved_split_sec < duration_sec:
        raise FeatureSplitError(
            "split_sec must be inside the video duration: "
            f"split_sec={resolved_split_sec}, duration_sec={duration_sec}"
        )

    split_index = int(round(resolved_split_sec * feature_fps))
    if split_index <= 0 or split_index >= total_rows:
        raise FeatureSplitError(
            "Invalid split_index. The split would create an empty half: "
            f"split_index={split_index}, total_rows={total_rows}, "
            f"split_sec={resolved_split_sec}, feature_fps={feature_fps}"
        )

    half1_path.parent.mkdir(parents=True, exist_ok=True)
    half2_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not half1_path.exists():
        np.save(half1_path, np.asarray(merged[:split_index]))
    if overwrite or not half2_path.exists():
        np.save(half2_path, np.asarray(merged[split_index:]))

    half1 = np.load(half1_path, mmap_mode="r")
    half2 = np.load(half2_path, mmap_mode="r")
    _validate_feature_array(
        half1,
        path=half1_path,
        expected_dim=expected_dim,
        allow_empty=False,
    )
    _validate_feature_array(
        half2,
        path=half2_path,
        expected_dim=expected_dim,
        allow_empty=False,
    )

    return FeatureSplitResult(
        merged_path=merged_path,
        half1_path=half1_path,
        half2_path=half2_path,
        merged_shape=tuple(int(v) for v in merged.shape),
        half1_shape=tuple(int(v) for v in half1.shape),
        half2_shape=tuple(int(v) for v in half2.shape),
        feature_fps=float(feature_fps),
        split_strategy=split_strategy,
        split_sec=float(resolved_split_sec),
        split_index=int(split_index),
        duration_sec=float(duration_sec),
    )


def _validate_feature_array(
    array: Any,
    *,
    path: Path,
    expected_dim: int,
    allow_empty: bool,
) -> None:
    if not hasattr(array, "shape"):
        raise FeatureMergeError(f"Loaded object is not an array: {path.as_posix()}")

    shape = tuple(int(v) for v in array.shape)
    if len(shape) != 2:
        raise FeatureMergeError(
            f"Feature array must be 2-D, got shape={shape} at {path.as_posix()}"
        )

    if not allow_empty and shape[0] <= 0:
        raise FeatureMergeError(f"Feature array is empty: shape={shape} at {path.as_posix()}")

    if shape[1] != expected_dim:
        raise FeatureMergeError(
            "Feature dim mismatch: "
            f"expected {expected_dim}, got {shape[1]} at {path.as_posix()}"
        )
