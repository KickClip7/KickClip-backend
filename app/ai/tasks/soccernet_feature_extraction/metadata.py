from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ai.tasks.soccernet_feature_extraction.chunk_runner import (
    FeatureChunkRunResult,
    FeatureChunkSpec,
)
from app.ai.tasks.soccernet_feature_extraction.config import (
    SoccerNetFeatureExtractionConfig,
)
from app.ai.tasks.soccernet_feature_extraction.splitter import (
    FeatureMergeResult,
    FeatureSplitResult,
)
from app.ai.tasks.soccernet_feature_extraction.video_probe import VideoProbeResult
from app.core.paths import get_project_root


@dataclass(frozen=True)
class FeatureMetadataFile:
    path: Path
    metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "metadata": self.metadata,
        }


def build_feature_metadata(
    *,
    config: SoccerNetFeatureExtractionConfig,
    video_probe: VideoProbeResult,
    merge_result: FeatureMergeResult,
    split_result: FeatureSplitResult,
    source_asset_id: str | None = None,
    source_video_path: str | Path | None = None,
    match_id: str | None = None,
    chunk_results: list[FeatureChunkRunResult] | None = None,
    chunk_specs: list[FeatureChunkSpec] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build serializable feature metadata for feature_metadata.json."""

    source_video_path = source_video_path or video_probe.path
    chunk_payload: list[dict[str, Any]]
    if chunk_results is not None:
        chunk_payload = [result.to_metadata(include_command=False) for result in chunk_results]
    elif chunk_specs is not None:
        chunk_payload = [spec.to_metadata() for spec in chunk_specs]
    else:
        chunk_payload = []

    metadata: dict[str, Any] = {
        "task_type": "SOCCERNET_FEATURE_EXTRACTION",
        "extractor_name": config.extractor_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_id": match_id,
        "source_asset_id": source_asset_id,
        "source_video_path": _path_to_text(source_video_path),
        "source_video_resolved_path": video_probe.path.as_posix(),
        "source_video_duration_sec": video_probe.duration_sec,
        "source_video_fps": video_probe.fps,
        "source_video_frame_count": video_probe.frame_count,
        "source_video_width": video_probe.width,
        "source_video_height": video_probe.height,
        "chunk_sec": config.chunk_sec,
        "num_chunks": merge_result.num_chunks,
        "chunks": chunk_payload,
        "merged_path": _project_relative_or_absolute(split_result.merged_path),
        "half1_path": _project_relative_or_absolute(split_result.half1_path),
        "half2_path": _project_relative_or_absolute(split_result.half2_path),
        "merged_shape": list(split_result.merged_shape),
        "half1_shape": list(split_result.half1_shape),
        "half2_shape": list(split_result.half2_shape),
        "feature_dim": merge_result.feature_dim,
        "feature_dtype": merge_result.dtype,
        "feature_fps": split_result.feature_fps,
        "split_strategy": split_result.split_strategy,
        "split_sec": split_result.split_sec,
        "split_index": split_result.split_index,
        "pca_path": _project_relative_or_absolute(config.pca_path),
        "pca_scaler_path": _project_relative_or_absolute(config.pca_scaler_path),
        "video_feature_extractor": _project_relative_or_absolute(
            config.video_feature_extractor
        ),
        "sn_spotting_root": _project_relative_or_absolute(config.sn_spotting_root),
        "config": config.to_metadata(),
    }

    if extra:
        metadata["extra"] = extra

    return metadata


def save_feature_metadata(
    *,
    metadata: dict[str, Any],
    output_path: str | Path,
) -> FeatureMetadataFile:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return FeatureMetadataFile(path=output_path, metadata=metadata)


def build_and_save_feature_metadata(
    *,
    config: SoccerNetFeatureExtractionConfig,
    video_probe: VideoProbeResult,
    merge_result: FeatureMergeResult,
    split_result: FeatureSplitResult,
    output_path: str | Path,
    source_asset_id: str | None = None,
    source_video_path: str | Path | None = None,
    match_id: str | None = None,
    chunk_results: list[FeatureChunkRunResult] | None = None,
    chunk_specs: list[FeatureChunkSpec] | None = None,
    extra: dict[str, Any] | None = None,
) -> FeatureMetadataFile:
    metadata = build_feature_metadata(
        config=config,
        video_probe=video_probe,
        merge_result=merge_result,
        split_result=split_result,
        source_asset_id=source_asset_id,
        source_video_path=source_video_path,
        match_id=match_id,
        chunk_results=chunk_results,
        chunk_specs=chunk_specs,
        extra=extra,
    )
    return save_feature_metadata(metadata=metadata, output_path=output_path)


def _path_to_text(path: str | Path) -> str:
    if isinstance(path, Path):
        return _project_relative_or_absolute(path)
    return str(path)


def _project_relative_or_absolute(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(get_project_root().resolve()).as_posix()
    except Exception:
        return path.as_posix()
