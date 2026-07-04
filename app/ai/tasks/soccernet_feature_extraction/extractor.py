from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ai.tasks.soccernet_feature_extraction.chunk_runner import (
    FeatureChunkRunResult,
    SoccerNetChunkRunner,
)
from app.ai.tasks.soccernet_feature_extraction.config import (
    SoccerNetFeatureExtractionConfig,
)
from app.ai.tasks.soccernet_feature_extraction.metadata import (
    FeatureMetadataFile,
    build_and_save_feature_metadata,
)
from app.ai.tasks.soccernet_feature_extraction.splitter import (
    FeatureMergeResult,
    FeatureSplitResult,
    merge_chunk_features,
    split_merged_feature,
)
from app.ai.tasks.soccernet_feature_extraction.video_probe import (
    VideoProbeResult,
    probe_video,
)
from app.storage.local_storage import LocalStorage
from app.storage.workspace import (
    get_match_soccernet_feature_chunks_subdir,
    get_match_soccernet_features_subdir,
)


@dataclass(frozen=True)
class SoccerNetFeatureExtractionPaths:
    base_dir: Path
    chunks_dir: Path
    merged_path: Path
    half1_path: Path
    half2_path: Path
    metadata_path: Path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "base_dir": self.base_dir.as_posix(),
            "chunks_dir": self.chunks_dir.as_posix(),
            "merged_path": self.merged_path.as_posix(),
            "half1_path": self.half1_path.as_posix(),
            "half2_path": self.half2_path.as_posix(),
            "metadata_path": self.metadata_path.as_posix(),
        }


@dataclass(frozen=True)
class SoccerNetFeatureExtractionResult:
    paths: SoccerNetFeatureExtractionPaths
    video_probe: VideoProbeResult
    chunk_results: list[FeatureChunkRunResult]
    merge_result: FeatureMergeResult
    split_result: FeatureSplitResult
    metadata_file: FeatureMetadataFile

    @property
    def merged_path(self) -> Path:
        return self.paths.merged_path

    @property
    def half1_path(self) -> Path:
        return self.paths.half1_path

    @property
    def half2_path(self) -> Path:
        return self.paths.half2_path

    @property
    def metadata_path(self) -> Path:
        return self.paths.metadata_path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "paths": self.paths.to_metadata(),
            "video_probe": self.video_probe.to_metadata(),
            "chunks": [result.to_metadata(include_command=False) for result in self.chunk_results],
            "merge": self.merge_result.to_metadata(),
            "split": self.split_result.to_metadata(),
            "metadata_file": {
                "path": self.metadata_file.path.as_posix(),
            },
        }


class SoccerNetFeatureExtractor:
    """Orchestrate SoccerNet-style feature extraction for one raw video.

    This class performs file-level work only:
    probe -> chunk extraction -> merge -> split -> metadata JSON.

    It deliberately does not create MediaAsset or Artifact database rows. That is
    the responsibility of SoccerNetFeatureExtractionTask in the next round.
    """

    def __init__(
        self,
        config: SoccerNetFeatureExtractionConfig,
        *,
        storage: LocalStorage | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or LocalStorage()
        self.chunk_runner = SoccerNetChunkRunner(config)

    def build_paths(
        self,
        *,
        match_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> SoccerNetFeatureExtractionPaths:
        if output_dir is None:
            if not match_id:
                raise ValueError("match_id is required when output_dir is not provided.")
            base_dir = self.storage.storage_root / get_match_soccernet_features_subdir(
                match_id
            )
            chunks_dir = self.storage.storage_root / get_match_soccernet_feature_chunks_subdir(
                match_id
            )
        else:
            base_dir = Path(output_dir)
            chunks_dir = base_dir / "chunks"

        return SoccerNetFeatureExtractionPaths(
            base_dir=base_dir,
            chunks_dir=chunks_dir,
            merged_path=base_dir / "merged_feature.npy",
            half1_path=base_dir / "half1_feature.npy",
            half2_path=base_dir / "half2_feature.npy",
            metadata_path=base_dir / "feature_metadata.json",
        )

    def extract(
        self,
        *,
        video_path: str | Path,
        match_id: str | None = None,
        source_asset_id: str | None = None,
        source_video_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        video_probe_result: VideoProbeResult | None = None,
    ) -> SoccerNetFeatureExtractionResult:
        if self.config.should_skip:
            raise RuntimeError("SoccerNet feature extraction is disabled or skipped by config.")

        paths = self.build_paths(match_id=match_id, output_dir=output_dir)
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        paths.chunks_dir.mkdir(parents=True, exist_ok=True)

        probe_result = video_probe_result or probe_video(video_path)

        chunk_results = self.chunk_runner.run_chunks(
            video_path=video_path,
            total_duration_sec=probe_result.duration_sec,
            output_dir=paths.chunks_dir,
        )

        merge_result = merge_chunk_features(
            chunk_paths=[result.output_path for result in chunk_results],
            merged_path=paths.merged_path,
            expected_dim=self.config.output_dim,
            overwrite=self.config.overwrite,
        )

        split_sec = (
            self.config.halftime_split_sec
            if self.config.split_strategy == "manual"
            else None
        )
        split_result = split_merged_feature(
            merged_path=paths.merged_path,
            half1_path=paths.half1_path,
            half2_path=paths.half2_path,
            duration_sec=probe_result.duration_sec,
            split_strategy=self.config.split_strategy,
            split_sec=split_sec,
            expected_dim=self.config.output_dim,
            overwrite=self.config.overwrite,
        )

        metadata_file = build_and_save_feature_metadata(
            config=self.config,
            video_probe=probe_result,
            merge_result=merge_result,
            split_result=split_result,
            output_path=paths.metadata_path,
            source_asset_id=source_asset_id,
            source_video_path=source_video_path or video_path,
            match_id=match_id,
            chunk_results=chunk_results,
        )

        return SoccerNetFeatureExtractionResult(
            paths=paths,
            video_probe=probe_result,
            chunk_results=chunk_results,
            merge_result=merge_result,
            split_result=split_result,
            metadata_file=metadata_file,
        )
