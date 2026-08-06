from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from app.ai.tasks.soccernet_feature_extraction.config import (
    SoccerNetFeatureExtractionConfig,
)


@dataclass(frozen=True)
class FeatureChunkSpec:
    """A single temporal chunk passed to VideoFeatureExtractor.py."""

    index: int
    start_sec: float
    duration_sec: float
    output_path: Path

    @property
    def filename(self) -> str:
        return self.output_path.name

    def to_metadata(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_sec": self.start_sec,
            "duration_sec": self.duration_sec,
            "output_path": self.output_path.as_posix(),
            "filename": self.filename,
        }


@dataclass(frozen=True)
class FeatureChunkRunResult:
    """Execution result for one chunk extraction subprocess."""

    spec: FeatureChunkSpec
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    skipped: bool = False

    @property
    def output_path(self) -> Path:
        return self.spec.output_path

    def to_metadata(self, *, include_command: bool = True) -> dict[str, Any]:
        payload = {
            "chunk": self.spec.to_metadata(),
            "returncode": self.returncode,
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
            "skipped": self.skipped,
        }
        if include_command:
            payload["command"] = self.command
        return payload


class FeatureChunkExtractionError(RuntimeError):
    """Raised when VideoFeatureExtractor.py fails for a chunk."""

    def __init__(
        self,
        message: str,
        *,
        spec: FeatureChunkSpec | None = None,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        details: list[str] = [message]
        if spec is not None:
            details.append(
                "chunk="
                f"{spec.index}, start={spec.start_sec:.3f}, "
                f"duration={spec.duration_sec:.3f}, "
                f"output={spec.output_path.as_posix()}"
            )
        if returncode is not None:
            details.append(f"returncode={returncode}")
        if command is not None:
            details.append(f"command={command}")
        if stdout:
            details.append(f"stdout_tail={_tail(stdout)}")
        if stderr:
            details.append(f"stderr_tail={_tail(stderr)}")
        super().__init__(" | ".join(details))
        self.spec = spec
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SoccerNetChunkRunner:
    """Run SoccerNet sn-spotting feature extraction chunk by chunk."""

    def __init__(self, config: SoccerNetFeatureExtractionConfig) -> None:
        self.config = config

    def build_chunk_specs(
        self,
        *,
        total_duration_sec: float,
        output_dir: str | Path,
    ) -> list[FeatureChunkSpec]:
        return build_chunk_specs(
            total_duration_sec=total_duration_sec,
            chunk_sec=self.config.chunk_sec,
            output_dir=output_dir,
        )

    def run_chunks(
        self,
        *,
        video_path: str | Path,
        total_duration_sec: float,
        output_dir: str | Path,
    ) -> list[FeatureChunkRunResult]:
        self.validate_environment(video_path=video_path)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        specs = self.build_chunk_specs(
            total_duration_sec=total_duration_sec,
            output_dir=output_dir,
        )
        if not specs:
            raise FeatureChunkExtractionError(
                f"No chunks were generated for duration={total_duration_sec}."
            )

        results: list[FeatureChunkRunResult] = []
        for spec in specs:
            result = self.run_one_chunk(video_path=video_path, spec=spec)
            results.append(result)

        return results

    def run_one_chunk(
        self,
        *,
        video_path: str | Path,
        spec: FeatureChunkSpec,
    ) -> FeatureChunkRunResult:
        video_path = Path(video_path)
        spec.output_path.parent.mkdir(parents=True, exist_ok=True)

        command = self.build_command(video_path=video_path, spec=spec)

        # Cache-friendly behavior: if a chunk already exists and the current
        # config does not request overwrite, keep it. The force mode sets
        # config.overwrite=True in config.py.
        if spec.output_path.exists() and not self.config.overwrite:
            return FeatureChunkRunResult(
                spec=spec,
                command=command,
                returncode=0,
                stdout="",
                stderr="",
                skipped=True,
            )

        completed = subprocess.run(
            command,
            cwd=str(self._subprocess_cwd()),
            capture_output=True,
            text=True,
            check=False,
        )

        result = FeatureChunkRunResult(
            spec=spec,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            skipped=False,
        )

        if completed.returncode != 0:
            raise FeatureChunkExtractionError(
                "VideoFeatureExtractor.py failed.",
                spec=spec,
                command=command,
                returncode=completed.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        if not spec.output_path.exists():
            raise FeatureChunkExtractionError(
                "VideoFeatureExtractor.py finished successfully but chunk output was not created.",
                spec=spec,
                command=command,
                returncode=completed.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        self._validate_chunk_output(
            output_path=spec.output_path,
            spec=spec,
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        return result

    def build_command(
        self,
        *,
        video_path: str | Path,
        spec: FeatureChunkSpec,
    ) -> list[str]:
        command = [
            self._resolve_python_executable(),
            str(self.config.video_feature_extractor),
            "--path_video",
            str(Path(video_path)),
            "--path_features",
            str(spec.output_path),
            "--start",
            _format_seconds(spec.start_sec),
            "--duration",
            _format_seconds(spec.duration_sec),
            "--PCA",
            str(self.config.pca_path),
            "--PCA_scaler",
            str(self.config.pca_scaler_path),
            "--batch_size",
            str(self.config.batch_size),
            "--overwrite",
        ]

        return command

    def validate_environment(self, *, video_path: str | Path) -> None:
        video_path = Path(video_path)
        required_files = {
            "video": video_path,
            "video_feature_extractor": self.config.video_feature_extractor,
            "pca_path": self.config.pca_path,
            "pca_scaler_path": self.config.pca_scaler_path,
        }

        missing = [
            f"{name}={path.as_posix()}"
            for name, path in required_files.items()
            if not path.exists()
        ]
        if missing:
            raise FeatureChunkExtractionError(
                "Required file(s) for SoccerNet feature extraction are missing: "
                + ", ".join(missing)
            )

        if not video_path.is_file():
            raise FeatureChunkExtractionError(
                f"Input video path is not a file: {video_path.as_posix()}"
            )

        if not self.config.video_feature_extractor.is_file():
            raise FeatureChunkExtractionError(
                "video_feature_extractor is not a file: "
                f"{self.config.video_feature_extractor.as_posix()}"
            )

        python_executable = self._resolve_python_executable()
        probe_code = (
            "import tensorflow; import SoccerNet; import cv2; import sklearn; "
            "import skvideo.io; import imutils; import numpy; "
            "print('feature-runtime-ok')"
        )
        try:
            completed = subprocess.run(
                [python_executable, "-c", probe_code],
                cwd=str(self._subprocess_cwd()),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.runtime_probe_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise FeatureChunkExtractionError(
                "SoccerNet feature extraction runtime probe timed out. "
                f"python={python_executable}"
            ) from exc

        if completed.returncode != 0:
            raise FeatureChunkExtractionError(
                "SoccerNet feature extraction Python runtime is not ready. "
                "Install the feature-extraction dependencies in the Python used by "
                "ACTION_SPOTTING_PYTHON_EXECUTABLE (or the backend Python when unset). "
                f"python={python_executable} | "
                f"stdout_tail={_tail(completed.stdout or '')} | "
                f"stderr_tail={_tail(completed.stderr or '')}"
            )

    def _resolve_python_executable(self) -> str:
        configured = self.config.python_executable.strip()
        candidate = Path(configured)
        if candidate.is_file():
            return str(candidate)

        discovered = shutil.which(configured)
        if discovered:
            return discovered

        raise FeatureChunkExtractionError(
            "Configured SoccerNet feature extraction Python executable was not found: "
            f"{configured}"
        )

    def _subprocess_cwd(self) -> Path:
        if self.config.sn_spotting_root.exists() and self.config.sn_spotting_root.is_dir():
            return self.config.sn_spotting_root
        return self.config.video_feature_extractor.parent

    def _validate_chunk_output(
        self,
        *,
        output_path: Path,
        spec: FeatureChunkSpec,
        command: list[str],
        stdout: str,
        stderr: str,
    ) -> None:
        try:
            array = np.load(output_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise FeatureChunkExtractionError(
                f"Failed to load generated chunk feature: {exc}",
                spec=spec,
                command=command,
                returncode=0,
                stdout=stdout,
                stderr=stderr,
            ) from exc

        shape = tuple(int(v) for v in array.shape)

        if len(shape) == 3 and shape[0] == 1:
            feature_dim = shape[2]
        elif len(shape) == 2:
            feature_dim = shape[1]
        else:
            raise FeatureChunkExtractionError(
                "Generated chunk feature has invalid shape. "
                f"Expected (T, {self.config.output_dim}) or (1, T, {self.config.output_dim}), "
                f"got shape={shape}.",
                spec=spec,
                command=command,
                returncode=0,
                stdout=stdout,
                stderr=stderr,
            )

        if feature_dim != self.config.output_dim:
            raise FeatureChunkExtractionError(
                "Generated chunk feature dimension does not match config.output_dim. "
                f"Expected {self.config.output_dim}, got {feature_dim}. "
                "This usually means PCA512 was not applied. "
                "Check VideoFeatureExtractor.py PCA arguments and PCA file paths.",
                spec=spec,
                command=command,
                returncode=0,
                stdout=stdout,
                stderr=stderr,
            )
    

def build_chunk_specs(
    *,
    total_duration_sec: float,
    chunk_sec: float,
    output_dir: str | Path,
) -> list[FeatureChunkSpec]:
    if not math.isfinite(total_duration_sec) or total_duration_sec <= 0:
        raise ValueError("total_duration_sec must be greater than 0.")
    if not math.isfinite(chunk_sec) or chunk_sec <= 0:
        raise ValueError("chunk_sec must be greater than 0.")

    output_dir = Path(output_dir)
    num_chunks = int(math.ceil(total_duration_sec / chunk_sec))
    specs: list[FeatureChunkSpec] = []

    for index in range(num_chunks):
        start_sec = index * chunk_sec
        duration_sec = min(chunk_sec, total_duration_sec - start_sec)

        # Floating point safety for extremely short trailing chunks.
        if duration_sec <= 0:
            continue

        specs.append(
            FeatureChunkSpec(
                index=index,
                start_sec=float(start_sec),
                duration_sec=float(duration_sec),
                output_path=output_dir / f"chunk_{index:03d}.npy",
            )
        )

    return specs


def _format_seconds(value: float) -> str:
    # Keep enough precision for chunk boundaries while avoiding noisy repr values.
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _tail(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    return text[-max_chars:]
