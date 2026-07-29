from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.ai.tasks.highlight_spotting.adapters.base import (
    ChampionAdapterPreflightReport,
    ChampionCheckpointSummary,
    HighlightRawPrediction,
)
from app.core.paths import get_project_root
from app.domains.action_spotting.errors import (
    ActionSpottingError,
    checkpoint_mismatch,
    class_order_mismatch,
    feature_mismatch,
    inference_policy_incomplete,
    runtime_unavailable,
)

if TYPE_CHECKING:
    from app.ai.tasks.highlight_spotting.feature_loader import HighlightFeatureBundle


CHAMPION_IDENTIFIER = "soccer_spotter_v9"
CHAMPION_CLASS_ORDER = ["goal", "shot", "penalty", "card", "corner"]
CHECKPOINT_CLASS_ORDER = ["Goal", "Shot", "Penalty", "Card", "Corner"]
EXPECTED_CHECKPOINT_SHA256 = (
    "b58db272f1feb43f548e0e8ee82df9b8bdbfe1e441fb14da0376f09ec056150c"
)
EXPECTED_PARAMETER_COUNT = 4_052_880
DEFAULT_CHAMPION_MODEL_DIR = "storage/models/action_spotting/soccer_spotter_v9"
DEFAULT_CHAMPION_CONFIG_DIR = "configs/models/action_spotting/soccer_spotter_v9"


@dataclass(frozen=True)
class V9ArtifactPaths:
    model_dir: Path
    checkpoint_path: Path
    model_config_path: Path
    inference_policy_path: Path
    manifest_path: Path

    def to_metadata(self, *, include_paths: bool = False) -> dict[str, Any]:
        if include_paths:
            return {
                "model_dir": self.model_dir.as_posix(),
                "checkpoint_path": self.checkpoint_path.as_posix(),
                "model_config_path": self.model_config_path.as_posix(),
                "inference_policy_path": self.inference_policy_path.as_posix(),
                "manifest_path": self.manifest_path.as_posix(),
            }
        return {
            "model_dir_exists": self.model_dir.is_dir(),
            "checkpoint_exists": self.checkpoint_path.is_file(),
            "model_config_exists": self.model_config_path.is_file(),
            "inference_policy_exists": self.inference_policy_path.is_file(),
            "manifest_exists": self.manifest_path.is_file(),
        }


@dataclass(frozen=True)
class V9ModelSpec:
    labels: list[str]
    checkpoint_labels: list[str]
    feature_dim: int
    feature_fps: float
    window_size: int
    stride_size: int
    batch_size: int
    base_thresholds: dict[str, float]
    peak_distances_sec: dict[str, float]
    max_events_per_class: dict[str, int]
    max_candidates_per_match: int
    raw_model_config: dict[str, Any]
    raw_inference_policy: dict[str, Any]
    raw_manifest: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": "SoccerSpotterV9",
            "model_type": "tcn_transformer_eventness_uncertainty_offset",
            "labels": self.labels,
            "feature_dim": self.feature_dim,
            "feature_fps": self.feature_fps,
            "window_size": self.window_size,
            "window_sec": self.window_size / self.feature_fps,
            "stride_size": self.stride_size,
            "stride_sec": self.stride_size / self.feature_fps,
            "base_thresholds": self.base_thresholds,
            "peak_distances_sec": self.peak_distances_sec,
            "max_events_per_class": self.max_events_per_class,
            "max_candidates_per_match": self.max_candidates_per_match,
            "champion_identifier": CHAMPION_IDENTIFIER,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        }


@dataclass(frozen=True)
class _HalfTimeline:
    half: int
    timesteps: int
    scores: np.ndarray
    offsets: np.ndarray
    eventness: np.ndarray


@dataclass(frozen=True)
class _Candidate:
    half: int
    label: str
    label_id: int
    timestamp_sec: float
    raw_timestamp_sec: float
    offset_sec: float
    timestep_idx: int
    score: float
    eventness: float


class SoccerSpotterV9Adapter:
    """Production adapter for the supplied SoccerSpotter v9 bundle."""

    def __init__(self, paths: V9ArtifactPaths) -> None:
        self.paths = paths
        self.spec = load_v9_model_spec(paths)
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path | None = None,
    ) -> "SoccerSpotterV9Adapter":
        return cls(resolve_v9_artifact_paths(model_dir or DEFAULT_CHAMPION_MODEL_DIR))

    def preflight(self) -> ChampionAdapterPreflightReport:
        reasons: list[str] = []
        checkpoint = inspect_v9_checkpoint(self.paths.checkpoint_path)
        try:
            self._validate_contract()
            self._load_model(device="cpu")
        except ActionSpottingError as exc:
            reasons.append(f"{exc.code}: {exc.public_message}")
        except Exception as exc:  # pragma: no cover - defensive boundary
            reasons.append(
                "ACTION_SPOTTING_RUNTIME_UNAVAILABLE: "
                f"unexpected v9 preflight failure ({type(exc).__name__})"
            )
        return ChampionAdapterPreflightReport(
            ready_for_real_adapter=not reasons,
            reasons=reasons,
            paths=self.paths.to_metadata(),
            spec=self.spec.to_metadata(),
            checkpoint=checkpoint.to_metadata(),
            implementation_todos=[],
        )

    def predict(
        self,
        *,
        feature_bundle: HighlightFeatureBundle,
        match_duration_sec: float | None = None,
        device: str = "auto",
        batch_size: int | None = None,
        max_candidates: int | None = None,
        debug_min_threshold: float | None = None,
    ) -> list[HighlightRawPrediction]:
        del match_duration_sec
        if debug_min_threshold is not None:
            raise inference_policy_incomplete(
                "v9 threshold overrides are not allowed.",
                requested_threshold=debug_min_threshold,
            )
        if feature_bundle.layout != "halves" or len(feature_bundle.resolved_paths) != 2:
            raise feature_mismatch(
                "v9 inference requires first-half and second-half PCA512 features.",
                feature_bundle=feature_bundle.to_metadata(include_paths=False),
            )
        if not feature_bundle.available:
            raise feature_mismatch(
                "SoccerNet PCA512 feature assets are missing or invalid.",
                feature_bundle=feature_bundle.to_metadata(include_paths=False),
            )
        policy_limit = self.spec.max_candidates_per_match
        if max_candidates is not None and int(max_candidates) != policy_limit:
            raise inference_policy_incomplete(
                "v9 max-candidate policy cannot be overridden.",
                expected=policy_limit,
                actual=int(max_candidates),
            )

        self._validate_contract()
        model, torch_module, resolved_device = self._load_model(device=device)
        resolved_batch_size = int(batch_size or self.spec.batch_size)
        half_predictions: list[_HalfTimeline] = []
        with torch_module.no_grad():
            for half, path in enumerate(feature_bundle.resolved_paths, start=1):
                half_predictions.append(
                    self._predict_half(
                        model=model,
                        torch_module=torch_module,
                        device=resolved_device,
                        path=path,
                        half=half,
                        batch_size=resolved_batch_size,
                    )
                )

        half1_duration = (
            float(half_predictions[0].timesteps) / self.spec.feature_fps
        )
        candidates = self._extract_candidates(half_predictions)
        predictions: list[HighlightRawPrediction] = []
        for item in candidates:
            combined_timestamp = (
                item.timestamp_sec
                if item.half == 1
                else half1_duration + item.timestamp_sec
            )
            predictions.append(
                HighlightRawPrediction(
                    label=item.label,
                    timestamp_sec=combined_timestamp,
                    confidence=item.score,
                    highlight_score=item.score,
                    half=item.half,
                    metadata={
                        "champion_identifier": CHAMPION_IDENTIFIER,
                        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                        "class_id": item.label_id,
                        "half_timestamp_sec": item.timestamp_sec,
                        "raw_half_timestamp_sec": item.raw_timestamp_sec,
                        "offset_sec": item.offset_sec,
                        "offset_unit": "seconds",
                        "timestep_idx": item.timestep_idx,
                        "eventness": item.eventness,
                        "feature_fps": self.spec.feature_fps,
                        "window_size": self.spec.window_size,
                        "stride_size": self.spec.stride_size,
                        "threshold": self.spec.base_thresholds[item.label],
                        "peak_distance_sec": self.spec.peak_distances_sec[item.label],
                        "player_involvement": {
                            "status": "unknown",
                            "basis": "action_spotting_is_temporal_only",
                        },
                    },
                )
            )
        return predictions

    def _validate_contract(self) -> None:
        missing = [
            path.name
            for path in (
                self.paths.checkpoint_path,
                self.paths.model_config_path,
                self.paths.inference_policy_path,
                self.paths.manifest_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise runtime_unavailable(
                "SoccerSpotter v9 bundle is incomplete.",
                missing_files=missing,
            )
        actual_sha = sha256_file(self.paths.checkpoint_path)
        if actual_sha != EXPECTED_CHECKPOINT_SHA256:
            raise checkpoint_mismatch(
                "SoccerSpotter v9 checkpoint integrity validation failed.",
                expected_sha256=EXPECTED_CHECKPOINT_SHA256,
                actual_sha256=actual_sha,
            )
        if self.spec.labels != CHAMPION_CLASS_ORDER:
            raise class_order_mismatch(
                "v9 class order does not match the backend contract.",
                expected=CHAMPION_CLASS_ORDER,
                actual=self.spec.labels,
            )
        if self.spec.checkpoint_labels != CHECKPOINT_CLASS_ORDER:
            raise class_order_mismatch(
                "v9 checkpoint class order is inconsistent.",
                expected=CHECKPOINT_CLASS_ORDER,
                actual=self.spec.checkpoint_labels,
            )
        if (
            self.spec.raw_inference_policy.get("model_id")
            != CHAMPION_IDENTIFIER
            or self.spec.raw_inference_policy.get("checkpoint_sha256")
            != EXPECTED_CHECKPOINT_SHA256
        ):
            raise checkpoint_mismatch(
                "v9 inference policy is paired with a different checkpoint."
            )
        if (
            self.spec.raw_manifest.get("model_id") != CHAMPION_IDENTIFIER
            or self.spec.raw_manifest.get("checkpoint_sha256")
            != EXPECTED_CHECKPOINT_SHA256
            or int(self.spec.raw_manifest.get("parameter_count", 0))
            != EXPECTED_PARAMETER_COUNT
        ):
            raise checkpoint_mismatch(
                "v9 manifest provenance is missing or inconsistent."
            )
        required_predictor = {
            "feature_dim": 512,
            "feature_fps": 2.0,
            "window_size": 128,
            "stride_size": 8,
            "batch_size": 64,
        }
        actual_predictor = {
            "feature_dim": self.spec.feature_dim,
            "feature_fps": self.spec.feature_fps,
            "window_size": self.spec.window_size,
            "stride_size": self.spec.stride_size,
            "batch_size": self.spec.batch_size,
        }
        if actual_predictor != required_predictor:
            raise inference_policy_incomplete(
                "v9 predictor policy is missing or inconsistent.",
                expected=required_predictor,
                actual=actual_predictor,
            )
        required_postprocess = {
            "base_thresholds": {
                "goal": 0.25,
                "shot": 0.55,
                "penalty": 0.20,
                "card": 0.75,
                "corner": 0.30,
            },
            "peak_distances_sec": {
                "goal": 25.0,
                "shot": 8.0,
                "penalty": 35.0,
                "card": 18.0,
                "corner": 10.0,
            },
            "max_events_per_class": {
                "goal": 15,
                "shot": 40,
                "penalty": 8,
                "card": 25,
                "corner": 25,
            },
            "use_adaptive_calibration": False,
            "apply_offset": True,
        }
        raw_postprocess = (
            self.spec.raw_inference_policy.get("postprocess") or {}
        )
        actual_postprocess = {
            "base_thresholds": self.spec.base_thresholds,
            "peak_distances_sec": self.spec.peak_distances_sec,
            "max_events_per_class": self.spec.max_events_per_class,
            "use_adaptive_calibration": raw_postprocess.get(
                "use_adaptive_calibration"
            ),
            "apply_offset": raw_postprocess.get("apply_offset"),
        }
        if actual_postprocess != required_postprocess:
            raise inference_policy_incomplete(
                "v9 postprocess policy is missing or inconsistent.",
                expected=required_postprocess,
                actual=actual_postprocess,
            )
        expected_max = sum(self.spec.max_events_per_class.values())
        if self.spec.max_candidates_per_match != expected_max:
            raise inference_policy_incomplete(
                "v9 candidate limit must equal the class-limit sum.",
                expected=expected_max,
                actual=self.spec.max_candidates_per_match,
            )

    def _load_model(self, *, device: str) -> tuple[Any, Any, Any]:
        try:
            import torch
            from app.ai.vendor.kickclip_v9 import SoccerSpotterV9
        except Exception as exc:
            raise runtime_unavailable(
                "SoccerSpotter v9 runtime could not be imported.",
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc

        resolved_device = _select_device(torch, device)
        if (
            self._model is not None
            and self._device is not None
            and str(self._device) == str(resolved_device)
        ):
            return self._model, self._torch, self._device

        try:
            checkpoint = _torch_load(torch, self.paths.checkpoint_path, resolved_device)
            checkpoint_config = checkpoint.get("config")
            state_dict = checkpoint.get("model_state_dict")
            if checkpoint_config != self.spec.raw_model_config:
                raise checkpoint_mismatch(
                    "v9 checkpoint-embedded config does not match v9_config.json."
                )
            if not isinstance(state_dict, dict):
                raise KeyError("model_state_dict")
            model_config = self.spec.raw_model_config["model"]
            model = SoccerSpotterV9(
                dim=self.spec.feature_dim,
                n_cls=len(self.spec.labels),
                d_model=int(model_config["d_model"]),
                tcn_channels=int(model_config["tcn_channels"]),
                nhead=int(model_config["nhead"]),
                num_transformer_layers=int(
                    model_config["num_transformer_layers"]
                ),
                dim_feedforward=int(model_config["dim_feedforward"]),
                dropout=float(model_config["dropout"]),
                use_eventness_head=bool(model_config["use_eventness_head"]),
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(resolved_device)
            model.eval()
        except ActionSpottingError:
            raise
        except Exception as exc:
            raise checkpoint_mismatch(
                "SoccerSpotter v9 checkpoint could not be loaded strictly.",
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise checkpoint_mismatch(
                "SoccerSpotter v9 parameter count does not match its provenance.",
                expected=EXPECTED_PARAMETER_COUNT,
                actual=parameter_count,
            )
        self._model = model
        self._torch = torch
        self._device = resolved_device
        return model, torch, resolved_device

    def _predict_half(
        self,
        *,
        model: Any,
        torch_module: Any,
        device: Any,
        path: Path,
        half: int,
        batch_size: int,
    ) -> _HalfTimeline:
        try:
            features = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise feature_mismatch(
                "A PCA512 feature file could not be read.",
                half=half,
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc
        if features.ndim != 2 or features.shape[1] != self.spec.feature_dim:
            raise feature_mismatch(
                "v9 features must have shape [T, 512].",
                half=half,
                shape=list(features.shape),
            )

        timesteps = int(features.shape[0])
        starts = get_window_start_indices(
            timesteps,
            self.spec.window_size,
            self.spec.stride_size,
        )
        num_classes = len(self.spec.labels)
        score_sum = np.zeros((timesteps, num_classes), dtype=np.float32)
        count_sum = np.zeros((timesteps, 1), dtype=np.float32)
        offset_sum = np.zeros((timesteps, num_classes), dtype=np.float32)
        eventness_sum = np.zeros(timesteps, dtype=np.float32)

        for batch_start in range(0, len(starts), batch_size):
            batch_indices = starts[batch_start : batch_start + batch_size]
            windows = [
                read_feature_window(features, start, self.spec.window_size)
                for start in batch_indices
            ]
            batch = torch_module.from_numpy(np.stack(windows)).float().to(device)
            logits, offset_mean, _offset_logvar, eventness = model(batch)
            if eventness is None:
                raise checkpoint_mismatch("v9 eventness output is missing.")
            scores = torch_module.sigmoid(logits).detach().cpu().numpy()
            offsets = offset_mean.detach().cpu().numpy()
            eventness_scores = (
                torch_module.sigmoid(eventness).detach().cpu().numpy()
            )
            expected_shape = (
                len(batch_indices),
                self.spec.window_size,
                num_classes,
            )
            if scores.shape != expected_shape or offsets.shape != expected_shape:
                raise class_order_mismatch(
                    "v9 output tensor shape does not match its class contract.",
                    heatmap_shape=list(scores.shape),
                    offset_shape=list(offsets.shape),
                )
            for index, start in enumerate(batch_indices):
                valid_len = min(self.spec.window_size, timesteps - start)
                end = start + valid_len
                score_sum[start:end] += scores[index, :valid_len]
                offset_sum[start:end] += offsets[index, :valid_len]
                eventness_sum[start:end] += eventness_scores[index, :valid_len]
                count_sum[start:end] += 1.0

        safe_count = np.maximum(count_sum, 1.0)
        return _HalfTimeline(
            half=half,
            timesteps=timesteps,
            scores=(score_sum / safe_count).astype(np.float32),
            offsets=(offset_sum / safe_count).astype(np.float32),
            eventness=(
                eventness_sum / np.maximum(count_sum[:, 0], 1.0)
            ).astype(np.float32),
        )

    def _extract_candidates(
        self,
        half_predictions: list[_HalfTimeline],
    ) -> list[_Candidate]:
        try:
            from scipy.signal import find_peaks
        except Exception as exc:
            raise runtime_unavailable(
                "scipy is required for the v9 peak extraction policy.",
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc

        candidates: list[_Candidate] = []
        for class_id, label in enumerate(self.spec.labels):
            class_candidates: list[_Candidate] = []
            threshold = self.spec.base_thresholds[label]
            distance = max(
                1,
                int(self.spec.peak_distances_sec[label] * self.spec.feature_fps),
            )
            for prediction in half_predictions:
                peaks, _ = find_peaks(
                    prediction.scores[:, class_id],
                    height=threshold,
                    distance=distance,
                )
                for raw_index in peaks:
                    index = int(raw_index)
                    raw_timestamp = index / self.spec.feature_fps
                    offset = float(prediction.offsets[index, class_id])
                    class_candidates.append(
                        _Candidate(
                            half=prediction.half,
                            label=label,
                            label_id=class_id,
                            timestamp_sec=max(0.0, raw_timestamp + offset),
                            raw_timestamp_sec=raw_timestamp,
                            offset_sec=offset,
                            timestep_idx=index,
                            score=float(prediction.scores[index, class_id]),
                            eventness=float(prediction.eventness[index]),
                        )
                    )
            class_candidates.sort(key=lambda item: item.score, reverse=True)
            candidates.extend(
                class_candidates[: self.spec.max_events_per_class[label]]
            )
        half1_duration = half_predictions[0].timesteps / self.spec.feature_fps
        return sorted(
            candidates,
            key=lambda item: (
                item.timestamp_sec
                if item.half == 1
                else half1_duration + item.timestamp_sec
            ),
        )


def resolve_v9_artifact_paths(model_dir: str | Path) -> V9ArtifactPaths:
    project_root = get_project_root()
    resolved_model_dir = Path(model_dir)
    if not resolved_model_dir.is_absolute():
        resolved_model_dir = project_root / resolved_model_dir
    config_dir = project_root / DEFAULT_CHAMPION_CONFIG_DIR
    return V9ArtifactPaths(
        model_dir=resolved_model_dir,
        checkpoint_path=resolved_model_dir / "v9_best_model.pth",
        model_config_path=config_dir / "v9_config.json",
        inference_policy_path=config_dir / "inference_policy.json",
        manifest_path=config_dir / "manifest.json",
    )


def load_v9_model_spec(paths: V9ArtifactPaths) -> V9ModelSpec:
    model_config = _load_json(paths.model_config_path)
    policy = _load_json(paths.inference_policy_path)
    manifest = _load_json(paths.manifest_path)
    predictor = policy.get("predictor") or {}
    postprocess = policy.get("postprocess") or {}
    return V9ModelSpec(
        labels=list(policy.get("class_order") or []),
        checkpoint_labels=list(policy.get("checkpoint_class_order") or []),
        feature_dim=int(predictor.get("feature_dim", 0)),
        feature_fps=float(predictor.get("feature_fps", 0)),
        window_size=int(predictor.get("window_size", 0)),
        stride_size=int(predictor.get("stride_size", 0)),
        batch_size=int(predictor.get("batch_size", 0)),
        base_thresholds={
            str(key): float(value)
            for key, value in (postprocess.get("base_thresholds") or {}).items()
        },
        peak_distances_sec={
            str(key): float(value)
            for key, value in (
                postprocess.get("peak_distances_sec") or {}
            ).items()
        },
        max_events_per_class={
            str(key): int(value)
            for key, value in (
                postprocess.get("max_events_per_class") or {}
            ).items()
        },
        max_candidates_per_match=int(
            postprocess.get("max_candidates_per_match", 0)
        ),
        raw_model_config=model_config,
        raw_inference_policy=policy,
        raw_manifest=manifest,
    )


def inspect_v9_checkpoint(checkpoint_path: str | Path) -> ChampionCheckpointSummary:
    path = Path(checkpoint_path)
    if not path.is_file():
        return ChampionCheckpointSummary(loadable=False, reason="checkpoint missing")
    try:
        import torch

        checkpoint = _torch_load(torch, path, "cpu")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            return ChampionCheckpointSummary(
                loadable=False,
                reason="checkpoint has no model_state_dict",
                checkpoint_type=type(checkpoint).__name__,
                checkpoint_keys=sorted(checkpoint),
            )
        return ChampionCheckpointSummary(
            loadable=True,
            checkpoint_type=type(checkpoint).__name__,
            checkpoint_keys=sorted(checkpoint),
            state_dict_key="model_state_dict",
            state_dict_num_tensors=len(state_dict),
            state_dict_sample_keys=list(state_dict)[:12],
            tensor_shapes={
                key: list(value.shape)
                for key, value in list(state_dict.items())[:12]
                if hasattr(value, "shape")
            },
            metrics={"best_frame_map": checkpoint.get("best_frame_map")},
        )
    except Exception as exc:
        return ChampionCheckpointSummary(
            loadable=False,
            reason=f"{type(exc).__name__}: {exc}",
        )


def get_window_start_indices(
    num_timesteps: int,
    window_size: int,
    stride_size: int,
) -> list[int]:
    if num_timesteps <= 0:
        return []
    if num_timesteps <= window_size:
        return [0]
    last_start = num_timesteps - window_size
    starts = list(range(0, last_start + 1, stride_size))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def read_feature_window(
    feature_array: np.ndarray,
    start: int,
    window_size: int,
) -> np.ndarray:
    window = np.zeros(
        (window_size, int(feature_array.shape[1])),
        dtype=np.float32,
    )
    end = min(int(feature_array.shape[0]), start + window_size)
    if start < end:
        window[: end - start] = np.asarray(
            feature_array[start:end],
            dtype=np.float32,
        )
    return window


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_device(torch_module: Any, name: str) -> Any:
    if name == "auto":
        return torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
    if name == "cuda" and not torch_module.cuda.is_available():
        raise runtime_unavailable("CUDA was requested but is unavailable.")
    if name not in {"cpu", "cuda"}:
        raise runtime_unavailable(
            "SoccerSpotter v9 supports only cpu, cuda, or auto devices.",
            requested_device=name,
        )
    return torch_module.device(name)


def _torch_load(torch_module: Any, path: Path, device: Any) -> dict[str, Any]:
    try:
        value = torch_module.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        value = torch_module.load(path, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(value).__name__}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}
