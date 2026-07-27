from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import yaml

from app.ai.tasks.highlight_spotting.adapters.base import (
    ChampionAdapterPreflightReport,
    ChampionArtifactPaths,
    ChampionCheckpointSummary,
    ChampionModelSpec,
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


CHAMPION_IDENTIFIER = "sampling_v1_loss_v2_ms_stem_v1"
CHAMPION_CLASS_ORDER = [
    "goal",
    "shot",
    "foul",
    "card",
    "free_kick",
    "corner",
]
EXPECTED_CHECKPOINT_SHA256 = (
    "c3aa72c3d5be98fb8c6104818da69d2e8ac9a993696830fa1a0588cd58ffaee1"
)
DEFAULT_CHAMPION_MODEL_DIR = (
    "storage/models/action_spotting/sampling_v1_loss_v2_ms_stem_v1"
)
DEFAULT_CHAMPION_CONFIG_DIR = (
    "configs/models/action_spotting/sampling_v1_loss_v2_ms_stem_v1"
)


@dataclass(frozen=True)
class _HalfTimeline:
    half: int
    timesteps: int
    scores: np.ndarray
    offsets: np.ndarray


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


class SoccerHighlightFormerAdapter:
    """Checkpoint-paired Champion adapter using the original model contract."""

    def __init__(self, paths: ChampionArtifactPaths) -> None:
        self.paths = paths
        self.spec = load_champion_model_spec(paths)
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_model_dir(
        cls, model_dir: str | Path | None = None
    ) -> "SoccerHighlightFormerAdapter":
        return cls(
            resolve_champion_artifact_paths(model_dir or DEFAULT_CHAMPION_MODEL_DIR)
        )

    def preflight(self) -> ChampionAdapterPreflightReport:
        reasons: list[str] = []
        checkpoint = inspect_champion_checkpoint(self.paths.checkpoint_path)

        try:
            self._validate_contract()
            self._load_model(device="cpu")
        except ActionSpottingError as exc:
            reasons.append(f"{exc.code}: {exc.public_message}")
        except Exception as exc:  # pragma: no cover - defensive boundary
            reasons.append(
                "ACTION_SPOTTING_RUNTIME_UNAVAILABLE: "
                f"unexpected Champion preflight failure ({type(exc).__name__})"
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
                "Champion threshold overrides are not allowed.",
                requested_threshold=debug_min_threshold,
            )
        if feature_bundle.layout != "halves" or len(feature_bundle.resolved_paths) != 2:
            raise feature_mismatch(
                "Champion inference requires explicit first-half and second-half PCA512 features.",
                feature_bundle=feature_bundle.to_metadata(include_paths=False),
            )
        if not feature_bundle.available:
            raise feature_mismatch(
                "SoccerNet PCA512 feature assets are missing or invalid.",
                feature_bundle=feature_bundle.to_metadata(include_paths=False),
            )

        self._validate_contract()
        model, torch_module, resolved_device = self._load_model(device=device)
        requested_batch_size = int(batch_size or self.spec.raw_inference_policy[
            "predictor"
        ]["batch_size"])
        policy_limit = int(self.spec.max_candidates_per_match or 80)
        if max_candidates is not None and int(max_candidates) != policy_limit:
            raise inference_policy_incomplete(
                "Champion max-candidate policy cannot be overridden.",
                expected=policy_limit,
                actual=int(max_candidates),
            )

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
                        batch_size=requested_batch_size,
                    )
                )

        candidates: list[_Candidate] = []
        for half_prediction in half_predictions:
            candidates.extend(self._extract_half_candidates(half_prediction))
        candidates = self._apply_class_aware_nms(candidates)[:policy_limit]

        half1_duration = (
            float(half_predictions[0].timesteps) / float(self.spec.feature_fps or 2.0)
        )
        return [
            HighlightRawPrediction(
                label=item.label,
                timestamp_sec=(
                    item.timestamp_sec if item.half == 1
                    else half1_duration + item.timestamp_sec
                ),
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
                    "feature_fps": self.spec.feature_fps,
                    "window_size": self.spec.window_size,
                    "stride_size": self.spec.stride_size,
                    "threshold": self.spec.threshold,
                    "local_peak_window_sec": self.spec.local_peak_window_sec,
                    "nms_window_sec": self.spec.nms_window_sec,
                    "player_involvement": {
                        "status": "unknown",
                        "basis": "action_spotting_is_temporal_only",
                    },
                },
            )
            for item in candidates
        ]

    def _validate_contract(self) -> None:
        if not self.paths.checkpoint_path.is_file():
            raise runtime_unavailable(
                "Champion checkpoint is unavailable.",
                missing_file=self.paths.checkpoint_path.name,
            )
        if not self.paths.inference_policy_path.is_file():
            raise inference_policy_incomplete(
                "Champion inference policy is unavailable.",
                missing_file=self.paths.inference_policy_path.name,
            )
        paired_configs = [
            self.paths.model_config_path,
            self.paths.train_config_path,
            self.paths.config_snapshot_path,
            self.paths.manifest_path,
        ]
        missing_configs = [
            path.name for path in paired_configs if not path.is_file()
        ]
        if missing_configs:
            raise checkpoint_mismatch(
                "Champion checkpoint-paired configuration is unavailable.",
                missing_files=missing_configs,
            )

        actual_sha = sha256_file(self.paths.checkpoint_path)
        if actual_sha != EXPECTED_CHECKPOINT_SHA256:
            raise checkpoint_mismatch(
                "Champion checkpoint integrity validation failed.",
                expected_sha256=EXPECTED_CHECKPOINT_SHA256,
                actual_sha256=actual_sha,
            )
        if self.spec.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise checkpoint_mismatch(
                "Champion checkpoint and inference policy do not match.",
                expected_sha256=EXPECTED_CHECKPOINT_SHA256,
                policy_sha256=self.spec.checkpoint_sha256,
            )
        if self.spec.labels != CHAMPION_CLASS_ORDER:
            raise class_order_mismatch(
                "Champion class order does not match the checkpoint contract.",
                expected=CHAMPION_CLASS_ORDER,
                actual=self.spec.labels,
            )

        snapshot_labels = (
            self.spec.raw_config_snapshot.get("data_config", {})
            .get("labels", {})
            .get("classes")
        )
        if snapshot_labels != CHAMPION_CLASS_ORDER:
            raise class_order_mismatch(
                "Champion training snapshot class order is inconsistent.",
                expected=CHAMPION_CLASS_ORDER,
                actual=snapshot_labels,
            )

        required_policy = {
            "feature_dim": 512,
            "feature_fps": 2.0,
            "window_size": 128,
            "stride_size": 32,
            "offset_merge": "weighted",
            "threshold": 0.2,
            "local_peak_window_sec": 3.0,
            "nms_window_sec": 10.0,
            "apply_offset": True,
            "class_aware_nms": True,
            "max_candidates_per_class_per_half": 100,
            "max_candidates_per_match": 80,
        }
        actual_policy = {
            "feature_dim": self.spec.feature_dim,
            "feature_fps": self.spec.feature_fps,
            "window_size": self.spec.window_size,
            "stride_size": self.spec.stride_size,
            "offset_merge": self.spec.offset_merge,
            "threshold": self.spec.threshold,
            "local_peak_window_sec": self.spec.local_peak_window_sec,
            "nms_window_sec": self.spec.nms_window_sec,
            "apply_offset": self.spec.apply_offset,
            "class_aware_nms": self.spec.class_aware_nms,
            "max_candidates_per_class_per_half": (
                self.spec.max_candidates_per_class_per_half
            ),
            "max_candidates_per_match": self.spec.max_candidates_per_match,
        }
        if actual_policy != required_policy:
            raise inference_policy_incomplete(
                "Champion inference policy is missing or inconsistent.",
                expected=required_policy,
                actual=actual_policy,
            )

    def _load_model(self, *, device: str) -> tuple[Any, Any, Any]:
        try:
            import torch
            from app.ai.vendor.kickclip_champion.models import (
                build_soccer_highlight_former_from_config,
            )
        except Exception as exc:
            raise runtime_unavailable(
                "Champion model runtime could not be imported.",
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
            model = build_soccer_highlight_former_from_config(
                self.spec.raw_model_config
            )
            try:
                checkpoint = torch.load(
                    self.paths.checkpoint_path,
                    map_location=resolved_device,
                    weights_only=False,
                )
            except TypeError:  # torch < 2.6
                checkpoint = torch.load(
                    self.paths.checkpoint_path, map_location=resolved_device
                )
            state_dict = checkpoint.get("model_state_dict")
            if not isinstance(state_dict, dict):
                raise KeyError("model_state_dict")
            embedded_classes = (
                checkpoint.get("configs", {})
                .get("data_config", {})
                .get("labels", {})
                .get("classes")
            )
            if embedded_classes != CHAMPION_CLASS_ORDER:
                raise class_order_mismatch(
                    "Champion checkpoint embeds a different class order.",
                    expected=CHAMPION_CLASS_ORDER,
                    actual=embedded_classes,
                )
            model.load_state_dict(state_dict, strict=True)
            model.to(resolved_device)
            model.eval()
        except ActionSpottingError:
            raise
        except Exception as exc:
            raise checkpoint_mismatch(
                "Champion checkpoint could not be loaded strictly.",
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 4_098_828:
            raise checkpoint_mismatch(
                "Champion model parameter count does not match its provenance.",
                expected=4_098_828,
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
            feature_array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise feature_mismatch(
                "A PCA512 feature file could not be read.",
                half=half,
                exception_type=type(exc).__name__,
                detail=str(exc),
            ) from exc
        if feature_array.ndim != 2 or feature_array.shape[1] != 512:
            raise feature_mismatch(
                "Champion features must have shape [T, 512].",
                half=half,
                shape=list(feature_array.shape),
            )
        timesteps = int(feature_array.shape[0])
        starts = get_window_start_indices(
            timesteps,
            int(self.spec.window_size or 128),
            int(self.spec.stride_size or 32),
        )
        num_classes = len(CHAMPION_CLASS_ORDER)
        score_sum = np.zeros((timesteps, num_classes), dtype=np.float32)
        count_sum = np.zeros((timesteps, 1), dtype=np.float32)
        offset_sum = np.zeros((timesteps, num_classes), dtype=np.float32)
        offset_weight_sum = np.zeros((timesteps, num_classes), dtype=np.float32)

        for batch_start in range(0, len(starts), batch_size):
            batch_indices = starts[batch_start : batch_start + batch_size]
            windows = [
                read_feature_window(
                    feature_array,
                    start,
                    int(self.spec.window_size or 128),
                )
                for start in batch_indices
            ]
            features = torch_module.from_numpy(np.stack(windows)).float().to(device)
            outputs = model(features)
            if "heatmap_logits" not in outputs or "offset" not in outputs:
                raise checkpoint_mismatch(
                    "Champion output heads do not match the checkpoint contract.",
                    output_keys=sorted(outputs),
                )
            scores = (
                torch_module.sigmoid(outputs["heatmap_logits"]).detach().cpu().numpy()
            )
            offsets = outputs["offset"].detach().cpu().numpy()
            if scores.shape[2] != num_classes or offsets.shape != scores.shape:
                raise class_order_mismatch(
                    "Champion output tensor shape does not match its class contract.",
                    heatmap_shape=list(scores.shape),
                    offset_shape=list(offsets.shape),
                )

            for local_index, start in enumerate(batch_indices):
                valid_len = min(int(self.spec.window_size or 128), timesteps - start)
                end = start + valid_len
                window_scores = scores[local_index, :valid_len]
                window_offsets = offsets[local_index, :valid_len]
                score_sum[start:end] += window_scores
                count_sum[start:end] += 1.0
                weights = np.clip(window_scores, a_min=1e-6, a_max=None)
                offset_sum[start:end] += window_offsets * weights
                offset_weight_sum[start:end] += weights

        return _HalfTimeline(
            half=half,
            timesteps=timesteps,
            scores=(score_sum / np.maximum(count_sum, 1.0)).astype(np.float32),
            offsets=(
                offset_sum / np.maximum(offset_weight_sum, 1e-6)
            ).astype(np.float32),
        )

    def _extract_half_candidates(
        self, prediction: _HalfTimeline
    ) -> list[_Candidate]:
        fps = float(self.spec.feature_fps or 2.0)
        radius = max(
            1, int(round(float(self.spec.local_peak_window_sec or 3.0) * fps))
        )
        duration_sec = float(prediction.timesteps) / fps
        result: list[_Candidate] = []
        for class_id, label in enumerate(CHAMPION_CLASS_ORDER):
            class_scores = prediction.scores[:, class_id]
            class_offsets = prediction.offsets[:, class_id]
            class_candidates: list[_Candidate] = []
            for raw_index in np.where(
                class_scores >= float(self.spec.threshold or 0.2)
            )[0]:
                index = int(raw_index)
                if not is_local_peak(class_scores, index, radius=radius):
                    continue
                raw_timestamp = float(index) / fps
                offset = float(class_offsets[index])
                timestamp = raw_timestamp + offset
                timestamp = max(0.0, min(timestamp, duration_sec))
                class_candidates.append(
                    _Candidate(
                        half=prediction.half,
                        label=label,
                        label_id=class_id,
                        timestamp_sec=timestamp,
                        raw_timestamp_sec=raw_timestamp,
                        offset_sec=offset,
                        timestep_idx=index,
                        score=float(class_scores[index]),
                    )
                )
            class_candidates.sort(key=lambda item: item.score, reverse=True)
            result.extend(
                class_candidates[
                    : int(self.spec.max_candidates_per_class_per_half or 100)
                ]
            )
        return result

    def _apply_class_aware_nms(
        self, candidates: list[_Candidate]
    ) -> list[_Candidate]:
        kept: list[_Candidate] = []
        nms_window = float(self.spec.nms_window_sec or 10.0)
        for label in CHAMPION_CLASS_ORDER:
            label_candidates = sorted(
                (item for item in candidates if item.label == label),
                key=lambda item: item.score,
                reverse=True,
            )
            label_kept: list[_Candidate] = []
            for candidate in label_candidates:
                if any(
                    candidate.half == other.half
                    and abs(candidate.timestamp_sec - other.timestamp_sec)
                    <= nms_window
                    for other in label_kept
                ):
                    continue
                label_kept.append(candidate)
            kept.extend(label_kept)
        return sorted(kept, key=lambda item: item.score, reverse=True)


def resolve_champion_artifact_paths(
    model_dir: str | Path,
) -> ChampionArtifactPaths:
    project_root = get_project_root()
    resolved_model_dir = Path(model_dir)
    if not resolved_model_dir.is_absolute():
        resolved_model_dir = project_root / resolved_model_dir
    config_dir = project_root / DEFAULT_CHAMPION_CONFIG_DIR
    return ChampionArtifactPaths(
        model_dir=resolved_model_dir,
        checkpoint_path=resolved_model_dir / "transformer_best.pt",
        model_config_path=config_dir / "model.yaml",
        train_config_path=config_dir / "train.yaml",
        config_snapshot_path=config_dir / "config_snapshot.yaml",
        inference_policy_path=config_dir / "inference_policy.yaml",
        manifest_path=config_dir / "manifest.yaml",
        valid_eval_path=resolved_model_dir / "valid_eval.json",
        train_history_path=resolved_model_dir / "transformer_history.json",
    )


def load_champion_model_spec(paths: ChampionArtifactPaths) -> ChampionModelSpec:
    model_config = _load_yaml(paths.model_config_path)
    snapshot = _load_yaml(paths.config_snapshot_path)
    policy = _load_yaml(paths.inference_policy_path)
    manifest = _load_yaml(paths.manifest_path)
    model_section = model_config.get("model", {})
    predictor = policy.get("predictor", {})
    postprocess = policy.get("postprocess", {})
    labels = list(policy.get("class_order") or [])
    return ChampionModelSpec(
        model_name=model_section.get("name"),
        model_type=model_section.get("type"),
        input_dim=int(model_section.get("input_dim", 0)) or None,
        num_classes=int(model_section.get("num_classes", 0)) or None,
        labels=labels,
        feature_dim=int(predictor.get("feature_dim", 0)) or None,
        feature_fps=float(predictor.get("feature_fps", 0)) or None,
        window_size=int(predictor.get("window_size", 0)) or None,
        window_sec=(
            float(predictor["window_size"]) / float(predictor["feature_fps"])
            if predictor.get("window_size") and predictor.get("feature_fps")
            else None
        ),
        stride_size=int(predictor.get("stride_size", 0)) or None,
        stride_sec=(
            float(predictor["stride_size"]) / float(predictor["feature_fps"])
            if predictor.get("stride_size") and predictor.get("feature_fps")
            else None
        ),
        threshold=_optional_float(postprocess.get("threshold")),
        local_peak_window_sec=_optional_float(
            postprocess.get("local_peak_window_sec")
        ),
        nms_window_sec=_optional_float(postprocess.get("nms_window_sec")),
        apply_offset=postprocess.get("apply_offset"),
        class_aware_nms=postprocess.get("class_aware_nms"),
        max_candidates_per_class_per_half=_optional_int(
            postprocess.get("max_candidates_per_class_per_half")
        ),
        max_candidates_per_match=_optional_int(
            postprocess.get("max_candidates_per_match")
        ),
        offset_merge=predictor.get("offset_merge"),
        champion_identifier=policy.get("champion_identifier"),
        checkpoint_sha256=policy.get("checkpoint_sha256"),
        raw_model_config=model_config,
        raw_inference_policy=policy,
        raw_config_snapshot=snapshot,
        raw_manifest=manifest,
    )


def inspect_champion_checkpoint(
    checkpoint_path: str | Path,
) -> ChampionCheckpointSummary:
    path = Path(checkpoint_path)
    if not path.is_file():
        return ChampionCheckpointSummary(loadable=False, reason="checkpoint missing")
    try:
        import torch
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
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
            epoch=_optional_int(checkpoint.get("epoch")),
            best=checkpoint.get("best"),
            metrics=checkpoint.get("metrics"),
        )
    except Exception as exc:
        return ChampionCheckpointSummary(
            loadable=False,
            reason=f"{type(exc).__name__}: {exc}",
        )


def get_window_start_indices(
    num_timesteps: int, window_size: int, stride_size: int
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
    feature_array: np.ndarray, start: int, window_size: int
) -> np.ndarray:
    feature_dim = int(feature_array.shape[1])
    window = np.zeros((window_size, feature_dim), dtype=np.float32)
    end = min(int(feature_array.shape[0]), start + window_size)
    if start < end:
        window[: end - start] = np.asarray(
            feature_array[start:end], dtype=np.float32
        )
    return window


def build_sliding_windows(
    *,
    feature_array: np.ndarray,
    window_size: int,
    stride_size: int,
) -> tuple[np.ndarray, list[int], int]:
    if feature_array.ndim != 2:
        raise ValueError(
            f"feature array must be 2-D [T, D], got {feature_array.shape}"
        )
    starts = get_window_start_indices(
        int(feature_array.shape[0]), window_size, stride_size
    )
    windows = np.stack(
        [read_feature_window(feature_array, start, window_size) for start in starts]
    )
    return windows, starts, int(feature_array.shape[0])


def is_local_peak(scores: np.ndarray, index: int, *, radius: int) -> bool:
    if radius <= 0:
        return True
    start = max(0, index - radius)
    end = min(len(scores), index + radius + 1)
    current = float(scores[index])
    neighborhood = scores[start:end]
    maximum = float(np.max(neighborhood))
    if current < maximum:
        return False
    maximum_indices = np.where(neighborhood == maximum)[0]
    return bool(len(maximum_indices)) and index == start + int(maximum_indices[0])


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
            "Champion runtime supports only cpu, cuda, or auto devices.",
            requested_device=name,
        )
    return torch_module.device(name)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def build_implementation_todos(
    spec: ChampionModelSpec,
    checkpoint: ChampionCheckpointSummary,
    *,
    round25_done: bool = True,
) -> list[str]:
    del spec, checkpoint, round25_done
    return []
