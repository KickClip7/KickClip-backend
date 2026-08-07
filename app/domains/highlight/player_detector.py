from __future__ import annotations

import hashlib
import importlib.metadata
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.ai.runtime.gpu_coordinator import claim_gpu_slot


COMPONENT_RUNTIME_VERSION = "kickclip-player-detector-rfdetr/1.1.0"
EXPECTED_RFDETR_VERSION = "1.8.3"
EXPECTED_MODEL_CLASS = "RFDETRSmall"
EXPECTED_CLASS_NAMES = [
    "player",
    "goalkeeper",
    "referee",
    "staff",
    "ball",
]
PLAYER_CLASS_IDS = frozenset({0, 1})
ALL_RFDETR_CLASS_IDS = frozenset(range(len(EXPECTED_CLASS_NAMES)))
INSTALL_MESSAGE = (
    "RF-DETR player detector is not installed. Create or activate a Python "
    "environment compatible with the fine-tuned model and run "
    "`python -m pip install -r requirements.txt`. "
    f"The required runtime is rfdetr=={EXPECTED_RFDETR_VERSION}."
)

_ARCHITECTURE_FIELDS = (
    "encoder",
    "out_feature_indexes",
    "dec_layers",
    "two_stage",
    "projector_scale",
    "hidden_dim",
    "patch_size",
    "num_windows",
    "sa_nheads",
    "ca_nheads",
    "dec_n_points",
    "num_queries",
    "num_select",
    "bbox_reparam",
    "lite_refpoint_refine",
    "layer_norm",
    "num_channels",
    "num_classes",
    "resolution",
    "group_detr",
    "positional_encoding_size",
    "segmentation_head",
    "use_grouppose_keypoints",
)


class PlayerDetectorError(RuntimeError):
    code = "PLAYER_DETECTOR_ERROR"

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class PlayerDetectorUnavailableError(PlayerDetectorError):
    code = "PLAYER_DETECTOR_UNAVAILABLE"


class PlayerDetectorDeviceError(PlayerDetectorError):
    code = "PLAYER_DETECTOR_DEVICE_UNAVAILABLE"


class PlayerDetectorCheckpointError(PlayerDetectorError):
    code = "PLAYER_DETECTOR_CHECKPOINT_INCOMPATIBLE"


class PlayerDetectorInferenceError(PlayerDetectorError):
    code = "PLAYER_DETECTOR_INFERENCE_FAILED"


@dataclass(frozen=True)
class DeviceDiagnostics:
    requested_device: str
    effective_device: str | None
    device_reason: str
    fallback_used: bool
    torch_version: str | None
    cuda_available: bool
    mps_available: bool
    gpu_name: str | None
    component_runtime_version: str = COMPONENT_RUNTIME_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerDetection:
    bbox_xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str


class PlayerDetector(Protocol):
    @property
    def runtime_metadata(self) -> dict[str, Any]: ...

    def detect_batch(
        self,
        frames_bgr: Sequence[Any],
    ) -> list[list[PlayerDetection]]: ...


def _torch_capabilities(torch_module: Any) -> tuple[bool, bool, str | None]:
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        cuda_available = False

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    try:
        mps_available = bool(mps_backend and mps_backend.is_available())
    except Exception:
        mps_available = False

    gpu_name: str | None = None
    if cuda_available:
        try:
            gpu_name = str(torch_module.cuda.get_device_name(0))
        except Exception:
            gpu_name = "CUDA device 0"
    elif mps_available:
        gpu_name = "Apple GPU (Metal Performance Shaders)"

    return cuda_available, mps_available, gpu_name


def resolve_device(requested_device: str, torch_module: Any) -> DeviceDiagnostics:
    requested = requested_device.strip().lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            "PLAYER_DETECTOR_DEVICE must be one of auto, cuda, mps, cpu."
        )

    cuda_available, mps_available, gpu_name = _torch_capabilities(torch_module)
    torch_version = str(getattr(torch_module, "__version__", "unknown"))

    if requested == "auto":
        if cuda_available:
            return DeviceDiagnostics(
                requested_device=requested,
                effective_device="cuda",
                device_reason="torch.cuda.is_available=true",
                fallback_used=False,
                torch_version=torch_version,
                cuda_available=True,
                mps_available=mps_available,
                gpu_name=gpu_name,
            )
        if mps_available:
            return DeviceDiagnostics(
                requested_device=requested,
                effective_device="mps",
                device_reason=(
                    "CUDA unavailable; torch.backends.mps.is_available=true"
                ),
                fallback_used=False,
                torch_version=torch_version,
                cuda_available=False,
                mps_available=True,
                gpu_name=gpu_name,
            )
        return DeviceDiagnostics(
            requested_device=requested,
            effective_device="cpu",
            device_reason=(
                "CUDA unavailable; MPS unavailable; "
                "CPU fallback allowed for device=auto"
            ),
            fallback_used=True,
            torch_version=torch_version,
            cuda_available=False,
            mps_available=False,
            gpu_name=None,
        )

    if requested == "cuda" and not cuda_available:
        diagnostics = DeviceDiagnostics(
            requested_device=requested,
            effective_device=None,
            device_reason=(
                "CUDA was explicitly requested but "
                "torch.cuda.is_available=false; CPU fallback is forbidden"
            ),
            fallback_used=False,
            torch_version=torch_version,
            cuda_available=False,
            mps_available=mps_available,
            gpu_name=gpu_name,
        )
        raise PlayerDetectorDeviceError(
            diagnostics.device_reason,
            diagnostics=diagnostics.as_dict(),
        )

    if requested == "mps" and not mps_available:
        diagnostics = DeviceDiagnostics(
            requested_device=requested,
            effective_device=None,
            device_reason=(
                "MPS was explicitly requested but "
                "torch.backends.mps.is_available=false; CPU fallback is forbidden"
            ),
            fallback_used=False,
            torch_version=torch_version,
            cuda_available=cuda_available,
            mps_available=False,
            gpu_name=gpu_name,
        )
        raise PlayerDetectorDeviceError(
            diagnostics.device_reason,
            diagnostics=diagnostics.as_dict(),
        )

    return DeviceDiagnostics(
        requested_device=requested,
        effective_device=requested,
        device_reason=f"{requested.upper()} explicitly requested and available",
        fallback_used=False,
        torch_version=torch_version,
        cuda_available=cuda_available,
        mps_available=mps_available,
        gpu_name=gpu_name,
    )


def _checkpoint_arg(checkpoint: dict[str, Any], key: str) -> Any:
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return getattr(args, key, None)


def validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    runtime_version: str,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"),
        dict,
    ):
        raise PlayerDetectorCheckpointError(
            "RF-DETR checkpoint must contain a top-level model state dictionary."
        )

    checkpoint_version = str(checkpoint.get("rfdetr_version") or "")
    if checkpoint_version != EXPECTED_RFDETR_VERSION:
        raise PlayerDetectorCheckpointError(
            "RF-DETR checkpoint runtime mismatch: "
            f"expected {EXPECTED_RFDETR_VERSION}, found "
            f"{checkpoint_version or 'missing'}."
        )
    if runtime_version != EXPECTED_RFDETR_VERSION:
        raise PlayerDetectorCheckpointError(
            "Installed RF-DETR runtime is incompatible: "
            f"expected {EXPECTED_RFDETR_VERSION}, found {runtime_version}."
        )

    model_name = checkpoint.get("model_name")
    model_config = checkpoint.get("model_config")
    if model_name != EXPECTED_MODEL_CLASS or not isinstance(model_config, dict):
        raise PlayerDetectorCheckpointError(
            "Checkpoint is not the expected RFDETRSmall architecture."
        )
    if model_config.get("model_name") != EXPECTED_MODEL_CLASS:
        raise PlayerDetectorCheckpointError(
            "Checkpoint model_config.model_name is not RFDETRSmall."
        )

    class_names = _checkpoint_arg(checkpoint, "class_names")
    if list(class_names or []) != EXPECTED_CLASS_NAMES:
        raise PlayerDetectorCheckpointError(
            "Checkpoint class mapping does not match "
            "player/goalkeeper/referee/staff/ball."
        )
    if model_config.get("num_classes") != len(EXPECTED_CLASS_NAMES):
        raise PlayerDetectorCheckpointError(
            "Checkpoint num_classes does not match the five-class soccer model."
        )

    return {
        "model_class": EXPECTED_MODEL_CLASS,
        "rfdetr_version": checkpoint_version,
        "class_names": list(class_names),
        "architecture": {
            key: model_config.get(key)
            for key in _ARCHITECTURE_FIELDS
        },
    }


def strict_verify_loaded_weights(
    *,
    torch_module: Any,
    checkpoint_state: dict[str, Any],
    loaded_state: dict[str, Any],
) -> None:
    missing = sorted(set(loaded_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(loaded_state))
    if missing or unexpected:
        raise PlayerDetectorCheckpointError(
            "RF-DETR strict weight audit failed: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )

    shape_mismatches = [
        (
            key,
            tuple(checkpoint_state[key].shape),
            tuple(loaded_state[key].shape),
        )
        for key in sorted(checkpoint_state)
        if getattr(checkpoint_state[key], "shape", None)
        != getattr(loaded_state[key], "shape", None)
    ]
    if shape_mismatches:
        raise PlayerDetectorCheckpointError(
            "RF-DETR strict weight audit found shape mismatches: "
            f"{shape_mismatches[:10]}."
        )

    unequal = [
        key
        for key in sorted(checkpoint_state)
        if not torch_module.equal(
            checkpoint_state[key].detach().cpu(),
            loaded_state[key].detach().cpu(),
        )
    ]
    if unequal:
        raise PlayerDetectorCheckpointError(
            "RF-DETR strict weight audit found tensors that were not loaded "
            f"exactly: {unequal[:10]}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _LoadedRFDETR:
    model: Any
    checkpoint_sha256: str
    checkpoint_contract: dict[str, Any]


_MODEL_CACHE: dict[tuple[str, int, int, str], _LoadedRFDETR] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


class RFDETRPlayerDetector:
    """Exact adapter for the fine-tuned RFDETRSmall soccer checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        requested_device: str,
        confidence_threshold: float = 0.25,
        batch_size: int = 6,
        inference_threshold: float | None = None,
        output_class_ids: frozenset[int] | set[int] | None = None,
        profile: str = "play",
    ):
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.confidence_threshold = float(confidence_threshold)
        self.inference_threshold = float(
            inference_threshold
            if inference_threshold is not None
            else confidence_threshold
        )
        self.output_class_ids = frozenset(
            PLAYER_CLASS_IDS if output_class_ids is None else output_class_ids
        )
        self.profile = str(profile or "play").strip().lower()
        self.batch_size = batch_size

        try:
            import torch
        except ImportError as exc:
            raise PlayerDetectorUnavailableError(INSTALL_MESSAGE) from exc
        self._torch = torch
        self.device = resolve_device(requested_device, torch)

        if not self.checkpoint_path.is_file():
            raise PlayerDetectorUnavailableError(
                "RF-DETR checkpoint was not found at "
                f"{self.checkpoint_path}. Configure PLAYER_DETECTOR_CHECKPOINT "
                "or install the fine-tuned model.",
                diagnostics=self.device.as_dict(),
            )
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("RF-DETR confidence threshold must be in (0, 1].")
        if not 0 < self.inference_threshold <= 1:
            raise ValueError("RF-DETR inference threshold must be in (0, 1].")
        if self.inference_threshold > self.confidence_threshold:
            raise ValueError(
                "RF-DETR inference threshold cannot be greater than the "
                "output confidence threshold."
            )
        if not self.output_class_ids:
            raise ValueError("RF-DETR output_class_ids cannot be empty.")
        unsupported = sorted(self.output_class_ids - ALL_RFDETR_CLASS_IDS)
        if unsupported:
            raise ValueError(
                "RF-DETR output_class_ids contains unsupported ids: "
                f"{unsupported}."
            )
        if batch_size < 1:
            raise ValueError("RF-DETR batch size must be positive.")

        try:
            runtime_version = importlib.metadata.version("rfdetr")
        except importlib.metadata.PackageNotFoundError as exc:
            raise PlayerDetectorUnavailableError(
                INSTALL_MESSAGE,
                diagnostics=self.device.as_dict(),
            ) from exc
        if runtime_version != EXPECTED_RFDETR_VERSION:
            raise PlayerDetectorCheckpointError(
                "Installed RF-DETR runtime is incompatible with this "
                f"checkpoint: expected {EXPECTED_RFDETR_VERSION}, found "
                f"{runtime_version}.",
                diagnostics=self.device.as_dict(),
            )
        self.rfdetr_version = runtime_version
        self._loaded = self._load_strict()

    def _load_strict(self) -> _LoadedRFDETR:
        stat = self.checkpoint_path.stat()
        cache_key = (
            str(self.checkpoint_path),
            stat.st_mtime_ns,
            stat.st_size,
            str(self.device.effective_device),
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached

            try:
                from rfdetr import RFDETRSmall
            except ImportError as exc:
                raise PlayerDetectorUnavailableError(
                    INSTALL_MESSAGE,
                    diagnostics=self.device.as_dict(),
                ) from exc

            checkpoint = self._torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            contract = validate_checkpoint_contract(
                checkpoint,
                runtime_version=self.rfdetr_version,
            )
            model = RFDETRSmall(
                pretrain_weights=str(self.checkpoint_path),
                device=self.device.effective_device,
                num_classes=len(EXPECTED_CLASS_NAMES),
            )

            loaded_config = model.model_config.model_dump(mode="python")
            config_mismatches = {
                key: {
                    "checkpoint": contract["architecture"].get(key),
                    "loaded": loaded_config.get(key),
                }
                for key in _ARCHITECTURE_FIELDS
                if contract["architecture"].get(key) != loaded_config.get(key)
            }
            if config_mismatches:
                raise PlayerDetectorCheckpointError(
                    "Loaded RFDETRSmall architecture does not exactly match "
                    f"the checkpoint: {config_mismatches}.",
                    diagnostics=self.device.as_dict(),
                )

            loaded_state = model.model.model.state_dict()
            try:
                strict_verify_loaded_weights(
                    torch_module=self._torch,
                    checkpoint_state=checkpoint["model"],
                    loaded_state=loaded_state,
                )
            except PlayerDetectorCheckpointError as exc:
                exc.diagnostics = self.device.as_dict()
                raise

            loaded = _LoadedRFDETR(
                model=model,
                checkpoint_sha256=_sha256(self.checkpoint_path),
                checkpoint_contract=contract,
            )
            _MODEL_CACHE[cache_key] = loaded
            return loaded

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backend": "rfdetr",
            **self.device.as_dict(),
            "rfdetr_version": self.rfdetr_version,
            "model_class": EXPECTED_MODEL_CLASS,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self._loaded.checkpoint_sha256,
            "confidence_threshold": self.confidence_threshold,
            "inference_threshold": self.inference_threshold,
            "output_confidence_threshold": self.confidence_threshold,
            "detector_profile": self.profile,
            "class_mapping": {
                str(index): name
                for index, name in enumerate(EXPECTED_CLASS_NAMES)
            },
            "candidate_class_ids": sorted(self.output_class_ids),
            "output_class_ids": sorted(self.output_class_ids),
            "input_color": "RGB",
            "preprocessing": (
                "RF-DETR 1.8.3 native predict preprocessing "
                "(checkpoint resolution=512)"
            ),
            "bbox_semantics": "original-frame xyxy",
            "bbox_bounds_policy": "inclusive_width_minus_1_height_minus_1_v1",
            "additional_nms": False,
            "batch_size": self.batch_size,
            "strict_checkpoint_audit": True,
        }

    def detect_batch(
        self,
        frames_bgr: Sequence[Any],
    ) -> list[list[PlayerDetection]]:
        if not frames_bgr:
            return []

        try:
            import cv2
        except ImportError as exc:
            raise PlayerDetectorUnavailableError(
                "opencv-python is required for RF-DETR video frame decoding.",
                diagnostics=self.runtime_metadata,
            ) from exc

        output: list[list[PlayerDetection]] = []
        try:
            with _INFERENCE_LOCK:
                slot = (
                    claim_gpu_slot()
                    if self.device.effective_device in {"cuda", "mps"}
                    else _null_context()
                )
                with slot:
                    for start in range(0, len(frames_bgr), self.batch_size):
                        chunk = frames_bgr[start : start + self.batch_size]
                        rgb_images = [
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            for frame in chunk
                        ]
                        with self._torch.inference_mode():
                            raw = self._loaded.model.predict(
                                rgb_images,
                                threshold=self.inference_threshold,
                            )
                        raw_rows = list(raw) if isinstance(raw, list) else [raw]
                        if len(raw_rows) != len(chunk):
                            raise RuntimeError(
                                "RF-DETR batch output count does not match "
                                "the input frame count."
                            )
                        output.extend(
                            self._normalize_result(result, frame)
                            for result, frame in zip(raw_rows, chunk)
                        )
        except PlayerDetectorError:
            raise
        except Exception as exc:
            raise PlayerDetectorInferenceError(
                "RF-DETR player candidate inference failed: "
                f"{type(exc).__name__}: {exc}",
                diagnostics=self.runtime_metadata,
            ) from exc
        return output

    def _normalize_result(
        self,
        result: Any,
        frame: Any,
    ) -> list[PlayerDetection]:
        height, width = frame.shape[:2]
        boxes = getattr(result, "xyxy", [])
        class_ids = getattr(result, "class_id", [])
        confidences = getattr(result, "confidence", [])
        detections: list[PlayerDetection] = []

        for raw_box, raw_class_id, raw_confidence in zip(
            boxes,
            class_ids,
            confidences,
        ):
            class_id = int(raw_class_id)
            confidence = float(raw_confidence)
            if (
                class_id not in self.output_class_ids
                or confidence < self.confidence_threshold
            ):
                continue
            values = [float(value) for value in raw_box]
            if len(values) != 4:
                continue
            # The frozen Stage-1/Stage-2 contract uses inclusive image bounds:
            # x/y coordinates must stay within [0, width-1] / [0, height-1].
            # RF-DETR can return sub-pixel values such as x2=1919.74 for a
            # 1920px frame, so clamp here before any cache/artifact is written.
            max_x = float(max(0, width - 1))
            max_y = float(max(0, height - 1))
            x1 = min(max(values[0], 0.0), max_x)
            y1 = min(max(values[1], 0.0), max_y)
            x2 = min(max(values[2], 0.0), max_x)
            y2 = min(max(values[3], 0.0), max_y)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                PlayerDetection(
                    bbox_xyxy=[x1, y1, x2, y2],
                    confidence=confidence,
                    class_id=class_id,
                    class_name=EXPECTED_CLASS_NAMES[class_id],
                )
            )

        # RF-DETR predict() already performs its native post-processing.
        # Additional NMS would change the frozen inference contract.
        return sorted(
            detections,
            key=lambda detection: detection.confidence,
            reverse=True,
        )


class HOGPlayerDetector:
    """Explicit development fallback; never selected unless configured."""

    def __init__(
        self,
        *,
        requested_device: str,
        fallback_reason: str,
        fallback_used: bool,
        source_diagnostics: dict[str, Any] | None = None,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise PlayerDetectorUnavailableError(
                "opencv-python is required for the explicit HOG fallback."
            ) from exc
        self._cv2 = cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        source = source_diagnostics or {}
        self._metadata = {
            "backend": "opencv_hog",
            "requested_device": requested_device,
            "effective_device": "cpu",
            "device_reason": fallback_reason,
            "fallback_used": fallback_used,
            "torch_version": source.get("torch_version"),
            "cuda_available": bool(source.get("cuda_available", False)),
            "mps_available": bool(source.get("mps_available", False)),
            "gpu_name": source.get("gpu_name"),
            "component_runtime_version": (
                f"opencv-hog/{getattr(cv2, '__version__', 'unknown')}"
            ),
            "confidence_threshold": None,
            "candidate_class_ids": [0],
            "input_color": "BGR",
            "bbox_semantics": "original-frame xyxy",
            "bbox_bounds_policy": "inclusive_width_minus_1_height_minus_1_v1",
            "additional_nms": False,
            "batch_size": 1,
            "strict_checkpoint_audit": False,
        }

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def detect_batch(
        self,
        frames_bgr: Sequence[Any],
    ) -> list[list[PlayerDetection]]:
        results: list[list[PlayerDetection]] = []
        for frame in frames_bgr:
            boxes, weights = self._hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            frame_rows: list[PlayerDetection] = []
            frame_height, frame_width = frame.shape[:2]
            max_x = float(max(0, frame_width - 1))
            max_y = float(max(0, frame_height - 1))
            for (x, y, width, height), weight in zip(boxes, weights):
                x1 = min(max(float(x), 0.0), max_x)
                y1 = min(max(float(y), 0.0), max_y)
                x2 = min(max(float(x + width), 0.0), max_x)
                y2 = min(max(float(y + height), 0.0), max_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                frame_rows.append(
                    PlayerDetection(
                        bbox_xyxy=[x1, y1, x2, y2],
                        confidence=min(1.0, max(0.0, float(weight) / 2.0)),
                        class_id=0,
                        class_name="player",
                    )
                )
            results.append(frame_rows)
        return results


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def create_player_detector(
    settings: Any,
    *,
    profile: str = "play",
) -> PlayerDetector:
    """Create the fine-tuned RF-DETR adapter with an explicit inference profile.

    `play` keeps only player/goalkeeper detections at the wide-shot threshold.
    `observation` keeps all configured observation classes from the lower base
    threshold; callers then apply close-up/role policy without losing staff or
    referee evidence.
    """

    backend = str(settings.PLAYER_DETECTOR_BACKEND).lower()
    requested_device = str(settings.PLAYER_DETECTOR_DEVICE).lower()
    normalized_profile = str(profile or "play").strip().lower()
    if normalized_profile not in {"play", "observation"}:
        raise ValueError(f"Unsupported RF-DETR detector profile: {profile}.")

    if backend == "hog":
        return HOGPlayerDetector(
            requested_device=requested_device,
            fallback_reason="HOG backend was explicitly configured",
            fallback_used=False,
        )
    if backend != "rfdetr":
        raise ValueError(f"Unsupported player detector backend: {backend}.")

    base_threshold = float(
        getattr(settings, "RFDETR_BASE_CONF_THRESHOLD", 0.15)
    )
    if normalized_profile == "observation":
        output_threshold = base_threshold
        output_class_ids = frozenset(
            getattr(settings, "observation_class_ids", ALL_RFDETR_CLASS_IDS)
        )
    else:
        output_threshold = float(
            getattr(
                settings,
                "TRACKING_PLAY_CONF_THRESHOLD",
                settings.PLAYER_DETECTOR_CONFIDENCE_THRESHOLD,
            )
        )
        output_class_ids = frozenset(
            getattr(settings, "tracking_play_class_ids", PLAYER_CLASS_IDS)
        )

    inference_threshold = min(base_threshold, output_threshold)

    try:
        return RFDETRPlayerDetector(
            checkpoint_path=Path(settings.PLAYER_DETECTOR_CHECKPOINT),
            requested_device=requested_device,
            confidence_threshold=output_threshold,
            inference_threshold=inference_threshold,
            output_class_ids=output_class_ids,
            profile=normalized_profile,
            batch_size=int(settings.PLAYER_DETECTOR_BATCH_SIZE),
        )
    except PlayerDetectorUnavailableError as exc:
        if not settings.PLAYER_DETECTOR_ALLOW_HOG_FALLBACK:
            raise
        return HOGPlayerDetector(
            requested_device=requested_device,
            fallback_reason=(
                "RF-DETR unavailable; explicit "
                f"PLAYER_DETECTOR_ALLOW_HOG_FALLBACK=true: {exc}"
            ),
            fallback_used=True,
            source_diagnostics=exc.diagnostics,
        )


def create_observation_detector(settings: Any) -> PlayerDetector:
    """Create RF-DETR in all-class observation mode for candidate discovery."""

    return create_player_detector(settings, profile="observation")


def discovery_failure_payload(
    exc: Exception,
    *,
    settings: Any,
) -> dict[str, Any]:
    diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
    if not diagnostics:
        try:
            import torch

            diagnostics.update(
                resolve_device(
                    str(settings.PLAYER_DETECTOR_DEVICE),
                    torch,
                ).as_dict()
            )
        except PlayerDetectorDeviceError as device_exc:
            diagnostics.update(device_exc.diagnostics)
        except Exception:
            # The required keys are still added below. This branch is used only
            # when even the optional torch runtime cannot be inspected.
            pass
    diagnostics.setdefault(
        "requested_device",
        str(settings.PLAYER_DETECTOR_DEVICE).lower(),
    )
    diagnostics.setdefault("effective_device", None)
    diagnostics.setdefault("device_reason", str(exc))
    diagnostics.setdefault("fallback_used", False)
    diagnostics.setdefault("torch_version", None)
    diagnostics.setdefault("cuda_available", False)
    diagnostics.setdefault("mps_available", False)
    diagnostics.setdefault("gpu_name", None)
    diagnostics.setdefault(
        "component_runtime_version",
        COMPONENT_RUNTIME_VERSION,
    )
    return {
        "status": "FAILED",
        "backend": str(settings.PLAYER_DETECTOR_BACKEND).lower(),
        "code": getattr(exc, "code", "PLAYER_CANDIDATE_DISCOVERY_FAILED"),
        "reason": str(exc),
        "runtime": diagnostics,
    }
