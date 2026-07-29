from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from app.domains.highlight.player_detector import (
    INSTALL_MESSAGE,
    PlayerDetectorCheckpointError,
    PlayerDetectorDeviceError,
    RFDETRPlayerDetector,
    resolve_device,
    strict_verify_loaded_weights,
    validate_checkpoint_contract,
)


class _FakeCuda:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self) -> bool:
        return self.available

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Fake CUDA GPU"


class _FakeMPS:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self) -> bool:
        return self.available


def _fake_torch(*, cuda: bool, mps: bool):
    return SimpleNamespace(
        __version__="test-torch",
        cuda=_FakeCuda(cuda),
        backends=SimpleNamespace(mps=_FakeMPS(mps)),
    )


class PlayerDetectorDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_and_records_selected_device(self) -> None:
        result = resolve_device(
            "auto",
            _fake_torch(cuda=True, mps=False),
        )

        self.assertEqual(result.effective_device, "cuda")
        self.assertEqual(
            result.device_reason,
            "torch.cuda.is_available=true",
        )
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.gpu_name, "Fake CUDA GPU")

    def test_auto_uses_mps_then_cpu_with_explicit_fallback_record(self) -> None:
        mps = resolve_device(
            "auto",
            _fake_torch(cuda=False, mps=True),
        )
        cpu = resolve_device(
            "auto",
            _fake_torch(cuda=False, mps=False),
        )

        self.assertEqual(mps.effective_device, "mps")
        self.assertFalse(mps.fallback_used)
        self.assertEqual(cpu.effective_device, "cpu")
        self.assertTrue(cpu.fallback_used)
        self.assertIn("CPU fallback allowed", cpu.device_reason)

    def test_explicit_unavailable_accelerator_never_falls_back(self) -> None:
        for requested in ("cuda", "mps"):
            with self.subTest(requested=requested):
                with self.assertRaises(PlayerDetectorDeviceError) as raised:
                    resolve_device(
                        requested,
                        _fake_torch(cuda=False, mps=False),
                    )
                self.assertIsNone(
                    raised.exception.diagnostics["effective_device"]
                )
                self.assertFalse(
                    raised.exception.diagnostics["fallback_used"]
                )
                self.assertIn(
                    "CPU fallback is forbidden",
                    str(raised.exception),
                )


class PlayerDetectorContractTests(unittest.TestCase):
    def test_checkpoint_requires_exact_rfdetrsmall_contract(self) -> None:
        checkpoint = {
            "model": {},
            "model_name": "RFDETRBase",
            "rfdetr_version": "1.8.3",
            "model_config": {
                "model_name": "RFDETRBase",
                "num_classes": 5,
            },
            "args": {
                "class_names": [
                    "player",
                    "goalkeeper",
                    "referee",
                    "staff",
                    "ball",
                ]
            },
        }

        with self.assertRaises(PlayerDetectorCheckpointError):
            validate_checkpoint_contract(
                checkpoint,
                runtime_version="1.8.3",
            )

    def test_strict_weight_audit_rejects_changed_tensor(self) -> None:
        with self.assertRaises(PlayerDetectorCheckpointError):
            strict_verify_loaded_weights(
                torch_module=torch,
                checkpoint_state={"weight": torch.tensor([1.0])},
                loaded_state={"weight": torch.tensor([2.0])},
            )

    def test_native_postprocess_filters_player_classes_and_clamps_xyxy(
        self,
    ) -> None:
        detector = object.__new__(RFDETRPlayerDetector)
        detector.confidence_threshold = 0.25
        frame = SimpleNamespace(shape=(100, 200, 3))
        result = SimpleNamespace(
            xyxy=[
                [-10, 5, 220, 95],
                [10, 10, 50, 80],
                [10, 10, 50, 80],
                [0, 0, 10, 10],
            ],
            class_id=[0, 1, 1, 2],
            confidence=[0.9, 0.8, 0.7, 0.99],
        )

        rows = detector._normalize_result(result, frame)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].bbox_xyxy, [0.0, 5.0, 200.0, 95.0])
        self.assertEqual({row.class_id for row in rows}, {0, 1})
        # No extra NMS is applied on top of RF-DETR native post-processing.
        self.assertEqual(
            sum(row.bbox_xyxy == [10.0, 10.0, 50.0, 80.0] for row in rows),
            2,
        )

    def test_missing_runtime_message_has_install_command(self) -> None:
        self.assertIn("requirements.txt", INSTALL_MESSAGE)
        self.assertIn("rfdetr==1.8.3", INSTALL_MESSAGE)


if __name__ == "__main__":
    unittest.main()
