from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = (
        Path(__file__).resolve().parents[1]
        / "app/domains/candidate_handoff_r1/runtime/verify_r1_v1_v2_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("verify_r13_1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase4c_finalizer_replaces_legacy_unsupported_gate() -> None:
    module = _load()
    adapter = "\n".join(
        [
            "--tracking-launch-manifest",
            "--target-selection",
            "--target-reference-set",
            "--earlier-anchor-decision",
            "--target-memory-revision",
            "--candidate-scoring-generation",
            "SELECTION_VIEW_SCHEMA",
            "--confirmed-candidate",
            "post_confirmation_finalizer",
            "finalize_confirmed_candidate",
            "if args.resume and args.confirmed_candidate",
            "args.target_memory_revision is not None",
            "memory_sha256=str(memory_sha)",
        ]
    )
    finalizer = "\n".join(
        [
            'memory.get("source_candidate_id") != candidate_id',
            'decision.get("state") != "SAME_PLAYER"',
            "identity_purity_gate_passed",
            "identity_observability_gate_passed",
            '"interpolation_used": False',
            '"synthetic_tracking_used": False',
            '"automatic_target_confirmation": False',
            "_sha256_file(memory_path)",
        ]
    )
    result = module.evaluate_adapter_contract(
        adapter_text=adapter,
        finalizer_text=finalizer,
    )
    assert result["legacy_memory_update_block_verified"] is False
    assert result["phase4c_post_confirmation_finalizer_verified"] is True
    assert result["memory_revision_safety_gate_verified"] is True
    assert result["backend_r1_v1_v2_adapter_verified"] is True


def test_missing_finalizer_contract_fails_closed() -> None:
    module = _load()
    adapter = "\n".join(
        [
            "--tracking-launch-manifest",
            "--target-selection",
            "--target-reference-set",
            "--earlier-anchor-decision",
            "--target-memory-revision",
            "--candidate-scoring-generation",
            "SELECTION_VIEW_SCHEMA",
            "--confirmed-candidate",
            "post_confirmation_finalizer",
            "finalize_confirmed_candidate",
            "if args.resume and args.confirmed_candidate",
            "args.target_memory_revision is not None",
            "memory_sha256=str(memory_sha)",
        ]
    )
    result = module.evaluate_adapter_contract(
        adapter_text=adapter,
        finalizer_text="",
    )
    assert result["memory_revision_safety_gate_verified"] is False
    assert result["backend_r1_v1_v2_adapter_verified"] is False
