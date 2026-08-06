from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    PROJECT_ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_r8_test_path_is_replaced_by_r9_policy_contract() -> None:
    adapter = _load("kickclip_r9_compatibility_test", ADAPTER_PATH)
    assert adapter.PHASE4B_POLICY.endswith("R9_PROVENANCE_ROLE_TRAJECTORY_REVIEW")
    assert adapter.PHASE4B_DETECTOR_ROLE_SCORING_POLICY.endswith("SOFT_EVIDENCE_R1")
    assert adapter.PHASE4B_ASSISTED_REVIEW_SELECTION_POLICY.endswith("DIVERSITY_R1")
    assert adapter.PHASE4B_PREVIOUS_POLICY_R8 in adapter.PHASE4B_RESCORABLE_PREVIOUS_POLICIES
