from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    PROJECT_ROOT
    / "app"
    / "domains"
    / "candidate_handoff_r1"
    / "runtime"
    / "r1_v1_v2_adapter_cli.py"
)
VERIFIER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "verify_phase4b_target_context_corroboration_contract.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("kickclip_phase4b_r11_adapter_test", ADAPTER_PATH)
VERIFIER = _load("kickclip_phase4b_r11_verifier_test", VERIFIER_PATH)


def _unit(*values: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32)
    return row / np.linalg.norm(row)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(bgr: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((160, 80, 3), dtype=np.uint8)
    image[:] = bgr
    return image


def _candidate(
    candidate_id: str,
    parent_id: str,
    start: int,
    end: int,
    *,
    retrieval: float,
    prototype_similarity: float,
    context_similarity: float,
    prototype: np.ndarray,
) -> dict[str, object]:
    count = end - start + 1
    return {
        "candidate_id": candidate_id,
        "parent_tracklet_id": parent_id,
        "start_frame": start,
        "end_frame_inclusive": end,
        "tracklet_detection_count": count,
        "detection_count": count,
        "retrieval_score": retrieval,
        "prototype_target_similarity": prototype_similarity,
        "identity_observability": {
            "passed": True,
            "clean_frame_count": count,
            "clean_frame_ratio": 1.0,
        },
        "identity_purity": {
            "policy": ADAPTER.PHASE4B_IDENTITY_PURITY_POLICY,
            "passed": True,
            "parent_tracklet_id": parent_id,
            "internal_pairwise_cosine_median": 0.90,
            "target_memory_used_for_segmentation": False,
            "automatic_target_confirmation": False,
        },
        "negative_review_gate": {
            "passed": True,
            "user_confirmed_role_gate": {"passed": True},
            "identity_negative_gate": {"passed": True},
        },
        "candidate_scale_class_counts": {"wide": count},
        "target_context_policy": ADAPTER.PHASE4B_TARGET_CONTEXT_POLICY,
        "target_context_similarity": context_similarity,
        "target_context_is_soft_evidence_only": True,
        "_prototype_vector": prototype,
    }


def _assignments(
    candidate_id: str,
    start: int,
    end: int,
    *,
    x: float,
) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate_id,
            "analysis_local_frame_index": frame,
            "x1": x,
            "y1": 100.0,
            "x2": x + 50.0,
            "y2": 200.0,
        }
        for frame in range(start, end + 1)
    ]


def test_target_context_descriptor_prefers_matching_approved_reference_color() -> None:
    target = ADAPTER._phase4b_aggregate_target_context_descriptor(
        [_image((235, 235, 235)), _image((245, 245, 245))],
        cv2=cv2,
        np=np,
    )
    matching = ADAPTER._phase4b_aggregate_target_context_descriptor(
        [_image((240, 240, 240))], cv2=cv2, np=np
    )
    dark = ADAPTER._phase4b_aggregate_target_context_descriptor(
        [_image((35, 35, 35))], cv2=cv2, np=np
    )
    matching_similarity = ADAPTER._phase4b_target_context_similarity(
        matching, target, np=np
    )
    dark_similarity = ADAPTER._phase4b_target_context_similarity(
        dark, target, np=np
    )
    assert matching_similarity > 0.95
    assert matching_similarity > dark_similarity + 0.25


def test_independent_overlapping_tracklets_create_corroboration_group() -> None:
    rows = [
        _candidate(
            "candidate_a",
            "parent_a",
            100,
            120,
            retrieval=0.52,
            prototype_similarity=0.57,
            context_similarity=0.91,
            prototype=_unit(1.0, 0.0, 0.0),
        ),
        _candidate(
            "candidate_b",
            "parent_b",
            105,
            115,
            retrieval=0.50,
            prototype_similarity=0.55,
            context_similarity=0.89,
            prototype=_unit(0.99, 0.05, 0.0),
        ),
    ]
    assignments = {
        "candidate_a": _assignments("candidate_a", 100, 120, x=300.0),
        "candidate_b": _assignments("candidate_b", 105, 115, x=306.0),
    }
    groups = ADAPTER._phase4b_assign_corroboration_groups(
        rows, assignments_by_candidate=assignments, np=np
    )
    group = next(row for row in groups if row["corroborated"] is True)
    assert set(group["member_candidate_ids"]) == {"candidate_a", "candidate_b"}
    assert group["independent_parent_count"] == 2
    assert rows[0]["corroborated_by_independent_tracklet"] is True
    assert rows[1]["corroborated_by_independent_tracklet"] is True


def test_review_selection_reserves_slot_for_context_corroborated_candidate() -> None:
    policy = {
        "minimum_retrieval_score": 0.65,
        "minimum_prototype_similarity": 0.60,
        "minimum_plausible_score": 0.45,
        "minimum_plausible_prototype": 0.45,
        "review_candidate_count": 3,
    }
    rows = [
        _candidate(
            "dark_high_1",
            "dark_parent_1",
            10,
            25,
            retrieval=0.70,
            prototype_similarity=0.75,
            context_similarity=0.20,
            prototype=_unit(0.0, 1.0, 0.0),
        ),
        _candidate(
            "dark_high_2",
            "dark_parent_2",
            40,
            55,
            retrieval=0.68,
            prototype_similarity=0.72,
            context_similarity=0.25,
            prototype=_unit(0.0, 0.98, 0.1),
        ),
        _candidate(
            "target_track_a",
            "target_parent_a",
            100,
            130,
            retrieval=0.53,
            prototype_similarity=0.57,
            context_similarity=0.93,
            prototype=_unit(1.0, 0.0, 0.0),
        ),
        _candidate(
            "target_track_b",
            "target_parent_b",
            104,
            116,
            retrieval=0.50,
            prototype_similarity=0.54,
            context_similarity=0.91,
            prototype=_unit(0.99, 0.04, 0.0),
        ),
    ]
    assignments = {
        "dark_high_1": _assignments("dark_high_1", 10, 25, x=50.0),
        "dark_high_2": _assignments("dark_high_2", 40, 55, x=1300.0),
        "target_track_a": _assignments("target_track_a", 100, 130, x=600.0),
        "target_track_b": _assignments("target_track_b", 104, 116, x=606.0),
    }
    selected, rescue, _, _, corroboration_groups = (
        ADAPTER._phase4b_select_assisted_review_candidates(
            rows,
            policy=policy,
            assignments_by_candidate=assignments,
            np=np,
        )
    )
    selected_ids = {str(row["candidate_id"]) for row in selected}
    assert selected_ids & {"target_track_a", "target_track_b"}
    assert any(row["corroborated"] is True for row in corroboration_groups)
    assert any(
        row.get("rescue_review_required") is True
        and row.get("corroborated_by_independent_tracklet") is True
        for row in rescue
    )


def test_r11_source_has_no_sample_specific_target_rules() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8").lower()
    assert "shot_0010_track_0047" not in source
    assert "shot_0010_track_0022" not in source
    assert "frame_000918" not in source
    assert "jersey_10" not in source
    assert "white_uniform" not in source
    assert "goal_replay" not in source


def test_target_context_corroboration_verifier_accepts_grounded_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    ranked = job / "ranked_candidates.json"
    contact = job / "ranked_candidates.jpg"
    ranked.write_text(json.dumps({"reviewable_candidates": []}), encoding="utf-8")
    contact.write_bytes(b"contact")
    attempt = {
        "policy": VERIFIER.PHASE4B_POLICY,
        "shot_id": "shot_test",
        "target_context_policy": VERIFIER.TARGET_CONTEXT_POLICY,
        "target_context_is_soft_evidence_only": True,
        "corroboration_policy": VERIFIER.CORROBORATION_POLICY,
        "corroboration_group_count": 0,
        "corroboration_groups": [],
        "assisted_review_selection_policy": VERIFIER.SELECTION_POLICY,
        "review_candidate_count": 0,
        "review_candidates": [],
        "automatic_target_confirmation": False,
        "candidate_link_created": False,
        "ranked_candidates_path": str(ranked),
        "ranked_candidates_sha256": _sha(ranked),
        "contact_sheet_path": str(contact),
        "contact_sheet_sha256": _sha(contact),
    }
    (job / "pipeline_state.json").write_text(
        json.dumps({"pending_action": None}), encoding="utf-8"
    )
    (job / "phase4b_first_cross_shot_report.json").write_text(
        json.dumps({"policy": VERIFIER.PHASE4B_POLICY, "attempts": [attempt]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--job-root", str(job), "--shot-id", "shot_test"],
    )
    assert VERIFIER.main() == 0
