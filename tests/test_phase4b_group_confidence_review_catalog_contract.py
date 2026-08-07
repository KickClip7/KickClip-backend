from __future__ import annotations

import csv
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("kickclip_phase4b_r12_adapter_test", ADAPTER_PATH)


def _unit(*values: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32)
    return row / np.linalg.norm(row)


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
    purity: float,
    clean_ratio: float,
) -> dict[str, object]:
    count = end - start + 1
    clean_count = max(3, int(round(count * clean_ratio)))
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
            "clean_frame_count": clean_count,
            "clean_frame_ratio": clean_ratio,
        },
        "identity_purity": {
            "policy": ADAPTER.PHASE4B_IDENTITY_PURITY_POLICY,
            "passed": True,
            "parent_tracklet_id": parent_id,
            "internal_pairwise_cosine_median": purity,
            "target_memory_used_for_segmentation": False,
            "automatic_target_confirmation": False,
        },
        "role_confusion": {"passed": True},
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


def test_group_confidence_selects_clean_target_representative_despite_low_context() -> None:
    policy = {
        "minimum_retrieval_score": 0.65,
        "minimum_prototype_similarity": 0.60,
        "minimum_plausible_score": 0.45,
        "minimum_plausible_prototype": 0.45,
        "review_candidate_count": 3,
    }
    target_a = _candidate(
        "target_long_mixed",
        "parent_target_a",
        100,
        130,
        retrieval=0.546,
        prototype_similarity=0.583,
        context_similarity=0.11,
        prototype=_unit(1.0, 0.0, 0.0),
        purity=0.79,
        clean_ratio=0.53,
    )
    target_b = _candidate(
        "target_clean",
        "parent_target_b",
        118,
        145,
        retrieval=0.527,
        prototype_similarity=0.574,
        context_similarity=0.10,
        prototype=_unit(0.944, 0.329, 0.0),
        purity=0.87,
        clean_ratio=0.93,
    )
    false_a = _candidate(
        "false_context_a",
        "parent_false_a",
        200,
        225,
        retrieval=0.50,
        prototype_similarity=0.53,
        context_similarity=0.31,
        prototype=_unit(0.0, 1.0, 0.0),
        purity=0.78,
        clean_ratio=0.80,
    )
    false_b = _candidate(
        "false_context_b",
        "parent_false_b",
        210,
        230,
        retrieval=0.50,
        prototype_similarity=0.53,
        context_similarity=0.30,
        prototype=_unit(0.0, 0.874, 0.486),
        purity=0.80,
        clean_ratio=0.82,
    )
    strict_false = _candidate(
        "strict_false",
        "parent_strict_false",
        300,
        320,
        retrieval=0.67,
        prototype_similarity=0.73,
        context_similarity=0.34,
        prototype=_unit(0.0, 0.0, 1.0),
        purity=0.84,
        clean_ratio=1.0,
    )
    rows = [target_a, target_b, false_a, false_b, strict_false]
    assignments = {
        "target_long_mixed": _assignments("target_long_mixed", 100, 130, x=300.0),
        "target_clean": _assignments("target_clean", 118, 145, x=305.0),
        "false_context_a": _assignments("false_context_a", 200, 225, x=700.0),
        "false_context_b": _assignments("false_context_b", 210, 230, x=706.0),
        "strict_false": _assignments("strict_false", 300, 320, x=1100.0),
    }

    selected, _, unselected, _, groups = ADAPTER._phase4b_select_assisted_review_candidates(
        rows,
        policy=policy,
        assignments_by_candidate=assignments,
        np=np,
    )
    selected_ids = [str(row["candidate_id"]) for row in selected]
    assert "target_clean" in selected_ids
    assert selected_ids[0] == "target_clean"
    target_group = next(
        group
        for group in groups
        if set(group["member_candidate_ids"])
        == {"target_long_mixed", "target_clean"}
    )
    assert target_group["representative_candidate_id"] == "target_clean"
    assert target_group["group_confidence_policy"] == ADAPTER.PHASE4B_GROUP_CONFIDENCE_POLICY
    catalog = sorted(
        [*selected, *unselected], key=lambda row: int(row["review_catalog_rank"])
    )
    assert all(row["manual_review_promotion_allowed"] is True for row in catalog)
    assert all(row["automatic_target_confirmation"] is False for row in catalog)


def _write_video(path: Path, frame_count: int = 5) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 64),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((64, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_manual_promotion_changes_only_pending_review_navigation(tmp_path: Path) -> None:
    output = tmp_path / "job"
    shot_root = output / "phase4b_cross_shot" / "shot_0010"
    candidate_id = "safe_candidate"
    candidate_dir = shot_root / "candidates" / candidate_id
    references = candidate_dir / "references"
    references.mkdir(parents=True)
    image = np.full((32, 16, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(references / "reference_01_frame_000002.jpg"), image)
    strips = shot_root / "candidate_strips"
    strips.mkdir(parents=True)
    assert cv2.imwrite(str(strips / f"{candidate_id}.jpg"), image)
    video = output / "source.mp4"
    output.mkdir(parents=True, exist_ok=True)
    _write_video(video)
    (shot_root / "shot_clip.mp4").write_bytes(b"already-materialized")

    ambiguity_id = "ambiguity_phase4b_shot_0010_g019"
    state = {
        "video": {"path": str(video), "fps": 5.0, "frame_count": 5},
        "shots": [
            {
                "shot_id": "shot_0010",
                "start_frame": 0,
                "end_frame_inclusive": 4,
            }
        ],
        "ambiguities": [
            {
                "ambiguity_id": ambiguity_id,
                "shot_id": "shot_0010",
                "status": "PENDING",
                "review_candidates": [],
            }
        ],
        "pending_action": {
            "type": "CROSS_SHOT_CONFIRMATION",
            "ambiguity_id": ambiguity_id,
            "shot_id": "shot_0010",
            "candidate_ids": ["wrong_top_candidate"],
            "automatic_target_confirmation": False,
        },
        "runtime": {
            "memory_revision_path": "unchanged-memory.json",
            "memory_revision_sha256": "a" * 64,
            "automatic_target_confirmation": False,
        },
        "updated_at": "old",
    }
    ADAPTER.atomic_json(output / "pipeline_state.json", state)
    candidate = {
        "candidate_id": candidate_id,
        "parent_tracklet_id": candidate_id,
        "start_frame": 1,
        "end_frame_inclusive": 3,
        "retrieval_score": 0.53,
        "prototype_target_similarity": 0.57,
        "role_confusion": {"passed": True},
        "identity_observability": {"passed": True, "clean_frame_count": 3, "clean_frame_ratio": 1.0},
        "identity_purity": {"passed": True, "internal_pairwise_cosine_median": 0.9},
        "negative_review_gate": {"passed": True},
        "automatic_target_confirmation": False,
    }
    ranked_path = shot_root / "ranked_candidates.json"
    ADAPTER.atomic_json(ranked_path, {"all_unique_candidates": [candidate]})
    ADAPTER.atomic_json(
        shot_root / "phase4b_attempt_report.json",
        {"ranked_candidates_sha256": ADAPTER.sha256_file(ranked_path)},
    )
    with (shot_root / "candidate_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "candidate_id",
                "frame_index",
                "detection_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "clean_for_reid",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": candidate_id,
                "frame_index": 2,
                "detection_id": "d1",
                "confidence": 0.9,
                "x1": 10,
                "y1": 10,
                "x2": 30,
                "y2": 50,
                "clean_for_reid": True,
            }
        )

    result = ADAPTER._phase4b_promote_review_candidate(
        output_dir=output,
        ambiguity_id=ambiguity_id,
        candidate_id=candidate_id,
        reviewer="USER",
        note="selected after reviewing the full safe catalog",
    )
    updated = ADAPTER.read_object(output / "pipeline_state.json")
    assert result["status"] == "PASS"
    assert updated["pending_action"]["candidate_ids"] == [candidate_id]
    assert updated["pending_action"]["manual_review_promoted_candidate_id"] == candidate_id
    assert updated["runtime"]["memory_revision_path"] == "unchanged-memory.json"
    assert updated["runtime"]["memory_revision_sha256"] == "a" * 64
    assert updated["runtime"]["automatic_target_confirmation"] is False
    assert Path(updated["pending_action"]["review_promotion_path"]).is_file()
    promoted = updated["ambiguities"][0]["review_candidates"][0]
    assert promoted["candidate_id"] == candidate_id
    assert promoted["automatic_target_confirmation"] is False
