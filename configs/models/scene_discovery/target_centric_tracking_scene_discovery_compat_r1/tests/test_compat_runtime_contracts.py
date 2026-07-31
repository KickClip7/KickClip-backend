from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any


def _write_video(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("Synthetic MP4 writer could not be opened.")
    for frame_index in range(12):
        frame = np.full((64, 64, 3), 24, dtype=np.uint8)
        left = 16 + frame_index
        cv2.rectangle(frame, (left, 10), (left + 18, 54), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def _arguments(
    *,
    project_root: Path,
    video: Path,
    detections: Path,
    shots: Path,
    output: Path,
) -> Namespace:
    return Namespace(
        command="discover",
        project_root=project_root,
        scene_id="scene_compat_smoke",
        discovery_id="discovery_compat_repeat",
        video=video,
        detections_csv=detections,
        shot_boundaries=shots,
        output_root=output,
    )


def _assert_output_contract(
    manifest: dict[str, Any],
    *,
    output: Path,
) -> None:
    assert manifest["runtime_mode"] == "COMPAT_R1"
    assert manifest["automatic_target_selection"] is False
    assert manifest["candidate_count"] >= 1
    assert manifest["shot_count"] == 1
    assert len(manifest["candidate_cache_key"]) == 64
    assert len(manifest["scene_candidates"]["sha256"]) == 64
    candidates = json.loads(
        (output / "scene_candidates.json").read_text(encoding="utf-8")
    )
    assert candidates["provenance"]["automatic_target_selection"] is False
    assert candidates["provenance"]["cross_shot_identity_linked"] is False
    assert candidates["candidates"]
    assert all(
        row["provenance"]["cross_shot_identity_linked"] is False
        for row in candidates["candidates"]
    )


def run_contract_tests(
    *,
    runtime_module,
    project_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    runtime_root = Path(runtime_module.__file__).resolve().parent
    fixtures = runtime_root / "tests" / "fixtures"
    work_root.mkdir(parents=True, exist_ok=True)
    video = work_root / "synthetic_scene.mp4"
    detections = work_root / "detections.csv"
    shots = work_root / "reviewed_shots.json"
    _write_video(video)
    shutil.copyfile(fixtures / "synthetic_detections.csv", detections)
    template = (fixtures / "reviewed_shots_template.json").read_text(
        encoding="utf-8"
    )
    shots.write_text(
        template.replace(
            "__VIDEO_SHA256__", runtime_module.sha256_file(video)
        ),
        encoding="utf-8",
    )

    output_a = work_root / "output_a"
    output_b = work_root / "output_b"
    first = runtime_module.discover(
        _arguments(
            project_root=project_root,
            video=video,
            detections=detections,
            shots=shots,
            output=output_a,
        )
    )
    second = runtime_module.discover(
        _arguments(
            project_root=project_root,
            video=video,
            detections=detections,
            shots=shots,
            output=output_b,
        )
    )
    _assert_output_contract(first, output=output_a)
    _assert_output_contract(second, output=output_b)
    first_candidates = runtime_module.sha256_file(
        output_a / "scene_candidates.json"
    )
    second_candidates = runtime_module.sha256_file(
        output_b / "scene_candidates.json"
    )
    assert first_candidates == second_candidates
    assert first["candidate_cache_key"] == second["candidate_cache_key"]

    rejected_states: list[str] = []
    original = json.loads(shots.read_text(encoding="utf-8"))
    for state in ("UNREVIEWED", "REJECTED", "PASS"):
        invalid = json.loads(json.dumps(original))
        invalid["shots"][0]["review_state"] = state
        invalid_path = work_root / f"shots_{state.lower()}.json"
        invalid_path.write_text(
            json.dumps(invalid, sort_keys=True), encoding="utf-8"
        )
        try:
            runtime_module.audit_reviewed_shots(invalid_path)
        except ValueError as exc:
            if "WAITING_SHOT_BOUNDARY_REVIEW" not in str(exc):
                raise
            rejected_states.append(state)
        else:
            raise AssertionError(f"Unapproved shot state accepted: {state}")

    forbidden_modules = {
        "appearance",
        "earlier_anchor",
        "launch",
        "selection",
    }
    imported_forbidden = sorted(forbidden_modules.intersection(set(__import__("sys").modules)))
    assert not imported_forbidden
    absent_r3_directories = [
        name
        for name in (
            "target_centric_tracking_v2_production_r2",
            "target_centric_tracking_scene_target_selection_r2",
            "target_centric_tracking_v2_production_r3",
        )
        if not (project_root / name).exists()
    ]
    return {
        "synthetic_discovery_smoke": True,
        "deterministic_repeat": True,
        "candidate_count": first["candidate_count"],
        "scene_candidates_sha256": first_candidates,
        "unsupported_unreviewed_shot_rejection": True,
        "rejected_review_states": rejected_states,
        "automatic_target_selection": False,
        "r3_absence_does_not_block_discovery": True,
        "absent_r3_directories": absent_r3_directories,
        "forbidden_modules_imported": imported_forbidden,
    }

