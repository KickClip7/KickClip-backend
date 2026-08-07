from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.domains.candidate_handoff_r1.reviewed_input_recovery import (
    ReviewedInputRecoveryError,
    find_recoverable_reviewed_input_bundle,
    sha256_file,
)


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _frames(count: int, seed: int = 0) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for index in range(count):
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        frame[:] = ((index * 7 + seed) % 255, (index * 11) % 255, (index * 17) % 255)
        cv2.rectangle(frame, (10 + index % 30, 15), (45 + index % 30, 60), (255, 255, 255), -1)
        cv2.putText(frame, str(index), (3, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        result.append(frame)
    return result


def _bundle(
    storage: Path,
    *,
    project: str,
    revision: str,
    scene_id: str,
    scene_frames: list[np.ndarray],
    review_state: str = "REVIEWED_PASS",
    detection_confidence: str = "0.90",
) -> Path:
    root = storage / "candidate_pipeline_inputs" / project / revision / scene_id
    video = root / "scene.mp4"
    _write_video(video, scene_frames)
    video_sha = sha256_file(video)
    reviewed = {
        "scene_id": scene_id,
        "video": {"sha256": video_sha},
        "shots": [
            {
                "shot_id": "shot_0000",
                "start_frame": 0,
                "end_frame": len(scene_frames) - 1,
                "review_status": review_state,
            }
        ],
    }
    (root / "reviewed_shots.json").write_text(
        json.dumps(reviewed), encoding="utf-8"
    )
    with (root / "detections.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["frame_index", "detection_id", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_index": 0,
                "detection_id": "det_0",
                "confidence": detection_confidence,
                "x1": 10,
                "y1": 15,
                "x2": 45,
                "y2": 60,
            }
        )
    return root


def test_recovers_identical_bundle_after_database_reset(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    source_frames = _frames(60)
    source = storage / "matches" / "source.mp4"
    _write_video(source, source_frames)
    scene_frames = source_frames[20:40]
    first = _bundle(
        storage,
        project="old_project_a",
        revision="old_revision_a",
        scene_id="evt_scene",
        scene_frames=scene_frames,
    )
    second = storage / "candidate_pipeline_inputs" / "old_project_b" / "old_revision_b" / "evt_scene"
    second.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(first, second)

    recovered = find_recoverable_reviewed_input_bundle(
        storage_root=storage,
        scene_id="evt_scene",
        source_video_path=source,
        source_video_sha256=sha256_file(source),
        source_start_sec=2.0,
        source_end_sec=4.0,
        source_fps=10.0,
    )

    assert recovered.identity == (
        sha256_file(second / "scene.mp4"),
        sha256_file(second / "reviewed_shots.json"),
        sha256_file(second / "detections.csv"),
    )
    assert recovered.shot_count == 1
    assert recovered.source_match_similarity >= 0.93


def test_does_not_promote_pending_review(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    source_frames = _frames(40)
    source = storage / "source.mp4"
    _write_video(source, source_frames)
    _bundle(
        storage,
        project="old",
        revision="rev",
        scene_id="evt_scene",
        scene_frames=source_frames[10:30],
        review_state="PENDING_REVIEW",
    )

    with pytest.raises(ReviewedInputRecoveryError) as exc:
        find_recoverable_reviewed_input_bundle(
            storage_root=storage,
            scene_id="evt_scene",
            source_video_path=source,
            source_video_sha256=sha256_file(source),
            source_start_sec=1.0,
            source_end_sec=3.0,
            source_fps=10.0,
        )
    assert exc.value.code == "REVIEWED_SHOT_BOUNDARIES_NOT_READY"


def test_rejects_scene_from_different_source_interval(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    source_frames = _frames(50)
    source = storage / "source.mp4"
    _write_video(source, source_frames)
    _bundle(
        storage,
        project="old",
        revision="rev",
        scene_id="evt_scene",
        scene_frames=_frames(20, seed=130),
    )

    with pytest.raises(ReviewedInputRecoveryError) as exc:
        find_recoverable_reviewed_input_bundle(
            storage_root=storage,
            scene_id="evt_scene",
            source_video_path=source,
            source_video_sha256=sha256_file(source),
            source_start_sec=1.0,
            source_end_sec=3.0,
            source_fps=10.0,
        )
    assert exc.value.code == "REVIEWED_SHOT_BOUNDARIES_NOT_READY"


def test_different_matching_bundles_are_not_selected_silently(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    source_frames = _frames(40)
    source = storage / "source.mp4"
    _write_video(source, source_frames)
    scene = source_frames[10:30]
    _bundle(
        storage,
        project="old_a",
        revision="rev_a",
        scene_id="evt_scene",
        scene_frames=scene,
        detection_confidence="0.90",
    )
    _bundle(
        storage,
        project="old_b",
        revision="rev_b",
        scene_id="evt_scene",
        scene_frames=scene,
        detection_confidence="0.80",
    )

    with pytest.raises(ReviewedInputRecoveryError) as exc:
        find_recoverable_reviewed_input_bundle(
            storage_root=storage,
            scene_id="evt_scene",
            source_video_path=source,
            source_video_sha256=sha256_file(source),
            source_start_sec=1.0,
            source_end_sec=3.0,
            source_fps=10.0,
        )
    assert exc.value.code == "AMBIGUOUS_REVIEWED_INPUT_RECOVERY"


def test_preparation_contains_storage_recovery_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "app/domains/candidate_handoff_r1/preparation.py"
    ).read_text(encoding="utf-8")
    assert "find_recoverable_reviewed_input_bundle" in source
    assert '"input_resolution_mode": "VERIFIED_STORAGE_RECOVERY"' in source
    assert '"automatic_review_approval": False' in source
