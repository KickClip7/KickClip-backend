from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from common import (
    atomic_json,
    load_module,
    portable,
    sha256_file,
    write_csv,
)
from contracts import validate_reviewed_shot_boundaries


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _border_risk(box: list[float], width: int, height: int) -> bool:
    return bool(
        box[0] <= 1
        or box[1] <= 1
        or box[2] >= width - 1
        or box[3] >= height - 1
    )


def _frame(capture, cv2, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, value = capture.read()
    if not ok or value is None:
        raise RuntimeError(f"Cannot decode frame {frame_index}")
    return value


def _crop(frame, box: list[float]):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * 0.12
    pad_y = (y2 - y1) * 0.08
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    return frame[top:bottom, left:right]


def _draw_box(cv2, frame, box: list[float], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)
    cv2.putText(
        frame,
        label,
        (max(0, x1), max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )


def _representative_observation(
    observations: list[dict[str, Any]],
    *,
    by_frame: dict[int, list[Any]],
    capture,
    cv2,
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[int, float]]:
    blur_by_frame: dict[int, float] = {}
    scored = []
    for observation in observations:
        index = int(observation["frame_index"])
        frame = _frame(capture, cv2, index)
        crop = _crop(frame, observation["bbox_xyxy"])
        blur = (
            float(cv2.Laplacian(crop, cv2.CV_64F).var())
            if crop.size
            else 0.0
        )
        blur_by_frame[index] = blur
        box = observation["bbox_xyxy"]
        area_ratio = (
            max(0.0, box[2] - box[0])
            * max(0.0, box[3] - box[1])
            / float(width * height)
        )
        max_overlap = max(
            (
                _iou(box, list(other.bbox))
                for other in by_frame.get(index, [])
                if str(other.detection_id) != str(observation["detection_id"])
            ),
            default=0.0,
        )
        score = (
            0.38 * float(observation["confidence"])
            + 0.28 * min(1.0, math.sqrt(max(area_ratio, 0.0) * 15.0))
            + 0.20 * min(1.0, blur / 250.0)
            + 0.14 * (1.0 - min(1.0, max_overlap))
            - (0.20 if _border_risk(box, width, height) else 0.0)
        )
        scored.append((score, -index, observation, max_overlap, area_ratio))
    _, _, selected, overlap, area_ratio = max(scored, key=lambda row: row[:2])
    return {
        **selected,
        "blur_laplacian_variance": blur_by_frame[int(selected["frame_index"])],
        "max_other_player_iou": overlap,
        "bbox_area_ratio": area_ratio,
    }, blur_by_frame


def _initialization_observation(
    observations: list[dict[str, Any]],
    *,
    by_frame: dict[int, list[Any]],
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    stable_min = int(policy["minimum_stable_observations"])
    required_after = float(policy["minimum_tracking_duration_sec"])
    ordered = sorted(observations, key=lambda row: int(row["frame_index"]))
    for offset, observation in enumerate(ordered):
        box = observation["bbox_xyxy"]
        max_overlap = max(
            (
                _iou(box, list(other.bbox))
                for other in by_frame.get(int(observation["frame_index"]), [])
                if str(other.detection_id) != str(observation["detection_id"])
            ),
            default=0.0,
        )
        remaining_stable = len(ordered) - offset
        remaining_scene = (frame_count - int(observation["frame_index"])) / fps
        if (
            remaining_stable >= stable_min
            and remaining_scene >= required_after
            and not _border_risk(box, width, height)
            and max_overlap <= float(policy["maximum_initialization_overlap_iou"])
        ):
            return {
                **observation,
                "validation_state": "VALID",
                "stable_observation_count": remaining_stable,
                "remaining_scene_duration_sec": remaining_scene,
                "max_other_player_iou": max_overlap,
            }
    fallback = ordered[0]
    return {
        **fallback,
        "validation_state": "INVALID",
        "validation_reason": "NO_STABLE_OBSERVATION_WITH_REQUIRED_FORWARD_DURATION",
        "stable_observation_count": len(ordered),
        "remaining_scene_duration_sec": (
            frame_count - int(fallback["frame_index"])
        )
        / fps,
    }


def _render_candidate(
    *,
    cv2,
    capture,
    video: Path,
    output_root: Path,
    candidate_id: str,
    observations: list[dict[str, Any]],
    representative: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    root = output_root / "candidates" / candidate_id
    crops_root = root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    selected_indexes = sorted(
        {
            0,
            len(observations) // 4,
            len(observations) // 2,
            (len(observations) * 3) // 4,
            len(observations) - 1,
        }
    )
    crop_paths: list[str] = []
    crop_images = []
    for number, index in enumerate(selected_indexes, start=1):
        observation = observations[index]
        frame = _frame(capture, cv2, int(observation["frame_index"]))
        crop = _crop(frame, observation["bbox_xyxy"])
        if not crop.size:
            continue
        path = crops_root / f"crop_{number:02d}.jpg"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(path)
        crop_paths.append(portable(path, output_root))
        crop_images.append(crop)
    rep_frame = _frame(capture, cv2, int(representative["frame_index"]))
    rep_crop = _crop(rep_frame, representative["bbox_xyxy"])
    representative_path = root / "representative.jpg"
    if not cv2.imwrite(str(representative_path), rep_crop):
        raise RuntimeError(representative_path)
    contact_sheet_path = root / "contact_sheet.jpg"
    if crop_images:
        cell_width, cell_height = 260, 300
        import numpy as np

        sheet = np.full(
            (cell_height, cell_width * len(crop_images), 3),
            32,
            dtype=np.uint8,
        )
        for index, image in enumerate(crop_images):
            scale = min(cell_width / image.shape[1], cell_height / image.shape[0])
            resized = cv2.resize(
                image,
                (
                    max(1, int(image.shape[1] * scale)),
                    max(1, int(image.shape[0] * scale)),
                ),
            )
            x = index * cell_width + (cell_width - resized.shape[1]) // 2
            y = (cell_height - resized.shape[0]) // 2
            sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        if not cv2.imwrite(str(contact_sheet_path), sheet):
            raise RuntimeError(contact_sheet_path)

    start = int(observations[0]["frame_index"])
    end = int(observations[-1]["frame_index"])
    capture_video = cv2.VideoCapture(str(video))
    capture_video.set(cv2.CAP_PROP_POS_FRAMES, start)
    output_video = root / "tracklet_review.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    by_index = {int(row["frame_index"]): row for row in observations}
    for index in range(start, end + 1):
        ok, frame = capture_video.read()
        if not ok or frame is None:
            break
        observation = by_index.get(index)
        if observation:
            _draw_box(cv2, frame, observation["bbox_xyxy"], candidate_id[-28:])
        writer.write(frame)
    writer.release()
    capture_video.release()
    return {
        "representative": portable(representative_path, output_root),
        "contact_sheet": portable(contact_sheet_path, output_root),
        "tracklet_review_video": portable(output_video, output_root),
        "crops": crop_paths,
    }


def discover_from_frozen_detections(
    *,
    project_root: Path,
    scene_id: str,
    candidate_namespace: str,
    video: Path,
    detections_csv: Path,
    shot_boundaries_path: Path,
    output_root: Path,
    policy: dict[str, Any],
    candidate_policy_sha256: str,
    package_manifest_sha256: str,
) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_sha = sha256_file(video)
    boundaries = validate_reviewed_shot_boundaries(
        path=shot_boundaries_path,
        video_sha256=video_sha,
        frame_count=frame_count,
    )
    stage2 = load_module(
        "scene_selection_frozen_v1_stage2",
        project_root
        / "target_centric_tracking_v1"
        / "stage2_run_conservative_target_association.py",
    )
    by_frame, _ = stage2.load_detections(
        detections_csv,
        frame_count,
        width,
        height,
    )
    frozen_b0 = load_module(
        "scene_selection_frozen_v2_stage3b0",
        project_root
        / "target_centric_tracking_v2"
        / "stage3b0_build_postcut_candidate_tracklets.py",
    )
    tracklet_policy = policy["local_tracklets"]
    candidate_rows: list[dict[str, Any]] = []
    gallery_shots: list[dict[str, Any]] = []
    shot_boundary_sha = sha256_file(shot_boundaries_path)
    rfdetr_sha = sha256_file(
        project_root / "weights/rfdetr/checkpoint_best_regular.pth"
    )
    for shot in boundaries["shots"]:
        tracks, _ = frozen_b0.build_tracklets(
            by_frame=by_frame,
            start_frame=int(shot["start_frame"]),
            end_frame_inclusive=int(shot["end_frame_inclusive"]),
            max_age=int(tracklet_policy["max_age"]),
            minimum_predicted_iou=float(
                tracklet_policy["minimum_predicted_iou"]
            ),
            maximum_center_distance=float(
                tracklet_policy["maximum_center_distance"]
            ),
            minimum_area_ratio=float(tracklet_policy["minimum_area_ratio"]),
            maximum_area_ratio=float(tracklet_policy["maximum_area_ratio"]),
            minimum_match_score=float(tracklet_policy["minimum_match_score"]),
        )
        tracks = [
            track
            for track in tracks
            if len(track.observations)
            >= int(tracklet_policy["minimum_tracklet_frames"])
        ]
        tracks.sort(
            key=lambda track: (
                track.start_frame,
                track.end_frame,
                track.internal_id,
            )
        )
        shot_candidate_ids = []
        for number, track in enumerate(tracks, start=1):
            candidate_id = (
                f"scene_candidate_{candidate_namespace}_{shot['shot_id']}_"
                f"track_{number:04d}"
            )
            observations = [
                {
                    "frame_index": int(item.frame_index),
                    "time_sec": int(item.frame_index) / fps,
                    "detection_id": str(item.detection_id),
                    "bbox_xyxy": [float(value) for value in item.bbox_xyxy],
                    "confidence": float(item.confidence),
                }
                for item in track.observations
            ]
            representative, blur_by_frame = _representative_observation(
                observations,
                by_frame=by_frame,
                capture=capture,
                cv2=cv2,
                width=width,
                height=height,
            )
            initialization = _initialization_observation(
                observations,
                by_frame=by_frame,
                width=width,
                height=height,
                frame_count=frame_count,
                fps=fps,
                policy=policy["initialization"],
            )
            artifacts = _render_candidate(
                cv2=cv2,
                capture=capture,
                video=video,
                output_root=output_root,
                candidate_id=candidate_id,
                observations=observations,
                representative=representative,
                fps=fps,
            )
            mean_confidence = sum(
                float(row["confidence"]) for row in observations
            ) / len(observations)
            mean_blur = sum(blur_by_frame.values()) / len(blur_by_frame)
            border = _border_risk(
                representative["bbox_xyxy"], width, height
            )
            trackability = max(
                0.0,
                min(
                    1.0,
                    0.50 * mean_confidence
                    + 0.25 * min(1.0, len(observations) / 30.0)
                    + 0.15 * min(1.0, mean_blur / 250.0)
                    + 0.10 * (0.0 if border else 1.0),
                ),
            )
            row = {
                "candidate_id": candidate_id,
                "scene_id": scene_id,
                "shot_id": str(shot["shot_id"]),
                "shot_index": int(shot["shot_index"]),
                "local_tracklet_id": f"track_{number:04d}",
                "first_frame": int(track.start_frame),
                "last_frame": int(track.end_frame),
                "observation_count": len(observations),
                "representative_observation": {
                    "frame_index": int(representative["frame_index"]),
                    "time_sec": int(representative["frame_index"]) / fps,
                    "bbox_xyxy": representative["bbox_xyxy"],
                    "detector_confidence": representative["confidence"],
                    "thumbnail_artifact": artifacts["representative"],
                },
                "tracking_initialization_observation": {
                    "frame_index": int(initialization["frame_index"]),
                    "time_sec": int(initialization["frame_index"]) / fps,
                    "bbox_xyxy": initialization["bbox_xyxy"],
                    "validation_state": initialization["validation_state"],
                    "validation_reason": initialization.get(
                        "validation_reason"
                    ),
                    "stable_observation_count": initialization[
                        "stable_observation_count"
                    ],
                },
                "quality": {
                    "trackability_score": round(trackability, 6),
                    "mean_detection_confidence": round(mean_confidence, 6),
                    "blur_score": round(1.0 / (1.0 + mean_blur / 100.0), 6),
                    "occlusion_score": round(
                        float(representative["max_other_player_iou"]), 6
                    ),
                    "border_risk": border,
                    "identity_switch_risk": (
                        "HIGH"
                        if float(representative["max_other_player_iou"]) > 0.6
                        else "LOW"
                    ),
                },
                "artifacts": {
                    "contact_sheet": artifacts["contact_sheet"],
                    "tracklet_review_video": artifacts[
                        "tracklet_review_video"
                    ],
                    "crops": artifacts["crops"],
                },
                "observations": observations,
                "gallery_visibility": (
                    "VISIBLE"
                    if trackability
                    >= float(policy["gallery"]["default_visible_score"])
                    else "HIDDEN_LOW_QUALITY"
                ),
                "provenance": {
                    "detector_checkpoint_sha256": rfdetr_sha,
                    "candidate_policy_sha256": candidate_policy_sha256,
                    "shot_boundary_sha256": shot_boundary_sha,
                    "package_manifest_sha256": package_manifest_sha256,
                    "detector_source": "FROZEN_V1_RFDETR_DETECTIONS",
                    "local_association": "FROZEN_V2_STAGE3B0",
                    "cross_shot_identity_linked": False,
                },
            }
            candidate_rows.append(row)
            shot_candidate_ids.append(candidate_id)
            atomic_json(
                output_root / "candidates" / candidate_id / "candidate.json",
                row,
            )
        gallery_shots.append(
            {
                "shot_id": str(shot["shot_id"]),
                "shot_index": int(shot["shot_index"]),
                "start_frame": int(shot["start_frame"]),
                "end_frame_inclusive": int(shot["end_frame_inclusive"]),
                "start_time_sec": float(shot["start_time_sec"]),
                "end_time_sec": float(shot["end_time_sec"]),
                "candidate_ids": shot_candidate_ids,
            }
        )
    capture.release()
    candidate_rows.sort(
        key=lambda row: (int(row["shot_index"]), int(row["first_frame"]))
    )
    result = {
        "schema_version": "kickclip.scene_player_candidates.v1",
        "scene_id": scene_id,
        "status": "WAITING_TARGET_SELECTION",
        "video": {
            "sha256": video_sha,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },
        "shot_boundary": {
            "sha256": shot_boundary_sha,
            "schema_version": boundaries.get("schema_version"),
            "review_status": "COMPLETE",
            "shot_count": len(boundaries["shots"]),
            "review_provenance": boundaries.get("review_contract", {}).get(
                "review_history", []
            ),
        },
        "candidates": candidate_rows,
        "provenance": {
            "ground_truth_used": False,
            "automatic_target_selection": False,
            "cross_shot_identity_linked": False,
            "detections_sha256": sha256_file(detections_csv),
        },
    }
    atomic_json(output_root / "scene_candidates.json", result)
    atomic_json(
        output_root / "candidate_gallery.json",
        {
            "schema_version": "kickclip.scene_candidate_gallery.v1",
            "scene_id": scene_id,
            "status": "WAITING_TARGET_SELECTION",
            "sort": ["shot_index", "first_frame"],
            "hidden_candidates_preserved": True,
            "shots": gallery_shots,
        },
    )
    write_csv(
        output_root / "scene_candidates.csv",
        (
            {
                "candidate_id": row["candidate_id"],
                "shot_id": row["shot_id"],
                "shot_index": row["shot_index"],
                "first_frame": row["first_frame"],
                "last_frame": row["last_frame"],
                "observation_count": row["observation_count"],
                "representative_frame": row[
                    "representative_observation"
                ]["frame_index"],
                "initialization_frame": row[
                    "tracking_initialization_observation"
                ]["frame_index"],
                "initialization_state": row[
                    "tracking_initialization_observation"
                ]["validation_state"],
                "trackability_score": row["quality"]["trackability_score"],
                "gallery_visibility": row["gallery_visibility"],
            }
            for row in candidate_rows
        ),
        [
            "candidate_id",
            "shot_id",
            "shot_index",
            "first_frame",
            "last_frame",
            "observation_count",
            "representative_frame",
            "initialization_frame",
            "initialization_state",
            "trackability_score",
            "gallery_visibility",
        ],
    )
    return result
