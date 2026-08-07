from __future__ import annotations

from pathlib import Path
from typing import Any

from common import atomic_json, canonical_sha256, now_iso, portable, read_json


def _candidate(candidates: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for row in candidates.get("candidates", []):
        if str(row.get("candidate_id")) == candidate_id:
            return row
    raise ValueError("Candidate does not belong to this scene candidate manifest.")


def next_selection_revision(output_root: Path) -> int:
    revisions = []
    for path in output_root.glob("target_selection_r*.json"):
        try:
            revisions.append(int(path.stem.rsplit("_r", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(revisions, default=0) + 1


def create_target_selection(
    *,
    output_root: Path,
    selected_candidate_id: str,
    reviewer: str,
    production_manifest_sha256: str,
    selection_revision: int | None = None,
) -> dict[str, Any]:
    candidates_path = output_root / "scene_candidates.json"
    candidates = read_json(candidates_path)
    candidate = _candidate(candidates, selected_candidate_id)
    revision = selection_revision or next_selection_revision(output_root)
    target_path = output_root / f"target_selection_r{revision:04d}.json"
    if target_path.exists():
        raise FileExistsError(
            "Target selection revisions are immutable and cannot be overwritten."
        )
    representative = candidate["representative_observation"]
    selection_id = "target_selection_" + canonical_sha256(
        {
            "scene_id": candidates["scene_id"],
            "candidate_id": selected_candidate_id,
            "revision": revision,
            "candidate_manifest_sha256": canonical_sha256(candidates),
        }
    )[:20]
    result = {
        "schema_version": "kickclip.target_selection.v1",
        "selection_id": selection_id,
        "target_selection_revision": revision,
        "scene_id": candidates["scene_id"],
        "selected_candidate_id": selected_candidate_id,
        "selected_shot_id": candidate["shot_id"],
        "selected_frame_index": int(representative["frame_index"]),
        "selected_bbox_xyxy": representative["bbox_xyxy"],
        "selection_source": "USER_SELECTED_SCENE_WIDE_CANDIDATE",
        "reviewer": reviewer,
        "selected_at": now_iso(),
        "target_reference_set": [],
        "earlier_anchor_resolution": {"state": "NOT_RUN"},
        "production_manifest_sha256": production_manifest_sha256,
        "candidate_manifest_sha256": canonical_sha256(candidates),
        "automatic_target_selection": False,
    }
    atomic_json(target_path, result)
    atomic_json(output_root / "target_selection.json", result)
    return result


def create_target_reference_set(
    *,
    output_root: Path,
    video: Path,
    selection: dict[str, Any],
    maximum_references: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    candidates = read_json(output_root / "scene_candidates.json")
    candidate = _candidate(candidates, selection["selected_candidate_id"])
    if candidate["quality"]["identity_switch_risk"] != "LOW":
        raise ValueError(
            "Selected local tracklet has identity-switch risk and cannot seed references."
        )
    observations = list(candidate.get("observations") or [])
    if not observations:
        raise ValueError("Selected candidate has no local observations.")
    selected_indexes = sorted(
        {
            round(index * (len(observations) - 1) / max(1, maximum_references - 1))
            for index in range(maximum_references)
        }
    )
    root = output_root / "target_references" / selection["selection_id"]
    crops_root = root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(video)
    references = []
    images = []
    for number, index in enumerate(selected_indexes, start=1):
        observation = observations[index]
        frame_index = int(observation["frame_index"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = observation["bbox_xyxy"]
        left = max(0, int(x1))
        top = max(0, int(y1))
        right = min(width, int(x2))
        bottom = min(height, int(y2))
        crop = frame[top:bottom, left:right]
        if not crop.size:
            continue
        crop_path = crops_root / f"reference_{number:02d}.jpg"
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(crop_path)
        images.append(crop)
        references.append(
            {
                "reference_id": (
                    f"{selection['selection_id']}_reference_{number:02d}"
                ),
                "source_candidate_id": candidate["candidate_id"],
                "frame_index": frame_index,
                "bbox_xyxy": observation["bbox_xyxy"],
                "crop_artifact": portable(crop_path, output_root),
                "quality": {
                    "detector_confidence": observation["confidence"],
                    "identity_switch_risk": "LOW",
                },
                "embedding_artifact": None,
                "selection_reason": (
                    "TEMPORALLY_DIVERSE_LOCAL_TRACKLET_REFERENCE"
                ),
            }
        )
    capture.release()
    if not references:
        raise RuntimeError("No target reference crops could be decoded.")
    contact = root / "target_reference_contact_sheet.jpg"
    cell_width, cell_height = 240, 300
    sheet = np.full(
        (cell_height, cell_width * len(images), 3), 32, dtype=np.uint8
    )
    for index, image in enumerate(images):
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
    if not cv2.imwrite(str(contact), sheet):
        raise RuntimeError(contact)
    preview = root / "target_reference_preview.mp4"
    source_preview = output_root / candidate["artifacts"]["tracklet_review_video"]
    if source_preview.is_file():
        preview.write_bytes(source_preview.read_bytes())
    else:
        preview_writer = cv2.VideoWriter(
            str(preview),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (cell_width, cell_height),
        )
        for image in images:
            canvas = np.full((cell_height, cell_width, 3), 32, dtype=np.uint8)
            scale = min(
                cell_width / image.shape[1],
                cell_height / image.shape[0],
            )
            resized = cv2.resize(
                image,
                (
                    max(1, int(image.shape[1] * scale)),
                    max(1, int(image.shape[0] * scale)),
                ),
            )
            x = (cell_width - resized.shape[1]) // 2
            y = (cell_height - resized.shape[0]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            for _ in range(5):
                preview_writer.write(canvas)
        preview_writer.release()
    result = {
        "schema_version": "kickclip.target_reference_set.v1",
        "selection_id": selection["selection_id"],
        "scene_id": selection["scene_id"],
        "source_candidate_id": candidate["candidate_id"],
        "references": references,
        "artifacts": {
            "contact_sheet": portable(contact, output_root),
            "preview_video": portable(preview, output_root),
        },
        "provenance": {
            "source_scope": "SINGLE_CAMERA_CUT_FREE_LOCAL_TRACKLET",
            "identity_switch_risk_frames_excluded": True,
            "frozen_v2_memory_modified": False,
        },
    }
    path = output_root / "target_reference_set.json"
    atomic_json(path, result)
    selection = dict(selection)
    selection["target_reference_set"] = [
        row["reference_id"] for row in references
    ]
    atomic_json(output_root / "target_selection.json", selection)
    return result
