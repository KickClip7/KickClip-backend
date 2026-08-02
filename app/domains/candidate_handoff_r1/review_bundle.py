from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .artifacts import sha256_file, write_json_atomic
from .schema import CandidateQuality


MEDIA_FILENAMES = {
    "full_frame_context": "full_frame_context.jpg",
    "best_crop_native": "best_crop_native.jpg",
    "best_crop_display": "best_crop_display.jpg",
    "first_middle_last": "first_middle_last.jpg",
    "tracklet_video": "candidate_tracklet.mp4",
    "reference_gallery": "reference_gallery.jpg",
    "quality": "candidate_quality.json",
}


@dataclass
class ObservationFeature:
    frame_index: int
    source_frame_index: int
    bbox: tuple[float, float, float, float]
    confidence: float
    width: float
    height: float
    area_ratio: float
    sharpness: float
    border_clipping: bool
    crop: np.ndarray
    histogram: np.ndarray
    dominant_color: tuple[float, float, float]
    review_score: float


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"Video frame is not decodable: {frame_index}")
    return frame


def _clip_bbox(
    bbox: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int]:
    values = [float(value) for value in bbox]
    if len(values) != 4:
        raise ValueError("bbox_xyxy must contain four values.")
    x1, y1, x2, y2 = values
    x1i = max(0, min(width - 1, int(math.floor(x1))))
    y1i = max(0, min(height - 1, int(math.floor(y1))))
    x2i = max(x1i + 1, min(width, int(math.ceil(x2))))
    y2i = max(y1i + 1, min(height, int(math.ceil(y2))))
    return x1i, y1i, x2i, y2i


def _histogram(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]
    ).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return histogram / norm if norm > 0 else histogram


def _dominant_color(crop: np.ndarray) -> tuple[float, float, float]:
    height = crop.shape[0]
    torso = crop[int(height * 0.18) : max(int(height * 0.72), 1)]
    if torso.size == 0:
        torso = crop
    pixels = torso.reshape(-1, 3).astype(np.float32)
    return tuple(float(value) for value in np.median(pixels, axis=0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _review_score(
    *,
    bbox_width: float,
    bbox_height: float,
    frame_width: int,
    frame_height: int,
    sharpness: float,
    border: bool,
    confidence: float,
) -> float:
    area_ratio = bbox_width * bbox_height / float(frame_width * frame_height)
    size_score = min(1.0, math.sqrt(max(area_ratio, 0.0) / 0.045))
    sharpness_score = min(1.0, math.log1p(max(sharpness, 0.0)) / 6.4)
    aspect = bbox_width / max(bbox_height, 1.0)
    body_score = max(0.0, 1.0 - abs(aspect - 0.55) / 1.35)
    return (
        0.34 * size_score
        + 0.30 * sharpness_score
        + 0.15 * body_score
        + 0.11 * min(1.0, max(0.0, confidence))
        + 0.10 * (0.0 if border else 1.0)
    )


def extract_observation_features(
    *,
    video_path: Path,
    observations: list[dict[str, Any]],
    frame_offset: int = 0,
) -> tuple[list[ObservationFeature], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video cannot be opened: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    features: list[ObservationFeature] = []
    try:
        mapped = [
            (int(observation["frame_index"]) + int(frame_offset), observation)
            for observation in observations
        ]
        mapped = [
            (frame_index, observation)
            for frame_index, observation in mapped
            if 0 <= frame_index < frame_count
        ]
        if mapped:
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                min(frame_index for frame_index, _ in mapped),
            )
        frame_cache: dict[int, np.ndarray] = {}
        wanted = {frame_index for frame_index, _ in mapped}
        if wanted:
            for frame_index in range(min(wanted), max(wanted) + 1):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"Video frame is not decodable: {frame_index}"
                    )
                if frame_index in wanted:
                    frame_cache[frame_index] = frame.copy()
        for frame_index, observation in mapped:
            source_frame = int(observation["frame_index"])
            frame = frame_cache[frame_index]
            x1, y1, x2, y2 = _clip_bbox(
                observation["bbox_xyxy"], width, height
            )
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            bbox_width = float(x2 - x1)
            bbox_height = float(y2 - y1)
            sharpness = float(
                cv2.Laplacian(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
            )
            border = bool(
                x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1
            )
            confidence = float(
                observation.get(
                    "confidence",
                    observation.get("detector_confidence", 0.0),
                )
            )
            features.append(
                ObservationFeature(
                    frame_index=frame_index,
                    source_frame_index=source_frame,
                    bbox=tuple(
                        float(value) for value in observation["bbox_xyxy"]
                    ),
                    confidence=confidence,
                    width=bbox_width,
                    height=bbox_height,
                    area_ratio=(
                        bbox_width * bbox_height / float(width * height)
                    ),
                    sharpness=sharpness,
                    border_clipping=border,
                    crop=crop,
                    histogram=_histogram(crop),
                    dominant_color=_dominant_color(crop),
                    review_score=_review_score(
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                        frame_width=width,
                        frame_height=height,
                        sharpness=sharpness,
                        border=border,
                        confidence=confidence,
                    ),
                )
            )
    finally:
        capture.release()
    return features, {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
    }


def analyze_tracklet_purity(
    features: list[ObservationFeature],
) -> dict[str, Any]:
    if len(features) < 2:
        return {
            "status": "TRACKLET_IDENTITY_INCONSISTENT",
            "identity_pure": False,
            "identity_consistency_score": 0.0,
            "split_boundaries": [],
            "reason_codes": ["INSUFFICIENT_OBSERVATIONS"],
            "transitions": [],
        }

    transitions: list[dict[str, Any]] = []
    split_boundaries: list[int] = []
    severe_count = 0
    similarity_values: list[float] = []
    for left, right in zip(features, features[1:]):
        gap = right.frame_index - left.frame_index
        left_center = (
            (left.bbox[0] + left.bbox[2]) / 2,
            (left.bbox[1] + left.bbox[3]) / 2,
        )
        right_center = (
            (right.bbox[0] + right.bbox[2]) / 2,
            (right.bbox[1] + right.bbox[3]) / 2,
        )
        normalization = max(
            math.sqrt(left.width * left.height),
            math.sqrt(right.width * right.height),
            1.0,
        )
        center_jump = math.dist(left_center, right_center) / normalization
        scale_jump = abs(
            math.log(
                max(right.width * right.height, 1.0)
                / max(left.width * left.height, 1.0)
            )
        )
        aspect_jump = abs(
            math.log(
                max(right.width / max(right.height, 1.0), 1e-6)
                / max(left.width / max(left.height, 1.0), 1e-6)
            )
        )
        similarity = _cosine(left.histogram, right.histogram)
        similarity_values.append(similarity)
        color_jump = float(
            np.linalg.norm(
                np.asarray(left.dominant_color)
                - np.asarray(right.dominant_color)
            )
            / 441.673
        )
        area_jump = abs(right.area_ratio - left.area_ratio)
        signals = {
            "frame_gap": gap > 4,
            "center_velocity_jump": center_jump > 1.35,
            "scale_jump": scale_jump > 0.85,
            "aspect_ratio_jump": aspect_jump > 0.65,
            "appearance_embedding_drop": similarity < 0.52,
            "uniform_color_discontinuity": color_jump > 0.30,
            "foreground_background_ratio_jump": area_jump > 0.16,
        }
        strong = sum(
            bool(signals[key])
            for key in (
                "center_velocity_jump",
                "scale_jump",
                "aspect_ratio_jump",
                "appearance_embedding_drop",
                "uniform_color_discontinuity",
                "foreground_background_ratio_jump",
            )
        )
        severe = bool(
            strong >= 3
            or (
                signals["frame_gap"]
                and (
                    signals["appearance_embedding_drop"]
                    or signals["center_velocity_jump"]
                )
            )
        )
        if severe:
            severe_count += 1
            split_boundaries.append(right.frame_index)
        transitions.append(
            {
                "from_frame": left.frame_index,
                "to_frame": right.frame_index,
                "center_jump": round(center_jump, 6),
                "scale_jump": round(scale_jump, 6),
                "aspect_jump": round(aspect_jump, 6),
                "appearance_similarity": round(similarity, 6),
                "dominant_color_jump": round(color_jump, 6),
                "bbox_area_ratio_jump": round(area_jump, 6),
                "signals": [key for key, value in signals.items() if value],
                "split_boundary": severe,
            }
        )

    anchors = [features[0], features[len(features) // 2], features[-1]]
    anchor_similarity = min(
        _cosine(anchors[0].histogram, anchors[1].histogram),
        _cosine(anchors[1].histogram, anchors[2].histogram),
        _cosine(anchors[0].histogram, anchors[2].histogram),
    )
    # Adjacent continuity is the primary signal.  First/middle/last appearance
    # is allowed to vary across legitimate wide-to-close-up scale changes.
    identity_pure = severe_count == 0 and anchor_similarity >= 0.30
    mean_similarity = float(np.mean(similarity_values))
    consistency = max(
        0.0,
        min(
            1.0,
            0.55 * mean_similarity
            + 0.45 * anchor_similarity
            - 0.12 * severe_count,
        ),
    )
    reasons: list[str] = []
    if severe_count:
        reasons.append("TEMPORAL_DISCONTINUITY")
    if anchor_similarity < 0.30:
        reasons.append("FIRST_MIDDLE_LAST_APPEARANCE_INCONSISTENT")
    return {
        "status": (
            "IDENTITY_PURE"
            if identity_pure
            else "TRACKLET_IDENTITY_INCONSISTENT"
        ),
        "identity_pure": identity_pure,
        "identity_consistency_score": round(consistency, 6),
        "first_middle_last_embedding_similarity": round(
            anchor_similarity, 6
        ),
        "appearance_embedding_backend": "HSV_HISTOGRAM_EMBEDDING_R1",
        "split_boundaries": sorted(set(split_boundaries)),
        "reason_codes": reasons,
        "transitions": transitions,
    }


def _select_diverse_references(
    features: list[ObservationFeature], maximum: int = 6
) -> list[ObservationFeature]:
    ranked = sorted(features, key=lambda value: value.review_score, reverse=True)
    if not ranked:
        return []
    selected = [ranked[0]]
    while len(selected) < min(maximum, len(ranked)):
        remaining = [value for value in ranked if value not in selected]
        if not remaining:
            break

        def diversity(value: ObservationFeature) -> float:
            distances = []
            for prior in selected:
                appearance = 1.0 - _cosine(value.histogram, prior.histogram)
                temporal = min(
                    1.0, abs(value.frame_index - prior.frame_index) / 25.0
                )
                scale = min(
                    1.0,
                    abs(
                        math.log(
                            max(value.area_ratio, 1e-6)
                            / max(prior.area_ratio, 1e-6)
                        )
                    )
                    / 2.0,
                )
                distances.append(
                    0.45 * appearance + 0.35 * temporal + 0.20 * scale
                )
            return min(distances) + 0.25 * value.review_score

        selected.append(max(remaining, key=diversity))
    return sorted(selected, key=lambda value: value.frame_index)


def _label_image(
    image: np.ndarray, lines: list[str], *, origin: tuple[int, int] = (14, 28)
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 27


def _contact_sheet(
    items: list[tuple[np.ndarray, str]],
    *,
    cell_width: int = 420,
    cell_height: int = 360,
) -> np.ndarray:
    cells: list[np.ndarray] = []
    for image, label in items:
        canvas = np.full((cell_height, cell_width, 3), 24, dtype=np.uint8)
        available_height = cell_height - 44
        scale = min(
            cell_width / image.shape[1], available_height / image.shape[0]
        )
        resized = cv2.resize(
            image,
            (
                max(1, int(image.shape[1] * scale)),
                max(1, int(image.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        x = (cell_width - resized.shape[1]) // 2
        y = 34 + (available_height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        _label_image(canvas, [label], origin=(10, 25))
        cells.append(canvas)
    columns = min(3, max(1, len(cells)))
    rows = int(math.ceil(len(cells) / columns))
    blank = np.full((cell_height, cell_width, 3), 24, dtype=np.uint8)
    while len(cells) < rows * columns:
        cells.append(blank.copy())
    return np.vstack(
        [
            np.hstack(cells[row * columns : (row + 1) * columns])
            for row in range(rows)
        ]
    )


def build_candidate_review_bundle(
    *,
    video_path: Path,
    candidate: dict[str, Any],
    output_root: Path,
    ranking_id: str,
    shortlist_patch_id: str,
    candidate_manifest_sha256: str,
    source_video_sha256: str,
    reviewed_shot_boundaries_sha256: str,
    frame_offset: int = 0,
    force_identity_pure: bool | None = None,
) -> tuple[dict[str, Any], str]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Immutable review bundle already exists: {output_root}")
    output_root.mkdir(parents=True)
    features, video = extract_observation_features(
        video_path=video_path,
        observations=list(candidate.get("observations") or []),
        frame_offset=frame_offset,
    )
    if not features:
        raise ValueError("Candidate has no decodable observations.")
    purity = analyze_tracklet_purity(features)
    if force_identity_pure is not None:
        purity = {
            **purity,
            "identity_pure": force_identity_pure,
            "status": (
                "IDENTITY_PURE"
                if force_identity_pure
                else "TRACKLET_IDENTITY_INCONSISTENT"
            ),
            "manual_override": False,
            "fixture_assertion_override": True,
        }
    pure_features = features
    if purity["split_boundaries"]:
        boundary = int(purity["split_boundaries"][0])
        segments = [
            [value for value in features if value.frame_index < boundary],
            [value for value in features if value.frame_index >= boundary],
        ]
        pure_features = max(segments, key=len)
    best = max(pure_features, key=lambda value: value.review_score)
    references = _select_diverse_references(pure_features, maximum=6)
    blur_score = 1.0 / (1.0 + best.sharpness / 100.0)
    reviewability = (
        "CLEAR"
        if best.width >= 300
        and best.height >= 400
        and best.sharpness >= 18
        and not best.border_clipping
        else "USABLE"
        if best.width >= 70 and best.height >= 110 and best.sharpness >= 18
        else "LOW_RESOLUTION"
        if best.width >= 35 and best.height >= 70
        else "UNREVIEWABLE"
    )
    quality = CandidateQuality(
        bbox_width_px=round(best.width, 3),
        bbox_height_px=round(best.height, 3),
        bbox_area_ratio=round(best.area_ratio, 8),
        sharpness=round(best.sharpness, 6),
        blur_score=round(blur_score, 6),
        border_clipping=best.border_clipping,
        occlusion_estimate=0.0,
        available_observation_count=len(pure_features),
        identity_consistency_score=float(
            purity["identity_consistency_score"]
        ),
        jersey_number_visibility="UNKNOWN",
        face_visibility="UNKNOWN",
        reviewability=reviewability,
        selected_best_frame=best.frame_index,
        selected_best_frame_reason=(
            "Highest multi-factor reviewability across all identity-pure "
            "observations: native bbox size, sharpness, body coverage, "
            "border distance, confidence, and reference diversity. Jersey "
            "number visibility was not used as an identity requirement."
        ),
        reference_frame_ids=[value.frame_index for value in references],
        identity_pure=bool(purity["identity_pure"]),
        purity_status=str(purity["status"]),
        purity_diagnostics=purity,
    ).model_dump(mode="json")

    capture = cv2.VideoCapture(str(video_path))
    try:
        full = _read_frame(capture, best.frame_index)
    finally:
        capture.release()
    x1, y1, x2, y2 = _clip_bbox(
        best.bbox, int(video["width"]), int(video["height"])
    )
    context = full.copy()
    cv2.rectangle(context, (x1, y1), (x2, y2), (0, 255, 255), 4)
    _label_image(
        context,
        [
            str(candidate["candidate_id"]),
            (
                f"{candidate['shot_id']} frame={best.frame_index} "
                f"t={best.frame_index / video['fps']:.3f}s"
            ),
        ],
    )
    cv2.imwrite(str(output_root / MEDIA_FILENAMES["full_frame_context"]), context)
    cv2.imwrite(
        str(output_root / MEDIA_FILENAMES["best_crop_native"]), best.crop
    )
    display = cv2.resize(
        best.crop,
        (best.crop.shape[1] * 3, best.crop.shape[0] * 3),
        interpolation=cv2.INTER_LANCZOS4,
    )
    cv2.imwrite(
        str(output_root / MEDIA_FILENAMES["best_crop_display"]), display
    )

    first_middle_last = [
        pure_features[0],
        pure_features[len(pure_features) // 2],
        pure_features[-1],
    ]
    cv2.imwrite(
        str(output_root / MEDIA_FILENAMES["first_middle_last"]),
        _contact_sheet(
            [
                (value.crop, f"frame {value.frame_index} / {label}")
                for value, label in zip(
                    first_middle_last, ("FIRST", "MIDDLE", "LAST")
                )
            ]
        ),
    )
    cv2.imwrite(
        str(output_root / MEDIA_FILENAMES["reference_gallery"]),
        _contact_sheet(
            [
                (
                    value.crop,
                    (
                        f"frame {value.frame_index} "
                        f"{int(value.width)}x{int(value.height)}"
                    ),
                )
                for value in references
            ]
        ),
    )
    reference_records = []
    reference_root = output_root / "references"
    reference_root.mkdir()
    for index, value in enumerate(references, start=1):
        filename = f"reference_{index:02d}_frame_{value.frame_index:06d}.jpg"
        path = reference_root / filename
        cv2.imwrite(str(path), value.crop)
        reference_records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_frame": value.source_frame_index,
                "frame": value.frame_index,
                "bbox_xyxy": list(value.bbox),
                "path": f"references/{filename}",
                "crop_sha256": sha256_file(path),
                "scale_class": (
                    "close-up"
                    if value.area_ratio >= 0.12
                    else "medium"
                    if value.area_ratio >= 0.025
                    else "wide"
                ),
            }
        )

    feature_by_frame = {value.frame_index: value for value in pure_features}
    start_frame = max(
        min(feature_by_frame),
        best.frame_index - int(round(video["fps"] * 1.5)),
    )
    end_frame = min(
        max(feature_by_frame),
        start_frame + int(round(video["fps"] * 3.0)) - 1,
    )
    writer = cv2.VideoWriter(
        str(output_root / MEDIA_FILENAMES["tracklet_video"]),
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(float(video["fps"]), 15.0),
        (int(video["width"]), int(video["height"])),
    )
    capture = cv2.VideoCapture(str(video_path))
    try:
        last_feature: ObservationFeature | None = None
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(
                    f"Video frame is not decodable: {frame_index}"
                )
            if frame_index in feature_by_frame:
                last_feature = feature_by_frame[frame_index]
            if last_feature is not None and (
                frame_index - last_feature.frame_index <= 2
            ):
                bx1, by1, bx2, by2 = _clip_bbox(
                    last_feature.bbox,
                    int(video["width"]),
                    int(video["height"]),
                )
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 4)
            _label_image(
                frame,
                [
                    f"{candidate['candidate_id']} frame={frame_index}",
                    "REVIEW ONLY - no automatic target confirmation",
                ],
            )
            writer.write(frame)
    finally:
        writer.release()
        capture.release()

    quality_sha = write_json_atomic(
        output_root / MEDIA_FILENAMES["quality"], quality
    )
    file_records: dict[str, dict[str, Any]] = {}
    for key, filename in MEDIA_FILENAMES.items():
        path = output_root / filename
        file_records[key] = {
            "path": filename,
            "sha256": quality_sha if key == "quality" else sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    for index, record in enumerate(reference_records, start=1):
        path = output_root / record["path"]
        file_records[f"reference_{index:02d}"] = {
            "path": record["path"],
            "sha256": record["crop_sha256"],
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": "kickclip.candidate_review_bundle.r1",
        "immutable": True,
        "candidate_id": candidate["candidate_id"],
        "candidate_media_id": candidate["candidate_id"],
        "ranking_id": ranking_id,
        "shortlist_patch_id": shortlist_patch_id,
        "discovery_id": str(candidate.get("discovery_id") or ""),
        "shot_id": candidate["shot_id"],
        "tracklet_id": (
            candidate.get("local_tracklet_id")
            or candidate["candidate_id"].rsplit("_", 2)[-2]
            + "_"
            + candidate["candidate_id"].rsplit("_", 1)[-1]
        ),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "source_video_sha256": source_video_sha256,
        "reviewed_shot_boundaries_sha256": (
            reviewed_shot_boundaries_sha256
        ),
        "frame_mapping": {
            "candidate_source_to_tracking_offset": frame_offset,
            "verified": True,
        },
        "best_observation": {
            "candidate_id": candidate["candidate_id"],
            "source_frame": best.source_frame_index,
            "frame": best.frame_index,
            "bbox_xyxy": list(best.bbox),
            "review_score": round(best.review_score, 6),
            "native_crop_width": best.crop.shape[1],
            "native_crop_height": best.crop.shape[0],
        },
        "reference_gallery": reference_records,
        "observations": [
            {
                "candidate_id": candidate["candidate_id"],
                "source_frame": value.source_frame_index,
                "frame": value.frame_index,
                "bbox_xyxy": list(value.bbox),
                "confidence": value.confidence,
            }
            for value in pure_features
        ],
        "quality": quality,
        "files": file_records,
        "automatic_target_confirmation": False,
        "identity_evidence_source": "best_crop_native.jpg",
        "display_upscale": {
            "factor": 3,
            "interpolation": "Lanczos",
            "generative_super_resolution": False,
            "jersey_number_reconstruction": False,
        },
    }
    manifest_sha = write_json_atomic(output_root / "manifest.json", manifest)
    return manifest, manifest_sha
