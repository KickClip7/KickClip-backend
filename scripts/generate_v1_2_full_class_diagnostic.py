from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from app.domains.highlight.player_detector import (
    EXPECTED_CLASS_NAMES,
    RFDETRPlayerDetector,
)


CSV_FIELDS = (
    "frame_index",
    "scene_local_time_sec",
    "detection_index",
    "detection_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(first: list[float], second: list[float]) -> float:
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


def read_frames(
    video: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> tuple[list[tuple[int, Any]], float, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open scene video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {frame_index}.")
        frames.append((frame_index, frame))
    capture.release()
    return frames, fps, width, height


def raw_full_class_predictions(
    detector: RFDETRPlayerDetector,
    frames: list[tuple[int, Any]],
    *,
    fps: float,
    width: int,
    height: int,
    batch_size: int,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        rgb = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _, frame in chunk
        ]
        with detector._torch.inference_mode():
            raw = detector._loaded.model.predict(rgb, threshold=threshold)
        results = list(raw) if isinstance(raw, list) else [raw]
        if len(results) != len(chunk):
            raise RuntimeError("RF-DETR full-class output count mismatch.")
        for (frame_index, _), result in zip(chunk, results):
            detections = []
            for raw_box, raw_class, raw_confidence in zip(
                getattr(result, "xyxy", []),
                getattr(result, "class_id", []),
                getattr(result, "confidence", []),
            ):
                class_id = int(raw_class)
                confidence = float(raw_confidence)
                if (
                    class_id < 0
                    or class_id >= len(EXPECTED_CLASS_NAMES)
                    or confidence < threshold
                ):
                    continue
                values = [float(value) for value in raw_box]
                if len(values) != 4 or any(
                    not math.isfinite(value) for value in values
                ):
                    continue
                x1 = min(max(values[0], 0.0), width - 1.0)
                y1 = min(max(values[1], 0.0), height - 1.0)
                x2 = min(max(values[2], x1 + 1e-6), width - 1.0)
                y2 = min(max(values[3], y1 + 1e-6), height - 1.0)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "frame_index": frame_index,
                        "scene_local_time_sec": frame_index / fps,
                        "class_id": class_id,
                        "class_name": EXPECTED_CLASS_NAMES[class_id],
                        "confidence": confidence,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )
            detections.sort(
                key=lambda row: (
                    -row["confidence"],
                    row["class_id"],
                    row["bbox_xyxy"],
                )
            )
            for detection_index, row in enumerate(detections):
                rows.append(
                    {
                        **row,
                        "detection_index": detection_index,
                        "detection_id": (
                            f"fullclass_frame_{frame_index:06d}_"
                            f"det_{detection_index:04d}"
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "processed_frames": min(start + batch_size, len(frames)),
                    "total_frames": len(frames),
                    "detections": len(rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows


def top5_class_audit(
    *,
    scene_candidates: dict[str, Any],
    ranking: dict[str, Any],
    detections: list[dict[str, Any]],
    frame_start: int,
    frame_end: int,
    minimum_iou: float,
) -> list[dict[str, Any]]:
    candidates = {
        row["candidate_id"]: row
        for row in scene_candidates["candidates"]
    }
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        by_frame.setdefault(detection["frame_index"], []).append(detection)
    audits = []
    for shortlist_row in ranking["shortlist"]:
        candidate_id = shortlist_row["candidate_id"]
        candidate = candidates[candidate_id]
        matches = []
        class_counts: Counter[str] = Counter()
        for observation in candidate["observations"]:
            frame = int(observation["frame_index"])
            if frame < frame_start or frame > frame_end:
                continue
            box = [float(value) for value in observation["bbox_xyxy"]]
            best = max(
                by_frame.get(frame, []),
                key=lambda row: iou(box, row["bbox_xyxy"]),
                default=None,
            )
            overlap = iou(box, best["bbox_xyxy"]) if best else 0.0
            matched = best is not None and overlap >= minimum_iou
            class_name = best["class_name"] if matched else None
            if class_name is not None:
                class_counts[class_name] += 1
            matches.append(
                {
                    "frame_index": frame,
                    "candidate_bbox_xyxy": box,
                    "matched": matched,
                    "matched_iou": overlap if best is not None else None,
                    "matched_detection_id": (
                        best["detection_id"] if matched else None
                    ),
                    "matched_class_id": (
                        best["class_id"] if matched else None
                    ),
                    "matched_class_name": class_name,
                    "matched_confidence": (
                        best["confidence"] if matched else None
                    ),
                }
            )
        majority = (
            class_counts.most_common(1)[0][0] if class_counts else None
        )
        audits.append(
            {
                "candidate_id": candidate_id,
                "original_shortlist_rank": shortlist_row["rank"],
                "observation_count_in_window": len(matches),
                "matched_observation_count": sum(
                    bool(row["matched"]) for row in matches
                ),
                "matched_class_counts": dict(sorted(class_counts.items())),
                "majority_matched_class": majority,
                "staff_classified": class_counts.get("staff", 0) > 0,
                "matches": matches,
            }
        )
    return audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-candidates", type=Path, required=True)
    parser.add_argument("--ranking-report", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=303)
    parser.add_argument("--frame-end", type=int, default=398)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--match-iou", type=float, default=0.50)
    arguments = parser.parse_args()
    output_csv = arguments.output_csv.resolve()
    output_json = arguments.output_json.resolve()
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("Diagnostic artifact already exists.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(".tmp.csv")

    video = arguments.video.resolve()
    checkpoint = arguments.checkpoint.resolve()
    scene_candidates = json.loads(
        arguments.scene_candidates.read_text(encoding="utf-8")
    )
    report = json.loads(
        arguments.ranking_report.read_text(encoding="utf-8")
    )
    ranking = report["ranking"]
    frames, fps, width, height = read_frames(
        video,
        start_frame=arguments.frame_start,
        end_frame=arguments.frame_end,
    )
    detector = RFDETRPlayerDetector(
        checkpoint_path=checkpoint,
        requested_device="auto",
        confidence_threshold=arguments.threshold,
        batch_size=arguments.batch_size,
    )
    detections = raw_full_class_predictions(
        detector,
        frames,
        fps=fps,
        width=width,
        height=height,
        batch_size=arguments.batch_size,
        threshold=arguments.threshold,
    )
    try:
        with temporary_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in detections:
                x1, y1, x2, y2 = row["bbox_xyxy"]
                writer.writerow(
                    {
                        "frame_index": row["frame_index"],
                        "scene_local_time_sec": (
                            f"{row['scene_local_time_sec']:.9f}"
                        ),
                        "detection_index": row["detection_index"],
                        "detection_id": row["detection_id"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "confidence": f"{row['confidence']:.9f}",
                        "x1": f"{x1:.6f}",
                        "y1": f"{y1:.6f}",
                        "x2": f"{x2:.6f}",
                        "y2": f"{y2:.6f}",
                    }
                )
        os.replace(temporary_csv, output_csv)
    finally:
        temporary_csv.unlink(missing_ok=True)

    class_counts = Counter(row["class_name"] for row in detections)
    audit = top5_class_audit(
        scene_candidates=scene_candidates,
        ranking=ranking,
        detections=detections,
        frame_start=arguments.frame_start,
        frame_end=arguments.frame_end,
        minimum_iou=arguments.match_iou,
    )
    document = {
        "schema_version": "kickclip.rfdetr_full_class_diagnostic.v1",
        "status": "DIAGNOSTIC_ONLY",
        "frame_range": {
            "start_frame": arguments.frame_start,
            "end_frame_inclusive": arguments.frame_end,
            "frame_count": len(frames),
        },
        "video": {
            "sha256": sha256_file(video),
            "fps": fps,
            "width": width,
            "height": height,
        },
        "checkpoint": {
            "sha256": sha256_file(checkpoint),
            "class_mapping": {
                str(index): name
                for index, name in enumerate(EXPECTED_CLASS_NAMES)
            },
            "threshold": arguments.threshold,
            "runtime": detector.runtime_metadata,
        },
        "detections": {
            "csv_relative_path": output_csv.name,
            "csv_sha256": sha256_file(output_csv),
            "total_count": len(detections),
            "class_counts": {
                name: class_counts.get(name, 0)
                for name in EXPECTED_CLASS_NAMES
            },
            "frame_count_with_detection": len(
                {row["frame_index"] for row in detections}
            ),
        },
        "top5_box_full_class_audit": audit,
        "score_input": False,
        "shortlist_input": False,
        "human_labels_used": False,
        "automatic_target_confirmation": False,
    }
    output_json.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(document, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
