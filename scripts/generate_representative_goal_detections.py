from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2

from app.domains.highlight.player_detector import RFDETRPlayerDetector


FIELDS = (
    "frame_index",
    "time_ms",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.25)
    arguments = parser.parse_args()

    video = arguments.video.resolve()
    checkpoint = arguments.checkpoint.resolve()
    output = arguments.output.resolve()
    metadata_path = arguments.metadata.resolve()
    if output.exists() or metadata_path.exists():
        raise RuntimeError("Detection output is immutable and already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open scene video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detector = RFDETRPlayerDetector(
        checkpoint_path=checkpoint,
        requested_device="auto",
        confidence_threshold=arguments.threshold,
        batch_size=arguments.batch_size,
    )

    frame_index = 0
    detection_count = 0
    class_counts: dict[str, int] = {}
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            while frame_index < expected_frames:
                frames = []
                indexes = []
                for _ in range(arguments.batch_size):
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    frames.append(frame)
                    indexes.append(frame_index)
                    frame_index += 1
                if not frames:
                    break
                detections_by_frame = detector.detect_batch(frames)
                for index, detections in zip(indexes, detections_by_frame):
                    for detection_index, detection in enumerate(detections):
                        x1, y1, x2, y2 = detection.bbox_xyxy
                        x1 = min(max(float(x1), 0.0), width - 1.0)
                        y1 = min(max(float(y1), 0.0), height - 1.0)
                        x2 = min(max(float(x2), x1 + 1e-6), width - 1.0)
                        y2 = min(max(float(y2), y1 + 1e-6), height - 1.0)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        writer.writerow(
                            {
                                "frame_index": index,
                                "time_ms": int(round(index * 1000.0 / fps)),
                                "detection_index": detection_index,
                                "detection_id": (
                                    f"frame_{index:06d}_det_"
                                    f"{detection_index:04d}"
                                ),
                                "class_id": detection.class_id,
                                "class_name": detection.class_name,
                                "confidence": f"{detection.confidence:.9f}",
                                "x1": f"{x1:.6f}",
                                "y1": f"{y1:.6f}",
                                "x2": f"{x2:.6f}",
                                "y2": f"{y2:.6f}",
                            }
                        )
                        detection_count += 1
                        class_counts[detection.class_name] = (
                            class_counts.get(detection.class_name, 0) + 1
                        )
                if frame_index % 60 < arguments.batch_size:
                    print(
                        json.dumps(
                            {
                                "decoded_frames": frame_index,
                                "expected_frames": expected_frames,
                                "detections": detection_count,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        if frame_index != expected_frames:
            raise RuntimeError(
                f"Scene decode ended early: {frame_index}/{expected_frames}"
            )
        os.replace(temporary, output)
    finally:
        capture.release()
        temporary.unlink(missing_ok=True)

    metadata = {
        "schema_version": "kickclip.frozen_scene_detections.v1",
        "source_video": str(video),
        "source_video_sha256": sha256_file(video),
        "checkpoint_sha256": sha256_file(checkpoint),
        "detections_sha256": sha256_file(output),
        "fps": fps,
        "frame_count": frame_index,
        "width": width,
        "height": height,
        "detection_count": detection_count,
        "class_counts": class_counts,
        "detector_runtime": detector.runtime_metadata,
        "automatic_target_selection": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
