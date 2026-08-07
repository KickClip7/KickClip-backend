# global_ID_linker/stage1_rfdetr_hybrid_track_video.py
"""
Stage 1-D / 1-E: RF-DETR hybrid tracking baseline

출력:
  - detections.jsonl
  - shot_modes.jsonl
  - closeup_crops/*.jpg
  - closeup_crops.jsonl
  - local_tracks.jsonl
  - summary.json
  - video_preview.mp4

핵심:
  - RF-DETR로 모든 프레임 detection
  - tuned1 기준 scene mode 분류
  - closeup_observation 프레임에서는 crop top-K 저장
  - play_tracking 프레임에서만 player/goalkeeper detection을 tracker에 입력
  - closeup_observation / non_player_or_crowd 프레임에서는 tracker 입력 금지

현재 tracker:
  - SimpleIoUTracker baseline
  - 나중에 BoT-SORT / ByteTrack으로 교체 가능
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np


CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "staff",
    4: "ball",
}

PLAYER_GK_CLASS_IDS = {0, 1}
PERSON_CLASS_IDS = {0, 1, 2, 3}

CLASS_COLORS_BGR = {
    0: (0, 255, 0),
    1: (255, 0, 255),
    2: (0, 255, 255),
    3: (255, 255, 0),
    4: (0, 128, 255),
}

MODE_COLORS_BGR = {
    "play_tracking": (80, 220, 80),
    "closeup_observation": (80, 180, 255),
    "non_player_or_crowd": (180, 180, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1-D/E: RF-DETR detection + scene mode + crops + local tracks"
    )

    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)

    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--repo-root", default=None, type=str)
    parser.add_argument("--model-size", default="small", choices=["small", "base"])
    parser.add_argument(
        "--class-id-offset",
        default=0,
        type=int,
        help="Use -1 if RF-DETR returns 1-based class ids.",
    )

    parser.add_argument("--max-frames", default=0, type=int)
    parser.add_argument("--preview-name", default="video_preview.mp4", type=str)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-every", default=50, type=int)

    # tuned1 scene mode defaults
    parser.add_argument("--play-min-player-gk", default=3, type=int)
    parser.add_argument("--play-max-box-height-ratio", default=0.55, type=float)
    parser.add_argument("--play-mean-box-height-ratio", default=0.32, type=float)

    parser.add_argument("--closeup-min-box-height-ratio", default=0.35, type=float)
    parser.add_argument("--closeup-min-box-area-ratio", default=0.10, type=float)
    parser.add_argument("--closeup-max-persons", default=5, type=int)

    parser.add_argument("--crowd-min-persons", default=10, type=int)
    parser.add_argument("--crowd-max-mean-height-ratio", default=0.18, type=float)
    parser.add_argument("--crowd-max-mean-center-y-ratio", default=0.45, type=float)
    parser.add_argument("--crowd-max-y2-ratio", default=0.60, type=float)

    # close-up crop settings
    parser.add_argument("--top-k-closeup", default=3, type=int)
    parser.add_argument("--crop-classes", default="0,1", type=str)
    parser.add_argument("--min-crop-conf", default=0.25, type=float)
    parser.add_argument("--min-crop-box-height-ratio", default=0.20, type=float)
    parser.add_argument("--min-crop-box-area-ratio", default=0.03, type=float)
    parser.add_argument("--crop-pad-ratio", default=0.08, type=float)
    parser.add_argument("--no-save-closeup-crops", action="store_true")

    # local tracker settings
    parser.add_argument("--tracker-type", default="simple_iou", choices=["simple_iou"])
    parser.add_argument("--track-classes", default="0,1", type=str)
    parser.add_argument("--track-iou-thresh", default=0.35, type=float)
    parser.add_argument("--track-max-age", default=12, type=int)
    parser.add_argument("--min-track-conf", default=0.25, type=float)
    parser.add_argument("--min-track-box-height-ratio", default=0.025, type=float)
    parser.add_argument("--min-track-box-area-ratio", default=0.00015, type=float)
    parser.add_argument(
        "--keep-tracker-through-non-play",
        action="store_true",
        help="If set, do not reset tracker when mode is not play_tracking.",
    )

    return parser.parse_args()


def resolve_project_root(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parents[1]


def add_import_paths(project_root: Path) -> None:
    candidates = [
        project_root,
        project_root / "v1",
        project_root / "v1" / "rfdetr",
        project_root / "global_ID_linker",
    ]
    for p in candidates:
        if p.exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_class_ids(text: str) -> Set[int]:
    result: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def first_existing_key(d: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def clamp_xyxy(
    xyxy: Iterable[float],
    width: int,
    height: int,
) -> Optional[List[float]]:
    vals = list(xyxy)
    if len(vals) != 4:
        return None

    x1, y1, x2, y2 = [safe_float(v) for v in vals]

    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def pad_and_clamp_xyxy(
    xyxy: List[float],
    width: int,
    height: int,
    pad_ratio: float,
) -> Optional[List[int]]:
    x1, y1, x2, y2 = xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    padded = [
        x1 - box_w * pad_ratio,
        y1 - box_h * pad_ratio,
        x2 + box_w * pad_ratio,
        y2 + box_h * pad_ratio,
    ]
    clamped = clamp_xyxy(padded, width, height)
    if clamped is None:
        return None
    return [int(round(v)) for v in clamped]


def xyxy_to_xywh(xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [
        round(x1, 2),
        round(y1, 2),
        round(x2 - x1, 2),
        round(y2 - y1, 2),
    ]


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return float(inter_area / union)


def box_ratios(
    xyxy: List[float],
    frame_width: int,
    frame_height: int,
) -> Dict[str, float]:
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h

    return {
        "box_width_ratio": round(w / max(frame_width, 1), 6),
        "box_height_ratio": round(h / max(frame_height, 1), 6),
        "box_area_ratio": round(area / max(frame_width * frame_height, 1), 6),
        "center_x_ratio": round(((x1 + x2) / 2.0) / max(frame_width, 1), 6),
        "center_y_ratio": round(((y1 + y2) / 2.0) / max(frame_height, 1), 6),
        "y2_ratio": round(y2 / max(frame_height, 1), 6),
    }


def normalize_sv_detections(
    detections: Any,
    frame_width: int,
    frame_height: int,
    conf_thresh: float,
    class_id_offset: int = 0,
) -> List[Dict[str, Any]]:
    xyxy_arr = to_numpy(getattr(detections, "xyxy", None))
    conf_arr = to_numpy(getattr(detections, "confidence", None))
    cls_arr = to_numpy(getattr(detections, "class_id", None))

    if xyxy_arr.size == 0:
        return []

    if conf_arr.size == 0:
        conf_arr = np.ones((len(xyxy_arr),), dtype=np.float32)
    if cls_arr.size == 0:
        cls_arr = np.full((len(xyxy_arr),), -1, dtype=np.int32)

    results: List[Dict[str, Any]] = []
    for i in range(len(xyxy_arr)):
        conf = safe_float(conf_arr[i])
        if conf < conf_thresh:
            continue

        class_id = safe_int(cls_arr[i]) + class_id_offset
        xyxy = clamp_xyxy(xyxy_arr[i], frame_width, frame_height)
        if xyxy is None:
            continue

        results.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "conf": round(conf, 6),
                "xyxy": xyxy,
                "xywh": xyxy_to_xywh(xyxy),
            }
        )

    results.sort(key=lambda d: d["conf"], reverse=True)
    return results


def normalize_dict_detections(
    detections: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    conf_thresh: float,
    class_id_offset: int = 0,
) -> List[Dict[str, Any]]:
    xyxy = first_existing_key(detections, ["xyxy", "boxes", "bboxes", "bbox"])
    conf = first_existing_key(detections, ["confidence", "conf", "scores", "score"])
    cls = first_existing_key(detections, ["class_id", "class_ids", "labels", "classes"])

    xyxy_arr = to_numpy(xyxy)
    conf_arr = to_numpy(conf)
    cls_arr = to_numpy(cls)

    if xyxy_arr.size == 0:
        return []

    if xyxy_arr.ndim == 1 and xyxy_arr.shape[0] == 4:
        xyxy_arr = xyxy_arr.reshape(1, 4)

    if conf_arr.size == 0:
        conf_arr = np.ones((len(xyxy_arr),), dtype=np.float32)
    if cls_arr.size == 0:
        cls_arr = np.full((len(xyxy_arr),), -1, dtype=np.int32)

    results: List[Dict[str, Any]] = []
    for i in range(len(xyxy_arr)):
        c = safe_float(conf_arr[i] if len(conf_arr) > i else 1.0)
        if c < conf_thresh:
            continue

        class_id = safe_int(cls_arr[i] if len(cls_arr) > i else -1) + class_id_offset
        box = clamp_xyxy(xyxy_arr[i], frame_width, frame_height)
        if box is None:
            continue

        results.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "conf": round(c, 6),
                "xyxy": box,
                "xywh": xyxy_to_xywh(box),
            }
        )

    results.sort(key=lambda d: d["conf"], reverse=True)
    return results


def normalize_list_detections(
    detections: List[Any],
    frame_width: int,
    frame_height: int,
    conf_thresh: float,
    class_id_offset: int = 0,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for item in detections:
        if isinstance(item, dict):
            box = first_existing_key(item, ["xyxy", "bbox", "box", "bboxes"])
            conf = item.get("confidence", item.get("conf", item.get("score", 1.0)))
            cls = item.get("class_id", item.get("label", item.get("class", -1)))

            c = safe_float(conf)
            if c < conf_thresh:
                continue

            class_id = safe_int(cls) + class_id_offset
            xyxy = clamp_xyxy(box, frame_width, frame_height) if box is not None else None
            if xyxy is None:
                continue

            results.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "conf": round(c, 6),
                    "xyxy": xyxy,
                    "xywh": xyxy_to_xywh(xyxy),
                }
            )

        else:
            arr = list(item) if isinstance(item, (list, tuple, np.ndarray)) else []
            if len(arr) < 4:
                continue

            xyxy = clamp_xyxy(arr[:4], frame_width, frame_height)
            if xyxy is None:
                continue

            conf = safe_float(arr[4], 1.0) if len(arr) >= 5 else 1.0
            if conf < conf_thresh:
                continue

            class_id = safe_int(arr[5], -1) + class_id_offset if len(arr) >= 6 else -1

            results.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "conf": round(conf, 6),
                    "xyxy": xyxy,
                    "xywh": xyxy_to_xywh(xyxy),
                }
            )

    results.sort(key=lambda d: d["conf"], reverse=True)
    return results


def normalize_detections(
    detections: Any,
    frame_width: int,
    frame_height: int,
    conf_thresh: float,
    class_id_offset: int = 0,
) -> List[Dict[str, Any]]:
    if detections is None:
        return []

    if hasattr(detections, "xyxy"):
        return normalize_sv_detections(
            detections, frame_width, frame_height, conf_thresh, class_id_offset
        )

    if isinstance(detections, dict):
        for key in ["detections", "predictions", "results"]:
            if key in detections:
                return normalize_detections(
                    detections[key],
                    frame_width,
                    frame_height,
                    conf_thresh,
                    class_id_offset,
                )

        return normalize_dict_detections(
            detections, frame_width, frame_height, conf_thresh, class_id_offset
        )

    if isinstance(detections, (list, tuple)):
        if len(detections) >= 3:
            maybe_dict = {
                "xyxy": detections[0],
                "confidence": detections[1],
                "class_id": detections[2],
            }
            parsed = normalize_dict_detections(
                maybe_dict,
                frame_width,
                frame_height,
                conf_thresh,
                class_id_offset,
            )
            if parsed:
                return parsed

        return normalize_list_detections(
            list(detections), frame_width, frame_height, conf_thresh, class_id_offset
        )

    return []


def load_rfdetr_model(
    checkpoint_path: Path,
    device: str,
    model_size: str,
) -> Any:
    try:
        from rfdetr import RFDETRBase, RFDETRSmall
    except Exception as e:
        raise ImportError(
            "\nRF-DETR import에 실패했습니다.\n"
            "확인할 것:\n"
            "  1) 현재 위치가 D:\\HAESUNG\\prometheus\\YOLO-train 인지\n"
            "  2) v1\\rfdetr 폴더가 존재하는지\n"
            "  3) 또는 RF-DETR 패키지가 설치되어 있는지\n"
            f"원래 에러: {repr(e)}\n"
        )

    model_cls = RFDETRSmall if model_size == "small" else RFDETRBase

    constructor_trials = [
        {"pretrain_weights": str(checkpoint_path), "device": device},
        {"pretrain_weights": str(checkpoint_path)},
        {"checkpoint": str(checkpoint_path), "device": device},
        {"checkpoint": str(checkpoint_path)},
        {"weights": str(checkpoint_path), "device": device},
        {"weights": str(checkpoint_path)},
    ]

    last_error: Optional[BaseException] = None

    for kwargs in constructor_trials:
        try:
            print(f"[INFO] Trying RF-DETR constructor: {model_cls.__name__}({kwargs})")
            model = model_cls(**kwargs)
            print("[INFO] RF-DETR model loaded.")
            return model
        except Exception as e:
            last_error = e

    try:
        print(f"[INFO] Trying RF-DETR empty constructor: {model_cls.__name__}()")
        model = model_cls()

        for method_name in ["load", "load_weights", "load_checkpoint"]:
            if hasattr(model, method_name):
                print(f"[INFO] Trying model.{method_name}({checkpoint_path})")
                getattr(model, method_name)(str(checkpoint_path))
                print("[INFO] RF-DETR model loaded.")
                return model

        raise RuntimeError("RF-DETR 모델은 생성됐지만 checkpoint 로드 메서드를 찾지 못했습니다.")

    except Exception as e:
        last_error = e

    raise RuntimeError(
        "\nRF-DETR checkpoint 로드에 실패했습니다.\n"
        f"checkpoint: {checkpoint_path}\n"
        f"model_size: {model_size}\n"
        f"device: {device}\n"
        f"last_error: {repr(last_error)}\n"
    )


def predict_rfdetr(model: Any, frame_bgr: np.ndarray, conf: float) -> Any:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        return model.predict(frame_rgb, threshold=conf)
    except Exception:
        pass

    try:
        return model.predict(frame_rgb, conf=conf)
    except Exception:
        pass

    try:
        return model.predict(frame_rgb)
    except Exception:
        pass

    try:
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        return model.predict(pil_img, threshold=conf)
    except Exception:
        pass

    try:
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        return model.predict(pil_img)
    except Exception as e:
        raise RuntimeError("RF-DETR predict 호출에 실패했습니다.") from e


def compute_detection_stats(
    detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> Dict[str, Any]:
    person_dets = [d for d in detections if int(d["class_id"]) in PERSON_CLASS_IDS]
    player_gk_dets = [d for d in detections if int(d["class_id"]) in PLAYER_GK_CLASS_IDS]

    def ratios_for(dets: List[Dict[str, Any]]) -> Dict[str, float]:
        if not dets:
            return {
                "max_box_height_ratio": 0.0,
                "mean_box_height_ratio": 0.0,
                "max_box_area_ratio": 0.0,
                "mean_box_area_ratio": 0.0,
                "mean_center_y_ratio": 0.0,
                "max_y2_ratio": 0.0,
                "x_spread_ratio": 0.0,
            }

        heights = []
        areas = []
        centers_x = []
        centers_y = []
        y2s = []

        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            heights.append(h / max(frame_height, 1))
            areas.append((w * h) / max(frame_width * frame_height, 1))
            centers_x.append(((x1 + x2) / 2.0) / max(frame_width, 1))
            centers_y.append(((y1 + y2) / 2.0) / max(frame_height, 1))
            y2s.append(y2 / max(frame_height, 1))

        return {
            "max_box_height_ratio": round(float(np.max(heights)), 6),
            "mean_box_height_ratio": round(float(np.mean(heights)), 6),
            "max_box_area_ratio": round(float(np.max(areas)), 6),
            "mean_box_area_ratio": round(float(np.mean(areas)), 6),
            "mean_center_y_ratio": round(float(np.mean(centers_y)), 6),
            "max_y2_ratio": round(float(np.max(y2s)), 6),
            "x_spread_ratio": round(float(np.max(centers_x) - np.min(centers_x)), 6)
            if len(centers_x) >= 2
            else 0.0,
        }

    class_counts = {str(k): 0 for k in CLASS_NAMES.keys()}
    for d in detections:
        cid = int(d["class_id"])
        if str(cid) in class_counts:
            class_counts[str(cid)] += 1

    return {
        "num_detections": len(detections),
        "num_person": len(person_dets),
        "num_player_gk": len(player_gk_dets),
        "class_counts": {
            str(k): {"class_name": CLASS_NAMES[int(k)], "count": int(v)}
            for k, v in class_counts.items()
        },
        "person": ratios_for(person_dets),
        "player_gk": ratios_for(player_gk_dets),
    }


def classify_scene_mode(
    detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stats = compute_detection_stats(detections, frame_width, frame_height)

    num_person = stats["num_person"]
    num_player_gk = stats["num_player_gk"]

    person = stats["person"]
    player_gk = stats["player_gk"]

    max_person_h = person["max_box_height_ratio"]
    max_person_area = person["max_box_area_ratio"]
    mean_person_h = person["mean_box_height_ratio"]
    mean_person_center_y = person["mean_center_y_ratio"]
    max_person_y2 = person["max_y2_ratio"]

    max_pg_h = player_gk["max_box_height_ratio"]
    mean_pg_h = player_gk["mean_box_height_ratio"]

    if num_player_gk == 0:
        mode = "non_player_or_crowd"
        reason = "no_player_or_goalkeeper_detection"
    else:
        is_closeup = (
            num_person <= args.closeup_max_persons
            and (
                max_person_h >= args.closeup_min_box_height_ratio
                or max_person_area >= args.closeup_min_box_area_ratio
            )
        )

        is_play = (
            num_player_gk >= args.play_min_player_gk
            and max_pg_h < args.play_max_box_height_ratio
            and mean_pg_h < args.play_mean_box_height_ratio
        )

        is_crowd_like = (
            num_person >= args.crowd_min_persons
            and mean_person_h <= args.crowd_max_mean_height_ratio
            and mean_person_center_y <= args.crowd_max_mean_center_y_ratio
            and max_person_y2 <= args.crowd_max_y2_ratio
        )

        if is_closeup:
            mode = "closeup_observation"
            reason = (
                "large_person_box_detected:"
                f"num_person={num_person},"
                f"max_h={max_person_h:.3f},"
                f"max_area={max_person_area:.3f}"
            )
        elif is_play:
            mode = "play_tracking"
            reason = (
                "many_player_gk_small_or_mid_boxes:"
                f"num_player_gk={num_player_gk},"
                f"max_pg_h={max_pg_h:.3f},"
                f"mean_pg_h={mean_pg_h:.3f}"
            )
        elif is_crowd_like:
            mode = "non_player_or_crowd"
            reason = (
                "crowd_like_upper_small_boxes:"
                f"num_person={num_person},"
                f"mean_h={mean_person_h:.3f},"
                f"mean_center_y={mean_person_center_y:.3f},"
                f"max_y2={max_person_y2:.3f}"
            )
        elif num_player_gk >= args.play_min_player_gk:
            mode = "play_tracking"
            reason = (
                "fallback_many_player_gk:"
                f"num_player_gk={num_player_gk},"
                f"max_pg_h={max_pg_h:.3f},"
                f"mean_pg_h={mean_pg_h:.3f}"
            )
        else:
            mode = "non_player_or_crowd"
            reason = (
                "too_few_player_gk_for_tracking_and_not_closeup:"
                f"num_player_gk={num_player_gk},"
                f"num_person={num_person},"
                f"max_person_h={max_person_h:.3f}"
            )

    return {"mode": mode, "reason": reason, "stats": stats}


def crop_quality_score(det: Dict[str, Any], frame_width: int, frame_height: int) -> float:
    ratios = box_ratios(det["xyxy"], frame_width, frame_height)
    conf = float(det["conf"])
    h = ratios["box_height_ratio"]
    area = ratios["box_area_ratio"]
    score = 0.70 * conf + 0.20 * min(h / 0.60, 1.0) + 0.10 * min(area / 0.25, 1.0)
    return round(float(score), 6)


def select_closeup_crop_candidates(
    detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    crop_class_ids: Set[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for det_idx, det in enumerate(detections):
        class_id = int(det["class_id"])
        conf = float(det["conf"])

        if class_id not in crop_class_ids:
            continue
        if conf < args.min_crop_conf:
            continue

        ratios = box_ratios(det["xyxy"], frame_width, frame_height)

        if ratios["box_height_ratio"] < args.min_crop_box_height_ratio:
            continue
        if ratios["box_area_ratio"] < args.min_crop_box_area_ratio:
            continue

        candidates.append(
            {
                "det_idx": det_idx,
                "det": det,
                "ratios": ratios,
                "quality_score": crop_quality_score(det, frame_width, frame_height),
            }
        )

    candidates.sort(
        key=lambda x: (
            x["quality_score"],
            x["det"]["conf"],
            x["ratios"]["box_area_ratio"],
        ),
        reverse=True,
    )
    return candidates[: max(args.top_k_closeup, 0)]


def save_closeup_crops_for_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    time_sec: float,
    detections: List[Dict[str, Any]],
    scene_info: Dict[str, Any],
    source_path: Path,
    out_dir: Path,
    crops_dir: Path,
    crop_class_ids: Set[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if args.no_save_closeup_crops:
        return []
    if scene_info["mode"] != "closeup_observation":
        return []

    frame_height, frame_width = frame_bgr.shape[:2]

    candidates = select_closeup_crop_candidates(
        detections=detections,
        frame_width=frame_width,
        frame_height=frame_height,
        crop_class_ids=crop_class_ids,
        args=args,
    )

    records: List[Dict[str, Any]] = []

    for rank, item in enumerate(candidates):
        det_idx = item["det_idx"]
        det = item["det"]
        ratios = item["ratios"]

        padded_xyxy = pad_and_clamp_xyxy(
            det["xyxy"],
            width=frame_width,
            height=frame_height,
            pad_ratio=args.crop_pad_ratio,
        )
        if padded_xyxy is None:
            continue

        x1, y1, x2, y2 = padded_xyxy
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        class_id = int(det["class_id"])
        class_name = det["class_name"]
        conf_tag = int(round(float(det["conf"]) * 1000))

        crop_id = f"frame_{frame_idx:06d}_det_{det_idx:03d}"
        filename = f"{crop_id}_{class_name}_conf{conf_tag:03d}.jpg"
        crop_path = crops_dir / filename

        ok = cv2.imwrite(str(crop_path), crop)
        if not ok:
            print(f"[WARN] failed to save crop: {crop_path}")
            continue

        try:
            rel_crop_path = str(crop_path.relative_to(out_dir))
        except ValueError:
            rel_crop_path = str(crop_path)

        records.append(
            {
                "frame": frame_idx,
                "time": round(time_sec, 4),
                "crop_id": crop_id,
                "rank": rank,
                "det_id": f"f{frame_idx:06d}_d{det_idx:03d}",
                "class_id": class_id,
                "class_name": class_name,
                "conf": det["conf"],
                "xyxy": det["xyxy"],
                "xywh": det["xywh"],
                "padded_xyxy": padded_xyxy,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "ratios": ratios,
                "quality_score": item["quality_score"],
                "mode": scene_info["mode"],
                "mode_reason": scene_info["reason"],
                "crop_path": rel_crop_path.replace("\\", "/"),
                "source": str(source_path),
            }
        )

    return records


@dataclass
class SimpleTrack:
    track_id: int
    class_id: int
    class_name: str
    xyxy: List[float]
    conf: float
    start_frame: int
    last_frame: int
    hits: int = 1
    missed: int = 0


class SimpleIoUTracker:
    def __init__(self, iou_thresh: float = 0.35, max_age: int = 12):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: List[SimpleTrack] = []

    def reset(self) -> None:
        self.tracks = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        detections:
          [{"det_idx": int, "det_id": str, "class_id": int, "xyxy": [...], ...}, ...]

        return:
          local track records for current frame
        """
        outputs: List[Dict[str, Any]] = []

        # old tracks age update
        for trk in self.tracks:
            trk.missed += 1

        pairs: List[Tuple[float, int, int]] = []
        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if trk.class_id != int(det["class_id"]):
                    continue
                iou = bbox_iou(trk.xyxy, det["xyxy"])
                if iou >= self.iou_thresh:
                    pairs.append((iou, ti, di))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        for iou, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue

            trk = self.tracks[ti]
            det = detections[di]

            trk.xyxy = det["xyxy"]
            trk.conf = float(det["conf"])
            trk.last_frame = frame_idx
            trk.hits += 1
            trk.missed = 0

            matched_tracks.add(ti)
            matched_dets.add(di)

            outputs.append(
                {
                    "track_id": trk.track_id,
                    "track_age": trk.hits,
                    "track_start_frame": trk.start_frame,
                    "match_iou": round(float(iou), 6),
                    "is_new_track": False,
                    "det": det,
                }
            )

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            trk = SimpleTrack(
                track_id=track_id,
                class_id=int(det["class_id"]),
                class_name=det["class_name"],
                xyxy=det["xyxy"],
                conf=float(det["conf"]),
                start_frame=frame_idx,
                last_frame=frame_idx,
                hits=1,
                missed=0,
            )
            self.tracks.append(trk)

            outputs.append(
                {
                    "track_id": trk.track_id,
                    "track_age": trk.hits,
                    "track_start_frame": trk.start_frame,
                    "match_iou": None,
                    "is_new_track": True,
                    "det": det,
                }
            )

        self.tracks = [trk for trk in self.tracks if trk.missed <= self.max_age]

        outputs.sort(key=lambda x: x["track_id"])
        return outputs


def select_tracking_candidates(
    detections: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    track_class_ids: Set[int],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for det_idx, det in enumerate(detections):
        class_id = int(det["class_id"])
        conf = float(det["conf"])

        if class_id not in track_class_ids:
            continue
        if conf < args.min_track_conf:
            continue

        ratios = box_ratios(det["xyxy"], frame_width, frame_height)

        if ratios["box_height_ratio"] < args.min_track_box_height_ratio:
            continue
        if ratios["box_area_ratio"] < args.min_track_box_area_ratio:
            continue

        item = dict(det)
        item["det_idx"] = det_idx
        item["det_id"] = f"f{{FRAME_IDX_PLACEHOLDER}}_d{det_idx:03d}"
        item["ratios"] = ratios
        candidates.append(item)

    candidates.sort(key=lambda d: float(d["conf"]), reverse=True)
    return candidates


def make_local_track_records(
    frame_idx: int,
    time_sec: float,
    frame_width: int,
    frame_height: int,
    source_path: Path,
    tracker_outputs: List[Dict[str, Any]],
    mode: str,
    tracker_type: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for item in tracker_outputs:
        det = item["det"]
        det_id = f"f{frame_idx:06d}_d{int(det['det_idx']):03d}"

        records.append(
            {
                "frame": frame_idx,
                "time": round(time_sec, 4),
                "track_id": int(item["track_id"]),
                "track_age": int(item["track_age"]),
                "track_start_frame": int(item["track_start_frame"]),
                "is_new_track": bool(item["is_new_track"]),
                "match_iou": item["match_iou"],
                "det_id": det_id,
                "class_id": int(det["class_id"]),
                "class_name": det["class_name"],
                "conf": det["conf"],
                "xyxy": det["xyxy"],
                "xywh": det["xywh"],
                "ratios": det["ratios"],
                "mode": mode,
                "tracker_type": tracker_type,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "source": str(source_path),
            }
        )

    return records


def draw_preview(
    frame_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    frame_idx: int,
    time_sec: float,
    scene_info: Dict[str, Any],
    saved_crop_det_ids: Set[str],
    local_track_records: List[Dict[str, Any]],
) -> np.ndarray:
    vis = frame_bgr.copy()

    det_id_to_track = {
        rec["det_id"]: rec
        for rec in local_track_records
    }

    for det_idx, det in enumerate(detections):
        class_id = int(det["class_id"])
        class_name = det["class_name"]
        conf = float(det["conf"])
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]

        det_id = f"f{frame_idx:06d}_d{det_idx:03d}"
        is_saved_crop = det_id in saved_crop_det_ids
        track_rec = det_id_to_track.get(det_id)

        color = CLASS_COLORS_BGR.get(class_id, (255, 255, 255))
        thickness = 2

        if track_rec is not None:
            thickness = 4
            label = f"T{track_rec['track_id']} {class_name} {conf:.2f}"
        elif is_saved_crop:
            thickness = 4
            label = f"CROP {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_w, label_h = label_size
        y_text_top = max(0, y1 - label_h - baseline - 4)

        cv2.rectangle(
            vis,
            (x1, y_text_top),
            (x1 + label_w + 6, y_text_top + label_h + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (x1 + 3, y_text_top + label_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    mode = scene_info["mode"]
    stats = scene_info["stats"]
    mode_color = MODE_COLORS_BGR.get(mode, (255, 255, 255))

    header_1 = (
        f"frame={frame_idx} time={time_sec:.2f}s "
        f"dets={len(detections)} mode={mode}"
    )
    header_2 = (
        f"num_pg={stats['num_player_gk']} "
        f"num_person={stats['num_person']} "
        f"tracks={len(local_track_records)} "
        f"crops={len(saved_crop_det_ids)}"
    )

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 64), (0, 0, 0), -1)

    cv2.putText(
        vis,
        header_1,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        header_2,
        (12, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    reason_short = scene_info["reason"][:120]
    cv2.putText(
        vis,
        reason_short,
        (12, vis.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


def open_video(source: Path) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    return cap, {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def create_video_writer(
    out_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"preview video writer를 열 수 없습니다: {out_path}")
    return writer


def main() -> None:
    args = parse_args()

    project_root = resolve_project_root(args)
    add_import_paths(project_root)

    source_path = (
        (project_root / args.source).resolve()
        if not Path(args.source).is_absolute()
        else Path(args.source).resolve()
    )
    checkpoint_path = (
        (project_root / args.checkpoint).resolve()
        if not Path(args.checkpoint).is_absolute()
        else Path(args.checkpoint).resolve()
    )
    out_dir = (
        (project_root / args.out).resolve()
        if not Path(args.out).is_absolute()
        else Path(args.out).resolve()
    )

    ensure_dir(out_dir)

    crops_dir = out_dir / "closeup_crops"
    if not args.no_save_closeup_crops:
        ensure_dir(crops_dir)

    crop_class_ids = parse_class_ids(args.crop_classes)
    track_class_ids = parse_class_ids(args.track_classes)

    detections_path = out_dir / "detections.jsonl"
    shot_modes_path = out_dir / "shot_modes.jsonl"
    closeup_crops_path = out_dir / "closeup_crops.jsonl"
    local_tracks_path = out_dir / "local_tracks.jsonl"
    summary_path = out_dir / "summary.json"
    preview_path = out_dir / args.preview_name

    if not source_path.exists():
        raise FileNotFoundError(f"source video not found: {source_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("[Stage 1-D/E] RF-DETR hybrid local tracking")
    print(f"[INFO] project_root : {project_root}")
    print(f"[INFO] source       : {source_path}")
    print(f"[INFO] checkpoint   : {checkpoint_path}")
    print(f"[INFO] out          : {out_dir}")
    print(f"[INFO] conf         : {args.conf}")
    print(f"[INFO] device       : {args.device}")
    print(f"[INFO] crop_classes : {sorted(crop_class_ids)}")
    print(f"[INFO] track_classes: {sorted(track_class_ids)}")
    print(f"[INFO] tracker      : {args.tracker_type}")
    print("=" * 80)

    model = load_rfdetr_model(
        checkpoint_path=checkpoint_path,
        device=args.device,
        model_size=args.model_size,
    )

    tracker = SimpleIoUTracker(
        iou_thresh=args.track_iou_thresh,
        max_age=args.track_max_age,
    )

    cap, video_info = open_video(source_path)
    fps = float(video_info["fps"])
    width = int(video_info["width"])
    height = int(video_info["height"])
    total_frames = int(video_info["frame_count"])

    writer = None
    if not args.no_preview:
        writer = create_video_writer(preview_path, fps, width, height)

    started_at = time.time()

    frame_idx = 0
    processed_frames = 0
    total_detections = 0
    total_closeup_crops = 0
    closeup_crop_frame_count = 0
    total_local_track_records = 0
    play_tracking_frame_count = 0

    unique_track_ids: Set[int] = set()

    per_class_counts = {str(k): 0 for k in CLASS_NAMES.keys()}
    crop_class_counts = {str(k): 0 for k in CLASS_NAMES.keys()}
    track_class_counts = {str(k): 0 for k in CLASS_NAMES.keys()}

    mode_counts = {
        "play_tracking": 0,
        "closeup_observation": 0,
        "non_player_or_crowd": 0,
    }

    failed_frames: List[Dict[str, Any]] = []

    with open(detections_path, "w", encoding="utf-8") as f_det, open(
        shot_modes_path, "w", encoding="utf-8"
    ) as f_mode, open(closeup_crops_path, "w", encoding="utf-8") as f_crop, open(
        local_tracks_path, "w", encoding="utf-8"
    ) as f_track:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            time_sec = frame_idx / fps

            try:
                raw_detections = predict_rfdetr(model, frame, args.conf)
                detections = normalize_detections(
                    raw_detections,
                    frame_width=width,
                    frame_height=height,
                    conf_thresh=args.conf,
                    class_id_offset=args.class_id_offset,
                )
            except Exception as e:
                detections = []
                failed_frames.append(
                    {
                        "frame": frame_idx,
                        "time": round(time_sec, 4),
                        "error": repr(e),
                    }
                )
                print(f"[WARN] frame {frame_idx} inference failed: {repr(e)}")

            scene_info = classify_scene_mode(
                detections=detections,
                frame_width=width,
                frame_height=height,
                args=args,
            )

            mode = scene_info["mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

            # 1) detections.jsonl
            for det_idx, det in enumerate(detections):
                class_id = int(det["class_id"])
                if str(class_id) in per_class_counts:
                    per_class_counts[str(class_id)] += 1

                record = {
                    "frame": frame_idx,
                    "time": round(time_sec, 4),
                    "det_id": f"f{frame_idx:06d}_d{det_idx:03d}",
                    "class_id": class_id,
                    "class_name": det["class_name"],
                    "conf": det["conf"],
                    "xyxy": det["xyxy"],
                    "xywh": det["xywh"],
                    "frame_width": width,
                    "frame_height": height,
                    "source": str(source_path),
                }
                f_det.write(json_dumps(record) + "\n")

            # 2) shot_modes.jsonl
            mode_record = {
                "frame": frame_idx,
                "time": round(time_sec, 4),
                "mode": scene_info["mode"],
                "reason": scene_info["reason"],
                "num_detections": scene_info["stats"]["num_detections"],
                "num_person": scene_info["stats"]["num_person"],
                "num_player_gk": scene_info["stats"]["num_player_gk"],
                "stats": scene_info["stats"],
            }
            f_mode.write(json_dumps(mode_record) + "\n")

            # 3) closeup crops
            crop_records = save_closeup_crops_for_frame(
                frame_bgr=frame,
                frame_idx=frame_idx,
                time_sec=time_sec,
                detections=detections,
                scene_info=scene_info,
                source_path=source_path,
                out_dir=out_dir,
                crops_dir=crops_dir,
                crop_class_ids=crop_class_ids,
                args=args,
            )

            saved_crop_det_ids: Set[str] = set()
            if crop_records:
                closeup_crop_frame_count += 1

            for crop_record in crop_records:
                f_crop.write(json_dumps(crop_record) + "\n")
                total_closeup_crops += 1

                cid = int(crop_record["class_id"])
                if str(cid) in crop_class_counts:
                    crop_class_counts[str(cid)] += 1

                saved_crop_det_ids.add(crop_record["det_id"])

            # 4) local tracks: play_tracking frame only
            local_track_records: List[Dict[str, Any]] = []

            if mode == "play_tracking":
                play_tracking_frame_count += 1

                tracking_candidates = select_tracking_candidates(
                    detections=detections,
                    frame_width=width,
                    frame_height=height,
                    track_class_ids=track_class_ids,
                    args=args,
                )

                # det_id placeholder 보정
                for cand in tracking_candidates:
                    cand["det_id"] = f"f{frame_idx:06d}_d{int(cand['det_idx']):03d}"

                tracker_outputs = tracker.update(
                    detections=tracking_candidates,
                    frame_idx=frame_idx,
                )

                local_track_records = make_local_track_records(
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    frame_width=width,
                    frame_height=height,
                    source_path=source_path,
                    tracker_outputs=tracker_outputs,
                    mode=mode,
                    tracker_type=args.tracker_type,
                )

                for rec in local_track_records:
                    f_track.write(json_dumps(rec) + "\n")
                    total_local_track_records += 1
                    unique_track_ids.add(int(rec["track_id"]))

                    cid = int(rec["class_id"])
                    if str(cid) in track_class_counts:
                        track_class_counts[str(cid)] += 1

            else:
                if not args.keep_tracker_through_non_play:
                    tracker.reset()

            total_detections += len(detections)

            if writer is not None:
                vis = draw_preview(
                    frame_bgr=frame,
                    detections=detections,
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    scene_info=scene_info,
                    saved_crop_det_ids=saved_crop_det_ids,
                    local_track_records=local_track_records,
                )
                writer.write(vis)

            processed_frames += 1

            if args.print_every > 0 and processed_frames % args.print_every == 0:
                elapsed = time.time() - started_at
                fps_proc = processed_frames / max(elapsed, 1e-6)
                print(
                    f"[INFO] processed={processed_frames}/{total_frames} "
                    f"frame={frame_idx} dets={total_detections} "
                    f"mode={mode} crops={total_closeup_crops} "
                    f"tracks={total_local_track_records} "
                    f"uniqueT={len(unique_track_ids)} "
                    f"speed={fps_proc:.2f} fps"
                )

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - started_at

    summary = {
        "stage": "stage1_de_rfdetr_detection_scene_mode_closeup_crops_local_tracks",
        "source": str(source_path),
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "detections_jsonl": str(detections_path),
        "shot_modes_jsonl": str(shot_modes_path),
        "closeup_crops_jsonl": str(closeup_crops_path),
        "closeup_crops_dir": str(crops_dir),
        "local_tracks_jsonl": str(local_tracks_path),
        "preview_video": str(preview_path) if writer is not None else None,
        "conf": args.conf,
        "device": args.device,
        "model_size": args.model_size,
        "class_id_offset": args.class_id_offset,
        "video": video_info,
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "mode_counts": mode_counts,
        "play_tracking_frame_count": play_tracking_frame_count,
        "total_closeup_crops": total_closeup_crops,
        "closeup_crop_frame_count": closeup_crop_frame_count,
        "total_local_track_records": total_local_track_records,
        "unique_local_track_ids": len(unique_track_ids),
        "per_class_counts": {
            str(k): {"class_name": CLASS_NAMES[int(k)], "count": int(v)}
            for k, v in per_class_counts.items()
        },
        "crop_class_counts": {
            str(k): {"class_name": CLASS_NAMES[int(k)], "count": int(v)}
            for k, v in crop_class_counts.items()
        },
        "track_class_counts": {
            str(k): {"class_name": CLASS_NAMES[int(k)], "count": int(v)}
            for k, v in track_class_counts.items()
        },
        "scene_mode_thresholds": {
            "play_min_player_gk": args.play_min_player_gk,
            "play_max_box_height_ratio": args.play_max_box_height_ratio,
            "play_mean_box_height_ratio": args.play_mean_box_height_ratio,
            "closeup_min_box_height_ratio": args.closeup_min_box_height_ratio,
            "closeup_min_box_area_ratio": args.closeup_min_box_area_ratio,
            "closeup_max_persons": args.closeup_max_persons,
            "crowd_min_persons": args.crowd_min_persons,
            "crowd_max_mean_height_ratio": args.crowd_max_mean_height_ratio,
            "crowd_max_mean_center_y_ratio": args.crowd_max_mean_center_y_ratio,
            "crowd_max_y2_ratio": args.crowd_max_y2_ratio,
        },
        "closeup_crop_settings": {
            "top_k_closeup": args.top_k_closeup,
            "crop_classes": sorted(crop_class_ids),
            "min_crop_conf": args.min_crop_conf,
            "min_crop_box_height_ratio": args.min_crop_box_height_ratio,
            "min_crop_box_area_ratio": args.min_crop_box_area_ratio,
            "crop_pad_ratio": args.crop_pad_ratio,
            "save_closeup_crops": not args.no_save_closeup_crops,
        },
        "tracker_settings": {
            "tracker_type": args.tracker_type,
            "track_classes": sorted(track_class_ids),
            "track_iou_thresh": args.track_iou_thresh,
            "track_max_age": args.track_max_age,
            "min_track_conf": args.min_track_conf,
            "min_track_box_height_ratio": args.min_track_box_height_ratio,
            "min_track_box_area_ratio": args.min_track_box_area_ratio,
            "keep_tracker_through_non_play": args.keep_tracker_through_non_play,
        },
        "failed_frame_count": len(failed_frames),
        "failed_frames_sample": failed_frames[:20],
        "elapsed_sec": round(elapsed, 3),
        "processing_fps": round(processed_frames / max(elapsed, 1e-6), 3),
        "outputs": {
            "detections_jsonl": "detections.jsonl",
            "shot_modes_jsonl": "shot_modes.jsonl",
            "closeup_crops_jsonl": "closeup_crops.jsonl",
            "closeup_crops_dir": "closeup_crops",
            "local_tracks_jsonl": "local_tracks.jsonl",
            "summary_json": "summary.json",
            "preview_video": args.preview_name if not args.no_preview else None,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE] Stage 1-D/E completed.")
    print(f"[OUT] detections      : {detections_path}")
    print(f"[OUT] shot_modes      : {shot_modes_path}")
    print(f"[OUT] closeup_crops   : {closeup_crops_path}")
    print(f"[OUT] crops_dir       : {crops_dir}")
    print(f"[OUT] local_tracks    : {local_tracks_path}")
    print(f"[OUT] summary         : {summary_path}")
    if not args.no_preview:
        print(f"[OUT] preview         : {preview_path}")
    print(f"[INFO] processed_frames         : {processed_frames}")
    print(f"[INFO] total_detections         : {total_detections}")
    print(f"[INFO] total_closeup_crops      : {total_closeup_crops}")
    print(f"[INFO] total_local_track_records: {total_local_track_records}")
    print(f"[INFO] unique_local_track_ids   : {len(unique_track_ids)}")
    print(f"[INFO] mode_counts              : {mode_counts}")
    print(f"[INFO] elapsed_sec              : {elapsed:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Stage 1-D/E failed.")
        traceback.print_exc()
        sys.exit(1)