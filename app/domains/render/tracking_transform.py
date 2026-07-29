from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CropPolicy:
    version: str = "target_centered_crop_v1"
    target_padding: float = 2.2
    minimum_crop_width_ratio: float = 0.32
    minimum_crop_height_ratio: float = 0.32
    low_confidence_scale: float = 1.25
    smoothing_alpha: float = 0.22
    max_center_step_ratio: float = 0.05
    # Keep the target slightly above center so bottom captions have room.
    target_vertical_position: float = 0.45


class TrackingTransformBuilder:
    """Convert trusted target boxes to bounded, smoothed crop keyframes."""

    def __init__(self, policy: CropPolicy | None = None):
        self.policy = policy or CropPolicy()

    def build(
        self,
        *,
        keyframes: list[dict[str, Any]],
        source_width: int,
        source_height: int,
        output_width: int,
        output_height: int,
    ) -> dict[str, Any]:
        if not keyframes:
            raise ValueError("TARGET_CENTERED segment has no trusted crop keyframes.")
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Source dimensions are required for target crop.")

        output_aspect = output_width / output_height
        raw: list[dict[str, float | str]] = []
        for frame in keyframes:
            bbox = frame.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Trusted crop keyframe has an invalid bbox.")
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
                raise ValueError("Trusted crop bbox is outside source boundaries.")

            box_width = x2 - x1
            box_height = y2 - y1
            padding = self.policy.target_padding
            if frame.get("state") == "ACTIVE_LOW_CONFIDENCE":
                padding *= self.policy.low_confidence_scale
            crop_height = max(
                box_height * padding,
                source_height * self.policy.minimum_crop_height_ratio,
            )
            crop_width = max(
                box_width * padding,
                source_width * self.policy.minimum_crop_width_ratio,
                crop_height * output_aspect,
            )
            crop_height = max(crop_height, crop_width / output_aspect)
            if crop_width > source_width:
                crop_width = float(source_width)
                crop_height = crop_width / output_aspect
            if crop_height > source_height:
                crop_height = float(source_height)
                crop_width = crop_height * output_aspect

            raw.append(
                {
                    "time": float(frame.get("relative_time_sec") or 0.0),
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "state": str(frame.get("state") or "ACTIVE"),
                }
            )

        # Use one conservative crop size for the segment. This avoids zoom
        # pumping; low-confidence observations can only widen it.
        crop_width = min(
            float(source_width),
            max(float(row["crop_width"]) for row in raw),
        )
        crop_height = min(
            float(source_height),
            max(float(row["crop_height"]) for row in raw),
        )
        if crop_width / crop_height > output_aspect:
            crop_height = crop_width / output_aspect
        else:
            crop_width = crop_height * output_aspect
        crop_width = min(crop_width, float(source_width))
        crop_height = min(crop_height, float(source_height))

        smoothed: list[dict[str, Any]] = []
        previous_x: float | None = None
        previous_y: float | None = None
        max_step_x = source_width * self.policy.max_center_step_ratio
        max_step_y = source_height * self.policy.max_center_step_ratio
        for row in raw:
            center_x = float(row["center_x"])
            center_y = float(row["center_y"]) + crop_height * (
                0.5 - self.policy.target_vertical_position
            )
            if previous_x is not None and previous_y is not None:
                alpha = self.policy.smoothing_alpha
                center_x = previous_x + self._clamp(
                    (center_x - previous_x) * alpha,
                    -max_step_x,
                    max_step_x,
                )
                center_y = previous_y + self._clamp(
                    (center_y - previous_y) * alpha,
                    -max_step_y,
                    max_step_y,
                )
            half_width = crop_width / 2
            half_height = crop_height / 2
            center_x = self._clamp(center_x, half_width, source_width - half_width)
            center_y = self._clamp(center_y, half_height, source_height - half_height)
            previous_x, previous_y = center_x, center_y
            smoothed.append(
                {
                    "time_sec": round(float(row["time"]), 6),
                    "x": round(center_x - half_width, 3),
                    "y": round(center_y - half_height, 3),
                    "state": row["state"],
                }
            )
        return {
            "policy_version": self.policy.version,
            "crop_width": max(2, int(round(crop_width)) // 2 * 2),
            "crop_height": max(2, int(round(crop_height)) // 2 * 2),
            "keyframes": smoothed,
        }

    @classmethod
    def ffmpeg_filter(
        cls,
        *,
        transform: dict[str, Any],
        output_width: int,
        output_height: int,
    ) -> str:
        keyframes = transform.get("keyframes") or []
        if not keyframes:
            raise ValueError("Target transform contains no keyframes.")
        x_expr = cls._piecewise_expression(keyframes, "x")
        y_expr = cls._piecewise_expression(keyframes, "y")
        crop_width = int(transform["crop_width"])
        crop_height = int(transform["crop_height"])
        return (
            f"crop={crop_width}:{crop_height}:x='{x_expr}':y='{y_expr}',"
            f"scale={output_width}:{output_height}:flags=lanczos,setsar=1"
        )

    @staticmethod
    def _piecewise_expression(keyframes: list[dict[str, Any]], field: str) -> str:
        if len(keyframes) == 1:
            return f"{float(keyframes[0][field]):.3f}"
        expression = f"{float(keyframes[-1][field]):.3f}"
        for current, following in reversed(list(zip(keyframes, keyframes[1:]))):
            start = float(current["time_sec"])
            end = float(following["time_sec"])
            start_value = float(current[field])
            end_value = float(following[field])
            if end <= start:
                segment = f"{end_value:.3f}"
            else:
                segment = (
                    f"{start_value:.3f}+({end_value - start_value:.3f})"
                    f"*(t-{start:.6f})/{end - start:.6f}"
                )
            expression = (
                f"if(between(t,{start:.6f},{end:.6f}),{segment},{expression})"
            )
        first = keyframes[0]
        return (
            f"if(lt(t,{float(first['time_sec']):.6f}),"
            f"{float(first[field]):.3f},{expression})"
        )

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        if maximum < minimum:
            return (minimum + maximum) / 2
        return max(minimum, min(maximum, value))
