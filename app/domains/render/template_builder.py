from __future__ import annotations

from dataclasses import dataclass


RESOLUTION_BY_QUALITY_AND_RATIO: dict[str, dict[str, tuple[int, int]]] = {
    "720p": {
        "9:16": (720, 1280),
        "1:1": (720, 720),
        "16:9": (1280, 720),
    },
    "1080p": {
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
        "16:9": (1920, 1080),
    },
    "4K": {
        "9:16": (2160, 3840),
        "1:1": (2160, 2160),
        "16:9": (3840, 2160),
    },
}


@dataclass(frozen=True)
class RenderTemplate:
    ratio: str
    quality: str
    width: int
    height: int
    video_filter: str
    resolution_label: str


class RenderTemplateBuilder:
    """RenderJob 옵션을 FFmpeg에서 사용할 수 있는 템플릿으로 변환한다."""

    def build(
        self,
        *,
        ratio: str,
        quality: str,
    ) -> RenderTemplate:
        normalized_quality = self._normalize_quality(quality)
        if ratio not in RESOLUTION_BY_QUALITY_AND_RATIO.get(normalized_quality, {}):
            raise ValueError("ratio must be one of: 9:16, 1:1, 16:9")

        width, height = RESOLUTION_BY_QUALITY_AND_RATIO[normalized_quality][ratio]
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )

        return RenderTemplate(
            ratio=ratio,
            quality=normalized_quality,
            width=width,
            height=height,
            video_filter=video_filter,
            resolution_label=f"{width}x{height}",
        )

    @staticmethod
    def _normalize_quality(quality: str | None) -> str:
        if quality is None:
            return "1080p"

        normalized = quality.strip()
        if normalized.lower() == "4k":
            normalized = "4K"

        if normalized not in RESOLUTION_BY_QUALITY_AND_RATIO:
            raise ValueError("quality must be one of: 720p, 1080p, 4K")

        return normalized
