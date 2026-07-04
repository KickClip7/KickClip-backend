from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BrowserPlayability:
    is_browser_playable: bool
    reason: str


def check_browser_playability(
    metadata: dict[str, Any],
    mime_type: str | None = None,
    filename: str | None = None,
) -> BrowserPlayability:
    """Decide whether uploaded video is likely playable in browsers.

    이 함수는 완벽한 브라우저 호환성 판정기가 아니라,
    KickClip MVP에서 프리뷰 변환이 필요한지 판단하기 위한 보수적인 검사다.

    현재는 가장 안전한 브라우저 호환 MP4 조건을 다음처럼 본다.

    - container/mime: mp4 계열
    - video codec: h264
    - codec tag: avc1 또는 h264 계열
    - pixel format: yuv420p 또는 yuvj420p

    mpeg4/mp4v, hevc/h265, av1 등은 브라우저에서 안 열릴 수 있으므로
    WEB_PREVIEW_VIDEO 변환 대상으로 본다.
    """
    codec_name = _lower(metadata.get("codec_name"))
    codec_tag_string = _lower(metadata.get("codec_tag_string"))
    pix_fmt = _lower(metadata.get("pix_fmt"))

    suffix = _lower(Path(filename or "").suffix)
    mime = _lower(mime_type)

    is_mp4_like = (
        "mp4" in mime
        or suffix in {".mp4", ".m4v", ".mov"}
    )

    if not is_mp4_like:
        return BrowserPlayability(
            is_browser_playable=False,
            reason="not_mp4_like_container",
        )

    if codec_name != "h264":
        return BrowserPlayability(
            is_browser_playable=False,
            reason=f"unsupported_video_codec:{codec_name or 'unknown'}",
        )

    if codec_tag_string and codec_tag_string not in {"avc1", "h264"}:
        return BrowserPlayability(
            is_browser_playable=False,
            reason=f"unsupported_codec_tag:{codec_tag_string}",
        )

    if pix_fmt and pix_fmt not in {"yuv420p", "yuvj420p"}:
        return BrowserPlayability(
            is_browser_playable=False,
            reason=f"unsupported_pixel_format:{pix_fmt}",
        )

    return BrowserPlayability(
        is_browser_playable=True,
        reason="browser_playable_h264_mp4",
    )


def _lower(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()