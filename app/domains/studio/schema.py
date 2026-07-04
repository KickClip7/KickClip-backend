from pydantic import BaseModel

from app.domains.player.schema import FrontendPlayer
from app.domains.timeline.schema import FrontendTimelineEvent


class VideoCodecInfo(BaseModel):
    codec_name: str | None = None
    codec_tag_string: str | None = None
    pix_fmt: str | None = None
    format_name: str | None = None


class UploadedVideoInfo(BaseModel):
    filename: str | None
    content_type: str | None
    duration_sec: float | None
    fps: float | None
    width: int | None
    height: int | None
    size_bytes: int | None
    codec: VideoCodecInfo


class UploadMatchVideoResponse(BaseModel):
    project_id: str
    match_id: str

    # 원본 보관/AI 분석용 asset
    raw_video_asset_id: str

    # 브라우저에서 보여줄 대표 asset
    # 원본이 재생 가능하면 RAW_VIDEO를 가리키고,
    # 원본이 비호환이면 WEB_PREVIEW_VIDEO를 가리킨다.
    video_asset_id: str
    video_url: str

    # 원본이 비호환일 때 생성된 짧은 preview asset
    # 생성 실패 또는 원본 재생 가능 시 None일 수 있다.
    preview_video_asset_id: str | None = None
    preview_url: str | None = None

    # RAW_VIDEO_PLAYABLE
    # PREVIEW_READY
    # PREVIEW_FAILED_RAW_ONLY
    preview_status: str

    is_browser_playable: bool
    browser_playability_reason: str

    upload: UploadedVideoInfo


class EditStateMatch(BaseModel):
    match_id: str
    project_id: str
    home_team: str | None = None
    away_team: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    duration_sec: float | None = None


class EditStateVideo(BaseModel):
    url: str | None = None
    asset_id: str | None = None
    asset_type: str | None = None
    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None


class EditStateFilters(BaseModel):
    # 프론트 필터 버튼과 호환되는 category 값
    eventTypes: list[str]

    # 백엔드 내부 표준 label. 특히 free_kick은 여기에서 유지한다.
    backendLabels: list[str]


class StudioEditStateResponse(BaseModel):
    match: EditStateMatch
    video: EditStateVideo
    events: list[FrontendTimelineEvent]
    players: list[FrontendPlayer]
    filters: EditStateFilters

