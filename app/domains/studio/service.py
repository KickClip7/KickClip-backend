from datetime import datetime

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.domains.match.model import Match
from app.domains.match.repository import MatchRepository
from app.domains.media.metadata_extractor import extract_video_metadata
from app.domains.media.model import MediaAsset
from app.domains.media.repository import MediaAssetRepository
from app.domains.media.video_compatibility import check_browser_playability
from app.domains.media.video_transcoder import create_browser_preview_clip
from app.domains.player.repository import PlayerRepository
from app.domains.project.repository import ProjectRepository
from app.domains.studio.schema import (
    EditStateFilters,
    EditStateMatch,
    EditStateVideo,
    StudioEditStateResponse,
    UploadedVideoInfo,
    UploadMatchVideoResponse,
    VideoCodecInfo,
)
from app.domains.timeline.fusion import (
    BACKEND_EVENT_LABELS,
    FRONTEND_EVENT_TYPES,
    build_frontend_event,
    build_frontend_player,
)
from app.domains.timeline.repository import TimelineEventRepository
from app.storage.local_storage import LocalStorage
from app.storage.workspace import (
    get_match_preview_video_subdir,
    get_match_raw_video_subdir,
)


PREVIEW_CLIP_DURATION_SEC = 30


class StudioService:
    """BFF style service for frontend studio flow.

    4회차 보강 구조:
    - RAW_VIDEO: 원본 전체 영상. AI 분석/최종 렌더링용.
    - WEB_PREVIEW_VIDEO: 브라우저 재생용 짧은 30초 H.264 preview clip.
    """

    def __init__(self, db: Session):
        self.db = db
        self.project_repository = ProjectRepository(db)
        self.match_repository = MatchRepository(db)
        self.media_asset_repository = MediaAssetRepository(db)
        self.timeline_event_repository = TimelineEventRepository(db)
        self.player_repository = PlayerRepository(db)
        self.storage = LocalStorage()

    def upload_match_video(
        self,
        file: UploadFile,
        project_title: str | None = None,
        project_description: str | None = None,
        owner_id: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        home_score: int | None = None,
        away_score: int | None = None,
        match_date: datetime | None = None,
        competition: str | None = None,
        season: str | None = None,
        duration_sec: float | None = None,
        match_metadata: dict | None = None,
    ) -> UploadMatchVideoResponse:
        created_file_paths: list[str] = []

        try:
            title = project_title or self._build_project_title(home_team, away_team)

            project = self.project_repository.create(
                owner_id=owner_id,
                title=title,
                description=project_description,
                status="UPLOADED",
            )

            match = self.match_repository.create(
                project_id=project.project_id,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                match_date=match_date,
                competition=competition,
                season=season,
                duration_sec=duration_sec,
                metadata_=match_metadata or {},
            )

            # 1. 원본 전체 영상 저장
            stored_raw = self.storage.save_upload_file(
                upload_file=file,
                subdir=get_match_raw_video_subdir(match.match_id),
            )
            created_file_paths.append(stored_raw.relative_path)

            # 2. 원본 metadata + codec 추출
            raw_metadata = extract_video_metadata(stored_raw.absolute_path)

            final_duration_sec = duration_sec
            if final_duration_sec is None:
                final_duration_sec = raw_metadata.get("duration_sec")

            if match.duration_sec is None:
                match.duration_sec = final_duration_sec

            # 3. RAW_VIDEO asset 저장
            raw_asset = self.media_asset_repository.create(
                match_id=match.match_id,
                asset_type="RAW_VIDEO",
                file_path=stored_raw.relative_path,
                original_filename=file.filename,
                mime_type=file.content_type or "application/octet-stream",
                duration_sec=final_duration_sec,
                fps=raw_metadata.get("fps"),
                width=raw_metadata.get("width"),
                height=raw_metadata.get("height"),
                size_bytes=raw_metadata.get("size_bytes") or stored_raw.size_bytes,
            )

            # 4. 브라우저 재생 가능 여부 판단
            playability = check_browser_playability(
                metadata=raw_metadata,
                mime_type=file.content_type,
                filename=file.filename,
            )

            # 기본값: 원본을 대표 video asset으로 사용
            video_asset = raw_asset
            preview_asset = None
            preview_status = "RAW_VIDEO_PLAYABLE"

            # 5. 비호환이면 전체 변환이 아니라 30초 preview clip만 생성
            if not playability.is_browser_playable:
                preview_status = "PREVIEW_FAILED_RAW_ONLY"

                preview_clip = create_browser_preview_clip(
                    input_path=stored_raw.absolute_path,
                    output_dir=(
                        self.storage.storage_root
                        / get_match_preview_video_subdir(match.match_id)
                    ),
                    original_filename=file.filename,
                    start_sec=0,
                    duration_sec=PREVIEW_CLIP_DURATION_SEC,
                )

                if preview_clip is not None:
                    created_file_paths.append(preview_clip.relative_path)
                    preview_metadata = extract_video_metadata(preview_clip.absolute_path)

                    preview_asset = self.media_asset_repository.create(
                        match_id=match.match_id,
                        asset_type="WEB_PREVIEW_VIDEO",
                        file_path=preview_clip.relative_path,
                        original_filename=preview_clip.filename,
                        mime_type="video/mp4",
                        duration_sec=preview_metadata.get("duration_sec")
                        or min(
                            PREVIEW_CLIP_DURATION_SEC,
                            final_duration_sec or PREVIEW_CLIP_DURATION_SEC,
                        ),
                        fps=preview_metadata.get("fps") or raw_metadata.get("fps"),
                        width=preview_metadata.get("width") or raw_metadata.get("width"),
                        height=preview_metadata.get("height") or raw_metadata.get("height"),
                        size_bytes=(
                            preview_metadata.get("size_bytes")
                            or preview_clip.absolute_path.stat().st_size
                        ),
                    )

                    video_asset = preview_asset
                    preview_status = "PREVIEW_READY"

            self.db.commit()
            self.db.refresh(project)
            self.db.refresh(match)
            self.db.refresh(raw_asset)

            if preview_asset is not None:
                self.db.refresh(preview_asset)

            return UploadMatchVideoResponse(
                project_id=project.project_id,
                match_id=match.match_id,
                raw_video_asset_id=raw_asset.asset_id,
                video_asset_id=video_asset.asset_id,
                video_url=f"/api/v1/media/{video_asset.asset_id}/stream",
                preview_video_asset_id=(
                    preview_asset.asset_id if preview_asset is not None else None
                ),
                preview_url=(
                    f"/api/v1/media/{preview_asset.asset_id}/stream"
                    if preview_asset is not None
                    else None
                ),
                preview_status=preview_status,
                is_browser_playable=playability.is_browser_playable,
                browser_playability_reason=playability.reason,
                upload=UploadedVideoInfo(
                    filename=file.filename,
                    content_type=file.content_type,
                    duration_sec=raw_asset.duration_sec,
                    fps=raw_asset.fps,
                    width=raw_asset.width,
                    height=raw_asset.height,
                    size_bytes=raw_asset.size_bytes,
                    codec=VideoCodecInfo(
                        codec_name=raw_metadata.get("codec_name"),
                        codec_tag_string=raw_metadata.get("codec_tag_string"),
                        pix_fmt=raw_metadata.get("pix_fmt"),
                        format_name=raw_metadata.get("format_name"),
                    ),
                ),
            )

        except Exception:
            self.db.rollback()

            for path in created_file_paths:
                self.storage.delete_file_if_exists(path)

            raise

    def get_edit_state(
        self,
        match_id: str,
        *,
        event_weights: dict[str, float] | None = None,
    ) -> StudioEditStateResponse | None:
        """Build the edit screen initial state for KickClip Studio.

        이 메서드는 DB 표준 모델인 Match / MediaAsset / TimelineEvent / Player를
        프론트 더미 데이터와 호환되는 events / players 형태로 변환한다.
        저장된 내부 label은 그대로 유지하고, API 응답 경계에서만 free_kick -> freekick
        변환을 적용한다.
        """
        match = self.match_repository.get_by_id(match_id)
        if match is None:
            return None

        video_asset = self._select_representative_video_asset(match_id)
        events = self.timeline_event_repository.list_by_match(match_id)
        players = self.player_repository.list_by_match(match_id)

        return StudioEditStateResponse(
            match=self._build_edit_state_match(match),
            video=self._build_edit_state_video(video_asset),
            events=[
                build_frontend_event(
                    event,
                    match_duration_sec=match.duration_sec,
                    event_weights=event_weights,
                )
                for event in events
            ],
            players=[build_frontend_player(player) for player in players],
            filters=EditStateFilters(
                eventTypes=FRONTEND_EVENT_TYPES,
                backendLabels=BACKEND_EVENT_LABELS,
            ),
        )

    def _select_representative_video_asset(
        self,
        match_id: str,
    ) -> MediaAsset | None:
        """Prefer WEB_PREVIEW_VIDEO for browser playback, fallback to RAW_VIDEO."""
        assets = self.media_asset_repository.list_by_match(match_id)

        preview_asset = next(
            (asset for asset in assets if asset.asset_type == "WEB_PREVIEW_VIDEO"),
            None,
        )
        if preview_asset is not None:
            return preview_asset

        return next(
            (asset for asset in assets if asset.asset_type == "RAW_VIDEO"),
            None,
        )

    @staticmethod
    def _build_edit_state_match(match: Match) -> EditStateMatch:
        return EditStateMatch(
            match_id=match.match_id,
            project_id=match.project_id,
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=match.home_score,
            away_score=match.away_score,
            duration_sec=match.duration_sec,
        )

    @staticmethod
    def _build_edit_state_video(asset: MediaAsset | None) -> EditStateVideo:
        if asset is None:
            return EditStateVideo()

        return EditStateVideo(
            url=f"/api/v1/media/{asset.asset_id}/stream",
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            duration_sec=asset.duration_sec,
            width=asset.width,
            height=asset.height,
        )

    @staticmethod
    def _build_project_title(
        home_team: str | None,
        away_team: str | None,
    ) -> str:
        if home_team and away_team:
            return f"{home_team} vs {away_team}"
        if home_team:
            return f"{home_team} 경기 영상"
        if away_team:
            return f"{away_team} 경기 영상"
        return "KickClip 업로드 프로젝트"
