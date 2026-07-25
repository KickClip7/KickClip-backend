from sqlalchemy import BigInteger, Float, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class MediaAsset(Base, TimestampMixin):
    __tablename__ = "media_assets"
    __table_args__ = (
        Index(
            "uq_media_assets_match_singleton_video",
            "match_id",
            "asset_type",
            unique=True,
            postgresql_where=text(
                "asset_type IN ('RAW_VIDEO', 'WEB_PREVIEW_VIDEO')"
            ),
            sqlite_where=text(
                "asset_type IN ('RAW_VIDEO', 'WEB_PREVIEW_VIDEO')"
            ),
        ),
    )

    asset_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("asset"),
    )
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    asset_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 2GB 이상 영상 파일도 저장 가능하도록 BigInteger 사용
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    match = relationship("Match", back_populates="media_assets")
