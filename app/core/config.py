import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "KickClip Studio Backend"
    SERVICE_NAME: str = "kickclip-backend"
    VERSION: str = "0.1.0"
    ENV: str = Field(default="local", description="local, dev, prod 등 실행 환경")
    DEBUG: bool = Field(default=True, validation_alias="KICKCLIP_DEBUG")

    API_V1_PREFIX: str = "/api/v1"

    # 중요:
    # pydantic-settings는 list[str] 환경변수를 JSON으로 먼저 파싱하려고 한다.
    # 그래서 환경변수에서는 문자열로 받고, 아래 cors_origins property에서 직접 파싱한다.
    BACKEND_CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:5173"
    )

    STORAGE_ROOT: str = "storage"

    DATABASE_URL: str = "postgresql+psycopg://kickclip:kickclip@localhost:5432/kickclip"
    DB_ECHO: bool = False

    QWEN_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_EXPORT_SAMPLE_FRAMES: int = 12
    QWEN_MAX_NEW_TOKENS: int = 384

    @property
    def cors_origins(self) -> list[str]:
        raw = self.BACKEND_CORS_ORIGINS.strip()

        if not raw:
            return []

        # JSON 배열 형식도 허용
        # 예: ["http://localhost:3000", "http://localhost:5173"]
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(origin).strip() for origin in parsed if str(origin).strip()]
            except json.JSONDecodeError:
                pass

        # 쉼표 구분 문자열 허용
        # 예: http://localhost:3000,http://localhost:5173
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
