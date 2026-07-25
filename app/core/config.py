"""Application configuration.

환경변수와 프로젝트 루트의 .env 파일에서 애플리케이션 설정을 불러온다.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """KickClip 애플리케이션 설정."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    PROJECT_NAME: str = "KickClip Studio Backend"
    SERVICE_NAME: str = "kickclip-backend"
    VERSION: str = "0.1.0"

    ENV: Literal["local", "dev", "test", "prod"] = Field(
        default="local",
        description="애플리케이션 실행 환경",
    )
    DEBUG: bool = Field(
        default=True,
        validation_alias="KICKCLIP_DEBUG",
    )

    API_V1_PREFIX: str = "/api/v1"

    # -------------------------------------------------------------------------
    # CORS
    # -------------------------------------------------------------------------
    # pydantic-settings는 list[str] 환경변수를 JSON 배열로 파싱하려고 하므로,
    # 문자열로 받은 뒤 cors_origins 속성에서 직접 변환한다.
    BACKEND_CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:5173"
    )

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------
    STORAGE_ROOT: Path = Path("storage")
    USE_MOCK_DATA: bool = True
    MOCK_TIMELINE_SOURCE_MATCH_ID: str = "korjpn_2026"
    AGENT_DEV_MATCH_ID: str = ""

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    DATABASE_URL: str = (
        "postgresql+psycopg://kickclip:kickclip@localhost:5432/kickclip"
    )
    DB_ECHO: bool = False

    # -------------------------------------------------------------------------
    # Qwen
    # -------------------------------------------------------------------------
    QWEN_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_EXPORT_SAMPLE_FRAMES: int = Field(default=12, ge=1)
    QWEN_MAX_NEW_TOKENS: int = Field(default=384, ge=1)

    # -------------------------------------------------------------------------
    # OpenAI / LangGraph
    # -------------------------------------------------------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_AGENT_MODEL: str = "gpt-4o-mini"
    OPENAI_AGENT_TIMEOUT_SECONDS: float = Field(default=90.0, gt=0)

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    AUTH_SECRET_KEY: str = "local-development-only-change-me-please-32chars"
    AUTH_ACCESS_TOKEN_MINUTES: int = Field(default=15, ge=1)
    AUTH_REFRESH_TOKEN_DAYS: int = Field(default=30, ge=1)
    MEDIA_SIGNED_URL_MINUTES: int = Field(default=10, ge=1)

    # -------------------------------------------------------------------------
    # Developer mode
    # -------------------------------------------------------------------------
    # 다른 사용자의 리소스에 접근할 수 있는 개발 전용 기능이다.
    # 운영 환경에서는 항상 비활성화해야 한다.
    DEVELOPER_MODE_ENABLED: bool = False
    DEVELOPER_ACCESS_KEY: str = ""

    @property
    def cors_origins(self) -> list[str]:
        """CORS origin 문자열을 정규화된 리스트로 반환한다.

        다음 두 형식을 지원한다.

        1. 쉼표 구분 문자열
           http://localhost:3000,http://localhost:5173

        2. JSON 배열
           ["http://localhost:3000", "http://localhost:5173"]
        """

        raw_origins = self.BACKEND_CORS_ORIGINS.strip()

        if not raw_origins:
            return []

        if raw_origins.startswith("["):
            try:
                parsed_origins = json.loads(raw_origins)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "BACKEND_CORS_ORIGINS의 JSON 배열 형식이 올바르지 않습니다."
                ) from exc

            if not isinstance(parsed_origins, list):
                raise ValueError(
                    "BACKEND_CORS_ORIGINS는 JSON 배열 또는 "
                    "쉼표로 구분된 문자열이어야 합니다."
                )

            origins = parsed_origins
        else:
            origins = raw_origins.split(",")

        normalized_origins: list[str] = []

        for origin in origins:
            normalized_origin = str(origin).strip().rstrip("/")

            if (
                normalized_origin
                and normalized_origin not in normalized_origins
            ):
                normalized_origins.append(normalized_origin)

        return normalized_origins

    @property
    def is_production(self) -> bool:
        """현재 운영 환경인지 반환한다."""

        return self.ENV == "prod"

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """운영 환경에서 위험한 설정이 활성화되지 않도록 검증한다."""

        if not self.is_production:
            return self

        if self.DEBUG:
            raise ValueError(
                "운영 환경에서는 KICKCLIP_DEBUG=false로 설정해야 합니다."
            )

        if self.DEVELOPER_MODE_ENABLED:
            raise ValueError(
                "운영 환경에서는 DEVELOPER_MODE_ENABLED를 "
                "활성화할 수 없습니다."
            )

        insecure_secret_keys = {
            "local-development-only-change-me-please-32chars",
            "replace-with-at-least-32-random-characters",
        }

        if (
            len(self.AUTH_SECRET_KEY) < 32
            or self.AUTH_SECRET_KEY in insecure_secret_keys
        ):
            raise ValueError(
                "운영 환경에서는 32자 이상의 안전한 "
                "AUTH_SECRET_KEY를 설정해야 합니다."
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """애플리케이션 설정을 생성하고 프로세스 내에서 캐싱한다."""

    return Settings()
