"""KickClip FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings


class ServiceInfoResponse(BaseModel):
    """루트 엔드포인트 응답 스키마."""

    model_config = ConfigDict(frozen=True)

    service: str
    version: str
    environment: str
    docs_url: str
    health_url: str


def configure_middleware(
    application: FastAPI,
    settings: Settings,
) -> None:
    """애플리케이션 공통 미들웨어를 등록한다."""

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
            "HEAD",
        ],
        allow_headers=["*"],
    )


def configure_routes(
    application: FastAPI,
    settings: Settings,
) -> None:
    """API 및 공통 엔드포인트를 등록한다."""

    application.include_router(
        api_router,
        prefix=settings.API_V1_PREFIX,
    )

    @application.get(
        "/",
        response_model=ServiceInfoResponse,
        tags=["Root"],
        summary="서비스 정보 조회",
        description=(
            "KickClip 백엔드 서비스 정보와 주요 API 경로를 반환합니다."
        ),
    )
    async def get_service_info() -> ServiceInfoResponse:
        return ServiceInfoResponse(
            service=settings.SERVICE_NAME,
            version=settings.VERSION,
            environment=settings.ENV,
            docs_url="/docs",
            health_url=f"{settings.API_V1_PREFIX}/health",
        )


def create_app() -> FastAPI:
    """KickClip FastAPI 애플리케이션을 생성하고 구성한다.

    애플리케이션 팩토리 패턴을 사용하여 설정, 미들웨어,
    라우터 등록 책임을 분리한다.

    이후 데이터베이스 연결, AI 모델 초기화, 렌더링 리소스 등의
    생명주기 관리가 필요한 경우 lifespan을 추가할 수 있다.
    """

    settings = get_settings()

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="KickClip Studio Backend API",
        debug=settings.DEBUG,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 미들웨어, 의존성, lifespan 등에서 공통 설정을 참조할 수 있도록 저장한다.
    application.state.settings = settings

    configure_middleware(application, settings)
    configure_routes(application, settings)

    return application


app = create_app()