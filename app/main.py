from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the KickClip FastAPI application.

    1회차에서는 DB, AI 모델, 렌더링 기능을 붙이지 않고,
    이후 도메인/AI task/router를 자연스럽게 확장할 수 있는 앱 골격만 만든다.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="KickClip Studio Backend API",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {
            "service": settings.SERVICE_NAME,
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_url": f"{settings.API_V1_PREFIX}/health",
        }

    return app


app = create_app()
