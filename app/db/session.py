from collections.abc import Generator

from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

# 중요:
# SQLAlchemy가 relationship("MediaAsset"), relationship("Artifact") 같은
# 문자열 기반 관계를 해석하려면 모든 모델 클래스가 먼저 registry에 등록되어야 한다.
# 이 import는 사용하지 않는 것처럼 보여도 반드시 필요하다.
from app.db import models  # noqa: F401


settings = get_settings()
_database_backend = make_url(settings.DATABASE_URL).get_backend_name()


def _queue_engine_kwargs(*, pool_size: int, max_overflow: int) -> dict:
    """Build QueuePool options without breaking SQLite-based local/tests."""

    kwargs: dict = {
        "echo": settings.DB_ECHO,
        "pool_pre_ping": True,
    }
    if _database_backend != "sqlite":
        kwargs.update(
            {
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": settings.DB_POOL_TIMEOUT_SECONDS,
                "pool_recycle": settings.DB_POOL_RECYCLE_SECONDS,
                "pool_use_lifo": True,
            }
        )
    return kwargs


# Request/API pool.  This is the pool FastAPI dependencies should use.
engine = create_engine(
    settings.DATABASE_URL,
    **_queue_engine_kwargs(
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
    ),
)

# Long-running background executors deliberately use a separate pool.  A scene
# AI task may keep a Session alive while an immutable runtime is executing; that
# must not consume request-serving pool capacity.  SQLite (especially :memory:)
# must share the same engine so tests still see the same database.
if _database_backend == "sqlite":
    background_engine = engine
    advisory_lock_engine = engine
else:
    background_engine = create_engine(
        settings.DATABASE_URL,
        **_queue_engine_kwargs(
            pool_size=settings.DB_BACKGROUND_POOL_SIZE,
            max_overflow=settings.DB_BACKGROUND_MAX_OVERFLOW,
        ),
    )

    # PostgreSQL advisory locks are intentionally held for the duration of a
    # tracking subprocess.  NullPool keeps that physical lock connection out of
    # both the API pool and the general background-worker pool.
    advisory_lock_engine = create_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        poolclass=NullPool,
    )


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

BackgroundSessionLocal = sessionmaker(
    bind=background_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for request-scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
