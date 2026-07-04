from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

# 중요:
# SQLAlchemy가 relationship("MediaAsset"), relationship("Artifact") 같은
# 문자열 기반 관계를 해석하려면 모든 모델 클래스가 먼저 registry에 등록되어야 한다.
# 이 import는 사용하지 않는 것처럼 보여도 반드시 필요하다.
from app.db import models  # noqa: F401


settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()