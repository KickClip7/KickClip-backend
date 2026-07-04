from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import engine


def init_db() -> None:
    """Create all tables directly.

    운영/협업 환경에서는 Alembic migration을 사용하고,
    이 함수는 로컬 개발 확인용으로만 사용한다.
    """
    Base.metadata.create_all(bind=engine)