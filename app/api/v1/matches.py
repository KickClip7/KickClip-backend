from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.match.schema import MatchRead


router = APIRouter()


@router.get(
    "/{match_id}",
    response_model=MatchRead,
    summary="경기 단건 조회",
)
def get_match(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MatchRead:
    return require_match_access(db, match_id, current_user)
