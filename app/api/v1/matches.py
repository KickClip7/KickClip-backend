from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.match.schema import MatchRead
from app.domains.match.service import MatchService


router = APIRouter()


@router.get(
    "/{match_id}",
    response_model=MatchRead,
    summary="경기 단건 조회",
)
def get_match(
    match_id: str,
    db: Session = Depends(get_db),
) -> MatchRead:
    service = MatchService(db)
    match = service.get_match(match_id)

    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found",
        )

    return match