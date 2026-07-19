from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.auth.access import require_match_access
from app.domains.auth.dependencies import get_current_user
from app.domains.auth.model import User
from app.domains.player.repository import PlayerRepository
from app.domains.player.schema import PlayersResponse
from app.domains.timeline.fusion import build_frontend_player


router = APIRouter()


@router.get(
    "/matches/{match_id}/players",
    response_model=PlayersResponse,
    summary="경기 선수 목록 조회",
)
def list_match_players(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PlayersResponse:
    require_match_access(db, match_id, current_user)
    repository = PlayerRepository(db)
    players = repository.list_by_match(match_id)
    frontend_players = [build_frontend_player(player) for player in players]

    return PlayersResponse(
        match_id=match_id,
        players=frontend_players,
        count=len(frontend_players),
    )
