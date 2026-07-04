from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
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
) -> PlayersResponse:
    repository = PlayerRepository(db)
    players = repository.list_by_match(match_id)
    frontend_players = [build_frontend_player(player) for player in players]

    return PlayersResponse(
        match_id=match_id,
        players=frontend_players,
        count=len(frontend_players),
    )
