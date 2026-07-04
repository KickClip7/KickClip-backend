from sqlalchemy.orm import Session

from app.domains.player.model import Player, PlayerTrack
from app.domains.player.repository import PlayerRepository, PlayerTrackRepository
from app.domains.player.schema import PlayerCreate, PlayerTrackCreate


class PlayerService:
    def __init__(self, db: Session):
        self.db = db
        self.player_repository = PlayerRepository(db)
        self.track_repository = PlayerTrackRepository(db)

    def create_player(self, data: PlayerCreate) -> Player:
        payload = data.model_dump(exclude_none=True)
        payload["metadata_"] = payload.pop("metadata", {})

        player = self.player_repository.create(**payload)
        self.db.commit()
        self.db.refresh(player)
        return player

    def list_match_players(self, match_id: str) -> list[Player]:
        return self.player_repository.list_by_match(match_id)

    def list_player_tracks(self, player_id: str) -> list[PlayerTrack]:
        return self.track_repository.list_by_player(player_id)

    def replace_players_and_tracks_for_match(
        self,
        match_id: str,
        players: list[PlayerCreate],
        tracks: list[PlayerTrackCreate],
        commit: bool = True,
    ) -> tuple[list[Player], list[PlayerTrack]]:
        self.track_repository.delete_by_match(match_id)
        self.player_repository.delete_by_match(match_id)

        created_players: list[Player] = []
        for player_data in players:
            payload = player_data.model_dump(exclude_none=True)
            payload["metadata_"] = payload.pop("metadata", {})
            created_players.append(self.player_repository.create(**payload))

        track_payloads: list[dict] = []
        for track_data in tracks:
            payload = track_data.model_dump()
            payload["metadata_"] = payload.pop("metadata", {})
            track_payloads.append(payload)

        created_tracks = self.track_repository.bulk_create(track_payloads)

        if commit:
            self.db.commit()
            for player in created_players:
                self.db.refresh(player)
            for track in created_tracks:
                self.db.refresh(track)

        return created_players, created_tracks