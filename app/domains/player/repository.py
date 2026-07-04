from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.domains.player.model import Player, PlayerTrack


class PlayerRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Player:
        player = Player(**kwargs)
        self.db.add(player)
        self.db.flush()
        return player

    def get_by_id(self, player_id: str) -> Player | None:
        stmt = select(Player).where(Player.player_id == player_id)
        return self.db.scalar(stmt)

    def update_profile(
        self,
        *,
        player: Player,
        display_name: str,
        identity_status: str,
        profile_source: str | None = None,
        metadata_: dict | None = None,
    ) -> Player:
        player.display_name = display_name
        player.identity_status = identity_status
        player.profile_source = profile_source
        if metadata_ is not None:
            player.metadata_ = metadata_
        self.db.flush()
        return player

    def list_by_match(self, match_id: str) -> list[Player]:
        stmt = (
            select(Player)
            .where(Player.match_id == match_id)
            .order_by(Player.team_name.asc().nulls_last(), Player.number.asc().nulls_last())
        )
        return list(self.db.scalars(stmt).all())

    def delete_by_match(self, match_id: str) -> int:
        stmt = delete(Player).where(Player.match_id == match_id)
        result = self.db.execute(stmt)
        return int(result.rowcount or 0)


class PlayerTrackRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> PlayerTrack:
        track = PlayerTrack(**kwargs)
        self.db.add(track)
        self.db.flush()
        return track

    def bulk_create(self, rows: list[dict]) -> list[PlayerTrack]:
        tracks: list[PlayerTrack] = []

        for row in rows:
            track = PlayerTrack(**row)
            self.db.add(track)
            tracks.append(track)

        self.db.flush()
        return tracks

    def list_by_match(self, match_id: str) -> list[PlayerTrack]:
        stmt = (
            select(PlayerTrack)
            .where(PlayerTrack.match_id == match_id)
            .order_by(PlayerTrack.start_sec.asc())
        )
        return list(self.db.scalars(stmt).all())

    def list_by_player(self, player_id: str) -> list[PlayerTrack]:
        stmt = (
            select(PlayerTrack)
            .where(PlayerTrack.player_id == player_id)
            .order_by(PlayerTrack.start_sec.asc())
        )
        return list(self.db.scalars(stmt).all())

    def delete_by_match(self, match_id: str) -> int:
        stmt = delete(PlayerTrack).where(PlayerTrack.match_id == match_id)
        result = self.db.execute(stmt)
        return int(result.rowcount or 0)

    def delete_by_source_job(self, source_job_id: str) -> int:
        stmt = delete(PlayerTrack).where(PlayerTrack.source_job_id == source_job_id)
        result = self.db.execute(stmt)
        return int(result.rowcount or 0)