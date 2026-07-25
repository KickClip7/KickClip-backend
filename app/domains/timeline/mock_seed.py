from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.auth.model import User
from app.domains.match.mock import build_mock_match_metadata, is_mock_match
from app.domains.match.repository import MatchRepository
from app.domains.project.repository import ProjectRepository
from app.domains.timeline.repository import TimelineEventRepository
from app.domains.timeline.schema import TimelineEventRead


MOCK_SEED_METADATA_KEY = "kickclip_mock_seed"


@dataclass(frozen=True)
class MockTimelineSeedResult:
    project_id: str
    match_id: str
    owner_id: str | None
    event_count: int
    replaced_event_count: int


def build_mock_timeline_rows(
    payload: dict[str, Any],
    *,
    target_match_id: str,
    source_path: Path,
) -> list[dict[str, Any]]:
    """Convert a timeline fixture into repeatable DB rows for one dev match."""

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("목업 timeline JSON의 events 필드는 배열이어야 합니다.")

    rows: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()

    for raw_event in raw_events:
        event = TimelineEventRead.model_validate(raw_event)
        source_event_id = event.timeline_event_id
        if source_event_id in seen_source_ids:
            raise ValueError(
                f"목업 timeline JSON에 중복 event ID가 있습니다: {source_event_id}"
            )
        seen_source_ids.add(source_event_id)

        digest = hashlib.sha256(
            f"{target_match_id}:{source_event_id}".encode("utf-8")
        ).hexdigest()[:24]
        metadata = {
            **event.metadata,
            MOCK_SEED_METADATA_KEY: {
                "source_event_id": source_event_id,
                "source_match_id": event.match_id,
                "source_path": str(source_path),
            },
        }

        rows.append(
            {
                "timeline_event_id": f"mock_evt_{digest}",
                "match_id": target_match_id,
                # Fixture IDs do not point at real artifact/job DB rows.
                "source_artifact_id": None,
                "source_job_id": None,
                "event_type": event.event_type,
                "label": event.label,
                "half": event.half,
                "timestamp_sec": event.timestamp_sec,
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "duration_sec": event.duration_sec,
                "confidence": event.confidence,
                "highlight_score": event.highlight_score,
                "title": event.title,
                "description": event.description,
                "team_name": event.team_name,
                "player_ids": event.player_ids,
                "metadata_": metadata,
            }
        )

    return rows


def seed_mock_timeline_fixture(
    db: Session,
    payload: dict[str, Any],
    *,
    target_match_id: str,
    source_path: Path,
    project_title: str = "KickClip shared mock project",
    owner_id: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    home_score: int | None = None,
    away_score: int | None = None,
) -> MockTimelineSeedResult:
    """Create or refresh a DB-backed mock Match and its timeline events.

    The operation is intentionally idempotent:

    - the project and match IDs are deterministic;
    - only rows marked with ``MOCK_SEED_METADATA_KEY`` are replaced;
    - an existing real match with the same ID is never overwritten.

    The mock match itself receives a separate metadata marker. In local/dev/test,
    ``MOCK_SHARED_ACCESS_ENABLED=true`` lets every authenticated teammate read
    that explicitly seeded match without weakening access control for real data.
    """

    source_path = source_path.resolve()
    rows = build_mock_timeline_rows(
        payload,
        target_match_id=target_match_id,
        source_path=source_path,
    )
    if not rows:
        raise ValueError("목업 timeline event가 1개 이상 필요합니다.")

    project_repository = ProjectRepository(db)
    match_repository = MatchRepository(db)
    timeline_repository = TimelineEventRepository(db)

    owner_id = owner_id or _resolve_single_active_user_id(db)
    project_id = _mock_project_id(target_match_id)
    match = match_repository.get_by_id(target_match_id)

    if match is not None and not is_mock_match(match):
        raise ValueError(
            "동일한 match_id의 실제 Match가 이미 존재하여 목업으로 덮어쓸 수 없습니다: "
            f"{target_match_id}"
        )

    try:
        if match is None:
            project = project_repository.get_by_id(project_id)
            if project is None:
                project = project_repository.create(
                    project_id=project_id,
                    owner_id=owner_id,
                    title=project_title,
                    description=(
                        "Local/shared development fixture. "
                        "Do not use as production match data."
                    ),
                    status="MOCK",
                )
            else:
                project.title = project_title
                if owner_id is not None:
                    project.owner_id = owner_id

            match = match_repository.create(
                match_id=target_match_id,
                project_id=project.project_id,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                duration_sec=_infer_duration(rows),
                metadata_=build_mock_match_metadata(
                    source_path=str(source_path),
                    event_count=len(rows),
                ),
            )
        else:
            project_id = match.project_id
            match.home_team = home_team
            match.away_team = away_team
            match.home_score = home_score
            match.away_score = away_score
            match.duration_sec = _infer_duration(rows)
            match.metadata_ = {
                **(match.metadata_ or {}),
                **build_mock_match_metadata(
                    source_path=str(source_path),
                    event_count=len(rows),
                ),
            }

            project = project_repository.get_by_id(match.project_id)
            if project is not None:
                project.title = project_title
                if owner_id is not None:
                    project.owner_id = owner_id

        replaced_event_count = 0
        for event in timeline_repository.list_by_match(target_match_id):
            if (event.metadata_ or {}).get(MOCK_SEED_METADATA_KEY):
                db.delete(event)
                replaced_event_count += 1

        db.flush()
        timeline_repository.bulk_create(rows)
        db.commit()
        db.refresh(match)

        return MockTimelineSeedResult(
            project_id=project_id,
            match_id=target_match_id,
            owner_id=owner_id,
            event_count=len(rows),
            replaced_event_count=replaced_event_count,
        )
    except Exception:
        db.rollback()
        raise


def _resolve_single_active_user_id(db: Session) -> str | None:
    """Use the only active local account as owner when it is unambiguous."""

    stmt = select(User.user_id).where(User.is_active.is_(True)).limit(2)
    user_ids = list(db.scalars(stmt).all())
    return user_ids[0] if len(user_ids) == 1 else None


def _mock_project_id(match_id: str) -> str:
    digest = hashlib.sha256(match_id.encode("utf-8")).hexdigest()[:24]
    return f"proj_mock_{digest}"


def _infer_duration(rows: list[dict[str, Any]]) -> float:
    return max(float(row["end_sec"]) for row in rows)
