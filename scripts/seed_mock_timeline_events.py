"""고정 개발용 경기에 timeline fixture를 DB 이벤트로 시드한다.

기본 입력은 storage/matches/{MOCK_TIMELINE_SOURCE_MATCH_ID}/timeline_events.json이고,
대상 경기는 --match-id 또는 AGENT_DEV_MATCH_ID로 지정한다.

사용 예시:
    python scripts/seed_mock_timeline_events.py
    python scripts/seed_mock_timeline_events.py --match-id match_abc123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.paths import get_storage_root
from app.db.session import SessionLocal
from app.domains.match.repository import MatchRepository
from app.domains.timeline.mock_seed import (
    MOCK_SEED_METADATA_KEY,
    build_mock_timeline_rows,
)
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository
from app.storage.workspace import get_match_timeline_events_path


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_input = (
        get_storage_root()
        / get_match_timeline_events_path(settings.MOCK_TIMELINE_SOURCE_MATCH_ID)
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-id",
        default=settings.AGENT_DEV_MATCH_ID or None,
        help="시드 대상 경기 ID. 생략하면 AGENT_DEV_MATCH_ID를 사용합니다.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="timeline fixture JSON 경로",
    )
    return parser.parse_args()


def is_mock_seed_event(event: TimelineEvent) -> bool:
    return bool((event.metadata_ or {}).get(MOCK_SEED_METADATA_KEY))


def main() -> None:
    args = parse_args()
    if not args.match_id:
        raise SystemExit(
            "--match-id 또는 .env의 AGENT_DEV_MATCH_ID를 지정해야 합니다."
        )
    if not args.input.is_file():
        raise SystemExit(f"timeline fixture를 찾을 수 없습니다: {args.input}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_mock_timeline_rows(
        payload,
        target_match_id=args.match_id,
        source_path=args.input,
    )

    with SessionLocal() as db:
        match = MatchRepository(db).get_by_id(args.match_id)
        if match is None:
            raise SystemExit(f"DB에서 대상 경기를 찾을 수 없습니다: {args.match_id}")

        repository = TimelineEventRepository(db)
        existing_events = repository.list_by_match(args.match_id)
        old_mock_events = [
            event for event in existing_events if is_mock_seed_event(event)
        ]

        try:
            for event in old_mock_events:
                db.delete(event)
            db.flush()

            repository.bulk_create(rows)
            db.commit()
        except Exception:
            db.rollback()
            raise

    print(f"고정 개발 경기: {args.match_id}")
    print(f"목업 원본: {args.input}")
    print(f"기존 목업 이벤트 교체: {len(old_mock_events)}개")
    print(f"DB 시드 완료: {len(rows)}개")
    max_event_end_sec = max((row["end_sec"] for row in rows), default=0.0)
    if match.duration_sec and max_event_end_sec > match.duration_sec:
        print(
            "경고: 목업 이벤트의 마지막 시각"
            f"({max_event_end_sec:.3f}초)이 영상 길이"
            f"({match.duration_sec:.3f}초)를 초과합니다. "
            "에이전트 로직 개발에는 사용할 수 있지만 재생/렌더링에는 맞지 않습니다."
        )


if __name__ == "__main__":
    main()
