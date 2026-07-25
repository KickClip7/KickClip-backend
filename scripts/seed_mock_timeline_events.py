"""Shared mock timeline fixture를 로컬 개발 DB에 시드한다.

팀 저장소에는 작은 fixture 파일만 공유한다.

    storage/matches/korjpn_2026/timeline_events.json

각 팀원은 DB 마이그레이션과 회원가입을 마친 뒤 이 스크립트를 한 번 실행한다.
스크립트는 deterministic Project/Match를 만들고, 이전에 시드된 mock event만 교체한다.
실제 Match나 실제 timeline event는 덮어쓰지 않는다.

사용 예시:
    python scripts/seed_mock_timeline_events.py --match-id korjpn_2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.paths import get_storage_root
from app.db.session import SessionLocal
from app.domains.timeline.mock_seed import seed_mock_timeline_fixture
from app.storage.workspace import get_match_timeline_events_path


DEFAULT_MATCH_ID = "korjpn_2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", default=DEFAULT_MATCH_ID)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Timeline fixture JSON 경로. 생략하면 "
            "storage/matches/{match_id}/timeline_events.json을 사용한다."
        ),
    )
    parser.add_argument("--owner-user-id", default=None)
    parser.add_argument("--project-title", default="KickClip 한일전 공유 목업")
    parser.add_argument("--home-team", default="대한민국")
    parser.add_argument("--away-team", default="일본")
    parser.add_argument("--home-score", type=int, default=2)
    parser.add_argument("--away-score", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if settings.is_production:
        raise RuntimeError("운영 환경에서는 목업 DB 시드를 실행할 수 없습니다.")
    if not settings.USE_MOCK_DATA:
        raise RuntimeError(
            ".env에서 USE_MOCK_DATA=true를 설정한 뒤 다시 실행하세요."
        )

    input_path = args.input or (
        get_storage_root() / get_match_timeline_events_path(args.match_id)
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"목업 timeline fixture가 없습니다: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))

    with SessionLocal() as db:
        result = seed_mock_timeline_fixture(
            db,
            payload,
            target_match_id=args.match_id,
            source_path=input_path,
            project_title=args.project_title,
            owner_id=args.owner_user_id,
            home_team=args.home_team,
            away_team=args.away_team,
            home_score=args.home_score,
            away_score=args.away_score,
        )

    print("KickClip mock timeline DB 시드 완료")
    print(f"Fixture     : {input_path}")
    print(f"Project ID  : {result.project_id}")
    print(f"Match ID    : {result.match_id}")
    print(f"Owner ID    : {result.owner_id}")
    print(f"Events      : {result.event_count}")
    print(f"Replaced    : {result.replaced_event_count}")
    print()
    print("프론트 활성 match_id도 동일하게 설정하세요:")
    print(f"  {result.match_id}")


if __name__ == "__main__":
    main()
