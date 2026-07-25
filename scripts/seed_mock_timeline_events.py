"""Shared mock match를 로컬 개발 DB에 시드한다.

저장소에는 작은 timeline fixture만 공유하고, 영상은 각 팀원이 로컬의
``test_data.mp4``를 사용한다. 스크립트는 다음을 반복 실행 가능하게 만든다.

1. deterministic mock Project / Match 생성 또는 갱신
2. mock TimelineEvent 교체
3. 로컬 영상을 STORAGE_ROOT 아래 hard link 또는 copy로 배치
4. RAW_VIDEO MediaAsset 생성 또는 갱신
5. 영상 길이가 모든 이벤트를 포함하는지 검증

사용 예시:
    python scripts/seed_mock_timeline_events.py --match-id korjpn_2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.paths import get_storage_root
from app.db.session import SessionLocal
from app.domains.media.mock_seed import seed_mock_video_asset
from app.domains.timeline.mock_seed import seed_mock_timeline_fixture
from app.storage.workspace import get_match_timeline_events_path


DEFAULT_MATCH_ID = "korjpn_2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", default=DEFAULT_MATCH_ID)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="TimelineEvent만 시드한다. 에이전트 렌더링 개발에는 권장하지 않는다.",
    )
    parser.add_argument(
        "--video-link-mode",
        choices=["auto", "hardlink", "copy"],
        default=None,
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
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"목업 timeline fixture가 없습니다: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    required_duration_sec = _max_event_end_sec(payload)

    with SessionLocal() as db:
        timeline_result = seed_mock_timeline_fixture(
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

    video_result = None
    if not args.skip_video:
        video_path = args.video_path or settings.MOCK_VIDEO_SOURCE_PATH
        video_link_mode = args.video_link_mode or settings.MOCK_VIDEO_LINK_MODE
        with SessionLocal() as db:
            video_result = seed_mock_video_asset(
                db,
                match_id=args.match_id,
                source_path=video_path,
                required_duration_sec=required_duration_sec,
                link_mode=video_link_mode,
            )

    print("KickClip 공유 mock match 시드 완료")
    print(f"Fixture       : {input_path}")
    print(f"Project ID    : {timeline_result.project_id}")
    print(f"Match ID      : {timeline_result.match_id}")
    print(f"Owner ID      : {timeline_result.owner_id}")
    print(f"Events        : {timeline_result.event_count}")
    print(f"Replaced      : {timeline_result.replaced_event_count}")
    print(f"Max event end : {required_duration_sec:.3f}s")

    if video_result is None:
        print("Video          : SKIPPED")
    else:
        print(f"Video source   : {video_result.source_path}")
        print(f"Video stored   : {video_result.stored_path}")
        print(f"Materialized   : {video_result.materialization}")
        print(f"Video asset ID : {video_result.asset_id}")
        print(f"Video duration : {video_result.duration_sec:.3f}s")
        print(
            "Video geometry : "
            f"{video_result.width}x{video_result.height} @ {video_result.fps}fps"
        )

    print()
    print("프론트 VITE_AGENT_DEV_MATCH_ID도 다음 값으로 설정하세요:")
    print(f"  {timeline_result.match_id}")


def _max_event_end_sec(payload: dict[str, Any]) -> float:
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("목업 timeline fixture에 events가 없습니다.")
    try:
        return max(float(event["end_sec"]) for event in events)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("모든 목업 event에는 유효한 end_sec가 필요합니다.") from exc


if __name__ == "__main__":
    main()
