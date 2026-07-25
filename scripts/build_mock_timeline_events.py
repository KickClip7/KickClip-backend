"""한일전 raw action spotting JSON을 TimelineEventRead 스키마의 목업 timeline_events.json으로 변환한다.

raw JSON 파일은 팀원마다 로컬 경로가 다르므로 --input은 필수 인자다(하드코딩된 기본 경로 없음).

사용 예시:
    # JSON fixture만 생성
    python scripts/build_mock_timeline_events.py --input path/to/events.json --match-id korjpn_2026

    # JSON fixture 생성 + Project/Match/TimelineEvent DB 시드
    python scripts/build_mock_timeline_events.py --input path/to/events.json --match-id korjpn_2026 --seed-db
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.paths import get_storage_root
from app.domains.action_spotting.schema import ActionSpottingEventsResponse
from app.domains.timeline.schema import TimelineEventRead
from app.storage.workspace import get_match_timeline_events_path

DEFAULT_MATCH_ID = "korjpn_2026"
ANALYSIS_JOB_ID = "job_actionspotting_v1"
SOURCE_ARTIFACT_ID_TEMPLATE = "video_{match_id}"

# label -> (pre_offset_sec, post_offset_sec, class_weight)
CLASS_CONFIG: dict[str, tuple[float, float, float]] = {
    "Goal": (7.0, 13.0, 1.0),
    "Shot": (5.0, 4.0, 0.45),
    "Card": (3.0, 8.0, 0.7),
    "Corner": (5.0, 6.0, 0.5),
    "Penalty": (8.0, 10.0, 0.75),
    "Substitution": (3.0, 5.0, 0.3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="원본 action spotting raw JSON 경로. 로컬 파일 위치라 팀원마다 다르므로 기본값 없이 항상 직접 지정해야 한다.",
    )
    parser.add_argument("--match-id", type=str, default=DEFAULT_MATCH_ID)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="--seed-db와 함께 사용할 로컬 원본 영상 경로. 생략 시 MOCK_VIDEO_SOURCE_PATH 사용.",
    )
    parser.add_argument(
        "--seed-db",
        action="store_true",
        help=(
            "생성한 fixture를 DB의 mock Project/Match/TimelineEvent로 함께 시드한다. "
            "동일 match_id의 기존 mock event만 안전하게 교체한다."
        ),
    )
    parser.add_argument(
        "--owner-user-id",
        default=None,
        help=(
            "선택 사항. mock project owner로 저장할 user_id. "
            "MOCK_SHARED_ACCESS_ENABLED=true이면 생략해도 모든 로그인 팀원이 접근할 수 있다."
        ),
    )
    parser.add_argument("--project-title", default="KickClip 한일전 공유 목업")
    parser.add_argument("--home-team", default="대한민국")
    parser.add_argument("--away-team", default="일본")
    parser.add_argument("--home-score", type=int, default=2)
    parser.add_argument("--away-score", type=int, default=0)
    return parser.parse_args()


def compute_half_boundary(events: list[dict]) -> int:
    """연속 이벤트 간 시간 간격이 가장 큰 지점을 하프타임 경계로 판단한다 (45분 고정 컷 금지)."""
    if len(events) < 2:
        return len(events)

    gaps = [
        (events[i + 1]["time_sec"] - events[i]["time_sec"], i)
        for i in range(len(events) - 1)
    ]
    _, boundary_index = max(gaps, key=lambda item: item[0])
    return boundary_index + 1  # 이 개수만큼이 half 1


def build_events(raw_events: list[dict], match_id: str) -> list[dict]:
    sorted_events = sorted(raw_events, key=lambda event: event["time_sec"])
    half1_count = compute_half_boundary(sorted_events)
    now = datetime.now(timezone.utc)
    source_artifact_id = SOURCE_ARTIFACT_ID_TEMPLATE.format(match_id=match_id)

    built: list[dict] = []
    for index, raw_event in enumerate(sorted_events):
        label = raw_event["label"]
        if label not in CLASS_CONFIG:
            raise ValueError(f"class_weight 테이블에 없는 라벨입니다: {label!r}")
        pre_offset, post_offset, weight = CLASS_CONFIG[label]

        timestamp_sec = float(raw_event["time_sec"])
        start_sec = max(0.0, timestamp_sec - pre_offset)
        end_sec = timestamp_sec + post_offset
        duration_sec = end_sec - start_sec
        confidence = float(raw_event["confidence"])

        built.append(
            {
                "timeline_event_id": raw_event["event_id"],
                "match_id": match_id,
                "source_artifact_id": source_artifact_id,
                "source_job_id": ANALYSIS_JOB_ID,
                "event_type": "action_spotting",
                "label": label,
                "half": 1 if index < half1_count else 2,
                "timestamp_sec": timestamp_sec,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "confidence": confidence,
                "highlight_score": weight * confidence,
                "title": None,
                "description": None,
                "team_name": None,
                "player_ids": [],
                "metadata": {
                    "raw_time_sec": raw_event["raw_time_sec"],
                    "frame_index": raw_event["frame_index"],
                    "time_hms": raw_event["time_hms"],
                    "min_start_sec": max(0.0, start_sec - 5.0),
                    "max_end_sec": end_sec + 5.0,
                },
                "created_at": now,
                "updated_at": now,
            }
        )
    return built


def print_summary(events: list[dict]) -> None:
    half_counts = Counter(event["half"] for event in events)
    label_counts = Counter(event["label"] for event in events)
    top5 = sorted(events, key=lambda event: event["highlight_score"], reverse=True)[:5]

    print(f"총 이벤트: {len(events)}개")
    print(f"half 분류: half1={half_counts.get(1, 0)}개, half2={half_counts.get(2, 0)}개")
    print("라벨별 개수:")
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count}개")
    print("highlight_score 상위 5개:")
    for event in top5:
        print(
            f"  - {event['timeline_event_id']} {event['label']} "
            f"score={event['highlight_score']:.3f} "
            f"({event['metadata']['time_hms']})"
        )


def seed_database(
    *,
    payload: dict,
    source_path: Path,
    args: argparse.Namespace,
) -> None:
    from app.core.config import get_settings
    from app.db.session import SessionLocal
    from app.domains.media.mock_seed import seed_mock_video_asset
    from app.domains.timeline.mock_seed import seed_mock_timeline_fixture

    settings = get_settings()
    if settings.is_production:
        raise RuntimeError("운영 환경에서는 목업 DB 시드를 실행할 수 없습니다.")
    if not settings.USE_MOCK_DATA:
        raise RuntimeError(
            ".env에서 USE_MOCK_DATA=true를 설정한 뒤 --seed-db를 실행하세요."
        )

    with SessionLocal() as db:
        result = seed_mock_timeline_fixture(
            db,
            payload,
            target_match_id=args.match_id,
            source_path=source_path,
            project_title=args.project_title,
            owner_id=args.owner_user_id,
            home_team=args.home_team,
            away_team=args.away_team,
            home_score=args.home_score,
            away_score=args.away_score,
        )

    required_duration_sec = max(float(event["end_sec"]) for event in payload["events"])
    with SessionLocal() as db:
        video_result = seed_mock_video_asset(
            db,
            match_id=args.match_id,
            source_path=args.video_path or settings.MOCK_VIDEO_SOURCE_PATH,
            required_duration_sec=required_duration_sec,
            link_mode=settings.MOCK_VIDEO_LINK_MODE,
        )

    print("DB 시드 완료:")
    print(f"  - project_id: {result.project_id}")
    print(f"  - match_id: {result.match_id}")
    print(f"  - owner_id: {result.owner_id}")
    print(f"  - event_count: {result.event_count}")
    print(f"  - replaced_event_count: {result.replaced_event_count}")
    print(f"  - video_asset_id: {video_result.asset_id}")
    print(f"  - video_duration_sec: {video_result.duration_sec:.3f}")


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"원본 action spotting JSON이 없습니다: {args.input}")

    raw_payload = json.loads(args.input.read_text(encoding="utf-8"))
    events = build_events(raw_payload["events"], args.match_id)

    validated_events = [TimelineEventRead.model_validate(event) for event in events]
    response = ActionSpottingEventsResponse.model_validate(
        {
            "analysis_job_id": ANALYSIS_JOB_ID,
            "status": "completed",
            "count": len(validated_events),
            "events": validated_events,
        }
    )

    output_path = args.output or (get_storage_root() / get_match_timeline_events_path(args.match_id))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        response.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"저장 위치: {output_path}")
    print_summary(events)

    if args.seed_db:
        seed_database(
            payload=response.model_dump(mode="json", by_alias=True),
            source_path=output_path,
            args=args,
        )


if __name__ == "__main__":
    main()
