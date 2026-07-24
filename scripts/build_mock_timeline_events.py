"""한일전 raw action spotting JSON을 TimelineEventRead 스키마의 목업 timeline_events.json으로 변환한다.

raw JSON 파일은 팀원마다 로컬 경로가 다르므로 --input은 필수 인자다(하드코딩된 기본 경로 없음).

사용 예시:
    python scripts/build_mock_timeline_events.py --input path/to/events.json --match-id korjpn_2026
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


def main() -> None:
    args = parse_args()

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


if __name__ == "__main__":
    main()
