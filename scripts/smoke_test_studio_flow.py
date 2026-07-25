"""KickClip Studio E2E smoke test.

14회차 목적:
1. 영상 업로드
2. Match 아래 Project 생성
3. 분석 job 생성 및 polling
4. project_id 기준 edit-state 조회
5. timeline-events / players 조회
6. Agent ClipPlan 생성
7. export-options 저장
8. RenderJob 생성 및 polling
9. 결과 mp4 다운로드

사용 예시:
    python scripts/smoke_test_studio_flow.py --video-path sample.mp4

비디오를 지정하지 않으면 ffmpeg로 짧은 demo video를 생성한다.
단, 기존 dummy TimelineEvent 시간이 긴 경우 render-safe manual plan을 자동 생성해
짧은 demo video에서도 E2E 렌더링 검증이 가능하게 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "requests 패키지가 필요합니다. `pip install requests` 후 다시 실행하세요."
    ) from exc


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_PATH = Path("storage/outputs/e2e_final_short.mp4")
DEFAULT_DEMO_VIDEO_PATH = Path("storage/demo/e2e_demo_90s.mp4")


class SmokeTestError(RuntimeError):
    pass


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    video_path = Path(args.video_path) if args.video_path else DEFAULT_DEMO_VIDEO_PATH
    if args.video_path is None:
        ensure_demo_video(video_path, duration_sec=args.demo_duration_sec)

    if not video_path.exists():
        raise SmokeTestError(f"Video file not found: {video_path}")

    print_step("1. Health check")
    health = request_json("GET", f"{base_url}/api/v1/health")
    print_json(health)

    print_step("2. Upload match video")
    upload = upload_match_video(
        base_url=base_url,
        video_path=video_path,
        home_team=args.home_team,
        away_team=args.away_team,
        home_score=args.home_score,
        away_score=args.away_score,
        duration_sec=args.duration_sec,
    )
    print_json(upload)

    match_id = upload["match_id"]
    raw_video_asset_id = upload.get("raw_video_asset_id")
    if not raw_video_asset_id:
        raise SmokeTestError("Upload response did not include raw_video_asset_id.")

    print_step("3. Create project under match")
    project = request_json(
        "POST",
        f"{base_url}/api/v1/matches/{match_id}/projects",
        json_body={
            "title": args.project_title,
            "description": "Studio E2E smoke test project",
        },
    )
    project_id = project["project_id"]
    print_json(project)

    print_step("4. Start analysis job")
    job = request_json(
        "POST",
        f"{base_url}/api/v1/matches/{match_id}/analysis-jobs",
        json_body={
            "job_type": "FULL_MATCH_ANALYSIS",
            "options": {
                "run_highlight_spotting": True,
                "run_player_tracking": True,
                "build_timeline": True,
                "dummy_mode": True,
            },
        },
    )
    print_json(job)

    job_id = job.get("analysis_job_id") or job.get("job_id")
    if not job_id:
        raise SmokeTestError("Analysis job response did not include analysis_job_id.")

    print_step("5. Poll analysis job")
    job_status = poll_status(
        url=f"{base_url}/api/v1/analysis-jobs/{job_id}",
        timeout_sec=args.analysis_timeout_sec,
        interval_sec=args.poll_interval_sec,
        status_key="status",
        progress_key="progress",
        success_statuses={"completed", "COMPLETED"},
        failure_statuses={"failed", "FAILED", "canceled", "CANCELED"},
    )
    print_json(job_status)

    print_step("6. Fetch project edit-state")
    edit_state = request_json(
        "GET",
        f"{base_url}/api/v1/projects/{project_id}/edit-state",
    )
    print_json(compact_edit_state(edit_state))

    video = edit_state.get("video") or {}
    if not video.get("url"):
        raise SmokeTestError(
            "edit-state.video.url is empty. Upload-created match should have RAW_VIDEO or WEB_PREVIEW_VIDEO."
        )

    events = edit_state.get("events") or []
    players = edit_state.get("players") or []
    if not events:
        raise SmokeTestError("No timeline events found. Analysis job did not create editable events.")

    print_step("7. Fetch timeline-events and players")
    timeline_events = request_json("GET", f"{base_url}/api/v1/matches/{match_id}/timeline-events")
    player_list = request_json("GET", f"{base_url}/api/v1/matches/{match_id}/players")
    print_json({"timeline_event_count": timeline_events.get("count"), "player_count": player_list.get("count")})

    selected_player_id = None
    if players:
        selected_player_id = players[0].get("id")

    print_step("8. Create Agent ClipPlan")
    agent_plan = request_json(
        "POST",
        f"{base_url}/api/v1/agent/clip-plan",
        json_body={
            "project_id": project_id,
            "mode": "AGENT_GENERATED",
            "prompt": args.prompt,
            "target_duration_sec": args.target_duration_sec,
            "selected_player_id": selected_player_id if args.use_selected_player else None,
            "options": {
                "allow_event_types": ["goal", "shot", "foul", "card", "free_kick", "corner"],
                "ratio": args.ratio,
            },
        },
    )
    print_json(agent_plan)

    if not agent_plan.get("items"):
        raise SmokeTestError("Agent ClipPlan has no items. Cannot continue to render.")

    render_clip_plan_id = agent_plan["clip_plan_id"]

    if args.auto_render_safe_plan:
        source_duration = infer_source_duration(upload=upload, edit_state=edit_state, fallback=args.demo_duration_sec)
        if plan_exceeds_source_duration(agent_plan, source_duration):
            print_step("8-1. Agent plan exceeds source video duration. Create render-safe manual plan.")
            first_event_id = events[0]["id"]
            safe_end = max(1.0, min(float(args.safe_clip_duration_sec), source_duration))
            manual_plan = request_json(
                "POST",
                f"{base_url}/api/v1/clip-plans",
                json_body={
                    "project_id": project_id,
                    "mode": "MANUAL",
                    "summary": "E2E smoke test render-safe manual plan",
                    "target_duration_sec": safe_end,
                    "options": {"ratio": args.ratio, "source": "e2e_smoke_test"},
                    "items": [
                        {
                            "timeline_event_id": first_event_id,
                            "start_sec": 0,
                            "end_sec": safe_end,
                            "order_index": 0,
                            "reason": "Short demo video compatibility for E2E render smoke test",
                        }
                    ],
                },
            )
            print_json(manual_plan)
            render_clip_plan_id = manual_plan["clip_plan_id"]

    print_step("9. Save export options")
    export_options = request_json(
        "PATCH",
        f"{base_url}/api/v1/clip-plans/{render_clip_plan_id}/export-options",
        json_body={
            "title": args.title,
            "ratio": args.ratio,
            "captions_enabled": args.captions_enabled,
            "music": args.music,
            "quality": args.quality,
        },
    )
    print_json({
        "clip_plan_id": export_options.get("clip_plan_id"),
        "options": export_options.get("options"),
    })

    if args.skip_render:
        print_step("Render skipped by --skip-render")
        return

    print_step("10. Start render job")
    render = request_json(
        "POST",
        f"{base_url}/api/v1/renders",
        json_body={
            "clip_plan_id": render_clip_plan_id,
            "options": {
                "ratio": args.ratio,
                "quality": args.quality,
                "captions_enabled": args.captions_enabled,
                "music": args.music,
            },
        },
    )
    print_json(render)

    render_job_id = render["render_job_id"]

    print_step("11. Poll render job")
    render_status = poll_status(
        url=f"{base_url}/api/v1/renders/{render_job_id}",
        timeout_sec=args.render_timeout_sec,
        interval_sec=args.poll_interval_sec,
        status_key="status",
        progress_key="progress",
        success_statuses={"completed", "COMPLETED"},
        failure_statuses={"failed", "FAILED", "canceled", "CANCELED"},
    )
    print_json(render_status)

    print_step("12. Download rendered video")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(
        url=f"{base_url}/api/v1/renders/{render_job_id}/download",
        output_path=output_path,
    )

    if output_path.stat().st_size <= 1024:
        raise SmokeTestError(f"Downloaded file is too small: {output_path} ({output_path.stat().st_size} bytes)")

    print_step("E2E smoke test completed")
    print_json(
        {
            "match_id": match_id,
            "project_id": project_id,
            "analysis_job_id": job_id,
            "agent_clip_plan_id": agent_plan["clip_plan_id"],
            "render_clip_plan_id": render_clip_plan_id,
            "render_job_id": render_job_id,
            "output_path": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KickClip Studio E2E smoke test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--project-title", default="KickClip E2E 테스트 프로젝트")
    parser.add_argument("--home-team", default="프로메테우스")
    parser.add_argument("--away-team", default="한빛 FC")
    parser.add_argument("--home-score", type=int, default=3)
    parser.add_argument("--away-score", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--prompt", default="골 장면과 슈팅 장면을 섞어서 30초 쇼츠로 만들어줘")
    parser.add_argument("--target-duration-sec", type=int, default=30)
    parser.add_argument("--ratio", choices=["9:16", "1:1", "16:9"], default="9:16")
    parser.add_argument("--quality", choices=["720p", "1080p", "4K"], default="720p")
    parser.add_argument("--title", default="KickClip E2E Smoke Test")
    parser.add_argument("--music", default="Hype Beat 01")
    parser.add_argument("--captions-enabled", action="store_true", default=True)
    parser.add_argument("--no-captions", dest="captions_enabled", action="store_false")
    parser.add_argument("--use-selected-player", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--analysis-timeout-sec", type=int, default=180)
    parser.add_argument("--render-timeout-sec", type=int, default=300)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--demo-duration-sec", type=int, default=90)
    parser.add_argument("--safe-clip-duration-sec", type=int, default=8)
    parser.add_argument("--auto-render-safe-plan", action="store_true", default=True)
    parser.add_argument("--no-auto-render-safe-plan", dest="auto_render_safe_plan", action="store_false")
    return parser.parse_args()


def ensure_demo_video(path: Path, duration_sec: int) -> None:
    if path.exists() and path.stat().st_size > 1024:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print_step(f"Create demo video: {path}")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=640x360:rate=24:duration={duration_sec}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=880:duration={duration_sec}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    run_command(cmd)


def upload_match_video(
    *,
    base_url: str,
    video_path: Path,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    duration_sec: float | None,
) -> dict[str, Any]:
    url = f"{base_url}/api/v1/matches"
    data: dict[str, str] = {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": str(home_score),
        "away_score": str(away_score),
    }
    if duration_sec is not None:
        data["duration_sec"] = str(duration_sec)

    with video_path.open("rb") as fp:
        files = {"file": (video_path.name, fp, guess_video_mime(video_path))}
        response = requests.post(url, data=data, files=files, timeout=120)

    if not response.ok:
        raise SmokeTestError(f"Upload failed: {response.status_code} {response.text}")
    return response.json()


def request_json(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    response = requests.request(method, url, json=json_body, timeout=timeout)
    if not response.ok:
        raise SmokeTestError(f"{method} {url} failed: {response.status_code} {response.text}")
    return response.json()


def poll_status(
    *,
    url: str,
    timeout_sec: int,
    interval_sec: float,
    status_key: str,
    progress_key: str,
    success_statuses: set[str],
    failure_statuses: set[str],
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] | None = None

    while time.time() < deadline:
        payload = request_json("GET", url)
        last_payload = payload
        status = str(payload.get(status_key, "")).lower()
        progress = payload.get(progress_key)
        print(f"  - status={status} progress={progress}")

        if status in {s.lower() for s in success_statuses}:
            return payload
        if status in {s.lower() for s in failure_statuses}:
            raise SmokeTestError(f"Job failed: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        time.sleep(interval_sec)

    raise SmokeTestError(f"Polling timeout. Last payload: {last_payload}")


def download_file(*, url: str, output_path: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        if not response.ok:
            raise SmokeTestError(f"Download failed: {response.status_code} {response.text}")
        with output_path.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)


def plan_exceeds_source_duration(plan: dict[str, Any], source_duration_sec: float) -> bool:
    items = plan.get("items") or []
    if not items:
        return False
    max_end_sec = max(float(item.get("end_sec") or 0) for item in items)
    return max_end_sec > max(1.0, source_duration_sec - 0.5)


def infer_source_duration(
    *,
    upload: dict[str, Any],
    edit_state: dict[str, Any],
    fallback: float,
) -> float:
    upload_info = upload.get("upload") or {}
    video_info = edit_state.get("video") or {}
    value = video_info.get("duration_sec") or upload_info.get("duration_sec") or fallback
    return float(value)


def compact_edit_state(edit_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "match": edit_state.get("match"),
        "video": edit_state.get("video"),
        "event_count": len(edit_state.get("events") or []),
        "player_count": len(edit_state.get("players") or []),
        "filters": edit_state.get("filters"),
    }


def guess_video_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".mkv":
        return "video/x-matroska"
    return "application/octet-stream"


def run_command(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SmokeTestError(
            f"Command not found: {cmd[0]}. ffmpeg가 PATH에 잡혀 있는지 확인하세요."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SmokeTestError(f"Command failed: {' '.join(cmd)}") from exc


def print_step(message: str) -> None:
    print(f"\n{'=' * 80}\n{message}\n{'=' * 80}")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except SmokeTestError as exc:
        print(f"\n[E2E SMOKE TEST FAILED] {exc}", file=sys.stderr)
        sys.exit(1)
