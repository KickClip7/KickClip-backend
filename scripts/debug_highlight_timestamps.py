from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class TimestampCheckRow:
    index: int
    label: str
    timestamp_sec: float
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float | None
    predictor_mode: str | None
    feature_index: int | None
    offset_feature_index: float | None
    feature_fps: float | None
    segment_offset_sec: float | None
    expected_timestamp_sec: float | None
    timestamp_delta_sec: float | None
    half: int | None
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "timestamp_sec": self.timestamp_sec,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "confidence": self.confidence,
            "predictor_mode": self.predictor_mode,
            "feature_index": self.feature_index,
            "offset_feature_index": self.offset_feature_index,
            "feature_fps": self.feature_fps,
            "segment_offset_sec": self.segment_offset_sec,
            "expected_timestamp_sec": self.expected_timestamp_sec,
            "timestamp_delta_sec": self.timestamp_delta_sec,
            "half": self.half,
            "source": self.source,
        }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    event_json_path = resolve_event_json_path(
        project_root=project_root,
        match_id=args.match_id,
        job_id=args.job_id,
        event_json=args.event_json,
    )
    event_payload = load_json(event_json_path)
    match_id = args.match_id or str(event_payload.get("match_id") or "").strip()
    job_id = args.job_id or str(event_payload.get("analysis_job_id") or "").strip()

    if not match_id:
        raise SystemExit("match_id is required. Pass --match-id or use an event JSON containing match_id.")
    if not job_id:
        raise SystemExit("job_id is required. Pass --job-id or use an event JSON containing analysis_job_id.")

    feature_metadata_path = resolve_feature_metadata_path(
        project_root=project_root,
        match_id=match_id,
        feature_metadata=args.feature_metadata,
    )
    feature_metadata = load_json(feature_metadata_path) if feature_metadata_path.exists() else {}

    video_path = resolve_video_path(
        project_root=project_root,
        match_id=match_id,
        video_path=args.video,
    )

    candidates = event_payload.get("candidates") or []
    if not isinstance(candidates, list):
        raise SystemExit(f"Invalid event candidate JSON: candidates must be a list: {event_json_path}")

    rows = [build_row(index=i, candidate=candidate) for i, candidate in enumerate(candidates)]
    summary = build_summary(
        rows=rows,
        event_json_path=event_json_path,
        feature_metadata_path=feature_metadata_path,
        feature_metadata=feature_metadata,
        video_path=video_path,
        match_id=match_id,
        job_id=job_id,
    )

    output_dir = Path(args.out_dir) if args.out_dir else project_root / "storage" / "debug" / "highlight_timestamps" / job_id
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "timestamp_debug_report.json"
    csv_path = output_dir / "timestamp_debug_candidates.csv"
    commands_path = output_dir / "ffmpeg_extract_commands.txt"

    write_json(report_path, {"summary": summary, "candidates": [row.to_dict() for row in rows]})
    write_csv(csv_path, rows)
    commands = build_ffmpeg_commands(
        rows=rows,
        video_path=video_path,
        output_dir=output_dir / "clips",
        around_sec=float(args.around_sec),
        max_candidates=int(args.max_clips),
    )
    commands_path.write_text("\n".join(commands) + ("\n" if commands else ""), encoding="utf-8")

    if args.make_clips:
        make_debug_clips(commands=commands, dry_run=args.dry_run)

    print_summary(summary, rows)
    print(f"\n[OK] wrote report: {to_display_path(report_path)}")
    print(f"[OK] wrote csv   : {to_display_path(csv_path)}")
    print(f"[OK] wrote ffmpeg commands: {to_display_path(commands_path)}")
    if video_path:
        print(f"[INFO] source video: {to_display_path(video_path)}")
    else:
        print("[WARN] source video was not resolved. Pass --video to generate usable ffmpeg commands.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect real highlight event timestamps and generate ffmpeg debug commands."
    )
    parser.add_argument("--project-root", default=".", help="KickClip backend project root. Default: current directory.")
    parser.add_argument("--match-id", help="Match id, e.g. match_7cc8c21e1813.")
    parser.add_argument("--job-id", help="Analysis job id, e.g. job_dd190191f611.")
    parser.add_argument("--event-json", help="Path to event_candidates_<job_id>.json. Optional when --match-id and --job-id are provided.")
    parser.add_argument("--feature-metadata", help="Path to feature_metadata.json. Optional when --match-id is provided.")
    parser.add_argument("--video", help="Path to source video. Optional; otherwise the first file under storage/matches/<match_id>/raw_video is used.")
    parser.add_argument("--out-dir", help="Output directory for debug report/csv/commands.")
    parser.add_argument("--around-sec", type=float, default=6.0, help="Seconds around timestamp to include in debug clips. Default: 6.")
    parser.add_argument("--max-clips", type=int, default=10, help="Max number of ffmpeg clip commands to generate. Default: 10.")
    parser.add_argument("--make-clips", action="store_true", help="Execute generated ffmpeg commands.")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without executing them.")
    return parser.parse_args()


def resolve_event_json_path(
    *,
    project_root: Path,
    match_id: str | None,
    job_id: str | None,
    event_json: str | None,
) -> Path:
    if event_json:
        path = Path(event_json)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise SystemExit(f"event JSON does not exist: {path}")
        return path

    if not match_id or not job_id:
        raise SystemExit("Either --event-json or both --match-id and --job-id are required.")

    path = project_root / "storage" / "matches" / match_id / "artifacts" / "event_candidates" / f"event_candidates_{job_id}.json"
    if not path.exists():
        raise SystemExit(f"event JSON does not exist: {path}")
    return path


def resolve_feature_metadata_path(
    *,
    project_root: Path,
    match_id: str,
    feature_metadata: str | None,
) -> Path:
    if feature_metadata:
        path = Path(feature_metadata)
        if not path.is_absolute():
            path = project_root / path
        return path
    return project_root / "storage" / "matches" / match_id / "features" / "soccernet_pca512" / "feature_metadata.json"


def resolve_video_path(*, project_root: Path, match_id: str, video_path: str | None) -> Path | None:
    if video_path:
        path = Path(video_path)
        if not path.is_absolute():
            path = project_root / path
        return path

    raw_dir = project_root / "storage" / "matches" / match_id / "raw_video"
    if not raw_dir.exists():
        return None
    candidates = [path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    if not candidates:
        candidates = [path for path in raw_dir.iterdir() if path.is_file()]
    return candidates[0] if candidates else None


def build_row(index: int, candidate: dict[str, Any]) -> TimestampCheckRow:
    metadata = candidate.get("metadata") or {}
    feature_index = optional_int(metadata.get("feature_index"))
    offset_feature_index = optional_float(metadata.get("offset_feature_index"))
    feature_fps = optional_float(metadata.get("feature_fps"))
    segment_offset_sec = optional_float(metadata.get("segment_offset_sec")) or 0.0

    expected_timestamp = None
    delta = None
    if feature_index is not None and feature_fps and feature_fps > 0:
        expected_timestamp = segment_offset_sec + ((feature_index + (offset_feature_index or 0.0)) / feature_fps)
        delta = float(candidate.get("timestamp_sec") or 0.0) - expected_timestamp

    return TimestampCheckRow(
        index=index,
        label=str(candidate.get("label") or candidate.get("event_type") or "unknown"),
        timestamp_sec=float(candidate.get("timestamp_sec") or 0.0),
        start_sec=float(candidate.get("start_sec") or 0.0),
        end_sec=float(candidate.get("end_sec") or 0.0),
        duration_sec=float(candidate.get("duration_sec") or 0.0),
        confidence=optional_float(candidate.get("confidence")),
        predictor_mode=metadata.get("predictor_mode"),
        feature_index=feature_index,
        offset_feature_index=offset_feature_index,
        feature_fps=feature_fps,
        segment_offset_sec=segment_offset_sec,
        expected_timestamp_sec=expected_timestamp,
        timestamp_delta_sec=delta,
        half=optional_int(candidate.get("half")),
        source=(metadata.get("source") or candidate.get("source")),
    )


def build_summary(
    *,
    rows: list[TimestampCheckRow],
    event_json_path: Path,
    feature_metadata_path: Path,
    feature_metadata: dict[str, Any],
    video_path: Path | None,
    match_id: str,
    job_id: str,
) -> dict[str, Any]:
    real_count = sum(1 for row in rows if row.predictor_mode == "real")
    fallback_count = sum(1 for row in rows if row.predictor_mode == "fallback")
    max_abs_delta = max((abs(row.timestamp_delta_sec) for row in rows if row.timestamp_delta_sec is not None), default=None)

    warnings: list[str] = []
    if not rows:
        warnings.append("No candidates found in event JSON.")
    if rows and real_count == 0:
        warnings.append("No candidate has metadata.predictor_mode='real'.")
    if max_abs_delta is not None and max_abs_delta > 0.2:
        warnings.append(f"Max feature-index timestamp delta is {max_abs_delta:.3f}s; inspect offset/timestamp formula.")
    if video_path is None or not video_path.exists():
        warnings.append("Source video could not be resolved; ffmpeg commands may not be executable.")

    return {
        "match_id": match_id,
        "job_id": job_id,
        "event_json_path": event_json_path.as_posix(),
        "feature_metadata_path": feature_metadata_path.as_posix(),
        "feature_metadata_exists": feature_metadata_path.exists(),
        "video_path": video_path.as_posix() if video_path else None,
        "num_candidates": len(rows),
        "real_candidate_count": real_count,
        "fallback_candidate_count": fallback_count,
        "labels": sorted({row.label for row in rows}),
        "min_timestamp_sec": min((row.timestamp_sec for row in rows), default=None),
        "max_timestamp_sec": max((row.timestamp_sec for row in rows), default=None),
        "max_abs_feature_timestamp_delta_sec": max_abs_delta,
        "feature_metadata_summary": {
            "source_video_duration_sec": feature_metadata.get("source_video_duration_sec"),
            "feature_fps": feature_metadata.get("feature_fps"),
            "split_strategy": feature_metadata.get("split_strategy"),
            "split_sec": feature_metadata.get("split_sec"),
            "merged_shape": feature_metadata.get("merged_shape"),
            "half1_shape": feature_metadata.get("half1_shape"),
            "half2_shape": feature_metadata.get("half2_shape"),
        },
        "warnings": warnings,
    }


def build_ffmpeg_commands(
    *,
    rows: list[TimestampCheckRow],
    video_path: Path | None,
    output_dir: Path,
    around_sec: float,
    max_candidates: int,
) -> list[str]:
    if video_path is None:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []
    half_span = max(around_sec / 2.0, 0.5)

    for row in rows[:max_candidates]:
        start = max(row.timestamp_sec - half_span, 0.0)
        duration = max(around_sec, 1.0)
        safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in row.label)
        out_path = output_dir / f"cand_{row.index:03d}_{safe_label}_{row.timestamp_sec:.1f}s.mp4"
        commands.append(
            "ffmpeg -y "
            f"-ss {start:.3f} "
            f"-i {quote_path(video_path)} "
            f"-t {duration:.3f} "
            "-c copy "
            f"{quote_path(out_path)}"
        )
    return commands


def make_debug_clips(*, commands: list[str], dry_run: bool) -> None:
    if not commands:
        print("[WARN] no ffmpeg commands to execute")
        return
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg was not found on PATH; commands were generated but not executed")
        return

    for command in commands:
        print(command)
        if dry_run:
            continue
        completed = subprocess.run(command, shell=True, check=False)
        if completed.returncode != 0:
            print(f"[WARN] ffmpeg command failed with returncode={completed.returncode}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[TimestampCheckRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TimestampCheckRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def print_summary(summary: dict[str, Any], rows: list[TimestampCheckRow]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCandidates:")
    print("idx | label | ts | start-end | conf | mode | feat_idx | expected_ts | delta")
    for row in rows[:20]:
        expected = f"{row.expected_timestamp_sec:.3f}" if row.expected_timestamp_sec is not None else "-"
        delta = f"{row.timestamp_delta_sec:+.3f}" if row.timestamp_delta_sec is not None else "-"
        confidence = f"{row.confidence:.4f}" if row.confidence is not None else "-"
        feature_index = str(row.feature_index) if row.feature_index is not None else "-"
        print(
            f"{row.index:>3} | {row.label:<9} | {row.timestamp_sec:>7.3f} | "
            f"{row.start_sec:>7.3f}-{row.end_sec:<7.3f} | {confidence:>6} | "
            f"{str(row.predictor_mode):<8} | {feature_index:>8} | {expected:>11} | {delta:>7}"
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def quote_path(path: Path) -> str:
    return '"' + str(path).replace('"', '\\"') + '"'


def to_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


if __name__ == "__main__":
    main()
