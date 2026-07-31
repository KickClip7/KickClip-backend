from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.highlight.model import HighlightRevision
from app.domains.media.model import MediaAsset
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent


MATCH_ID = "match_preloaded_kor_jpn"
SOURCE_ASSET_ID = "asset_preloaded_215d278fb5a41438ed19dfea"
ACTION_JOB_ID = "job_087614b078d0"
EVENT_ID = "evt_5eb4be390208"
PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
SCENE_ASSET_ID = "asset_eval_goal_evt_5eb4be390208_scene"
PROJECT_TITLE = "Representative Goal Shadow Evaluation"
STORAGE_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def extract_scene(
    source: Path,
    destination: Path,
    *,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(source_fps - fps) > 0.01:
        raise RuntimeError("Representative source FPS changed.")
    start_frame = int(round(start_sec * source_fps))
    frame_count = int(round((end_sec - start_sec) * fps))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open representative scene writer.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < frame_count:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1
    writer.release()
    capture.release()
    if written != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Scene decode ended early: {written}/{frame_count}"
        )
    temporary.replace(destination)
    return {
        "source_start_sec": start_sec,
        "source_end_sec": end_sec,
        "source_start_frame": start_frame,
        "source_end_frame_inclusive": start_frame + frame_count - 1,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def cut_scores(video: Path) -> tuple[list[np.ndarray], np.ndarray, float]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    signatures: list[tuple[np.ndarray, np.ndarray]] = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]
        )
        cv2.normalize(histogram, histogram)
        frames.append(frame)
        signatures.append((gray, histogram))
    capture.release()
    scores = np.zeros(len(frames), dtype=np.float64)
    for index in range(1, len(signatures)):
        previous_gray, previous_histogram = signatures[index - 1]
        current_gray, current_histogram = signatures[index]
        pixel_delta = float(
            np.mean(
                np.abs(
                    current_gray.astype(np.float32)
                    - previous_gray.astype(np.float32)
                )
            )
            / 255.0
        )
        histogram_delta = float(
            cv2.compareHist(
                previous_histogram,
                current_histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        scores[index] = 0.45 * pixel_delta + 0.55 * histogram_delta
    return frames, scores, fps


def propose_cuts(scores: np.ndarray, fps: float) -> tuple[list[int], float]:
    nonzero = scores[1:]
    median = float(np.median(nonzero))
    mad = float(np.median(np.abs(nonzero - median)))
    threshold = max(0.28, median + 8.0 * max(mad, 0.01))
    candidates = [
        index
        for index in range(2, len(scores) - 1)
        if scores[index] >= threshold
        and scores[index] >= scores[index - 1]
        and scores[index] >= scores[index + 1]
    ]
    minimum_gap = max(3, int(round(fps * 0.35)))
    selected: list[int] = []
    for candidate in sorted(
        candidates, key=lambda index: scores[index], reverse=True
    ):
        if all(abs(candidate - existing) >= minimum_gap for existing in selected):
            selected.append(candidate)
    return sorted(selected), threshold


def render_review_artifacts(
    root: Path,
    frames: list[np.ndarray],
    scores: np.ndarray,
    cuts: list[int],
    *,
    fps: float,
    video_name: str,
) -> tuple[Path, Path]:
    pairs: list[np.ndarray] = []
    for cut in cuts:
        left = cv2.resize(frames[max(0, cut - 1)], (320, 180))
        right = cv2.resize(frames[min(len(frames) - 1, cut)], (320, 180))
        cv2.putText(
            left,
            f"before f{cut - 1}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            right,
            f"after f{cut} {cut / fps:.2f}s",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        pairs.append(np.hstack([left, right]))
    if pairs:
        cut_sheet = np.vstack(pairs)
    else:
        cut_sheet = np.full((180, 640, 3), 32, dtype=np.uint8)
        cv2.putText(
            cut_sheet,
            "No cut proposals",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
    cut_sheet_path = root / "shot_boundary_review_contact_sheet.jpg"
    cv2.imwrite(str(cut_sheet_path), cut_sheet)

    html_path = root / "shot_boundary_review.html"
    rows = "\n".join(
        (
            f"<tr><td>{index + 1}</td><td>{cut}</td>"
            f"<td>{cut / fps:.3f}</td><td>{scores[cut]:.6f}</td>"
            "<td>UNREVIEWED</td></tr>"
        )
        for index, cut in enumerate(cuts)
    )
    html_path.write_text(
        """<!doctype html><html><head><meta charset="utf-8">
<title>Representative Goal shot review</title>
<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto}
video,img{max-width:100%}table{border-collapse:collapse}
td,th{border:1px solid #aaa;padding:.4rem}</style></head><body>
<h1>Representative Goal — camera-shot review</h1>
<p>This artifact is pending human review. Discovery remains blocked until
every proposed boundary is accepted or corrected and every resulting shot is
REVIEWED_PASS or CONFIRMED.</p>
<video controls src=\""""
        + video_name
        + """\"></video>
<h2>Before/after contact sheet</h2>
<img src="shot_boundary_review_contact_sheet.jpg">
<h2>Proposals</h2><table><thead><tr><th>#</th><th>Frame</th>
<th>Scene-local sec</th><th>Score</th><th>State</th></tr></thead><tbody>"""
        + rows
        + """</tbody></table></body></html>
""",
        encoding="utf-8",
    )
    return cut_sheet_path, html_path


def main() -> None:
    with SessionLocal() as db:
        event = db.get(TimelineEvent, EVENT_ID)
        source_asset = db.get(MediaAsset, SOURCE_ASSET_ID)
        if event is None or source_asset is None:
            raise RuntimeError("Representative source event or asset is missing.")
        if (
            event.match_id != MATCH_ID
            or event.source_job_id != ACTION_JOB_ID
            or source_asset.match_id != MATCH_ID
            or event.label.lower() != "goal"
        ):
            raise RuntimeError("Representative Goal provenance changed.")
        source = (STORAGE_ROOT / source_asset.file_path).resolve()
        if (
            not source.is_relative_to(STORAGE_ROOT)
            or not source.is_file()
            or sha256_file(source) != source_asset.sha256
        ):
            raise RuntimeError("Canonical source MediaAsset SHA mismatch.")

        project = db.get(Project, PROJECT_ID)
        if project is None:
            project = Project(
                project_id=PROJECT_ID,
                match_id=MATCH_ID,
                owner_id=None,
                title=PROJECT_TITLE,
                description=(
                    "Development-only representative Goal shadow evaluation."
                ),
                status="DRAFT",
            )
            db.add(project)
            db.flush()
        elif project.match_id != MATCH_ID:
            raise RuntimeError("Evaluation Project ID collision.")

        revision = db.get(HighlightRevision, REVISION_ID)
        if revision is None:
            revision = HighlightRevision(
                revision_id=REVISION_ID,
                project_id=PROJECT_ID,
                revision_number=1,
                action_spotting_job_id=ACTION_JOB_ID,
                user_request="Representative Goal shadow evaluation",
                structured_request={
                    "event_id": EVENT_ID,
                    "mode": "PROVISIONAL_SHADOW_ONLY",
                },
                selected_scene_ids=[EVENT_ID],
                scene_selection=[
                    {
                        "scene_id": EVENT_ID,
                        "source_event_ids": [EVENT_ID],
                        "status": "SHOT_BOUNDARY_REVIEW_REQUIRED",
                    }
                ],
                focus_mode="NONE",
                status="SHOT_BOUNDARY_REVIEW_REQUIRED",
                pending_action="REVIEW_SHOT_BOUNDARIES",
                options={
                    "representative_goal": {
                        "event_id": EVENT_ID,
                        "source_job_id": ACTION_JOB_ID,
                        "source_asset_id": SOURCE_ASSET_ID,
                        "scene_start_sec": float(event.start_sec),
                        "scene_end_sec": float(event.end_sec),
                    }
                },
            )
            db.add(revision)
            db.flush()

        review_root = (
            STORAGE_ROOT
            / "storage"
            / "matches"
            / MATCH_ID
            / "projects"
            / PROJECT_ID
            / "highlight"
            / REVISION_ID
            / "representative-goal-review"
        ).resolve()
        if not review_root.is_relative_to(STORAGE_ROOT / "storage"):
            raise RuntimeError("Review output escapes storage.")
        review_root.mkdir(parents=True, exist_ok=True)
        scene_video = review_root / "representative_goal_scene.mp4"
        if not scene_video.is_file():
            clip = extract_scene(
                source,
                scene_video,
                start_sec=float(event.start_sec),
                end_sec=float(event.end_sec),
                fps=float(source_asset.fps or 25.0),
            )
        else:
            capture = cv2.VideoCapture(str(scene_video))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            clip = {
                "source_start_sec": float(event.start_sec),
                "source_end_sec": float(event.end_sec),
                "source_start_frame": int(
                    round(float(event.start_sec) * float(source_asset.fps or 25))
                ),
                "source_end_frame_inclusive": int(
                    round(float(event.start_sec) * float(source_asset.fps or 25))
                )
                + frame_count
                - 1,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_sec": frame_count / fps,
                "size_bytes": scene_video.stat().st_size,
                "sha256": sha256_file(scene_video),
            }
        if abs(clip["duration_sec"] - 45.0) > 0.05:
            raise RuntimeError("Representative scene duration is invalid.")

        scene_asset = db.get(MediaAsset, SCENE_ASSET_ID)
        relative_scene = scene_video.relative_to(STORAGE_ROOT).as_posix()
        if scene_asset is None:
            scene_asset = MediaAsset(
                asset_id=SCENE_ASSET_ID,
                match_id=MATCH_ID,
                asset_type="HIGHLIGHT_SCENE_CLIP",
                file_path=relative_scene,
                original_filename=scene_video.name,
                mime_type="video/mp4",
                duration_sec=clip["duration_sec"],
                fps=clip["fps"],
                width=clip["width"],
                height=clip["height"],
                size_bytes=clip["size_bytes"],
                sha256=clip["sha256"],
            )
            db.add(scene_asset)
            db.flush()
        elif scene_asset.sha256 != clip["sha256"]:
            raise RuntimeError("Scene MediaAsset immutable SHA mismatch.")

        frames, scores, scene_fps = cut_scores(scene_video)
        cuts, threshold = propose_cuts(scores, scene_fps)
        boundaries = [0, *cuts, len(frames)]
        shots = []
        for index, (start, next_start) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            shots.append(
                {
                    "shot_id": f"shot_{index:04d}",
                    "shot_index": index,
                    "start_frame": start,
                    "end_frame_inclusive": next_start - 1,
                    "start_time_sec": start / scene_fps,
                    "end_time_sec": next_start / scene_fps,
                    "review_state": "UNREVIEWED",
                }
            )
        boundaries_path = review_root / "shot_boundaries_pending_review.json"
        boundary_document = {
            "schema_version": "kickclip.reviewed_shot_boundaries.v1",
            "status": "PENDING_HUMAN_REVIEW",
            "video": {
                "sha256": clip["sha256"],
                "frame_count": len(frames),
                "fps": scene_fps,
                "width": clip["width"],
                "height": clip["height"],
            },
            "diagnostics": {
                "review_required": True,
                "retrieval_authorized": False,
                "pending_cut_frames": cuts,
                "cut_score_threshold": threshold,
            },
            "review_contract": {
                "pending_cut_frames": cuts,
                "approved_states": ["REVIEWED_PASS", "CONFIRMED"],
                "reviewer": None,
                "reviewed_at": None,
            },
            "shots": shots,
        }
        write_json(boundaries_path, boundary_document)
        contact_sheet, html = render_review_artifacts(
            review_root,
            frames,
            scores,
            cuts,
            fps=scene_fps,
            video_name=scene_video.name,
        )
        provenance_path = review_root / "representative_goal_provenance.json"
        write_json(
            provenance_path,
            {
                "project_id": PROJECT_ID,
                "revision_id": REVISION_ID,
                "event_id": EVENT_ID,
                "scene_id": EVENT_ID,
                "action_spotting_job_id": ACTION_JOB_ID,
                "source_asset_id": SOURCE_ASSET_ID,
                "source_video_sha256": source_asset.sha256,
                "scene_video_asset_id": SCENE_ASSET_ID,
                "scene_video_sha256": clip["sha256"],
                "scene_source_start_sec": float(event.start_sec),
                "scene_source_end_sec": float(event.end_sec),
                "event_timestamp_sec": float(event.timestamp_sec),
                "event_confidence": event.confidence,
                "highlight_goal_reference_sha256": (
                    "3c06257ca633a50f31545e5e4c3e318c"
                    "9c6f5871466b535638ae341317bfb597"
                ),
                "frame_provenance_mean_correlation": 0.9991548420470437,
                "shot_review_status": "PENDING_HUMAN_REVIEW",
                "automatic_target_confirmation": False,
            },
        )

        artifact_specs = (
            (
                "REPRESENTATIVE_GOAL_PROVENANCE",
                provenance_path,
                "application/json",
            ),
            (
                "SHOT_BOUNDARIES_PENDING_REVIEW",
                boundaries_path,
                "application/json",
            ),
            (
                "SHOT_BOUNDARY_REVIEW_CONTACT_SHEET",
                contact_sheet,
                "image/jpeg",
            ),
            ("SHOT_BOUNDARY_REVIEW_HTML", html, "text/html"),
        )
        artifact_ids = {}
        for artifact_type, path, mime in artifact_specs:
            existing = db.scalar(
                select(Artifact).where(
                    Artifact.project_id == PROJECT_ID,
                    Artifact.artifact_type == artifact_type,
                )
            )
            if existing is None:
                existing = Artifact(
                    match_id=MATCH_ID,
                    project_id=PROJECT_ID,
                    analysis_job_id=None,
                    artifact_type=artifact_type,
                    file_path=path.relative_to(STORAGE_ROOT).as_posix(),
                    mime_type=mime,
                    metadata_={
                        "revision_id": REVISION_ID,
                        "scene_id": EVENT_ID,
                        "sha256": sha256_file(path),
                        "review_state": "PENDING_HUMAN_REVIEW",
                    },
                )
                db.add(existing)
                db.flush()
            artifact_ids[artifact_type] = existing.artifact_id
        revision.options = {
            **(revision.options or {}),
            "representative_goal": {
                **((revision.options or {}).get("representative_goal") or {}),
                "scene_video_asset_id": SCENE_ASSET_ID,
                "scene_video_sha256": clip["sha256"],
                "shot_review_artifact_id": artifact_ids[
                    "SHOT_BOUNDARIES_PENDING_REVIEW"
                ],
                "shot_review_contact_sheet_artifact_id": artifact_ids[
                    "SHOT_BOUNDARY_REVIEW_CONTACT_SHEET"
                ],
                "shot_review_html_artifact_id": artifact_ids[
                    "SHOT_BOUNDARY_REVIEW_HTML"
                ],
                "shot_review_status": "PENDING_HUMAN_REVIEW",
            },
        }
        db.commit()
        print(
            json.dumps(
                {
                    "project_id": PROJECT_ID,
                    "revision_id": REVISION_ID,
                    "event_id": EVENT_ID,
                    "scene_id": EVENT_ID,
                    "scene_video_asset_id": SCENE_ASSET_ID,
                    "scene_video": str(scene_video),
                    "scene_video_sha256": clip["sha256"],
                    "frame_count": clip["frame_count"],
                    "proposed_cut_count": len(cuts),
                    "proposed_shot_count": len(shots),
                    "pending_cut_frames": cuts,
                    "shot_boundaries": str(boundaries_path),
                    "contact_sheet": str(contact_sheet),
                    "review_html": str(html),
                    "artifact_ids": artifact_ids,
                    "status": "PENDING_HUMAN_REVIEW",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
