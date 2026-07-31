from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image
from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.highlight.model import HighlightRevision


PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
EVENT_ID = "evt_5eb4be390208"
MATCH_ID = "match_preloaded_kor_jpn"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = (
    PROJECT_ROOT
    / "storage/matches/match_preloaded_kor_jpn/projects"
    / PROJECT_ID
    / "highlight"
    / REVISION_ID
    / "representative-goal-review"
)
REPORT_PATH = REVIEW_ROOT / "representative_goal_shadow_run_report.json"
BUNDLE_ROOT = REVIEW_ROOT / "human-event-role-review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extended(path: Path) -> Path:
    return Path("\\\\?\\" + str(path.resolve()))


def copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Review copy differs: {destination}")
        return
    shutil.copy2(source, destination)


def gif_preview(video: Path, destination: Path) -> None:
    if destination.exists():
        return
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open tracklet review video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    stride = max(1, int(round(fps / 5.0)))
    frames: list[Image.Image] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if index % stride == 0:
            height, width = frame.shape[:2]
            target_width = min(480, width)
            target_height = max(1, int(height * target_width / width))
            small = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(
                Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            )
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No GIF frames decoded: {video}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=False,
    )


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    ranking = report["ranking"]
    discovery_relative = Path(
        ranking["discovery_snapshot"]["discovery_artifact_root"]
    )
    discovery_root = extended(PROJECT_ROOT / discovery_relative)
    candidates_document = json.loads(
        (discovery_root / "scene_candidates.json").read_text(encoding="utf-8")
    )
    candidates = {
        row["candidate_id"]: row
        for row in candidates_document["candidates"]
    }
    top_five = ranking["shortlist"]
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)

    top_cards = []
    media_manifest = []
    for ranked in top_five:
        rank = int(ranked["rank"])
        candidate = candidates[ranked["candidate_id"]]
        contact_source = discovery_root / candidate["artifacts"]["contact_sheet"]
        video_source = (
            discovery_root
            / candidate["artifacts"]["tracklet_review_video"]
        )
        contact_destination = (
            BUNDLE_ROOT / "top5" / f"rank_{rank:02d}_contact_sheet.jpg"
        )
        video_destination = (
            BUNDLE_ROOT / "top5" / f"rank_{rank:02d}_tracklet_review.mp4"
        )
        gif_destination = (
            BUNDLE_ROOT / "top5" / f"rank_{rank:02d}_tracklet_preview.gif"
        )
        copy_immutable(contact_source, contact_destination)
        copy_immutable(video_source, video_destination)
        gif_preview(video_source, gif_destination)
        top_cards.append(
            {
                "rank": rank,
                "candidate_id": candidate["candidate_id"],
                "shot_id": candidate["shot_id"],
                "score": ranked["recommendation_score"],
                "reasons": ranked["reason_codes"],
                "risks": ranked["risk_codes"],
                "contact": contact_destination.relative_to(
                    BUNDLE_ROOT
                ).as_posix(),
                "video": video_destination.relative_to(
                    BUNDLE_ROOT
                ).as_posix(),
                "gif": gif_destination.relative_to(
                    BUNDLE_ROOT
                ).as_posix(),
            }
        )
        for kind, path in (
            ("contact_sheet", contact_destination),
            ("tracklet_review", video_destination),
            ("browser_preview", gif_destination),
        ):
            media_manifest.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "rank": rank,
                    "kind": kind,
                    "path": path.relative_to(BUNDLE_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
            )

    gallery_cards = []
    for index, candidate in enumerate(
        candidates_document["candidates"],
        start=1,
    ):
        source = (
            discovery_root
            / candidate["representative_observation"]["thumbnail_artifact"]
        )
        destination = (
            BUNDLE_ROOT / "gallery" / f"candidate_{index:04d}.jpg"
        )
        copy_immutable(source, destination)
        gallery_cards.append(
            {
                "candidate_id": candidate["candidate_id"],
                "shot_id": candidate["shot_id"],
                "shot_index": candidate["shot_index"],
                "first_frame": candidate["first_frame"],
                "image": destination.relative_to(BUNDLE_ROOT).as_posix(),
            }
        )

    annotation = {
        "schema_version": "kickclip.event_role_annotation.v1_1_2a",
        "ranking_artifact_id": report["ranking_artifact_id"],
        "ranking_package": ranking["package"],
        "ranking_schema_version": ranking["schema_version"],
        "ranking_policy_sha256": ranking["freeze"][
            "ranking_policy_sha256"
        ],
        "event_id": EVENT_ID,
        "scene_id": EVENT_ID,
        "annotation_status": "PENDING_INDEPENDENT_REVIEW",
        "primary_actor_visible": None,
        "primary_actor_in_candidate_set": None,
        "primary_actor_candidate_ids": [],
        "directly_related_candidate_ids": [],
        "broadcast_closeup_non_actor_candidate_ids": [],
        "unrelated_candidate_ids": [],
        "uncertain_candidate_ids": [],
        "actor_missing_reason": None,
        "reviewer": None,
        "reviewed_at": None,
        "allowed_roles": [
            "PRIMARY_EVENT_ACTOR",
            "DIRECTLY_RELATED_PLAYER",
            "BROADCAST_CLOSEUP_NON_ACTOR",
            "UNRELATED_PLAYER",
            "UNCERTAIN",
            "ACTOR_NOT_IN_CANDIDATE_SET",
        ],
        "ground_truth_used_for_ranking": False,
    }
    annotation_path = BUNDLE_ROOT / "human_annotation_template.json"
    annotation_path.write_text(
        json.dumps(annotation, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    top_html = []
    for card in top_cards:
        top_html.append(
            f"""
            <article class="top-card">
              <h3>Rank {card['rank']}</h3>
              <code>{html.escape(card['candidate_id'])}</code>
              <p>shot: {html.escape(card['shot_id'])}</p>
              <img src="{card['contact']}" alt="contact sheet">
              <img src="{card['gif']}" alt="tracklet preview">
              <p><a href="{card['video']}">원본 tracklet MP4 열기</a></p>
              <p>reason: {html.escape(', '.join(card['reasons']))}</p>
              <p>risk: {html.escape(', '.join(card['risks']))}</p>
            </article>
            """
        )
    gallery_html = []
    for card in gallery_cards:
        gallery_html.append(
            f"""
            <article class="gallery-card" data-shot="{card['shot_index']}">
              <img src="{card['image']}" alt="candidate thumbnail">
              <code>{html.escape(card['candidate_id'])}</code>
              <span>{html.escape(card['shot_id'])}, frame {card['first_frame']}</span>
            </article>
            """
        )
    review_html = BUNDLE_ROOT / "human_event_role_review.html"
    review_html.write_text(
        """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Representative Goal — Human Event Role Review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111827;color:#f9fafb}
h1,h2{margin-top:30px}.notice{padding:14px;background:#312e81;border-radius:10px}
.top{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
.top-card,.gallery-card{background:#1f2937;padding:14px;border-radius:10px}
.top-card img{display:block;width:100%;margin-top:10px;border-radius:8px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.gallery-card img{width:100%;height:180px;object-fit:contain;background:#030712}
code,span{display:block;overflow-wrap:anywhere;margin-top:7px}
a{color:#93c5fd}
</style></head><body>
<h1>Representative Goal — Offline Human Review</h1>
<div class="notice">이 페이지는 ranking 입력이 아닌 사후 사람 검토용입니다.
선수 이름·등번호·정답 candidate는 production feature에 사용되지 않았습니다.
먼저 전체 234명 안에 실제 주인공이 있는지 확인한 뒤 annotation template에
역할을 기록하세요.</div>
<h2>Provisional Top 5</h2><section class="top">
"""
        + "\n".join(top_html)
        + """
</section><h2>Full gallery fallback (scene/shot/time order)</h2>
<section class="gallery">
"""
        + "\n".join(gallery_html)
        + """
</section></body></html>
""",
        encoding="utf-8",
        newline="\n",
    )

    manifest = {
        "schema_version": "kickclip.human_event_role_review_bundle.v1",
        "project_id": PROJECT_ID,
        "revision_id": REVISION_ID,
        "event_id": EVENT_ID,
        "scene_id": EVENT_ID,
        "discovery_id": report["discovery_id"],
        "ranking_artifact_id": report["ranking_artifact_id"],
        "candidate_count": len(gallery_cards),
        "top5": top_cards,
        "media": media_manifest,
        "html": {
            "path": review_html.relative_to(BUNDLE_ROOT).as_posix(),
            "sha256": sha256_file(review_html),
        },
        "annotation_template": {
            "path": annotation_path.relative_to(BUNDLE_ROOT).as_posix(),
            "sha256": sha256_file(annotation_path),
        },
        "automatic_target_confirmation": False,
    }
    manifest_path = BUNDLE_ROOT / "human_review_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with SessionLocal() as db:
        revision = db.get(HighlightRevision, REVISION_ID)
        if revision is None:
            raise RuntimeError("HighlightRevision is missing.")
        artifact_ids = {}
        for artifact_type, path, mime in (
            ("EVENT_ROLE_REVIEW_HTML", review_html, "text/html"),
            (
                "EVENT_ROLE_ANNOTATION_TEMPLATE",
                annotation_path,
                "application/json",
            ),
            (
                "EVENT_ROLE_REVIEW_BUNDLE_MANIFEST",
                manifest_path,
                "application/json",
            ),
        ):
            artifact = db.scalar(
                select(Artifact).where(
                    Artifact.project_id == PROJECT_ID,
                    Artifact.artifact_type == artifact_type,
                )
            )
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if artifact is None:
                artifact = Artifact(
                    match_id=MATCH_ID,
                    project_id=PROJECT_ID,
                    analysis_job_id=None,
                    artifact_type=artifact_type,
                    file_path=relative,
                    mime_type=mime,
                    metadata_={
                        "revision_id": REVISION_ID,
                        "scene_id": EVENT_ID,
                        "ranking_artifact_id": report["ranking_artifact_id"],
                        "sha256": sha256_file(path),
                        "annotation_status": (
                            "PENDING_INDEPENDENT_REVIEW"
                        ),
                    },
                )
                db.add(artifact)
                db.flush()
            artifact_ids[artifact_type] = artifact.artifact_id
        revision.options = {
            **(revision.options or {}),
            "representative_goal_human_review": {
                "status": "PENDING_INDEPENDENT_REVIEW",
                "html_artifact_id": artifact_ids[
                    "EVENT_ROLE_REVIEW_HTML"
                ],
                "annotation_template_artifact_id": artifact_ids[
                    "EVENT_ROLE_ANNOTATION_TEMPLATE"
                ],
                "bundle_manifest_artifact_id": artifact_ids[
                    "EVENT_ROLE_REVIEW_BUNDLE_MANIFEST"
                ],
                "candidate_count": len(gallery_cards),
                "automatic_target_confirmation": False,
            },
        }
        db.commit()

    print(
        json.dumps(
            {
                "html": str(review_html),
                "annotation_template": str(annotation_path),
                "manifest": str(manifest_path),
                "candidate_count": len(gallery_cards),
                "top5_tracklet_artifacts": len(top_cards),
                "artifact_ids": artifact_ids,
                "annotation_status": "PENDING_INDEPENDENT_REVIEW",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
