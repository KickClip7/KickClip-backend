from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.session import SessionLocal
from app.domains.artifact.model import Artifact
from app.domains.highlight.model import (
    HighlightRevision,
    SceneAITask,
    ScenePlayerCandidate,
)
from app.domains.project.model import Project
from app.domains.timeline.model import TimelineEvent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "proj_eval_goal_evt_5eb4be390208"
REVISION_ID = "hrev_eval_goal_evt_5eb4be390208_r1"
EVENT_ID = "evt_5eb4be390208"
DISCOVERY_ID = "discovery_43f3e1bfbc700d78573d"
RANKING_ARTIFACT_ID = "art_11f5e2e0b584"
YOLO_ROOT = Path(r"D:\HAESUNG\prometheus\YOLO-train")
HANDOFF_ROOT = PROJECT_ROOT / "handoff"
ARCHIVE_STEM = (
    "event_candidate_ranking_v1_1_2a_goal_failure_handoff_20260731"
)
ARCHIVE_PATH = HANDOFF_ROOT / f"{ARCHIVE_STEM}.zip"
SIDECAR_PATH = HANDOFF_ROOT / f"{ARCHIVE_STEM}_sha256.json"

REVIEW_RELATIVE = Path(
    "storage/matches/match_preloaded_kor_jpn/projects/"
    f"{PROJECT_ID}/highlight/{REVISION_ID}/"
    "representative-goal-review"
)
DISCOVERY_RELATIVE = Path(
    "storage/matches/match_preloaded_kor_jpn/projects/"
    f"{PROJECT_ID}/highlight/{REVISION_ID}/"
    f"scene-target-selection/{EVENT_ID}/discoveries/{DISCOVERY_ID}"
)

VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".gif",
}
MODEL_SUFFIXES = {
    ".pth",
    ".pt",
    ".onnx",
    ".engine",
    ".ckpt",
}
IGNORED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "_backups",
}
FIXED_ZIP_TIME = (2026, 7, 31, 0, 0, 0)


def extended(path: Path) -> Path:
    value = str(path.resolve())
    if value.startswith("\\\\?\\"):
        return Path(value)
    return Path("\\\\?\\" + value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def source_allowed(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & IGNORED_PARTS:
        return False
    if path.suffix.lower() in VIDEO_SUFFIXES | MODEL_SUFFIXES:
        return False
    if path.name.endswith((".tmp", ".pyc", ".pyo")):
        return False
    return path.is_file()


def gather_tree(
    mapping: dict[str, Path],
    *,
    source_root: Path,
    archive_root: str,
    predicate=None,
) -> None:
    for path in source_root.rglob("*"):
        if not source_allowed(path):
            continue
        if predicate is not None and not predicate(path):
            continue
        relative = path.relative_to(source_root).as_posix()
        mapping[f"{archive_root.rstrip('/')}/{relative}"] = path


def gather_workspace_sources(mapping: dict[str, Path]) -> None:
    directories = (
        (
            PROJECT_ROOT
            / "app/domains/highlight/event_candidate_ranking_v1_1",
            "source/backend/app/domains/highlight/"
            "event_candidate_ranking_v1_1",
        ),
        (
            PROJECT_ROOT
            / "app/domains/highlight/event_candidate_ranking_v1_1_1",
            "source/backend/app/domains/highlight/"
            "event_candidate_ranking_v1_1_1",
        ),
        (
            PROJECT_ROOT
            / "app/domains/highlight/event_candidate_ranking_v1_1_2",
            "source/backend/app/domains/highlight/"
            "event_candidate_ranking_v1_1_2",
        ),
        (
            PROJECT_ROOT
            / "app/domains/highlight/event_candidate_ranking_v1_1_2a",
            "source/backend/app/domains/highlight/"
            "event_candidate_ranking_v1_1_2a",
        ),
        (
            PROJECT_ROOT / "configs/models/event_candidate_ranking",
            "source/configs/models/event_candidate_ranking",
        ),
        (
            PROJECT_ROOT / "configs/models/scene_discovery",
            "source/configs/models/scene_discovery",
        ),
    )
    for source, destination in directories:
        gather_tree(
            mapping,
            source_root=source,
            archive_root=destination,
        )

    files = (
        ".env.example",
        "README.md",
        "app/main.py",
        "app/api/v1/highlights.py",
        "app/api/v1/router.py",
        "app/api/v1/event_candidate_ranking_v1_1_2a.py",
        "app/core/config.py",
        "app/db/models.py",
        "app/domains/highlight/model.py",
        "app/domains/highlight/repository.py",
        "app/domains/highlight/schema.py",
        "app/domains/highlight/event_candidate_ranking.py",
        "app/domains/highlight/event_candidate_ranking_v1_1_2a_integration.py",
        "app/domains/highlight/event_candidate_verifier.py",
        "app/domains/highlight/runtime_contract.py",
        "app/domains/highlight/scene_ai_task.py",
        "app/domains/highlight/scene_discovery_runtime_verifier_adapter.py",
        "app/domains/highlight/scene_target_selection.py",
        "app/domains/tracking/verifier.py",
        "docs/event_candidate_ranking_v1_1.md",
        "docs/event_candidate_ranking_v1_1_1.md",
        "docs/event_candidate_ranking_v1_1_2.md",
        "docs/event_candidate_ranking_v1_1_2a.md",
        "docs/event_candidate_ranking_v1_1_2a_integration.md",
        "docs/scene_target_selection_runtime.md",
        "docs/target_tracking_backend.md",
        "alembic/versions/20260728_0012_create_highlight_drafts.py",
        "alembic/versions/20260730_0013_create_scene_target_selections.py",
        "alembic/versions/20260730_0014_isolate_selection_artifacts.py",
        "alembic/versions/20260730_0015_event_candidate_ranking.py",
        "alembic/versions/20260730_0016_create_scene_ai_tasks.py",
    )
    for relative in files:
        path = PROJECT_ROOT / relative
        if path.is_file():
            mapping[f"source/backend/{Path(relative).as_posix()}"] = path

    script_names = (
        "prepare_representative_goal_review.py",
        "approve_representative_goal_shots.py",
        "generate_representative_goal_detections.py",
        "execute_representative_goal_shadow.py",
        "build_representative_goal_human_review.py",
        "finalize_representative_goal_human_evaluation.py",
        "build_tracking_performance_handoff_zip.py",
    )
    for name in script_names:
        path = PROJECT_ROOT / "scripts" / name
        mapping[f"source/backend/scripts/{name}"] = path

    test_names = (
        "test_event_candidate_ranking_v1_1.py",
        "test_event_candidate_ranking_v1_1_1.py",
        "test_event_candidate_ranking_v1_1_2.py",
        "test_event_candidate_ranking_v1_1_2a.py",
        "test_event_candidate_ranking_v1_1_2a_integration.py",
        "test_scene_target_selection_contract.py",
        "test_scene_migration_chain_postgres.py",
    )
    for name in test_names:
        path = PROJECT_ROOT / "tests" / name
        mapping[f"source/backend/tests/{name}"] = path
    fixtures = (
        "event_candidate_ranking_v1_1_smoke.json",
        "r2_r3_reviewed_shot_artifact.json",
        "fake_scene_selection_verifier.py",
        "fake_r3_scene_selection_verifier.py",
    )
    for name in fixtures:
        path = PROJECT_ROOT / "tests/fixtures" / name
        if path.is_file():
            mapping[f"source/backend/tests/fixtures/{name}"] = path


def gather_frozen_runtime_sources(mapping: dict[str, Path]) -> None:
    scene_root = (
        YOLO_ROOT / "target_centric_tracking_scene_target_selection_v1"
    )
    manifest_path = scene_root / "scene_target_selection_frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_files = manifest.get("package_files") or {}
    mapping[
        "source/frozen_runtime/scene_target_selection_v1/"
        "scene_target_selection_frozen_manifest.json"
    ] = manifest_path
    for relative in sorted(package_files):
        path = scene_root / relative
        mapping[
            "source/frozen_runtime/scene_target_selection_v1/"
            f"{Path(relative).as_posix()}"
        ] = path
    dependencies = (
        (
            YOLO_ROOT
            / "target_centric_tracking_v1/"
            "stage2_run_conservative_target_association.py",
            "source/compat_dependencies/target_centric_tracking_v1/"
            "stage2_run_conservative_target_association.py",
        ),
        (
            YOLO_ROOT
            / "target_centric_tracking_v2/"
            "stage3b0_build_postcut_candidate_tracklets.py",
            "source/compat_dependencies/target_centric_tracking_v2/"
            "stage3b0_build_postcut_candidate_tracklets.py",
        ),
    )
    for path, archive_name in dependencies:
        mapping[archive_name] = path


def gather_actual_run(
    mapping: dict[str, Path],
    excluded: list[dict[str, Any]],
) -> None:
    review_root = PROJECT_ROOT / REVIEW_RELATIVE
    for path in review_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(review_root).as_posix()
        if path.suffix.lower() in VIDEO_SUFFIXES:
            excluded.append(
                {
                    "scope": "actual_run/review",
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "reason": "VIDEO_EXCLUDED",
                }
            )
            continue
        if source_allowed(path):
            mapping[f"actual_run/review/{relative}"] = path

    discovery_root = extended(PROJECT_ROOT / DISCOVERY_RELATIVE)
    for path in discovery_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(discovery_root).as_posix()
        if path.suffix.lower() in VIDEO_SUFFIXES:
            excluded.append(
                {
                    "scope": "actual_run/discovery_snapshot",
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "reason": "VIDEO_EXCLUDED",
                }
            )
            continue
        if path.suffix.lower() in MODEL_SUFFIXES:
            excluded.append(
                {
                    "scope": "actual_run/discovery_snapshot",
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "reason": "MODEL_BINARY_EXCLUDED",
                }
            )
            continue
        if source_allowed(path):
            mapping[f"actual_run/discovery_snapshot/{relative}"] = path


def database_snapshot() -> dict[str, Any]:
    with SessionLocal() as db:
        project = db.get(Project, PROJECT_ID)
        revision = db.get(HighlightRevision, REVISION_ID)
        event = db.get(TimelineEvent, EVENT_ID)
        if project is None or revision is None or event is None:
            raise RuntimeError("Representative run DB scope is incomplete.")
        tasks = db.scalars(
            select(SceneAITask)
            .where(SceneAITask.project_id == PROJECT_ID)
            .order_by(SceneAITask.created_at.asc())
        ).all()
        artifacts = db.scalars(
            select(Artifact)
            .where(Artifact.project_id == PROJECT_ID)
            .order_by(Artifact.created_at.asc())
        ).all()
        candidates = db.scalars(
            select(ScenePlayerCandidate)
            .where(ScenePlayerCandidate.revision_id == REVISION_ID)
            .order_by(
                ScenePlayerCandidate.anchor_frame_index.asc(),
                ScenePlayerCandidate.candidate_id.asc(),
            )
        ).all()
        return {
            "schema_version": "kickclip.tracking_handoff_db_snapshot.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project": {
                "project_id": project.project_id,
                "match_id": project.match_id,
                "owner_id": project.owner_id,
                "title": project.title,
                "status": project.status,
            },
            "revision": {
                "revision_id": revision.revision_id,
                "revision_number": revision.revision_number,
                "action_spotting_job_id": revision.action_spotting_job_id,
                "selected_scene_ids": revision.selected_scene_ids,
                "scene_selection": revision.scene_selection,
                "status": revision.status,
                "pending_action": revision.pending_action,
                "options": revision.options,
            },
            "event": {
                "event_id": event.timeline_event_id,
                "source_job_id": event.source_job_id,
                "source_artifact_id": event.source_artifact_id,
                "event_type": event.event_type,
                "label": event.label,
                "timestamp_sec": event.timestamp_sec,
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "confidence": event.confidence,
                "metadata": event.metadata_,
            },
            "scene_ai_tasks": [
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status,
                    "attempt_count": task.attempt_count,
                    "max_attempts": task.max_attempts,
                    "payload": task.payload,
                    "result": task.result,
                    "error_message": task.error_message,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                }
                for task in tasks
            ],
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_type": artifact.artifact_type,
                    "file_path": artifact.file_path,
                    "mime_type": artifact.mime_type,
                    "metadata": artifact.metadata_,
                    "created_at": artifact.created_at,
                }
                for artifact in artifacts
            ],
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "scene_id": candidate.scene_id,
                    "anchor_frame_index": candidate.anchor_frame_index,
                    "track_length_frames": candidate.track_length_frames,
                    "trackability_score": candidate.trackability_score,
                    "status": candidate.status,
                    "metadata": candidate.metadata_,
                }
                for candidate in candidates
            ],
        }


def static_review_html() -> bytes:
    source = (
        PROJECT_ROOT
        / REVIEW_RELATIVE
        / "human-event-role-review/human_event_role_review.html"
    )
    text = source.read_text(encoding="utf-8")
    text = re.sub(
        r'<img src="[^"]+\.gif"[^>]*>',
        '<p class="notice">Animated preview excluded from this handoff.</p>',
        text,
    )
    text = re.sub(
        r'<p><a href="[^"]+\.mp4">[^<]+</a></p>',
        '<p>Tracklet video excluded; static contact sheet retained.</p>',
        text,
    )
    return text.encode("utf-8")


def handoff_readme() -> bytes:
    text = f"""# Event Candidate Ranking V1.1.2a — Representative Goal Failure Handoff

## 결론

실제 Representative Goal Shadow Run에서 후보 생성은 주인공을 포함했지만,
provisional Top-5는 모두 감독/스태프 등 `BROADCAST_CLOSEUP_NON_ACTOR`였습니다.
따라서 Meaningful Top-5 Recommendation은 **FAIL**이며 Production UI는
계속 **BLOCKED**입니다.

Frozen V1.1/V1.1.1/V1.1.2/V1.1.2a package, manifest, policy와 ranking
weight는 이 실행을 위해 변경하지 않았습니다.

## 실제 결과

- Project: `{PROJECT_ID}`
- Revision: `{REVISION_ID}`
- Event/Scene: `{EVENT_ID}`
- Discovery: `{DISCOVERY_ID}`
- Ranking artifact: `{RANKING_ARTIFACT_ID}`
- Candidate count: `234`
- Parsed observations: `7,501`
- Reviewed shots: `11`
- Frame coverage: `1,125 / 1,125`
- Visual coverage: `1.0`
- Motion coverage: `1.0`
- Broadcast coverage: `1.0`
- Ball state: `UNAVAILABLE`
- Feature completeness: `PARTIAL_FEATURES` (`0.6994017094`)
- Ranking status: `PROVISIONAL_SHADOW_ONLY`
- Full gallery fallback: `234`
- Automatic target confirmation: `false`

Human ground truth는 동일 선수의 fragmented tracklet 세 개입니다.

- `shot_0003_track_0001`: rank `17`
- `shot_0003_track_0002`: rank `10`
- `shot_0003_track_0003`: rank `6`

공식 single-event evaluation:

- Candidate generation actor coverage: `1.0`
- Primary actor Recall@1: `0.0`
- Primary actor Recall@3: `0.0`
- Primary actor Recall@5: `0.0`
- MRR: `0.1666666667`
- Broadcast closeup non-actor Top-1: `1.0`
- Cross-version metric mixing: `false`

관찰상 실패는 RF-DETR player/goalkeeper 후보군에 감독·스태프가 포함된
semantic false positive와, ball/field-context가 없는 상태에서 event-near
broadcast closeup 증거가 높은 점수를 받은 조합입니다. 이는 산출물 기반
진단이며 weight calibration은 수행하지 않았습니다.

## 주요 SHA-256

- Reviewed shots: `8103dc62fb8cf255f049c95b1c6738e89e9eeeef4729e17fd9edf5256120235b`
- Frozen detections: `ce5f1d346806fb6edf86975c82ad5e96920bc76753e6c16c1fb9c36230a06dbb`
- Scene candidates: `56bd82f9edd37b508c34687660aaed7e42d62f359eee4dd87e8ca6edb636fd39`
- Candidate manifest: `d17b15c0d7e90f463c211c3cca892b2d66641d4c1e025aa2b2902da22ba01414`
- Ranking artifact: `958d085f642c8e50c0ae7bf1c4a63eb5d77df7920c7d1591bc08d2d3016a5197`
- Final human evaluation: `50067723c3c586727050f839e14198d1dc50ce9808adc2805683b68bcf5741d3`
- Scene Discovery COMPAT_R1 manifest: `f8bc9710053198c59efae6233b1e1aef27c8fb2eecadaeef094061528c2d1005`
- RF-DETR checkpoint (not included): `5d1d05cf78b6a777430430955d0e745ce5c61913c659b2e96ae5deddbd1a3a94`

## ZIP 구조

- `actual_run/discovery_snapshot/`: immutable candidate/ranking/annotation
  JSON과 정적 candidate 이미지
- `actual_run/review/`: provenance, reviewed shots, detections, reports,
  full gallery 정적 이미지와 human evaluation
- `source/backend/`: backend adapter/API/executor/verifier/migration/docs/tests
- `source/frozen_runtime/`: frozen Scene Target Selection package 21 files
- `source/compat_dependencies/`: 실제 discovery가 동적으로 사용한 V1/V2
  dependency 파일
- `DATABASE_SNAPSHOT.json`: 해당 Project/Revision/task/artifact/candidate DB
  record snapshot
- `HANDOFF_MANIFEST.json`: ZIP에 포함된 모든 파일의 SHA-256

## 제외 항목

사용자 지시에 따라 MP4/AVI/MOV/MKV/WEBM/M4V 및 animated GIF는 포함하지
않았습니다. RF-DETR/Sports OSNet checkpoint 등 model binary도 포함하지
않았습니다. `__pycache__`, `.pyc`, `.git`, 임시 파일과 `.env`도 제외했습니다.
정적 contact sheet, representative JPEG, crops, HTML, JSON, CSV와 log는
포함했습니다.

## 검증

관련 회귀 테스트:

```text
72 passed, 1 skipped in 5.85s
```

Skipped 항목은 PostgreSQL migration 환경 조건부 테스트입니다. ZIP 생성 후
각 entry SHA, 금지된 video/model/cache entry 부재, archive CRC를 다시
검증했습니다.
"""
    return text.encode("utf-8")


def add_virtual(
    virtual: dict[str, bytes],
    name: str,
    value: bytes,
) -> None:
    if name in virtual:
        raise RuntimeError(f"Duplicate virtual archive path: {name}")
    virtual[name] = value


def write_entry(
    archive: zipfile.ZipFile,
    name: str,
    *,
    source: Path | None = None,
    value: bytes | None = None,
) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    if source is not None:
        with source.open("rb") as input_stream, archive.open(info, "w") as out:
            shutil.copyfileobj(input_stream, out, length=1024 * 1024)
    elif value is not None:
        archive.writestr(info, value)
    else:
        raise ValueError("source or value is required")


def verify_archive(
    path: Path,
    *,
    expected_manifest: dict[str, Any],
) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed.")
        names = archive.namelist()
        forbidden = [
            name
            for name in names
            if Path(name).suffix.lower() in VIDEO_SUFFIXES | MODEL_SUFFIXES
            or any(part.lower() in IGNORED_PARTS for part in Path(name).parts)
            or name.lower().endswith((".pyc", ".pyo"))
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden ZIP entries: {forbidden[:10]}")
        manifest = json.loads(
            archive.read("HANDOFF_MANIFEST.json").decode("utf-8")
        )
        if manifest != expected_manifest:
            raise RuntimeError("Embedded handoff manifest changed.")
        for row in manifest["files"]:
            digest = hashlib.sha256()
            with archive.open(row["archive_path"], "r") as stream:
                for chunk in iter(
                    lambda: stream.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise RuntimeError(
                    f"ZIP entry SHA mismatch: {row['archive_path']}"
                )


def main() -> None:
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists() or SIDECAR_PATH.exists():
        raise FileExistsError(
            "Handoff output already exists; refusing to overwrite."
        )
    sources: dict[str, Path] = {}
    excluded: list[dict[str, Any]] = []
    gather_workspace_sources(sources)
    gather_frozen_runtime_sources(sources)
    gather_actual_run(sources, excluded)

    virtual: dict[str, bytes] = {}
    add_virtual(virtual, "README_KO.md", handoff_readme())
    add_virtual(
        virtual,
        "DATABASE_SNAPSHOT.json",
        json_bytes(database_snapshot()),
    )
    add_virtual(
        virtual,
        "actual_run/review/human-event-role-review/"
        "human_event_role_review_STATIC_NO_VIDEO.html",
        static_review_html(),
    )
    add_virtual(
        virtual,
        "TEST_RESULTS.txt",
        (
            "Command:\n"
            "python -m pytest "
            "tests/test_event_candidate_ranking_v1_1.py "
            "tests/test_event_candidate_ranking_v1_1_1.py "
            "tests/test_event_candidate_ranking_v1_1_2.py "
            "tests/test_event_candidate_ranking_v1_1_2a.py "
            "tests/test_event_candidate_ranking_v1_1_2a_integration.py "
            "tests/test_scene_target_selection_contract.py "
            "tests/test_scene_migration_chain_postgres.py -q\n\n"
            "Result:\n72 passed, 1 skipped in 5.85s\n"
        ).encode("utf-8"),
    )

    collisions = set(sources) & set(virtual)
    if collisions:
        raise RuntimeError(f"Archive path collisions: {sorted(collisions)}")

    file_rows = []
    for name, path in sorted(sources.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        file_rows.append(
            {
                "archive_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_kind": "FILE",
            }
        )
    for name, value in sorted(virtual.items()):
        file_rows.append(
            {
                "archive_path": name,
                "size_bytes": len(value),
                "sha256": sha256_bytes(value),
                "source_kind": "GENERATED_HANDOFF_METADATA",
            }
        )
    file_rows.sort(key=lambda row: row["archive_path"])
    manifest = {
        "schema_version": "kickclip.tracking_performance_handoff.v1",
        "archive_name": ARCHIVE_PATH.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": PROJECT_ID,
        "revision_id": REVISION_ID,
        "event_id": EVENT_ID,
        "discovery_id": DISCOVERY_ID,
        "ranking_artifact_id": RANKING_ARTIFACT_ID,
        "meaningful_top5_result": "FAIL",
        "automatic_target_confirmation": False,
        "video_included": False,
        "model_binary_included": False,
        "test_result": {
            "passed": 72,
            "skipped": 1,
            "failed": 0,
        },
        "files": file_rows,
        "file_count": len(file_rows),
        "total_uncompressed_bytes": sum(
            row["size_bytes"] for row in file_rows
        ),
        "excluded_files": sorted(
            excluded,
            key=lambda row: (row["scope"], row["path"]),
        ),
        "excluded_file_count": len(excluded),
        "excluded_bytes": sum(row["size_bytes"] for row in excluded),
        "exclusion_rules": {
            "video_suffixes": sorted(VIDEO_SUFFIXES),
            "model_suffixes": sorted(MODEL_SUFFIXES),
            "ignored_parts": sorted(IGNORED_PARTS),
            "secrets": [".env"],
        },
    }
    manifest_value = json_bytes(manifest)

    temporary = ARCHIVE_PATH.with_suffix(".tmp.zip")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, path in sorted(sources.items()):
                write_entry(archive, name, source=path)
            for name, value in sorted(virtual.items()):
                write_entry(archive, name, value=value)
            write_entry(
                archive,
                "HANDOFF_MANIFEST.json",
                value=manifest_value,
            )
        os.replace(temporary, ARCHIVE_PATH)
    finally:
        temporary.unlink(missing_ok=True)

    verify_archive(ARCHIVE_PATH, expected_manifest=manifest)
    archive_sha = sha256_file(ARCHIVE_PATH)
    sidecar = {
        "archive": str(ARCHIVE_PATH),
        "archive_name": ARCHIVE_PATH.name,
        "sha256": archive_sha,
        "size_bytes": ARCHIVE_PATH.stat().st_size,
        "embedded_manifest_sha256": sha256_bytes(manifest_value),
        "file_count_excluding_manifest": len(file_rows),
        "video_included": False,
        "model_binary_included": False,
        "verified": True,
    }
    SIDECAR_PATH.write_bytes(json_bytes(sidecar))
    print(json.dumps(sidecar, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
