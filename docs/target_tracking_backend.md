# Target-Centric Tracking 백엔드 통합

KickClip의 target tracking은 전체 선수 Global ID를 생성하는 기능이 아니다. 사용자가
선택한 한 선수를 같은 shot 안에서 추적하고, 카메라 컷 뒤에는 frozen appearance
evidence로 후보를 찾되 확실하지 않으면 다른 선수로 전환하지 않고 사용자 확인을
기다린다.

백엔드는 AI 알고리즘을 포함하거나 재구현하지 않는다.

```text
Tracking API
→ tracking_jobs DB row
→ durable local executor
→ yolo-soccer Python subprocess
→ target_centric_tracking_e2e_v1 runner
→ pipeline_state.json 수집
→ 인증된 timeline/artifact API
```

## 안전 계약

- 운영 API는 `assisted` reacquisition만 허용한다.
- `NEEDS_CONFIRMATION`과 runner exit code 3은 실패가 아니다.
- `COMPLETE_WITH_SAFE_BLOCK`은 정상적인 안전 완료다.
- `SEARCHING`, `AMBIGUOUS`, `ABSENT`, `LOST`, `TERMINATED`에는 bbox가 없어야 한다.
- 후보 rank 1을 자동 승인하지 않는다.
- 카메라 컷 전후 motion track을 연결하지 않는다.
- V1/V2 코드, frozen manifest, checkpoint, threshold, ranking weight, gate를 수정하지 않는다.
- 원본 영상 및 tracking output의 절대 경로를 API에 노출하지 않는다.

## Windows 로컬 설치

첨부한 `target_centric_tracking_e2e_v1` 폴더를 AI 프로젝트 루트에 배치한다.
백엔드 저장소로 복사하지 않는다.

```text
<YOLO_TRAIN_ROOT>\
  target_centric_tracking_e2e_v1\
    run_target_centric_pipeline.py
    verify_e2e_installation.py
    e2e_output_schema.json
```

`yolo-soccer` 환경의 Python 경로는 다음 명령으로 확인한다.

```powershell
conda env list
conda run -n yolo-soccer python -c "import sys; print(sys.executable)"
```

로컬 `.env`에만 실제 절대 경로를 기록한다.

```dotenv
TRACKING_ENABLED=true
TRACKING_PROJECT_ROOT="<YOLO_TRAIN_ROOT>"
TRACKING_E2E_SCRIPT_PATH="<YOLO_TRAIN_ROOT>\target_centric_tracking_e2e_v1\run_target_centric_pipeline.py"
TRACKING_VERIFY_SCRIPT_PATH="<YOLO_TRAIN_ROOT>\target_centric_tracking_e2e_v1\verify_e2e_installation.py"
TRACKING_PYTHON_EXECUTABLE="<CONDA_ROOT>\envs\yolo-soccer\python.exe"
TRACKING_OUTPUT_ROOT="<YOLO_TRAIN_ROOT>\runs\target_centric_tracking_e2e_v1"
TRACKING_DEVICE="cuda"
TRACKING_REACQUISITION_MODE="assisted"
TRACKING_MAX_CONCURRENT_JOBS=1
TRACKING_PROCESS_TIMEOUT_SECONDS=21600
TRACKING_VERIFY_TIMEOUT_SECONDS=300
TRACKING_PREVIEW_ENABLED=true
```

설치 verifier를 직접 실행한다.

```powershell
& "<CONDA_ROOT>\envs\yolo-soccer\python.exe" `
  "<YOLO_TRAIN_ROOT>\target_centric_tracking_e2e_v1\verify_e2e_installation.py" `
  --project-root "<YOLO_TRAIN_ROOT>"
```

백엔드에서도 인증 후 진단할 수 있다.

```http
GET /api/v1/tracking/diagnostics?refresh=true
Authorization: Bearer <access-token>
```

검증 실패는 백엔드 전체를 중단시키지 않는다. tracking job 생성만
`503 TRACKING_RUNTIME_UNAVAILABLE`로 거절하며 fake 결과를 반환하지 않는다.

## DB migration

```powershell
alembic upgrade head
alembic heads
```

head는 `20260730_0016` 하나여야 한다. `tracking_jobs`는 실행 입력, 소유권,
backend/pipeline 상태, pending action, 내부 경로, provenance, process 결과와 시간을
보존한다. frame별 row를 생성하지 않고 원본 `target_timeline.json`을 artifact로
보존한다. 기존 action spotting `timeline_events`와 분리되어 있다.

## API

모든 endpoint에는 Bearer 인증이 필요하다. `media_asset_id`의 Match 소유자가 아니면
다른 사용자의 리소스 존재 여부를 드러내지 않도록 404를 반환한다.

### 작업 시작

```http
POST /api/v1/tracking/jobs
Content-Type: application/json
Authorization: Bearer <access-token>

{
  "media_asset_id": "asset_...",
  "project_id": "proj_...",
  "initial_bbox_xyxy": [1290, 727, 1354, 871],
  "bbox_format": "xyxy_pixels",
  "reacquisition_mode": "assisted"
}
```

응답은 GPU 실행을 기다리지 않고 `202`와 job ID를 반환한다.

```json
{
  "job_id": "trk_...",
  "status": "QUEUED",
  "status_url": "/api/v1/tracking/jobs/trk_..."
}
```

### 상태 조회

```http
GET /api/v1/tracking/jobs/{job_id}
Authorization: Bearer <access-token>
```

backend 상태:

| 상태 | 의미 |
|---|---|
| `QUEUED` | DB에 저장되었고 executor claim 대기 |
| `RUNNING` | 외부 runtime 실행 중 |
| `WAITING_MEMORY_REVIEW` | pre-cut target memory 확인 필요 |
| `WAITING_CROSS_SHOT_CONFIRMATION` | 후보 선택 또는 target 부재 확인 필요 |
| `WAITING_SEGMENT_REVIEW` | `SEGMENT`/`STAGE2*` 시각 검수 필요 |
| `COMPLETED` | 전체 target timeline 완료 |
| `COMPLETED_SAFE_BLOCK` | 다른 선수로 전환하지 않고 안전하게 종료 |
| `FAILED` | 설치 외 runtime/계약/timeout fatal 오류 |
| `CANCELLED` | 취소 상태(현재 공개 cancel API 없음) |

### Memory/segment review

```http
POST /api/v1/tracking/jobs/{job_id}/reviews
Content-Type: application/json
Authorization: Bearer <access-token>

{
  "stage": "MEMORY",
  "decision": "approve",
  "note": "contact sheet와 tracking preview에서 모두 선택 선수 확인"
}
```

거절은 `"decision": "reject"`를 사용한다. 가능한 stage는 `MEMORY`, `SEGMENT`,
`STAGE2B`, `STAGE2D`, `STAGE2D1`, `STAGE2D2`다. 현재
`pipeline_state.json`이 요청한 정확한 stage만 허용된다.

### Cross-shot candidate 선택

```http
POST /api/v1/tracking/jobs/{job_id}/ambiguities/{ambiguity_id}/confirm
Content-Type: application/json
Authorization: Bearer <access-token>

{
  "decision": "candidate",
  "candidate_id": "shot_0004_track_0003",
  "note": "contact sheet에서 target 확인"
}
```

현재 review candidate 목록에 없는 ID는 runner에 전달하지 않는다.

### Target 부재 확인

```http
POST /api/v1/tracking/jobs/{job_id}/ambiguities/{ambiguity_id}/confirm
Content-Type: application/json
Authorization: Bearer <access-token>

{
  "decision": "absent",
  "note": "이 shot에는 target이 보이지 않음"
}
```

동일한 review/confirmation 요청은 action key로 idempotent하게 처리하며, 하나의 job에
동시에 두 resume process가 실행되지 않는다.

### Timeline

```http
GET /api/v1/tracking/jobs/{job_id}/timeline?start_frame=0&end_frame=500
Authorization: Bearer <access-token>
```

range는 양끝을 포함한다. 응답은 `kickclip.target_centric_e2e.v1`을 유지하되
`video.path`를 `media_asset:<id>`로 바꾸고 다른 절대 경로를 제거한다. backend는
불확실 상태의 non-null bbox를 발견하면 임의로 고치지 않고 output contract 오류로
거절한다.

### Artifact

```http
GET /api/v1/tracking/jobs/{job_id}/artifacts
Authorization: Bearer <access-token>
```

응답의 `url`만 다운로드에 사용한다.

```http
GET /api/v1/tracking/jobs/{job_id}/artifacts/{artifact_key}
Authorization: Bearer <access-token>
```

artifact key는 backend가 `pipeline_state.json`에서 생성한 allowlist만 허용한다.
경로·파일명을 클라이언트가 지정할 수 없다. JSON artifact는 원본을 그대로 보존하되
다운로드용 사본에서 Windows/Unix 절대 경로를 제거한다. memory/segment review 파일이
외부 V1/V2 run 디렉터리에 있으면 job root 아래 인증 영역으로 복사한 뒤 제공한다.

## 실행·복구·동시성

- HTTP handler는 subprocess 완료를 기다리지 않는다.
- executor는 `tracking_jobs.status = QUEUED`를 조건부 update해 한 worker만 claim한다.
- 기본 GPU concurrency는 1이다.
- PostgreSQL advisory slot lock이 여러 backend process 사이의 GPU 동시성을 제한한다.
- job별 in-process active set과 DB claim이 중복 resume을 차단한다.
- stdout/stderr는 `TRACKING_OUTPUT_ROOT/_backend_process_logs/<test_name>/`에 보존하며
  client artifact로 제공하지 않는다.
- subprocess는 `shell=False`와 argument list를 사용한다.
- timeout 또는 backend shutdown 시 process group/tree를 종료한다.
- active PID는 실행 중에만 저장하고 종료 후 지운다. 마지막 PID는 audit metadata에만
  남겨 OS PID 재사용을 실행 상태로 오인하지 않는다.
- startup에서 `QUEUED`/`RUNNING` row를 reconcile한다. terminal/waiting JSON state가
  있으면 DB를 복구하고, 안전하게 resume할 수 있는 RUNNING state는 resume하며,
  초기 불완전 run은 같은 server-generated test name으로 overwrite 재시작한다.

현재 executor는 durable local-development adapter다. production queue로 전환할 때는
HTTP/service와 `TrackingProcessRunner`를 유지하고 `TrackingJobExecutor.submit` 및
startup reconcile을 Celery/RQ/Dramatiq worker enqueue/claim으로 교체한다. worker도
동일한 DB claim, advisory GPU slot, state mapper를 사용해야 한다.

## 테스트

CUDA 없는 CI에서는 production runner를 바꾸지 않고
`tests/fixtures/fake_tracking_runner.py`를 별도 executable로 실행한다.

```powershell
python -m unittest discover -s tests -p "test_target_tracking.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

fake fixture는 memory review, cross-shot candidate, candidate/absent resume,
`COMPLETE_WITH_SAFE_BLOCK`, fatal, timeout 계약을 파일과 exit code로 재현한다.
production Settings는 이 fixture를 참조하지 않는다.

## 실제 Betis GPU smoke test

먼저 ZIP 폴더 배치와 verifier PASS를 확인한 뒤 AI 프로젝트에서 직접 실행한다.

```powershell
& "<CONDA_ROOT>\envs\yolo-soccer\python.exe" `
  "<YOLO_TRAIN_ROOT>\target_centric_tracking_e2e_v1\run_target_centric_pipeline.py" `
  --project-root "<YOLO_TRAIN_ROOT>" `
  --video "<YOLO_TRAIN_ROOT>\sample_videos\betis_vs_barcelona_2025_26_01h06m53s_23s.mp4" `
  --test-name "betis_barcelona_target_e2e" `
  --initial-bbox 1290 727 1354 871 `
  --device cuda `
  --reacquisition-mode assisted `
  --output-root "<YOLO_TRAIN_ROOT>\runs\target_centric_tracking_e2e_v1"
```

API smoke test에서는 같은 영상이 먼저 KickClip `MediaAsset`으로 등록되어 있어야 한다.
클라이언트는 위 영상 절대 경로가 아니라 해당 `media_asset_id`를 전송한다.

가중치, 원본 영상, AI 소스 복사본, `runs/` 결과는 Git에 커밋하지 않는다.
