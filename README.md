# KickClip Studio Backend

KickClip Studio Backend는 축구 경기 영상을 업로드하면 AI가 주요 장면을 분석하고, 사용자가 하나의 하이라이트 프로젝트 안에서 장면 선택과 선택적 선수 포커스를 이어서 편집해 숏츠 영상으로 내보낼 수 있게 하는 FastAPI 기반 백엔드입니다.

영상 업로드와 미디어 관리부터 AI 분석 작업, 프레임 단위 선수 추적, 타임라인 편집, Agent 기반 ClipPlan 생성, FFmpeg 렌더링까지 KickClip Studio의 전체 제작 흐름을 하나의 API로 제공합니다.

## 주요 기능

- 이메일 기반 회원가입, 로그인, Access/Refresh Token 인증
- 축구 경기 영상 업로드 및 MediaAsset 관리
- 브라우저 재생용 서명 URL, 스트리밍, 다운로드
- SoccerNet 기반 Action Spotting과 이벤트 타임라인 생성
- 사용자가 선택한 선수의 target-centric tracking
- 카메라 컷 이후 안전한 재탐색과 사용자 확인 기반 resume
- 경기별 선수, 이벤트, 편집 상태 통합 조회
- 자연어 프롬프트를 이용한 하이라이트 ClipPlan 생성
- 제목, 해시태그, 썸네일 및 내보내기 옵션 추천
- FFmpeg 기반 숏폼 영상 렌더링과 결과 다운로드

## 서비스 흐름

```mermaid
flowchart LR
    A["경기 영상 업로드"] --> B["MediaAsset 생성"]
    B --> C["Action Spotting 1회 실행/재사용"]
    C --> D["HighlightScene 정규화 및 선택"]
    D --> E{"선수 포커스?"}
    E -->|"아니오"| G["Agent ClipPlan"]
    E -->|"예"| F["선택 Scene 후보 확인 및 Tracking"]
    F --> G
    G --> H["미리보기 및 내보내기 설정"]
    H --> I["FFmpeg 렌더링"]
    I --> J["숏츠 영상 다운로드"]
```

## Target-Centric Tracking

KickClip의 tracking은 모든 선수의 Global ID를 만드는 기능이 아니라, 사용자가 선택한 한 선수를 안전하게 따라가는 기능입니다.

- 같은 shot 안에서는 선택한 선수를 정밀 추적합니다.
- 카메라 컷을 motion 정보만으로 강제 연결하지 않습니다.
- 확실하지 않은 프레임은 `SEARCHING`, `AMBIGUOUS`, `ABSENT` 등으로 남기며 bbox를 생성하지 않습니다.
- 카메라 컷 이후 후보가 여러 명이면 사용자 확인을 기다립니다.
- 사용자가 후보 또는 target 부재를 확인하면 기존 작업을 이어서 실행합니다.
- 완성된 frame-level timeline은 선수 중심 crop, zoom, blur 등 후속 편집 작업에서 사용할 수 있습니다.

Tracking AI는 별도 frozen runtime을 subprocess로 실행합니다. 모델과 checkpoint를 백엔드 저장소에 포함하지 않으며, 미설치 환경에서는 나머지 백엔드 기능을 그대로 사용할 수 있습니다.

자세한 설정과 상태 계약은 [Target Tracking 통합 문서](docs/target_tracking_backend.md)를 참고하세요.

## 통합 하이라이트 워크플로

일반 하이라이트와 선수 중심 하이라이트는 별도 제품이나 별도 Project가 아닙니다.
Action Spotting 결과는 source video, video hash, model version, prediction policy
version 조합으로 한 번만 실행·재사용하고, Project별 편집 상태는
`HighlightRevision`으로 보존합니다.

```text
POST /api/v1/projects/{project_id}/highlight/analyze
GET  /api/v1/projects/{project_id}/highlight/scenes
POST /api/v1/projects/{project_id}/highlight/scenes/select
POST /api/v1/projects/{project_id}/highlight/player-focus
GET  /api/v1/projects/{project_id}/highlight/player-candidates
POST /api/v1/projects/{project_id}/highlight/player-focus/select
POST /api/v1/projects/{project_id}/highlight/scenes/{scene_id}/player-confirmation
GET  /api/v1/projects/{project_id}/highlight/status
POST /api/v1/projects/{project_id}/highlight/clip-plan
POST /api/v1/projects/{project_id}/highlight/render
```

선수 후보는 선택된 scene 안에서만 생성되며 실제 선수 이름이나 경기 전체 Global
ID로 자동 확정되지 않습니다. 사용자가 scene-local 후보를 확인한 뒤에만 frozen
target-centric runner가 실행됩니다. 자세한 revision, 좌표계, 상태 및 렌더링 계약은
[통합 하이라이트 워크플로 문서](docs/highlight_workflow.md)를 참고하세요.

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| API | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL, SQLAlchemy 2.x, Alembic |
| Authentication | Bearer Access Token, Refresh Token |
| AI orchestration | LangGraph, OpenAI API, Qwen |
| Video | FFmpeg, ffprobe, OpenCV |
| Action analysis | SoccerNet feature pipeline |
| Player candidate detector | fine-tuned RF-DETR Small 1.8.3 |
| Tracking | 외부 RF-DETR / Sports ReID CUDA runtime |
| Storage | 서버 로컬 파일 스토리지, DB artifact metadata |

## 프로젝트 구조

```text
KickClip-backend/
├── app/
│   ├── api/v1/              # FastAPI v1 router
│   ├── ai/                  # 분석 task와 model adapter
│   ├── core/                # 애플리케이션 설정
│   ├── db/                  # SQLAlchemy session과 model registry
│   ├── domains/             # 도메인별 model/schema/repository/service
│   ├── storage/             # 미디어 및 artifact 저장 로직
│   └── main.py              # FastAPI application entry point
├── alembic/                 # 데이터베이스 migration
├── configs/                 # AI 및 runtime 설정
├── docs/                    # 기능별 상세 문서
├── scripts/                 # 설치, seed, smoke test 도구
├── tests/                   # unit/integration tests
├── .env.example
├── alembic.ini
└── requirements.txt
```

## 시작하기

### 요구 사항

- Python 3.11 이상
- PostgreSQL 16 권장
- FFmpeg와 ffprobe
- 선택 사항: OpenAI API key
- 선수 후보 탐지: RF-DETR 1.8.3 호환 PyTorch 환경과 checkpoint
- 선택 사항: Target Tracking용 CUDA 환경과 외부 AI 프로젝트

### 1. 가상환경과 패키지 설치

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`에는 공통 백엔드 의존성과 `rfdetr==1.8.3`이 포함됩니다.
PyTorch는 팀원 또는 배포 환경에 맞는 CUDA/MPS/CPU build를 먼저 설치하세요.
모델 runtime이 없으면 API가 `requirements.txt` 설치 명령을 포함한 명시적
오류를 기록합니다. HOG로의 자동 전환은 기본적으로 비활성화되어 있습니다.

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 환경변수 설정

```powershell
Copy-Item .env.example .env
```

macOS/Linux에서는 다음 명령을 사용합니다.

```bash
cp .env.example .env
```

로컬 실행에 필요한 주요 설정은 다음과 같습니다.

```dotenv
ENV=local
KICKCLIP_DEBUG=true

DATABASE_URL="postgresql+psycopg://kickclip:kickclip@localhost:5432/kickclip"
AUTH_SECRET_KEY="replace-with-a-random-secret-key-at-least-32-characters"
STORAGE_ROOT="storage"

OPENAI_API_KEY=""
TRACKING_ENABLED=false
```

전체 설정과 설명은 [.env.example](.env.example)을 참고하세요. 실제 API key, 모델 경로, 영상 경로와 checkpoint는 Git에 커밋하지 마세요.

### 3. PostgreSQL 실행

Docker를 사용하는 경우:

```bash
docker run --name kickclip-postgres \
  -e POSTGRES_USER=kickclip \
  -e POSTGRES_PASSWORD=kickclip \
  -e POSTGRES_DB=kickclip \
  -p 5432:5432 \
  -d postgres:16
```

이미 컨테이너를 생성했다면 다음 명령으로 다시 시작할 수 있습니다.

```bash
docker start kickclip-postgres
```

### 4. 데이터베이스 migration

```bash
alembic upgrade head
```

현재 migration 상태와 단일 head 여부는 다음 명령으로 확인합니다.

```bash
alembic current
alembic heads
```

### 5. API 서버 실행

```bash
uvicorn app.main:app --reload
```

서버가 실행되면 다음 주소를 사용할 수 있습니다.

| 항목 | URL |
| --- | --- |
| 서비스 정보 | `http://127.0.0.1:8000/` |
| Health Check | `http://127.0.0.1:8000/api/v1/health` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |
| OpenAPI JSON | `http://127.0.0.1:8000/openapi.json` |

## 빠른 API 사용 흐름

보호된 API에는 로그인 응답의 Access Token을 전달해야 합니다.

```http
Authorization: Bearer <access_token>
```

### 1. 회원가입

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email":"editor@example.com","password":"kickclip1234","display_name":"KickClip Editor"}'
```

### 2. 경기 영상 업로드

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/matches" \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@/path/to/match.mp4;type=video/mp4" \
  -F "home_team=FC Barcelona" \
  -F "away_team=Real Betis" \
  -F "competition=La Liga"
```

응답의 `match_id`, `raw_video_asset_id`, `video_asset_id`는 분석과 편집 API에서 사용합니다.

### 3. 편집 프로젝트 생성

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/matches/<match_id>/projects" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"title":"선수 하이라이트","description":"숏폼 편집 프로젝트"}'
```

### 4. Action Spotting 작업 시작

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/action-spotting/matches/<match_id>/jobs" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"run_feature_extraction":true,"feature_extraction_mode":"auto","device":"auto","options":{}}'
```

작업 상태는 `GET /api/v1/analysis-jobs/{job_id}`, 결과 이벤트는 `GET /api/v1/action-spotting/jobs/{job_id}/events`에서 확인합니다.

### 5. Target Tracking 작업 시작

Tracking runtime이 설치된 환경에서만 사용할 수 있습니다.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/tracking/jobs" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "media_asset_id":"<raw_video_asset_id>",
    "project_id":"<project_id>",
    "initial_bbox_xyxy":[1290,727,1354,871],
    "bbox_format":"xyxy_pixels",
    "reacquisition_mode":"assisted"
  }'
```

API는 GPU 추론 완료를 기다리지 않고 `202 Accepted`와 `job_id`를 반환합니다.

### 6. Agent ClipPlan 생성

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agent/clip-plan" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id":"<project_id>",
    "match_id":"<match_id>",
    "prompt":"득점과 결정적인 슈팅 장면을 30초 숏츠로 만들어줘",
    "target_duration_sec":30
  }'
```

### 7. 렌더링과 다운로드

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/renders" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "clip_plan_id":"<clip_plan_id>",
    "options":{
      "ratio":"9:16",
      "quality":"1080p",
      "captions_enabled":true
    }
  }'
```

렌더링 상태는 `GET /api/v1/renders/{render_job_id}`, 완료 파일은 `GET /api/v1/renders/{render_job_id}/download`에서 확인합니다.

## 주요 API

모든 API의 최신 요청·응답 계약은 Swagger UI에서 확인할 수 있습니다.

| 영역 | 주요 Endpoint |
| --- | --- |
| Auth | `/api/v1/auth/*` |
| Matches | `/api/v1/matches/*` |
| Projects | `/api/v1/projects/*` |
| Media | `/api/v1/media/*` |
| Analysis Jobs | `/api/v1/matches/{match_id}/analysis-jobs` |
| Action Spotting | `/api/v1/action-spotting/*` |
| Target Tracking | `/api/v1/tracking/*` |
| Timeline / Players | `/api/v1/matches/{match_id}/timeline-events`, `/players` |
| Agent | `/api/v1/agent/*` |
| ClipPlan | `/api/v1/clip-plans/*` |
| Session | `/api/v1/session/*` |
| Render | `/api/v1/renders/*` |

## Tracking Runtime 설정

외부 frozen tracking pipeline을 사용하려면 `.env`에 다음 항목을 설정합니다.

```dotenv
TRACKING_ENABLED=true
TRACKING_PROJECT_ROOT="<AI_PROJECT_ROOT>"
TRACKING_E2E_SCRIPT_PATH="<AI_PROJECT_ROOT>/target_centric_tracking_e2e_v1/run_target_centric_pipeline.py"
TRACKING_VERIFY_SCRIPT_PATH="<AI_PROJECT_ROOT>/target_centric_tracking_e2e_v1/verify_e2e_installation.py"
TRACKING_PYTHON_EXECUTABLE="<YOLO_SOCCER_ENV>/python"
TRACKING_OUTPUT_ROOT="<AI_PROJECT_ROOT>/runs/target_centric_tracking_e2e_v1"
TRACKING_DEVICE="cuda"
TRACKING_REACQUISITION_MODE="assisted"
TRACKING_MAX_CONCURRENT_JOBS=1
```

설정 후 runtime 상태를 확인합니다.

```bash
curl "http://127.0.0.1:8000/api/v1/tracking/diagnostics?refresh=true" \
  -H "Authorization: Bearer <access_token>"
```

Verifier가 실패하면 tracking job 생성은 `503 Service Unavailable`로 거부되며 mock 결과를 반환하지 않습니다.

## 테스트

전체 테스트:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Target Tracking integration 테스트:

```bash
python -m unittest discover -s tests -p "test_target_tracking.py" -v
```

정적 import/문법 확인:

```bash
python -m compileall -q app scripts
```

CUDA와 실제 모델이 없는 환경에서도 tracking backend integration은 테스트 전용 fake runner로 검증할 수 있습니다. Production runtime은 fake runner로 대체하지 않습니다.

전체 업로드 → 분석 → 편집 → 렌더링 흐름은 다음 smoke test로 확인할 수 있습니다.

```bash
python scripts/smoke_test_studio_flow.py --video-path "/path/to/sample.mp4"
```

## 운영 시 주의사항

- 운영 환경에서는 충분히 긴 무작위 `AUTH_SECRET_KEY`를 사용해야 합니다.
- `ENV=prod`에서는 debug, mock data, developer mode를 활성화할 수 없습니다.
- 업로드 영상, 생성 artifact, 모델 파일, checkpoint와 `.env`는 Git에 커밋하지 않습니다.
- FFmpeg와 ffprobe가 실행 환경의 `PATH`에 있어야 합니다.
- 현재 media와 render artifact는 서버 로컬 스토리지에 저장됩니다.
- 장기 실행 AI 작업은 상태 조회 방식으로 사용하며 HTTP 요청에서 완료까지 기다리지 않습니다.
- Target Tracking은 기본 GPU 동시 실행 수를 1로 제한합니다.

## 문서

- [Mock 데이터 개발 가이드](docs/mock-data-development.md)
- [Target Tracking 백엔드 통합](docs/target_tracking_backend.md)
- API 문서: 서버 실행 후 `/docs`
