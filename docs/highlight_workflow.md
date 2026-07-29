# 통합 하이라이트 워크플로

## 데이터 흐름

```text
Project
  → cached AnalysisJob(HIGHLIGHT_SPOTTING)
  → TimelineEvent as HighlightScene
  → HighlightRevision(scene selection)
  → optional ScenePlayerCandidate + PlayerFocusSubject
  → SceneTrackingBinding + existing TrackingJob
  → tracking-aware ClipPlan
  → existing RenderJob
```

`AnalysisJob`은 `media_asset_id`, `video_sha256`, `model_version`,
`policy_version`으로 만든 `cache_key`를 저장합니다. 여러 Project가 같은 Match의
MediaAsset을 사용하면 Action Spotting job과 정규화된 `TimelineEvent`를 공유합니다.
자연어 요청, scene 선택, focus, tracking, ClipPlan, Render는 revision별 데이터입니다.

## HighlightScene 정규화

기존 `HighlightPostprocessor`의 threshold, pre/post-roll, temporal NMS, 최대 clip
길이를 재사용합니다. 같은 label의 근접 prediction은 NMS로 정리하되 suppression된
원본도 `metadata.source_predictions`에 남깁니다. 서로 다른 label은 pre/post-roll
구간이 실제로 겹치고 기존 최대 길이를 넘지 않을 때만 병합합니다. primary label은
설정의 class priority로 결정합니다.

전반/후반은 `Match.metadata.period_boundaries` 또는 prediction의 `half`를 사용합니다.
경계가 없으면 0초~2700초를 가정하지 않으며 `SELECT_SCENES` 상태로 사용자 확인을
기다립니다.

## Revision

- analyze: 새 revision을 만들고 cached Action Spotting 결과를 연결합니다.
- scene 선택 변경: 부모 revision을 보존한 새 revision을 만듭니다.
- player focus: 현재 selected scene set을 복사한 새 revision을 만듭니다.
- ClipPlan/Render ID는 해당 revision에 연결되므로 이전 결과를 덮어쓰지 않습니다.
- scene 상태는 `CANDIDATE`, `AGENT_SELECTED`, `USER_SELECTED`,
  `USER_EXCLUDED`, `INCLUDED_IN_CLIP_PLAN`으로 저장합니다.

## 후보 탐지와 tracking

후보 탐지는 selected scene의 대표 프레임만 샘플링합니다. 기본 detector는
fine-tuned `RFDETRSmall`이며 학습 프로젝트의 `rfdetr==1.8.3` native
`predict()` 계약을 그대로 사용합니다.

- checkpoint architecture: `RFDETRSmall`, resolution 512
- 입력: OpenCV BGR frame을 RGB로 변환한 batch
- confidence threshold: `0.25`
- candidate class: `player(0)`, `goalkeeper(1)`
- bbox: native post-processing이 원본 frame 좌표로 복원한 `xyxy`
- additional NMS: 없음
- checkpoint load: architecture/class/runtime 선검증 후 모든 state key, shape,
  tensor가 정확히 로드됐는지 strict audit

후보는 `선수 후보 N`으로만 표시합니다. OCR, 명단 연결, 실제 선수 이름 추론은
하지 않으며 사용자 선택이 identity의 유일한 근거입니다. HOG는
`PLAYER_DETECTOR_ALLOW_HOG_FALLBACK=true`를 명시한 개발 환경에서만 허용되고
기본값은 `false`입니다.

`PLAYER_DETECTOR_DEVICE=auto`는 CUDA, MPS, CPU 순서로 선택합니다. 명시적으로
`cuda` 또는 `mps`를 요청했는데 사용할 수 없으면 CPU로 변경하지 않고 실패합니다.
완료/실패 결과에는 아래 runtime 진단이 저장됩니다.

- requested/effective device, 선택 이유와 fallback 여부
- torch version, CUDA/MPS availability, GPU name
- detector component와 RF-DETR runtime version
- checkpoint SHA-256, confidence/class/batch/bbox/post-processing 계약

frozen runner는 첫 frame bbox 계약을 유지합니다. 후보 anchor가 scene 중간이면
anchor source timestamp부터 scene 끝까지 새 clip을 추출하고 local frame 0으로
재기준화합니다. binding은 아래 값을 보존합니다.

- source start/end timestamp와 advisory frame index
- source/clip FPS와 clip frame count
- scene clip MediaAsset
- selected candidate, focus subject, TrackingJob
- extraction command와 mapping provenance

timeline의 source timestamp는 `source_start_time_sec + time_seconds`로 계산합니다.
이는 VFR에서도 시간 좌표를 우선 보존합니다. source frame index는 FPS 기반
advisory 값으로 명시합니다.

## Agent와 fallback

RuleBasedClipPlanner의 “선수 이벤트가 없으면 선수 필터를 제거” fallback은
제거되었습니다. PLAYER revision의 ClipPlan은 user-confirmed binding에서
`ACTIVE`, `ACTIVE_LOW_CONFIDENCE`, `REACQUIRED`, `USER_CONFIRMED`인 crop 가능
구간만 사용합니다.

- 일부 scene ABSENT: 기본 `EXCLUDE`
- 사용자가 scene ABSENT + `FULL_FRAME`을 명시하고 plan 요청에서도
  `allow_absent_full_frame=true`: 전체 화면 유지
- 모든 scene에 crop 가능 target 없음: `NO_TARGET_SCENES`

다른 선수 bbox나 경기 전체 event로 자동 fallback하지 않습니다.

## Renderer

tracking timeline bbox는 ClipPlanItem의 `tracking_transform`으로 변환됩니다.
transform은 output aspect ratio, source boundary, target padding, 최소 crop 크기,
low-confidence 완화, center smoothing, 이동 속도 제한을 적용합니다.

- trusted bbox: target-centered dynamic crop
- 0.5초 이하 OCCLUDED: 마지막 stable center 유지
- LOST/SEARCHING/AMBIGUOUS/ABSENT: crop keyframe 생성 금지
- explicit FULL_FRAME: 전체 원본을 보존하고 letterbox/pillarbox padding

Render artifact provenance에는 revision, source scenes, tracking jobs, timeline schema,
focus subject, crop policy version이 기록됩니다.

## Background job과 동시성

- Action Spotting, candidate discovery, clip extraction, tracking, rendering은 API
  응답과 분리된 background 실행입니다.
- Action Spotting cache key와 DB unique index가 duplicate 실행을 막습니다.
- AnalysisJob/TrackingJob은 DB에서 QUEUED 상태를 원자적으로 claim합니다.
- Action Spotting, RF-DETR 후보 탐지, Tracking은 하나의 in-process GPU
  semaphore를 공유합니다.
- Tracking은 기존 PostgreSQL advisory lock과
  `TRACKING_MAX_CONCURRENT_JOBS`도 그대로 사용합니다.
- TrackingExecutor는 서버 시작 시 QUEUED/RUNNING job을 reconcile합니다.

## 로컬 실행

```powershell
alembic upgrade head
python -m pytest -q
uvicorn app.main:app --reload
```

RF-DETR 후보 탐지가 필요한 runtime에는 환경에 맞는 CUDA/MPS/CPU PyTorch를
먼저 설치한 다음 공통 백엔드 requirements를 설치합니다.

```powershell
python -m pip install -r requirements.txt
```

`rfdetr` 또는 checkpoint가 없고 HOG fallback도 허용하지 않았다면 workflow는
설치 명령이 포함된 명시적 `PLAYER_DETECTOR_UNAVAILABLE` 오류를 기록합니다.

실제 target tracking은 `docs/target_tracking_backend.md`의 환경변수와 외부 frozen
runtime 설치가 필요합니다. 외부 runtime이 없으면 일반 하이라이트 workflow와
fake-runner 테스트는 사용할 수 있지만 실제 PLAYER tracking은 unavailable 상태로
남습니다.
