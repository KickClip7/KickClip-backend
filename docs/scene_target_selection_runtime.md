# Scene Target Selection / R3 runtime contract

현재 Alembic 단일 head는 `20260730_0016`이며 `20260728_0012`부터
`20260730_0016`까지의 migration이 모두 필요하다.

## 설치 경계

generic same-shot tracking 검증과 selection-assisted R3 검증은 별개다.
R3 작업은 아래 항목이 모두 검증되기 전에는 생성되지 않는다.

- Scene Target Selection package verifier
- Scene Target Selection manifest SHA-256
- R2 manifest SHA-256
- R3 manifest SHA-256
- R3 wrapper import
- Sports OSNet strict loader
- selection/reference schema compatibility
- synthetic assisted smoke

설정은 `.env.example`의 `SCENE_TARGET_SELECTION_*`,
`TRACKING_SCENE_SELECTION_*`, `TRACKING_R2_*`, `TRACKING_R3_*`를
사용한다. manifest SHA 값은 파일을 읽어 계산한 실제 SHA-256과 일치해야
한다. metadata에 문자열만 기록하는 것은 검증으로 인정하지 않는다.

candidate discovery, target reference build, earlier retrieval/decision,
reviewability render, manual anchor, event ranking, R3 preparation은
`scene_ai_tasks` durable queue에서 실행한다. POST는 `202 Accepted`와
`task_id`를 반환하며 `GET /api/v1/scene-ai-tasks/{task_id}`로 조회한다.
동일 payload 재요청은 idempotency key로 기존 task를 재사용하고, 실패
task는 retry endpoint와 최대 시도 횟수를 사용한다. backend 재시작 시
RUNNING task는 QUEUED로 복구된다.

generic verifier:

```bash
<TRACKING_PYTHON_EXECUTABLE> <TRACKING_VERIFY_SCRIPT_PATH> \
  --project-root <TRACKING_PROJECT_ROOT>
```

selection package verifier:

```bash
<SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE> \
  <SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH> \
  --project-root <SCENE_TARGET_SELECTION_PROJECT_ROOT>
```

R3 smoke verifier:

```bash
<TRACKING_PYTHON_EXECUTABLE> \
  <TRACKING_SCENE_SELECTION_VERIFY_SCRIPT_PATH> \
  --project-root <TRACKING_PROJECT_ROOT> \
  --r3-script <TRACKING_SCENE_SELECTION_R3_SCRIPT_PATH> \
  --synthetic-assisted-smoke
```

R3 smoke verifier stdout은 다음 boolean claim을 포함한 JSON이어야 한다.

```json
{
  "r3_wrapper_verified": true,
  "sports_osnet_strict_loader_verified": true,
  "selection_schema_compatible": true,
  "reference_schema_compatible": true,
  "synthetic_assisted_smoke_verified": true
}
```

## Immutable selection artifact

각 selection은 discovery 공용 root가 아니라 다음 독립 root를 사용한다.

```text
discoveries/<discovery-id>/
├─ candidates/
└─ selections/
   ├─ r0001_<selection-id>/
   └─ r0002_<selection-id>/
```

DB에는 selection root, selection/reference/proposal/decision JSON의 경로와
SHA-256을 저장한다. R3 dispatch 직전에도 root containment와 hash를 다시
검증한다. runtime 상대 경로는 실파일, root containment, suffix/MIME
allowlist를 통과해야 하며 API는 runtime path 대신 인증된 artifact ID를
반환한다.

## Event-aware ranking

`target_centric_tracking_event_candidate_ranking_v1`은 현재
`PROVISIONAL_SHADOW_ONLY`다. 입력 시간은 원본 영상 기준
`SOURCE_VIDEO_SECONDS`, candidate observation 시간은 scene-local seconds로
고정한다. API는 raw feature, event relevance, trackability,
recommendation score, reason/risk code와 Top 3~5를 생성하지만 자동 target
확정은 항상 `false`다.

human role label은 production feature와 분리 저장하며 evaluation endpoint가
Recall@1/3/5, MRR, broadcast non-actor Top-1을 계산한다. human label이
없으면 측정값은 `NOT_RUN`이다.
