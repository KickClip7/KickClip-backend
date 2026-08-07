# KickClip Target-Centric Tracking E2E V1

사용자가 영상의 한 프레임에서 선수를 한 명 선택하면 다음 흐름을 하나의 canonical 실행 스크립트로 관리합니다.

```text
선수 선택
→ 카메라 컷 탐지 및 shot 분리
→ 첫 shot에서 frozen Phase-1 추적
→ pre-cut target memory 생성
→ 이후 shot의 선수 tracklet 생성
→ frozen Sports OSNet 순위화
→ frozen Stage 3-B2 안전 gate
→ 불확실하면 후보 2~3명 사용자 확인
→ 확인된 anchor부터 frozen Phase-1 추적 재개
→ 전체 영상 target timeline 및 미리보기 생성
```

기존 폴더는 수정하지 않습니다.

```text
target_centric_tracking_v1  : 읽기 전용
target_centric_tracking_v2  : 읽기 전용
target_centric_tracking_e2e_v1 : canonical orchestration 코드
```

## 설치

압축 파일에서 `target_centric_tracking_e2e_v1` 폴더를 프로젝트 루트에 배치합니다.

```text
D:\HAESUNG\prometheus\YOLO-train\target_centric_tracking_e2e_v1
```

필수 기존 파일:

```text
target_centric_tracking_v1\phase1_frozen_manifest.json
target_centric_tracking_v2\stage3*.py
global_ID_tracking_upgrade_v6\stage2b1_extract_frozen_tracking_reid_embeddings_v6.py
weights\rfdetr\checkpoint_best_regular.pth
global_ID_tracking_upgrade_v6\third_party\Deep-EIoU\Deep-EIoU\checkpoints\sports_model.pth.tar-60
```

`global_ID_tracking_upgrade_v7`은 필수 파일이 아닙니다. 폴더가 없어도 canonical E2E가 동작해야 하며, V7은 제품 tracking runtime에 사용하지 않습니다.

## 중간 프레임에서 선수를 선택한 경우

백엔드의 scene target selection처럼 bbox가 원본 영상의 중간 프레임에서 선택된 경우 `--initial-frame`을 함께 전달합니다.

```bat
python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --video "..." --test-name example --initial-frame 1234 --initial-bbox X1 Y1 X2 Y2 --device cuda --reacquisition-mode assisted
```

내부적으로는 확인된 anchor부터 forward tracking을 시작하며, 최종 `target_timeline.json`의 `frame_index`와 `time_ms`는 원본 영상 좌표계로 다시 기록됩니다.

## 현재 Betis 영상 첫 실행

```bat
cd /d D:\HAESUNG\prometheus\YOLO-train && python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --video "sample_videos\betis_vs_barcelona_2025_26_01h06m53s_23s.mp4" --test-name betis_barcelona_target_e2e --initial-bbox 1290 727 1354 871 --device cuda --reacquisition-mode assisted
```

정상적인 첫 중단은 오류가 아닙니다.

```text
Status         : NEEDS_CONFIRMATION
Pending action : MEMORY_REVIEW
```

콘솔에 출력되는 memory contact sheet와 tracking preview를 확인합니다. 모두 선택한 target이면 다음 명령을 실행합니다.

```bat
cd /d D:\HAESUNG\prometheus\YOLO-train && python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --test-name betis_barcelona_target_e2e --resume --approve-review MEMORY --device cuda
```

## Cross-shot 후보 확인

후보 확인이 필요하면 다음과 같이 종료됩니다.

```text
Status         : NEEDS_CONFIRMATION
Pending action : CROSS_SHOT_CONFIRMATION
Ambiguity      : ambiguity_0001
Candidates     : shot_0001_track_0001, ...
Contact sheet  : ...\ambiguity_candidates\ambiguity_0001\candidates.jpg
```

해당 shot에 target이 없으면:

```bat
cd /d D:\HAESUNG\prometheus\YOLO-train && python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --test-name betis_barcelona_target_e2e --resume --ambiguity-id ambiguity_0001 --confirm-absent --device cuda
```

후보 중 target을 찾았으면 콘솔에 표시된 후보 ID를 사용합니다.

```bat
cd /d D:\HAESUNG\prometheus\YOLO-train && python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --test-name betis_barcelona_target_e2e --resume --ambiguity-id ambiguity_0002 --confirmed-candidate shot_0004_track_0003 --device cuda
```

확인된 후보의 anchor부터 같은 shot 내부 frozen Phase-1 추적이 실행됩니다.

## Segment 시각 검수

확인된 후보 anchor부터 frozen Phase-1을 실행하며, legacy V7 경고와 연구용 10~30초 duration 경고는 원본 Stage-0을 보존한 derived compatibility audit에서만 정규화합니다. 모델/hash/bbox/video 오류는 그대로 차단됩니다.

```bat
cd /d D:\HAESUNG\prometheus\YOLO-train && python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --test-name betis_barcelona_target_e2e --resume --approve-review SEGMENT --device cuda
```

10~30초 segment에서 frozen Phase-1 내부 검수가 필요하면 콘솔에 표시된 단계만 승인합니다.

```bat
cd /d D:\HAESUNG\prometheus\YOLO-train && python target_centric_tracking_e2e_v1\run_target_centric_pipeline.py --test-name betis_barcelona_target_e2e --resume --approve-review STAGE2B --device cuda
```

가능한 단계는 `STAGE2B`, `STAGE2D`, `STAGE2D1`, `STAGE2D2`입니다. 영상을 직접 확인하지 않고 승인하면 안 됩니다.

## 출력

```text
runs\target_centric_tracking_e2e_v1\<test_name>\
├─ pipeline_manifest.json
├─ pipeline_state.json
├─ shot_boundaries.csv
├─ target_timeline.json
├─ target_timeline.csv
├─ target_segments.json
├─ ambiguities.json
├─ confirmations.json
├─ ambiguity_candidates\
├─ full_frame_tracking_preview.mp4
├─ target_centered_preview.mp4
├─ pipeline_summary.json
├─ report.md
└─ work\
```

## 상태 의미

```text
ACTIVE / REACQUIRED / USER_CONFIRMED
- 선택 target bbox를 신뢰할 수 있음

SEARCHING
- target을 찾지 못했으며 다른 선수에게 자동 전환하지 않음

AMBIGUOUS
- 유력 후보가 있지만 사용자 확인 전에는 bbox를 확정하지 않음

ABSENT
- 사용자가 해당 shot에 target이 없다고 확인함
```

## 안전 원칙

- `assisted` 모드에서는 Stage 3-B2 자동 gate를 통과해도 사용자 확인 없이 연결하지 않습니다.
- 불확실한 프레임에는 target bbox를 만들지 않습니다.
- 카메라 컷을 가로질러 motion track을 이어 붙이지 않습니다.
- 기존 V1/V2 파일과 threshold를 수정하지 않습니다.
- `global_ID_tracking_upgrade_v7`을 필요로 하거나 호출하지 않습니다.
- target memory는 최초 pre-cut tracking에서 한 번 만들며, 검수되지 않은 episode로 자동 갱신하지 않습니다.
- `auto-safe`는 선택 옵션이지만 MVP 검증 단계에서는 `assisted` 사용을 권장합니다.

## 현재 구현 범위

이 버전은 서비스형 assisted MVP입니다. 여러 컷을 순차 처리하며, target이 없는 클로즈업 shot은 `ABSENT` 또는 `SEARCHING`으로 건너뛴 뒤 다음 shot에서 다시 후보를 찾습니다.

최종 자동 cross-shot 일반화 성능을 증명하는 평가 도구는 아닙니다. 잘못된 선수를 조용히 연결하지 않고 최소한의 사용자 확인으로 timeline을 완성하는 것이 목적입니다.

## KickClip-backend canonical 연결

백엔드에 포함할 때 `TRACKING_PROJECT_ROOT`는 **backend 루트가 아니라 `tracking_source`** 를 가리켜야 합니다.

```text
TRACKING_PROJECT_ROOT=D:\HAESUNG\prometheus\KickClip-backend\tracking_source
TRACKING_E2E_SCRIPT_PATH=D:\HAESUNG\prometheus\KickClip-backend\tracking_source\target_centric_tracking_e2e_v1\run_target_centric_pipeline.py
TRACKING_VERIFY_SCRIPT_PATH=D:\HAESUNG\prometheus\KickClip-backend\tracking_source\target_centric_tracking_e2e_v1\verify_e2e_installation.py
TRACKING_OUTPUT_ROOT=D:\HAESUNG\prometheus\KickClip-backend\storage\tracking_runtime
TRACKING_REACQUISITION_MODE=assisted
```

`TRACKING_OUTPUT_ROOT`는 candidate-handoff DB/API가 immutable review artifact를 다시 읽을 수 있도록 반드시 backend `STORAGE_ROOT` 아래에 둡니다.

Scene target selection이 원본 영상 중간 프레임을 선택하면 backend process runner가 immutable anchor와 reviewed shot boundaries를 검증한 후 `--initial-frame`, `--initial-bbox`, `--cut-frames`만 canonical E2E에 전달합니다. R1/R2/R3 tracking algorithm이나 V7로 routing하지 않습니다.

Cross-shot ambiguity candidate에는 backend review UI를 위한 manifest, full-frame context, shot clip, reference gallery, immutable crop hashes가 생성됩니다. 이 evidence bundle은 **후보 점수 계산 이후**에만 생성되며 Stage 3-B1 score나 Stage 3-B2 safety gate에는 입력되지 않습니다. 사용자가 `SAME_PLAYER`로 확인한 경우 backend memory revision은 audit/provenance로 저장할 수 있지만 이후 canonical Stage 3-B1 scoring memory를 대체하지 않습니다.
