# Tracking 로컬 환경 설정

각 개발자는 저장소를 clone한 뒤 자신의 로컬 절대경로를 `.env`에 설정해야
합니다. `.env` 자체는 Git에 올리지 않습니다.

## 1. RF-DETR 체크포인트 배치

별도로 전달받은 체크포인트를 다음 위치에 둡니다.

```text
<BACKEND_ROOT>/tracking_source/weights/rfdetr/checkpoint_best_regular.pth
```

파일 검증값:

```text
Size:   127470159 bytes
SHA256: 5d1d05cf78b6a777430430955d0e745ce5c61913c659b2e96ae5deddbd1a3a94
```

macOS/Linux:

```bash
shasum -a 256 tracking_source/weights/rfdetr/checkpoint_best_regular.pth
```

Windows PowerShell:

```powershell
Get-FileHash tracking_source/weights/rfdetr/checkpoint_best_regular.pth -Algorithm SHA256
```

## 2. Python 환경

Python 3.11 가상환경 하나를 tracking, scene discovery, scene target selection에
공통으로 사용하는 구성이 가장 단순합니다. 백엔드 의존성을 설치하고,
PyTorch와 torchvision은 각 개발자의 CPU/CUDA 플랫폼에 맞는 wheel을 별도로
설치합니다. macOS에서는 CUDA wheel을 설치하지 않으며
`TRACKING_DEVICE=cpu`를 사용합니다. 현재 런타임은 `mps` 값을 지원하지
않습니다.

Python 실행 파일은 반드시 실제 파일의 절대경로로 설정합니다.

```text
macOS/Linux: <BACKEND_ROOT>/venv311/bin/python
Windows:     C:/path/to/KickClip-backend/venv311/Scripts/python.exe
```

## 3. `.env` 설정

`.env.example`을 `.env`로 복사한 후 아래의 두 placeholder를 치환합니다.

```text
<ABS_BACKEND_ROOT>
<ABS_TRACKING_PYTHON_EXECUTABLE>
```

경로 설정은 모두 절대경로여야 합니다. Windows에서도 `C:/...` 형태의
forward slash를 사용하면 복사와 검토가 편합니다. 경로에 공백이 있으면 값을
큰따옴표로 감쌉니다.

핵심 설정 예시:

```dotenv
TRACKING_ENABLED=true
TRACKING_PROJECT_ROOT="<ABS_BACKEND_ROOT>/tracking_source"
TRACKING_E2E_SCRIPT_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_e2e_v1/run_target_centric_pipeline.py"
TRACKING_VERIFY_SCRIPT_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_e2e_v1/verify_e2e_installation.py"
TRACKING_PYTHON_EXECUTABLE="<ABS_TRACKING_PYTHON_EXECUTABLE>"
TRACKING_OUTPUT_ROOT="<ABS_BACKEND_ROOT>/storage/tracking_runtime"
TRACKING_DEVICE=cpu
TRACKING_REACQUISITION_MODE=assisted

SCENE_DISCOVERY_PROJECT_ROOT="<ABS_BACKEND_ROOT>/tracking_source"
SCENE_DISCOVERY_PYTHON_EXECUTABLE="<ABS_TRACKING_PYTHON_EXECUTABLE>"
SCENE_DISCOVERY_SCRIPT_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_scene_discovery_compat_r1/compat_runtime.py"
SCENE_DISCOVERY_VERIFY_SCRIPT_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_scene_discovery_compat_r1/verify_compat_runtime.py"
SCENE_DISCOVERY_MANIFEST_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_scene_discovery_compat_r1/scene_discovery_compat_r1_manifest.json"
SCENE_DISCOVERY_MANIFEST_SHA256="f8bc9710053198c59efae6233b1e1aef27c8fb2eecadaeef094061528c2d1005"

SCENE_TARGET_SELECTION_PROJECT_ROOT="<ABS_BACKEND_ROOT>/tracking_source"
SCENE_TARGET_SELECTION_PYTHON_EXECUTABLE="<ABS_TRACKING_PYTHON_EXECUTABLE>"
SCENE_TARGET_SELECTION_SCRIPT_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_scene_target_selection_v1/run_scene_target_selection.py"
SCENE_TARGET_SELECTION_VERIFY_SCRIPT_PATH="<ABS_BACKEND_ROOT>/app/domains/candidate_handoff_r1/runtime/verify_scene_target_selection_source.py"
SCENE_TARGET_SELECTION_MANIFEST_PATH="<ABS_BACKEND_ROOT>/tracking_source/target_centric_tracking_scene_target_selection_v1/scene_target_selection_frozen_manifest.json"
SCENE_TARGET_SELECTION_MANIFEST_SHA256="de16b76b89fa7585b70e121a693b538a1a7ddf859f92cd2cb175977e99c15a55"
```

CUDA가 정상 설치된 Windows/Linux 개발자는 `TRACKING_DEVICE=cuda` 또는
`auto`를 사용할 수 있습니다. 처음 설치를 확인할 때는 `cpu`가 문제 범위를
줄이기 쉽습니다.

## 4. 설치 검증

백엔드 루트에서 다음 명령을 실행합니다.

```bash
<ABS_TRACKING_PYTHON_EXECUTABLE> \
  tracking_source/target_centric_tracking_e2e_v1/verify_e2e_installation.py \
  --project-root "<ABS_BACKEND_ROOT>/tracking_source"
```

정상 결과에는 다음 항목이 포함됩니다.

```text
Status=PASS
Frozen models verified= 2
V6 ReID helper verified=True
E2E dependencies verified=True
```

백엔드가 `.env`를 실제로 읽은 결과까지 확인하려면 다음 명령을 실행합니다.

```bash
<ABS_TRACKING_PYTHON_EXECUTABLE> -c "from app.core.config import get_settings; from app.domains.tracking.verifier import TrackingInstallationVerifier, SceneTargetTrackingInstallationVerifier; s=get_settings(); g=TrackingInstallationVerifier(s); print(g.check(force=True)); print(SceneTargetTrackingInstallationVerifier(s, generic=g).check(force=True))"
```

최종 상태 코드는 각각 `TRACKING_AVAILABLE`과
`SCENE_TARGET_TRACKING_AVAILABLE`이어야 합니다.

## 5. 자주 발생하는 오류

- `*_CONFIGURATION_INVALID`: 상대경로가 들어갔는지 확인합니다.
- `*_RUNTIME_MISSING`: Python 실행 파일 또는 script/manifest 경로를 확인합니다.
- `*_MANIFEST_HASH_MISMATCH`: 다른 버전의 `tracking_source`와 `.env` 값을 섞지
  않았는지 확인합니다.
- 모델 검증 실패: RF-DETR 파일 위치, 크기, SHA256을 다시 확인합니다.
- CUDA 오류: 우선 `TRACKING_DEVICE=cpu`로 설치 상태를 분리해서 확인합니다.
