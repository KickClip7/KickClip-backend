# KickClip 공유 목업 데이터 개발

## 목적

목업 JSON 파일을 단순히 `storage`에 놓는 것만으로는 API가 동작하지 않습니다. API는 먼저 DB의 `Project`와 `Match`를 조회하고 접근 권한을 검사합니다.

이 패치는 다음 구조로 목업과 실제 데이터를 분리합니다.

- 목업 fixture: `storage/matches/korjpn_2026/timeline_events.json`
- 로컬 DB: deterministic mock Project, Match, TimelineEvent
- 인증: 로그인은 그대로 필수
- 공유 접근: 로컬·개발·테스트 환경에서 명시적으로 시드된 mock Match만 모든 로그인 팀원에게 허용
- 운영: mock 관련 설정이 켜져 있으면 애플리케이션 시작 단계에서 차단

## 팀 저장소에 공유할 파일

```text
storage/matches/korjpn_2026/timeline_events.json
```

원본 action spotting `events.json`은 팀원마다 가질 필요가 없습니다.

## 최초 fixture 생성 담당자

```cmd
python scripts/build_mock_timeline_events.py --input "C:\Users\sunca\Documents\카카오톡 받은 파일\events.json" --match-id korjpn_2026
```

생성된 `storage\matches\korjpn_2026\timeline_events.json`을 저장소에 커밋합니다.

JSON 생성과 DB 시드를 한 번에 수행하려면 다음을 사용합니다.

```cmd
python scripts/build_mock_timeline_events.py --input "C:\Users\sunca\Documents\카카오톡 받은 파일\events.json" --match-id korjpn_2026 --seed-db
```

## 모든 팀원의 로컬 설정

기존 `.env`에 다음을 추가합니다.

```env
ENV=local
USE_MOCK_DATA=true
MOCK_SHARED_ACCESS_ENABLED=true
AGENT_DEV_MATCH_ID=korjpn_2026
```

DB 마이그레이션과 회원가입을 마친 뒤 다음 명령을 실행합니다.

```cmd
python scripts/seed_mock_timeline_events.py --match-id korjpn_2026
```

프론트의 활성 Match ID도 `korjpn_2026`으로 통일합니다.

## 확인

Bearer 토큰을 사용해 확인합니다.

```cmd
curl -i http://127.0.0.1:8000/api/v1/matches/korjpn_2026/timeline-events -H "Authorization: Bearer ACCESS_TOKEN"
```

```cmd
curl -i http://127.0.0.1:8000/api/v1/studio/matches/korjpn_2026/edit-state -H "Authorization: Bearer ACCESS_TOKEN"
```

```cmd
curl -i -X POST http://127.0.0.1:8000/api/v1/session/start -H "Authorization: Bearer ACCESS_TOKEN" -H "Content-Type: application/json" -d "{\"match_id\":\"korjpn_2026\"}"
```

`edit-state.video.url`은 MediaAsset을 별도로 시드하지 않았으므로 비어 있을 수 있습니다. Timeline event 목록과 편집 세션은 정상 조회됩니다. 영상까지 백엔드 MediaAsset으로 제공해야 할 때는 실제 업로드 API를 사용해 Match를 만들거나 별도의 mock video asset 시더를 추가합니다.

## 실제 데이터 운영으로 복귀

`.env`를 다음처럼 변경하고 서버를 재시작합니다.

```env
USE_MOCK_DATA=false
MOCK_SHARED_ACCESS_ENABLED=false
AGENT_DEV_MATCH_ID=
```

실제 Match 접근 권한, 업로드, 분석 job, DB timeline 조회 경로는 변경하지 않았습니다. 목업 공유 접근은 `kickclip_mock_match` metadata가 있는 시드 Match에만 적용됩니다.
