# Tapormi

Fn 키를 길게 눌러 실시간 받아쓰기를 하는 macOS 앱입니다.

## MVP 목표
- `fn` 300ms 홀드 시 캡슐형 팝업 표시
- 홀드 중 실시간 partial STT 표시
- 키 릴리즈 시 final 텍스트를 포커스된 입력창에 삽입
- 로컬 ASR 모델: `mlx-community/Qwen3-ASR-1.7B-8bit`

## 현재 상태
- [x] 프로젝트 부트스트랩
- [x] Python WebSocket 워커 스켈레톤
- [x] Qwen3 실시간 디코더 연결 (`mlx-audio` 백엔드)
- [ ] Swift 메뉴바 앱 구현
- [ ] fn long-press + 팝업 + 텍스트 삽입

## 구조
- `docs/` 설계 문서
- `worker/` Python 실시간 STT 워커
- `tapormi-macos/` Swift 앱(생성 예정)

## 빠른 시작 (워커)
```bash
cd worker
uv sync --python 3.10
uv run python -m tapormi_worker.main --host 127.0.0.1 --port 8765
```

Qwen3 백엔드 강제 실행:
```bash
cd worker
TAPORMI_ASR_BACKEND=mlx_qwen3 uv run python -m tapormi_worker.main --host 127.0.0.1 --port 8765
```

참고:
- 기본적으로 워커 시작 시 Qwen3 prewarm을 백그라운드로 시도합니다.
- 비활성화: `TAPORMI_PREWARM_ON_START=0`

## 프로토콜 요약
- Client -> Worker
  - `session_start`
  - `audio_chunk`
  - `session_stop`
  - `session_cancel`
- Worker -> Client
  - `ready`
  - `partial`
  - `final`
  - `error`

세부 스펙은 `docs/realtime-protocol.md` 참고.
