# tapormi-worker

Tapormi macOS 앱과 통신하는 로컬 WebSocket STT 워커입니다.

## 실행
```bash
uv sync --python 3.14
uv run python -m tapormi_worker.main --host 127.0.0.1 --port 8765
```

기본값은 `TAPORMI_ASR_BACKEND=auto`이며, `mlx-audio`가 없으면 자동으로 `mock` 백엔드로 폴백합니다.

## Qwen3(mlx-audio) 활성화
```bash
TAPORMI_ASR_BACKEND=mlx_qwen3 \
TAPORMI_ASR_MODEL=mlx-community/Qwen3-ASR-1.7B-8bit \
uv run python -m tapormi_worker.main --host 127.0.0.1 --port 8765
```

선택 환경변수:
- `TAPORMI_PARTIAL_EVERY_CHUNKS` (기본 4)
- `TAPORMI_MIN_PARTIAL_SECONDS` (기본 0.4)

## 엔드포인트
- WebSocket: `ws://127.0.0.1:8765/stt`
- Health: `http://127.0.0.1:8765/health`

## 참고
- 런타임 요구사항: Python `>=3.10`
- 입력 오디오는 `16kHz mono PCM16` 기준입니다.
- `ready` 이벤트에 `backend`, `model`, `warm`, `warning(optional)`이 포함됩니다.
- Qwen3 모델은 첫 디코드 시 약 2.5GB 가중치 다운로드가 발생할 수 있습니다.
