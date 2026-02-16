# Tapormi Implementation Plan

## 1. Product
- 트리거: `fn` long-press
- UI: 작은 캡슐 팝업 (Listening / Transcribing)
- 결과: 실시간 partial + final 입력창 삽입

## 2. 시스템 구성
1. macOS Swift 메뉴바 앱
2. Python 상주 STT 워커(WebSocket)
3. 로컬 모델(Qwen3 ASR MLX)

## 3. 단계
1. Worker 프로토콜/세션 관리 완성
2. Swift 앱 생성 및 `fn` 감지 구현
3. 오디오 스트리밍 + partial UI 연결
4. 텍스트 삽입(Clipboard + Cmd+V, 이후 AX direct)
5. 지연 튜닝/VAD/오류 복구

## 4. 성능 목표
- First partial < 500ms
- Partial 업데이트 100~200ms
- Stop 후 final < 700ms
