# Tapormi macOS App

이 디렉토리는 SwiftUI macOS 앱 소스 위치입니다.

## 생성 방법
1. Xcode에서 macOS App 프로젝트 생성
2. Product Name: `Tapormi`
3. Interface: SwiftUI
4. Language: Swift
5. 프로젝트 위치를 이 폴더로 지정

## 연결 대상
- Worker WebSocket: `ws://127.0.0.1:8765/stt`
- Protocol: `docs/realtime-protocol.md`

## 첫 구현 우선순위
1. `fn` long-press 감지 (`CGEventTap`)
2. 캡슐형 팝업 UI
3. 오디오 chunk 스트리밍
4. partial/final 표시
5. 입력창 텍스트 삽입
