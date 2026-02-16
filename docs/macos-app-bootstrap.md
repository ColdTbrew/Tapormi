# macOS App Bootstrap Checklist

## Xcode 생성
1. Xcode -> New Project -> macOS App
2. Product Name: `Tapormi`
3. Interface: SwiftUI
4. Language: Swift
5. Create Git repository 옵션은 끄기(이미 상위 git 사용)

## 1차 구현 파일
- `AppDelegate.swift`: 상태바/라이프사이클
- `FnHoldDetector.swift`: `CGEventTap` + `flagsChanged`
- `PopupPanel.swift`: non-activating 캡슐 패널
- `AudioStreamRecorder.swift`: 16k mono PCM chunk 생성
- `STTWebSocketClient.swift`: worker 프로토콜 연결
- `TextInjector.swift`: pasteboard + Cmd+V

## 권한
- Microphone
- Accessibility
- Input Monitoring(필요 시)

## 기본 동작 검증
- fn 홀드 300ms -> 팝업 표시
- 홀드 중 오디오 레벨 애니메이션
- 릴리즈 시 session_stop + final 수신
- final 텍스트 입력창 삽입
