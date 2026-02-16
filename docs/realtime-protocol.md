# Realtime STT Protocol (v0)

## WebSocket
- URL: `ws://127.0.0.1:8765/stt`

## Client -> Worker

### session_start
```json
{"type":"session_start","session_id":"uuid","sample_rate":16000,"lang":"ko"}
```

### audio_chunk
```json
{"type":"audio_chunk","session_id":"uuid","seq":1,"pcm16_base64":"...","rms":0.04}
```

### session_stop
```json
{"type":"session_stop","session_id":"uuid"}
```

### session_cancel
```json
{"type":"session_cancel","session_id":"uuid"}
```

## Worker -> Client

### ready
```json
{"type":"ready","backend":"mlx_qwen3","model":"mlx-community/Qwen3-ASR-1.7B-8bit","warm":false,"t_ms":1739730000000}
```

`warning` 필드는 선택입니다. (`auto` 백엔드가 `mock`으로 폴백한 경우 포함)

### partial
```json
{"type":"partial","session_id":"uuid","text":"안녕","stability":0.72,"t_ms":1739730000100}
```

### final
```json
{"type":"final","session_id":"uuid","text":"안녕하세요","t_ms":1739730000900}
```

### error
```json
{"type":"error","session_id":"uuid","code":"unknown_session","message":"...","t_ms":1739730000050}
```

주요 `code`:
- `missing_session_id`
- `invalid_sample_rate`
- `invalid_session_start`
- `missing_audio_chunk`
- `invalid_audio_chunk`
- `decode_error`
- `engine_unavailable`
- `unknown_session`
- `unknown_type`
