from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from .engine import EngineDecodeError, EngineUnavailableError, RealtimeAsrEngine, SessionState

engine = RealtimeAsrEngine()
logger = logging.getLogger("tapormi-worker")
_prewarm_task: asyncio.Task[None] | None = None
_prewarm_error: str | None = None

def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _prewarm_in_progress() -> bool:
    return _prewarm_task is not None and not _prewarm_task.done()


async def _run_prewarm() -> None:
    global _prewarm_error
    try:
        await asyncio.to_thread(engine.warmup)
        _prewarm_error = None
        logger.info("ASR engine prewarm completed (backend=%s)", engine.backend_name)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _prewarm_error = str(exc)
        logger.exception("ASR engine prewarm failed: %s", exc)
    except BaseException as exc:
        _prewarm_error = str(exc)
        logger.exception("ASR engine prewarm interrupted: %s", exc)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _prewarm_task
    should_prewarm = _env_bool(
        "TAPORMI_PREWARM_ON_START",
        default=(engine.backend_name != "mock"),
    )
    if should_prewarm:
        _prewarm_task = asyncio.create_task(_run_prewarm())
    try:
        yield
    finally:
        if _prewarm_task is not None:
            if _prewarm_task.done():
                try:
                    _prewarm_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                except BaseException:
                    pass
            else:
                _prewarm_task.cancel()
                try:
                    await _prewarm_task
                except asyncio.CancelledError:
                    pass
            _prewarm_task = None


app = FastAPI(title="tapormi-worker", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "backend": engine.backend_name,
        "model": engine.model_name,
        "warm": engine.warm,
        "prewarm_in_progress": _prewarm_in_progress(),
    }
    if engine.warning:
        payload["warning"] = engine.warning
    if _prewarm_error:
        payload["prewarm_error"] = _prewarm_error
    return payload


def _parse_message(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json: {exc}") from exc
    if not isinstance(payload, dict) or "type" not in payload:
        raise ValueError("invalid_message: missing type")
    return payload


@app.websocket("/stt")
async def stt_socket(ws: WebSocket) -> None:
    await ws.accept()
    ready_payload: dict[str, Any] = {
        "type": "ready",
        "model": engine.model_name,
        "backend": engine.backend_name,
        "warm": engine.warm,
        "prewarm_in_progress": _prewarm_in_progress(),
        "t_ms": int(time.time() * 1000),
    }
    if engine.warning:
        ready_payload["warning"] = engine.warning
    if _prewarm_error:
        ready_payload["prewarm_error"] = _prewarm_error
    await ws.send_json(ready_payload)

    sessions: dict[str, SessionState] = {}

    try:
        while True:
            raw = await ws.receive_text()
            msg = _parse_message(raw)
            msg_type = msg["type"]
            now = int(time.time() * 1000)

            if msg_type == "session_start":
                session_id = str(msg.get("session_id", ""))
                if not session_id:
                    await ws.send_json({"type": "error", "code": "missing_session_id", "t_ms": now})
                    continue

                raw_sample_rate = msg.get("sample_rate", 16000)
                try:
                    sample_rate = int(raw_sample_rate)
                except (TypeError, ValueError):
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "invalid_sample_rate",
                            "message": f"invalid sample_rate: {raw_sample_rate!r}",
                            "t_ms": now,
                        }
                    )
                    continue

                language = str(msg.get("lang", "ko") or "ko")
                try:
                    sessions[session_id] = engine.start(
                        session_id,
                        sample_rate=sample_rate,
                        language=language,
                    )
                except ValueError as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "invalid_session_start",
                            "message": str(exc),
                            "t_ms": now,
                        }
                    )
                except EngineUnavailableError as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "engine_unavailable",
                            "message": str(exc),
                            "t_ms": now,
                        }
                    )
                continue

            if msg_type == "audio_chunk":
                session_id = str(msg.get("session_id", ""))
                state = sessions.get(session_id)
                if state is None:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "unknown_session",
                            "t_ms": now,
                        }
                    )
                    continue
                chunk_b64 = msg.get("pcm16_base64", "")
                if not isinstance(chunk_b64, str) or not chunk_b64:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "missing_audio_chunk",
                            "t_ms": now,
                        }
                    )
                    continue
                try:
                    chunk = base64.b64decode(chunk_b64, validate=True)
                except Exception:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "invalid_audio_chunk",
                            "t_ms": now,
                        }
                    )
                    continue

                try:
                    partial = await asyncio.to_thread(engine.push_chunk, state, chunk)
                except EngineDecodeError as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "decode_error",
                            "message": str(exc),
                            "t_ms": now,
                        }
                    )
                    continue
                except EngineUnavailableError as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "engine_unavailable",
                            "message": str(exc),
                            "t_ms": now,
                        }
                    )
                    continue

                if partial:
                    await ws.send_json(
                        {
                            "type": "partial",
                            "session_id": session_id,
                            "text": partial,
                            "stability": 0.2,
                            "t_ms": now,
                        }
                    )
                continue

            if msg_type == "session_stop":
                session_id = str(msg.get("session_id", ""))
                state = sessions.pop(session_id, None)
                if state is None:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "unknown_session",
                            "t_ms": now,
                        }
                    )
                    continue

                try:
                    final_text = await asyncio.to_thread(engine.stop, state)
                except EngineDecodeError as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "decode_error",
                            "message": str(exc),
                            "t_ms": now,
                        }
                    )
                    continue
                except EngineUnavailableError as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "code": "engine_unavailable",
                            "message": str(exc),
                            "t_ms": now,
                        }
                    )
                    continue

                await ws.send_json(
                    {
                        "type": "final",
                        "session_id": session_id,
                        "text": final_text,
                        "t_ms": now,
                    }
                )
                continue

            if msg_type == "session_cancel":
                session_id = str(msg.get("session_id", ""))
                sessions.pop(session_id, None)
                continue

            await ws.send_json({"type": "error", "code": "unknown_type", "t_ms": now})

    except WebSocketDisconnect:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Tapormi realtime STT worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
