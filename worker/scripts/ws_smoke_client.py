from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8765/stt"
    session_id = str(uuid.uuid4())

    async with websockets.connect(uri) as ws:
        print("<-", await ws.recv())

        await ws.send(
            json.dumps(
                {
                    "type": "session_start",
                    "session_id": session_id,
                    "sample_rate": 16000,
                    "lang": "ko",
                }
            )
        )

        for i in range(12):
            fake_pcm = os.urandom(640)  # placeholder bytes
            payload = {
                "type": "audio_chunk",
                "session_id": session_id,
                "seq": i,
                "pcm16_base64": base64.b64encode(fake_pcm).decode("ascii"),
                "rms": 0.1,
            }
            await ws.send(json.dumps(payload))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                print("<-", msg)
            except asyncio.TimeoutError:
                pass

        await ws.send(json.dumps({"type": "session_stop", "session_id": session_id}))
        print("<-", await ws.recv())


if __name__ == "__main__":
    asyncio.run(main())
