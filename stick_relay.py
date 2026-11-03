"""
M5StickS3 <-> Gemini Live 중계 서버 (stick_relay.py)

스틱이 보내는 16kHz PCM16 mono 음성을 Gemini Live로 넘기고,
Gemini가 돌려주는 24kHz PCM16 음성을 스틱으로 전달합니다.

설치:   pip install aiohttp google-genai
환경변수:
  GEMINI_API_KEY  (필수)
  DEVICE_TOKEN    (필수, 스틱 스케치의 TOKEN과 동일하게)
  LIVE_MODEL      (기본값은 아래. token_server.py에서 쓰는 모델명과 맞추세요)
  VOICE_NAME      (기본 Aoede)
  PERSONA_FILE    (선택, 태연 페르소나 텍스트 파일 경로)
  PORT            (기본 8765)
실행:   python stick_relay.py
"""
import asyncio
import os

from aiohttp import web, WSMsgType
from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
DEVICE_TOKEN = os.environ["DEVICE_TOKEN"]
MODEL = os.environ.get("LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025")
VOICE = os.environ.get("VOICE_NAME", "Aoede")
PORT = int(os.environ.get("PORT", "8765"))

CHUNK_BYTES = 4800  # 24kHz * 2byte * 0.1초 (스틱 쪽 CHUNK_BYTES와 동일해야 함)

DEFAULT_PERSONA = (
    "너는 가수 태연처럼 다정하고 친근하게 말하는 음성 비서야. "
    "음성 대화이므로 대답은 한두 문장으로 짧고 자연스럽게 해."
)


def load_persona() -> str:
    path = os.environ.get("PERSONA_FILE")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return DEFAULT_PERSONA


client = genai.Client(api_key=API_KEY)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)

    if request.query.get("token") != DEVICE_TOKEN:
        await ws.close(code=4401, message=b"unauthorized")
        return ws

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=load_persona(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
    )

    print("device connected")
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:

            async def gemini_to_device():
                buf = bytearray()
                while True:
                    async for resp in session.receive():
                        if resp.data:
                            buf.extend(resp.data)
                            while len(buf) >= CHUNK_BYTES:
                                await ws.send_bytes(bytes(buf[:CHUNK_BYTES]))
                                del buf[:CHUNK_BYTES]
                        sc = resp.server_content
                        if sc and sc.turn_complete:
                            if buf:
                                if len(buf) % 2:
                                    buf = buf[:-1]
                                if buf:
                                    await ws.send_bytes(bytes(buf))
                                buf.clear()
                            await ws.send_str("turn_end")

            async def device_to_gemini():
                async for msg in ws:
                    if msg.type == WSMsgType.BINARY:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=msg.data, mime_type="audio/pcm;rate=16000"
                            )
                        )
                    elif msg.type == WSMsgType.TEXT and msg.data == "end":
                        await session.send_realtime_input(audio_stream_end=True)

            t1 = asyncio.create_task(gemini_to_device())
            t2 = asyncio.create_task(device_to_gemini())
            done, pending = await asyncio.wait(
                {t1, t2}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            for t in done:
                if t.exception():
                    print("error:", t.exception())
    except Exception as e:
        print("session error:", e)
    finally:
        print("device disconnected")
        if not ws.closed:
            await ws.close()
    return ws


app = web.Application()
app.router.add_get("/ws", ws_handler)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT)
