"""Vexo speaker service (FastAPI): streaming diarization + wearer matching for the band's audio.

Start (from the repository):
    .venv-speaker/bin/uvicorn speaker_service.app:app --host 127.0.0.1 --port 8790

  GET    /health                  the process is up
  GET    /ready                   models loaded? (or exactly what is missing, e.g. HF_TOKEN / gated model access)
  GET    /enrollment              the confirmed wearer profile (summary)
  POST   /enrollment/clips        one clean clip from the band (WAV body, or raw 16 kHz s16le) -> checked
  POST   /enrollment/commit       build and save the profile from the accepted clips (3 or more)
  DELETE /enrollment              forget the profile and any pending clips
  GET    /sessions/{id}           timing and speaker summary of a conversation
  WS     /sessions/{id}/audio     stream audio, receive speaker events (protocol below)

WebSocket protocol (all JSON text except audio):
  -> {"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le", "channels": 1, "start": 0.0, "resume": false}
  <- {"type": "ready", ...}            or {"type": "unavailable", "reason": ...} (then the client keeps transcribing)
  -> binary frames: 16-bit little-endian mono PCM; {"type": "at", "start": t} before a frame that does not follow on
  <- {"type": "speaker" | "revision" | "status" | "gap" | "generation", "session_id", "generation", "revision", ...}
  -> {"type": "end"}                   <- {"type": "summary", ...}
"""
import asyncio
import json
import os
import threading

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from speaker_service import enrollment
from speaker_service.config import Settings
from speaker_service.engine import Models, Session, Worker

settings = Settings()
models = Models(settings)
if os.environ.get("SPEAKER_TEST_STANDIN_MODELS") == "1":       # tests only: tones stand in for voices, no accuracy
    from speaker_service.tests.fakes import FakeEmbedding, FakeSegmentation
    models = Models(settings, segmentation_loader=lambda: FakeSegmentation(), embedding_loader=lambda: FakeEmbedding())
    models.specs = {"duration": 5.0}
sessions = {}                                   # id -> (Session, Worker)
pending_clips = []                              # accepted enrolment clips not yet committed
app = FastAPI(title="Vexo speaker service")


@app.on_event("startup")
def _startup():
    if models.status == "not loaded":                 # never reload models that are already up
        threading.Thread(target=models.load, name="load-models", daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    profile = enrollment.load(settings.profile, settings)
    body = {"ready": models.ready(), "models": models.summary(), "settings": settings.public(),
            "wearer_profile": bool(profile), "sessions": len(sessions)}
    return JSONResponse(body, status_code=200 if models.ready() else 503)


def _profile_summary(profile):
    if not profile:
        return {"enrolled": False}
    return {"enrolled": True, **{k: v for k, v in profile.items() if k != "embedding"}}


@app.get("/enrollment")
def get_enrollment():
    return dict(_profile_summary(enrollment.load(settings.profile, settings)), pending_clips=len(pending_clips))


@app.post("/enrollment/clips")
async def add_clip(request: Request):
    if not models.ready():
        return JSONResponse({"ok": False, "problem": "models not ready: %s" % (models.error or models.status)}, 503)
    try:
        audio = enrollment.decode(await request.body(), request.headers.get("content-type"))
    except Exception as problem:
        return JSONResponse({"ok": False, "problem": "could not read the audio: %s" % problem}, 400)
    result = await asyncio.to_thread(enrollment.analyse_clip, audio, models)
    if result["ok"]:
        pending_clips.append(result)
    return {k: v for k, v in result.items() if k != "embedding"} | {"accepted_clips": len(pending_clips)}


@app.post("/enrollment/commit")
def commit_enrollment():
    if len(pending_clips) < 3:
        return JSONResponse({"ok": False, "problem": "need 3 accepted clips, have %d" % len(pending_clips)}, 400)
    profile = enrollment.build_profile(pending_clips, models, settings)
    enrollment.save(profile, settings.profile)
    pending_clips.clear()
    return {"ok": True, **_profile_summary(profile)}


@app.delete("/enrollment")
def delete_enrollment():
    pending_clips.clear()
    existed = os.path.exists(settings.profile)
    if existed:
        os.remove(settings.profile)
    return {"deleted": existed}


@app.get("/sessions/{session_id}")
def session_summary(session_id: str):
    if session_id not in sessions:
        return JSONResponse({"error": "no such session"}, 404)
    return sessions[session_id][0].summary()


@app.websocket("/sessions/{session_id}/audio")
async def stream(ws: WebSocket, session_id: str):
    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue(maxsize=2000)

    def on_event(event):                                  # called from the worker thread
        def put():
            if outbox.full():                             # the client is not reading: drop the oldest
                outbox.get_nowait()
            outbox.put_nowait(event)
        try:
            loop.call_soon_threadsafe(put)
        except RuntimeError:                              # this connection is gone; the session stays for a resume
            pass

    async def sender():
        while True:
            event = await outbox.get()
            if event is None:
                return
            await ws.send_text(json.dumps(event))

    send_task = asyncio.create_task(sender())
    session = worker = None
    try:
        first = json.loads(await ws.receive_text())
        if first.get("type") != "start" or int(first.get("sample_rate", 0)) != settings.sample_rate \
                or first.get("encoding", "pcm_s16le") != "pcm_s16le" or int(first.get("channels", 1)) != 1:
            await ws.send_text(json.dumps({"type": "unavailable", "reason": "expected 16 kHz mono pcm_s16le start"}))
            return
        if not models.ready():
            await ws.send_text(json.dumps({"type": "unavailable", "reason": models.error or "models still loading"}))
            return
        start = float(first.get("start", 0.0))
        old = sessions.get(session_id)
        if first.get("resume") and old and abs(old[0].received - start) <= 1.0:
            session, worker = old                          # same conversation, same speaker state
            session.on_event = on_event
        else:
            if old:
                old[1].stop(flush=False)
            profile = enrollment.load(settings.profile, settings)
            session = Session(models, session_id, settings, profile=profile, on_event=on_event)
            if start > 0:
                session.received = start
                session._reset_audio(start)
            worker = Worker(session).start()
            sessions[session_id] = (session, worker)
        await ws.send_text(json.dumps({"type": "ready", "session_id": session_id, "generation": session.generation,
                                       "wearer_profile": session.profile is not None, "step": session.step,
                                       "latency": session.latency, "window": session.duration,
                                       "segmentation": settings.segmentation, "embedding": settings.embedding}))
        at = None
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                pcm = np.frombuffer(message["bytes"], dtype="<i2").astype(np.float32) / 32768
                session.feed(pcm, start=at)
                at = None
            elif message.get("text"):
                cmd = json.loads(message["text"])
                if cmd.get("type") == "at":
                    at = float(cmd["start"])
                elif cmd.get("type") == "end":
                    await asyncio.to_thread(worker.stop, True)
                    on_event(dict(session.summary(), type="summary"))   # same queue: after every speaker event
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None and session.on_event is on_event:
            session.on_event = lambda event: None        # nobody is listening until a resume
        loop.call_soon(outbox.put_nowait, None)
        try:
            await asyncio.wait_for(send_task, timeout=2)
        except Exception:
            send_task.cancel()
