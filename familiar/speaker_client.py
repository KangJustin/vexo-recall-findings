"""Streams the band's decoded audio to the speaker service (speaker_service/app.py) and passes its events on.

The audio is the SAME decoded 16 kHz PCM the app keeps for voice matching, in the same order Deepgram receives
the Opus packets, so the service's times and Deepgram's word timestamps share one clock.

Never in the way of transcription: feed() only queues. The queue is bounded (MAX_BUFFER seconds); if the service
is unreachable or too slow, the oldest audio is dropped, the drop is reported as a gap, and attribution is marked
unavailable until the service is caught up again. Reconnects resume the same session at the right time.
"""
import json
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vexo-open-kit",
                                "examples", "python"))

SR = 16000
MAX_BUFFER = 10.0        # seconds of audio queued at most while the service is away or slow


class SpeakerClient:
    def __init__(self, url, session_id, on_event, log=None, max_buffer=MAX_BUFFER, connect=None):
        self.url = url.rstrip("/")
        self.session_id = session_id
        self.on_event = on_event
        self.log = log or (lambda text: print(text, file=sys.stderr))
        self.max_samples = int(max_buffer * SR)
        self._connect = connect or self._websocket
        self._queue = deque()                 # (start seconds, pcm bytes)
        self._queued = 0
        self._sent_until = 0.0                # seconds of audio handed to the service
        self._clock = 0.0                     # seconds of audio fed so far (the session clock)
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._ws = None
        self.status = "connecting"
        self.stats = {"dropped_seconds": 0.0, "reconnects": 0, "events": 0}

    # ------------------------------------------------------------------ audio in (band thread)

    def feed(self, pcm):
        n = len(pcm) // 2
        with self._lock:
            self._queue.append((self._clock, pcm))
            self._queued += n
            self._clock += n / SR
            while self._queued > self.max_samples:              # bounded: drop the oldest, and say so
                start, old = self._queue.popleft()
                self._queued -= len(old) // 2
                self.stats["dropped_seconds"] += len(old) / 2 / SR
                self._mark("unavailable", "speaker service not keeping up: audio from %.1f s dropped" % start)
            self._lock.notify()

    # ------------------------------------------------------------------ connection (own threads)

    def start(self):
        threading.Thread(target=self._run, name="speaker-client", daemon=True).start()
        return self

    def restart(self, session_id):
        """A new conversation: a new service session (fresh speaker state), from the current point in time."""
        self.session_id = session_id
        self._fresh = True
        ws = self._ws
        if ws is not None:
            try:
                ws.close()                                  # the loops reconnect, without resuming
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        with self._lock:
            self._lock.notify()
        ws = self._ws
        if ws is not None:
            try:
                ws.send(json.dumps({"type": "end"}))
            except Exception:
                pass

    def _websocket(self):
        from tiny_websocket import WebSocket
        return WebSocket("%s/sessions/%s/audio" % (self.url, self.session_id), timeout=5)

    def _mark(self, status, reason):
        if status != self.status:
            self.status = status
            self.on_event({"type": "status", "attribution": status, "reason": reason, "source": "client"})

    def _run(self):
        delay, resume = 1.0, False
        while not self._stop.is_set():
            try:
                ws = self._connect()
                if getattr(self, "_fresh", False):
                    resume, self._fresh = False, False
                ws.send(json.dumps({"type": "start", "sample_rate": SR, "encoding": "pcm_s16le", "channels": 1,
                                    "start": round(self._next_start(), 3), "resume": resume}))
                first = json.loads(ws.recv(timeout=10) or "{}")
                if first.get("type") != "ready":
                    self._mark("unavailable", "speaker service: %s" % first.get("reason", "no answer"))
                    ws.close()
                    self._stop.wait(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                self._ws, delay, resume = ws, 1.0, True
                self._mark("available", "connected")
                reader = threading.Thread(target=self._read, args=(ws,), daemon=True)
                reader.start()
                self._send_loop(ws)
            except Exception as problem:
                self._mark("unavailable", "speaker service unreachable (%s)" % type(problem).__name__)
            self._ws = None
            if not self._stop.is_set():
                self.stats["reconnects"] += 1
                self._stop.wait(delay)
                delay = min(delay * 2, 30.0)

    def _next_start(self):
        with self._lock:
            return self._queue[0][0] if self._queue else self._clock

    def _send_loop(self, ws):
        while not self._stop.is_set():
            with self._lock:
                while not self._queue and not self._stop.is_set():
                    self._lock.wait(0.5)
                if self._stop.is_set():
                    return
                start, pcm = self._queue.popleft()
                self._queued -= len(pcm) // 2
            if abs(start - self._sent_until) > 0.001:             # audio was dropped: tell the service where we are
                ws.send(json.dumps({"type": "at", "start": round(start, 3)}))
            ws.send(pcm)
            self._sent_until = start + len(pcm) / 2 / SR

    def _read(self, ws):
        while True:
            try:
                message = ws.recv()
            except Exception:
                return
            if message is None:
                return
            if isinstance(message, str):
                try:
                    event = json.loads(message)
                except ValueError:
                    continue
                self.stats["events"] += 1
                if event.get("type") == "status":
                    self.status = event.get("attribution", self.status)
                self.on_event(event)
