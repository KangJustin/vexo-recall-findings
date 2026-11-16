"""A tiny local web server for the demo screen. Standard library only.

  GET  /              the page
  GET  /events        live updates (server-sent events)
  GET  /state         the current state as JSON
  POST /api/name      {"speaker": 1, "name": "Maya"}      the person typed a name
  POST /api/clear     {"speaker": 1}                       "Not Maya?"
  POST /api/me        {"speaker": 0}                       "That's me"
  POST /api/merge     {"from": 2, "to": 1}                 "Same person"
  POST /api/say       {"speaker": 1, "text": "..."}        type a line (for testing or as a fallback)
  POST /api/suggestion {"id": 3, "same": true}              answer a "Same person?" question
  POST /api/proposal  {"id": 4, "apply": true}              apply (or dismiss) a who-said-it suggestion
  POST /api/revert    {"id": 4}                             undo an applied correction
  POST /api/reset

It listens on 127.0.0.1 only, so nobody else on the Wi-Fi can see the conversation.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Hub:
    """Wakes every open /events connection when the state changes."""

    def __init__(self):
        self.cond = threading.Condition()
        self.rev = 0
        self.snapshot = None

    def publish(self, snapshot):
        with self.cond:
            self.rev += 1
            self.snapshot = snapshot
            self.cond.notify_all()


def make_server(resolver, hub, port=8765):
    hub.snapshot = resolver.snapshot()
    resolver.on_change = hub.publish

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if self.path == "/debug":
                return self._send(200, json.dumps(resolver.diagnostics))
            if self.path == "/state":
                return self._send(200, json.dumps(resolver.snapshot()))
            if self.path == "/events":
                return self._events()
            self._send(404, '{"error":"not found"}')

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with hub.cond:
                        if hub.rev == seen:
                            hub.cond.wait(timeout=10)
                        rev, snap = hub.rev, hub.snapshot
                    if rev != seen:
                        seen = rev
                        self.wfile.write(("data: %s\n\n" % json.dumps(snap)).encode("utf-8"))
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(size) or b"{}")
            except (ValueError, TypeError):
                return self._send(400, '{"error":"bad json"}')
            path = self.path
            try:
                if path == "/api/name":
                    resolver.set_name(int(body["speaker"]), str(body.get("name", "")))
                elif path == "/api/clear":
                    resolver.clear_name(int(body["speaker"]))
                elif path == "/api/me":
                    resolver.set_me(int(body["speaker"]), bool(body.get("me", True)))
                elif path == "/api/merge":
                    resolver.merge(int(body["from"]), int(body["to"]))
                elif path == "/api/say":
                    resolver.feed(int(body.get("speaker", 0)), str(body.get("text", "")), True)
                elif path == "/api/proposal":            # a second-pass / review suggestion: apply or dismiss
                    if body.get("apply"):
                        resolver.apply_proposal(int(body["id"]))
                    else:
                        resolver.dismiss_proposal(int(body["id"]))
                elif path == "/api/revert":
                    resolver.revert_correction(int(body["id"]))
                elif path == "/api/suggestion":
                    resolver.answer_suggestion(int(body["id"]), bool(body.get("same")))
                elif path == "/api/reset":
                    resolver.reset()
                else:
                    return self._send(404, '{"error":"not found"}')
            except (KeyError, ValueError, TypeError):
                return self._send(400, '{"error":"bad request"}')
            self._send(200, '{"ok":true}')

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def make_voice_server(tracker, port=8765, started=None):
    """Voice only: GET /voice (the page), GET /voice-state (who is talking now, recent changes)."""
    import time as _time
    started = started or _time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/voice"):
                with open(os.path.join(WEB, "voice.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if self.path == "/voice-state":
                last = dict(tracker.last)
                now = _time.monotonic()
                last["elapsed"] = round(now - started, 2)
                last["history"] = [[round(t - started, 2), s, sc] for t, s, sc in tracker.history[-80:]]
                last["has_baseline"] = tracker.me is not None
                last["threshold"] = tracker.me["threshold"] if tracker.me else None
                return self._send(200, json.dumps(last))
            self._send(404, '{"error":"not found"}')

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
