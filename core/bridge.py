"""Tiny local HTTP endpoint the browser extension talks to (127.0.0.1 only).

POST /speakers  {"platform": "meet"|"teams", "speaking": ["Name", ...], "t": epoch_ms, "source": "captions"|"tile",
                 "self": "My name in the meeting"}
GET  /status    -> {"app": "MeetingScribe", "recording": bool, "version": 1}

Only requests from a browser extension (Origin chrome-extension://...) are accepted, so ordinary web pages
can't feed fake names into a meeting.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

PORT = 47833


class Bridge:
    def __init__(self, on_speakers: Callable[[dict], None], is_recording: Callable[[], bool]):
        self.on_speakers, self.is_recording = on_speakers, is_recording
        self.server: ThreadingHTTPServer | None = None

    def start(self) -> bool:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _origin_ok(self) -> bool:
                origin = self.headers.get("Origin", "")
                return origin.startswith("chrome-extension://") or origin.startswith("extension://")

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", ""))
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def _reply(self, code: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(code)
                if self._origin_ok():
                    self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                if not self._origin_ok():
                    return self._reply(403, {"error": "forbidden"})
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self):
                if not self._origin_ok():
                    return self._reply(403, {"error": "forbidden"})
                if self.path == "/status":
                    return self._reply(200, {"app": "MeetingScribe", "version": 1, "recording": bridge.is_recording()})
                self._reply(404, {"error": "not found"})

            def do_POST(self):
                if not self._origin_ok() or self.path != "/speakers":
                    return self._reply(403, {"error": "forbidden"})
                try:
                    n = min(int(self.headers.get("Content-Length", "0")), 65536)
                    msg = json.loads(self.rfile.read(n) or b"{}")
                    speaking = [str(x)[:60] for x in msg.get("speaking", [])][:16]
                    event = {"platform": str(msg.get("platform", ""))[:20], "speaking": speaking,
                             "t": float(msg.get("t", 0)) / 1000.0, "source": str(msg.get("source", ""))[:20],
                             "self": str(msg.get("self", ""))[:60]}
                except (ValueError, TypeError):
                    return self._reply(400, {"error": "bad request"})
                bridge.on_speakers(event)
                self._reply(200, {"ok": True, "recording": bridge.is_recording()})

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
        except OSError:  # port taken (another copy running)
            return False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return True

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server = None
