"""Stdlib HTTP server for the training dashboard.

Routes (see :mod:`lts2_agent.dashboard` for the data contract):

* ``GET /``                              -> the single-page UI (index.html)
* ``GET /api/runs``                      -> newest-first run summaries
* ``GET /api/runs/<id>/meta``            -> metric names, tag keys, maxStep
* ``GET /api/runs/<id>/series?name=&group_by=&bucket=`` -> downsampled series

No third-party deps and no external assets: the UI is one self-contained HTML file
served from disk, so the whole thing works offline.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from .store import RunStore

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_PATH = os.path.join(_HERE, "index.html")


class DashboardHandler(BaseHTTPRequestHandler):
    # Set by make_server via a subclass attribute.
    store: RunStore = None  # type: ignore[assignment]

    server_version = "Lts2Dashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib name
        # Quiet by default; the dashboard is a background convenience, not a service.
        pass

    # -- helpers ------------------------------------------------------------
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_index(self) -> None:
        try:
            with open(_INDEX_PATH, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send_json({"error": "index.html missing"}, status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    # -- routing ------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802 - stdlib name
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query)

        if path == "/" or path == "/index.html":
            self._send_index()
            return
        if path == "/api/runs":
            self._send_json(self.store.list_runs())
            return

        seg = [unquote(p) for p in path.split("/") if p != ""]
        # ["api", "runs", "<id>", "meta"|"series"]
        if len(seg) == 4 and seg[0] == "api" and seg[1] == "runs":
            run_id = seg[2]
            if seg[3] == "meta":
                meta = self.store.meta(run_id)
                if meta is None:
                    self._send_json({"error": "run not found"}, status=404)
                else:
                    self._send_json(meta)
                return
            if seg[3] == "series":
                self._handle_series(run_id, query)
                return

        self._send_json({"error": "not found", "path": path}, status=404)

    def _handle_series(self, run_id: str, query: dict) -> None:
        name = _first(query, "name")
        if not name:
            self._send_json({"error": "missing name"}, status=400)
            return
        group_by = _first(query, "group_by") or "none"
        bucket_raw = _first(query, "bucket") or "auto"
        bucket: Any
        if bucket_raw == "auto":
            bucket = "auto"
        else:
            try:
                bucket = int(bucket_raw)
            except (TypeError, ValueError):
                bucket = "auto"
        result = self.store.series(run_id, name, group_by=group_by, bucket=bucket)
        if result is None:
            self._send_json({"error": "run not found"}, status=404)
            return
        self._send_json(result)


def _first(query: dict, key: str) -> Optional[str]:
    vals = query.get(key)
    if not vals:
        return None
    return vals[0]


def make_server(root: str, host: str, port: int) -> ThreadingHTTPServer:
    store = RunStore(root)

    class _Handler(DashboardHandler):
        pass

    _Handler.store = store
    httpd = ThreadingHTTPServer((host, port), _Handler)
    return httpd
