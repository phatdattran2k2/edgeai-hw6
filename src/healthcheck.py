#!/usr/bin/env python3
# Copyright (c) 2026 <Tên bạn và Wei-Chun>
# Tatung University — I4210 AI實務專題
"""src/healthcheck.py — Minimal /healthz endpoint for the inference container."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("HEALTHZ_PORT", "8000"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")


def _current_power_mode() -> str:
    """Read live nvpmodel state. Returns empty string if unavailable."""
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in out.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


class HealthCheckServer:
    """Simple HTTP server exposing /healthz endpoint."""

    def __init__(self, port: int = PORT) -> None:
        """Initialize server on given port."""
        self.port = port

    def start_in_thread(self) -> threading.Thread:
        """Start the healthz server on a daemon thread."""
        server = HTTPServer(("0.0.0.0", self.port), _Handler)
        t = threading.Thread(
            target=server.serve_forever, daemon=True, name="healthz",
        )
        t.start()
        return t


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler for /healthz endpoint."""

    def do_GET(self) -> None:
        """Handle GET /healthz request."""
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = json.dumps({
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "power_mode": _current_power_mode(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        """Silence per-request stderr spam."""


def start_in_thread() -> threading.Thread:
    """Start the healthz server on a daemon thread so it dies with main()."""
    server = HealthCheckServer()
    return server.start_in_thread()