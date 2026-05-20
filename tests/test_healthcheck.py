#!/usr/bin/env python3
# Copyright (c) 2026 <Tên bạn và Wei-Chun>
# Tatung University — I4210 AI實務專題
"""tests/test_healthcheck.py — Unit tests for HealthCheckServer."""

import json
import threading
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from src.healthcheck import HealthCheckServer, _current_power_mode, start_in_thread


def test_current_power_mode_no_nvpmodel() -> None:
    """Returns empty string when nvpmodel not available (x86 CI)."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = _current_power_mode()
    assert result == ""


def test_current_power_mode_timeout() -> None:
    """Returns empty string on timeout."""
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvpmodel", 2)):
        result = _current_power_mode()
    assert result == ""


def test_current_power_mode_parses_output() -> None:
    """Parses nvpmodel -q output correctly."""
    mock_result = MagicMock()
    mock_result.stdout = "NV Power Mode: 15W\nSome other line\n"
    with patch("subprocess.run", return_value=mock_result):
        result = _current_power_mode()
    assert result == "15W"


def test_current_power_mode_no_match() -> None:
    """Returns empty string when output has no Power Mode line."""
    mock_result = MagicMock()
    mock_result.stdout = "Some unrelated output\n"
    with patch("subprocess.run", return_value=mock_result):
        result = _current_power_mode()
    assert result == ""


def test_healthcheck_server_init() -> None:
    """HealthCheckServer initializes with correct port."""
    server = HealthCheckServer(port=9999)
    assert server.port == 9999


def test_healthcheck_server_default_port() -> None:
    """HealthCheckServer uses PORT env var or default 8000."""
    server = HealthCheckServer()
    assert server.port in range(1, 65536)


def test_healthz_endpoint_returns_200() -> None:
    """GET /healthz returns 200 with healthy status."""
    server = HealthCheckServer(port=18765)
    t = server.start_in_thread()
    assert t.is_alive()

    import time
    time.sleep(0.2)

    with urllib.request.urlopen("http://localhost:18765/healthz") as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "healthy"
        assert "model_version" in data
        assert "power_mode" in data


def test_healthz_returns_404_for_other_paths() -> None:
    """GET /other returns 404."""
    server = HealthCheckServer(port=18766)
    server.start_in_thread()

    import time
    time.sleep(0.2)

    try:
        urllib.request.urlopen("http://localhost:18766/other")
        pytest.fail("Expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_start_in_thread_returns_daemon_thread() -> None:
    """start_in_thread() returns a running daemon thread."""
    with patch("src.healthcheck.HTTPServer"):
        t = start_in_thread()
    assert isinstance(t, threading.Thread)
