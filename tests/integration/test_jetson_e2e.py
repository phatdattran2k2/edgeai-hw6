#!/usr/bin/env python3
# Copyright (c) 2026 Tran Phat Dat & Yang Wei-Chun
# Tatung University — I4210 AI實務專題
"""tests/integration/test_jetson_e2e.py — End-to-end inference test on Jetson."""

import os
import subprocess
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import pytest

IMAGE = os.environ.get("IMAGE", "")
SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
MQTT_TOPIC = "/sense/vision/detections"
BROKER = "localhost"
CONTAINER_NAME = "hw6-integration-test"


@pytest.fixture(scope="module")
def inference_container():
    """Start inference container, yield, cleanup."""
    assert IMAGE, "Set IMAGE env var to the GHCR image to test"
    assert SAMPLE_FRAME.exists(), f"Missing {SAMPLE_FRAME}"

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--runtime", "nvidia",
        "--network", "host",
        "-v", "lab12-models:/opt/models",
        "-v", f"{SAMPLE_FRAME.resolve()}:/opt/data/test_video.mp4:ro",
        "-e", "MQTT_BROKER=localhost",
        IMAGE,
    ]
    subprocess.run(cmd, check=True)
    time.sleep(30)

    yield CONTAINER_NAME

    subprocess.run(["docker", "stop", CONTAINER_NAME], check=False)
    subprocess.run(["docker", "rm", CONTAINER_NAME], check=False)


def test_image_is_per_commit_sha_tagged() -> None:
    """Image tag must be sha-<short>, not :latest."""
    assert IMAGE != "", "IMAGE env var must be set"
    assert "sha-" in IMAGE, f"Expected sha- tag, got: {IMAGE}"


def test_container_is_running(inference_container: str) -> None:
    """Container must be running after startup."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", inference_container],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "true", \
        f"Container {inference_container} is not running"


def test_inference_publishes_mqtt_within_window(
    inference_container: str,
) -> None:
    """Container must publish at least 1 MQTT message within 30s."""
    received = threading.Event()
    messages: list[bytes] = []

    def on_message(
        client: mqtt.Client,
        userdata: None,
        msg: mqtt.MQTTMessage,
    ) -> None:
        messages.append(msg.payload)
        received.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(BROKER, 1883, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_start()

    got_message = received.wait(timeout=30)

    client.loop_stop()
    client.disconnect()

    assert got_message, f"No MQTT message on {MQTT_TOPIC} within 30s"
    assert len(messages) >= 1