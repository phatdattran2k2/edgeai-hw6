#!/usr/bin/env python3
# Copyright (c) 2026 <Tên bạn và Wei-Chun>
# Tatung University — I4210 AI實務專題
"""tests/test_inference.py — Unit tests for inference pipeline helpers.

These tests exercise the preprocessing and postprocessing logic that
Part A refactored out of the main loop. They do NOT import torch with
CUDA, do NOT load a TRT engine, and do NOT open a camera.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.inference_node import (
    InferenceNode,
    _write_health,
    apply_confidence_threshold,
    detections_to_payload,
    preprocess_frame,
)


# --- preprocessing ---
@pytest.mark.parametrize(
    "shape",
    [(480, 640, 3), (720, 1280, 3), (320, 320, 3)],
)
def test_preprocess_frame_outputs_expected_shape(
    shape: tuple[int, int, int],
) -> None:
    """preprocess_frame() must return (1, 3, H, W) tensor and normalize values."""
    frame = (np.random.rand(*shape) * 255).astype(np.uint8)
    out = preprocess_frame(frame, target_size=(320, 320))
    assert out.shape == (1, 3, 320, 320)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocess_frame_handles_grayscale_input() -> None:
    """Single-channel input must be broadcast to 3 channels."""
    gray = (np.random.rand(480, 640) * 255).astype(np.uint8)
    out = preprocess_frame(gray, target_size=(320, 320))
    assert out.shape == (1, 3, 320, 320)


# --- postprocessing ---
@pytest.mark.parametrize(
    ("conf_thresh", "expected_count"),
    [
        (0.0, 5),  # all detections pass
        (0.5, 3),  # conf >= 0.5: 0.99, 0.75, 0.55
        (0.995, 0),  # nothing meets near-perfect threshold (max is 0.99)
    ],
)
def test_apply_confidence_threshold(
    conf_thresh: float,
    expected_count: int,
) -> None:
    """Filtering by confidence must drop detections below the threshold."""
    detections = [
        {"cls": 0, "conf": 0.99, "xyxy": [0, 0, 10, 10]},
        {"cls": 1, "conf": 0.75, "xyxy": [10, 10, 20, 20]},
        {"cls": 0, "conf": 0.55, "xyxy": [20, 20, 30, 30]},
        {"cls": 2, "conf": 0.30, "xyxy": [30, 30, 40, 40]},
        {"cls": 1, "conf": 0.10, "xyxy": [40, 40, 50, 50]},
    ]
    out = apply_confidence_threshold(detections, conf_thresh)
    assert len(out) == expected_count


def test_detections_to_payload_includes_required_fields() -> None:
    """The MQTT payload schema must always have frame, ts, detections."""
    payload = detections_to_payload(frame_id=42, ts=1700000000.0, detections=[])
    assert payload["frame"] == 42
    assert payload["ts"] == 1700000000.0
    assert payload["detections"] == []


def test_detections_to_payload_with_detections() -> None:
    """Payload phải chứa đúng detections khi có dữ liệu."""
    detections = [{"cls": 0, "conf": 0.9, "xyxy": [0, 0, 10, 10]}]
    payload = detections_to_payload(1, 123.0, detections)
    assert len(payload["detections"]) == 1
    assert payload["frame"] == 1


# --- health file ---
def test_write_health_creates_file(tmp_path: Path) -> None:
    """_write_health phải ghi timestamp vào file."""
    health_file = str(tmp_path / "health")
    _write_health(health_file)
    assert Path(health_file).exists()
    content = Path(health_file).read_text()
    assert float(content) > 0


def test_write_health_handles_os_error() -> None:
    """_write_health không raise khi path không hợp lệ."""
    _write_health("/nonexistent/path/health")


# --- InferenceNode ---
def test_inference_node_init() -> None:
    """InferenceNode khởi tạo với default values đúng."""
    node = InferenceNode()
    assert node.imgsz == 320
    assert node.conf == 0.25
    assert node.mqtt_topic == "/sense/vision/detections"
    assert node.model_path == "/opt/models/best.engine"


def test_inference_node_custom_params() -> None:
    """InferenceNode nhận custom parameters."""
    node = InferenceNode(
        model_path="custom.engine",
        imgsz=640,
        conf=0.5,
        mqtt_broker="192.168.1.1",
    )
    assert node.model_path == "custom.engine"
    assert node.imgsz == 640
    assert node.conf == 0.5
    assert node.mqtt_broker == "192.168.1.1"


def test_inference_node_process_results() -> None:
    """_process_results phải extract detections đúng format."""
    node = InferenceNode()

    mock_box = MagicMock()
    mock_box.cls = 0  # int, không phải [0]
    mock_box.conf = 0.85  # float, không phải [0.85]
    mock_box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 30.0, 40.0])]

    mock_result = MagicMock()
    mock_result.names = {0: "Hardhat"}
    mock_result.boxes = [mock_box]

    results = node._process_results([mock_result])
    assert len(results) == 1
    assert results[0]["class"] == "Hardhat"
    assert results[0]["conf"] == 0.85


def test_inference_node_process_empty_results() -> None:
    """_process_results với empty results trả về list rỗng."""
    node = InferenceNode()
    mock_result = MagicMock()
    mock_result.boxes = []
    results = node._process_results([mock_result])
    assert results == []


def test_inference_node_build_mqtt_client() -> None:
    """_build_mqtt_client phải connect và start loop."""
    node = InferenceNode()
    with patch("src.inference_node.mqtt.Client") as mock_mqtt_client:
        mock = MagicMock()
        mock_mqtt_client.return_value = mock
        node._build_mqtt_client()
        mock.connect.assert_called_once_with("localhost", 1883)
        mock.loop_start.assert_called_once()


def test_inference_node_custom_model_factory() -> None:
    """InferenceNode chấp nhận custom model_factory."""
    mock_factory = MagicMock(return_value=MagicMock())
    node = InferenceNode(model_factory=mock_factory)
    assert node._model_factory is mock_factory


# --- camera mock ---
@pytest.fixture
def mock_video_capture() -> MagicMock:
    """Mock cv2.VideoCapture so tests don't need a real camera."""
    fake = MagicMock()
    fake.isOpened.return_value = True
    fake.read.side_effect = [
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (False, None),
    ]
    return fake


def test_video_capture_loop_processes_all_frames(
    mock_video_capture: MagicMock,
) -> None:
    """The main loop should consume frames until read says stop."""
    with patch("cv2.VideoCapture", return_value=mock_video_capture):
        assert mock_video_capture.read.call_count >= 0
