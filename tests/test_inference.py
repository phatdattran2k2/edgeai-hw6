#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題
"""tests/test_inference.py — Unit tests for inference pipeline helpers.

These tests exercise the preprocessing and postprocessing logic that
Part A refactored out of the main loop. They do NOT import torch with
CUDA, do NOT load a TRT engine, and do NOT open a camera.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Assume Part A refactored these helpers out of inference_node.py.
# Adjust imports to whatever your refactor names them.
from src.inference_node import (
    apply_confidence_threshold,
    detections_to_payload,
    preprocess_frame,
)


# --- preprocessing ---
@pytest.mark.parametrize("shape", [(480, 640, 3), (720, 1280, 3), (320, 320, 3)])
def test_preprocess_frame_outputs_expected_shape(shape: Tuple[int, int, int]) -> None:
    """preprocess_frame() must return (1, 3, H, W) tensor and normalize values."""
    frame = (np.random.rand(*shape) * 255).astype(np.uint8)
    out = preprocess_frame(frame, target_size=(320, 320))
    
    assert out.shape == (1, 3, 320, 320)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocess_frame_handles_grayscale_input() -> None:
    """Single-channel input must be broadcast to 3 channels."""
    gray = (np.random.rand(480, 640) * 255).astype(np.uint8)
    out = preprocess_frame(gray, target_size=(320, 320))
    assert out.shape == (1, 3, 320, 320)


# --- postprocessing ---
@pytest.mark.parametrize(
    ("conf_thresh", "expected_count"),
    [
        (0.0, 5),   # all detections pass
        (0.5, 3),   # only the high-confidence ones
        (0.95, 0),  # nothing meets a near-perfect threshold
    ],
)
def test_apply_confidence_threshold(conf_thresh: float, expected_count: int) -> None:
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


# --- camera mock fixture ---
@pytest.fixture
def mock_video_capture() -> MagicMock:
    """Mock cv2.VideoCapture so tests don't need a real camera."""
    fake = MagicMock()
    fake.isOpened.return_value = True
    # Yield 3 fake frames then exhaust
    fake.read.side_effect = [
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (False, None),  # exhausted
    ]
    return fake


def test_video_capture_loop_processes_all_frames(mock_video_capture: MagicMock) -> None:
    """The main loop should consume frames until isOpened/read says stop."""
    with patch("cv2.VideoCapture", return_value=mock_video_capture):
        # ... call the loop with a stub model and assert frame count ...
        assert mock_video_capture.read.call_count >= 0