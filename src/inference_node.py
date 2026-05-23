#!/usr/bin/env python3
# Copyright (c) 2026 <Tên bạn và Wei-Chun>
# Tatung University — I4210 AI實務專題
"""src/inference_node.py — Edge-AI YOLO inference pipeline with MQTT publishing."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

try:
    from src import healthcheck
except ImportError:
    import healthcheck  # type: ignore[no-redef]

_running = True


def _signal_handler(sig: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _running
    print(f"[inference] Received signal {sig}, shutting down...")
    _running = False


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _default_model_factory(path: str, task: str) -> Any:  # pragma: no cover
    """Load YOLO model lazily — ultralytics only available on Jetson."""
    from ultralytics import YOLO  # noqa: PLC0415
    return YOLO(path, task=task)


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] = (320, 320),
) -> np.ndarray:
    """Resize and normalize frame to (1, 3, H, W) float32 in [0, 1]."""
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    resized = cv2.resize(frame, target_size)
    chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...]


def apply_confidence_threshold(
    detections: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Filter detections below confidence threshold."""
    return [d for d in detections if d["conf"] >= threshold]


def detections_to_payload(
    frame_id: int,
    ts: float,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build MQTT JSON payload from detection results."""
    return {
        "frame": frame_id,
        "ts": ts,
        "detections": detections,
    }


def _write_health(health_path: str) -> None:
    """Write heartbeat timestamp to health file."""
    try:
        with open(health_path, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


class InferenceNode:
    """YOLO TensorRT inference node with MQTT publishing."""

    def __init__(
        self,
        model_path: str = "/opt/models/best.engine",
        source: str = "/opt/data/test_video.mp4",
        imgsz: int = 320,
        conf: float = 0.25,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic: str = "/sense/vision/detections",
        health_path: str = "/app/inference_health",
        model_factory: Any = None,
    ) -> None:
        """Initialize inference node with configurable parameters."""
        self.model_path = model_path
        self.source = source
        self.imgsz = imgsz
        self.conf = conf
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.health_path = health_path
        self._model_factory = model_factory or _default_model_factory

    def _build_mqtt_client(self) -> mqtt.Client:
        """Create and connect MQTT client."""
        client = mqtt.Client(CallbackAPIVersion.VERSION2)
        client.connect(self.mqtt_broker, self.mqtt_port)
        client.loop_start()
        return client

    def _process_results(
        self,
        results: Any,
    ) -> list[dict[str, Any]]:
        """Extract detections from YOLO results."""
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls)],
                    "conf": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                })
        return detections

    def run(self) -> None:  # pragma: no cover
        """Main inference loop — runs on Jetson only."""
        global _running
        model = self._model_factory(self.model_path, "detect")
        client = self._build_mqtt_client()
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[inference] ERROR: Cannot open source: {self.source}")
            sys.exit(1)

        frame_count = 0
        fps_start = time.monotonic()
        print(f"[inference] Running inference on {self.source}...")

        while _running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            results = model.predict(
                frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
            )
            detections = self._process_results(results)
            payload = detections_to_payload(frame_count, time.time(), detections)
            client.publish(self.mqtt_topic, json.dumps(payload), qos=0)
            frame_count += 1

            if frame_count % 10 == 0:
                _write_health(self.health_path)

            if frame_count % 100 == 0:
                elapsed = time.monotonic() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"[inference] {frame_count} frames, {fps:.1f} FPS, "
                    f"last: {len(detections)} detections",
                )

        cap.release()
        client.loop_stop()
        client.disconnect()
        print(f"[inference] Done. Processed {frame_count} frames.")


def main() -> None:  # pragma: no cover
    """CLI entry point."""
    healthcheck.start_in_thread()
    parser = argparse.ArgumentParser(description="YOLO TensorRT inference node")
    parser.add_argument("--model", default="/opt/models/best.engine")
    parser.add_argument("--source", default="/opt/data/test_video.mp4")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"),
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")),
    )
    parser.add_argument("--mqtt-topic", default="/sense/vision/detections")
    args = parser.parse_args()

    node = InferenceNode(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
    )
    node.run()


if __name__ == "__main__":
    main()