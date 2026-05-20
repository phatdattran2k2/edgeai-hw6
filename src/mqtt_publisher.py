#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題
"""src/mqtt_publisher.py — Thin paho-mqtt wrapper with reconnect + JSON encoding.

Pulled out of inference_node so it's unit-testable without starting a real broker.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


@dataclass
class PublisherConfig:
    """Configuration data class for the MqttPublisher."""

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    client_id: str = ""
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 30.0


class MqttPublisher:
    """Publish JSON messages to MQTT with automatic reconnection.

    The constructor takes an optional `client_factory` so tests can
    inject a mock paho-mqtt Client without monkeypatching globals.
    """

    def __init__(self, config: PublisherConfig,
                client_factory=None):
        self.config = config
        factory = client_factory or (lambda: mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=config.client_id,
        ))
        self.client = factory()
        # Thêm dòng này:
        self.client.reconnect_delay_set(
            config.reconnect_min_delay,
            config.reconnect_max_delay,
        )
        self._connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        self.client.reconnect_delay_set(
            self.config.reconnect_min_delay,
            self.config.reconnect_max_delay
        )
        def on_connect(client, userdata, flags, rc, properties=None):
            self._connected = (rc == 0)
        self.client.on_connect = on_connect
        self.client.connect(self.config.host, self.config.port, self.config.keepalive)
        self.client.loop_start()
        import time
        deadline = time.time() + timeout
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)
        return self._connected

    def publish(self, topic: str, payload: Any) -> bool:
        if not self._connected:
            return False
        body = payload if isinstance(payload, str) else json.dumps(payload)
        info = self.client.publish(topic, body)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def disconnect(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        """Return the current connection status of the MQTT client."""
        return self._connected