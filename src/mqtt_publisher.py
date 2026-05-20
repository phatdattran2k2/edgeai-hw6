#!/usr/bin/env python3
# Copyright (c) 2026 <Ten ban va Wei-Chun>
# Tatung University - I4210 AI Topic
"""src/mqtt_publisher.py - Thin paho-mqtt wrapper with reconnect + JSON encoding."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


@dataclass
class PublisherConfig:
    """Configuration for MqttPublisher."""

    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    client_id: str = ""
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 30.0


class MqttPublisher:
    """Publish JSON messages to MQTT with automatic reconnection.

    The constructor takes an optional client_factory so tests can
    inject a mock paho-mqtt Client without monkeypatching globals.
    """

    def __init__(
        self,
        config: PublisherConfig,
        client_factory: Callable[[], mqtt.Client] | None = None,
    ) -> None:
        """Initialize publisher with config and optional mock client factory."""
        self.config = config
        factory: Callable[[], mqtt.Client] = client_factory or (
            lambda: mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=config.client_id,
            )
        )
        self.client = factory()
        self.client.reconnect_delay_set(
            config.reconnect_min_delay,
            config.reconnect_max_delay,
        )
        self._connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to broker and wait up to timeout seconds for confirmation."""

        def on_connect(
            client: mqtt.Client,
            userdata: str,
            flags: dict[str, int],
            rc: int,
            properties: mqtt.Properties | None = None,
        ) -> None:
            self._connected = rc == 0

        self.client.on_connect = on_connect
        self.client.connect(self.config.host, self.config.port, self.config.keepalive)
        self.client.loop_start()
        deadline = time.time() + timeout
        while time.time() < deadline and not self._connected:
            time.sleep(0.05)
        return self._connected

    def publish(self, topic: str, payload: str | dict | list) -> bool:
        """Publish payload to topic; JSON-encode dicts, pass strings through."""
        if not self._connected:
            return False
        body = payload if isinstance(payload, str) else json.dumps(payload)
        info = self.client.publish(topic, body)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def disconnect(self) -> None:
        """Stop loop and disconnect from broker."""
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        """Return current connection state."""
        return self._connected
