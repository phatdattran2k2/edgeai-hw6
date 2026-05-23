#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題
"""tests/test_mqtt.py — Unit tests for MqttPublisher."""

import json
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest

from src.mqtt_publisher import MqttPublisher, PublisherConfig


@pytest.fixture
def mock_client() -> MagicMock:
    """Return a MagicMock that behaves enough like paho.mqtt.client.Client."""
    client = MagicMock(spec=mqtt.Client)
    # publish() returns a paho-mqtt MQTTMessageInfo-like object
    info = MagicMock()
    info.rc = mqtt.MQTT_ERR_SUCCESS
    client.publish.return_value = info
    return client


@pytest.fixture
def publisher(mock_client: MagicMock) -> MqttPublisher:
    """Build a publisher wired to the mock client (no real network)."""
    return MqttPublisher(PublisherConfig(host="test"), client_factory=lambda: mock_client)


def test_publish_sends_json_payload(publisher: MqttPublisher, mock_client: MagicMock) -> None:
    """publish() must JSON-encode dicts and call client.publish()."""
    # Force connected state for the test
    publisher._connected = True
    payload = {"frame": 1, "ts": 1234567890.0}
    assert publisher.publish("jetson/vision/detections", payload) is True

    args, _ = mock_client.publish.call_args
    topic, body = args
    assert topic == "jetson/vision/detections"
    assert json.loads(body) == payload


def test_publish_when_disconnected_returns_false(
    publisher: MqttPublisher, mock_client: MagicMock
) -> None:
    """publish() before connect() must NOT raise — just return False."""
    assert publisher.connected is False
    assert publisher.publish("any/topic", {"x": 1}) is False
    mock_client.publish.assert_not_called()


def test_publish_string_payload_is_passed_through(
    publisher: MqttPublisher, mock_client: MagicMock
) -> None:
    """If caller passes a str, don't double-JSON-encode it."""
    publisher._connected = True
    publisher.publish("topic", "already-a-string")
    args, _ = mock_client.publish.call_args
    assert args[1] == "already-a-string"  # not '"already-a-string"'


def test_disconnect_stops_loop(publisher: MqttPublisher, mock_client: MagicMock) -> None:
    """disconnect() must stop the paho loop, disconnect client, and update state."""
    publisher._connected = True
    publisher.disconnect()
    mock_client.loop_stop.assert_called_once()
    mock_client.disconnect.assert_called_once()
    assert publisher.connected is False


def test_reconnect_delays_set(publisher: MqttPublisher, mock_client: MagicMock) -> None:
    """Verify the publisher configured paho's exponential reconnect."""
    mock_client.reconnect_delay_set.assert_called_once()


def test_connect_returns_false_on_timeout(mock_client: MagicMock) -> None:
    """connect() phải trả về False nếu timeout."""
    from src.mqtt_publisher import MqttPublisher, PublisherConfig

    pub = MqttPublisher(
        PublisherConfig(host="unreachable"),
        client_factory=lambda: mock_client,
    )
    # Không trigger on_connect callback → timeout
    result = pub.connect(timeout=0.01)
    assert result is False


def test_publish_dict_encodes_to_json(mock_client: MagicMock) -> None:
    """publish() với dict phải encode JSON đúng."""
    from src.mqtt_publisher import MqttPublisher, PublisherConfig

    pub = MqttPublisher(PublisherConfig(), client_factory=lambda: mock_client)
    pub._connected = True
    pub.publish("topic", {"key": "value", "num": 42})
    args, _ = mock_client.publish.call_args
    import json

    assert json.loads(args[1]) == {"key": "value", "num": 42}


def test_publisher_config_defaults() -> None:
    """PublisherConfig defaults phải đúng."""
    from src.mqtt_publisher import PublisherConfig

    cfg = PublisherConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 1883
    assert cfg.keepalive == 60
