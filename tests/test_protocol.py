from __future__ import annotations

import json

from preact.mode import OperatingMode
from preact.models import ServoTarget, ToFSample
from preact.protocol import encode_mode, encode_servo_target, parse_device_message


def test_parses_tof_message() -> None:
    message = parse_device_message(
        '{"type":"tof","device_time_ms":1234,"distance_mm":712,'
        '"valid":true,"range_status":0}\n',
        received_at_s=8.5,
    )
    assert isinstance(message, ToFSample)
    assert message.received_at_s == 8.5
    assert message.device_time_ms == 1234
    assert message.distance_mm == 712
    assert message.valid


def test_encodes_servo_target_as_newline_delimited_json() -> None:
    encoded = encode_servo_target(
        ServoTarget(
            command_id=7,
            angle_deg=123.5,
            target_x_m=2.1,
            confidence=0.83,
            time_to_impact_ms=620,
        )
    )
    assert encoded.endswith(b"\n")
    value = json.loads(encoded)
    assert value["type"] == "servo_target"
    assert value["protocol_version"] == 1
    assert value["command_id"] == 7
    assert value["angle_deg"] == 123.5


def test_encodes_mode_indicator_command() -> None:
    value = json.loads(
        encode_mode(command_id=11, mode=OperatingMode.TRACKING)
    )
    assert value == {
        "type": "mode",
        "protocol_version": 1,
        "command_id": 11,
        "mode": "tracking",
    }
