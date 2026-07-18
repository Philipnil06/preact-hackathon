"""Newline-delimited JSON protocol between the laptop and Genesis Mini."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from preact.mode import OperatingMode
from preact.models import ServoTarget, ToFSample

PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class DeviceHello:
    device: str
    protocol_version: int
    firmware_version: str


@dataclass(frozen=True, slots=True)
class DeviceAck:
    command_id: int
    accepted: bool
    detail: str


DeviceMessage = ToFSample | DeviceHello | DeviceAck


def parse_device_message(line: str | bytes, *, received_at_s: float) -> DeviceMessage:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Genesis JSON message") from error
    if not isinstance(value, dict):
        raise ValueError("Genesis message must be a JSON object")

    message_type = value.get("type")
    if message_type == "tof":
        return ToFSample(
            received_at_s=float(received_at_s),
            device_time_ms=_required_int(value, "device_time_ms"),
            distance_mm=_required_int(value, "distance_mm"),
            valid=bool(value.get("valid", False)),
            range_status=int(value.get("range_status", 0)),
        )
    if message_type == "hello":
        return DeviceHello(
            device=str(value.get("device", "unknown")),
            protocol_version=_required_int(value, "protocol_version"),
            firmware_version=str(value.get("firmware_version", "unknown")),
        )
    if message_type == "ack":
        return DeviceAck(
            command_id=_required_int(value, "command_id"),
            accepted=bool(value.get("accepted", False)),
            detail=str(value.get("detail", "")),
        )
    raise ValueError(f"unsupported Genesis message type: {message_type!r}")


def encode_servo_target(target: ServoTarget) -> bytes:
    return _encode(
        {
            "type": "servo_target",
            "protocol_version": PROTOCOL_VERSION,
            **asdict(target),
        }
    )


def encode_neutral(*, command_id: int, angle_deg: float) -> bytes:
    return _encode(
        {
            "type": "neutral",
            "protocol_version": PROTOCOL_VERSION,
            "command_id": command_id,
            "angle_deg": float(angle_deg),
        }
    )


def encode_mode(*, command_id: int, mode: OperatingMode) -> bytes:
    return _encode(
        {
            "type": "mode",
            "protocol_version": PROTOCOL_VERSION,
            "command_id": command_id,
            "mode": mode.value,
        }
    )


def encode_ping(*, nonce: int) -> bytes:
    return _encode(
        {
            "type": "ping",
            "protocol_version": PROTOCOL_VERSION,
            "nonce": int(nonce),
        }
    )


def _encode(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")


IntegerField = Literal[
    "device_time_ms",
    "distance_mm",
    "protocol_version",
    "command_id",
]


def _required_int(value: dict[str, Any], key: IntegerField) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{key} must be an integer")
    return item
