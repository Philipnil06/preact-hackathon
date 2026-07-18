from __future__ import annotations

import json

from preact.genesis import GenesisSerial
from preact.mode import OperatingMode
from preact.models import ServoTarget, ToFSample


class FakeSerial:
    def __init__(self, *, fail_writes: bool = False) -> None:
        self.incoming = bytearray()
        self.writes: list[bytes] = []
        self.closed = False
        self.fail_writes = fail_writes

    @property
    def in_waiting(self) -> int:
        return len(self.incoming)

    def read(self, size: int) -> bytes:
        value = bytes(self.incoming[:size])
        del self.incoming[:size]
        return value

    def write(self, value: bytes) -> None:
        if self.fail_writes:
            raise OSError("simulated USB disconnect")
        self.writes.append(value)

    def close(self) -> None:
        self.closed = True


def test_serial_transport_preserves_partial_json_lines() -> None:
    connection = FakeSerial()
    genesis = GenesisSerial("COM-TEST", connection=connection)
    line = (
        b'{"type":"tof","device_time_ms":123,"distance_mm":700,'
        b'"valid":true,"range_status":0}\n'
    )

    connection.incoming.extend(line[:30])
    assert genesis.poll() == []

    connection.incoming.extend(line[30:])
    messages = genesis.poll()

    assert len(messages) == 1
    assert isinstance(messages[0], ToFSample)
    assert messages[0].distance_mm == 700


def test_serial_transport_writes_servo_json_and_closes() -> None:
    connection = FakeSerial()
    genesis = GenesisSerial("COM-TEST", connection=connection)

    genesis.set_servo_target(
        ServoTarget(
            command_id=8,
            angle_deg=125.0,
            target_x_m=2.0,
            confidence=0.8,
            time_to_impact_ms=500,
        )
    )
    value = json.loads(connection.writes[-1])

    assert value["type"] == "servo_target"
    assert value["command_id"] == 8
    genesis.set_mode(command_id=9, mode=OperatingMode.LANDING)
    mode_value = json.loads(connection.writes[-1])
    assert mode_value["type"] == "mode"
    assert mode_value["mode"] == "landing"
    genesis.close()
    assert connection.closed
    assert not genesis.connected


def test_serial_reconnect_replays_latest_mode_then_safe_actuator(
    monkeypatch: object,
) -> None:
    first = FakeSerial(fail_writes=True)
    second = FakeSerial()
    monkeypatch.setattr("preact.genesis.serial.Serial", lambda **_: second)
    genesis = GenesisSerial(
        "COM-TEST",
        connection=first,
        reconnect_interval_s=0.0,
    )

    genesis.set_mode(command_id=9, mode=OperatingMode.LANDING)
    assert not genesis.connected
    genesis.neutral(command_id=10, angle_deg=90.0)

    assert genesis.connected
    assert [json.loads(payload)["type"] for payload in second.writes] == [
        "mode",
        "neutral",
    ]
    assert json.loads(second.writes[0])["mode"] == "landing"
    assert json.loads(second.writes[1])["angle_deg"] == 90.0
