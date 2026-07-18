"""Serial transport and in-memory Genesis emulator."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import serial

from preact.mode import OperatingMode
from preact.models import ServoTarget
from preact.protocol import (
    DeviceMessage,
    encode_mode,
    encode_neutral,
    encode_servo_target,
    parse_device_message,
)


class GenesisTransport(Protocol):
    def poll(self) -> list[DeviceMessage]: ...

    def set_servo_target(self, target: ServoTarget) -> None: ...

    def neutral(self, *, command_id: int, angle_deg: float) -> None: ...

    def set_mode(self, *, command_id: int, mode: OperatingMode) -> None: ...

    def close(self) -> None: ...


class GenesisSerial:
    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        timeout_s: float = 0.0,
        reconnect_interval_s: float = 0.75,
        connection: Any | None = None,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout_s = timeout_s
        self._reconnect_interval_s = reconnect_interval_s
        self._serial = connection or serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout_s,
        )
        self._receive_buffer = bytearray()
        self._closed = False
        self._next_reconnect_at_s = 0.0
        self._desired_mode: bytes | None = None
        self._desired_actuator: bytes | None = None
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._serial is not None and not self._closed

    def poll(self) -> list[DeviceMessage]:
        messages: list[DeviceMessage] = []
        connection, reconnected = self._ensure_connection()
        if connection is None:
            return messages
        if reconnected and not self._replay_desired_state(connection):
            return messages
        try:
            waiting = int(connection.in_waiting)
            if waiting > 0:
                self._receive_buffer.extend(connection.read(waiting))
        except (OSError, serial.SerialException) as error:
            self._disconnect(error)
            return messages

        while b"\n" in self._receive_buffer:
            raw_line, _, remainder = self._receive_buffer.partition(b"\n")
            self._receive_buffer = bytearray(remainder)
            if not raw_line:
                continue
            try:
                messages.append(
                    parse_device_message(
                        raw_line,
                        received_at_s=time.monotonic(),
                    )
                )
            except ValueError:
                continue
        if len(self._receive_buffer) > 16_384:
            self._receive_buffer.clear()
            self.last_error = "Genesis receive buffer exceeded 16384 bytes"
        return messages

    def set_servo_target(self, target: ServoTarget) -> None:
        payload = encode_servo_target(target)
        self._desired_actuator = payload
        self._write(payload)

    def neutral(self, *, command_id: int, angle_deg: float) -> None:
        payload = encode_neutral(command_id=command_id, angle_deg=angle_deg)
        self._desired_actuator = payload
        self._write(payload)

    def set_mode(self, *, command_id: int, mode: OperatingMode) -> None:
        payload = encode_mode(command_id=command_id, mode=mode)
        self._desired_mode = payload
        self._write(payload)

    def close(self) -> None:
        self._closed = True
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _write(self, value: bytes) -> None:
        connection, reconnected = self._ensure_connection()
        if connection is None:
            return
        if reconnected:
            self._replay_desired_state(connection)
            return
        try:
            connection.write(value)
        except (OSError, serial.SerialException) as error:
            self._disconnect(error)

    def _ensure_connection(self) -> tuple[Any | None, bool]:
        if self._closed:
            return None, False
        if self._serial is not None:
            return self._serial, False
        now = time.monotonic()
        if now < self._next_reconnect_at_s:
            return None, False
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self._timeout_s,
            )
            self.last_error = None
            self._receive_buffer.clear()
        except (OSError, serial.SerialException) as error:
            self.last_error = str(error)
            self._next_reconnect_at_s = now + self._reconnect_interval_s
            return None, False
        return self._serial, True

    def _replay_desired_state(self, connection: Any) -> bool:
        """Restore mode first, then the latest safe actuator state."""

        try:
            if self._desired_mode is not None:
                connection.write(self._desired_mode)
            if self._desired_actuator is not None:
                connection.write(self._desired_actuator)
        except (OSError, serial.SerialException) as error:
            self._disconnect(error)
            return False
        return True

    def _disconnect(self, error: Exception) -> None:
        self.last_error = str(error)
        connection = self._serial
        self._serial = None
        self._receive_buffer.clear()
        self._next_reconnect_at_s = time.monotonic() + self._reconnect_interval_s
        if connection is not None:
            try:
                connection.close()
            except (OSError, serial.SerialException):
                pass


@dataclass(frozen=True, slots=True)
class EmulatedServoState:
    angle_deg: float
    command_id: int


class GenesisEmulator:
    """In-memory transport used by tests and the synthetic demo."""

    def __init__(self, *, neutral_angle_deg: float = 90.0) -> None:
        self._messages: deque[DeviceMessage] = deque()
        self.servo_state = EmulatedServoState(
            angle_deg=neutral_angle_deg,
            command_id=0,
        )
        self.targets: list[ServoTarget] = []
        self.mode = OperatingMode.IDLE
        self.tracking_led_on = False
        self.landing_led_on = False
        self.closed = False

    def queue_message(self, message: DeviceMessage) -> None:
        self._messages.append(message)

    def poll(self) -> list[DeviceMessage]:
        messages = list(self._messages)
        self._messages.clear()
        return messages

    def set_servo_target(self, target: ServoTarget) -> None:
        self.targets.append(target)
        self.servo_state = EmulatedServoState(
            angle_deg=target.angle_deg,
            command_id=target.command_id,
        )

    def neutral(self, *, command_id: int, angle_deg: float) -> None:
        self.servo_state = EmulatedServoState(
            angle_deg=angle_deg,
            command_id=command_id,
        )

    def set_mode(self, *, command_id: int, mode: OperatingMode) -> None:
        del command_id
        self.mode = mode
        self.tracking_led_on = mode is OperatingMode.TRACKING
        self.landing_led_on = mode is OperatingMode.LANDING

    def close(self) -> None:
        self.closed = True
