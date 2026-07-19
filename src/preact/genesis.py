"""Serial transport and in-memory Genesis emulator."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import serial

from preact.mode import OperatingMode
from preact.models import PanTiltTarget, ServoTarget
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

    def set_pan_tilt_target(self, target: PanTiltTarget) -> None: ...

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

    def set_pan_tilt_target(self, target: PanTiltTarget) -> None:
        del target
        raise RuntimeError("JSON Genesis firmware does not support pan/tilt commands")

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
        self.pan_tilt_targets: list[PanTiltTarget] = []
        self.pan_angle_deg = neutral_angle_deg
        self.tilt_angle_deg = neutral_angle_deg
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

    def set_pan_tilt_target(self, target: PanTiltTarget) -> None:
        self.pan_tilt_targets.append(target)
        self.pan_angle_deg = target.pan_angle_deg
        self.tilt_angle_deg = target.tilt_angle_deg

    def neutral(self, *, command_id: int, angle_deg: float) -> None:
        self.servo_state = EmulatedServoState(
            angle_deg=angle_deg,
            command_id=command_id,
        )
        self.pan_angle_deg = angle_deg
        self.tilt_angle_deg = angle_deg

    def set_mode(self, *, command_id: int, mode: OperatingMode) -> None:
        del command_id
        self.mode = mode
        self.tracking_led_on = mode is OperatingMode.TRACKING
        self.landing_led_on = mode is OperatingMode.LANDING

    def close(self) -> None:
        self.closed = True


class PanTiltSerial:
    """Serial transport for the Axiometa S1:/S3: text servo firmware."""

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
        self._closed = False
        self._next_reconnect_at_s = 0.0
        self._receive_buffer = bytearray()
        self._desired_angles: tuple[int, int] = (90, 90)
        self._sent_angles: tuple[int, int] | None = None
        self._acknowledged_angles: tuple[int | None, int | None] = (None, None)
        self._last_write_at_s = 0.0
        self._ack_retry_interval_s = 0.25
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._serial is not None and not self._closed

    def poll(self) -> list[DeviceMessage]:
        connection, reconnected = self._ensure_connection()
        if connection is None:
            return []
        if reconnected and not self._replay_desired_state(connection):
            return []
        try:
            waiting = int(connection.in_waiting)
            if waiting > 0:
                self._receive_buffer.extend(connection.read(waiting))
        except (OSError, serial.SerialException) as error:
            self._disconnect(error)
            return []
        while b"\n" in self._receive_buffer:
            raw_line, _, remainder = self._receive_buffer.partition(b"\n")
            self._receive_buffer = bytearray(remainder)
            self._record_acknowledgement(raw_line)
        if len(self._receive_buffer) > 1024:
            self._receive_buffer.clear()
        if (
            self._acknowledged_angles != self._desired_angles
            and time.monotonic() - self._last_write_at_s
            >= self._ack_retry_interval_s
        ):
            self._write(self._encode_replay(*self._desired_angles))
        return []

    @property
    def acknowledged_angles(self) -> tuple[int | None, int | None]:
        return self._acknowledged_angles

    def _record_acknowledgement(self, raw_line: bytes) -> None:
        try:
            line = raw_line.decode("ascii").strip()
            axis, raw_angle = line.split(":", 1)
            angle = int(raw_angle)
        except (UnicodeDecodeError, ValueError):
            return
        pan, tilt = self._acknowledged_angles
        if axis == "S1":
            pan = angle
        elif axis == "S3":
            tilt = angle
        else:
            return
        self._acknowledged_angles = (pan, tilt)

    def set_servo_target(self, target: ServoTarget) -> None:
        pan = int(round(target.angle_deg))
        _, tilt = self._desired_angles
        self._send_angles(pan, tilt)

    def set_pan_tilt_target(self, target: PanTiltTarget) -> None:
        self._send_angles(
            int(round(target.pan_angle_deg)),
            int(round(target.tilt_angle_deg)),
        )

    def neutral(self, *, command_id: int, angle_deg: float) -> None:
        del command_id
        angle = int(round(angle_deg))
        self._desired_angles = (angle, angle)
        self._write(self._encode_angles(angle, angle, force_both=True))

    def set_mode(self, *, command_id: int, mode: OperatingMode) -> None:
        del command_id, mode

    def close(self) -> None:
        self._closed = True
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _send_angles(self, pan: int, tilt: int) -> None:
        pan = max(0, min(180, pan))
        tilt = max(0, min(180, tilt))
        self._desired_angles = (pan, tilt)
        payload = self._encode_angles(pan, tilt)
        if payload:
            self._write(payload)

    def _encode_angles(self, pan: int, tilt: int, *, force_both: bool = False) -> bytes:
        previous = self._sent_angles
        commands: list[str] = []
        if force_both or previous is None or pan != previous[0]:
            commands.append(f"S1:{pan}")
        if force_both or previous is None or tilt != previous[1]:
            commands.append(f"S3:{tilt}")
        if not commands:
            return b""
        self._sent_angles = (pan, tilt)
        return ("\n".join(commands) + "\n").encode("ascii")

    def _write(self, value: bytes) -> None:
        connection, reconnected = self._ensure_connection()
        if connection is None:
            return
        if reconnected:
            self._replay_desired_state(connection)
            return
        try:
            connection.write(value)
            self._last_write_at_s = time.monotonic()
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
        pan, tilt = self._desired_angles
        self._sent_angles = (pan, tilt)
        try:
            connection.write(self._encode_replay(pan, tilt))
            self._last_write_at_s = time.monotonic()
        except (OSError, serial.SerialException) as error:
            self._disconnect(error)
            return False
        return True

    @staticmethod
    def _encode_replay(pan: int, tilt: int) -> bytes:
        return f"S1:{pan}\nS3:{tilt}\n".encode("ascii")

    def _disconnect(self, error: Exception) -> None:
        self.last_error = str(error)
        connection = self._serial
        self._serial = None
        self._sent_angles = None
        self._acknowledged_angles = (None, None)
        self._receive_buffer.clear()
        self._next_reconnect_at_s = time.monotonic() + self._reconnect_interval_s
        if connection is not None:
            try:
                connection.close()
            except (OSError, serial.SerialException):
                pass
