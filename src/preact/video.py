"""Video-source abstractions for synthetic, replay, and live camera input."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from preact.models import ToFSample
from preact.synthetic import SyntheticFrame, SyntheticThrow


@dataclass(frozen=True, slots=True)
class FramePacket:
    timestamp_s: float
    frame_bgr: np.ndarray
    tof_sample: ToFSample | None = None


class FrameSource(Protocol):
    @property
    def frames_per_second(self) -> float: ...

    def read(self) -> FramePacket | None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class SyntheticFrameSource:
    def __init__(
        self,
        scene: SyntheticThrow | None = None,
        *,
        loop: bool = False,
    ) -> None:
        self.scene = scene or SyntheticThrow()
        self._frames: list[SyntheticFrame] = self.scene.frames()
        self._loop = loop
        self._index = 0
        self._time_offset_s = 0.0

    @property
    def frames_per_second(self) -> float:
        return self.scene.config.frames_per_second

    def read(self) -> FramePacket | None:
        if self._index >= len(self._frames):
            if not self._loop:
                return None
            self._index = 0
            self._time_offset_s += (
                self._frames[-1].timestamp_s
                + 1.0 / self.scene.config.frames_per_second
            )
        item = self._frames[self._index]
        self._index += 1
        timestamp_s = item.timestamp_s + self._time_offset_s
        tof_sample = ToFSample(
            received_at_s=timestamp_s,
            device_time_ms=(
                item.tof_sample.device_time_ms
                + round(self._time_offset_s * 1000.0)
            ),
            distance_mm=item.tof_sample.distance_mm,
            valid=item.tof_sample.valid,
            range_status=item.tof_sample.range_status,
        )
        return FramePacket(
            timestamp_s=timestamp_s,
            frame_bgr=item.frame_bgr.copy(),
            tof_sample=tof_sample,
        )

    def reset(self) -> None:
        self._index = 0
        self._time_offset_s = 0.0

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ScriptedToFConfig:
    trigger_time_s: float = 0.40
    pulse_duration_s: float = 0.09
    baseline_distance_mm: int = 1100
    triggered_distance_mm: int = 700

    def __post_init__(self) -> None:
        if self.trigger_time_s <= 0:
            raise ValueError("trigger_time_s must leave time for baseline samples")
        if self.pulse_duration_s <= 0:
            raise ValueError("pulse_duration_s must be positive")
        if self.triggered_distance_mm >= self.baseline_distance_mm:
            raise ValueError("triggered ToF distance must be below the baseline")


class ScriptedToFFrameSource:
    """Inject deterministic ToF samples into a replay file for offline testing."""

    def __init__(
        self,
        source: FrameSource,
        config: ScriptedToFConfig | None = None,
    ) -> None:
        self.source = source
        self.config = config or ScriptedToFConfig()

    @property
    def frames_per_second(self) -> float:
        return self.source.frames_per_second

    def read(self) -> FramePacket | None:
        packet = self.source.read()
        if packet is None:
            return None
        config = self.config
        triggered = (
            config.trigger_time_s
            <= packet.timestamp_s
            < config.trigger_time_s + config.pulse_duration_s
        )
        distance_mm = (
            config.triggered_distance_mm
            if triggered
            else config.baseline_distance_mm
        )
        return FramePacket(
            timestamp_s=packet.timestamp_s,
            frame_bgr=packet.frame_bgr,
            tof_sample=ToFSample(
                received_at_s=packet.timestamp_s,
                device_time_ms=round(packet.timestamp_s * 1000.0),
                distance_mm=distance_mm,
                valid=True,
            ),
        )

    def reset(self) -> None:
        self.source.reset()

    def close(self) -> None:
        self.source.close()


class OpenCVFrameSource:
    """Open a replay file or camera only when explicitly constructed."""

    def __init__(
        self,
        source: str | Path | int,
        *,
        requested_width_px: int | None = None,
        requested_height_px: int | None = None,
        requested_fps: float | None = None,
    ) -> None:
        self.source = source
        self._capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")
        if isinstance(source, int):
            if requested_width_px is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width_px)
            if requested_height_px is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height_px)
            if requested_fps is not None:
                self._capture.set(cv2.CAP_PROP_FPS, requested_fps)
        nominal_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._fps = nominal_fps if nominal_fps > 1.0 else 30.0
        self._started_at_s = time.monotonic()
        self._frame_index = 0

    @property
    def frames_per_second(self) -> float:
        return self._fps

    def read(self) -> FramePacket | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        if isinstance(self.source, int):
            timestamp = time.monotonic()
        else:
            timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else (
                self._frame_index / self._fps
            )
        self._frame_index += 1
        return FramePacket(timestamp_s=timestamp, frame_bgr=frame)

    def reset(self) -> None:
        if isinstance(self.source, int):
            return
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._started_at_s = time.monotonic()
        self._frame_index = 0

    def close(self) -> None:
        self._capture.release()
