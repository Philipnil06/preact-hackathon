"""ToF baseline tracking and passage-trigger detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

from preact.models import ToFSample, ToFTrigger


@dataclass(frozen=True, slots=True)
class ToFGateConfig:
    baseline_window: int = 25
    minimum_baseline_samples: int = 10
    trigger_drop_mm: int = 80
    release_drop_mm: int = 35
    minimum_distance_mm: int = 30
    maximum_distance_mm: int = 4000
    debounce_s: float = 0.20
    recent_trigger_window_s: float = 3.0


class ToFGate:
    """Detect an object crossing a single-zone ToF beam.

    The sensor is aimed at a static background. A ball or hand entering the beam
    creates a shorter range. The transition arms a new camera trajectory.
    """

    def __init__(self, config: ToFGateConfig | None = None) -> None:
        self.config = config or ToFGateConfig()
        self._baseline_samples: deque[int] = deque(maxlen=self.config.baseline_window)
        self._occupied = False
        self._last_trigger: ToFTrigger | None = None
        self._last_sample: ToFSample | None = None

    @property
    def baseline_mm(self) -> float | None:
        if len(self._baseline_samples) < self.config.minimum_baseline_samples:
            return None
        return float(median(self._baseline_samples))

    @property
    def occupied(self) -> bool:
        return self._occupied

    @property
    def last_trigger(self) -> ToFTrigger | None:
        return self._last_trigger

    @property
    def last_sample(self) -> ToFSample | None:
        return self._last_sample

    def update(self, sample: ToFSample) -> ToFTrigger | None:
        self._last_sample = sample
        config = self.config
        if (
            not sample.valid
            or sample.distance_mm < config.minimum_distance_mm
            or sample.distance_mm > config.maximum_distance_mm
        ):
            return None

        baseline = self.baseline_mm
        if baseline is None:
            self._baseline_samples.append(sample.distance_mm)
            return None

        drop = baseline - sample.distance_mm
        if self._occupied:
            if drop <= config.release_drop_mm:
                self._occupied = False
                self._baseline_samples.append(sample.distance_mm)
            return None

        if drop >= config.trigger_drop_mm and self._debounce_elapsed(sample.received_at_s):
            self._occupied = True
            trigger = ToFTrigger(
                triggered_at_s=sample.received_at_s,
                distance_mm=sample.distance_mm,
                baseline_mm=baseline,
                drop_mm=drop,
            )
            self._last_trigger = trigger
            return trigger

        self._baseline_samples.append(sample.distance_mm)
        return None

    def has_recent_trigger(self, now_s: float) -> bool:
        trigger = self._last_trigger
        return bool(
            trigger
            and 0.0 <= now_s - trigger.triggered_at_s <= self.config.recent_trigger_window_s
        )

    def reset(self) -> None:
        self._baseline_samples.clear()
        self._occupied = False
        self._last_trigger = None
        self._last_sample = None

    def _debounce_elapsed(self, now_s: float) -> bool:
        return bool(
            self._last_trigger is None
            or now_s - self._last_trigger.triggered_at_s >= self.config.debounce_s
        )
