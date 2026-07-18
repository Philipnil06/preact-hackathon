"""Operating modes shared by voice, runtime, engine, and firmware protocol."""

from __future__ import annotations

from enum import StrEnum


class OperatingMode(StrEnum):
    IDLE = "idle"
    TRACKING = "tracking"
    LANDING = "landing"

    @property
    def active(self) -> bool:
        return self is not OperatingMode.IDLE
