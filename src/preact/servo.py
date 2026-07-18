"""Map landing positions to a one-axis positional servo."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Literal

import numpy as np

from preact.models import Prediction, ServoTarget

ServoMappingMode = Literal["linear", "planar-pointing"]


@dataclass(frozen=True, slots=True)
class ServoCalibration:
    landing_min_x_m: float = 0.0
    landing_max_x_m: float = 3.0
    angle_at_min_deg: float = 20.0
    angle_at_max_deg: float = 160.0
    neutral_angle_deg: float = 90.0
    minimum_confidence: float = 0.45
    minimum_tracking_quality: float = 0.35
    mapping_mode: ServoMappingMode = "linear"
    pointer_base_x_m: float = -0.4
    pointer_base_y_m: float = 0.5
    pointer_zero_angle_deg: float = 90.0
    pointer_direction: int = 1

    def __post_init__(self) -> None:
        if self.landing_max_x_m <= self.landing_min_x_m:
            raise ValueError("landing_max_x_m must be greater than landing_min_x_m")
        for angle in (
            self.angle_at_min_deg,
            self.angle_at_max_deg,
            self.neutral_angle_deg,
        ):
            if not 0.0 <= angle <= 180.0:
                raise ValueError("servo angles must be in the range 0..180 degrees")
        if not 0.0 <= self.pointer_zero_angle_deg <= 180.0:
            raise ValueError("pointer_zero_angle_deg must be in the range 0..180")
        if self.mapping_mode not in ("linear", "planar-pointing"):
            raise ValueError("unsupported servo mapping mode")
        if self.pointer_direction not in (-1, 1):
            raise ValueError("pointer_direction must be -1 or 1")
        if not 0.0 <= self.minimum_tracking_quality <= 1.0:
            raise ValueError("minimum_tracking_quality must be in the range 0..1")

    def angle_for_landing(self, landing_x_m: float) -> float:
        return self.angle_for_point(landing_x_m, 0.0)

    def angle_for_point(self, x_m: float, y_m: float) -> float:
        if self.mapping_mode == "planar-pointing":
            bearing_deg = degrees(
                atan2(
                    y_m - self.pointer_base_y_m,
                    x_m - self.pointer_base_x_m,
                )
            )
            angle = (
                self.pointer_zero_angle_deg
                + self.pointer_direction * bearing_deg
            )
            lower = min(self.angle_at_min_deg, self.angle_at_max_deg)
            upper = max(self.angle_at_min_deg, self.angle_at_max_deg)
            return float(np.clip(angle, lower, upper))

        fraction = (x_m - self.landing_min_x_m) / (
            self.landing_max_x_m - self.landing_min_x_m
        )
        fraction = float(np.clip(fraction, 0.0, 1.0))
        angle = self.angle_at_min_deg + fraction * (
            self.angle_at_max_deg - self.angle_at_min_deg
        )
        return float(np.clip(angle, 0.0, 180.0))


class ServoPlanner:
    def __init__(self, calibration: ServoCalibration | None = None) -> None:
        self.calibration = calibration or ServoCalibration()
        self._next_command_id = 1

    def target_for_prediction(
        self,
        prediction: Prediction,
        *,
        tof_triggered: bool,
    ) -> ServoTarget | None:
        if (
            not tof_triggered
            or not prediction.valid
            or prediction.landing_x_m is None
            or prediction.time_to_impact_s is None
            or prediction.confidence < self.calibration.minimum_confidence
        ):
            return None

        target = ServoTarget(
            command_id=self._next_command_id,
            angle_deg=self.calibration.angle_for_landing(prediction.landing_x_m),
            target_x_m=prediction.landing_x_m,
            confidence=prediction.confidence,
            time_to_impact_ms=max(0, round(prediction.time_to_impact_s * 1000.0)),
            target_y_m=0.0,
            purpose="landing",
        )
        self._next_command_id += 1
        return target

    def target_for_tracking(
        self,
        *,
        x_m: float,
        y_m: float,
        quality: float,
    ) -> ServoTarget | None:
        if quality < self.calibration.minimum_tracking_quality:
            return None
        target = ServoTarget(
            command_id=self._next_command_id,
            angle_deg=self.calibration.angle_for_point(x_m, y_m),
            target_x_m=x_m,
            target_y_m=y_m,
            confidence=quality,
            time_to_impact_ms=0,
            purpose="tracking",
        )
        self._next_command_id += 1
        return target
