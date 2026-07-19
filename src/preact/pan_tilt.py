"""Smooth pixel-to-servo mapping for the S1 pan and S3 tilt axes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preact.models import Detection, PanTiltTarget

CALIBRATION_POINT_FRACTIONS: dict[str, tuple[float, float]] = {
    "top_left": (0.1, 0.1),
    "top_right": (0.9, 0.1),
    "center": (0.5, 0.5),
    "bottom_left": (0.1, 0.9),
    "bottom_right": (0.9, 0.9),
}


class PanTiltImageCalibration:
    """Five-point image-to-servo calibration for a displaced pointer mount."""

    def __init__(self, points: dict[str, tuple[float, float]] | None = None) -> None:
        self.points = dict(points or {})
        self._coefficients: tuple[np.ndarray, np.ndarray] | None = None
        self._refit()

    @property
    def complete(self) -> bool:
        return all(name in self.points for name in CALIBRATION_POINT_FRACTIONS)

    def set_point(self, name: str, *, pan_deg: float, tilt_deg: float) -> None:
        if name not in CALIBRATION_POINT_FRACTIONS:
            raise ValueError(f"unknown calibration point {name!r}")
        self.points[name] = (float(pan_deg), float(tilt_deg))
        self._refit()

    def angles_for(self, x_fraction: float, y_fraction: float) -> tuple[float, float] | None:
        if self._coefficients is None:
            return None
        basis = _calibration_basis(x_fraction, y_fraction)
        pan_coefficients, tilt_coefficients = self._coefficients
        return (
            float(basis @ pan_coefficients),
            float(basis @ tilt_coefficients),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "points": {
                name: {"pan_deg": angles[0], "tilt_deg": angles[1]}
                for name, angles in self.points.items()
            },
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> PanTiltImageCalibration:
        source = Path(path)
        if not source.exists():
            return cls()
        value = json.loads(source.read_text(encoding="utf-8"))
        points = {
            name: (float(item["pan_deg"]), float(item["tilt_deg"]))
            for name, item in value.get("points", {}).items()
            if name in CALIBRATION_POINT_FRACTIONS
        }
        return cls(points)

    def _refit(self) -> None:
        if not self.complete:
            self._coefficients = None
            return
        names = tuple(CALIBRATION_POINT_FRACTIONS)
        matrix = np.stack(
            [
                _calibration_basis(*CALIBRATION_POINT_FRACTIONS[name])
                for name in names
            ]
        )
        pan_values = np.array([self.points[name][0] for name in names])
        tilt_values = np.array([self.points[name][1] for name in names])
        self._coefficients = (
            np.linalg.solve(matrix, pan_values),
            np.linalg.solve(matrix, tilt_values),
        )


def _calibration_basis(x_fraction: float, y_fraction: float) -> np.ndarray:
    lower = 0.1
    upper = 0.9
    midpoint = 0.5
    denominator = (
        (midpoint - lower)
        * (midpoint - upper)
        * (midpoint - lower)
        * (midpoint - upper)
    )
    center_weight = (
        (x_fraction - lower)
        * (x_fraction - upper)
        * (y_fraction - lower)
        * (y_fraction - upper)
        / denominator
    )
    return np.array(
        [1.0, x_fraction, y_fraction, x_fraction * y_fraction, center_weight],
        dtype=float,
    )


@dataclass(frozen=True, slots=True)
class PanTiltCalibration:
    minimum_angle_deg: float = 10.0
    maximum_angle_deg: float = 170.0
    neutral_angle_deg: float = 90.0
    minimum_quality: float = 0.35
    deadband_deg: float = 1.0
    smoothing_alpha: float = 0.55
    maximum_speed_deg_s: float = 200.0
    pan_response_multiplier: float = 1.7
    invert_pan: bool = False
    invert_tilt: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_angle_deg < self.maximum_angle_deg <= 180.0:
            raise ValueError("pan/tilt angle range must be within 0..180 degrees")
        if not self.minimum_angle_deg <= self.neutral_angle_deg <= self.maximum_angle_deg:
            raise ValueError("neutral angle must be inside the pan/tilt angle range")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be in the range 0..1")
        if self.deadband_deg < 0.0:
            raise ValueError("deadband_deg must not be negative")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in the range (0, 1]")
        if self.maximum_speed_deg_s <= 0.0:
            raise ValueError("maximum_speed_deg_s must be positive")
        if self.pan_response_multiplier < 1.0:
            raise ValueError("pan_response_multiplier must be at least 1.0")


class PanTiltPlanner:
    """Convert target centroids to smooth independent pan and tilt commands."""

    def __init__(
        self,
        calibration: PanTiltCalibration | None = None,
        *,
        image_calibration: PanTiltImageCalibration | None = None,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.calibration = calibration or PanTiltCalibration()
        self.image_calibration = image_calibration or PanTiltImageCalibration()
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self._next_command_id = 1
        self._pan_deg = self.calibration.neutral_angle_deg
        self._tilt_deg = self.calibration.neutral_angle_deg
        self._last_timestamp_s: float | None = None

    def reset(self) -> None:
        self._pan_deg = self.calibration.neutral_angle_deg
        self._tilt_deg = self.calibration.neutral_angle_deg
        self._last_timestamp_s = None

    @property
    def calibration_points(self) -> dict[str, tuple[float, float]]:
        return dict(self.image_calibration.points)

    def calibration_seed_angles(self, name: str) -> tuple[float, float]:
        if name not in CALIBRATION_POINT_FRACTIONS:
            raise ValueError(f"unknown calibration point {name!r}")
        existing = self.image_calibration.points.get(name)
        if existing is not None:
            return existing
        x_fraction, y_fraction = CALIBRATION_POINT_FRACTIONS[name]
        return self._default_angles(x_fraction, y_fraction)

    def save_calibration_point(self, name: str, *, pan_deg: float, tilt_deg: float) -> None:
        self.image_calibration.set_point(name, pan_deg=pan_deg, tilt_deg=tilt_deg)
        if self.calibration_path is not None:
            self.image_calibration.save(self.calibration_path)

    def target_for_detection(
        self,
        detection: Detection,
        *,
        frame_width_px: int,
        frame_height_px: int,
        timestamp_s: float,
    ) -> PanTiltTarget | None:
        return self.target_for_point(
            x_px=detection.x_px,
            y_px=detection.y_px,
            confidence=detection.quality,
            frame_width_px=frame_width_px,
            frame_height_px=frame_height_px,
            timestamp_s=timestamp_s,
        )

    def target_for_point(
        self,
        *,
        x_px: float,
        y_px: float,
        confidence: float,
        frame_width_px: int,
        frame_height_px: int,
        timestamp_s: float,
        immediate: bool = False,
    ) -> PanTiltTarget | None:
        """Point at an arbitrary image position, including a predicted landing."""

        if confidence < self.calibration.minimum_quality:
            return None
        if frame_width_px <= 1 or frame_height_px <= 1:
            return None

        x_fraction = float(np.clip(x_px / (frame_width_px - 1), 0.0, 1.0))
        y_fraction = float(np.clip(y_px / (frame_height_px - 1), 0.0, 1.0))
        calibrated = self.image_calibration.angles_for(x_fraction, y_fraction)
        if calibrated is None:
            desired_pan, desired_tilt = self._default_angles(x_fraction, y_fraction)
        else:
            desired_pan, desired_tilt = calibrated

        elapsed_s = (
            1.0 / 30.0
            if self._last_timestamp_s is None
            else max(1.0 / 120.0, timestamp_s - self._last_timestamp_s)
        )
        if immediate:
            self._pan_deg = float(
                np.clip(
                    desired_pan,
                    self.calibration.minimum_angle_deg,
                    self.calibration.maximum_angle_deg,
                )
            )
            self._tilt_deg = float(
                np.clip(
                    desired_tilt,
                    self.calibration.minimum_angle_deg,
                    self.calibration.maximum_angle_deg,
                )
            )
        else:
            self._pan_deg = self._smooth_axis(
                self._pan_deg,
                desired_pan,
                elapsed_s,
                response_multiplier=self.calibration.pan_response_multiplier,
            )
            self._tilt_deg = self._smooth_axis(
                self._tilt_deg,
                desired_tilt,
                elapsed_s,
            )
        self._last_timestamp_s = timestamp_s

        target = PanTiltTarget(
            command_id=self._next_command_id,
            pan_angle_deg=self._pan_deg,
            tilt_angle_deg=self._tilt_deg,
            x_px=x_px,
            y_px=y_px,
            confidence=confidence,
        )
        self._next_command_id += 1
        return target

    def _map_fraction(self, fraction: float, *, invert: bool) -> float:
        if invert:
            fraction = 1.0 - fraction
        span = self.calibration.maximum_angle_deg - self.calibration.minimum_angle_deg
        return self.calibration.minimum_angle_deg + fraction * span

    def _default_angles(self, x_fraction: float, y_fraction: float) -> tuple[float, float]:
        return (
            self._map_fraction(x_fraction, invert=self.calibration.invert_pan),
            self._map_fraction(y_fraction, invert=self.calibration.invert_tilt),
        )

    def _smooth_axis(
        self,
        current: float,
        desired: float,
        elapsed_s: float,
        *,
        response_multiplier: float = 1.0,
    ) -> float:
        difference = desired - current
        effective_deadband = self.calibration.deadband_deg / response_multiplier
        if abs(difference) <= effective_deadband:
            return current
        effective_alpha = 1.0 - (
            (1.0 - self.calibration.smoothing_alpha) ** response_multiplier
        )
        filtered = current + difference * effective_alpha
        maximum_step = (
            self.calibration.maximum_speed_deg_s
            * response_multiplier
            * elapsed_s
        )
        step = float(np.clip(filtered - current, -maximum_step, maximum_step))
        return float(
            np.clip(
                current + step,
                self.calibration.minimum_angle_deg,
                self.calibration.maximum_angle_deg,
            )
        )
