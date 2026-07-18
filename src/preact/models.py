"""Shared immutable data models for PREACT."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """A ball detection in image coordinates."""

    timestamp_s: float
    x_px: float
    y_px: float
    radius_px: float
    area_px2: float
    quality: float
    bbox_x_px: float | None = None
    bbox_y_px: float | None = None
    bbox_width_px: float | None = None
    bbox_height_px: float | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """A calibrated ball observation in world coordinates.

    The world frame uses x along the floor and y as height above the floor.
    """

    timestamp_s: float
    x_m: float
    y_m: float
    quality: float = 1.0


@dataclass(frozen=True, slots=True)
class TrajectoryFit:
    """Estimated motion state at the latest observation."""

    reference_time_s: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    ay_mps2: float
    sample_count: int
    observed_span_s: float
    rmse_m: float
    inlier_fraction: float
    mean_quality: float

    def position_at(self, seconds_after_reference: float) -> tuple[float, float]:
        """Return the fitted position after the reference time."""

        dt = float(seconds_after_reference)
        x_m = self.x_m + self.vx_mps * dt
        y_m = self.y_m + self.vy_mps * dt + 0.5 * self.ay_mps2 * dt * dt
        return x_m, y_m


@dataclass(frozen=True, slots=True)
class Prediction:
    """A future floor-intersection prediction."""

    valid: bool
    reason: str
    generated_at_s: float
    landing_x_m: float | None
    time_to_impact_s: float | None
    confidence: float
    fit: TrajectoryFit | None

    @classmethod
    def invalid(
        cls,
        reason: str,
        generated_at_s: float,
        fit: TrajectoryFit | None = None,
    ) -> Prediction:
        return cls(
            valid=False,
            reason=reason,
            generated_at_s=generated_at_s,
            landing_x_m=None,
            time_to_impact_s=None,
            confidence=0.0,
            fit=fit,
        )


@dataclass(frozen=True, slots=True)
class ToFSample:
    """A timestamped ToF range sample received from Genesis."""

    received_at_s: float
    device_time_ms: int
    distance_mm: int
    valid: bool
    range_status: int = 0


@dataclass(frozen=True, slots=True)
class ToFTrigger:
    """A near-object transition detected against the ToF background baseline."""

    triggered_at_s: float
    distance_mm: int
    baseline_mm: float
    drop_mm: float


@dataclass(frozen=True, slots=True)
class ServoTarget:
    """A physical pointer command derived from a valid landing prediction."""

    command_id: int
    angle_deg: float
    target_x_m: float
    confidence: float
    time_to_impact_ms: int
    target_y_m: float | None = None
    purpose: str = "landing"
