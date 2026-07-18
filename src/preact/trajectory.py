"""Robust physics-first trajectory estimation and floor intersection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from preact.models import Observation, Prediction, TrajectoryFit


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    gravity_mps2: float = 9.80665
    floor_y_m: float = 0.0
    min_samples: int = 6
    min_observed_span_s: float = 0.12
    max_prediction_horizon_s: float = 3.0
    min_prediction_horizon_s: float = 0.04
    huber_delta_m: float = 0.025
    robust_iterations: int = 8
    residual_scale_m: float = 0.04
    ideal_sample_count: int = 14
    ideal_observed_span_s: float = 0.35
    max_abs_horizontal_speed_mps: float = 15.0
    max_abs_vertical_speed_mps: float = 15.0


class TrajectoryPredictor:
    """Fit a ballistic trajectory and predict its next floor intersection.

    Gravity is fixed instead of freely estimated. This stabilises early predictions
    from a short, noisy webcam track once the image plane is calibrated to metres.
    """

    def __init__(self, config: TrajectoryConfig | None = None) -> None:
        self.config = config or TrajectoryConfig()

    def fit(self, observations: Sequence[Observation]) -> TrajectoryFit:
        config = self.config
        if len(observations) < config.min_samples:
            raise ValueError(f"at least {config.min_samples} observations are required")

        ordered = sorted(observations, key=lambda item: item.timestamp_s)
        timestamps = np.asarray([item.timestamp_s for item in ordered], dtype=np.float64)
        x_values = np.asarray([item.x_m for item in ordered], dtype=np.float64)
        y_values = np.asarray([item.y_m for item in ordered], dtype=np.float64)
        quality = np.clip(
            np.asarray([item.quality for item in ordered], dtype=np.float64),
            0.05,
            1.0,
        )
        if not (
            np.isfinite(timestamps).all()
            and np.isfinite(x_values).all()
            and np.isfinite(y_values).all()
        ):
            raise ValueError("observations must be finite")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("observation timestamps must be unique")

        reference_time = float(timestamps[-1])
        tau = timestamps - reference_time
        observed_span = float(timestamps[-1] - timestamps[0])
        if observed_span < config.min_observed_span_s:
            raise ValueError(
                f"observation span {observed_span:.3f}s is shorter than "
                f"{config.min_observed_span_s:.3f}s"
            )

        x_design = np.column_stack((np.ones_like(tau), tau))
        x_coefficients, x_residuals, x_weights = _robust_weighted_fit(
            x_design,
            x_values,
            quality,
            delta=config.huber_delta_m,
            iterations=config.robust_iterations,
        )

        acceleration = -abs(config.gravity_mps2)
        gravity_term = 0.5 * acceleration * np.square(tau)
        y_design = np.column_stack((np.ones_like(tau), tau))
        y_coefficients, y_adjusted_residuals, y_weights = _robust_weighted_fit(
            y_design,
            y_values - gravity_term,
            quality,
            delta=config.huber_delta_m,
            iterations=config.robust_iterations,
        )
        y_residuals = y_adjusted_residuals

        combined_residual = np.sqrt(np.square(x_residuals) + np.square(y_residuals))
        rmse = float(np.sqrt(np.average(np.square(combined_residual), weights=quality)))
        combined_weights = np.minimum(x_weights, y_weights)
        inlier_fraction = float(np.mean(combined_weights >= 0.75 * quality))

        return TrajectoryFit(
            reference_time_s=reference_time,
            x_m=float(x_coefficients[0]),
            y_m=float(y_coefficients[0]),
            vx_mps=float(x_coefficients[1]),
            vy_mps=float(y_coefficients[1]),
            ay_mps2=acceleration,
            sample_count=len(ordered),
            observed_span_s=observed_span,
            rmse_m=rmse,
            inlier_fraction=inlier_fraction,
            mean_quality=float(np.mean(quality)),
        )

    def predict(self, observations: Sequence[Observation]) -> Prediction:
        generated_at = max((item.timestamp_s for item in observations), default=0.0)
        try:
            fit = self.fit(observations)
        except ValueError as error:
            return Prediction.invalid(str(error), generated_at)

        config = self.config
        if fit.y_m < config.floor_y_m - 0.05:
            return Prediction.invalid(
                "latest observation is below the floor plane",
                generated_at,
                fit,
            )
        if abs(fit.vx_mps) > config.max_abs_horizontal_speed_mps:
            return Prediction.invalid(
                "horizontal speed is outside the demo envelope",
                generated_at,
                fit,
            )
        if abs(fit.vy_mps) > config.max_abs_vertical_speed_mps:
            return Prediction.invalid(
                "vertical speed is outside the demo envelope",
                generated_at,
                fit,
            )

        impact_time = _future_floor_intersection(
            y_m=fit.y_m,
            vy_mps=fit.vy_mps,
            ay_mps2=fit.ay_mps2,
            floor_y_m=config.floor_y_m,
        )
        if impact_time is None:
            return Prediction.invalid(
                "trajectory does not intersect the floor in the future",
                generated_at,
                fit,
            )
        if impact_time < config.min_prediction_horizon_s:
            return Prediction.invalid(
                "impact is too close to issue a useful prediction",
                generated_at,
                fit,
            )
        if impact_time > config.max_prediction_horizon_s:
            return Prediction.invalid("impact is outside the prediction horizon", generated_at, fit)

        landing_x = fit.x_m + fit.vx_mps * impact_time
        confidence = _confidence(fit, impact_time, config)
        return Prediction(
            valid=True,
            reason="ok",
            generated_at_s=generated_at,
            landing_x_m=float(landing_x),
            time_to_impact_s=float(impact_time),
            confidence=confidence,
            fit=fit,
        )


def _robust_weighted_fit(
    design: np.ndarray,
    values: np.ndarray,
    base_weights: np.ndarray,
    *,
    delta: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(max(1, iterations)):
        root_weights = np.sqrt(np.clip(weights, 1e-6, None))
        weighted_design = design * root_weights[:, None]
        weighted_values = values * root_weights
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        residuals = values - design @ coefficients
        absolute = np.abs(residuals)
        robust = np.ones_like(absolute)
        large = absolute > delta
        robust[large] = delta / absolute[large]
        weights = base_weights * robust
    residuals = values - design @ coefficients
    return coefficients, residuals, weights


def _future_floor_intersection(
    *,
    y_m: float,
    vy_mps: float,
    ay_mps2: float,
    floor_y_m: float,
) -> float | None:
    constant = y_m - floor_y_m
    if abs(ay_mps2) < 1e-9:
        if vy_mps >= 0:
            return None
        result = -constant / vy_mps
        return result if result > 0 else None

    a = 0.5 * ay_mps2
    b = vy_mps
    discriminant = b * b - 4.0 * a * constant
    if discriminant < 0:
        return None
    root = sqrt(discriminant)
    roots = ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a))
    positive = [value for value in roots if value > 1e-9]
    return min(positive) if positive else None


def _confidence(
    fit: TrajectoryFit,
    impact_time_s: float,
    config: TrajectoryConfig,
) -> float:
    sample_score = min(1.0, fit.sample_count / config.ideal_sample_count)
    span_score = min(1.0, fit.observed_span_s / config.ideal_observed_span_s)
    residual_score = exp(-fit.rmse_m / max(config.residual_scale_m, 1e-6))
    horizon_excess = max(0.0, impact_time_s - fit.observed_span_s)
    horizon_score = exp(-0.35 * horizon_excess)

    factors = (
        max(sample_score, 1e-3),
        max(span_score, 1e-3),
        max(residual_score, 1e-3),
        max(fit.inlier_fraction, 1e-3),
        max(fit.mean_quality, 1e-3),
        max(horizon_score, 1e-3),
    )
    geometric_mean = float(np.prod(factors) ** (1.0 / len(factors)))
    return float(np.clip(geometric_mean, 0.0, 1.0))
