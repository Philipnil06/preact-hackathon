from __future__ import annotations

import numpy as np

from preact.models import Observation
from preact.trajectory import TrajectoryConfig, TrajectoryPredictor


def _synthetic_observations(
    *,
    rng: np.random.Generator,
    sample_count: int = 16,
    noise_m: float = 0.006,
    include_outlier: bool = False,
) -> tuple[list[Observation], float]:
    gravity = 9.80665
    launch_x = 0.35
    launch_y = 1.15
    vx = 2.2
    vy = 4.8
    times = np.linspace(0.0, 0.46, sample_count)
    observations = []
    for index, timestamp in enumerate(times):
        x_m = launch_x + vx * timestamp + rng.normal(0.0, noise_m)
        y_m = launch_y + vy * timestamp - 0.5 * gravity * timestamp**2
        y_m += rng.normal(0.0, noise_m)
        if include_outlier and index == sample_count // 2:
            x_m += 0.32
            y_m -= 0.24
        observations.append(
            Observation(
                timestamp_s=float(timestamp),
                x_m=float(x_m),
                y_m=float(y_m),
                quality=1.0,
            )
        )

    impact_time = (vy + np.sqrt(vy**2 + 2 * gravity * launch_y)) / gravity
    expected_landing_x = launch_x + vx * impact_time
    return observations, float(expected_landing_x)


def test_predicts_synthetic_landing_with_noise() -> None:
    observations, expected_landing = _synthetic_observations(
        rng=np.random.default_rng(4),
    )
    prediction = TrajectoryPredictor().predict(observations)

    assert prediction.valid
    assert prediction.landing_x_m is not None
    assert abs(prediction.landing_x_m - expected_landing) < 0.035
    assert prediction.time_to_impact_s is not None
    assert prediction.time_to_impact_s > 0.3
    assert prediction.confidence > 0.65


def test_robust_fit_rejects_single_large_outlier() -> None:
    observations, expected_landing = _synthetic_observations(
        rng=np.random.default_rng(9),
        include_outlier=True,
    )
    prediction = TrajectoryPredictor().predict(observations)

    assert prediction.valid
    assert prediction.landing_x_m is not None
    assert abs(prediction.landing_x_m - expected_landing) < 0.12
    assert prediction.fit is not None
    assert prediction.fit.inlier_fraction < 1.0


def test_rejects_too_few_observations() -> None:
    predictor = TrajectoryPredictor(TrajectoryConfig(min_samples=6))
    observations = [
        Observation(timestamp_s=index * 0.03, x_m=0.1 * index, y_m=1.0)
        for index in range(5)
    ]

    prediction = predictor.predict(observations)

    assert not prediction.valid
    assert "at least 6" in prediction.reason


def test_rejects_too_short_observation_window() -> None:
    predictor = TrajectoryPredictor(TrajectoryConfig(min_observed_span_s=0.12))
    observations = [
        Observation(timestamp_s=index * 0.01, x_m=0.02 * index, y_m=1.0)
        for index in range(8)
    ]

    prediction = predictor.predict(observations)

    assert not prediction.valid
    assert "shorter" in prediction.reason
