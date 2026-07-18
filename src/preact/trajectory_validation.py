"""Deterministic Monte Carlo validation for the ballistic predictor."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np

from preact.models import Observation
from preact.trajectory import TrajectoryPredictor


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    scenario_count: int = 300
    random_seed: int = 20_260_717
    maximum_landing_x_m: float = 3.1
    minimum_confidence: float = 0.45
    maximum_accepted_error_m: float = 0.20
    minimum_success_rate: float = 0.80
    outlier_probability: float = 0.25

    def __post_init__(self) -> None:
        if self.scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        if self.maximum_landing_x_m <= 0:
            raise ValueError("maximum_landing_x_m must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in the range 0..1")
        if self.maximum_accepted_error_m <= 0:
            raise ValueError("maximum_accepted_error_m must be positive")
        if not 0.0 <= self.minimum_success_rate <= 1.0:
            raise ValueError("minimum_success_rate must be in the range 0..1")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must be in the range 0..1")


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    scenarios: int
    valid_predictions: int
    actuated_predictions: int
    successful_predictions: int
    success_rate: float
    median_error_m: float
    p90_error_m: float
    p95_error_m: float
    maximum_error_m: float
    outlier_cases: int
    accepted: bool


def run_validation(
    config: ValidationConfig | None = None,
) -> ValidationSummary:
    config = config or ValidationConfig()
    rng = np.random.default_rng(config.random_seed)
    predictor = TrajectoryPredictor()
    gravity_mps2 = predictor.config.gravity_mps2
    outlier_count = 0
    errors: list[float] = []
    successes = 0

    evaluated = 0
    valid_predictions = 0
    while evaluated < config.scenario_count:
        launch_x_m = rng.uniform(0.10, 0.45)
        launch_y_m = rng.uniform(0.75, 1.55)
        vx_mps = rng.uniform(1.1, 3.2)
        vy_mps = rng.uniform(2.5, 5.8)
        impact_time_s = (
            vy_mps
            + sqrt(vy_mps * vy_mps + 2.0 * gravity_mps2 * launch_y_m)
        ) / gravity_mps2
        expected_landing_x_m = launch_x_m + vx_mps * impact_time_s
        if expected_landing_x_m > config.maximum_landing_x_m:
            continue

        observed_span_s = rng.uniform(
            0.22,
            min(0.46, impact_time_s - 0.12),
        )
        sample_count = int(rng.integers(8, 16))
        timestamps = np.linspace(0.0, observed_span_s, sample_count)
        noise_m = rng.uniform(0.003, 0.015)
        outlier_index = (
            int(rng.integers(1, sample_count - 1))
            if rng.random() < config.outlier_probability
            else -1
        )
        if outlier_index >= 0:
            outlier_count += 1

        observations: list[Observation] = []
        for index, timestamp_s in enumerate(timestamps):
            x_m = (
                launch_x_m
                + vx_mps * timestamp_s
                + rng.normal(0.0, noise_m)
            )
            y_m = (
                launch_y_m
                + vy_mps * timestamp_s
                - 0.5 * gravity_mps2 * timestamp_s**2
                + rng.normal(0.0, noise_m)
            )
            quality = rng.uniform(0.75, 1.0)
            if index == outlier_index:
                x_m += rng.normal(0.0, 0.20)
                y_m += rng.normal(0.0, 0.20)
                quality = rng.uniform(0.35, 0.65)
            observations.append(
                Observation(
                    timestamp_s=float(timestamp_s),
                    x_m=float(x_m),
                    y_m=float(y_m),
                    quality=float(quality),
                )
            )

        prediction = predictor.predict(observations)
        evaluated += 1
        if not prediction.valid or prediction.landing_x_m is None:
            continue
        valid_predictions += 1
        if prediction.confidence < config.minimum_confidence:
            continue
        error_m = abs(prediction.landing_x_m - expected_landing_x_m)
        errors.append(error_m)
        if error_m <= config.maximum_accepted_error_m:
            successes += 1

    error_values = np.asarray(errors, dtype=np.float64)
    if len(error_values) == 0:
        median = p90 = p95 = maximum = float("inf")
    else:
        median = float(np.median(error_values))
        p90 = float(np.percentile(error_values, 90))
        p95 = float(np.percentile(error_values, 95))
        maximum = float(np.max(error_values))
    success_rate = successes / config.scenario_count
    return ValidationSummary(
        scenarios=config.scenario_count,
        valid_predictions=valid_predictions,
        actuated_predictions=len(errors),
        successful_predictions=successes,
        success_rate=success_rate,
        median_error_m=median,
        p90_error_m=p90,
        p95_error_m=p95,
        maximum_error_m=maximum,
        outlier_cases=outlier_count,
        accepted=success_rate >= config.minimum_success_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PREACT trajectory accuracy over seeded noisy throws"
    )
    parser.add_argument("--scenarios", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--maximum-error-m", type=float, default=0.20)
    parser.add_argument("--minimum-success-rate", type=float, default=0.80)
    arguments = parser.parse_args()
    summary = run_validation(
        ValidationConfig(
            scenario_count=arguments.scenarios,
            random_seed=arguments.seed,
            maximum_accepted_error_m=arguments.maximum_error_m,
            minimum_success_rate=arguments.minimum_success_rate,
        )
    )
    print(json.dumps(asdict(summary), indent=2))
    if not summary.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
