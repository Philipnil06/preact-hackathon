from __future__ import annotations

from preact.trajectory_validation import ValidationConfig, run_validation


def test_seeded_monte_carlo_validation_exceeds_accuracy_gate() -> None:
    summary = run_validation(
        ValidationConfig(
            scenario_count=150,
            random_seed=20_260_717,
        )
    )

    assert summary.accepted
    assert summary.success_rate >= 0.95
    assert summary.p95_error_m < 0.10
    assert summary.maximum_error_m < 0.20
    assert summary.outlier_cases > 20
