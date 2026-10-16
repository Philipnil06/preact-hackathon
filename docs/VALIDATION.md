# Validation

Checks reproduced on **16 October 2026**, on Windows with Python 3.12,
using a separate worktree containing the latest local runtime changes.

## Automated checks

| Check | Local result |
| --- | --- |
| `python -m pytest -q` | 92 passed, 2 skipped |
| `python -m ruff check src tests` | Passed |
| `python -m preact.trajectory_validation --scenarios 300` | Passed |
| Wheel build and separate target installation | Passed; packaged UI, synthetic prediction, and Stop verified |

The two skipped tests require a local biometric commissioning profile and its
reference images/models. They passed in the original development environment
(94 passed), but those personal assets are not distributed. The portable
checkout intentionally skips them. This does not validate face tracking for a
new user or a new environment.

Tests cover calibration, synthetic image detection, trajectory fitting,
confidence/servo gates, ToF handling, serial adapters/reconnection, mode
transitions, local voice parsing, mock cloud calls, and web endpoints.

## Seeded trajectory benchmark

```bash
python -m preact.trajectory_validation --scenarios 300 --seed 20260717
```

| Quantity | Result |
| --- | ---: |
| Scenarios / valid fits | 300 / 300 |
| Accepted predictions | 297 |
| Accepted predictions within 0.20 m | 297 |
| Success rate, all scenarios | 99% |
| Median error, accepted predictions | 0.015517 m |
| 90th-percentile error, accepted predictions | 0.043259 m |
| 95th-percentile error, accepted predictions | 0.051620 m |
| Maximum error, accepted predictions | 0.106359 m |
| Scenarios with a generated outlier | 88 |

The benchmark generates timestamped 2D world-coordinate observations of
ballistic throws, adds noise and occasional outliers, and runs the trajectory
predictor. Predictions require confidence of at least 0.45. The acceptance
gate requires at least 80% of all scenarios to produce a confident prediction
within 0.20 m. Three fits fell below the confidence threshold; they count
against the success rate and are excluded from the error percentiles.

This benchmark bypasses image detection, camera calibration, the runtime's
stable-landing lock, and serial/servo actuation. Its `actuated_predictions`
field means predictions above the benchmark confidence threshold; it does not
mean that any physical actuator moved. It measures the estimator under its
assumed model, not field accuracy or measured end-to-end performance.

## Interactive demo

The default `python -m preact` demo was opened in a browser with generated
frames, an emulated ToF stream, and an emulated pointer. Landing prediction,
preview rendering, tracking selection, and Stop are also exercised by the
web/runtime tests. The README screenshot is from that synthetic demo.

## Hardware evidence

No fresh physical camera/servo/ToF experiment was performed for this cleanup.
Historical firmware build notes and hackathon plans remain in the repository;
they are not a current hardware certification or repeatability result.

A useful physical evaluation should record all throws, accepted/rejected
predictions, predicted and actual contact positions, time before contact,
camera resolution, calibration, lighting, and servo response. Keep these
measurements separate from the synthetic benchmark above.
