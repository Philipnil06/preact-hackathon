# Project scope

PREACT explores a concrete robotics question: can a low-cost camera observe a
moving object early enough to predict its floor contact and move a pointer
before it arrives?

The demonstration uses a thrown ball, a calibrated camera plane, and a passive
servo pointer. A separate pan/tilt mode follows a coloured object or an
explicitly commissioned face reference. The engineering work connects
perception, geometry, serial communication, state transitions, and operator
interaction in one runnable system.

## What a reviewer can inspect

- A synthetic end-to-end demo with no hardware or credentials.
- Local image detection and pixel-to-world calibration.
- A robust ballistic fit with an explicit confidence threshold.
- Distinct adapters for two firmware protocols and a hardware emulator.
- Deterministic tests and reproducible synthetic validation.
- Optional language-model calls outside the time-critical control loop.

## What the evidence supports

The seeded validation measures prediction error on simulated 2D coordinate
observations. It does not establish camera accuracy, mechanical pointing
accuracy, arbitrary 3D tracking, or operation on drones. The face extension
requires local assets and separate commissioning.

Earlier hackathon discussions considered broader applications. Those ideas
are not implemented capabilities or evaluation results. The repository is
presented as the ball-prediction and tracking prototype that can be inspected
and exercised today.

## Next useful experiments

Measure physical throws against marked floor positions, report both accepted
and rejected attempts, and record error and time before contact. Separately
measure detector loss, calibration sensitivity, and servo response. These
measurements would show where the current pipeline works and which component
most limits repeatability.
