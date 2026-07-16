# PREACT

**See the impact before it happens.**

PREACT is a safe embodied-AI hackathon prototype that predicts where a thrown foam ball will first touch the floor and makes that prediction visible before impact.

The project is being built by Philip, Michelle, and Malte for the Axiometa × Anthropic Hardware Hack in London, 17–19 July 2026.

## Demo

1. A laptop camera observes a brightly coloured foam ball from the side.
2. Local computer-vision code converts camera pixels to a calibrated 2D measurement plane.
3. A physics-first model estimates the future floor intersection and confidence.
4. Preferred output: a one-axis servo pointer moves to the predicted landing position.
5. Guaranteed fallback: the Genesis Mini screen and 5×5 LED matrix show the predicted zone and confidence.
6. Claude helps commission the scene and diagnose weak predictions, but it is not in the real-time control loop.

## Success criteria

- Prediction at least 300 ms before first floor contact.
- Landing error at most 20 cm in at least 8 of 10 controlled throws.
- Reset within 5 seconds.
- Five successful demonstrations in a row.
- A replay mode and recorded backup demo.

## Current constraints

- One ordinary 720p laptop camera.
- Throws are constrained to a marked 2D/2.5D plane.
- Every attendee receives a Genesis Mini Starter Kit.
- Servo adapter, positional servo, ToF sensor, servo power, and USB data cable are not guaranteed by the published kit contents.
- Fifteen Bambu Lab printers are confirmed, but the exact models are unknown.
- No hardware will be brought from home or purchased at the venue.

The full moving-pointer demo is therefore conditional on event inventory. The camera predictor and LED/screen fallback remain buildable from the confirmed kits.

## Repository map

- [`AGENTS.md`](AGENTS.md) — persistent context and working rules for coding agents.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — data flow and component boundaries.
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — confirmed kit contents, conditional parts, and fallbacks.
- [`docs/HACKATHON_PLAN.md`](docs/HACKATHON_PLAN.md) — roles, milestones, and the 24-hour schedule.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — important decisions and rationale.
- [`docs/SAFETY.md`](docs/SAFETY.md) — project scope and safe-demo rules.

## Software direction

- Python 3.12
- OpenCV
- NumPy / SciPy
- PySerial
- Anthropic SDK
- Axiometa Studio or Arduino-compatible Genesis firmware
- Bambu Studio 02.07.01.62

Dependencies are listed in [`requirements.txt`](requirements.txt). Do not commit API keys; use `.env` locally.

## Sources

- [Axiometa Genesis Mini Starter Kit](https://www.axiometa.io/products/axiometa-genesis-mini-starter-kit)
- [Axiometa Genesis Mini](https://www.axiometa.io/products/axiometa-genesis-mini)
- [Axiometa Servo Adapter](https://www.axiometa.io/products/servo-adapter)
- [Axiometa ToF Distance Sensor](https://www.axiometa.io/products/distance-sensor-vl53l0cx)
- [OpenCV camera calibration](https://docs.opencv.org/master/d4/d94/tutorial_camera_calibration.html)
- [OpenCV ArUco markers](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html)

## Scope

This repository is for benign ball tracking, sports/education, industrial-safety exploration, and embodied-AI demonstration. It must not contain weaponization, firing, interception, or autonomous engagement functionality.
