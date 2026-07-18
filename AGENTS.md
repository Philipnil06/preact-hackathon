# PREACT agent context

## Objective

Build a reliable, safe voice-controlled demo that either follows a foam ball
continuously or predicts where it will first touch the floor and points there
before impact.

## Team

- Philip: product, integration, CAD/physical rig, measurement protocol, pitch.
- Malte: camera, calibration, computer vision, trajectory model, telemetry.
- Michelle: Genesis firmware, ToF, servo, power, serial link, recovery testing.

Keep these workstreams parallel until the first end-to-end integration.

## Non-negotiable decisions

1. Prove prediction first in a controlled side-on 2D/2.5D throwing plane; the
   physical pointer is a planned two-axis pan/tilt rig for the final 3D demo.
2. Build only the servo + ToF physical version.
3. Treat ToF as a required release/passage trigger and confidence gate.
4. Use the camera, not ToF, for trajectory geometry.
5. Use local deterministic code for all time-critical tracking and prediction.
6. Transcribe locally; exact wake commands are deterministic and Claude only
   classifies safe paraphrases or performs one-time object commissioning.
7. Keep the UI to two mode cards, Stop, camera image, red box, and small status
   text.
8. Leave the pointer neutral and send no command below the confidence threshold.
9. Do not activate a physical webcam until Philip explicitly approves it.

## Required physical path

- Axiometa Genesis Mini
- AX22-0015 VL53L0X ToF module
- AX22-0031 Servo Adapter
- two AX22-0036 Single NeoPixel modules
- two positional hobby servos, each with a horn
- regulated external servo power
- USB-C data cable
- laptop camera
- passive printed pan/tilt pointer and base

The two separate physical indicators are green for TRACKING and red for
CALCULATE LANDING. Synthetic and replay modes exist for software verification
and a recorded judging backup.

## Runtime contracts

- World coordinates: `x` along the floor, `y` upward, both in metres.
- Camera and ToF timestamps must share a monotonic time base in live mode.
- A new ToF trigger clears the previous observation track.
- Tracking mode follows the current calibrated ball position without a ToF
  trigger.
- Prediction output includes validity, landing x, time to impact, confidence,
  fit diagnostics, and optional servo target.
- Servo output requires a recent ToF trigger, a valid prediction, and sufficient
  confidence in landing mode, or sufficient detection quality in tracking mode.
- Stop always enters idle, neutralises the servo, and turns both mode LEDs off.
- Firmware protocol is versioned newline-delimited JSON at 115200 baud.

## Definition of done

- At least 8/10 controlled throws land within 20 cm of the prediction.
- First stable prediction is available at least 300 ms before impact.
- Five consecutive demonstrations complete without manual recovery.
- Stop and reset return the pointer to neutral.
- Replay and synthetic modes run without physical hardware.
- Telemetry is written for measured runs.
- No secrets or personal calibration footage are committed.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Preserve the safe synthetic default: `python -m preact` must never open a camera.
