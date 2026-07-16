# PREACT agent context

## Objective

Build a reliable hackathon demo that predicts where a thrown foam ball will land and communicates that prediction physically before impact.

## Team

- Philip: product, integration, CAD/physical rig, measurement protocol, pitch.
- Malte: camera, computer vision, calibration, trajectory model, dashboard.
- Michelle: Genesis firmware, output hardware, serial/Wi-Fi connection, testing and recovery.

Roles can be swapped to match actual strengths, but keep the three workstreams separate.

## Non-negotiable decisions

1. The MVP uses a controlled side-on 2D/2.5D throwing plane.
2. Local deterministic code owns the time-critical tracking and trajectory fit.
3. Claude is used for scene commissioning, configuration suggestions, and post-throw diagnosis only.
4. The ToF sensor is optional and must never be a dependency of the core prediction.
5. Preferred output is a one-axis positional servo pointer.
6. Guaranteed fallback is the Genesis Mini 0.96-inch screen plus 5×5 LED matrix.
7. If prediction confidence is low, show “no prediction” and leave the pointer neutral.
8. No firing, interception, catching, targeting, or weaponization features.

## Confirmed event inventory

From the event Discord summary supplied on 16 July 2026:

- One Genesis Mini Starter Kit per attendee.
- Team size maximum: three.
- Fifteen Bambu Lab printers; exact models unknown.
- PLA in black, white, grey, and green.
- Extra Axiometa modules and experimental parts, types and quantities not guaranteed.
- Anthropic, Modal, Linkup, and ElevenLabs credits.
- Venue open overnight.

## Starter kit contents

The official Axiometa product page lists:

- Genesis Mini
- Temperature and humidity sensor
- 0.96-inch screen
- LED button
- Rotary encoder
- 5×5 LED matrix
- Passive buzzer
- Light sensor
- Remote transceiver
- Genesis Studio access

It does not list a servo adapter, servo, ToF sensor, motor power, or USB data cable.

## Engineering priorities

1. Recorded-video ball detection.
2. Camera calibration and pixel-to-centimetre mapping.
3. Landing prediction and confidence.
4. Live camera loop.
5. Guaranteed LED/screen output.
6. Servo output only when hardware is confirmed.
7. ToF integration only after the entire MVP works.
8. Dashboard polish, Claude commissioning, pitch, and backup video.

## Definition of done

- 8/10 throws within 20 cm.
- Prediction at least 300 ms before impact.
- Five consecutive demos.
- Replay mode works without live hardware.
- README and runbook match the implemented system.
- No secrets committed.

## Working rules

- Prefer simple colour segmentation over a new ML detector.
- Timestamp real frames; do not rely only on nominal FPS.
- Log raw observations and fitted predictions for every throw.
- Keep interfaces small: prediction output should include target position, time to impact, confidence, and validity.
- Never let uncertain hardware block the guaranteed fallback.
- Keep 3D-printed parts within 170 × 170 mm and support-free until the printer model is known.
