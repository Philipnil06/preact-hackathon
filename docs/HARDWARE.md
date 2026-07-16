# Hardware and event inventory

## Confirmed Genesis Mini Starter Kit contents

Axiometa's official Starter Kit page lists the following items:

| Item | Planned PREACT use |
|---|---|
| Axiometa Genesis Mini | Physical I/O controller |
| Temperature and humidity sensor | Not needed for MVP |
| 0.96-inch screen | Landing zone, confidence, and status fallback |
| LED button | Arm/reset/demo trigger |
| Rotary encoder | Manual calibration or sensitivity adjustment |
| 5×5 LED matrix | Direction and confidence fallback |
| Passive buzzer | Optional valid/invalid prediction sound |
| Light sensor | Optional lighting warning |
| Remote transceiver | Not needed for MVP |
| Genesis Studio access | Firmware generation and flashing |

Official source: [Genesis Mini Starter Kit](https://www.axiometa.io/products/axiometa-genesis-mini-starter-kit)

## Confirmed event resources

- One starter kit per attendee; expected team total: three.
- Fifteen Bambu Lab printers.
- PLA: five black, five white, five grey, and five green units/spools; spool size not stated.
- Extra motors, screens, modules, and experimental parts, subject to availability.
- Venue open overnight.

## Not confirmed

- Exact Bambu Lab printer models.
- Nozzle diameter.
- AX22 Servo Adapter (AX22-0031).
- 180-degree positional hobby servo.
- AX22 ToF Distance Sensor (AX22-0015).
- External 5 V servo power.
- USB-C data cables.

“Extra motors” does not guarantee a positional servo, its AX22 adapter, or suitable power.

## Demo tiers

### Tier A — full pointer

Needs:

- Genesis Mini
- AX22 Servo Adapter
- 180-degree positional microservo
- suitable external 5 V motor power
- light pointer and base
- ToF optional

Output: pointer moves to predicted landing coordinate.

### Tier B — guaranteed kit

Needs:

- Genesis Mini
- 0.96-inch screen
- 5×5 LED matrix
- camera predictor running on laptop

Output: screen shows centimetres/zone and time to impact; LED matrix shows left/centre/right plus confidence.

### Tier C — laptop-only recovery

Needs:

- laptop and built-in camera

Output: on-screen trajectory, predicted impact, confidence, and replay mode.

## Servo power note

The Axiometa Servo Adapter exposes standard signal/Vin/GND servo headers and supports external motor power. Do not assume the Genesis 3.3 V rail can power a hobby servo. Use the event's approved servo-power arrangement and a common ground.

Source: [Axiometa Servo Adapter](https://www.axiometa.io/products/servo-adapter)

## ToF role

The VL53L0X-based sensor is a single-zone distance sensor. It is useful as a trigger, calibration aid, or occasional extra observation. It is not the primary ball tracker and is not required for the MVP.

Source: [Axiometa ToF Distance Sensor](https://www.axiometa.io/products/distance-sensor-vl53l0cx)

## 3D-print rules

- Use Bambu Studio.
- Keep every part within 170 × 170 mm until the exact printer is known.
- PLA only.
- Design without supports.
- Target 60–90 minutes maximum per print.
- Print the servo base first.
- Keep a cardboard/foamboard mechanical fallback.
