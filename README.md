# PREACT

**See the impact before it happens.**

PREACT is a safe embodied-AI prototype with two voice-controlled modes. It can
continuously point a pan/tilt robot arm at a marked demo object, or predict where a
hand-thrown foam ball will first touch the floor and point there before impact.

Philip, Michelle, and Malte are building it for the Axiometa × Anthropic
Hardware Hack in London, 17–19 July 2026.

## What the demo does

- **TRACKING:** “Hey Preact, start tracking” continuously finds the ball and
  points the arm at its current calibrated position. The green physical
  TRACKING indicator is on.
- **CALCULATE LANDING:** “Hey Preact, calculate where this will land” arms the
  ToF release gate, fits the camera trajectory, and points at predicted first
  floor contact. The red physical CALCULATE LANDING indicator is on.
- **IDLE:** “Hey Preact, stop” immediately returns the pointer to neutral,
  switches both indicators off, and waits for the next command.

The laptop camera detects the selected bright foam ball. A homography converts
pixels to metres in the side-on throwing plane. In landing mode a robust,
fixed-gravity fit predicts the future floor intersection. Genesis Mini controls
the servo, ToF gate, and two AX22-0036 NeoPixel indicators.

Speech is transcribed locally with faster-whisper. The exact three commands are
handled deterministically without an API call; Claude Sonnet 5 is a strict
command classifier only when the user paraphrases a wake-phrase command.
Claude may also select the intended safe demo object from one commissioning
image. It is never in the real-time geometry or servo loop.

## Current local status

- Tracking, trajectory prediction, confidence, ToF gating, planar servo
  pointing, the three-state controller, local speech transcription, Claude
  command fallback, serial protocol, two LED indicators, synthetic/replay
  input, calibration, local UI, and Genesis firmware are implemented.
- The safe default is synthetic input; it never opens the webcam.
- The camera-only landing bench mode is implemented and has produced correct
  predictions in clean controlled throws. It is not yet reliable enough to claim
  unconstrained 3D prediction.
- Venue integration remains: two servos, ToF placement, mechanical pan/tilt
  calibration, and repeatability testing.

## Install

Python 3.12, Bambu Studio 02.07.01.62, Arduino CLI 1.5.1, Espressif ESP32
3.3.10, and Adafruit_VL53L0X 1.2.5 are present on the development laptop.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Genesis firmware is in
[`firmware/genesis_mini/preact_genesis/preact_genesis.ino`](firmware/genesis_mini/preact_genesis/preact_genesis.ino).
It compiles against the actual `esp32:esp32:axiometa_genesis_mini` board target.

## Safe synthetic demo

```powershell
.\.venv\Scripts\python.exe -m preact
```

Open <http://127.0.0.1:8765>. Use **Start tracking**, **Calculate landing**, and
**Stop** to test both complete servo-emulator paths without opening a camera or
microphone.

## Voice setup

The local `.env` is already prepared for this laptop. Add only your Anthropic
API key:

```dotenv
ANTHROPIC_API_KEY=your_key_here
```

`ANTHROPIC_MODEL=claude-sonnet-5`, the timeout, and the verified microphone
device/rate/channel are already set. The local `base.en` Whisper model is
pre-downloaded, so no Hugging Face key is required. The three canonical phrases
also work without an Anthropic key; the key enables Claude paraphrase handling.

Voice is opt-in and does not start unless `--voice` is present:

```powershell
.\.venv\Scripts\python.exe -m preact --voice
```

The three canonical phrases are:

- “Hey Preact, start tracking”
- “Hey Preact, calculate where this will land”
- “Hey Preact, stop”

On the development laptop the verified raw microphone path is device 16,
channel 1 at 48 kHz. PREACT resamples it to 16 kHz for Whisper and calibrates a
local noise floor during the first 0.5 seconds.

## Camera + voice bench mode

When Genesis, ToF, servo, and LEDs are not connected, use the in-memory hardware
emulator while retaining the real camera and microphone:

```powershell
.\.venv\Scripts\python.exe -m preact `
  --source 0 `
  --camera-width 640 --camera-height 480 --camera-fps 30 `
  --calibration tmp\live-bench\plane.json `
  --emulate-genesis --voice `
  --servo-mapping planar-pointing
```

`--emulate-genesis` is a bench-only option. For camera-only development, add
`--vision-only-landing`; this bypasses the physical ToF gate and must not be
used to claim a complete physical demo.

## Calibration

Generate four printable markers:

```powershell
.\.venv\Scripts\python.exe -m preact.calibrate_cli markers `
  --output calibration\markers
```

Place marker centres at bottom-left (ID 0), bottom-right (ID 1), top-right
(ID 2), and top-left (ID 3) of the measured throwing plane:

```powershell
.\.venv\Scripts\python.exe -m preact.calibrate_cli plane `
  --image calibration\scene.jpg `
  --width-m 3.0 --height-m 2.0 `
  --output calibration\plane.json `
  --preview calibration\plane-preview.jpg
```

If markers are unavailable, use four measured pixel points in the same order:

```powershell
.\.venv\Scripts\python.exe -m preact.calibrate_cli plane-manual `
  --points "80,650;1200,650;1200,70;80,70" `
  --width-m 3.0 --height-m 2.0 `
  --output calibration\plane.json
```

Build a local colour profile from a tight rectangle around any bright,
saturated ball:

```powershell
.\.venv\Scripts\python.exe -m preact.calibrate_cli colour `
  --image calibration\ball.jpg `
  --roi 480,250,120,120 `
  --output calibration\ball.json `
  --preview calibration\ball-preview.jpg
```

With `ANTHROPIC_API_KEY` in a local `.env`, Claude can locate the object first:

```powershell
.\.venv\Scripts\python.exe -m preact.calibrate_cli colour-claude `
  --image calibration\ball.jpg `
  --output calibration\ball.json
```

## Physical run

The camera opens only when a numeric `--source` is explicitly provided:

```powershell
.\.venv\Scripts\python.exe -m preact `
  --source 0 `
  --camera-width 1280 --camera-height 720 --camera-fps 30 `
  --calibration calibration\plane.json `
  --detector-profile calibration\ball.json `
  --serial-port COM6 `
  --landing-min-x-m 0 --landing-max-x-m 3 `
  --servo-angle-min-deg 20 --servo-angle-max-deg 160 `
  --servo-mapping planar-pointing `
  --servo-base-x-m -0.4 --servo-base-y-m 0.5 `
  --servo-zero-angle-deg 90 --servo-direction 1 `
  --voice `
  --log logs\venue.jsonl
```

`planar-pointing` uses the measured servo-axis position in the same calibrated
side-on plane as the ball. This is what allows the same one-axis arm to point
at either the airborne ball or its future floor contact. Use `linear` only when
the pointer moves over a compact printed landing scale.

Replace the saved colour profile with `--claude-target-lock` to perform one
Claude vision call on the first frame. Do not use both flags together.

## Deterministic replay

```powershell
.\.venv\Scripts\python.exe -m preact `
  --source recordings\throw.mp4 `
  --calibration calibration\plane.json `
  --detector-profile calibration\ball.json `
  --replay-emulated-tof --replay-trigger-s 0.40
```

Replay injects a visible, deterministic ToF gate event but uses the same
prediction and servo logic as the physical path.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

## Documentation

- [`docs/MATH.md`](docs/MATH.md) — exact trajectory, confidence, ToF, and servo equations.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — component and data flow.
- [`docs/HARDWARE.md`](docs/HARDWARE.md) — exact venue parts and wiring.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — day-of setup and recovery.
- [`docs/HACKATHON_PLAN.md`](docs/HACKATHON_PLAN.md) — 24-hour team schedule.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — laptop ↔ Genesis JSON protocol.
- [`docs/SAFETY.md`](docs/SAFETY.md) — allowed demo scope.

## Scope

This repository is for benign marked-object tracking, sports and education,
industrial-safety exploration, and robotics perception. It must not contain
firing, interception, autonomous engagement, or weapon-control functionality.

## Primary references

- [Axiometa Genesis Mini](https://www.axiometa.io/products/axiometa-genesis-mini)
- [Axiometa Servo Adapter](https://www.axiometa.io/products/servo-adapter)
- [Axiometa ToF Distance Sensor](https://www.axiometa.io/products/distance-sensor-vl53l0cx)
- [Axiometa Single NeoPixel](https://www.axiometa.io/products/single-neopixel)
- [Claude Sonnet 5](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5)
- [Anthropic strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)
- [Anthropic vision](https://platform.claude.com/docs/en/build-with-claude/vision)
- [OpenCV ArUco detection](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html)
