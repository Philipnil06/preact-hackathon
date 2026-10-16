# Venue runbook

> Historical JSON-firmware integration checklist from the hackathon. For the
> current pan/tilt text-protocol path, use [Setup](SETUP.md). This document
> records planned procedures, not a completed hardware evaluation.

## Important scope note

Camera-only `--vision-only-landing` is a local bench mode used to tune the
predictor. It bypasses the real ToF gate and is not the venue demo claim. The
venue target is a two-servo pan/tilt pointer; retain the calibrated side-on
camera plane for the first end-to-end proof, then calibrate the physical pan,
tilt, and ToF placement before presenting any 3D behaviour.

## Before touching hardware

1. Reserve every required part in `HARDWARE.md`.
2. Choose a 3 m landing lane with a plain background and stable lighting.
3. Put the camera side-on, approximately perpendicular to the throw plane.
4. Keep the camera fixed after calibration.
5. Confirm the laptop is on mains power and sleep is disabled.

## Software health check

```powershell
cd "C:\Users\Philip Nilsson\Documents\Codex\2026-07-16\oke\work\preact-hackathon"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m preact
```

Open <http://127.0.0.1:8765>. Press **Start tracking**, verify that the green
mode card activates and the servo angle follows the synthetic ball. Press
**Calculate landing**, verify the red mode card and a predicted floor target,
then press **Stop** and verify idle. This does not open the webcam or microphone.

## Prepare voice without opening the microphone

The laptop's local `.env` contains the verified microphone profile. Fill in:

```dotenv
ANTHROPIC_API_KEY=your_key_here
```

The local Whisper model is already cached. The verified laptop profile is raw
microphone device 16, channel 1, 48 kHz, with a 0.0002 minimum energy threshold.
PREACT resamples to 16 kHz and measures the current noise floor for 0.5 seconds
at startup. At the venue, add `--voice` to the live command. Exact commands are
local and work even before the API key is entered:

- “Hey Preact, start tracking”
- “Hey Preact, calculate where this will land”
- “Hey Preact, stop”

Claude is called only for a paraphrase after the “Hey Preact” wake phrase.
Stop is an exact local fast path.

If hardware is not yet connected, the real camera/microphone can be tested with
`--emulate-genesis`. This mode never fabricates a ToF-ready state and is not a
replacement for the physical integration test.

## Flash Genesis

Arduino CLI 1.5.1, Espressif ESP32 3.3.10, the Axiometa Genesis Mini board
definition, Adafruit_VL53L0X 1.2.5, and Adafruit NeoPixel 1.15.5 are installed.

1. Connect Genesis Mini and identify its COM port.
2. Compile once with `scripts\verify.ps1`.
3. Upload with the CLI or install Arduino IDE 2.x if a graphical serial monitor
   is preferred.
4. Select **Axiometa Genesis Mini** and its COM port.
5. Open
   `firmware/genesis_mini/preact_genesis/preact_genesis.ino`.
6. Upload with servo power disconnected.
7. Open Serial Monitor at 115200 baud.

Expected first line:

```json
{"type":"hello","device":"genesis-mini","protocol_version":1,"firmware_version":"0.1.0"}
```

Then expect continuous valid `tof` lines. Before servo power is enabled, use
the UI or serial mode messages to verify:

- tracking: Port 2 lamp green, Port 3 lamp off;
- landing: Port 2 lamp off, Port 3 lamp red;
- idle: both lamps off.

## Prove the ToF gate

1. Aim ToF at a static background 0.8–1.5 m away.
2. Leave the beam clear for at least one second.
3. Move the foam ball through the intended release point.
4. Record baseline and near-object ranges.
5. Confirm the drop exceeds 80 mm with margin.
6. Repeat ten times.

If the margin is smaller, move the background farther away or use:

```powershell
--tof-trigger-drop-mm 50
```

Do not lower it until background noise has been measured.

## Servo proof

1. Keep horn and pointer detached.
2. Connect approved external 5 V servo power.
3. Start at neutral 90°.
4. Test 20° and 160°.
5. If direction is reversed, swap the runtime endpoint values:

```powershell
--servo-angle-min-deg 160 --servo-angle-max-deg 20
```

6. Narrow the firmware's 10–170° hard limits if the servo approaches a stop.
7. Attach the horn at the physical centre mark only after limits are safe.

## Calibrate the camera

Camera calibration and live capture must use the same resolution, normally
1280 × 720.

1. Generate IDs 0–3 with `preact.calibrate_cli markers`.
2. Put marker centres at the four measured plane corners:
   bottom-left, bottom-right, top-right, top-left.
3. Capture a still without moving the camera.
4. Run `preact.calibrate_cli plane`.
5. Inspect the preview and reported reprojection RMSE.
6. Capture a still with the foam ball held against the plain background.
7. Run `colour` with a tight ROI or `colour-claude`.
8. Inspect the ball preview; no background object should receive a red box.

If paper markers are unavailable, read four pixel coordinates from the still
and use `plane-manual`.

## Live command

```powershell
.\.venv\Scripts\python.exe -m preact `
  --source 0 `
  --camera-width 1280 --camera-height 720 --camera-fps 30 `
  --calibration calibration\plane.json `
  --detector-profile calibration\ball.json `
  --serial-port COM6 `
  --landing-min-x-m 0 --landing-max-x-m 3 `
  --servo-angle-min-deg 20 --servo-angle-max-deg 160 `
  --servo-neutral-deg 90 `
  --servo-mapping planar-pointing `
  --servo-base-x-m -0.4 --servo-base-y-m 0.5 `
  --servo-zero-angle-deg 90 --servo-direction 1 `
  --minimum-confidence 0.45 `
  --voice `
  --log logs\venue.jsonl
```

The browser UI mirrors the physical state. Start tracking does not require ToF.
Calculate landing resets the old track and ToF baseline; wait until the footer
reports a stable ToF range before throwing. Stop returns the pointer to neutral,
turns both indicators off, and leaves the voice listener ready.

## Servo-plane calibration

The pointer must rotate in the same side-on vertical plane used for camera
calibration. Measure the servo axis as world coordinates:

- `servo-base-x-m`: horizontal position along the same floor scale;
- `servo-base-y-m`: servo-axis height above the calibrated floor;
- `servo-zero-angle-deg`: physical servo angle that points along world +x;
- `servo-direction`: `1` or `-1` depending on mechanical rotation.

With the ball held at several known points, start tracking and tune these four
values until the arm follows. Then put the ball on several known floor points
and verify landing targets use the same geometry. Keep the configured
20–160-degree software envelope inside the mechanism's safe range.

## Ten-throw calibration

For each throw record:

- actual first-contact x;
- predicted x;
- absolute error;
- time to impact when the first servo command appears;
- confidence;
- missed ToF triggers;
- false colour detections.

Change one parameter class at a time:

1. plane geometry/homography;
2. colour profile/ROI;
3. ToF placement and drop threshold;
4. servo endpoint mapping;
5. confidence threshold.

Do not tune gravity to hide a bad plane calibration.

For `planar-pointing`, do not place the arm on a top-down axis: a single servo
must rotate in the calibrated vertical plane to point at both airborne and
floor targets.

## Recovery table

| Symptom | First check | Recovery |
|---|---|---|
| No `hello` | Wrong COM port or cable | Re-select port; use a data-capable USB cable |
| Invalid ToF samples | Module seating/I²C | Power down, align ground marks, reseat |
| No trigger | Baseline not ready or drop too small | Clear beam, wait 1 s, reposition background |
| Constant trigger | Ball/hand remains in beam | Clear beam until range returns to baseline |
| Servo jitters/resets board | Motor power | Use external regulated supply and common ground |
| Servo reversed | Mechanical orientation | Change `--servo-direction` between `1` and `-1` |
| Tracking points consistently high/low | Wrong servo-axis world position | Re-measure base x/y and zero angle |
| Red box on background | HSV profile too broad | Re-sample tighter ROI; simplify background |
| Good box, bad landing | Camera moved or wrong resolution | Restore camera and recalibrate plane |
| No prediction | Too few samples/span or low confidence | Slower throw, larger ball, wider visible path |
| UI unavailable | Local server/port | Restart command; check port 8765 |
| Voice says listening but no command | Input device, room noise, wake phrase | Select microphone device; tune energy threshold; use UI buttons |
| Whisper hears only part after a pause | Wake phrase and command split | PREACT joins a wake-only segment to the next segment for 2.5 s |
| Claude command error | API key/network | Use exact canonical phrase; exact commands remain local |
| Wrong physical lamp | Port order | ToF P1, green P2, red P3, servo P4; reseat with power off |

## Backup

Record at least three clean physical runs after the accuracy gate. Save:

- video;
- exact runtime command;
- `plane.json`;
- `ball.json`;
- JSONL telemetry;
- a photo of wiring and ToF placement.

Verify file replay with `--replay-emulated-tof` and set
`--replay-trigger-s` to the frame time where the real release occurs.
