# Setup and operating modes

Start with the installation and synthetic demo in the [README](../README.md).
Run all commands below from the repository root, with the environment active.
Commands use PowerShell continuation syntax; use backslashes instead of
backticks in a POSIX shell. Camera indices, COM ports, and coordinates below
are examples, not detected settings.

## Choose the hardware path

| Path | Firmware | Options | Behaviour |
| --- | --- | --- | --- |
| Synthetic | None | `python -m preact` | Generated frames, simulated ToF, emulated pointer |
| Camera bench | None | `--source 0 --emulate-genesis --vision-only-landing` | Real camera, emulated pointer, no physical release gate |
| Video replay | None | `--source FILE --replay-emulated-tof` | Recorded frames, scripted ToF event, emulated pointer |
| ToF + planar pointer | `firmware/genesis_mini/preact_genesis/preact_genesis.ino` | `--source 0 --serial-port COM6` | JSON; one pointer axis, ToF, two mode LEDs |
| Pan/tilt tracking | `servo/firmware/sketch.ino` | `--source 0 --serial-port COM3 --pan-tilt-tracking --vision-only-landing` | Text; S1 pan and S3 tilt; no ToF or mode LED messages |

The firmware variants have different protocols and port layouts. The original
[hardware guide](HARDWARE.md), [runbook](RUNBOOK.md), and
[protocol specification](PROTOCOL.md) describe the JSON variant. The pan/tilt
sketch uses the servo adapter on Port 1 instead. Close the standalone
[Web Serial panel](../browser/Servo%20Control.html) before running Python:
only one process can own the serial port.

## Camera calibration

The ball must remain close to the measured side-on plane. Use the same image
resolution for calibration and capture. Camera movement invalidates the map.

Generate printable ArUco markers:

```powershell
python -m preact.calibrate_cli markers --output work/calibration/markers
```

Place marker centres at bottom-left (ID 0), bottom-right (ID 1), top-right
(ID 2), and top-left (ID 3) of a measured rectangle in the throwing plane.
Capture a scene image yourself, then run:

```powershell
python -m preact.calibrate_cli plane `
  --image work/calibration/scene.jpg `
  --width-m 3.0 --height-m 2.0 `
  --output work/calibration/plane.json `
  --preview work/calibration/plane-preview.jpg
```

Without markers, supply measured image points in the same order:

```powershell
python -m preact.calibrate_cli plane-manual `
  --points "80,650;1200,650;1200,70;80,70" `
  --width-m 3.0 --height-m 2.0 `
  --output work/calibration/plane.json
```

Create a tracking colour profile from a tight pixel rectangle around the object:

```powershell
python -m preact.calibrate_cli colour `
  --image work/calibration/ball.jpg --roi 480,250,120,120 `
  --output work/calibration/ball.json `
  --preview work/calibration/ball-preview.jpg
```

Tracking uses `--detector-profile`. Live/replay landing mode uses its own
colour-and-motion profile, `--landing-detector-profile`, defaulting to
`config/red-landing-ball.json`. Tune that profile for the thrown ball and
lighting; it is independent of the balloon tracking profile. After selecting
**Calculate landing**, leave the scene empty while the detector learns its
background, then throw the ball through the calibrated plane.

## Camera bench without hardware

```powershell
python -m preact `
  --source 0 --camera-width 1280 --camera-height 720 --camera-fps 30 `
  --calibration work/calibration/plane.json `
  --detector-profile work/calibration/ball.json `
  --emulate-genesis --vision-only-landing `
  --log work/bench.jsonl
```

This explicitly opens the camera and bypasses the physical ToF gate. It tests
the vision/prediction path; it does not validate the physical sensor or pointer.

## JSON ToF and planar pointer

Follow the [firmware build and wiring instructions](../firmware/README.md).
Measure the pointer base in the same vertical plane as the ball:

```powershell
python -m preact `
  --source 0 --camera-width 1280 --camera-height 720 --camera-fps 30 `
  --calibration work/calibration/plane.json `
  --detector-profile work/calibration/ball.json `
  --serial-port COM6 --servo-mapping planar-pointing `
  --servo-base-x-m -0.4 --servo-base-y-m 0.5 `
  --servo-zero-angle-deg 90 --servo-direction 1 `
  --servo-angle-min-deg 20 --servo-angle-max-deg 160 `
  --log work/hardware.jsonl
```

Omit `--vision-only-landing` for the physical ToF-gated demonstration. `linear`
mapping is also available for a pointer over a printed landing scale.

## Two-axis pan/tilt tracking

Upload `servo/firmware/sketch.ino`. The Python text-protocol adapter drives
**S1** horizontally and **S3** vertically. The checked-in camera and five-point
mount files reflect an earlier setup and must be recalibrated for yours.

```powershell
python -m preact `
  --source 0 --camera-width 640 --camera-height 480 --camera-fps 30 `
  --calibration config/camera-calibration.json `
  --detector-profile config/blue-balloon.json `
  --serial-port COM3 --pan-tilt-tracking --vision-only-landing
```

In the UI, use **Calibrate mount** to measure the centre and four corners.
Select **Balloon**, then **Start tracking**. This path follows an image target;
it does not reconstruct its depth. The text firmware has no ToF stream or
physical mode indicators.

Face tracking is an optional experimental extension. It requires local
YuNet/SFace model files and a commissioned reference profile supplied with
`--face-detector-profile`. Those assets are not bundled and the default demo
does not need them. See [reference intake](../config/face-reference/README.md).

## Deterministic video replay

Supply your own recording and its matching calibration:

```powershell
python -m preact `
  --source recordings/throw.mp4 `
  --calibration work/calibration/plane.json `
  --detector-profile work/calibration/ball.json `
  --replay-emulated-tof --replay-trigger-s 0.40
```

Replay injects a deterministic ToF event at the selected video timestamp and
uses an emulated pointer. No sample recording is distributed with this repo.

## Optional voice and cloud features

Add `--voice` to enable microphone capture and local faster-whisper
transcription. The first run may download the `base.en` Whisper model.
Microphone device/rate/channel options are listed by `python -m preact --help`.

Canonical commands are parsed locally:

- “Hey Preact, start tracking”
- “Hey Preact, switch to balloon”
- “Hey Preact, switch to my face”
- “Hey Preact, calculate where this will land”
- “Hey Preact, stop”

Select a configured target before starting tracking. Face selection remains
unavailable until a commissioned profile is supplied.

For paraphrase classification or one-shot object selection, copy
`.env.example` to `.env` and set `ANTHROPIC_API_KEY` and an `ANTHROPIC_MODEL`
available to your account. The checked-in model name is a historical
configuration value, not a guarantee of API availability.

`--claude-target-lock` sends an initial image for object selection instead of
using `--detector-profile`; choose one. `colour-claude` is also available in
the calibration CLI. These optional features send command text or a selected
image to Anthropic. Local profiles avoid those calls.

The separate `preact-say "message"` helper supports ElevenLabs speech output
when its environment settings are present; it is not required by the demo.
Keep credentials, recordings, model weights, and personal reference data local.
