# Decision log

> Historical decisions from hackathon development. Later work added a separate
> pan/tilt text-protocol path and optional face tracking. Use [Setup](SETUP.md)
> and [Architecture](ARCHITECTURE.md) for the current interfaces; planned work
> below is not evidence of measured hardware performance.

## D-001 — Controlled 2D/2.5D throwing plane

**Decision:** Use a side-on camera and constrain throws near one calibrated
plane.

**Why:** One ordinary camera cannot reliably recover unconstrained room-scale
3D motion in 24 hours. A measured plane makes accuracy testable.

## D-002 — Physics-first real-time loop

**Decision:** Use local colour segmentation, a planar homography, and a robust
fixed-gravity fit.

**Why:** It is fast, inspectable, deterministic, and measurable.

## D-003 — Claude selects; local code measures

**Decision:** Claude Sonnet 5 may select the safe demo object once through a
strict bounding-box tool. Local pixels create the HSV profile.

**Why:** Claude adds semantic commissioning without controlling geometry,
timing, or actuation. A saved profile is the offline path.

## D-004 — ToF is required but is not a 3D tracker

**Decision:** Build only the ToF + servo path. ToF creates a release/passage
event and gates camera observations and servo commands.

**Why:** The user chose the most impressive target build and accepts the event
inventory dependency. A single-zone sensor still cannot supply the full ball
trajectory.

## D-005 — Servo pointer plus two mode indicators

**Decision:** Keep the servo pointer as the physical result and add two separate
AX22-0036 indicators: green for tracking and red for landing calculation.

**Why:** The indicators make the voice-controlled state immediately legible
without turning the demo into a dashboard.

## D-006 — Minimal UI

**Decision:** Provide two mode cards, matching green/red lamps, Stop, the camera
feed, a red tracking box, and small voice/ToF/servo status text.

**Why:** The physical pointer is the product moment. A dashboard would compete
with it and add failure surface.

## D-007 — Confidence-gated actuation

**Decision:** Send no target below the threshold and clamp angles on both laptop
and firmware.

**Why:** A physically quiet “no prediction” is more credible and safer than
pointing based on weak data.

## D-008 — Safe product framing

**Decision:** Frame PREACT as predictive perception for industrial safety,
robotics, logistics, sports, and training.

**Why:** The prototype validates sensing and early motion prediction, not a
weapon, engagement system, or counter-UAS product.

## D-009 — Private and local until explicit release

**Decision:** Work locally and make no GitHub push or PR until Philip explicitly
asks.

**Why:** The implementation is still being tested, and the user requested local
work only.

## D-010 — Synthetic input is the safe default

**Decision:** `python -m preact` starts a synthetic end-to-end demo. A numeric
camera index must be explicit.

**Why:** Tests and development must not unexpectedly activate the webcam.

## D-011 — Three explicit operating modes

**Decision:** Use `idle`, `tracking`, and `landing` as mutually exclusive states.
Every transition resets stale tracks. Stop neutralises the servo and turns both
indicators off.

**Why:** An explicit state machine prevents tracking commands and landing
predictions from competing for the same actuator.

## D-012 — Local speech, deterministic safety commands

**Decision:** Transcribe speech locally with faster-whisper. Handle the three
canonical “Hey Preact” phrases locally and use Claude Sonnet 5 only to classify
wake-phrase paraphrases with a strict four-value tool.

**Why:** Canonical commands, especially Stop, remain fast and available without
network latency. The only cloud credential is `ANTHROPIC_API_KEY`.

The microphone path supports non-16-kHz multi-channel devices, resamples locally
for Whisper, calibrates the noise floor at startup, and joins a wake-only speech
segment to the following command for 2.5 seconds.

## D-013 — One vertical plane for both pointer behaviours

**Decision:** Place the servo axis in or next to the calibrated side-on plane
and calculate the bearing to either current ball \((x,y)\) or predicted landing
\((x,0)\).

**Why:** A single rotational axis cannot point to arbitrary 3D positions, but it
can visibly follow both targets when the ball remains in one controlled plane.
