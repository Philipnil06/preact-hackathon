# Decision log

## D-001 — Controlled 2D/2.5D throwing plane

**Decision:** Use a side-on camera and constrain throws near a calibrated plane.

**Why:** A single ordinary camera cannot reliably recover unconstrained room-scale 3D motion. The controlled plane produces a credible demo within one day.

## D-002 — Physics-first real-time loop

**Decision:** Use deterministic computer vision and a trajectory fit for the real-time prediction.

**Why:** It is fast, inspectable, measurable, and robust to API latency.

## D-003 — Claude outside the time-critical loop

**Decision:** Claude handles commissioning and post-throw diagnosis.

**Why:** This gives the model a genuine role without using an LLM for precision geometry or actuation timing.

## D-004 — ToF is optional

**Decision:** The core MVP cannot depend on ToF.

**Why:** The kit does not include it, event quantity is unknown, and a single-zone ToF sensor is not a reliable room-scale ball tracker.

## D-005 — Guaranteed output before servo output

**Decision:** Implement the starter kit's LED matrix and screen first.

**Why:** Servo adapter, positional servo, and external power are not guaranteed.

## D-006 — Private repository

**Decision:** Start the repository as private.

**Why:** The project is early, contains unpublished hackathon work, and has a dual-use market narrative. Visibility can be changed later.

## D-007 — Safe product framing

**Decision:** Frame PREACT as sports, education, industrial safety, and robotics perception.

**Why:** The demo validates low-cost predictive perception, not a weapon or counter-UAS system.
