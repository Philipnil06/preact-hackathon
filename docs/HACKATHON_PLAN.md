# 24-hour hackathon plan

The software foundation is prepared before the event. The first half of the
hackathon is therefore hardware-first, not a fresh coding sprint.

## Fixed ownership

- **Philip:** product lead, physical rig, CAD/printing, integration, scoring,
  Greypoint-oriented industrial narrative, final pitch.
- **Malte:** camera, calibration, detector, trajectory tuning, local Whisper,
  voice validation, telemetry, replay.
- **Michelle:** Genesis firmware, ToF placement, servo/power, two mode LEDs,
  serial reliability, reset and fault recovery.

One person owns each subsystem. Integration decisions are made together at the
hard gates.

## Schedule

| Hour | Shared outcome | Philip | Malte | Michelle |
|---:|---|---|---|---|
| 0–1 | Required parts reserved and test zone selected | Inventory owner, mark 3 m strip | Verify both synthetic modes; create local `.env` | Reserve ToF, adapter, 2× AX22-0036, servo, supply, cable; find COM |
| 1–2 | Sensors, LEDs, and actuator prove life independently | Measure servo axis and mounting geometry | Generate markers; verify Whisper model without live demo loop | Flash firmware; prove ToF JSON, green/red/idle states, neutral servo |
| 2–4 | First mechanical prototype printing | CAD base, pointer, cable access; slice and queue | Capture at 1280×720; create plane and ball profiles | Tune servo pulse limits with pointer detached |
| 4–6 | Stable tracking mode and release gate | Mount camera and define throw/servo plane | Verify red-box and arm tracking over full path | Position ToF; measure baseline/drop; tune trigger |
| 6–8 | First complete physical throw | Integrate base, labelled lamps, landing scale | Run live predictor and inspect fit | Verify mode/servo commands, ACKs, LEDs, neutral/reset |
| 8–10 | Ten measured throws | Measure actual landing and record error | Analyse JSONL error, latency, confidence | Record trigger/servo failures |
| 10–12 | Accuracy gate: ≥8/10 within 20 cm | Fix geometry and physical alignment | Tune ROI, homography, thresholds only | Tune servo mapping and power stability |
| 12–13 | Freeze core architecture | Decide only evidence-backed fixes | Lock equations and protocol | Lock firmware and wiring |
| 13–15 | Voice and Claude fallback work | Prepare visible voice-mode story | Test 3 exact phrases plus paraphrases; validate target lock | Confirm API delay cannot affect Stop or servo loop |
| 15–17 | Product-quality enclosure and UI | Print final parts and signage | Keep UI minimal; record clean overlay | Cable management and safe pointer limits |
| 17–19 | Industrial-safety narrative and evidence | Pitch, market, Greypoint relevance | Produce accuracy/latency summary | Produce hardware BOM and reliability summary |
| 19–21 | Stress gate: five consecutive demos | Run rehearsals and score | Log each throw and watch drift | Inject disconnect/reset failures and recover |
| 21–22 | Backup package | Record best physical runs | Verify deterministic replay | Write 60-second hardware recovery note |
| 22–23 | Full presentation rehearsals | Present and time | Operate laptop | Operate Genesis and reset |
| 23–24 | Freeze and submit | Submission owner | Blockers only | Pack, label, photograph wiring |

## Hard gates

- **Hour 1:** every required part is physically on the team table.
- **Hour 2:** ToF JSON streams, LEDs show all three states, and the unattached
  servo reaches neutral.
- **Hour 6:** ToF baseline/trigger is repeatable at least 10 times.
- **Hour 6:** tracking follows the held ball across the calibrated plane.
- **Hour 8:** one real throw moves the pointer to impact before the ball lands.
- **Hour 12:** 8/10 accuracy or a measured explanation of the dominant error.
- **Hour 19:** no new features.
- **Hour 21:** five consecutive successful demos.
- **Hour 22:** physical backup video and replay command both verified.

## Decision order when something fails

1. Check calibration resolution and marker order.
2. Check ToF baseline, placement, and drop threshold.
3. Check servo power separately from USB logic power.
4. Increase ball size/contrast and simplify the background.
5. Reduce the throwing plane and speed envelope.
6. Reprint only the smallest affected mechanical part.
7. Never replace the physics model with an untested model during the final half.

## Demo sequence

1. Show the passive pointer, two labelled lamps, and selected foam ball.
2. Say “Hey Preact, start tracking”; the green lamp turns on and the arm follows
   the held ball.
3. Say “Hey Preact, stop”; the lamp turns off and the pointer returns neutral.
4. Say “Hey Preact, calculate where this will land”; the red lamp turns on.
5. Throw through the ToF gate in the marked plane.
6. Let the red box and pointer communicate prediction before impact.
7. Compare the physical landing and pointer, then say “Hey Preact, stop”.
8. Explain that local Whisper heard the phrase, Claude can understand safe
   paraphrases, and deterministic geometry handled every real-time target.

## Pitch emphasis

Lead with measured predictive perception, not counter-drone or weapon framing:

- low-cost fusion of commodity vision and range sensing;
- early motion prediction rather than reactive detection;
- an inspectable confidence gate before actuation;
- applications in industrial safety, robotics, logistics, sports, and training;
- a credible path from one camera/one ToF gate to richer sensor networks.
