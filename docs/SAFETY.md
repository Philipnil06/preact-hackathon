# Safety and scope

## Allowed scope

- Track one hand-thrown, lightweight foam ball in a controlled indoor lane.
- Continuously point a passive arm at that ball inside the calibrated plane.
- Predict its first contact with the marked floor.
- Move a passive, lightweight servo pointer to the predicted coordinate.
- Measure latency, error, confidence, trigger reliability, and recovery.
- Use local speech transcription and Claude only for safe command paraphrases
  and optional commissioning of the approved ball.
- Explore industrial-safety, robotics, logistics, sports, and education uses.

## Out of scope

- Launching, firing, catching, striking, intercepting, or pursuing any object.
- Selecting people, animals, vehicles, weapons, drones, or hazardous objects.
- Drone or missile engagement.
- Fire-control, aiming, autonomous engagement, or weapon integration.
- A laser pointer or any projected beam at the venue.
- Operation outside the marked demo lane.

## Physical rules

- Use a soft foam ball only.
- Mark throwing and landing zones and keep spectators outside them.
- Keep the camera side-on and prevent throws toward equipment or people.
- Use a lightweight pointer with limited 10–170° travel.
- Test servo neutral and limits with the pointer detached.
- Power the servo from the adapter's approved external supply.
- Suppress actuator commands below the confidence threshold.
- Stop returns the pointer to neutral.
- Idle and Stop turn both physical mode indicators off.
- Follow organizer and venue instructions over this runbook.

## Claude guardrail

The voice tool schema permits only tracking, landing, stop, or ignore after the
“Hey Preact” wake phrase. The target-selection tool permits only one bright,
inanimate, lightweight demo object. Returned image coordinates are validated.
If Claude does not select a suitable target, the system keeps the local
fallback profile; it does not broaden its search to people or other objects.

Claude never receives a servo tool or target coordinate tool.

## Data handling

- Avoid recording bystanders.
- Tell teammates when a commissioning image is sent to the Anthropic API.
- Keep `ANTHROPIC_API_KEY` in `.env`; never commit it.
- Store telemetry locally.
- Delete calibration images containing people when no longer needed.
- Do not commit event credentials, personal data, or raw venue footage.
