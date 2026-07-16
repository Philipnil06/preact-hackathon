# Safety and scope

## Allowed project scope

- Tracking a thrown foam ball in a controlled indoor zone.
- Predicting first floor contact.
- Displaying the prediction on a screen, LED matrix, or passive pointer.
- Measuring latency, error, and confidence.
- Exploring sports, education, industrial-safety, and robotics applications.

## Out of scope

- Launching or firing any object.
- Catching, striking, intercepting, or pursuing the ball.
- Drone or missile engagement.
- Autonomous target selection.
- Weapon aiming or fire-control integration.
- Lasers used as pointers in the venue.

## Physical demo rules

- Use a soft foam ball.
- Mark a clear throwing and landing zone.
- Keep spectators outside the throw path.
- Use a lightweight pointer with limited travel.
- Default the actuator to a neutral position.
- Suppress actuator movement when confidence is low.
- Follow venue and organizer instructions.

## Data handling

- Do not record bystanders unnecessarily.
- Keep API keys in local environment variables.
- Do not commit event credentials or personal information.
- Delete calibration recordings that contain people unless they are needed and consented.
