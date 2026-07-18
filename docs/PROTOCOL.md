# Laptop ↔ Genesis protocol

PREACT uses UTF-8, newline-delimited JSON at 115200 baud. Each JSON object must
fit on one line. Protocol version 1 is intentionally small enough to inspect in
Arduino Serial Monitor during the hackathon.

## Genesis to laptop

Ready:

```json
{"type":"hello","device":"genesis-mini","protocol_version":1,"firmware_version":"0.1.0"}
```

Distance sample:

```json
{"type":"tof","device_time_ms":1234,"distance_mm":712,"valid":true,"range_status":0}
```

Command acknowledgement:

```json
{"type":"ack","command_id":7,"accepted":true,"detail":"ok"}
```

## Laptop to Genesis

Point at a predicted landing position:

```json
{"type":"servo_target","protocol_version":1,"command_id":7,"angle_deg":123.5,"target_x_m":2.1,"confidence":0.83,"time_to_impact_ms":620,"target_y_m":0.0,"purpose":"landing"}
```

Or point at the ball's current position:

```json
{"type":"servo_target","protocol_version":1,"command_id":8,"angle_deg":88.2,"target_x_m":1.3,"confidence":0.91,"time_to_impact_ms":0,"target_y_m":1.1,"purpose":"tracking"}
```

Return to the neutral angle:

```json
{"type":"neutral","protocol_version":1,"command_id":10,"angle_deg":90.0}
```

Set the operating mode and physical indicators:

```json
{"type":"mode","protocol_version":1,"command_id":11,"mode":"tracking"}
```

Valid modes are:

- `tracking`: green TRACKING NeoPixel on, red LANDING NeoPixel off;
- `landing`: red LANDING NeoPixel on, green TRACKING NeoPixel off;
- `idle`: both NeoPixels off.

Landing servo commands require a recent ToF trigger and a valid,
confidence-gated ballistic prediction. Tracking mode emits current-ball servo
targets without a ToF trigger. Firmware clamps all angles again as a second
protection against mechanical hard stops.

The laptop retains only the latest desired mode and latest actuator state. If
USB serial disconnects, it reconnects and restores the mode first, followed by
the latest servo target or neutral command. Repeated Stop commands always
reassert both neutral and idle rather than relying on cached application state.
