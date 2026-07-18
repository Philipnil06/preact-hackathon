# Genesis Mini firmware

The sketch in
`genesis_mini/preact_genesis/preact_genesis.ino` is the hardware half of
PREACT. It sends Time-of-Flight readings to the laptop, accepts one servo angle
at a time, and mirrors the active mode on two separate physical indicators.

## Required hardware

- Axiometa Genesis Mini
- Axiometa Distance Sensor, ToF-VL53L0CX, AX22-0015
- Axiometa Servo Adapter, AX22-0031
- two Axiometa Single NeoPixel modules, AX22-0036
- one standard hobby servo
- regulated external servo supply, normally 5 V and at least 1 A
- USB-C data cable to the laptop

The adapter's external supply input should power the servo. Do not assume the
Genesis Mini's 3.3 V rail can power it. The AX22 adapter shares ground between
the controller signal and the motor supply.

Use this fixed port layout:

- Port 1: ToF module
- Port 2: NeoPixel under the **TRACKING** label
- Port 3: NeoPixel under the **CALCULATE LANDING** label
- Port 4: Servo Adapter, servo on its first header

The AX22-0036 schematic connects NeoPixel Data-In to AX22 IO1. The Genesis Mini
board definition maps Port 2 IO1 to GPIO 6 and Port 3 IO1 to GPIO 16.

## Build in Arduino IDE

1. Install the Espressif ESP32 board package.
2. Select **Axiometa Genesis Mini** as the board.
3. Install **Adafruit VL53L0X** and **Adafruit NeoPixel**.
4. Open `preact_genesis.ino`.
5. Connect the ToF module and servo adapter to AX22 ports, with their ground
   marks aligned.
6. Connect the servo to the adapter's first header and connect external motor
   power.
7. Upload and open Serial Monitor at 115200 baud.

On boot the servo moves to 90 degrees, both mode indicators are off, and the
board sends a `hello` message. Valid ToF messages then arrive continuously as
newline-delimited JSON.

## Verified CLI build

The development laptop has Arduino CLI 1.5.1, Espressif ESP32 3.3.10,
Adafruit_VL53L0X 1.2.5, and Adafruit NeoPixel. Compile with:

```powershell
& "C:\Program Files\Arduino CLI\arduino-cli.exe" compile `
  --fqbn esp32:esp32:axiometa_genesis_mini `
  --output-dir artifacts\firmware-build `
  firmware\genesis_mini\preact_genesis
```

The verified build uses 372,869 bytes of flash (28%) and 25,332 bytes of dynamic
memory (7%). Warnings emitted with `--warnings all` originate in the Espressif
core and dependencies, not this sketch.

## Day-of calibration

The servo adapter must use Port 4 for the default first-header signal
`P4_IO0/GPIO1`. PWM is 50 Hz.

Before attaching the printed pointer:

1. upload with no horn or pointer attached;
2. let the servo move to neutral;
3. attach the horn so the pointer faces the center mark;
4. test 10 and 170 degrees briefly;
5. narrow `SERVO_MIN_ANGLE_DEG` and `SERVO_MAX_ANGLE_DEG` if the mechanism
   approaches a hard stop.

The laptop and firmware use protocol version 1. Protocol details are in
`docs/PROTOCOL.md`.
