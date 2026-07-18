# Hardware and venue inventory

## Exact required parts

Because the team will bring nothing and buy nothing, every item below must be
available at the venue.

| Qty | Required item | Purpose | Stop condition |
|---:|---|---|---|
| 1 | Axiometa Genesis Mini | ToF acquisition, serial bridge, servo PWM | Required |
| 1 | AX22-0015 Distance Sensor (VL53L0X) | Release/passage trigger | Required |
| 1 | AX22-0031 Servo Adapter | Two servo signals and external motor power | Required |
| 2 | AX22-0036 Single NeoPixel | Separate green TRACKING and red CALCULATE LANDING indicators | Required |
| 1 | 180° positional hobby servo | Move the pointer | Required |
| 1 | Regulated 5 V servo supply, ≥1 A | Power servo independently | Required |
| 1 | USB-C data cable | Genesis ↔ laptop serial | Required |
| 1 | Laptop with working webcam and microphone | Ball trajectory and voice input | Required |
| 1 | Bright matte foam ball | Safe tracked object | Required |
| 1 | Small terminal screwdriver | Servo-adapter power terminal | Required if screw terminal is fitted |
| 1 set | Power leads/connectors | Supply ↔ servo adapter | Required |
| 1 set | Tape, cable ties, screws, or elastic bands | Secure modules and camera | Required |
| 1 | Printed pointer/base assembly | Physical output | Required |
| 1 | 3 m measured landing strip/tape measure | Calibration and scoring | Required |
| 4 | Printed ArUco markers, IDs 0–3 | Fast camera calibration | Preferred |

For the intended 3D pointer, the physical requirement is **two** 180-degree
positional micro servos, each with a horn, plus regulated external 5 V power
capable of at least 2 A. The first servo is pan and the second is tilt. The
older one-servo row above describes only the current camera-plane software
proof and is superseded for the venue build.

The published Genesis Mini Starter Kit does not itself list the ToF module,
servo adapter, Single NeoPixels, servo, external power, or USB data cable. The
chosen build therefore assumes these are in Axiometa's event module pool. The
two NeoPixels are the clean target implementation for separate labelled lamps;
they are not guaranteed merely because a Genesis Mini is available. If a
required electronics item is unavailable, the intended physical demo cannot be
completed as designed; synthetic/replay remains the software backup.

## Relevant official module facts

### Genesis Mini

- ESP32-S3-Mini-1
- four AX22 ports
- 4 MB flash and 2 MB PSRAM
- USB-C, STEMMA QT, and Arduino IDE support

### AX22-0015 ToF

- VL53L0X-compatible library and I²C interface
- single-zone absolute distance
- official page states 1–4000 mm and up to 50 Hz continuous mode
- default I²C address shown as `0x52`
- connects through any compatible AX22 port

### AX22-0031 Servo Adapter

- three standard signal/Vin/GND servo headers
- official example signal GPIOs: 1, 14, and 41
- external motor-power input
- 50 Hz PWM in the official Arduino example

The adapter exposes three servo headers. Before wiring, confirm that the Genesis
Mini pin map permits two independent PWM outputs on the selected adapter/header
combination. PREACT needs two: pan and tilt.

### AX22-0036 Single NeoPixel

- one RGB NeoPixel per AX22 module;
- the official schematic connects Data-In to AX22 `IO1`;
- firmware sets the Port 2 module to green only in tracking mode;
- firmware sets the Port 3 module to red only in landing mode;
- both modules are off in idle.

The Starter Kit's AX22-0050 Tactile LED Button is not a colour substitute: its
published built-in LED is white. Reserve two AX22-0036 modules if the separate
green/red physical indication is required.

## Fixed four-port layout

| Genesis port | Module | Signal used |
|---|---|---|
| Port 1 | AX22-0015 ToF | I²C, SDA GPIO 10 / SCL GPIO 11 |
| Port 2 | AX22-0036 under **TRACKING** | IO1 / GPIO 6, illuminated green |
| Port 3 | AX22-0036 under **CALCULATE LANDING** | IO1 / GPIO 16, illuminated red |
| Port 4 | AX22-0031 Servo Adapter | First header, IO0 / GPIO 1 |

## Wiring order

Connect the pan and tilt servos to separate adapter headers with correct
signal/Vin/GND orientation. Test each at neutral with its horn detached before
attaching the printed pan/tilt pointer. Never power the servos from the Genesis
3.3 V rail.

1. Keep servo motor power disconnected.
2. Align the white ground marks and insert the ToF module into Port 1.
3. Insert the TRACKING NeoPixel into Port 2 and the CALCULATE LANDING NeoPixel
   into Port 3.
4. Insert the Servo Adapter into Port 4.
5. Connect the hobby servo to the adapter's first header with correct
   signal/Vin/GND orientation.
6. Connect regulated 5 V and ground to the adapter's external power terminal.
7. Confirm controller and motor supply share ground through the adapter.
8. Connect Genesis Mini to the laptop over USB-C.
9. Upload firmware; verify both indicators are off at boot.
10. Power the servo and verify neutral with the horn/pointer detached.
11. Attach the horn at the physical centre mark.

Do not assume the Genesis 3.3 V rail can power a hobby servo. A stalled servo can
brown out or damage the controller path.

## Mechanical build

The final print is a light passive pan/tilt pointer: two servo cradles, a wide
stable base, removable horns, and no laser or projectile mechanism.

Use a light passive pointer, not a laser. The base needs:

- four Genesis mounting holes or straps;
- one servo cradle;
- a wide, stable footprint;
- a hard-stop-free 10–170° pointer envelope;
- access to USB-C and external power;
- a removable pointer/horn connection.

Until the printer model is confirmed:

- keep each part within 170 × 170 mm;
- use PLA;
- design without supports;
- target under 60 minutes for the first base;
- use 0.28 mm layer height and 10–15% infill for prototypes;
- print the servo cradle separately so dimensions can be corrected quickly.

Bambu Studio 02.07.01.62 is installed on the laptop.

## Venue inventory script

Ask Axiometa for these exact names immediately:

> We need one AX22-0015 VL53L0X distance sensor, one AX22-0031 Servo Adapter,
> two AX22-0036 Single NeoPixel modules, one 180-degree positional servo,
> external regulated 5 V servo power with leads, a USB-C data cable, and a
> small screwdriver. Can we reserve them for PREACT for the next 24 hours?

Do not accept “a motor” without confirming it is a positional hobby servo.

## Sources

- [Genesis Mini](https://www.axiometa.io/products/axiometa-genesis-mini)
- [Servo Adapter AX22-0031](https://www.axiometa.io/products/servo-adapter)
- [Distance Sensor AX22-0015](https://www.axiometa.io/products/distance-sensor-vl53l0cx)
- [Single NeoPixel AX22-0036](https://www.axiometa.io/products/single-neopixel)
- [Tactile LED Button AX22-0050](https://www.axiometa.io/products/tactile-led-button)
