# 24 productive hours

The event spans a weekend; this plan covers 24 focused build hours.

## Roles

- Philip: product, physical rig, CAD, integration, test protocol, pitch.
- Malte: camera, vision, calibration, prediction, dashboard.
- Michelle: Genesis, output hardware, communication, testing, recovery.

## Schedule

| Time | Shared outcome | Philip | Malte | Michelle |
|---|---|---|---|---|
| 00:00–00:30 | Scope and metrics frozen | Throwing zone | Data interface | Inventory exact modules |
| 00:30–02:00 | Three hello-worlds | First physical mock-up | Camera + ball centroid | LED/screen output |
| 02:00–04:00 | Live tracking | ArUco layout | Stable observations | Laptop–Genesis link |
| 04:00–06:00 | Prediction from recorded video | Ground-truth protocol | Calibration + fit | Output mapping |
| 06:00–08:00 | End-to-end MVP | Integrate | Real-time prediction | LED/screen, servo if available |
| 08:00–10:00 | Stable physical build | Print or cardboard | Filtering and overlay | Mount modules |
| 10:00–12:00 | Thirty-throw test | Record outcomes | Error and latency | Hardware/reset failures |
| 12:00–14:00 | Robustness | Light/background | Outliers/no-prediction | Watchdog/recovery |
| 14:00–16:00 | Claude adds real value | User flow | Commissioning and diagnosis | Apply bounded config |
| 16:00–18:00 | Product feel | Signage/landing strip | Dashboard | LED/status animation |
| 18:00–20:00 | Market and story | Pitch | Technical visual | Hardware/BOM visual |
| 20:00–21:30 | Backup demo | Record best runs | Replay mode | Manual reset notes |
| 21:30–23:00 | Five rehearsals | Present | Observe metrics | Inject/recover failures |
| 23:00–24:00 | Freeze and submit | Submission | Blockers only | Pack and document |

## Hard gates

- Hour 2: camera sees ball; LED/screen output works.
- Hour 6: recorded-video prediction works.
- Hour 8: guaranteed end-to-end chain works three times.
- Hour 12: no new core features.
- Hour 20: code freeze and backup video.

## Scope cuts

- Weak tracking: larger visible ball, simpler background, better light; do not add a new detector.
- No ToF: remove it.
- No servo: use the guaranteed matrix and screen.
- Printer queue: use cardboard or foamboard.
- Claude latency: use it only between throws.
