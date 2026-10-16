# Development

Install from the repository root with Python 3.12 and an active environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m preact.trajectory_validation --scenarios 300
```

Tests use synthetic observations, generated images, mock cloud clients,
and serial emulators. They do not require API credentials or physical devices.
Optional face-model assets are not part of the default test setup.

## Layout

| Directory | Purpose |
| --- | --- |
| `src/preact/` | Vision, geometry, runtime, serial adapters, voice, and local web UI |
| `tests/` | Unit and integration coverage for the Python runtime |
| `config/` | Example detector and calibration profiles; not portable measurements |
| `firmware/` | JSON-protocol ToF, LEDs, and planar-pointer sketch |
| `servo/` | Alternate text-protocol pan/tilt sketch and hardware notes |
| `browser/` | Standalone Web Serial servo panel |
| `docs/` | Architecture, setup, validation, and historical hackathon notes |
| `work/` | Ignored local calibration, diagnostics, and temporary assets |

## Verification helper

`scripts/verify.ps1` runs the Python checks. Add `-IncludeFirmware` to also
compile the JSON firmware, provided Arduino CLI, the Genesis Mini board
definition, and its libraries are installed. The pan/tilt sketch is a separate
firmware target and is not covered by that compilation step.

GitHub Actions defines Python checks on Windows and Linux, trajectory
validation, and a wheel-resource smoke check. Local results are documented in
[Validation](VALIDATION.md); a workflow definition alone is not a passing run.

## Packaging

`pyproject.toml` is the dependency source of truth. `requirements.txt` delegates
to it. Build a wheel with:

```bash
python -m pip wheel --no-deps --wheel-dir work/dist .
```

The wheel includes the local UI HTML. Camera, mount, and detector configuration
files remain repository assets, so run camera/hardware commands from a clone
with the appropriate calibration paths. The synthetic entry point does not
require those configuration files.

## Working on the hardware

Keep the JSON and pan/tilt firmware/protocol paths distinct. Start with the
synthetic demo, then use camera/emulator mode before a physical run. Recalibrate
after camera or mount changes and record predicted/actual contact points for
physical evaluation. The runtime can append per-throw results with
`--landing-results work/landing-results.jsonl`; frame telemetry uses `--log`.

Personal face images/profiles, local model files, environment files, recordings,
and working artifacts are ignored. Do not add an open-source license without
agreement from the project owners.
