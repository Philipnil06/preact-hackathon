from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from preact.__main__ import _build_synthetic_runtime
from preact.models import Prediction
from preact.runtime import _draw_landing_marker
from preact.session_log import SessionLogger
from preact.synthetic import SyntheticThrow


def test_synthetic_runtime_writes_machine_readable_session_log(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    runtime = _build_synthetic_runtime()
    runtime.session_logger = SessionLogger(path)

    runtime.play()
    deadline = 80
    while runtime.status().observations == 0 and deadline > 0:
        import time

        time.sleep(0.02)
        deadline -= 1
    runtime.stop()
    runtime.close()

    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert values[0]["event"] == "session_start"
    frames = [value for value in values if value["event"] == "frame"]
    assert frames
    assert "prediction" in frames[-1]
    assert "tof" in frames[-1]


def test_landing_marker_is_drawn_at_the_calibrated_floor_intersection() -> None:
    scene = SyntheticThrow()
    image = np.zeros((scene.config.frame_height_px, scene.config.frame_width_px, 3), dtype=np.uint8)
    landing_x = 1.4
    prediction = Prediction(
        valid=True,
        reason="ok",
        generated_at_s=0.0,
        landing_x_m=landing_x,
        time_to_impact_s=0.3,
        confidence=0.9,
        fit=None,
    )

    _draw_landing_marker(image, prediction, scene.calibration)

    x_px, y_px = scene.calibration.world_to_pixel(landing_x, 0.0)
    pixel = image[round(y_px), round(x_px)]
    assert int(pixel[1]) > 200
