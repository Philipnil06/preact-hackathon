from __future__ import annotations

import cv2
import numpy as np

from preact.ball_calibration import calibrate_ball_colour
from preact.detection import BallDetector, BallDetectorConfig


def test_calibrates_and_detects_an_arbitrary_bright_colour() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.circle(frame, (220, 140), 34, (255, 0, 255), -1)

    result = calibrate_ball_colour(frame, (180, 100, 80, 80))
    detection = BallDetector(result.config).detect(frame, timestamp_s=1.0)

    assert detection is not None
    assert abs(detection.x_px - 220) < 2
    assert abs(detection.y_px - 140) < 2


def test_red_hue_wraps_and_profile_round_trips(tmp_path: object) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 30, (0, 0, 255), -1)

    result = calibrate_ball_colour(frame, (125, 85, 70, 70))
    assert len(result.config.hsv_ranges) == 2

    path = tmp_path / "ball.json"  # type: ignore[operator]
    result.config.save(path)
    loaded = BallDetectorConfig.load(path)

    assert loaded == result.config
    assert BallDetector(loaded).detect(frame, timestamp_s=0.0) is not None
