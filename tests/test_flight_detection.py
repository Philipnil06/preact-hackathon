from __future__ import annotations

import cv2
import numpy as np

from preact.flight_detection import FlightBallDetector, FlightBallDetectorConfig


def test_flight_detector_learns_background_then_follows_small_colourful_ball() -> None:
    detector = FlightBallDetector(FlightBallDetectorConfig(background_frames=3))
    background = np.full((240, 320, 3), 35, dtype=np.uint8)

    assert detector.detect(background, timestamp_s=0.00) is None
    assert detector.detect(background, timestamp_s=0.03) is None
    assert detector.detect(background, timestamp_s=0.06) is None

    positions = ((270, 150), (250, 120), (229, 96), (207, 78), (184, 66), (160, 60))
    detections = []
    for index, position in enumerate(positions):
        frame = background.copy()
        cv2.circle(frame, position, 7, (0, 220, 255), -1)
        detections.append(detector.detect(frame, timestamp_s=0.10 + index / 30.0))

    assert detections[0] is None
    assert detections[1] is None
    assert all(item is not None for item in detections[2:])
    assert detections[-1] is not None
    assert abs(detections[-1].x_px - positions[-1][0]) < 1.0
    assert abs(detections[-1].y_px - positions[-1][1]) < 1.0
    assert detector.state == "locked"


def test_flight_detector_does_not_lock_static_colour_flicker() -> None:
    detector = FlightBallDetector(FlightBallDetectorConfig(background_frames=3))
    background = np.full((240, 320, 3), 35, dtype=np.uint8)
    for index in range(3):
        assert detector.detect(background, timestamp_s=index / 30.0) is None

    for index in range(12):
        frame = background.copy()
        cv2.circle(frame, (160 + index % 2, 180), 7, (0, 0, 255), -1)
        assert detector.detect(frame, timestamp_s=0.2 + index / 30.0) is None

    assert detector.state == "acquiring"


def test_clear_track_keeps_the_learned_background_for_the_next_throw() -> None:
    detector = FlightBallDetector(FlightBallDetectorConfig(background_frames=3))
    background = np.full((240, 320, 3), 35, dtype=np.uint8)
    for index in range(3):
        assert detector.detect(background, timestamp_s=index / 30.0) is None

    detector.clear_track()

    assert detector.state == "searching"
    frame = background.copy()
    cv2.circle(frame, (270, 150), 7, (0, 220, 255), -1)
    assert detector.detect(frame, timestamp_s=0.2) is None
    assert detector.state == "acquiring"
