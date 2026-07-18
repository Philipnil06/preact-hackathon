from __future__ import annotations

import cv2
import numpy as np

from preact.aruco_calibration import (
    ArucoPlaneSpec,
    calibrate_aruco_plane,
    calibrate_plane_from_corners,
)


def test_four_aruco_markers_calibrate_the_throwing_plane() -> None:
    frame = np.full((600, 900, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    positions = {
        0: (100, 450),
        1: (700, 450),
        2: (700, 100),
        3: (100, 100),
    }
    side = 80
    for marker_id, (x, y) in positions.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        frame[y : y + side, x : x + side] = cv2.cvtColor(
            marker,
            cv2.COLOR_GRAY2BGR,
        )

    result = calibrate_aruco_plane(
        frame,
        ArucoPlaneSpec(width_m=3.0, height_m=2.0),
    )

    bottom_left = result.marker_centres_px[0]
    world = result.calibration.pixel_to_world(*bottom_left)
    assert np.allclose(world, (0.0, 0.0), atol=1e-6)
    assert result.calibration.reprojection_rmse_m < 1e-6


def test_manual_corners_are_a_marker_free_fallback() -> None:
    pixels = np.asarray(
        [[100, 500], [800, 500], [800, 100], [100, 100]],
        dtype=np.float64,
    )
    calibration = calibrate_plane_from_corners(
        pixels,
        ArucoPlaneSpec(width_m=3.5, height_m=2.0),
    )

    assert np.allclose(
        calibration.pixel_to_world(800, 100),
        (3.5, 2.0),
        atol=1e-6,
    )
