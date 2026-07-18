from __future__ import annotations

import numpy as np

from preact.landing_calibration import calibration_from_floor_markers


def test_floor_marker_calibration_maps_baseline_and_height_to_metres() -> None:
    calibration = calibration_from_floor_markers(
        (117.0, 364.0),
        (438.0, 382.0),
        distance_m=1.0,
    )

    np.testing.assert_allclose(calibration.pixel_to_world(117.0, 364.0), (0.0, 0.0))
    np.testing.assert_allclose(calibration.pixel_to_world(438.0, 382.0), (1.0, 0.0))
    horizontal = np.asarray((321.0, 18.0))
    horizontal /= np.linalg.norm(horizontal)
    one_metre_up = np.asarray((117.0, 364.0)) + np.asarray(
        (horizontal[1], -horizontal[0])
    ) * np.linalg.norm((321.0, 18.0))
    np.testing.assert_allclose(
        calibration.pixel_to_world(*one_metre_up),
        (0.0, 1.0),
        atol=1e-9,
    )


def test_floor_marker_calibration_rejects_degenerate_inputs() -> None:
    try:
        calibration_from_floor_markers((10.0, 10.0), (10.0, 10.0), distance_m=1.0)
    except ValueError as error:
        assert "different pixel" in str(error)
    else:
        raise AssertionError("expected duplicate floor markers to fail")
