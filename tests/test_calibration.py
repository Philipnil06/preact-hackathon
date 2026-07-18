from __future__ import annotations

import numpy as np

from preact.calibration import PlanarCalibration


def test_planar_calibration_round_trip(tmp_path) -> None:
    pixel_points = np.array(
        [
            [120.0, 680.0],
            [1160.0, 660.0],
            [1080.0, 120.0],
            [180.0, 140.0],
        ]
    )
    world_points = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 2.0],
            [0.0, 2.0],
        ]
    )
    calibration = PlanarCalibration.from_correspondences(pixel_points, world_points)

    test_world = np.array([[0.4, 0.2], [1.5, 1.0], [2.7, 1.7]])
    pixels = calibration.world_to_pixels(test_world)
    recovered = calibration.pixels_to_world(pixels)

    np.testing.assert_allclose(recovered, test_world, atol=1e-8)
    assert calibration.reprojection_rmse_m < 1e-8

    output = tmp_path / "calibration.json"
    calibration.save(output)
    loaded = PlanarCalibration.load(output)
    np.testing.assert_allclose(
        loaded.pixel_to_world_matrix,
        calibration.pixel_to_world_matrix,
    )


def test_calibration_requires_four_points() -> None:
    points = np.zeros((3, 2))
    try:
        PlanarCalibration.from_correspondences(points, points)
    except ValueError as error:
        assert "four" in str(error)
    else:
        raise AssertionError("expected ValueError")
