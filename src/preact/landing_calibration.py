"""Camera-plane calibration from two floor markers a known distance apart."""

from __future__ import annotations

import numpy as np

from preact.calibration import PlanarCalibration


def calibration_from_floor_markers(
    first_marker_px: tuple[float, float],
    second_marker_px: tuple[float, float],
    *,
    distance_m: float,
) -> PlanarCalibration:
    """Build a metric affine plane with +Y perpendicular to the marker baseline.

    This is the controlled-demo fallback used before a four-point homography or
    ToF depth reference is available. The camera must remain fixed and the throw
    must stay in the same plane as the markers.
    """

    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    first = np.asarray(first_marker_px, dtype=np.float64)
    second = np.asarray(second_marker_px, dtype=np.float64)
    axis = second - first
    pixel_distance = float(np.linalg.norm(axis))
    if pixel_distance <= 1e-6:
        raise ValueError("floor markers must occupy different pixel positions")

    horizontal = axis / pixel_distance
    vertical = np.asarray((horizontal[1], -horizontal[0]), dtype=np.float64)
    pixels_per_metre = pixel_distance / distance_m
    linear = np.vstack((horizontal, vertical)) / pixels_per_metre
    translation = -(linear @ first)
    pixel_to_world = np.asarray(
        (
            (linear[0, 0], linear[0, 1], translation[0]),
            (linear[1, 0], linear[1, 1], translation[1]),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return PlanarCalibration(
        pixel_to_world_matrix=pixel_to_world,
        world_to_pixel_matrix=np.linalg.inv(pixel_to_world),
        reprojection_rmse_m=0.0,
    )
