"""Four-marker planar calibration for the controlled throwing plane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preact.calibration import PlanarCalibration


@dataclass(frozen=True, slots=True)
class ArucoPlaneSpec:
    width_m: float = 3.0
    height_m: float = 2.0
    marker_ids: tuple[int, int, int, int] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("plane dimensions must be positive")
        if len(set(self.marker_ids)) != 4:
            raise ValueError("four distinct marker IDs are required")

    @property
    def world_points_m(self) -> dict[int, tuple[float, float]]:
        bottom_left, bottom_right, top_right, top_left = self.marker_ids
        return {
            bottom_left: (0.0, 0.0),
            bottom_right: (self.width_m, 0.0),
            top_right: (self.width_m, self.height_m),
            top_left: (0.0, self.height_m),
        }


@dataclass(frozen=True, slots=True)
class ArucoCalibrationResult:
    calibration: PlanarCalibration
    marker_centres_px: dict[int, tuple[float, float]]


def calibrate_aruco_plane(
    frame_bgr: np.ndarray,
    spec: ArucoPlaneSpec | None = None,
) -> ArucoCalibrationResult:
    plane = spec or ArucoPlaneSpec()
    centres = detect_marker_centres(frame_bgr)
    missing = [marker_id for marker_id in plane.marker_ids if marker_id not in centres]
    if missing:
        raise ValueError(f"missing ArUco markers: {missing}")

    pixel_points = np.asarray(
        [centres[marker_id] for marker_id in plane.marker_ids],
        dtype=np.float64,
    )
    world_lookup = plane.world_points_m
    world_points = np.asarray(
        [world_lookup[marker_id] for marker_id in plane.marker_ids],
        dtype=np.float64,
    )
    calibration = PlanarCalibration.from_correspondences(pixel_points, world_points)
    return ArucoCalibrationResult(
        calibration=calibration,
        marker_centres_px={
            marker_id: centres[marker_id] for marker_id in plane.marker_ids
        },
    )


def calibrate_plane_from_corners(
    pixel_points: np.ndarray,
    spec: ArucoPlaneSpec | None = None,
) -> PlanarCalibration:
    """Manual fallback ordered bottom-left, bottom-right, top-right, top-left."""

    plane = spec or ArucoPlaneSpec()
    pixels = np.asarray(pixel_points, dtype=np.float64)
    if pixels.shape != (4, 2):
        raise ValueError("manual plane calibration requires four pixel points")
    world_lookup = plane.world_points_m
    world = np.asarray(
        [world_lookup[marker_id] for marker_id in plane.marker_ids],
        dtype=np.float64,
    )
    return PlanarCalibration.from_correspondences(pixels, world)


def detect_marker_centres(frame_bgr: np.ndarray) -> dict[int, tuple[float, float]]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3)")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        cv2.aruco.DetectorParameters(),
    )
    corners, identifiers, _ = detector.detectMarkers(frame_bgr)
    if identifiers is None:
        return {}
    centres: dict[int, tuple[float, float]] = {}
    for marker_corners, raw_id in zip(corners, identifiers.ravel(), strict=True):
        centre = np.mean(marker_corners.reshape(4, 2), axis=0)
        centres[int(raw_id)] = (float(centre[0]), float(centre[1]))
    return centres


def generate_calibration_markers(
    output_directory: str | Path,
    *,
    marker_ids: tuple[int, ...] = (0, 1, 2, 3),
    side_pixels: int = 1000,
) -> list[Path]:
    if side_pixels < 100:
        raise ValueError("marker side must be at least 100 pixels")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    paths: list[Path] = []
    for marker_id in marker_ids:
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels)
        path = directory / f"aruco-4x4-50-id-{marker_id}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write marker image: {path}")
        paths.append(path)
    return paths


def draw_calibration_preview(
    frame_bgr: np.ndarray,
    result: ArucoCalibrationResult,
) -> np.ndarray:
    output = frame_bgr.copy()
    ordered = list(result.marker_centres_px.items())
    for marker_id, (x_px, y_px) in ordered:
        point = (round(x_px), round(y_px))
        cv2.circle(output, point, 8, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"ID {marker_id}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output
