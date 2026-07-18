"""Planar camera calibration for the controlled throwing plane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PlanarCalibration:
    """A homography between image pixels and world-plane metres."""

    pixel_to_world_matrix: np.ndarray
    world_to_pixel_matrix: np.ndarray
    reprojection_rmse_m: float

    @classmethod
    def from_correspondences(
        cls,
        pixel_points: np.ndarray,
        world_points_m: np.ndarray,
        *,
        ransac_threshold_px: float = 3.0,
    ) -> PlanarCalibration:
        pixels = np.asarray(pixel_points, dtype=np.float64)
        world = np.asarray(world_points_m, dtype=np.float64)
        if pixels.shape != world.shape or pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixel_points and world_points_m must both have shape (N, 2)")
        if len(pixels) < 4:
            raise ValueError("at least four point correspondences are required")
        if not np.isfinite(pixels).all() or not np.isfinite(world).all():
            raise ValueError("calibration points must be finite")

        method = cv2.RANSAC if len(pixels) > 4 else 0
        matrix, mask = cv2.findHomography(
            pixels,
            world,
            method=method,
            ransacReprojThreshold=ransac_threshold_px,
        )
        if matrix is None:
            raise ValueError("homography could not be estimated")
        inverse = np.linalg.inv(matrix)

        projected = cv2.perspectiveTransform(
            pixels.reshape(-1, 1, 2),
            matrix,
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - world, axis=1)
        if mask is not None:
            inliers = mask.ravel().astype(bool)
            errors = errors[inliers]
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf")
        return cls(
            pixel_to_world_matrix=matrix,
            world_to_pixel_matrix=inverse,
            reprojection_rmse_m=rmse,
        )

    def pixels_to_world(self, points_px: np.ndarray) -> np.ndarray:
        points = _normalise_points(points_px)
        transformed = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self.pixel_to_world_matrix,
        )
        return transformed.reshape(-1, 2)

    def world_to_pixels(self, points_m: np.ndarray) -> np.ndarray:
        points = _normalise_points(points_m)
        transformed = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self.world_to_pixel_matrix,
        )
        return transformed.reshape(-1, 2)

    def pixel_to_world(self, x_px: float, y_px: float) -> tuple[float, float]:
        point = self.pixels_to_world(np.array([[x_px, y_px]], dtype=np.float64))[0]
        return float(point[0]), float(point[1])

    def world_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        point = self.world_to_pixels(np.array([[x_m, y_m]], dtype=np.float64))[0]
        return float(point[0]), float(point[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_to_world_matrix": self.pixel_to_world_matrix.tolist(),
            "world_to_pixel_matrix": self.world_to_pixel_matrix.tolist(),
            "reprojection_rmse_m": self.reprojection_rmse_m,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlanarCalibration:
        return cls(
            pixel_to_world_matrix=np.asarray(
                value["pixel_to_world_matrix"],
                dtype=np.float64,
            ),
            world_to_pixel_matrix=np.asarray(
                value["world_to_pixel_matrix"],
                dtype=np.float64,
            ),
            reprojection_rmse_m=float(value["reprojection_rmse_m"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> PlanarCalibration:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _normalise_points(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim == 1 and value.shape == (2,):
        value = value.reshape(1, 2)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(value).all():
        raise ValueError("points must be finite")
    return value
