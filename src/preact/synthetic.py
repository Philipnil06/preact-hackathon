"""Synthetic camera and ToF data for tests without activating a real camera."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from preact.calibration import PlanarCalibration
from preact.models import ToFSample


@dataclass(frozen=True, slots=True)
class SyntheticSceneConfig:
    frame_width_px: int = 960
    frame_height_px: int = 540
    world_width_m: float = 3.2
    world_height_m: float = 2.2
    margin_px: int = 45
    frames_per_second: float = 30.0
    pre_roll_frames: int = 12
    gravity_mps2: float = 9.80665
    launch_x_m: float = 0.25
    launch_y_m: float = 1.05
    launch_vx_mps: float = 2.15
    launch_vy_mps: float = 4.6
    ball_radius_px: int = 14
    background_tof_mm: int = 1050
    triggered_tof_mm: int = 720


@dataclass(frozen=True, slots=True)
class SyntheticFrame:
    timestamp_s: float
    frame_bgr: np.ndarray
    tof_sample: ToFSample
    true_position_m: tuple[float, float] | None


class SyntheticThrow:
    def __init__(self, config: SyntheticSceneConfig | None = None) -> None:
        self.config = config or SyntheticSceneConfig()
        self.calibration = self._build_calibration()

    @property
    def expected_landing_x_m(self) -> float:
        config = self.config
        impact_time = (
            config.launch_vy_mps
            + np.sqrt(
                config.launch_vy_mps**2
                + 2.0 * config.gravity_mps2 * config.launch_y_m
            )
        ) / config.gravity_mps2
        return float(config.launch_x_m + config.launch_vx_mps * impact_time)

    def frames(self) -> list[SyntheticFrame]:
        config = self.config
        dt = 1.0 / config.frames_per_second
        impact_time = (
            config.launch_vy_mps
            + np.sqrt(
                config.launch_vy_mps**2
                + 2.0 * config.gravity_mps2 * config.launch_y_m
            )
        ) / config.gravity_mps2
        flight_frames = int(np.ceil(impact_time / dt)) + 1
        output: list[SyntheticFrame] = []

        for index in range(config.pre_roll_frames + flight_frames):
            timestamp_s = index * dt
            device_time_ms = round(timestamp_s * 1000.0)
            if index < config.pre_roll_frames:
                output.append(
                    SyntheticFrame(
                        timestamp_s=timestamp_s,
                        frame_bgr=self._background(),
                        tof_sample=ToFSample(
                            received_at_s=timestamp_s,
                            device_time_ms=device_time_ms,
                            distance_mm=config.background_tof_mm,
                            valid=True,
                        ),
                        true_position_m=None,
                    )
                )
                continue

            flight_index = index - config.pre_roll_frames
            flight_time = flight_index * dt
            x_m = config.launch_x_m + config.launch_vx_mps * flight_time
            y_m = (
                config.launch_y_m
                + config.launch_vy_mps * flight_time
                - 0.5 * config.gravity_mps2 * flight_time**2
            )
            frame = self._background()
            x_px, y_px = self.calibration.world_to_pixel(x_m, y_m)
            if y_m >= -0.05:
                cv2.circle(
                    frame,
                    (round(x_px), round(y_px)),
                    config.ball_radius_px,
                    (0, 140, 255),
                    -1,
                    cv2.LINE_AA,
                )
            tof_distance = (
                config.triggered_tof_mm
                if flight_index in (0, 1)
                else config.background_tof_mm
            )
            output.append(
                SyntheticFrame(
                    timestamp_s=timestamp_s,
                    frame_bgr=frame,
                    tof_sample=ToFSample(
                        received_at_s=timestamp_s,
                        device_time_ms=device_time_ms,
                        distance_mm=tof_distance,
                        valid=True,
                    ),
                    true_position_m=(float(x_m), float(y_m)),
                )
            )
        return output

    def _build_calibration(self) -> PlanarCalibration:
        config = self.config
        left = float(config.margin_px)
        right = float(config.frame_width_px - config.margin_px)
        top = float(config.margin_px)
        bottom = float(config.frame_height_px - config.margin_px)
        pixels = np.array(
            [
                [left, bottom],
                [right, bottom],
                [right, top],
                [left, top],
            ],
            dtype=np.float64,
        )
        world = np.array(
            [
                [0.0, 0.0],
                [config.world_width_m, 0.0],
                [config.world_width_m, config.world_height_m],
                [0.0, config.world_height_m],
            ],
            dtype=np.float64,
        )
        return PlanarCalibration.from_correspondences(pixels, world)

    def _background(self) -> np.ndarray:
        config = self.config
        frame = np.full(
            (config.frame_height_px, config.frame_width_px, 3),
            (28, 30, 34),
            dtype=np.uint8,
        )
        floor_y = config.frame_height_px - config.margin_px
        cv2.line(
            frame,
            (config.margin_px, floor_y),
            (config.frame_width_px - config.margin_px, floor_y),
            (90, 90, 90),
            2,
        )
        return frame
