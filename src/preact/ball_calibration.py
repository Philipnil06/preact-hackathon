"""Create a reusable ball-detector profile from a tight sample rectangle."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from preact.detection import BallDetectorConfig, HSVRange


@dataclass(frozen=True, slots=True)
class ColourCalibrationResult:
    config: BallDetectorConfig
    dominant_hue: int
    sampled_pixels: int


def calibrate_ball_colour(
    frame_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    minimum_saturation: int = 40,
    minimum_value: int = 40,
    hue_search_radius: int = 12,
    hue_margin: int = 4,
) -> ColourCalibrationResult:
    """Estimate one or two HSV ranges from a rectangle tightly around the ball.

    OpenCV hue is circular. Red, for example, crosses 179 -> 0 and therefore
    produces two ranges. The detector supports both transparently.
    """

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3)")
    x, y, width, height = roi
    if width <= 0 or height <= 0:
        raise ValueError("colour calibration ROI must have positive dimensions")
    frame_height, frame_width = frame_bgr.shape[:2]
    x0 = int(np.clip(x, 0, frame_width))
    y0 = int(np.clip(y, 0, frame_height))
    x1 = int(np.clip(x + width, 0, frame_width))
    y1 = int(np.clip(y + height, 0, frame_height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("colour calibration ROI does not intersect the frame")

    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    colourful = pixels[
        (pixels[:, 1] >= minimum_saturation)
        & (pixels[:, 2] >= minimum_value)
    ]
    if len(colourful) < 25:
        raise ValueError(
            "not enough colourful pixels; draw a tighter rectangle or use a "
            "more saturated ball"
        )

    hues = colourful[:, 0].astype(np.int16)
    dominant = _dominant_circular_hue(hues, radius=hue_search_radius)
    distances = _signed_hue_distance(hues, dominant)
    core = colourful[np.abs(distances) <= max(18, hue_search_radius + hue_margin)]
    core_distances = _signed_hue_distance(core[:, 0].astype(np.int16), dominant)
    if len(core) < 25:
        raise ValueError("could not isolate a dominant ball colour")

    low_distance = int(np.floor(np.percentile(core_distances, 2))) - hue_margin
    high_distance = int(np.ceil(np.percentile(core_distances, 98))) + hue_margin
    low_distance = max(low_distance, -35)
    high_distance = min(high_distance, 35)

    saturation_low = max(20, int(np.percentile(core[:, 1], 5)) - 25)
    value_low = max(20, int(np.percentile(core[:, 2], 5)) - 25)
    ranges = _circular_ranges(
        dominant + low_distance,
        dominant + high_distance,
        saturation_low=saturation_low,
        value_low=value_low,
    )
    config = BallDetectorConfig(
        hsv=ranges[0],
        additional_hsv_ranges=tuple(ranges[1:]),
        minimum_area_px2=max(40.0, width * height * 0.025),
    )
    return ColourCalibrationResult(
        config=config,
        dominant_hue=dominant,
        sampled_pixels=len(core),
    )


def _dominant_circular_hue(hues: np.ndarray, *, radius: int) -> int:
    counts = np.bincount(hues, minlength=180)
    scores = np.zeros(180, dtype=np.int64)
    for offset in range(-radius, radius + 1):
        scores += np.roll(counts, offset)
    return int(np.argmax(scores))


def _signed_hue_distance(hues: np.ndarray, centre: int) -> np.ndarray:
    return ((hues.astype(np.int16) - centre + 90) % 180) - 90


def _circular_ranges(
    lower_hue: int,
    upper_hue: int,
    *,
    saturation_low: int,
    value_low: int,
) -> list[HSVRange]:
    if lower_hue >= 0 and upper_hue <= 179:
        return [
            HSVRange(
                lower=(lower_hue, saturation_low, value_low),
                upper=(upper_hue, 255, 255),
            )
        ]
    if lower_hue < 0:
        return [
            HSVRange(
                lower=(0, saturation_low, value_low),
                upper=(upper_hue, 255, 255),
            ),
            HSVRange(
                lower=(180 + lower_hue, saturation_low, value_low),
                upper=(179, 255, 255),
            ),
        ]
    return [
        HSVRange(
            lower=(lower_hue, saturation_low, value_low),
            upper=(179, 255, 255),
        ),
        HSVRange(
            lower=(0, saturation_low, value_low),
            upper=(upper_hue - 180, 255, 255),
        ),
    ]
