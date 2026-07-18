"""Small, fast juggling-ball detection against a fixed demo background."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from preact.calibration import PlanarCalibration
from preact.models import Detection


@dataclass(frozen=True, slots=True)
class FlightBallDetectorConfig:
    background_frames: int = 12
    foreground_threshold: int = 18
    minimum_saturation: int = 55
    minimum_value: int = 35
    minimum_area_px2: float = 10.0
    maximum_area_px2: float = 300.0
    maximum_box_size_px: int = 28
    maximum_aspect_ratio: float = 3.5
    frame_border_margin_px: int = 40
    association_distance_px: float = 50.0
    lock_timeout_s: float = 0.25
    confirmation_frames: int = 3
    minimum_confirmation_displacement_px: float = 12.0
    minimum_height_above_floor_m: float = 0.08


class FlightBallDetector:
    """Detect the rotating multi-colour ball from colour plus frame difference.

    The first few frames after Calculate Landing starts become the clean scene
    reference. This intentionally avoids reusing the balloon's identity lock.
    """

    def __init__(
        self,
        config: FlightBallDetectorConfig | None = None,
        *,
        calibration: PlanarCalibration | None = None,
    ) -> None:
        self.config = config or FlightBallDetectorConfig()
        self.calibration = calibration
        self._background_samples: list[np.ndarray] = []
        self._background: np.ndarray | None = None
        self._last_detection: Detection | None = None
        self._previous_detection: Detection | None = None
        self._pending_detections: list[Detection] = []
        self._state = "learning-background"

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        self._background_samples.clear()
        self._background = None
        self._last_detection = None
        self._previous_detection = None
        self._pending_detections.clear()
        self._state = "learning-background"

    def clear_track(self) -> None:
        """Forget a completed throw while keeping the learned empty scene."""

        self._last_detection = None
        self._previous_detection = None
        self._pending_detections.clear()
        self._state = "searching" if self._background is not None else "learning-background"

    def detect(self, frame_bgr: np.ndarray, *, timestamp_s: float) -> Detection | None:
        if self._background is None:
            self._learn_background(frame_bgr)
            return None

        candidates = self._candidates(frame_bgr, timestamp_s=timestamp_s)
        if self._last_detection is None:
            detection = self._confirm_launch(candidates)
            if detection is None:
                return None
            self._last_detection = detection
            self._state = "locked"
            return detection
        detection = self._select(candidates, timestamp_s=timestamp_s)
        if detection is None:
            self._state = "lost" if self._last_detection is not None else "searching"
            if (
                self._last_detection is not None
                and timestamp_s - self._last_detection.timestamp_s > self.config.lock_timeout_s
            ):
                self._last_detection = None
                self._previous_detection = None
                self._pending_detections.clear()
            return None

        self._previous_detection = self._last_detection
        self._last_detection = detection
        self._state = "locked"
        return detection

    def _learn_background(self, frame_bgr: np.ndarray) -> None:
        self._background_samples.append(frame_bgr.copy())
        if len(self._background_samples) < self.config.background_frames:
            return
        stack = np.stack(self._background_samples, axis=0)
        self._background = np.median(stack, axis=0).astype(np.uint8)
        self._background_samples.clear()
        self._state = "searching"

    def _candidates(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_s: float,
    ) -> list[Detection]:
        assert self._background is not None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        colourful = cv2.inRange(
            hsv,
            (0, self.config.minimum_saturation, self.config.minimum_value),
            (179, 255, 255),
        )
        difference = cv2.cvtColor(
            cv2.absdiff(frame_bgr, self._background),
            cv2.COLOR_BGR2GRAY,
        )
        foreground = cv2.threshold(
            difference,
            self.config.foreground_threshold,
            255,
            cv2.THRESH_BINARY,
        )[1]
        mask = cv2.bitwise_and(colourful, foreground)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output: list[Detection] = []
        frame_height, frame_width = frame_bgr.shape[:2]
        margin = self.config.frame_border_margin_px
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            smaller = max(1, min(width, height))
            aspect = max(width, height) / smaller
            if not (
                self.config.minimum_area_px2 <= area <= self.config.maximum_area_px2
                and 2 <= width <= self.config.maximum_box_size_px
                and 2 <= height <= self.config.maximum_box_size_px
                and aspect <= self.config.maximum_aspect_ratio
                and x >= margin
                and x + width <= frame_width - margin
                and y >= margin
                and y + height <= frame_height - margin
            ):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0.0:
                continue
            x_px = float(moments["m10"] / moments["m00"])
            y_px = float(moments["m01"] / moments["m00"])
            if self.calibration is not None:
                _, world_y = self.calibration.pixel_to_world(x_px, y_px)
                if world_y < self.config.minimum_height_above_floor_m:
                    continue
            fill = area / max(float(width * height), 1.0)
            size_score = min(1.0, area / 70.0)
            quality = float(np.clip(0.35 + 0.35 * fill + 0.30 * size_score, 0.0, 1.0))
            output.append(
                Detection(
                    timestamp_s=timestamp_s,
                    x_px=x_px,
                    y_px=y_px,
                    radius_px=0.5 * float(np.hypot(width, height)),
                    area_px2=area,
                    quality=quality,
                    bbox_x_px=float(x),
                    bbox_y_px=float(y),
                    bbox_width_px=float(width),
                    bbox_height_px=float(height),
                )
            )
        return output

    def _confirm_launch(self, candidates: list[Detection]) -> Detection | None:
        if not candidates:
            self._pending_detections.clear()
            self._state = "searching"
            return None
        if not self._pending_detections:
            chosen = max(candidates, key=lambda item: item.quality)
            self._pending_detections.append(chosen)
            self._state = "acquiring"
            return None

        previous = self._pending_detections[-1]
        chosen = min(
            candidates,
            key=lambda item: float(
                np.hypot(item.x_px - previous.x_px, item.y_px - previous.y_px)
            ),
        )
        distance = float(
            np.hypot(chosen.x_px - previous.x_px, chosen.y_px - previous.y_px)
        )
        if distance > self.config.association_distance_px:
            self._pending_detections = [max(candidates, key=lambda item: item.quality)]
            self._state = "acquiring"
            return None
        self._pending_detections.append(chosen)
        maximum_pending = max(self.config.confirmation_frames, 8)
        self._pending_detections = self._pending_detections[-maximum_pending:]
        if len(self._pending_detections) < self.config.confirmation_frames:
            self._state = "acquiring"
            return None
        first = self._pending_detections[0]
        displacement = float(np.hypot(chosen.x_px - first.x_px, chosen.y_px - first.y_px))
        if displacement < self.config.minimum_confirmation_displacement_px:
            self._state = "acquiring"
            return None
        if len(self._pending_detections) >= 2:
            self._previous_detection = self._pending_detections[-2]
        self._pending_detections.clear()
        return chosen

    def _select(
        self,
        candidates: list[Detection],
        *,
        timestamp_s: float,
    ) -> Detection | None:
        if not candidates:
            return None
        last = self._last_detection
        if last is None or timestamp_s - last.timestamp_s > self.config.lock_timeout_s:
            return max(candidates, key=lambda item: item.quality)

        predicted_x = last.x_px
        predicted_y = last.y_px
        previous = self._previous_detection
        if previous is not None and last.timestamp_s > previous.timestamp_s:
            scale = (timestamp_s - last.timestamp_s) / (last.timestamp_s - previous.timestamp_s)
            predicted_x += (last.x_px - previous.x_px) * scale
            predicted_y += (last.y_px - previous.y_px) * scale
        chosen = min(
            candidates,
            key=lambda item: float(np.hypot(item.x_px - predicted_x, item.y_px - predicted_y)),
        )
        distance = float(np.hypot(chosen.x_px - predicted_x, chosen.y_px - predicted_y))
        return chosen if distance <= self.config.association_distance_px else None
