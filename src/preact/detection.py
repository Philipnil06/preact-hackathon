"""Colour-based foam-ball detection for controlled hackathon lighting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from preact.models import Detection


@dataclass(frozen=True, slots=True)
class HSVRange:
    lower: tuple[int, int, int] = (4, 100, 90)
    upper: tuple[int, int, int] = (28, 255, 255)

    def __post_init__(self) -> None:
        if any(low > high for low, high in zip(self.lower, self.upper, strict=True)):
            raise ValueError("HSV lower bounds must not exceed upper bounds")
        if not 0 <= self.lower[0] <= 179 or not 0 <= self.upper[0] <= 179:
            raise ValueError("OpenCV hue bounds must be between 0 and 179")
        for value in (*self.lower[1:], *self.upper[1:]):
            if not 0 <= value <= 255:
                raise ValueError("saturation and value bounds must be between 0 and 255")

    def to_dict(self) -> dict[str, list[int]]:
        return {"lower": list(self.lower), "upper": list(self.upper)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HSVRange:
        lower = tuple(int(item) for item in value["lower"])
        upper = tuple(int(item) for item in value["upper"])
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("HSV bounds must each contain three integers")
        return cls(lower=lower, upper=upper)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BallDetectorConfig:
    hsv: HSVRange = HSVRange()
    additional_hsv_ranges: tuple[HSVRange, ...] = ()
    minimum_area_px2: float = 120.0
    maximum_area_px2: float = 80_000.0
    minimum_circularity: float = 0.45
    morphology_kernel_size: int = 5
    colour_grouping_kernel_size: int = 1
    minimum_hsv_range_count: int = 1
    minimum_hsv_range_fraction: float = 0.0
    region_of_interest: tuple[int, int, int, int] | None = None
    minimum_mean_saturation: float = 0.0
    minimum_mean_value: float = 0.0
    dominant_channel: Literal["blue", "green", "red"] | None = None
    minimum_channel_excess: float = 0.0
    minimum_aspect_ratio: float = 1.0
    maximum_aspect_ratio: float = 1000.0
    minimum_rectangularity: float = 0.0
    association_max_distance_px: float = 0.0
    lock_timeout_s: float = 0.45
    reacquire_confirmation_frames: int = 1
    identity_lock_enabled: bool = False
    commissioning_min_bbox_height_px: float = 0.0
    commissioning_min_vertical_ratio: float = 0.0
    identity_reacquire_min_vertical_ratio: float = 0.0
    maximum_appearance_distance: float = 0.38
    maximum_tracker_appearance_distance: float = 0.52
    appearance_update_rate: float = 0.05
    commissioning_min_orb_features: int = 0
    minimum_orb_identity_matches: int = 0
    maximum_orb_match_distance: float = 64.0
    maximum_identity_area_ratio: float = 2.5
    maximum_identity_aspect_ratio_change: float = 2.0
    visual_tracker: Literal["none", "csrt", "kcf"] = "none"
    reference_image_path: str | None = None
    reference_image_bbox: tuple[int, int, int, int] | None = None
    live_bootstrap_frames: int = 0
    live_bootstrap_update_rate: float = 0.25
    prediction_coast_s: float = 0.0
    velocity_update_rate: float = 0.45

    def __post_init__(self) -> None:
        if self.minimum_area_px2 <= 0:
            raise ValueError("minimum_area_px2 must be positive")
        if self.maximum_area_px2 < self.minimum_area_px2:
            raise ValueError("maximum_area_px2 must not be smaller than minimum_area_px2")
        if not 0.0 <= self.minimum_circularity <= 1.0:
            raise ValueError("minimum_circularity must be in the range 0..1")
        if not 1 <= self.minimum_hsv_range_count <= len(self.hsv_ranges):
            raise ValueError(
                "minimum_hsv_range_count must be between 1 and the number of HSV ranges"
            )
        if not 0.0 <= self.minimum_hsv_range_fraction <= 1.0:
            raise ValueError("minimum_hsv_range_fraction must be in the range 0..1")
        if self.colour_grouping_kernel_size < 1:
            raise ValueError("colour_grouping_kernel_size must be at least 1")
        if not 0.0 <= self.minimum_mean_saturation <= 255.0:
            raise ValueError("minimum_mean_saturation must be in the range 0..255")
        if not 0.0 <= self.minimum_mean_value <= 255.0:
            raise ValueError("minimum_mean_value must be in the range 0..255")
        if self.dominant_channel not in (None, "blue", "green", "red"):
            raise ValueError("dominant_channel must be blue, green, red, or null")
        if self.minimum_channel_excess < 0.0:
            raise ValueError("minimum_channel_excess must not be negative")
        if self.minimum_aspect_ratio < 1.0:
            raise ValueError("minimum_aspect_ratio must be at least 1")
        if self.maximum_aspect_ratio < self.minimum_aspect_ratio:
            raise ValueError(
                "maximum_aspect_ratio must not be smaller than minimum_aspect_ratio"
            )
        if not 0.0 <= self.minimum_rectangularity <= 1.0:
            raise ValueError("minimum_rectangularity must be in the range 0..1")
        if self.association_max_distance_px < 0.0:
            raise ValueError("association_max_distance_px must not be negative")
        if self.lock_timeout_s <= 0.0:
            raise ValueError("lock_timeout_s must be positive")
        if self.reacquire_confirmation_frames < 1:
            raise ValueError("reacquire_confirmation_frames must be at least 1")
        if self.commissioning_min_bbox_height_px < 0.0:
            raise ValueError("commissioning_min_bbox_height_px must not be negative")
        if self.commissioning_min_vertical_ratio < 0.0:
            raise ValueError("commissioning_min_vertical_ratio must not be negative")
        if self.identity_reacquire_min_vertical_ratio < 0.0:
            raise ValueError("identity_reacquire_min_vertical_ratio must not be negative")
        if not 0.0 <= self.maximum_appearance_distance <= 1.0:
            raise ValueError("maximum_appearance_distance must be in the range 0..1")
        if not 0.0 <= self.maximum_tracker_appearance_distance <= 1.0:
            raise ValueError(
                "maximum_tracker_appearance_distance must be in the range 0..1"
            )
        if not 0.0 <= self.appearance_update_rate <= 1.0:
            raise ValueError("appearance_update_rate must be in the range 0..1")
        if self.commissioning_min_orb_features < 0:
            raise ValueError("commissioning_min_orb_features must not be negative")
        if self.minimum_orb_identity_matches < 0:
            raise ValueError("minimum_orb_identity_matches must not be negative")
        if self.maximum_orb_match_distance <= 0.0:
            raise ValueError("maximum_orb_match_distance must be positive")
        if self.maximum_identity_area_ratio < 1.0:
            raise ValueError("maximum_identity_area_ratio must be at least 1")
        if self.maximum_identity_aspect_ratio_change < 1.0:
            raise ValueError("maximum_identity_aspect_ratio_change must be at least 1")
        if self.visual_tracker not in ("none", "csrt", "kcf"):
            raise ValueError("visual_tracker must be none, csrt, or kcf")
        if self.reference_image_bbox is not None:
            if self.reference_image_path is None:
                raise ValueError("reference_image_bbox requires reference_image_path")
            _, _, width, height = self.reference_image_bbox
            if width <= 0 or height <= 0:
                raise ValueError("reference_image_bbox width and height must be positive")
        if self.live_bootstrap_frames < 0:
            raise ValueError("live_bootstrap_frames must not be negative")
        if not 0.0 <= self.live_bootstrap_update_rate <= 1.0:
            raise ValueError("live_bootstrap_update_rate must be in the range 0..1")
        if self.prediction_coast_s < 0.0:
            raise ValueError("prediction_coast_s must not be negative")
        if not 0.0 <= self.velocity_update_rate <= 1.0:
            raise ValueError("velocity_update_rate must be in the range 0..1")

    @property
    def hsv_ranges(self) -> tuple[HSVRange, ...]:
        return (self.hsv, *self.additional_hsv_ranges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hsv_ranges": [value.to_dict() for value in self.hsv_ranges],
            "minimum_area_px2": self.minimum_area_px2,
            "maximum_area_px2": self.maximum_area_px2,
            "minimum_circularity": self.minimum_circularity,
            "morphology_kernel_size": self.morphology_kernel_size,
            "colour_grouping_kernel_size": self.colour_grouping_kernel_size,
            "minimum_hsv_range_count": self.minimum_hsv_range_count,
            "minimum_hsv_range_fraction": self.minimum_hsv_range_fraction,
            "region_of_interest": (
                list(self.region_of_interest)
                if self.region_of_interest is not None
                else None
            ),
            "minimum_mean_saturation": self.minimum_mean_saturation,
            "minimum_mean_value": self.minimum_mean_value,
            "dominant_channel": self.dominant_channel,
            "minimum_channel_excess": self.minimum_channel_excess,
            "minimum_aspect_ratio": self.minimum_aspect_ratio,
            "maximum_aspect_ratio": self.maximum_aspect_ratio,
            "minimum_rectangularity": self.minimum_rectangularity,
            "association_max_distance_px": self.association_max_distance_px,
            "lock_timeout_s": self.lock_timeout_s,
            "reacquire_confirmation_frames": self.reacquire_confirmation_frames,
            "identity_lock_enabled": self.identity_lock_enabled,
            "commissioning_min_bbox_height_px": self.commissioning_min_bbox_height_px,
            "commissioning_min_vertical_ratio": self.commissioning_min_vertical_ratio,
            "identity_reacquire_min_vertical_ratio": (
                self.identity_reacquire_min_vertical_ratio
            ),
            "maximum_appearance_distance": self.maximum_appearance_distance,
            "maximum_tracker_appearance_distance": (
                self.maximum_tracker_appearance_distance
            ),
            "appearance_update_rate": self.appearance_update_rate,
            "commissioning_min_orb_features": self.commissioning_min_orb_features,
            "minimum_orb_identity_matches": self.minimum_orb_identity_matches,
            "maximum_orb_match_distance": self.maximum_orb_match_distance,
            "maximum_identity_area_ratio": self.maximum_identity_area_ratio,
            "maximum_identity_aspect_ratio_change": (
                self.maximum_identity_aspect_ratio_change
            ),
            "visual_tracker": self.visual_tracker,
            "reference_image_path": self.reference_image_path,
            "reference_image_bbox": (
                list(self.reference_image_bbox)
                if self.reference_image_bbox is not None
                else None
            ),
            "live_bootstrap_frames": self.live_bootstrap_frames,
            "live_bootstrap_update_rate": self.live_bootstrap_update_rate,
            "prediction_coast_s": self.prediction_coast_s,
            "velocity_update_rate": self.velocity_update_rate,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BallDetectorConfig:
        raw_ranges = value.get("hsv_ranges")
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise ValueError("detector profile must contain at least one HSV range")
        ranges = tuple(HSVRange.from_dict(item) for item in raw_ranges)
        raw_roi = value.get("region_of_interest")
        roi = tuple(int(item) for item in raw_roi) if raw_roi is not None else None
        if roi is not None and len(roi) != 4:
            raise ValueError("region_of_interest must contain four integers")
        raw_reference_bbox = value.get("reference_image_bbox")
        reference_bbox = (
            tuple(int(item) for item in raw_reference_bbox)
            if raw_reference_bbox is not None
            else None
        )
        if reference_bbox is not None and len(reference_bbox) != 4:
            raise ValueError("reference_image_bbox must contain four integers")
        return cls(
            hsv=ranges[0],
            additional_hsv_ranges=ranges[1:],
            minimum_area_px2=float(value.get("minimum_area_px2", 120.0)),
            maximum_area_px2=float(value.get("maximum_area_px2", 80_000.0)),
            minimum_circularity=float(value.get("minimum_circularity", 0.45)),
            morphology_kernel_size=int(value.get("morphology_kernel_size", 5)),
            colour_grouping_kernel_size=int(
                value.get("colour_grouping_kernel_size", 1)
            ),
            minimum_hsv_range_count=int(value.get("minimum_hsv_range_count", 1)),
            minimum_hsv_range_fraction=float(
                value.get("minimum_hsv_range_fraction", 0.0)
            ),
            region_of_interest=roi,  # type: ignore[arg-type]
            minimum_mean_saturation=float(value.get("minimum_mean_saturation", 0.0)),
            minimum_mean_value=float(value.get("minimum_mean_value", 0.0)),
            dominant_channel=value.get("dominant_channel"),
            minimum_channel_excess=float(value.get("minimum_channel_excess", 0.0)),
            minimum_aspect_ratio=float(value.get("minimum_aspect_ratio", 1.0)),
            maximum_aspect_ratio=float(value.get("maximum_aspect_ratio", 1000.0)),
            minimum_rectangularity=float(value.get("minimum_rectangularity", 0.0)),
            association_max_distance_px=float(
                value.get("association_max_distance_px", 0.0)
            ),
            lock_timeout_s=float(value.get("lock_timeout_s", 0.45)),
            reacquire_confirmation_frames=int(
                value.get("reacquire_confirmation_frames", 1)
            ),
            identity_lock_enabled=bool(value.get("identity_lock_enabled", False)),
            commissioning_min_bbox_height_px=float(
                value.get("commissioning_min_bbox_height_px", 0.0)
            ),
            commissioning_min_vertical_ratio=float(
                value.get("commissioning_min_vertical_ratio", 0.0)
            ),
            identity_reacquire_min_vertical_ratio=float(
                value.get("identity_reacquire_min_vertical_ratio", 0.0)
            ),
            maximum_appearance_distance=float(
                value.get("maximum_appearance_distance", 0.38)
            ),
            maximum_tracker_appearance_distance=float(
                value.get("maximum_tracker_appearance_distance", 0.52)
            ),
            appearance_update_rate=float(value.get("appearance_update_rate", 0.05)),
            commissioning_min_orb_features=int(
                value.get("commissioning_min_orb_features", 0)
            ),
            minimum_orb_identity_matches=int(
                value.get("minimum_orb_identity_matches", 0)
            ),
            maximum_orb_match_distance=float(
                value.get("maximum_orb_match_distance", 64.0)
            ),
            maximum_identity_area_ratio=float(
                value.get("maximum_identity_area_ratio", 2.5)
            ),
            maximum_identity_aspect_ratio_change=float(
                value.get("maximum_identity_aspect_ratio_change", 2.0)
            ),
            visual_tracker=value.get("visual_tracker", "none"),
            reference_image_path=value.get("reference_image_path"),
            reference_image_bbox=reference_bbox,  # type: ignore[arg-type]
            live_bootstrap_frames=int(value.get("live_bootstrap_frames", 0)),
            live_bootstrap_update_rate=float(
                value.get("live_bootstrap_update_rate", 0.25)
            ),
            prediction_coast_s=float(value.get("prediction_coast_s", 0.0)),
            velocity_update_rate=float(value.get("velocity_update_rate", 0.45)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BallDetectorConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("detector profile must be a JSON object")
        return cls.from_dict(value)


@dataclass(slots=True)
class _Candidate:
    score: float
    detection: Detection
    appearance_histogram: np.ndarray
    orb_descriptors: np.ndarray | None
    rotated_aspect_ratio: float
    from_visual_tracker: bool = False


@dataclass(slots=True)
class _TargetSignature:
    appearance_histogram: np.ndarray
    orb_descriptors: np.ndarray | None
    area_px2: float
    rotated_aspect_ratio: float


class BallDetector:
    def __init__(self, config: BallDetectorConfig | None = None) -> None:
        self.config = config or BallDetectorConfig()
        self._last_detection: Detection | None = None
        self._pending_detection: Detection | None = None
        self._pending_identity_candidate: _Candidate | None = None
        self._pending_count = 0
        self._reference_signature = self._load_reference_signature()
        self._target_signature = self._copy_reference_signature()
        self._live_bootstrap_remaining = (
            self.config.live_bootstrap_frames
            if self._reference_signature is not None
            else 0
        )
        self._last_observed_detection: Detection | None = None
        self._velocity_x_px_s = 0.0
        self._velocity_y_px_s = 0.0
        self._visual_tracker: Any | None = None
        self._frames_since_tracker_initialisation = 0
        self._state = (
            "searching"
            if self._target_signature is not None or not self.config.identity_lock_enabled
            else "acquiring"
        )

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        """Clear temporal association without changing the colour profile."""

        self._last_detection = None
        self._pending_detection = None
        self._pending_identity_candidate = None
        self._pending_count = 0
        self._target_signature = self._copy_reference_signature()
        self._live_bootstrap_remaining = (
            self.config.live_bootstrap_frames
            if self._reference_signature is not None
            else 0
        )
        self._last_observed_detection = None
        self._velocity_x_px_s = 0.0
        self._velocity_y_px_s = 0.0
        self._visual_tracker = None
        self._frames_since_tracker_initialisation = 0
        self._state = (
            "searching"
            if self._target_signature is not None or not self.config.identity_lock_enabled
            else "acquiring"
        )

    def _load_reference_signature(self) -> _TargetSignature | None:
        path = self.config.reference_image_path
        if path is None:
            return None
        frame = cv2.imread(str(Path(path)))
        if frame is None:
            raise ValueError(f"could not read detector reference image: {path}")
        height, width = frame.shape[:2]
        if self.config.reference_image_bbox is None:
            box = (0, 0, width, height)
        else:
            x, y, box_width, box_height = self.config.reference_image_bbox
            x0 = int(np.clip(x, 0, width - 1))
            y0 = int(np.clip(y, 0, height - 1))
            x1 = int(np.clip(x + box_width, x0 + 1, width))
            y1 = int(np.clip(y + box_height, y0 + 1, height))
            box = (x0, y0, x1, y1)
        histogram = _appearance_histogram(frame, box)
        if float(histogram.sum()) <= 0.0:
            raise ValueError(f"detector reference image contains no usable colour: {path}")
        return _TargetSignature(
            appearance_histogram=histogram,
            orb_descriptors=(
                _orb_descriptors(frame, box) if self._needs_orb else None
            ),
            area_px2=0.0,
            rotated_aspect_ratio=0.0,
        )

    def _copy_reference_signature(self) -> _TargetSignature | None:
        reference = self._reference_signature
        if reference is None:
            return None
        return _TargetSignature(
            appearance_histogram=reference.appearance_histogram.copy(),
            orb_descriptors=(
                None
                if reference.orb_descriptors is None
                else reference.orb_descriptors.copy()
            ),
            area_px2=reference.area_px2,
            rotated_aspect_ratio=reference.rotated_aspect_ratio,
        )

    @property
    def _needs_orb(self) -> bool:
        return (
            self.config.commissioning_min_orb_features > 0
            or self.config.minimum_orb_identity_matches > 0
        )

    def detect(self, frame_bgr: np.ndarray, *, timestamp_s: float) -> Detection | None:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3)")

        roi, offset_x, offset_y = self._crop(frame_bgr)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._mask_from_hsv(hsv)
        kernel_size = max(1, int(self.config.morphology_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        grouping_size = max(1, int(self.config.colour_grouping_kernel_size))
        if grouping_size % 2 == 0:
            grouping_size += 1
        if grouping_size > 1:
            grouping_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (grouping_size, grouping_size),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, grouping_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[_Candidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.config.minimum_area_px2 <= area <= self.config.maximum_area_px2:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = float(4.0 * pi * area / (perimeter * perimeter))
            if circularity < self.config.minimum_circularity:
                continue
            (_, _), (rectangle_width, rectangle_height), _ = cv2.minAreaRect(contour)
            short_side = min(rectangle_width, rectangle_height)
            long_side = max(rectangle_width, rectangle_height)
            if short_side <= 1.0:
                continue
            aspect_ratio = long_side / short_side
            if not (
                self.config.minimum_aspect_ratio
                <= aspect_ratio
                <= self.config.maximum_aspect_ratio
            ):
                continue
            rectangle_area = rectangle_width * rectangle_height
            rectangularity = area / max(rectangle_area, 1.0)
            if rectangularity < self.config.minimum_rectangularity:
                continue

            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            minimum_band_pixels = max(
                1,
                round(area * self.config.minimum_hsv_range_fraction),
            )
            visible_band_count = sum(
                cv2.countNonZero(
                    cv2.bitwise_and(
                        cv2.inRange(
                            hsv,
                            np.asarray(value.lower, dtype=np.uint8),
                            np.asarray(value.upper, dtype=np.uint8),
                        ),
                        contour_mask,
                    )
                )
                >= minimum_band_pixels
                for value in self.config.hsv_ranges
            )
            if visible_band_count < self.config.minimum_hsv_range_count:
                continue
            mean_hsv = cv2.mean(hsv, mask=contour_mask)
            mean_saturation = float(mean_hsv[1])
            if mean_saturation < self.config.minimum_mean_saturation:
                continue
            if float(mean_hsv[2]) < self.config.minimum_mean_value:
                continue
            if self.config.dominant_channel is not None:
                mean_bgr = cv2.mean(roi, mask=contour_mask)[:3]
                channel_index = {"blue": 0, "green": 1, "red": 2}[
                    self.config.dominant_channel
                ]
                dominant_value = mean_bgr[channel_index]
                other_value = max(
                    value for index, value in enumerate(mean_bgr) if index != channel_index
                )
                if dominant_value - other_value < self.config.minimum_channel_excess:
                    continue
            (x_px, y_px), radius_px = cv2.minEnclosingCircle(contour)
            if radius_px <= 1.0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-9:
                x_px = float(moments["m10"] / moments["m00"])
                y_px = float(moments["m01"] / moments["m00"])
            box_x, box_y, box_width, box_height = cv2.boundingRect(contour)

            circle_area = pi * radius_px * radius_px
            fill_ratio = min(1.0, area / max(circle_area, 1.0))
            circularity_score = np.clip(
                (circularity - self.config.minimum_circularity)
                / max(1.0 - self.config.minimum_circularity, 1e-6),
                0.0,
                1.0,
            )
            area_score = np.clip(
                area / max(self.config.minimum_area_px2 * 8.0, 1.0),
                0.0,
                1.0,
            )
            quality = float(
                np.clip(
                    0.50 * circularity_score + 0.30 * fill_ratio + 0.20 * area_score,
                    0.05,
                    1.0,
                )
            )
            detection = Detection(
                timestamp_s=float(timestamp_s),
                x_px=float(x_px + offset_x),
                y_px=float(y_px + offset_y),
                radius_px=float(radius_px),
                area_px2=area,
                quality=quality,
                bbox_x_px=float(box_x + offset_x),
                bbox_y_px=float(box_y + offset_y),
                bbox_width_px=float(box_width),
                bbox_height_px=float(box_height),
            )
            identity_box = detection_box(detection, frame_bgr.shape, padding_px=0)
            candidates.append(
                _Candidate(
                    score=quality * sqrt_area(area),
                    detection=detection,
                    appearance_histogram=_appearance_histogram(
                        frame_bgr,
                        identity_box,
                    ),
                    orb_descriptors=(
                        _orb_descriptors(frame_bgr, identity_box)
                        if self._needs_orb
                        else None
                    ),
                    rotated_aspect_ratio=float(aspect_ratio),
                )
            )

        if self.config.identity_lock_enabled:
            return self._select_identity_aware(
                frame_bgr,
                candidates,
                timestamp_s=float(timestamp_s),
            )
        selected = self._select_temporally(
            [(candidate.score, candidate.detection) for candidate in candidates],
            timestamp_s=float(timestamp_s),
        )
        if selected is not None:
            self._state = "locked"
        elif self._last_detection is not None:
            self._state = "lost"
        else:
            self._state = "searching"
        return selected

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return the raw HSV mask for calibration/debugging."""

        roi, _, _ = self._crop(frame_bgr)
        return self._colour_mask(roi)

    def _colour_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        return self._mask_from_hsv(hsv)

    def _mask_from_hsv(self, hsv: np.ndarray) -> np.ndarray:
        result = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for value in self.config.hsv_ranges:
            current = cv2.inRange(
                hsv,
                np.asarray(value.lower, dtype=np.uint8),
                np.asarray(value.upper, dtype=np.uint8),
            )
            result = cv2.bitwise_or(result, current)
        return result

    def _select_identity_aware(
        self,
        frame_bgr: np.ndarray,
        candidates: list[_Candidate],
        *,
        timestamp_s: float,
    ) -> Detection | None:
        if self._target_signature is None:
            return self._commission_identity(frame_bgr, candidates)

        tracked = self._update_visual_tracker(frame_bgr, timestamp_s=timestamp_s)
        motion_prediction = self._motion_prediction(timestamp_s)
        reference = (
            tracked.detection
            if tracked is not None
            else motion_prediction or self._last_detection
        )
        matched: list[tuple[float, float, _Candidate]] = []
        for candidate in candidates:
            appearance_distance = self._identity_distance(candidate)
            if appearance_distance is None:
                continue
            spatial_distance = (
                0.0
                if reference is None
                else float(
                    np.hypot(
                        candidate.detection.x_px - reference.x_px,
                        candidate.detection.y_px - reference.y_px,
                    )
                )
            )
            last = self._last_detection
            lock_is_recent = (
                last is not None
                and timestamp_s - last.timestamp_s <= self.config.lock_timeout_s
            )
            allowance = self.config.association_max_distance_px
            if reference is not None:
                allowance += 0.25 * (
                    candidate.detection.radius_px + reference.radius_px
                )
            if lock_is_recent and spatial_distance > allowance:
                continue
            matched.append((appearance_distance, spatial_distance, candidate))

        if matched:
            _, _, chosen = min(matched, key=lambda item: (item[0], item[1]))
            if (
                tracked is None
                or self._frames_since_tracker_initialisation >= 30
                or _detection_distance(chosen.detection, tracked.detection)
                > self.config.association_max_distance_px * 0.5
            ):
                self._initialise_visual_tracker(frame_bgr, chosen.detection)
            self._update_target_signature(chosen)
            self._accept(chosen.detection)
            self._state = "locked"
            return chosen.detection

        if tracked is not None and self._identity_distance(tracked) is not None:
            self._accept(tracked.detection)
            self._state = "locked"
            return tracked.detection

        self._visual_tracker = None
        if motion_prediction is not None:
            self._state = "coasting"
            return motion_prediction
        self._state = "lost" if self._last_detection is not None else "searching"
        return None

    def _commission_identity(
        self,
        frame_bgr: np.ndarray,
        candidates: list[_Candidate],
    ) -> Detection | None:
        eligible: list[_Candidate] = []
        for candidate in candidates:
            detection = candidate.detection
            width = float(detection.bbox_width_px or 0.0)
            height = float(detection.bbox_height_px or 0.0)
            vertical_ratio = height / max(width, 1.0)
            if height < self.config.commissioning_min_bbox_height_px:
                continue
            if vertical_ratio < self.config.commissioning_min_vertical_ratio:
                continue
            descriptor_count = (
                0
                if candidate.orb_descriptors is None
                else len(candidate.orb_descriptors)
            )
            if descriptor_count < self.config.commissioning_min_orb_features:
                continue
            eligible.append(candidate)

        if not eligible:
            self._pending_identity_candidate = None
            self._pending_count = 0
            self._state = "acquiring"
            return None

        chosen = max(
            eligible,
            key=lambda candidate: (
                candidate.score
                * min(
                    1.5,
                    float(candidate.detection.bbox_height_px or 0.0)
                    / max(float(candidate.detection.bbox_width_px or 1.0), 1.0),
                )
            ),
        )
        pending = self._pending_identity_candidate
        same_candidate = False
        if pending is not None:
            appearance_distance = _appearance_distance(
                chosen.appearance_histogram,
                pending.appearance_histogram,
            )
            same_candidate = (
                _detection_distance(chosen.detection, pending.detection)
                <= self.config.association_max_distance_px
                and appearance_distance <= self.config.maximum_appearance_distance
            )
        self._pending_count = self._pending_count + 1 if same_candidate else 1
        self._pending_identity_candidate = chosen
        self._state = "acquiring"
        if self._pending_count < self.config.reacquire_confirmation_frames:
            return None

        self._target_signature = _TargetSignature(
            appearance_histogram=chosen.appearance_histogram.copy(),
            orb_descriptors=(
                None
                if chosen.orb_descriptors is None
                else chosen.orb_descriptors.copy()
            ),
            area_px2=_detection_box_area(chosen.detection),
            rotated_aspect_ratio=chosen.rotated_aspect_ratio,
        )
        self._live_bootstrap_remaining = 0
        self._pending_identity_candidate = None
        self._pending_count = 0
        self._initialise_visual_tracker(frame_bgr, chosen.detection)
        self._accept(chosen.detection)
        self._state = "locked"
        return chosen.detection

    def _identity_distance(self, candidate: _Candidate) -> float | None:
        signature = self._target_signature
        if signature is None:
            return None
        detection = candidate.detection
        if not candidate.from_visual_tracker:
            vertical_ratio = float(detection.bbox_height_px or 0.0) / max(
                float(detection.bbox_width_px or 0.0),
                1.0,
            )
            if vertical_ratio < self.config.identity_reacquire_min_vertical_ratio:
                return None
        appearance_distance = _appearance_distance(
            candidate.appearance_histogram,
            signature.appearance_histogram,
        )
        orb_matches = _orb_match_count(
            candidate.orb_descriptors,
            signature.orb_descriptors,
            maximum_distance=self.config.maximum_orb_match_distance,
        )
        orb_required = self.config.minimum_orb_identity_matches
        orb_matches_identity = orb_required <= 0 or orb_matches >= orb_required
        maximum_appearance_distance = (
            self.config.maximum_tracker_appearance_distance
            if candidate.from_visual_tracker
            else self.config.maximum_appearance_distance
        )
        if candidate.from_visual_tracker:
            if (
                appearance_distance > maximum_appearance_distance
                and not orb_matches_identity
            ):
                return None
        elif (
            appearance_distance > maximum_appearance_distance
            or not orb_matches_identity
        ):
            return None
        if signature.area_px2 > 0.0:
            area_ratio = _symmetric_ratio(
                _detection_box_area(candidate.detection),
                signature.area_px2,
            )
            if area_ratio > self.config.maximum_identity_area_ratio:
                return None
        if signature.rotated_aspect_ratio > 0.0:
            aspect_ratio_change = _symmetric_ratio(
                candidate.rotated_aspect_ratio,
                signature.rotated_aspect_ratio,
            )
            if aspect_ratio_change > self.config.maximum_identity_aspect_ratio_change:
                return None
        return appearance_distance

    def _update_target_signature(self, candidate: _Candidate) -> None:
        signature = self._target_signature
        bootstrapping = self._live_bootstrap_remaining > 0
        rate = (
            self.config.live_bootstrap_update_rate
            if bootstrapping
            else self.config.appearance_update_rate
        )
        if signature is None or rate <= 0.0:
            return
        if signature.area_px2 <= 0.0:
            signature.area_px2 = _detection_box_area(candidate.detection)
            signature.rotated_aspect_ratio = candidate.rotated_aspect_ratio
        histogram = (
            (1.0 - rate) * signature.appearance_histogram
            + rate * candidate.appearance_histogram
        )
        total = float(histogram.sum())
        if total > 0.0:
            histogram = histogram / total
        signature.appearance_histogram = np.asarray(histogram, dtype=np.float32)
        signature.area_px2 = (
            (1.0 - rate) * signature.area_px2
            + rate * _detection_box_area(candidate.detection)
        )
        signature.rotated_aspect_ratio = (
            (1.0 - rate) * signature.rotated_aspect_ratio
            + rate * candidate.rotated_aspect_ratio
        )
        if bootstrapping:
            self._live_bootstrap_remaining -= 1

    def _initialise_visual_tracker(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
    ) -> None:
        if self.config.visual_tracker == "none":
            self._visual_tracker = None
            return
        box = _tracker_box(detection, frame_bgr.shape)
        if box is None:
            self._visual_tracker = None
            return
        factory = (
            cv2.TrackerCSRT_create
            if self.config.visual_tracker == "csrt"
            else cv2.TrackerKCF_create
        )
        tracker = factory()
        tracker.init(frame_bgr, box)
        self._visual_tracker = tracker
        self._frames_since_tracker_initialisation = 0

    def _update_visual_tracker(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_s: float,
    ) -> _Candidate | None:
        tracker = self._visual_tracker
        if tracker is None:
            return None
        ok, raw_box = tracker.update(frame_bgr)
        self._frames_since_tracker_initialisation += 1
        if not ok:
            self._visual_tracker = None
            return None
        height, width = frame_bgr.shape[:2]
        x, y, box_width, box_height = (float(value) for value in raw_box)
        x = float(np.clip(x, 0.0, max(width - 1.0, 0.0)))
        y = float(np.clip(y, 0.0, max(height - 1.0, 0.0)))
        box_width = float(np.clip(box_width, 1.0, width - x))
        box_height = float(np.clip(box_height, 1.0, height - y))
        radius = 0.5 * float(np.hypot(box_width, box_height))
        detection = Detection(
            timestamp_s=timestamp_s,
            x_px=x + 0.5 * box_width,
            y_px=y + 0.5 * box_height,
            radius_px=radius,
            area_px2=box_width * box_height,
            quality=0.55,
            bbox_x_px=x,
            bbox_y_px=y,
            bbox_width_px=box_width,
            bbox_height_px=box_height,
        )
        histogram = _appearance_histogram(
            frame_bgr,
            detection_box(detection, frame_bgr.shape, padding_px=0),
        )
        box = detection_box(detection, frame_bgr.shape, padding_px=0)
        return _Candidate(
            score=detection.quality * sqrt_area(detection.area_px2),
            detection=detection,
            appearance_histogram=histogram,
            orb_descriptors=(
                _orb_descriptors(frame_bgr, box) if self._needs_orb else None
            ),
            rotated_aspect_ratio=max(box_width, box_height)
            / max(min(box_width, box_height), 1.0),
            from_visual_tracker=True,
        )

    def _select_temporally(
        self,
        candidates: list[tuple[float, Detection]],
        *,
        timestamp_s: float,
    ) -> Detection | None:
        maximum_distance = self.config.association_max_distance_px
        if maximum_distance <= 0.0:
            return max(candidates, key=lambda item: item[0])[1] if candidates else None

        last = self._last_detection
        if last is not None and timestamp_s - last.timestamp_s <= self.config.lock_timeout_s:
            associated: list[tuple[float, float, Detection]] = []
            for score, candidate in candidates:
                distance = float(
                    np.hypot(candidate.x_px - last.x_px, candidate.y_px - last.y_px)
                )
                radius_allowance = 0.25 * (candidate.radius_px + last.radius_px)
                if distance <= maximum_distance + radius_allowance:
                    associated.append((distance, -score, candidate))
            if not associated:
                return None
            chosen = min(associated, key=lambda item: (item[0], item[1]))[2]
            self._accept(chosen)
            return chosen

        self._last_detection = None
        if not candidates:
            self._pending_detection = None
            self._pending_count = 0
            return None

        chosen = max(candidates, key=lambda item: item[0])[1]
        required = self.config.reacquire_confirmation_frames
        if required <= 1:
            self._accept(chosen)
            return chosen

        pending = self._pending_detection
        if pending is not None:
            distance = float(
                np.hypot(chosen.x_px - pending.x_px, chosen.y_px - pending.y_px)
            )
        else:
            distance = float("inf")
        if pending is not None and distance <= maximum_distance:
            self._pending_count += 1
        else:
            self._pending_count = 1
        self._pending_detection = chosen
        if self._pending_count < required:
            return None
        self._accept(chosen)
        return chosen

    def _accept(self, detection: Detection) -> None:
        previous = self._last_observed_detection
        if previous is not None:
            elapsed_s = detection.timestamp_s - previous.timestamp_s
            if 1e-3 <= elapsed_s <= max(self.config.lock_timeout_s, 0.1):
                instantaneous_x = (detection.x_px - previous.x_px) / elapsed_s
                instantaneous_y = (detection.y_px - previous.y_px) / elapsed_s
                rate = self.config.velocity_update_rate
                self._velocity_x_px_s = (
                    (1.0 - rate) * self._velocity_x_px_s
                    + rate * instantaneous_x
                )
                self._velocity_y_px_s = (
                    (1.0 - rate) * self._velocity_y_px_s
                    + rate * instantaneous_y
                )
        self._last_observed_detection = detection
        self._last_detection = detection
        self._pending_detection = None
        self._pending_count = 0

    def _motion_prediction(self, timestamp_s: float) -> Detection | None:
        last = self._last_observed_detection
        coast_s = self.config.prediction_coast_s
        if last is None or coast_s <= 0.0:
            return None
        elapsed_s = timestamp_s - last.timestamp_s
        if elapsed_s <= 0.0 or elapsed_s > coast_s:
            return None
        offset_x = self._velocity_x_px_s * elapsed_s
        offset_y = self._velocity_y_px_s * elapsed_s
        confidence_scale = max(0.25, 1.0 - 0.7 * elapsed_s / coast_s)
        return Detection(
            timestamp_s=timestamp_s,
            x_px=last.x_px + offset_x,
            y_px=last.y_px + offset_y,
            radius_px=last.radius_px,
            area_px2=last.area_px2,
            quality=last.quality * confidence_scale,
            bbox_x_px=(
                None
                if last.bbox_x_px is None
                else last.bbox_x_px + offset_x
            ),
            bbox_y_px=(
                None
                if last.bbox_y_px is None
                else last.bbox_y_px + offset_y
            ),
            bbox_width_px=last.bbox_width_px,
            bbox_height_px=last.bbox_height_px,
        )

    def _crop(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
        roi = self.config.region_of_interest
        if roi is None:
            return frame_bgr, 0, 0
        x, y, width, height = roi
        if width <= 0 or height <= 0:
            raise ValueError("region_of_interest width and height must be positive")
        frame_height, frame_width = frame_bgr.shape[:2]
        x0 = int(np.clip(x, 0, frame_width))
        y0 = int(np.clip(y, 0, frame_height))
        x1 = int(np.clip(x + width, 0, frame_width))
        y1 = int(np.clip(y + height, 0, frame_height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("region_of_interest does not intersect the frame")
        return frame_bgr[y0:y1, x0:x1], x0, y0


def detection_box(
    detection: Detection,
    frame_shape: tuple[int, ...],
    *,
    padding_px: int = 8,
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    if (
        detection.bbox_x_px is not None
        and detection.bbox_y_px is not None
        and detection.bbox_width_px is not None
        and detection.bbox_height_px is not None
    ):
        x0 = int(np.clip(round(detection.bbox_x_px - padding_px), 0, width - 1))
        y0 = int(np.clip(round(detection.bbox_y_px - padding_px), 0, height - 1))
        x1 = int(
            np.clip(
                round(detection.bbox_x_px + detection.bbox_width_px + padding_px),
                0,
                width - 1,
            )
        )
        y1 = int(
            np.clip(
                round(detection.bbox_y_px + detection.bbox_height_px + padding_px),
                0,
                height - 1,
            )
        )
        return x0, y0, x1, y1
    radius = detection.radius_px + padding_px
    x0 = int(np.clip(round(detection.x_px - radius), 0, width - 1))
    y0 = int(np.clip(round(detection.y_px - radius), 0, height - 1))
    x1 = int(np.clip(round(detection.x_px + radius), 0, width - 1))
    y1 = int(np.clip(round(detection.y_px + radius), 0, height - 1))
    return x0, y0, x1, y1


def draw_tracking_box(
    frame_bgr: np.ndarray,
    detection: Detection | None,
    *,
    state: str = "searching",
) -> np.ndarray:
    output = frame_bgr.copy()
    if detection is None:
        label = {
            "acquiring": "SHOW TARGET CLEARLY",
            "lost": "TARGET LOST",
        }.get(state, "SEARCHING")
        cv2.putText(
            output,
            label,
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )
        return output

    x0, y0, x1, y1 = detection_box(detection, output.shape)
    coasting = state == "coasting"
    colour = (0, 210, 255) if coasting else (0, 0, 255)
    label = (
        f"PREDICTING TARGET {detection.quality:.0%}"
        if coasting
        else f"TARGET LOCKED {detection.quality:.0%}"
    )
    cv2.rectangle(output, (x0, y0), (x1, y1), colour, 3)
    cv2.putText(
        output,
        label,
        (x0, max(26, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
        2,
        cv2.LINE_AA,
    )
    return output


def _appearance_histogram(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = box
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((18, 8), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    valid = cv2.inRange(hsv, (0, 35, 30), (179, 255, 255))
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        valid,
        [18, 8],
        [0, 180, 0, 256],
    )
    total = float(histogram.sum())
    if total <= 0.0:
        return np.zeros((18, 8), dtype=np.float32)
    return np.asarray(histogram / total, dtype=np.float32)


def _appearance_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or float(first.sum()) <= 0.0 or float(second.sum()) <= 0.0:
        return 1.0
    return float(cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA))


def _orb_descriptors(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    scale = max(1.0, 140.0 / max(patch.shape[:2]))
    if scale > 1.0:
        patch = cv2.resize(
            patch,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    grayscale = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=160,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=8,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=15,
        fastThreshold=5,
    )
    _, descriptors = orb.detectAndCompute(grayscale, None)
    return descriptors


def _orb_match_count(
    first: np.ndarray | None,
    second: np.ndarray | None,
    *,
    maximum_distance: float,
) -> int:
    if first is None or second is None or len(first) == 0 or len(second) == 0:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(first, second)
    return sum(1 for match in matches if float(match.distance) <= maximum_distance)


def _detection_distance(first: Detection, second: Detection) -> float:
    return float(np.hypot(first.x_px - second.x_px, first.y_px - second.y_px))


def _detection_box_area(detection: Detection) -> float:
    if detection.bbox_width_px is not None and detection.bbox_height_px is not None:
        return max(1.0, detection.bbox_width_px * detection.bbox_height_px)
    return max(1.0, detection.area_px2)


def _symmetric_ratio(first: float, second: float) -> float:
    smaller = max(min(abs(first), abs(second)), 1e-6)
    return max(abs(first), abs(second)) / smaller


def _tracker_box(
    detection: Detection,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    if (
        detection.bbox_x_px is None
        or detection.bbox_y_px is None
        or detection.bbox_width_px is None
        or detection.bbox_height_px is None
    ):
        return None
    height, width = frame_shape[:2]
    x = int(np.clip(round(detection.bbox_x_px), 0, width - 1))
    y = int(np.clip(round(detection.bbox_y_px), 0, height - 1))
    box_width = int(np.clip(round(detection.bbox_width_px), 1, width - x))
    box_height = int(np.clip(round(detection.bbox_height_px), 1, height - y))
    return x, y, box_width, box_height


def sqrt_area(area: float) -> float:
    return float(np.sqrt(max(area, 0.0)))
