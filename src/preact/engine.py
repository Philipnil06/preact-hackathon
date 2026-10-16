"""End-to-end PREACT sensor-fusion engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from preact.calibration import PlanarCalibration
from preact.detection import BallDetector
from preact.flight_detection import FlightBallDetector
from preact.genesis import GenesisTransport
from preact.mode import OperatingMode
from preact.models import (
    Detection,
    Observation,
    PanTiltTarget,
    Prediction,
    ServoTarget,
    ToFSample,
    ToFTrigger,
)
from preact.pan_tilt import PanTiltPlanner
from preact.servo import ServoPlanner
from preact.tof import ToFGate
from preact.trajectory import TrajectoryPredictor


@dataclass(frozen=True, slots=True)
class EngineConfig:
    maximum_observations: int = 40
    maximum_detection_gap_s: float = 0.22
    minimum_servo_update_interval_s: float = 0.08
    minimum_servo_angle_change_deg: float = 0.5
    tracking_lost_neutral_s: float = 0.35
    require_tof_for_landing: bool = True
    landing_track_settle_s: float = 0.40
    # Calibrated from real 640x480 camera throws. The detector normally only
    # produces two clean fits before impact, and mild reflections can leave one
    # of eight samples outside the robust fit.
    minimum_landing_lock_confidence: float = 0.65
    minimum_landing_lock_time_to_impact_s: float = 0.08
    minimum_landing_lock_inlier_fraction: float = 0.85
    maximum_landing_lock_rmse_m: float = 0.035
    maximum_landing_lock_vertical_speed_mps: float = 0.0
    minimum_observed_descent_m: float = 0.015
    landing_stability_samples: int = 2
    maximum_landing_stability_span_m: float = 0.08
    landing_lock_hold_s: float = 10.0


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    mode: OperatingMode
    timestamp_s: float
    detection: Detection | None
    detector_state: str
    observation_count: int
    prediction: Prediction
    tof_trigger: ToFTrigger | None
    tof_baseline_mm: float | None
    tof_distance_mm: int | None
    servo_target: ServoTarget | None
    pan_tilt_target: PanTiltTarget | None
    landing_track_id: int
    landing_locked: bool


class PREACTEngine:
    def __init__(
        self,
        *,
        calibration: PlanarCalibration,
        detector: BallDetector,
        predictor: TrajectoryPredictor,
        tof_gate: ToFGate,
        servo_planner: ServoPlanner,
        genesis: GenesisTransport,
        pan_tilt_planner: PanTiltPlanner | None = None,
        landing_detector: FlightBallDetector | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.detector = detector
        self.landing_detector = landing_detector
        self.predictor = predictor
        self.tof_gate = tof_gate
        self.servo_planner = servo_planner
        self.genesis = genesis
        self.pan_tilt_planner = pan_tilt_planner
        self.config = config or EngineConfig()
        self._observations: deque[Observation] = deque(
            maxlen=self.config.maximum_observations
        )
        self._active_trigger: ToFTrigger | None = None
        self._last_detection_at_s: float | None = None
        self._last_servo_target: ServoTarget | None = None
        self._last_servo_update_at_s: float | None = None
        self._last_pan_tilt_target: PanTiltTarget | None = None
        self._locked_landing_prediction: Prediction | None = None
        self._locked_landing_servo_target: ServoTarget | None = None
        self._landing_lock_candidates: deque[Prediction] = deque(
            maxlen=max(1, self.config.landing_stability_samples)
        )
        self._landing_track_reset_at_s: float | None = None
        self._landing_track_id = 0

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def process_tof(self, sample: ToFSample) -> ToFTrigger | None:
        trigger = self.tof_gate.update(sample)
        if trigger is not None:
            self._active_trigger = trigger
            self._observations.clear()
            self._last_detection_at_s = None
            self._last_servo_target = None
            self._last_servo_update_at_s = None
            self._locked_landing_prediction = None
            self._locked_landing_servo_target = None
            self._landing_lock_candidates.clear()
            self._landing_track_reset_at_s = None
        return trigger

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_s: float,
        mode: OperatingMode = OperatingMode.LANDING,
    ) -> EngineSnapshot:
        if (
            mode is OperatingMode.LANDING
            and self._landing_track_reset_at_s is not None
            and timestamp_s >= self._landing_track_reset_at_s
        ):
            self._complete_landing_track(timestamp_s)
        active_detector = (
            self.landing_detector
            if mode is OperatingMode.LANDING and self.landing_detector is not None
            else self.detector
        )
        detection = active_detector.detect(frame_bgr, timestamp_s=timestamp_s)
        if mode is OperatingMode.TRACKING:
            return self._process_tracking_frame(
                detection=detection,
                detector_state=active_detector.state,
                timestamp_s=timestamp_s,
                frame_width_px=frame_bgr.shape[1],
                frame_height_px=frame_bgr.shape[0],
            )
        if mode is OperatingMode.IDLE:
            return self._snapshot(
                mode=mode,
                timestamp_s=timestamp_s,
                detection=detection,
                detector_state=active_detector.state,
                prediction=Prediction.invalid("idle mode", timestamp_s),
                servo_target=None,
                pan_tilt_target=None,
            )

        landing_is_armed = (
            not self.config.require_tof_for_landing
            or self.tof_gate.has_recent_trigger(timestamp_s)
        )
        if detection is not None and landing_is_armed:
            if (
                self._last_detection_at_s is not None
                and timestamp_s - self._last_detection_at_s
                > self.config.maximum_detection_gap_s
            ):
                self._observations.clear()
                self._landing_lock_candidates.clear()
            world_x, world_y = self.calibration.pixel_to_world(
                detection.x_px,
                detection.y_px,
            )
            self._observations.append(
                Observation(
                    timestamp_s=timestamp_s,
                    x_m=world_x,
                    y_m=world_y,
                    quality=detection.quality,
                )
            )
            self._last_detection_at_s = timestamp_s

        prediction = self.predictor.predict(self._observations)
        # Only count independently measured fits. Re-rendering the same fit on
        # frames where the detector found nothing must never create a lock.
        if detection is not None:
            if self._is_landing_lock_candidate(
                prediction,
                frame_width_px=frame_bgr.shape[1],
                frame_height_px=frame_bgr.shape[0],
            ):
                if (
                    not self._landing_lock_candidates
                    or prediction.generated_at_s
                    > self._landing_lock_candidates[-1].generated_at_s
                ):
                    self._landing_lock_candidates.append(prediction)
            else:
                self._landing_lock_candidates.clear()
        stable_prediction = self._stable_landing_prediction()
        if (
            stable_prediction is not None
            and self._locked_landing_prediction is None
        ):
            stable_target = self.servo_planner.target_for_prediction(
                stable_prediction,
                tof_triggered=landing_is_armed,
            )
            self._locked_landing_prediction = stable_prediction
            self._locked_landing_servo_target = stable_target
            self._landing_track_reset_at_s = (
                timestamp_s + self.config.landing_lock_hold_s
            )
            self._landing_lock_candidates.clear()
        locked_prediction = self._locked_landing_prediction
        locked_target = self._locked_landing_servo_target
        pan_tilt_target: PanTiltTarget | None = None
        if (
            locked_target is not None
            and self.pan_tilt_planner is not None
            and locked_prediction is not None
            and locked_prediction.landing_x_m is not None
        ):
            landing_x_px, landing_y_px = self.calibration.world_to_pixel(
                locked_prediction.landing_x_m,
                0.0,
            )
            pan_tilt_target = self.pan_tilt_planner.target_for_point(
                x_px=landing_x_px,
                y_px=landing_y_px,
                confidence=locked_prediction.confidence,
                frame_width_px=frame_bgr.shape[1],
                frame_height_px=frame_bgr.shape[0],
                timestamp_s=timestamp_s,
                immediate=True,
            )
            dispatched_target = None
        elif self.pan_tilt_planner is not None:
            # A two-axis mount must wait for the stable landing lock. Never send
            # a preliminary single-servo command from a lower-confidence fit.
            dispatched_target = None
        else:
            dispatched_target = self._dispatch_if_needed(locked_target, timestamp_s)
        dispatched_pan_tilt = self._dispatch_pan_tilt_if_needed(
            pan_tilt_target,
            timestamp_s,
        )
        return self._snapshot(
            mode=mode,
            timestamp_s=timestamp_s,
            detection=detection,
            detector_state=active_detector.state,
            prediction=locked_prediction or prediction,
            servo_target=(locked_target if pan_tilt_target is not None else dispatched_target)
            or self._last_servo_target,
            pan_tilt_target=dispatched_pan_tilt or self._last_pan_tilt_target,
        )

    def reset(self) -> None:
        self._observations.clear()
        self._active_trigger = None
        self._last_detection_at_s = None
        self._last_servo_target = None
        self._last_servo_update_at_s = None
        self._last_pan_tilt_target = None
        self._locked_landing_prediction = None
        self._locked_landing_servo_target = None
        self._landing_lock_candidates.clear()
        self._landing_track_reset_at_s = None
        self._landing_track_id += 1
        self.detector.reset()
        if self.landing_detector is not None:
            self.landing_detector.reset()
        self.tof_gate.reset()
        if self.pan_tilt_planner is not None:
            self.pan_tilt_planner.reset()
        self.genesis.neutral(
            command_id=0,
            angle_deg=self.servo_planner.calibration.neutral_angle_deg,
        )

    def _complete_landing_track(self, timestamp_s: float) -> None:
        """End the completed throw without relearning the background scene."""

        self._observations.clear()
        self._last_detection_at_s = None
        self._last_servo_target = None
        self._last_servo_update_at_s = timestamp_s
        self._last_pan_tilt_target = None
        self._locked_landing_prediction = None
        self._locked_landing_servo_target = None
        self._landing_lock_candidates.clear()
        self._landing_track_reset_at_s = None
        self._landing_track_id += 1
        if self.landing_detector is not None:
            self.landing_detector.clear_track()
        if self.pan_tilt_planner is not None:
            self.pan_tilt_planner.reset()
        if self.pan_tilt_planner is not None:
            self.prepare_for_landing(timestamp_s=timestamp_s)
        else:
            self.genesis.neutral(
                command_id=0,
                angle_deg=self.servo_planner.calibration.neutral_angle_deg,
            )

    def prepare_for_landing(self, *, timestamp_s: float) -> PanTiltTarget | None:
        """Pre-position the mount at the centre of the visible floor."""

        planner = self.pan_tilt_planner
        if planner is None:
            return None
        target = planner.target_for_point(
            x_px=0.5,
            y_px=0.9,
            confidence=1.0,
            frame_width_px=2,
            frame_height_px=2,
            timestamp_s=timestamp_s,
            immediate=True,
        )
        if target is not None:
            self.genesis.set_pan_tilt_target(target)
            self._last_pan_tilt_target = target
            self._last_servo_update_at_s = timestamp_s
        return target

    def _is_landing_lock_candidate(
        self,
        prediction: Prediction,
        *,
        frame_width_px: int,
        frame_height_px: int,
    ) -> bool:
        fit = prediction.fit
        if (
            not prediction.valid
            or prediction.landing_x_m is None
            or prediction.time_to_impact_s is None
            or fit is None
            or prediction.confidence < self.config.minimum_landing_lock_confidence
            or prediction.time_to_impact_s
            < self.config.minimum_landing_lock_time_to_impact_s
            or fit.inlier_fraction
            < self.config.minimum_landing_lock_inlier_fraction
            or fit.rmse_m > self.config.maximum_landing_lock_rmse_m
            or fit.vy_mps > self.config.maximum_landing_lock_vertical_speed_mps
        ):
            return False
        observations = tuple(self._observations)
        if len(observations) < 2:
            return False
        peak_before_latest_m = max(value.y_m for value in observations[:-1])
        if (
            peak_before_latest_m - observations[-1].y_m
            < self.config.minimum_observed_descent_m
        ):
            return False
        x_px, y_px = self.calibration.world_to_pixel(prediction.landing_x_m, 0.0)
        return 0.0 <= x_px < frame_width_px and 0.0 <= y_px < frame_height_px

    def _stable_landing_prediction(self) -> Prediction | None:
        candidates = tuple(self._landing_lock_candidates)
        if len(candidates) < self.config.landing_stability_samples:
            return None
        landing_values = [
            value.landing_x_m
            for value in candidates
            if value.landing_x_m is not None
        ]
        if len(landing_values) != len(candidates):
            return None
        if max(landing_values) - min(landing_values) > (
            self.config.maximum_landing_stability_span_m
        ):
            return None
        return candidates[-1]

    def _dispatch_if_needed(
        self,
        target: ServoTarget | None,
        timestamp_s: float,
    ) -> ServoTarget | None:
        if target is None:
            return None
        previous = self._last_servo_target
        elapsed = (
            float("inf")
            if self._last_servo_update_at_s is None
            else timestamp_s - self._last_servo_update_at_s
        )
        target_change = (
            float("inf")
            if previous is None
            else abs(target.angle_deg - previous.angle_deg)
        )
        if (
            elapsed < self.config.minimum_servo_update_interval_s
            and target_change < self.config.minimum_servo_angle_change_deg
        ):
            return None
        self.genesis.set_servo_target(target)
        self._last_servo_target = target
        self._last_servo_update_at_s = timestamp_s
        return target

    def _process_tracking_frame(
        self,
        *,
        detection: Detection | None,
        detector_state: str,
        timestamp_s: float,
        frame_width_px: int,
        frame_height_px: int,
    ) -> EngineSnapshot:
        prediction = Prediction.invalid("tracking mode", timestamp_s)
        target: ServoTarget | None = None
        pan_tilt_target: PanTiltTarget | None = None
        is_observed_detection = detector_state not in ("coasting", "lost")
        coast_is_safe = (
            self._last_detection_at_s is not None
            and timestamp_s - self._last_detection_at_s
            < self.config.tracking_lost_neutral_s
        )
        if detection is not None and (is_observed_detection or coast_is_safe):
            if self.pan_tilt_planner is not None:
                pan_tilt_target = self.pan_tilt_planner.target_for_detection(
                    detection,
                    frame_width_px=frame_width_px,
                    frame_height_px=frame_height_px,
                    timestamp_s=timestamp_s,
                )
            else:
                world_x, world_y = self.calibration.pixel_to_world(
                    detection.x_px,
                    detection.y_px,
                )
                target = self.servo_planner.target_for_tracking(
                    x_m=world_x,
                    y_m=world_y,
                    quality=detection.quality,
                )
            if (
                is_observed_detection
                and (target is not None or pan_tilt_target is not None)
            ):
                self._last_detection_at_s = timestamp_s
        if (
            target is None and pan_tilt_target is None
            and self._last_detection_at_s is not None
            and timestamp_s - self._last_detection_at_s
            >= self.config.tracking_lost_neutral_s
            and (self._last_servo_target is not None or self._last_pan_tilt_target is not None)
        ):
            self.genesis.neutral(
                command_id=0,
                angle_deg=self.servo_planner.calibration.neutral_angle_deg,
            )
            self._last_servo_target = None
            self._last_pan_tilt_target = None
            self._last_servo_update_at_s = timestamp_s

        dispatched = self._dispatch_if_needed(target, timestamp_s)
        dispatched_pan_tilt = self._dispatch_pan_tilt_if_needed(
            pan_tilt_target,
            timestamp_s,
        )
        return self._snapshot(
            mode=OperatingMode.TRACKING,
            timestamp_s=timestamp_s,
            detection=detection,
            detector_state=detector_state,
            prediction=prediction,
            servo_target=dispatched or self._last_servo_target,
            pan_tilt_target=dispatched_pan_tilt or self._last_pan_tilt_target,
        )

    def _dispatch_pan_tilt_if_needed(
        self,
        target: PanTiltTarget | None,
        timestamp_s: float,
    ) -> PanTiltTarget | None:
        if target is None:
            return None
        previous = self._last_pan_tilt_target
        elapsed = (
            float("inf")
            if self._last_servo_update_at_s is None
            else timestamp_s - self._last_servo_update_at_s
        )
        largest_change = (
            float("inf")
            if previous is None
            else max(
                abs(target.pan_angle_deg - previous.pan_angle_deg),
                abs(target.tilt_angle_deg - previous.tilt_angle_deg),
            )
        )
        if (
            elapsed < self.config.minimum_servo_update_interval_s
            and largest_change < self.config.minimum_servo_angle_change_deg
        ):
            return None
        self.genesis.set_pan_tilt_target(target)
        self._last_pan_tilt_target = target
        self._last_servo_update_at_s = timestamp_s
        return target

    def _snapshot(
        self,
        *,
        mode: OperatingMode,
        timestamp_s: float,
        detection: Detection | None,
        detector_state: str | None = None,
        prediction: Prediction,
        servo_target: ServoTarget | None,
        pan_tilt_target: PanTiltTarget | None = None,
    ) -> EngineSnapshot:
        last_sample = self.tof_gate.last_sample
        return EngineSnapshot(
            mode=mode,
            timestamp_s=timestamp_s,
            detection=detection,
            detector_state=detector_state or self.detector.state,
            observation_count=len(self._observations),
            prediction=prediction,
            tof_trigger=self._active_trigger,
            tof_baseline_mm=self.tof_gate.baseline_mm,
            tof_distance_mm=last_sample.distance_mm if last_sample else None,
            servo_target=servo_target,
            pan_tilt_target=pan_tilt_target,
            landing_track_id=self._landing_track_id,
            landing_locked=self._locked_landing_prediction is not None,
        )
