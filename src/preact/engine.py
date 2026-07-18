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
from preact.models import Detection, Observation, Prediction, ServoTarget, ToFSample, ToFTrigger
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
    landing_track_id: int


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
        self.config = config or EngineConfig()
        self._observations: deque[Observation] = deque(
            maxlen=self.config.maximum_observations
        )
        self._active_trigger: ToFTrigger | None = None
        self._last_detection_at_s: float | None = None
        self._last_servo_target: ServoTarget | None = None
        self._last_servo_update_at_s: float | None = None
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
            )
        if mode is OperatingMode.IDLE:
            return self._snapshot(
                mode=mode,
                timestamp_s=timestamp_s,
                detection=detection,
                detector_state=active_detector.state,
                prediction=Prediction.invalid("idle mode", timestamp_s),
                servo_target=None,
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
        if prediction.valid and prediction.time_to_impact_s is not None:
            reset_at = (
                timestamp_s
                + prediction.time_to_impact_s
                + self.config.landing_track_settle_s
            )
            if (
                self._landing_track_reset_at_s is None
                or reset_at < self._landing_track_reset_at_s
            ):
                self._landing_track_reset_at_s = reset_at
        target = self.servo_planner.target_for_prediction(
            prediction,
            tof_triggered=landing_is_armed,
        )
        dispatched_target = self._dispatch_if_needed(target, timestamp_s)
        return self._snapshot(
            mode=mode,
            timestamp_s=timestamp_s,
            detection=detection,
            detector_state=active_detector.state,
            prediction=prediction,
            servo_target=dispatched_target or self._last_servo_target,
        )

    def reset(self) -> None:
        self._observations.clear()
        self._active_trigger = None
        self._last_detection_at_s = None
        self._last_servo_target = None
        self._last_servo_update_at_s = None
        self._landing_track_reset_at_s = None
        self._landing_track_id += 1
        self.detector.reset()
        if self.landing_detector is not None:
            self.landing_detector.reset()
        self.tof_gate.reset()
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
        self._landing_track_reset_at_s = None
        self._landing_track_id += 1
        if self.landing_detector is not None:
            self.landing_detector.clear_track()
        self.genesis.neutral(
            command_id=0,
            angle_deg=self.servo_planner.calibration.neutral_angle_deg,
        )

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
    ) -> EngineSnapshot:
        prediction = Prediction.invalid("tracking mode", timestamp_s)
        target: ServoTarget | None = None
        if detection is not None:
            world_x, world_y = self.calibration.pixel_to_world(
                detection.x_px,
                detection.y_px,
            )
            target = self.servo_planner.target_for_tracking(
                x_m=world_x,
                y_m=world_y,
                quality=detection.quality,
            )
            if target is not None:
                self._last_detection_at_s = timestamp_s
        if (
            target is None
            and self._last_detection_at_s is not None
            and timestamp_s - self._last_detection_at_s
            >= self.config.tracking_lost_neutral_s
            and self._last_servo_target is not None
        ):
            self.genesis.neutral(
                command_id=0,
                angle_deg=self.servo_planner.calibration.neutral_angle_deg,
            )
            self._last_servo_target = None
            self._last_servo_update_at_s = timestamp_s

        dispatched = self._dispatch_if_needed(target, timestamp_s)
        return self._snapshot(
            mode=OperatingMode.TRACKING,
            timestamp_s=timestamp_s,
            detection=detection,
            detector_state=detector_state,
            prediction=prediction,
            servo_target=dispatched or self._last_servo_target,
        )

    def _snapshot(
        self,
        *,
        mode: OperatingMode,
        timestamp_s: float,
        detection: Detection | None,
        detector_state: str | None = None,
        prediction: Prediction,
        servo_target: ServoTarget | None,
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
            landing_track_id=self._landing_track_id,
        )
