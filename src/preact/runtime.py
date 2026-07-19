"""Threaded runtime used by the minimal local UI."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from preact.calibration import PlanarCalibration
from preact.detection import BallDetector, draw_tracking_box
from preact.engine import EngineSnapshot, PREACTEngine
from preact.face_detection import FaceDetector
from preact.mode import OperatingMode
from preact.models import PanTiltTarget, Prediction, ToFSample
from preact.pan_tilt import CALIBRATION_POINT_FRACTIONS
from preact.session_log import SessionLogger
from preact.video import FrameSource
from preact.voice import VoiceCommand, VoiceDecision


class VoiceService(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PublicStatus:
    mode: str
    playing: bool
    tracking: bool
    target_state: str
    observations: int
    tof_distance_mm: int | None
    tof_ready: bool
    landing_x_m: float | None
    actual_landing_x_m: float | None
    landing_error_cm: float | None
    time_to_impact_ms: int | None
    confidence: float
    servo_angle_deg: float | None
    pan_angle_deg: float | None
    tilt_angle_deg: float | None
    target_x_m: float | None
    target_y_m: float | None
    voice_enabled: bool
    voice_state: str
    last_transcript: str
    last_voice_command: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "playing": self.playing,
            "tracking": self.tracking,
            "target_state": self.target_state,
            "observations": self.observations,
            "tof_distance_mm": self.tof_distance_mm,
            "tof_ready": self.tof_ready,
            "landing_x_m": self.landing_x_m,
            "actual_landing_x_m": self.actual_landing_x_m,
            "landing_error_cm": self.landing_error_cm,
            "time_to_impact_ms": self.time_to_impact_ms,
            "confidence": self.confidence,
            "servo_angle_deg": self.servo_angle_deg,
            "pan_angle_deg": self.pan_angle_deg,
            "tilt_angle_deg": self.tilt_angle_deg,
            "target_x_m": self.target_x_m,
            "target_y_m": self.target_y_m,
            "voice_enabled": self.voice_enabled,
            "voice_state": self.voice_state,
            "last_transcript": self.last_transcript,
            "last_voice_command": self.last_voice_command,
            "reason": self.reason,
        }


class PREACTRuntime:
    def __init__(
        self,
        *,
        engine: PREACTEngine,
        source: FrameSource,
        session_logger: SessionLogger | None = None,
        landing_marker_hold_s: float = 10.0,
        landing_results_path: str | Path | None = None,
    ) -> None:
        if landing_marker_hold_s <= 0:
            raise ValueError("landing_marker_hold_s must be positive")
        self.engine = engine
        self.source = source
        self.session_logger = session_logger
        self.landing_marker_hold_s = landing_marker_hold_s
        self.landing_results_path = (
            Path(landing_results_path) if landing_results_path is not None else None
        )
        self._lock = threading.RLock()
        self._processing_lock = threading.RLock()
        self._mode = OperatingMode.IDLE
        self._closed = False
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._voice_service: VoiceService | None = None
        self._voice_state = "disabled"
        self._last_transcript = ""
        self._last_voice_command = ""
        self._last_landing_prediction: Prediction | None = None
        self._landing_marker_track_id: int | None = None
        self._landing_marker_expires_at_s: float | None = None
        self._actual_landing_candidate: tuple[float, float, float, float] | None = None
        self._previous_landing_world_y_m: float | None = None
        self._last_actual_landing_x_m: float | None = None
        self._last_landing_error_cm: float | None = None
        self._landing_result_track_id: int | None = None
        self._calibration_point: str | None = None
        self._manual_pan_deg = 90.0
        self._manual_tilt_deg = 90.0
        self._tracking_detectors: dict[str, BallDetector | FaceDetector] = {
            "balloon": engine.detector
        }
        self._tracking_target: str | None = None
        self._status = PublicStatus(
            mode=OperatingMode.IDLE.value,
            playing=False,
            tracking=False,
            target_state="idle",
            observations=0,
            tof_distance_mm=None,
            tof_ready=False,
            landing_x_m=None,
            actual_landing_x_m=None,
            landing_error_cm=None,
            time_to_impact_ms=None,
            confidence=0.0,
            servo_angle_deg=None,
            pan_angle_deg=None,
            tilt_angle_deg=None,
            target_x_m=None,
            target_y_m=None,
            voice_enabled=False,
            voice_state="disabled",
            last_transcript="",
            last_voice_command="",
            reason="ready",
        )

    def attach_voice_service(self, service: VoiceService) -> None:
        with self._lock:
            self._voice_service = service
            self._voice_state = "ready"

    def start_worker(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                if self._voice_service is not None:
                    self._voice_service.start()
                return
            self._closed = False
            self._thread = threading.Thread(
                target=self._run,
                name="preact-runtime",
                daemon=True,
            )
            self._thread.start()
            if self._voice_service is not None:
                self._voice_service.start()

    def play(self) -> None:
        self.set_mode(OperatingMode.LANDING)

    def start_tracking(self) -> bool:
        target = self._tracking_target
        detector = self._tracking_detectors.get(target or "")
        if target is None or detector is None:
            reason = (
                "choose face or balloon before starting tracking"
                if target is None
                else f"{target} reference is not ready"
            )
            with self._lock:
                self._status = replace(self._status, reason=reason)
            return False
        self.engine.detector = detector
        self.set_mode(OperatingMode.TRACKING)
        return True

    @property
    def tracking_target(self) -> str | None:
        return self._tracking_target

    def register_tracking_detector(
        self,
        target: str,
        detector: BallDetector | FaceDetector,
    ) -> None:
        if target not in {"face", "balloon"}:
            raise ValueError(f"unknown tracking target {target!r}")
        self._tracking_detectors[target] = detector

    def select_tracking_target(self, target: str) -> bool:
        if target not in {"face", "balloon"}:
            raise ValueError(f"unknown tracking target {target!r}")
        with self._lock:
            was_tracking = self._mode is OperatingMode.TRACKING
        if was_tracking:
            self.stop()
        self._tracking_target = target
        detector = self._tracking_detectors.get(target)
        ready = detector is not None
        if ready:
            self.engine.detector = detector
            self.engine.detector.reset()
        with self._lock:
            self._status = replace(
                self._status,
                reason=(
                    f"tracking target selected: {target}"
                    if ready
                    else f"{target} selected; reference images are required"
                ),
            )
        if was_tracking and ready:
            self.start_tracking()
        return ready

    def start_landing(self) -> None:
        self.set_mode(OperatingMode.LANDING)

    def set_mode(self, mode: OperatingMode) -> None:
        if mode is OperatingMode.IDLE:
            self.stop()
            return
        self.start_worker()
        with self._lock:
            previous_mode = self._mode
            if previous_mode is mode:
                return
            self._mode = OperatingMode.IDLE
        with self._processing_lock:
            self._finalize_landing_result()
            if self.session_logger is not None and previous_mode.active:
                self.session_logger.stop_session()
            self.source.reset()
            self.engine.reset()
            self._last_landing_prediction = None
            self._landing_marker_track_id = None
            self._landing_marker_expires_at_s = None
            self._reset_actual_landing_measurement()
            self.engine.genesis.set_mode(command_id=0, mode=mode)
            if mode is OperatingMode.LANDING:
                self.engine.prepare_for_landing(timestamp_s=time.monotonic())
            if self.session_logger is not None:
                self.session_logger.start_session()
        with self._lock:
            self._mode = mode
            self._status = _idle_status(
                mode=mode,
                playing=True,
                servo_angle_deg=self.engine.servo_planner.calibration.neutral_angle_deg,
                pan_angle_deg=(
                    self.engine.pan_tilt_planner.calibration.neutral_angle_deg
                    if self.engine.pan_tilt_planner is not None
                    else None
                ),
                tilt_angle_deg=(
                    self.engine.pan_tilt_planner.calibration.neutral_angle_deg
                    if self.engine.pan_tilt_planner is not None
                    else None
                ),
                reason=f"{mode.value} mode",
            )

    def stop(self) -> None:
        with self._lock:
            previous_mode = self._mode
            self._mode = OperatingMode.IDLE
            was_playing = previous_mode.active
        with self._processing_lock:
            self._finalize_landing_result()
            self.engine.reset()
            self._last_landing_prediction = None
            self._landing_marker_track_id = None
            self._landing_marker_expires_at_s = None
            self._reset_actual_landing_measurement()
            self.engine.genesis.set_mode(
                command_id=0,
                mode=OperatingMode.IDLE,
            )
            if self.session_logger is not None and was_playing:
                self.session_logger.stop_session()
        with self._lock:
            neutral = self.engine.servo_planner.calibration.neutral_angle_deg
            self._status = _idle_status(
                mode=OperatingMode.IDLE,
                playing=False,
                servo_angle_deg=neutral,
                pan_angle_deg=(
                    self.engine.pan_tilt_planner.calibration.neutral_angle_deg
                    if self.engine.pan_tilt_planner is not None
                    else None
                ),
                tilt_angle_deg=(
                    self.engine.pan_tilt_planner.calibration.neutral_angle_deg
                    if self.engine.pan_tilt_planner is not None
                    else None
                ),
                reason="stopped",
            )

    def close(self) -> None:
        if self._voice_service is not None:
            self._voice_service.close()
        self.stop()
        with self._lock:
            self._closed = True
            self._mode = OperatingMode.IDLE
        if self._thread:
            self._thread.join(timeout=2.0)
        self.source.close()
        self.engine.genesis.close()
        if self.session_logger is not None:
            self.session_logger.close()

    def status(self) -> PublicStatus:
        with self._lock:
            return replace(
                self._status,
                voice_enabled=self._voice_service is not None,
                voice_state=self._voice_state,
                last_transcript=self._last_transcript,
                last_voice_command=self._last_voice_command,
            )

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def landing_results(self) -> list[dict[str, object]]:
        path = self.landing_results_path
        if path is None or not path.exists():
            return []
        output: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                output.append(value)
        return output[-50:]

    def pan_tilt_calibration_status(self) -> dict[str, object]:
        planner = self.engine.pan_tilt_planner
        if planner is None:
            return {"available": False}
        with self._lock:
            points = planner.calibration_points
            return {
                "available": True,
                "active_point": self._calibration_point,
                "pan_deg": self._manual_pan_deg,
                "tilt_deg": self._manual_tilt_deg,
                "complete": planner.image_calibration.complete,
                "saved_count": len(points),
                "points": {
                    name: {
                        "pan_deg": angles[0],
                        "tilt_deg": angles[1],
                    }
                    for name, angles in points.items()
                },
                "positions": {
                    name: {"x": position[0], "y": position[1]}
                    for name, position in CALIBRATION_POINT_FRACTIONS.items()
                },
            }

    def start_pan_tilt_calibration(self, point: str) -> dict[str, object]:
        planner = self.engine.pan_tilt_planner
        if planner is None:
            raise RuntimeError("pan/tilt calibration is unavailable")
        if point not in CALIBRATION_POINT_FRACTIONS:
            raise ValueError(f"unknown calibration point {point!r}")
        self.stop()
        pan_deg, tilt_deg = planner.calibration_seed_angles(point)
        with self._processing_lock:
            with self._lock:
                self._calibration_point = point
                self._manual_pan_deg = pan_deg
                self._manual_tilt_deg = tilt_deg
            self._send_manual_pan_tilt()
        return self.pan_tilt_calibration_status()

    def adjust_pan_tilt_calibration(
        self,
        *,
        pan_delta_deg: float,
        tilt_delta_deg: float,
    ) -> dict[str, object]:
        planner = self.engine.pan_tilt_planner
        if planner is None or self._calibration_point is None:
            raise RuntimeError("choose a calibration point first")
        lower = planner.calibration.minimum_angle_deg
        upper = planner.calibration.maximum_angle_deg
        with self._processing_lock:
            with self._lock:
                self._manual_pan_deg = float(
                    np.clip(self._manual_pan_deg + pan_delta_deg, lower, upper)
                )
                self._manual_tilt_deg = float(
                    np.clip(self._manual_tilt_deg + tilt_delta_deg, lower, upper)
                )
            self._send_manual_pan_tilt()
        return self.pan_tilt_calibration_status()

    def set_pan_tilt_calibration_angles(
        self,
        *,
        pan_deg: float,
        tilt_deg: float,
    ) -> dict[str, object]:
        if self.engine.pan_tilt_planner is None or self._calibration_point is None:
            raise RuntimeError("choose a calibration point first")
        with self._processing_lock:
            with self._lock:
                self._manual_pan_deg = float(np.clip(pan_deg, 1.0, 180.0))
                self._manual_tilt_deg = float(np.clip(tilt_deg, 1.0, 180.0))
            self._send_manual_pan_tilt()
        return self.pan_tilt_calibration_status()

    def save_pan_tilt_calibration_point(self) -> dict[str, object]:
        planner = self.engine.pan_tilt_planner
        point = self._calibration_point
        if planner is None or point is None:
            raise RuntimeError("choose a calibration point first")
        with self._lock:
            planner.save_calibration_point(
                point,
                pan_deg=self._manual_pan_deg,
                tilt_deg=self._manual_tilt_deg,
            )
        return self.pan_tilt_calibration_status()

    def finish_pan_tilt_calibration(self) -> dict[str, object]:
        self.stop()
        with self._lock:
            self._calibration_point = None
        return self.pan_tilt_calibration_status()

    def _send_manual_pan_tilt(self) -> None:
        point = self._calibration_point or "center"
        x_fraction, y_fraction = CALIBRATION_POINT_FRACTIONS[point]
        self.engine.genesis.set_pan_tilt_target(
            PanTiltTarget(
                command_id=0,
                pan_angle_deg=self._manual_pan_deg,
                tilt_angle_deg=self._manual_tilt_deg,
                x_px=x_fraction,
                y_px=y_fraction,
                confidence=1.0,
            )
        )

    def handle_voice_decision(self, decision: VoiceDecision) -> None:
        with self._lock:
            self._last_transcript = decision.transcript
            self._last_voice_command = decision.command.value
        if decision.command is VoiceCommand.TRACKING:
            self.start_tracking()
        elif decision.command is VoiceCommand.LANDING:
            self.start_landing()
        elif decision.command is VoiceCommand.STOP:
            self.stop()
        elif decision.command is VoiceCommand.SWITCH_FACE:
            self.select_tracking_target("face")
        elif decision.command is VoiceCommand.SWITCH_BALLOON:
            self.select_tracking_target("balloon")

    def handle_voice_state(self, state: str) -> None:
        with self._lock:
            self._voice_state = state

    def _run(self) -> None:
        interval = 1.0 / max(self.source.frames_per_second, 1.0)
        while True:
            loop_started = time.monotonic()
            with self._lock:
                if self._closed:
                    return

            with self._processing_lock:
                with self._lock:
                    mode = self._mode
                packet = self.source.read()
                if packet is None:
                    if mode.active:
                        self.stop()
                    else:
                        self.source.reset()
                    time.sleep(interval)
                    continue

                if not mode.active:
                    self._publish_jpeg(packet.frame_bgr)
                    elapsed = time.monotonic() - loop_started
                    time.sleep(max(0.0, interval - elapsed))
                    continue

                if packet.tof_sample is not None:
                    self.engine.process_tof(packet.tof_sample)
                for message in self.engine.genesis.poll():
                    if isinstance(message, ToFSample):
                        self.engine.process_tof(message)

                snapshot = self.engine.process_frame(
                    packet.frame_bgr,
                    timestamp_s=packet.timestamp_s,
                    mode=mode,
                )
                self._expire_landing_marker_if_needed(loop_started)
                if self.session_logger is not None:
                    self.session_logger.record(snapshot)
                if (
                    snapshot.mode is OperatingMode.LANDING
                    and snapshot.landing_locked
                    and self._landing_marker_track_id != snapshot.landing_track_id
                ):
                    # Keep the first sufficiently early, stable estimate on screen.
                    # Later colour glitches must not move the judging reference.
                    self._last_landing_prediction = snapshot.prediction
                    self._landing_marker_track_id = snapshot.landing_track_id
                    self._landing_marker_expires_at_s = (
                        loop_started + self.landing_marker_hold_s
                    )
                    self._reset_actual_landing_measurement()
                    self._last_actual_landing_x_m = None
                    self._last_landing_error_cm = None
                self._observe_actual_landing(snapshot)
                rendered = _render(
                    packet.frame_bgr,
                    snapshot,
                    calibration=self.engine.calibration,
                    landing_marker_prediction=self._last_landing_prediction,
                    actual_landing_candidate=self._actual_landing_candidate,
                    landing_error_cm=self._current_landing_error_cm(),
                )
                public_status = replace(
                    _status_from_snapshot(snapshot, playing=True),
                    actual_landing_x_m=(
                        self._actual_landing_candidate[0]
                        if self._actual_landing_candidate is not None
                        else self._last_actual_landing_x_m
                    ),
                    landing_error_cm=(
                        self._current_landing_error_cm()
                        if self._actual_landing_candidate is not None
                        else self._last_landing_error_cm
                    ),
                )
                self._publish_jpeg(
                    rendered,
                    status=public_status,
                )

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, interval - elapsed))

    def _expire_landing_marker_if_needed(self, now_s: float) -> None:
        expires_at = self._landing_marker_expires_at_s
        if expires_at is not None and now_s >= expires_at:
            self._finalize_landing_result()
            self._last_landing_prediction = None
            self._landing_marker_expires_at_s = None

    def _observe_actual_landing(self, snapshot: EngineSnapshot) -> None:
        detection = snapshot.detection
        if self._last_landing_prediction is None or detection is None:
            return
        world_x, world_y = self.engine.calibration.pixel_to_world(
            detection.x_px,
            detection.y_px,
        )
        previous_y = self._previous_landing_world_y_m
        self._previous_landing_world_y_m = world_y
        if world_y > 0.15 or (previous_y is not None and world_y > previous_y + 0.03):
            return
        candidate = self._actual_landing_candidate
        if candidate is None or abs(world_y) < abs(candidate[1]):
            self._actual_landing_candidate = (
                world_x,
                world_y,
                detection.x_px,
                detection.y_px,
            )

    def _current_landing_error_cm(self) -> float | None:
        prediction = self._last_landing_prediction
        candidate = self._actual_landing_candidate
        if prediction is None or prediction.landing_x_m is None or candidate is None:
            return None
        return abs(candidate[0] - prediction.landing_x_m) * 100.0

    def _reset_actual_landing_measurement(self) -> None:
        self._actual_landing_candidate = None
        self._previous_landing_world_y_m = None

    def _finalize_landing_result(self) -> None:
        prediction = self._last_landing_prediction
        track_id = self._landing_marker_track_id
        if (
            prediction is None
            or prediction.landing_x_m is None
            or track_id is None
            or self._landing_result_track_id == track_id
        ):
            return
        candidate = self._actual_landing_candidate
        actual_x = candidate[0] if candidate is not None else None
        error_cm = self._current_landing_error_cm()
        result: dict[str, object] = {
            "wall_time_s": time.time(),
            "track_id": track_id,
            "predicted_landing_x_m": prediction.landing_x_m,
            "actual_landing_x_m": actual_x,
            "signed_error_m": (
                actual_x - prediction.landing_x_m if actual_x is not None else None
            ),
            "absolute_error_cm": error_cm,
            "confidence": prediction.confidence,
            "actual_floor_height_m": candidate[1] if candidate is not None else None,
            "actual_x_px": candidate[2] if candidate is not None else None,
            "actual_y_px": candidate[3] if candidate is not None else None,
            "measured": candidate is not None,
        }
        path = self.landing_results_path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
        self._landing_result_track_id = track_id
        self._last_actual_landing_x_m = actual_x
        self._last_landing_error_cm = error_cm

    def _publish_jpeg(
        self,
        frame_bgr: np.ndarray,
        *,
        status: PublicStatus | None = None,
    ) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 86],
        )
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = encoded.tobytes()
            if status is not None:
                self._status = status


def _render(
    frame_bgr: np.ndarray,
    snapshot: EngineSnapshot,
    *,
    calibration: PlanarCalibration | None = None,
    landing_marker_prediction: Prediction | None = None,
    actual_landing_candidate: tuple[float, float, float, float] | None = None,
    landing_error_cm: float | None = None,
) -> np.ndarray:
    visible_detection = (
        snapshot.detection
        if snapshot.mode is OperatingMode.TRACKING
        or landing_marker_prediction is None
        else None
    )
    output = draw_tracking_box(
        frame_bgr,
        visible_detection,
        state=snapshot.detector_state,
    )
    prediction = snapshot.prediction
    if (
        snapshot.mode is OperatingMode.LANDING
        and calibration is not None
        and landing_marker_prediction is not None
    ):
        _draw_landing_marker(output, landing_marker_prediction, calibration)
        if actual_landing_candidate is not None:
            _draw_actual_landing_marker(
                output,
                actual_landing_candidate,
                calibration=calibration,
                landing_error_cm=landing_error_cm,
            )
    tof_text = (
        f"ToF {snapshot.tof_distance_mm} mm"
        if snapshot.tof_distance_mm is not None
        else "ToF waiting"
    )
    if snapshot.mode is OperatingMode.TRACKING:
        target = snapshot.servo_target
        pan_tilt = snapshot.pan_tilt_target
        if snapshot.detector_state == "acquiring":
            prediction_text = "Show the target clearly to commission identity"
        elif snapshot.detector_state == "coasting":
            prediction_text = "Briefly occluded - predicting target motion"
        elif snapshot.detector_state == "lost":
            prediction_text = "Target lost - show the same target again"
        elif pan_tilt is not None:
            prediction_text = (
                f"S1 {pan_tilt.pan_angle_deg:.0f} deg  "
                f"S3 {pan_tilt.tilt_angle_deg:.0f} deg  "
                f"{pan_tilt.confidence:.0%}"
            )
        elif target is not None and target.target_y_m is not None:
            prediction_text = (
                f"Following {target.target_x_m:.2f}, "
                f"{target.target_y_m:.2f} m  {target.confidence:.0%}"
            )
        else:
            prediction_text = "Searching for ball..."
    elif (
        snapshot.landing_locked
        and prediction.valid
        and prediction.landing_x_m is not None
    ):
        impact_ms = round((prediction.time_to_impact_s or 0.0) * 1000.0)
        prediction_text = (
            f"Impact {prediction.landing_x_m:.2f} m  "
            f"{impact_ms} ms  {prediction.confidence:.0%}"
        )
    elif prediction.valid:
        prediction_text = "Validating trajectory..."
    else:
        prediction_text = "Predicting..."
    cv2.rectangle(
        output,
        (0, output.shape[0] - 54),
        (output.shape[1], output.shape[0]),
        (12, 12, 12),
        -1,
    )
    cv2.putText(
        output,
        tof_text,
        (20, output.shape[0] - 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        prediction_text,
        (20, output.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def _draw_landing_marker(
    frame_bgr: np.ndarray,
    prediction: Prediction,
    calibration: PlanarCalibration,
) -> None:
    """Draw the latest valid floor intersection directly in camera coordinates."""

    if not prediction.valid or prediction.landing_x_m is None:
        return
    x_px, y_px = calibration.world_to_pixel(prediction.landing_x_m, 0.0)
    height, width = frame_bgr.shape[:2]
    if not (0.0 <= x_px < width and 0.0 <= y_px < height):
        return
    centre = (round(x_px), round(y_px))
    green = (66, 255, 105)
    cv2.circle(frame_bgr, centre, 13, (8, 8, 8), -1, cv2.LINE_AA)
    cv2.circle(frame_bgr, centre, 10, green, 2, cv2.LINE_AA)
    cv2.circle(frame_bgr, centre, 4, green, -1, cv2.LINE_AA)
    label = "PREDICTED LANDING"
    label_y = max(24, centre[1] - 18)
    cv2.putText(
        frame_bgr,
        label,
        (max(8, centre[0] - 84), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        green,
        1,
        cv2.LINE_AA,
    )


def _draw_actual_landing_marker(
    frame_bgr: np.ndarray,
    candidate: tuple[float, float, float, float],
    *,
    calibration: PlanarCalibration,
    landing_error_cm: float | None,
) -> None:
    """Draw the measured floor contact next to the green prediction marker."""

    actual_x_m, _, _, _ = candidate
    x_px, y_px = calibration.world_to_pixel(actual_x_m, 0.0)
    height, width = frame_bgr.shape[:2]
    if not (0.0 <= x_px < width and 0.0 <= y_px < height):
        return
    centre = (round(x_px), round(y_px))
    orange = (45, 170, 255)
    cv2.circle(frame_bgr, centre, 12, (8, 8, 8), -1, cv2.LINE_AA)
    cv2.drawMarker(
        frame_bgr,
        centre,
        orange,
        cv2.MARKER_TILTED_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    error_text = (
        f" - ERROR {landing_error_cm:.0f} cm"
        if landing_error_cm is not None
        else ""
    )
    cv2.putText(
        frame_bgr,
        f"ACTUAL LANDING{error_text}",
        (max(8, centre[0] - 92), max(24, centre[1] - 36)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        orange,
        1,
        cv2.LINE_AA,
    )


def _status_from_snapshot(snapshot: EngineSnapshot, *, playing: bool) -> PublicStatus:
    prediction = snapshot.prediction
    target = snapshot.servo_target
    pan_tilt = snapshot.pan_tilt_target
    servo_angle = target.angle_deg if target else None
    target_x = target.target_x_m if target else None
    target_y = target.target_y_m if target else None
    confidence = (
        (pan_tilt.confidence if pan_tilt is not None else target.confidence)
        if snapshot.mode is OperatingMode.TRACKING
        and (target is not None or pan_tilt is not None)
        else prediction.confidence
    )
    return PublicStatus(
        mode=snapshot.mode.value,
        playing=playing,
        tracking=snapshot.detection is not None,
        target_state=snapshot.detector_state,
        observations=snapshot.observation_count,
        tof_distance_mm=snapshot.tof_distance_mm,
        tof_ready=snapshot.tof_baseline_mm is not None,
        landing_x_m=(prediction.landing_x_m if snapshot.landing_locked else None),
        actual_landing_x_m=None,
        landing_error_cm=None,
        time_to_impact_ms=(
            round(prediction.time_to_impact_s * 1000.0)
            if snapshot.landing_locked and prediction.time_to_impact_s is not None
            else None
        ),
        confidence=confidence if snapshot.landing_locked else 0.0,
        servo_angle_deg=servo_angle,
        pan_angle_deg=pan_tilt.pan_angle_deg if pan_tilt else None,
        tilt_angle_deg=pan_tilt.tilt_angle_deg if pan_tilt else None,
        target_x_m=target_x,
        target_y_m=target_y,
        voice_enabled=False,
        voice_state="disabled",
        last_transcript="",
        last_voice_command="",
        reason=prediction.reason,
    )


def _idle_status(
    *,
    mode: OperatingMode,
    playing: bool,
    servo_angle_deg: float,
    pan_angle_deg: float | None = None,
    tilt_angle_deg: float | None = None,
    reason: str,
) -> PublicStatus:
    return PublicStatus(
        mode=mode.value,
        playing=playing,
        tracking=False,
        target_state="idle" if mode is OperatingMode.IDLE else "acquiring",
        observations=0,
        tof_distance_mm=None,
        tof_ready=False,
        landing_x_m=None,
        actual_landing_x_m=None,
        landing_error_cm=None,
        time_to_impact_ms=None,
        confidence=0.0,
        servo_angle_deg=servo_angle_deg,
        pan_angle_deg=pan_angle_deg,
        tilt_angle_deg=tilt_angle_deg,
        target_x_m=None,
        target_y_m=None,
        voice_enabled=False,
        voice_state="disabled",
        last_transcript="",
        last_voice_command="",
        reason=reason,
    )
