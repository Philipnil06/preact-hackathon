"""Threaded runtime used by the minimal local UI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np

from preact.calibration import PlanarCalibration
from preact.detection import draw_tracking_box
from preact.engine import EngineSnapshot, PREACTEngine
from preact.mode import OperatingMode
from preact.models import Prediction, ToFSample
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
    time_to_impact_ms: int | None
    confidence: float
    servo_angle_deg: float | None
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
            "time_to_impact_ms": self.time_to_impact_ms,
            "confidence": self.confidence,
            "servo_angle_deg": self.servo_angle_deg,
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
    ) -> None:
        self.engine = engine
        self.source = source
        self.session_logger = session_logger
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
        self._status = PublicStatus(
            mode=OperatingMode.IDLE.value,
            playing=False,
            tracking=False,
            target_state="idle",
            observations=0,
            tof_distance_mm=None,
            tof_ready=False,
            landing_x_m=None,
            time_to_impact_ms=None,
            confidence=0.0,
            servo_angle_deg=None,
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

    def start_tracking(self) -> None:
        self.set_mode(OperatingMode.TRACKING)

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
            if self.session_logger is not None and previous_mode.active:
                self.session_logger.stop_session()
            self.source.reset()
            self.engine.reset()
            self._last_landing_prediction = None
            self._landing_marker_track_id = None
            self.engine.genesis.set_mode(command_id=0, mode=mode)
            if self.session_logger is not None:
                self.session_logger.start_session()
        with self._lock:
            self._mode = mode
            self._status = _idle_status(
                mode=mode,
                playing=True,
                servo_angle_deg=self.engine.servo_planner.calibration.neutral_angle_deg,
                reason=f"{mode.value} mode",
            )

    def stop(self) -> None:
        with self._lock:
            previous_mode = self._mode
            self._mode = OperatingMode.IDLE
            was_playing = previous_mode.active
        with self._processing_lock:
            self.engine.reset()
            self._last_landing_prediction = None
            self._landing_marker_track_id = None
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
                if self.session_logger is not None:
                    self.session_logger.record(snapshot)
                if (
                    snapshot.mode is OperatingMode.LANDING
                    and snapshot.prediction.valid
                    and snapshot.prediction.time_to_impact_s is not None
                    and snapshot.prediction.time_to_impact_s >= 0.18
                    and snapshot.prediction.confidence >= 0.55
                    and self._landing_marker_track_id != snapshot.landing_track_id
                ):
                    # Keep the first sufficiently early, stable estimate on screen.
                    # Later colour glitches must not move the judging reference.
                    self._last_landing_prediction = snapshot.prediction
                    self._landing_marker_track_id = snapshot.landing_track_id
                rendered = _render(
                    packet.frame_bgr,
                    snapshot,
                    calibration=self.engine.calibration,
                    landing_marker_prediction=self._last_landing_prediction,
                )
                self._publish_jpeg(
                    rendered,
                    status=_status_from_snapshot(snapshot, playing=True),
                )

            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, interval - elapsed))

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
) -> np.ndarray:
    output = draw_tracking_box(
        frame_bgr,
        snapshot.detection,
        state=snapshot.detector_state,
    )
    prediction = snapshot.prediction
    if (
        snapshot.mode is OperatingMode.LANDING
        and calibration is not None
        and landing_marker_prediction is not None
    ):
        _draw_landing_marker(output, landing_marker_prediction, calibration)
    tof_text = (
        f"ToF {snapshot.tof_distance_mm} mm"
        if snapshot.tof_distance_mm is not None
        else "ToF waiting"
    )
    if snapshot.mode is OperatingMode.TRACKING:
        target = snapshot.servo_target
        if snapshot.detector_state == "acquiring":
            prediction_text = "Show the target clearly to commission identity"
        elif snapshot.detector_state == "coasting":
            prediction_text = "Briefly occluded - predicting target motion"
        elif snapshot.detector_state == "lost":
            prediction_text = "Target lost - show the same target again"
        elif target is not None and target.target_y_m is not None:
            prediction_text = (
                f"Following {target.target_x_m:.2f}, "
                f"{target.target_y_m:.2f} m  {target.confidence:.0%}"
            )
        else:
            prediction_text = "Searching for ball..."
    elif prediction.valid and prediction.landing_x_m is not None:
        impact_ms = round((prediction.time_to_impact_s or 0.0) * 1000.0)
        prediction_text = (
            f"Impact {prediction.landing_x_m:.2f} m  "
            f"{impact_ms} ms  {prediction.confidence:.0%}"
        )
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


def _status_from_snapshot(snapshot: EngineSnapshot, *, playing: bool) -> PublicStatus:
    prediction = snapshot.prediction
    target = snapshot.servo_target
    servo_angle = target.angle_deg if target else None
    target_x = target.target_x_m if target else None
    target_y = target.target_y_m if target else None
    confidence = (
        target.confidence
        if snapshot.mode is OperatingMode.TRACKING and target is not None
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
        landing_x_m=prediction.landing_x_m,
        time_to_impact_ms=(
            round(prediction.time_to_impact_s * 1000.0)
            if prediction.time_to_impact_s is not None
            else None
        ),
        confidence=confidence,
        servo_angle_deg=servo_angle,
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
        time_to_impact_ms=None,
        confidence=0.0,
        servo_angle_deg=servo_angle_deg,
        target_x_m=None,
        target_y_m=None,
        voice_enabled=False,
        voice_state="disabled",
        last_transcript="",
        last_voice_command="",
        reason=reason,
    )
