"""Run PREACT locally. Synthetic mode is the safe default."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from preact.calibration import PlanarCalibration
from preact.claude_targeting import (
    ClaudeCommissionedDetector,
    ClaudeTargetSelector,
)
from preact.detection import BallDetector, BallDetectorConfig
from preact.engine import EngineConfig, PREACTEngine
from preact.face_detection import FaceDetector, FaceDetectorConfig
from preact.flight_detection import FlightBallDetector
from preact.genesis import GenesisEmulator, GenesisSerial, PanTiltSerial
from preact.pan_tilt import (
    PanTiltCalibration,
    PanTiltImageCalibration,
    PanTiltPlanner,
)
from preact.runtime import PREACTRuntime
from preact.servo import ServoCalibration, ServoPlanner
from preact.session_log import SessionLogger
from preact.synthetic import SyntheticThrow
from preact.tof import ToFGate, ToFGateConfig
from preact.trajectory import TrajectoryPredictor
from preact.video import (
    OpenCVFrameSource,
    ScriptedToFConfig,
    ScriptedToFFrameSource,
    SyntheticFrameSource,
)
from preact.voice import ClaudeVoiceInterpreter
from preact.voice_listener import MicrophoneVoiceService, VoiceListenerConfig
from preact.web import create_app


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="PREACT local runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--source",
        help="Replay file path or explicit camera index such as 0. Omit for synthetic mode.",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--detector-profile", type=Path)
    parser.add_argument(
        "--face-detector-profile",
        type=Path,
        help="commissioned face profile created from the user's reference images",
    )
    parser.add_argument(
        "--claude-target-lock",
        action="store_true",
        help="use one Claude vision call to select the safe demo object",
    )
    parser.add_argument("--serial-port")
    parser.add_argument(
        "--pan-tilt-tracking",
        action="store_true",
        help="drive the S1/S3 text-protocol firmware from blue-ball tracking",
    )
    parser.add_argument("--pan-tilt-min-deg", type=float, default=10.0)
    parser.add_argument("--pan-tilt-max-deg", type=float, default=170.0)
    parser.add_argument("--pan-tilt-speed-deg-s", type=float, default=200.0)
    parser.add_argument("--pan-response-multiplier", type=float, default=1.7)
    parser.add_argument("--pan-tilt-deadband-deg", type=float, default=1.0)
    parser.add_argument(
        "--pan-tilt-calibration",
        type=Path,
        default=Path("config/pan-tilt-mount.json"),
        help="saved five-point mount calibration",
    )
    parser.add_argument(
        "--emulate-genesis",
        action="store_true",
        help="use camera with an in-memory Genesis emulator for bench testing",
    )
    parser.add_argument(
        "--replay-emulated-tof",
        action="store_true",
        help="inject a ToF baseline and trigger when replaying a video file",
    )
    parser.add_argument(
        "--vision-only-landing",
        action="store_true",
        help=(
            "bench-test landing prediction without a ToF passage trigger; "
            "the physical demo should omit this flag"
        ),
    )
    parser.add_argument("--replay-trigger-s", type=float, default=0.40)
    parser.add_argument("--log", type=Path, help="append frame telemetry as JSONL")
    parser.add_argument(
        "--landing-results",
        type=Path,
        default=Path("work/landing-results.jsonl"),
        help="append one predicted-versus-actual result per throw",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--landing-min-x-m", type=float, default=0.0)
    parser.add_argument("--landing-max-x-m", type=float, default=3.0)
    parser.add_argument("--servo-angle-min-deg", type=float, default=20.0)
    parser.add_argument("--servo-angle-max-deg", type=float, default=160.0)
    parser.add_argument("--servo-neutral-deg", type=float, default=90.0)
    parser.add_argument(
        "--servo-mapping",
        choices=("linear", "planar-pointing"),
        default="linear",
    )
    parser.add_argument("--servo-base-x-m", type=float, default=-0.4)
    parser.add_argument("--servo-base-y-m", type=float, default=0.5)
    parser.add_argument("--servo-zero-angle-deg", type=float, default=90.0)
    parser.add_argument(
        "--servo-direction",
        type=int,
        choices=(-1, 1),
        default=1,
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    parser.add_argument("--tof-trigger-drop-mm", type=int, default=80)
    parser.add_argument(
        "--voice",
        action="store_true",
        help="listen for Hey Preact commands using local Whisper + Claude",
    )
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument(
        "--microphone-device",
        default=os.getenv("PREACT_MICROPHONE_DEVICE") or None,
    )
    parser.add_argument(
        "--microphone-sample-rate",
        type=int,
        default=int(os.getenv("PREACT_MICROPHONE_SAMPLE_RATE_HZ", "16000")),
    )
    parser.add_argument(
        "--microphone-channel",
        type=int,
        default=int(os.getenv("PREACT_MICROPHONE_CHANNEL", "0")),
    )
    parser.add_argument(
        "--voice-energy-threshold",
        type=float,
        default=float(os.getenv("PREACT_VOICE_ENERGY_THRESHOLD", "0.012")),
    )
    arguments = parser.parse_args()

    runtime = (
        _build_synthetic_runtime(log_path=arguments.log)
        if arguments.source is None
        else _build_hardware_runtime(arguments)
    )
    if arguments.voice:
        microphone_device: int | str | None = arguments.microphone_device
        if (
            isinstance(microphone_device, str)
            and microphone_device.isdigit()
        ):
            microphone_device = int(microphone_device)
        runtime.attach_voice_service(
            MicrophoneVoiceService(
                interpreter=ClaudeVoiceInterpreter.from_environment(),
                on_decision=runtime.handle_voice_decision,
                on_state=runtime.handle_voice_state,
                config=VoiceListenerConfig(
                    sample_rate_hz=arguments.microphone_sample_rate,
                    whisper_model=arguments.whisper_model,
                    microphone_device=microphone_device,
                    microphone_channel=arguments.microphone_channel,
                    energy_threshold=arguments.voice_energy_threshold,
                ),
            )
        )
    uvicorn.run(
        create_app(runtime),
        host=arguments.host,
        port=arguments.port,
        log_level="info",
    )


def _build_synthetic_runtime(*, log_path: Path | None = None) -> PREACTRuntime:
    scene = SyntheticThrow()
    source = SyntheticFrameSource(scene, loop=True)
    genesis = GenesisEmulator()
    engine = _build_engine(
        calibration=scene.calibration,
        genesis=genesis,
        landing_max_x_m=scene.config.world_width_m,
    )
    return PREACTRuntime(
        engine=engine,
        source=source,
        session_logger=SessionLogger(log_path) if log_path is not None else None,
    )


def _build_hardware_runtime(arguments: argparse.Namespace) -> PREACTRuntime:
    if arguments.calibration is None:
        raise SystemExit("--calibration is required for replay or camera input")
    source_is_camera = str(arguments.source).isdigit()
    source_value: str | int = (
        int(arguments.source) if source_is_camera else str(arguments.source)
    )
    source: OpenCVFrameSource | ScriptedToFFrameSource
    source = OpenCVFrameSource(
        source_value,
        requested_width_px=arguments.camera_width,
        requested_height_px=arguments.camera_height,
        requested_fps=arguments.camera_fps,
    )
    if source_is_camera:
        if arguments.serial_port is not None:
            genesis: GenesisSerial | PanTiltSerial | GenesisEmulator = (
                PanTiltSerial(arguments.serial_port)
                if arguments.pan_tilt_tracking
                else GenesisSerial(arguments.serial_port)
            )
        elif arguments.emulate_genesis:
            genesis = GenesisEmulator()
        else:
            source.close()
            raise SystemExit(
                "--serial-port is required for live ToF + servo; use "
                "--emulate-genesis only for camera/voice bench testing"
            )
    else:
        if not arguments.replay_emulated_tof:
            source.close()
            raise SystemExit(
                "--replay-emulated-tof is required for deterministic file replay"
            )
        source = ScriptedToFFrameSource(
            source,
            ScriptedToFConfig(trigger_time_s=arguments.replay_trigger_s),
        )
        genesis = GenesisEmulator()
    calibration = PlanarCalibration.load(arguments.calibration)
    if arguments.detector_profile is not None and arguments.claude_target_lock:
        raise SystemExit(
            "choose either --detector-profile or --claude-target-lock, not both"
        )
    if arguments.detector_profile is not None:
        detector = BallDetector(BallDetectorConfig.load(arguments.detector_profile))
    elif arguments.claude_target_lock:
        detector = ClaudeCommissionedDetector(
            ClaudeTargetSelector.from_environment()
        )
    else:
        detector = BallDetector()
    engine = _build_engine(
        calibration=calibration,
        genesis=genesis,
        detector=detector,
        servo_calibration=ServoCalibration(
            landing_min_x_m=arguments.landing_min_x_m,
            landing_max_x_m=arguments.landing_max_x_m,
            angle_at_min_deg=arguments.servo_angle_min_deg,
            angle_at_max_deg=arguments.servo_angle_max_deg,
            neutral_angle_deg=arguments.servo_neutral_deg,
            minimum_confidence=arguments.minimum_confidence,
            mapping_mode=arguments.servo_mapping,
            pointer_base_x_m=arguments.servo_base_x_m,
            pointer_base_y_m=arguments.servo_base_y_m,
            pointer_zero_angle_deg=arguments.servo_zero_angle_deg,
            pointer_direction=arguments.servo_direction,
        ),
        tof_gate=ToFGate(
            ToFGateConfig(trigger_drop_mm=arguments.tof_trigger_drop_mm)
        ),
        engine_config=EngineConfig(
            require_tof_for_landing=not arguments.vision_only_landing
        ),
        landing_detector=FlightBallDetector(calibration=calibration),
        pan_tilt_planner=(
            PanTiltPlanner(
                PanTiltCalibration(
                    minimum_angle_deg=arguments.pan_tilt_min_deg,
                    maximum_angle_deg=arguments.pan_tilt_max_deg,
                    neutral_angle_deg=arguments.servo_neutral_deg,
                    deadband_deg=arguments.pan_tilt_deadband_deg,
                    maximum_speed_deg_s=arguments.pan_tilt_speed_deg_s,
                    pan_response_multiplier=arguments.pan_response_multiplier,
                ),
                image_calibration=PanTiltImageCalibration.load(
                    arguments.pan_tilt_calibration
                ),
                calibration_path=arguments.pan_tilt_calibration,
            )
            if arguments.pan_tilt_tracking
            else None
        ),
    )
    runtime = PREACTRuntime(
        engine=engine,
        source=source,
        session_logger=(
            SessionLogger(arguments.log) if arguments.log is not None else None
        ),
        landing_results_path=arguments.landing_results,
    )
    if arguments.face_detector_profile is not None:
        runtime.register_tracking_detector(
            "face",
            FaceDetector(FaceDetectorConfig.load(arguments.face_detector_profile)),
        )
    return runtime


def _build_engine(
    *,
    calibration: PlanarCalibration,
    genesis: GenesisSerial | PanTiltSerial | GenesisEmulator,
    landing_max_x_m: float = 3.0,
    detector: BallDetector | None = None,
    servo_calibration: ServoCalibration | None = None,
    tof_gate: ToFGate | None = None,
    engine_config: EngineConfig | None = None,
    landing_detector: FlightBallDetector | None = None,
    pan_tilt_planner: PanTiltPlanner | None = None,
) -> PREACTEngine:
    return PREACTEngine(
        calibration=calibration,
        detector=detector or BallDetector(),
        predictor=TrajectoryPredictor(),
        tof_gate=tof_gate or ToFGate(),
        servo_planner=ServoPlanner(
            servo_calibration
            or ServoCalibration(landing_max_x_m=landing_max_x_m)
        ),
        genesis=genesis,
        pan_tilt_planner=pan_tilt_planner,
        landing_detector=landing_detector,
        config=engine_config,
    )


if __name__ == "__main__":
    main()
