from __future__ import annotations

from preact.detection import BallDetector
from preact.engine import EngineConfig, PREACTEngine
from preact.genesis import GenesisEmulator
from preact.mode import OperatingMode
from preact.servo import ServoCalibration, ServoPlanner
from preact.synthetic import SyntheticThrow
from preact.tof import ToFGate, ToFGateConfig
from preact.trajectory import TrajectoryPredictor


def test_synthetic_throw_runs_through_tof_camera_prediction_and_servo() -> None:
    scene = SyntheticThrow()
    genesis = GenesisEmulator()
    engine = PREACTEngine(
        calibration=scene.calibration,
        detector=BallDetector(),
        predictor=TrajectoryPredictor(),
        tof_gate=ToFGate(
            ToFGateConfig(
                minimum_baseline_samples=8,
                trigger_drop_mm=100,
                release_drop_mm=40,
            )
        ),
        servo_planner=ServoPlanner(
            ServoCalibration(
                landing_min_x_m=0.0,
                landing_max_x_m=3.2,
                angle_at_min_deg=20.0,
                angle_at_max_deg=160.0,
                minimum_confidence=0.40,
            )
        ),
        genesis=genesis,
    )

    valid_predictions = []
    for item in scene.frames():
        engine.process_tof(item.tof_sample)
        snapshot = engine.process_frame(item.frame_bgr, timestamp_s=item.timestamp_s)
        if snapshot.prediction.valid:
            valid_predictions.append(snapshot.prediction)

    assert valid_predictions
    final = valid_predictions[-1]
    assert final.landing_x_m is not None
    assert abs(final.landing_x_m - scene.expected_landing_x_m) < 0.08
    assert genesis.targets
    assert 20.0 <= genesis.servo_state.angle_deg <= 160.0


def test_tracking_mode_follows_ball_without_a_tof_trigger() -> None:
    scene = SyntheticThrow()
    genesis = GenesisEmulator()
    engine = PREACTEngine(
        calibration=scene.calibration,
        detector=BallDetector(),
        predictor=TrajectoryPredictor(),
        tof_gate=ToFGate(),
        servo_planner=ServoPlanner(
            ServoCalibration(
                mapping_mode="planar-pointing",
                pointer_base_x_m=-0.4,
                pointer_base_y_m=0.5,
                angle_at_min_deg=10.0,
                angle_at_max_deg=170.0,
            )
        ),
        genesis=genesis,
    )

    tracked = []
    for item in scene.frames():
        snapshot = engine.process_frame(
            item.frame_bgr,
            timestamp_s=item.timestamp_s,
            mode=OperatingMode.TRACKING,
        )
        if snapshot.servo_target is not None:
            tracked.append(snapshot.servo_target)

    assert tracked
    assert all(target.purpose == "tracking" for target in genesis.targets)
    assert all(target.target_y_m is not None for target in genesis.targets)
    assert not any(target.purpose == "landing" for target in genesis.targets)


def test_camera_only_bench_mode_predicts_without_a_tof_trigger() -> None:
    scene = SyntheticThrow()
    genesis = GenesisEmulator()
    engine = PREACTEngine(
        calibration=scene.calibration,
        detector=BallDetector(),
        predictor=TrajectoryPredictor(),
        tof_gate=ToFGate(),
        servo_planner=ServoPlanner(
            ServoCalibration(
                landing_min_x_m=0.0,
                landing_max_x_m=3.2,
                minimum_confidence=0.40,
            )
        ),
        genesis=genesis,
        config=EngineConfig(require_tof_for_landing=False),
    )

    predictions = [
        snapshot.prediction
        for item in scene.frames()
        if (
            snapshot := engine.process_frame(
                item.frame_bgr,
                timestamp_s=item.timestamp_s,
            )
        ).prediction.valid
    ]

    assert predictions
    assert genesis.targets
    assert all(target.purpose == "landing" for target in genesis.targets)
