from __future__ import annotations

import numpy as np

from preact.detection import BallDetector
from preact.engine import EngineConfig, PREACTEngine
from preact.genesis import GenesisEmulator
from preact.mode import OperatingMode
from preact.models import Observation, Prediction, TrajectoryFit
from preact.pan_tilt import PanTiltCalibration, PanTiltPlanner
from preact.servo import ServoCalibration, ServoPlanner
from preact.synthetic import SyntheticThrow
from preact.tof import ToFGate, ToFGateConfig
from preact.trajectory import TrajectoryPredictor


def test_landing_lock_quality_accepts_recorded_camera_profile() -> None:
    scene = SyntheticThrow()
    engine = PREACTEngine(
        calibration=scene.calibration,
        detector=BallDetector(),
        predictor=TrajectoryPredictor(),
        tof_gate=ToFGate(),
        servo_planner=ServoPlanner(),
        genesis=GenesisEmulator(),
    )
    fit = TrajectoryFit(
        reference_time_s=1.0,
        x_m=0.8,
        y_m=0.2,
        vx_mps=1.0,
        vy_mps=-1.0,
        ay_mps2=-9.81,
        sample_count=8,
        observed_span_s=0.2,
        rmse_m=0.03,
        inlier_fraction=0.875,
        mean_quality=0.8,
    )
    recorded_quality = Prediction(
        valid=True,
        reason="ok",
        generated_at_s=1.0,
        landing_x_m=1.0,
        time_to_impact_s=0.10,
        confidence=0.68,
        fit=fit,
    )
    engine._observations.extend(
        (
            Observation(timestamp_s=0.80, x_m=0.60, y_m=0.32),
            Observation(timestamp_s=0.84, x_m=0.64, y_m=0.35),
            Observation(timestamp_s=0.88, x_m=0.68, y_m=0.34),
            Observation(timestamp_s=0.92, x_m=0.72, y_m=0.31),
            Observation(timestamp_s=0.96, x_m=0.76, y_m=0.27),
            Observation(timestamp_s=1.00, x_m=0.80, y_m=0.22),
        )
    )

    assert engine._is_landing_lock_candidate(
        recorded_quality,
        frame_width_px=640,
        frame_height_px=540,
    )

    low_confidence = Prediction(
        valid=True,
        reason="ok",
        generated_at_s=1.0,
        landing_x_m=1.0,
        time_to_impact_s=0.10,
        confidence=0.57,
        fit=fit,
    )
    assert not engine._is_landing_lock_candidate(
        low_confidence,
        frame_width_px=640,
        frame_height_px=540,
    )

    engine._observations.clear()
    engine._observations.extend(
        Observation(timestamp_s=0.8 + index * 0.04, x_m=0.6 + index * 0.04, y_m=0.3)
        for index in range(6)
    )
    assert not engine._is_landing_lock_candidate(
        recorded_quality,
        frame_width_px=640,
        frame_height_px=540,
    )


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


def test_pan_tilt_tracking_drives_both_camera_axes() -> None:
    scene = SyntheticThrow()
    genesis = GenesisEmulator()
    engine = PREACTEngine(
        calibration=scene.calibration,
        detector=BallDetector(),
        predictor=TrajectoryPredictor(),
        tof_gate=ToFGate(),
        servo_planner=ServoPlanner(),
        genesis=genesis,
        pan_tilt_planner=PanTiltPlanner(
            PanTiltCalibration(minimum_quality=0.0)
        ),
    )

    snapshots = [
        engine.process_frame(
            item.frame_bgr,
            timestamp_s=item.timestamp_s,
            mode=OperatingMode.TRACKING,
        )
        for item in scene.frames()
    ]

    targets = [value.pan_tilt_target for value in snapshots if value.pan_tilt_target]
    assert targets
    assert genesis.pan_tilt_targets
    assert any(target.pan_angle_deg != 90.0 for target in targets)
    assert any(target.tilt_angle_deg != 90.0 for target in targets)
    assert not genesis.targets


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


def test_predicted_landing_drives_pan_and_tilt_servos() -> None:
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
        pan_tilt_planner=PanTiltPlanner(
            PanTiltCalibration(minimum_quality=0.0)
        ),
        config=EngineConfig(require_tof_for_landing=False),
    )

    snapshots = [
        engine.process_frame(
            item.frame_bgr,
            timestamp_s=item.timestamp_s,
            mode=OperatingMode.LANDING,
        )
        for item in scene.frames()
    ]

    commanded = [value for value in snapshots if value.pan_tilt_target is not None]
    assert commanded
    assert genesis.pan_tilt_targets
    assert not genesis.targets
    assert any(value.prediction.valid for value in commanded)
    assert len({round(target.x_px, 6) for target in genesis.pan_tilt_targets}) == 1
    assert len({round(target.y_px, 6) for target in genesis.pan_tilt_targets}) == 1
    locked = engine._locked_landing_prediction
    reset_at = engine._landing_track_reset_at_s
    assert locked is not None
    assert reset_at is not None
    assert abs(reset_at - locked.generated_at_s - 10.0) < 1e-6

    held = engine.process_frame(
        np.zeros_like(scene.frames()[0].frame_bgr),
        timestamp_s=reset_at - 0.01,
        mode=OperatingMode.LANDING,
    )
    assert held.landing_track_id == 0
    assert engine._locked_landing_prediction is locked

    released = engine.process_frame(
        np.zeros_like(scene.frames()[0].frame_bgr),
        timestamp_s=reset_at,
        mode=OperatingMode.LANDING,
    )
    assert released.landing_track_id == 1
    assert engine._locked_landing_prediction is None
    assert genesis.pan_angle_deg == 90.0
    assert abs(genesis.tilt_angle_deg - 26.0) < 1e-9
