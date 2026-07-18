from __future__ import annotations

from preact.models import Prediction
from preact.servo import ServoCalibration, ServoPlanner


def _prediction(*, landing_x_m: float, confidence: float = 0.9) -> Prediction:
    return Prediction(
        valid=True,
        reason="ok",
        generated_at_s=1.0,
        landing_x_m=landing_x_m,
        time_to_impact_s=0.65,
        confidence=confidence,
        fit=None,
    )


def test_servo_maps_landing_range_to_angle_range() -> None:
    calibration = ServoCalibration(
        landing_min_x_m=1.0,
        landing_max_x_m=3.0,
        angle_at_min_deg=30.0,
        angle_at_max_deg=150.0,
    )
    assert calibration.angle_for_landing(1.0) == 30.0
    assert calibration.angle_for_landing(2.0) == 90.0
    assert calibration.angle_for_landing(3.0) == 150.0
    assert calibration.angle_for_landing(4.0) == 150.0


def test_planar_pointing_mapping_uses_real_pointer_geometry() -> None:
    calibration = ServoCalibration(
        landing_min_x_m=0.0,
        landing_max_x_m=3.0,
        angle_at_min_deg=10.0,
        angle_at_max_deg=170.0,
        neutral_angle_deg=90.0,
        mapping_mode="planar-pointing",
        pointer_base_x_m=-0.4,
        pointer_base_y_m=0.5,
        pointer_zero_angle_deg=90.0,
    )

    assert 38.0 < calibration.angle_for_landing(0.0) < 39.0
    assert 75.0 < calibration.angle_for_landing(1.5) < 76.0
    assert 104.0 < calibration.angle_for_point(1.5, 1.0) < 105.0


def test_planar_pointing_direction_can_be_reversed() -> None:
    normal = ServoCalibration(
        mapping_mode="planar-pointing",
        pointer_base_x_m=-0.4,
        pointer_base_y_m=0.5,
        pointer_direction=1,
    )
    reversed_mapping = ServoCalibration(
        mapping_mode="planar-pointing",
        pointer_base_x_m=-0.4,
        pointer_base_y_m=0.5,
        pointer_direction=-1,
    )

    assert normal.angle_for_point(1.0, 1.0) > 90.0
    assert reversed_mapping.angle_for_point(1.0, 1.0) < 90.0


def test_servo_requires_tof_trigger_and_confident_prediction() -> None:
    planner = ServoPlanner(ServoCalibration(minimum_confidence=0.5))

    assert (
        planner.target_for_prediction(
            _prediction(landing_x_m=2.0),
            tof_triggered=False,
        )
        is None
    )
    assert (
        planner.target_for_prediction(
            _prediction(landing_x_m=2.0, confidence=0.2),
            tof_triggered=True,
        )
        is None
    )

    target = planner.target_for_prediction(
        _prediction(landing_x_m=2.0),
        tof_triggered=True,
    )
    assert target is not None
    assert target.angle_deg == 113.33333333333333
    assert target.time_to_impact_ms == 650
