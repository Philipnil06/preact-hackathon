from __future__ import annotations

from pathlib import Path

from preact.__main__ import _build_synthetic_runtime
from preact.genesis import PanTiltSerial
from preact.models import Detection, PanTiltTarget
from preact.pan_tilt import (
    CALIBRATION_POINT_FRACTIONS,
    PanTiltCalibration,
    PanTiltImageCalibration,
    PanTiltPlanner,
)


class FakeSerial:
    def __init__(self) -> None:
        self.incoming = bytearray()
        self.writes: list[bytes] = []
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self.incoming)

    def read(self, size: int) -> bytes:
        value = bytes(self.incoming[:size])
        del self.incoming[:size]
        return value

    def write(self, value: bytes) -> None:
        self.writes.append(value)

    def close(self) -> None:
        self.closed = True


def _detection(x_px: float, y_px: float, quality: float = 0.9) -> Detection:
    return Detection(
        timestamp_s=0.0,
        x_px=x_px,
        y_px=y_px,
        radius_px=20.0,
        area_px2=1200.0,
        quality=quality,
    )


def test_planner_maps_camera_axes_and_limits_speed() -> None:
    planner = PanTiltPlanner(
        PanTiltCalibration(smoothing_alpha=1.0, maximum_speed_deg_s=180.0)
    )

    target = planner.target_for_detection(
        _detection(639.0, 0.0),
        frame_width_px=640,
        frame_height_px=480,
        timestamp_s=1.0,
    )

    assert target is not None
    assert target.pan_angle_deg == 100.2
    assert target.tilt_angle_deg == 96.0


def test_planner_deadband_keeps_stationary_axis_still() -> None:
    planner = PanTiltPlanner(
        PanTiltCalibration(deadband_deg=2.0, smoothing_alpha=1.0)
    )
    first = planner.target_for_detection(
        _detection(420.0, 240.0),
        frame_width_px=640,
        frame_height_px=480,
        timestamp_s=1.0,
    )
    second = planner.target_for_detection(
        _detection(500.0, 241.0),
        frame_width_px=640,
        frame_height_px=480,
        timestamp_s=1.1,
    )

    assert first is not None and second is not None
    assert second.pan_angle_deg > first.pan_angle_deg
    assert second.tilt_angle_deg == first.tilt_angle_deg


def test_planner_can_point_at_a_predicted_image_position() -> None:
    planner = PanTiltPlanner(
        PanTiltCalibration(smoothing_alpha=1.0, maximum_speed_deg_s=9000.0)
    )

    target = planner.target_for_point(
        x_px=639.0,
        y_px=479.0,
        confidence=0.8,
        frame_width_px=640,
        frame_height_px=480,
        timestamp_s=1.0,
    )

    assert target is not None
    assert target.pan_angle_deg == 170.0
    assert target.tilt_angle_deg == 10.0


def test_text_transport_only_writes_the_axis_that_changed() -> None:
    connection = FakeSerial()
    transport = PanTiltSerial("COM-TEST", connection=connection)
    transport.set_pan_tilt_target(
        PanTiltTarget(1, 100.0, 80.0, 100.0, 100.0, 0.9)
    )
    transport.set_pan_tilt_target(
        PanTiltTarget(2, 105.0, 80.0, 120.0, 100.0, 0.9)
    )
    transport.set_pan_tilt_target(
        PanTiltTarget(3, 105.0, 75.0, 120.0, 120.0, 0.9)
    )

    assert connection.writes == [
        b"S1:100\nS3:80\n",
        b"S1:105\n",
        b"S3:75\n",
    ]

    transport.neutral(command_id=0, angle_deg=90.0)
    assert connection.writes[-1] == b"S1:90\nS3:90\n"


def test_text_transport_records_acknowledgements_and_retries_missing_echo() -> None:
    connection = FakeSerial()
    transport = PanTiltSerial("COM-TEST", connection=connection)
    transport.set_pan_tilt_target(
        PanTiltTarget(1, 82.0, 81.0, 500.0, 400.0, 0.9)
    )

    transport._last_write_at_s -= 1.0
    transport.poll()
    assert connection.writes[-1] == b"S1:82\nS3:81\n"

    connection.incoming.extend(b"S1:82\nS3:81\n")
    transport.poll()
    assert transport.acknowledged_angles == (82, 81)
    write_count = len(connection.writes)
    transport._last_write_at_s -= 1.0
    transport.poll()
    assert len(connection.writes) == write_count


def test_five_point_calibration_matches_every_saved_point(tmp_path: Path) -> None:
    path = tmp_path / "mount.json"
    calibration = PanTiltImageCalibration()
    expected: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(CALIBRATION_POINT_FRACTIONS):
        angles = (35.0 + index * 17.0, 145.0 - index * 13.0)
        expected[name] = angles
        calibration.set_point(name, pan_deg=angles[0], tilt_deg=angles[1])

    calibration.save(path)
    loaded = PanTiltImageCalibration.load(path)

    assert loaded.complete
    for name, position in CALIBRATION_POINT_FRACTIONS.items():
        actual = loaded.angles_for(*position)
        assert actual is not None
        assert abs(actual[0] - expected[name][0]) < 1e-9
        assert abs(actual[1] - expected[name][1]) < 1e-9


def test_pan_axis_responds_faster_without_changing_tilt_response() -> None:
    planner = PanTiltPlanner(PanTiltCalibration(minimum_quality=0.0))

    target = planner.target_for_point(
        x_px=639.0,
        y_px=0.0,
        confidence=1.0,
        frame_width_px=640,
        frame_height_px=480,
        timestamp_s=1.0,
    )

    assert target is not None
    assert target.pan_angle_deg - 90.0 > target.tilt_angle_deg - 90.0
    assert abs(target.tilt_angle_deg - (90.0 + 200.0 / 30.0)) < 1e-9


def test_immediate_landing_target_bypasses_tracking_smoothing() -> None:
    planner = PanTiltPlanner(PanTiltCalibration(minimum_quality=0.0))

    target = planner.target_for_point(
        x_px=639.0,
        y_px=479.0,
        confidence=1.0,
        frame_width_px=640,
        frame_height_px=480,
        timestamp_s=1.0,
        immediate=True,
    )

    assert target is not None
    assert target.pan_angle_deg == 170.0
    assert target.tilt_angle_deg == 10.0


def test_runtime_jogs_and_saves_a_mount_calibration_point(tmp_path: Path) -> None:
    path = tmp_path / "mount.json"
    runtime = _build_synthetic_runtime()
    runtime.engine.pan_tilt_planner = PanTiltPlanner(calibration_path=path)
    try:
        started = runtime.start_pan_tilt_calibration("center")
        assert started["active_point"] == "center"
        assert started["pan_deg"] == 90.0
        assert started["tilt_deg"] == 90.0

        adjusted = runtime.adjust_pan_tilt_calibration(
            pan_delta_deg=2.0,
            tilt_delta_deg=-3.0,
        )
        assert adjusted["pan_deg"] == 92.0
        assert adjusted["tilt_deg"] == 87.0

        positioned = runtime.set_pan_tilt_calibration_angles(
            pan_deg=121.0,
            tilt_deg=64.0,
        )
        assert positioned["pan_deg"] == 121.0
        assert positioned["tilt_deg"] == 64.0

        saved = runtime.save_pan_tilt_calibration_point()
        assert saved["saved_count"] == 1
        assert path.exists()
        assert runtime.engine.genesis.pan_angle_deg == 121.0
        assert runtime.engine.genesis.tilt_angle_deg == 64.0

        for name in ("top_left", "top_right", "bottom_left", "bottom_right"):
            runtime.start_pan_tilt_calibration(name)
            runtime.save_pan_tilt_calibration_point()
        before_shift = runtime.engine.pan_tilt_planner.calibration_points
        shifted = runtime.shift_pan_tilt_calibration(
            pan_delta_deg=1.0,
            tilt_delta_deg=-1.0,
        )
        assert shifted["complete"] is True
        for name, angles in before_shift.items():
            assert shifted["points"][name]["pan_deg"] == angles[0] + 1.0
            assert shifted["points"][name]["tilt_deg"] == angles[1] - 1.0

        reset = runtime.reset_pan_tilt_calibration()
        assert reset["saved_count"] == 0
        assert reset["complete"] is False
        assert reset["active_point"] is None
        assert PanTiltImageCalibration.load(path).points == {}
    finally:
        runtime.close()
