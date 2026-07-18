from __future__ import annotations

from preact.models import ToFSample
from preact.tof import ToFGate, ToFGateConfig


def _sample(timestamp_s: float, distance_mm: int, *, valid: bool = True) -> ToFSample:
    return ToFSample(
        received_at_s=timestamp_s,
        device_time_ms=round(timestamp_s * 1000),
        distance_mm=distance_mm,
        valid=valid,
    )


def test_tof_gate_calibrates_triggers_and_releases() -> None:
    gate = ToFGate(
        ToFGateConfig(
            baseline_window=12,
            minimum_baseline_samples=8,
            trigger_drop_mm=80,
            release_drop_mm=35,
        )
    )
    for index in range(10):
        assert gate.update(_sample(index * 0.02, 1000 + (index % 3 - 1) * 2)) is None

    trigger = gate.update(_sample(0.30, 760))

    assert trigger is not None
    assert trigger.drop_mm > 200
    assert gate.occupied
    assert gate.has_recent_trigger(0.60)

    assert gate.update(_sample(0.32, 755)) is None
    assert gate.update(_sample(0.42, 995)) is None
    assert not gate.occupied


def test_invalid_tof_samples_do_not_change_baseline() -> None:
    gate = ToFGate(ToFGateConfig(minimum_baseline_samples=2))
    gate.update(_sample(0.0, 900))
    gate.update(_sample(0.1, 0, valid=False))
    assert gate.baseline_mm is None
    gate.update(_sample(0.2, 902))
    assert gate.baseline_mm == 901.0
