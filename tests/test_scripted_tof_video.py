from __future__ import annotations

from preact.synthetic import SyntheticThrow
from preact.video import (
    ScriptedToFConfig,
    ScriptedToFFrameSource,
    SyntheticFrameSource,
)


def test_scripted_tof_adds_baseline_trigger_and_release() -> None:
    source = ScriptedToFFrameSource(
        SyntheticFrameSource(SyntheticThrow()),
        ScriptedToFConfig(
            trigger_time_s=0.20,
            pulse_duration_s=0.08,
            baseline_distance_mm=1000,
            triggered_distance_mm=650,
        ),
    )
    samples: list[tuple[float, int]] = []
    while packet := source.read():
        assert packet.tof_sample is not None
        samples.append((packet.timestamp_s, packet.tof_sample.distance_mm))

    assert samples[0][1] == 1000
    assert any(distance == 650 for _, distance in samples)
    assert samples[-1][1] == 1000
