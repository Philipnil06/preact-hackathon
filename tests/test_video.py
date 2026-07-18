from __future__ import annotations

from itertools import pairwise

from preact.video import SyntheticFrameSource


def test_synthetic_source_replays_after_reset() -> None:
    source = SyntheticFrameSource()
    first = source.read()
    assert first is not None
    source.reset()
    repeated = source.read()
    assert repeated is not None
    assert repeated.timestamp_s == first.timestamp_s
    assert repeated.tof_sample == first.tof_sample


def test_looping_synthetic_source_keeps_monotonic_time_until_stop() -> None:
    source = SyntheticFrameSource(loop=True)
    timestamps: list[float] = []
    for _ in range(len(source.scene.frames()) + 2):
        packet = source.read()
        assert packet is not None
        timestamps.append(packet.timestamp_s)

    assert all(later > earlier for earlier, later in pairwise(timestamps))
