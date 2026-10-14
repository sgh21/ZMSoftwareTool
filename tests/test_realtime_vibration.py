"""有状态抗混叠与环形显示缓冲的算法回归。"""

import numpy as np

from core.algorithms.realtime_vibration import PlotBuffer, StreamingAntiAliasResampler


def test_chunked_filter_matches_continuous_filter_and_rejects_alias():
    rate = 25600
    time_s = np.arange(rate * 2) / rate
    values = np.sin(2 * np.pi * 50 * time_s) + np.sin(2 * np.pi * 3000 * time_s)
    whole = StreamingAntiAliasResampler()
    blocks = StreamingAntiAliasResampler()
    for resampler in (whole, blocks):
        resampler.configure(rate, 4096 / 5)
        resampler.initialize(0)
    expected = whole.process(values)
    actual = np.concatenate([blocks.process(values[start:start + 511]) for start in range(0, len(values), 511)])
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert .65 < np.std(actual[500:]) < .75


def test_ring_wrap_keeps_tail_without_growing_or_copying_history():
    buffer = PlotBuffer(100, 5)
    allocation = buffer.y
    values = np.arange(1500, dtype=float)
    for start in range(0, len(values), 17):
        buffer.append_display(values[start:start + 17])
    segments = buffer.display_segments()
    np.testing.assert_array_equal(np.concatenate([y for _, y in segments]), values[-500:])
    assert buffer.y is allocation
    assert buffer.count == buffer.capacity == 500
    assert all(np.shares_memory(y, buffer.y) for _, y in segments)
    assert segments[-1][0][-1] == 0
