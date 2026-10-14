"""原始振动文件、连续时间轴与真实时间消费的正式回归。"""

import csv
import io
import json
import lzma
from time import monotonic, sleep

import numpy as np
import pytest

from core.services.vibration_replay import VibrationReplay, load_vibration_recording


def make_run_folder(tmp_path, sample_rate=25600, counts=(25600, 25600, 25600)):
    folder = tmp_path / "vibration_run"
    raw_dir = folder / "acceleration"
    raw_dir.mkdir(parents=True)
    channels = [f"device/ai{index}" for index in range(3)]
    manifest = {"configuration": {"groups": [{"signal_type": "acceleration",
                "channels": [{"physical_name": name} for name in channels],
                "settings": {"sample_rate_hz": sample_rate}}]}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    start = 0
    for index, count in enumerate(counts):
        time_s = (start + np.arange(count)) / sample_rate
        data = np.array([.004 * (channel + 1) * np.sin(2 * np.pi * 50 * time_s + channel)
                         for channel in range(3)])
        payload = io.BytesIO()
        np.savez(payload, time_s=time_s, data=data, channels=np.array(channels), sample_start_index=start,
                 sample_rate_hz=sample_rate, signal_type="acceleration", unit="g")
        filename = f"{index:06d}.npz.xz"
        (raw_dir / filename).write_bytes(lzma.compress(payload.getvalue()))
        rows.append({"signal_type": "acceleration", "sample_start_index": start,
                     "sample_count": count, "time_start_s": start / sample_rate,
                     "sample_rate_hz": sample_rate, "unit": "g", "data_file": filename})
        start += count
    with (folder / "segment_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return folder


def test_prepare_checks_all_samples_and_fixed_display_ranges(tmp_path):
    recording = load_vibration_recording(make_run_folder(tmp_path))
    assert recording.total_samples == 76800
    assert recording.duration_s == 3
    for window in (0.5, 1, 5, 10):
        assert recording.display_rates[window] <= 4096 / window
        assert all(low < 0 < high for low, high in recording.ranges[window])


def test_replay_delivers_only_due_samples_and_finishes_exactly(tmp_path):
    recording = load_vibration_recording(make_run_folder(tmp_path, sample_rate=1000, counts=(1000, 137)))
    replay = VibrationReplay(recording)
    replay.start()
    try:
        deadline = monotonic() + 3
        chunks = []
        while replay.samples_consumed < 500 and monotonic() < deadline:
            chunks.extend(replay.poll(.5))
            sleep(.005)
        assert replay.samples_consumed == 500
        assert all(chunk.end_sample <= 500 for chunk in chunks)
        while not replay.finished and monotonic() < deadline:
            chunks.extend(replay.poll(recording.duration_s))
            sleep(.005)
        assert replay.finished
        assert sum(chunk.source_count for chunk in chunks) == 1137
        assert replay.samples_consumed == recording.total_samples
        for window in (0.5, 1):
            # 此采样率下短窗口无需抽取，跨分片交付的样本应完整保留。
            values = np.concatenate([chunk.values[window] for chunk in chunks], axis=1)
            time_s = np.arange(recording.total_samples) / recording.sample_rate_hz
            expected = np.array([.004 * (channel + 1) * np.sin(2 * np.pi * 50 * time_s + channel)
                                 for channel in range(3)])
            np.testing.assert_allclose(values, expected)
    finally:
        replay.close()


def test_fixed_ranges_put_regular_measured_waveforms_at_seventy_percent(tmp_path):
    folder = make_run_folder(tmp_path, sample_rate=1024, counts=(10240, 10240))
    recording = load_vibration_recording(folder)
    for window in (0.5, 1, 5, 10):
        for channel, (lower, upper) in enumerate(recording.ranges[window]):
            measured_peak_to_peak = .008 * (channel + 1)
            assert measured_peak_to_peak / (upper - lower) == pytest.approx(.70, rel=.04)


@pytest.mark.parametrize("change", ["gap", "missing"])
def test_bad_source_is_rejected_before_playback(tmp_path, change):
    folder = make_run_folder(tmp_path, sample_rate=1000, counts=(1000, 1000))
    if change == "missing":
        next((folder / "acceleration").glob("*.npz.xz")).unlink()
    else:
        path = folder / "segment_records.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("acceleration,1000,1000", "acceleration,1001,1000"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_vibration_recording(folder)
