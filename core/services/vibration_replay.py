"""原始实测振动分片的只读预扫与真实时间回放。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import lzma
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable

import numpy as np

from core.algorithms.realtime_vibration import MAX_PLOT_POINTS, StreamingAntiAliasResampler


WINDOWS = (0.5, 1, 5, 10)


@dataclass(frozen=True)
class VibrationSegment:
    path: Path
    sample_start_index: int
    sample_count: int
    time_start_s: float


@dataclass(frozen=True)
class VibrationRecording:
    folder: Path
    name: str
    duration_s: float
    sample_rate_hz: float
    unit: str
    total_samples: int
    channels: tuple[str, ...]
    segments: tuple[VibrationSegment, ...]
    ranges: dict[float, list[tuple[float, float]]]
    display_rates: dict[float, float]
    passbands: dict[float, float]
    initial_values: tuple[float, ...]
    peak_ranges: dict[float, list[tuple[float, float]]]


@dataclass(frozen=True)
class ReplayChunk:
    source_count: int
    end_sample: int
    values: dict[float, np.ndarray]


def _resamplers(sample_rate: float, initial_values=None) -> dict[float, list[StreamingAntiAliasResampler]]:
    banks = {window: [StreamingAntiAliasResampler() for _ in range(3)] for window in WINDOWS}
    for window, bank in banks.items():
        for channel, resampler in enumerate(bank):
            resampler.configure(sample_rate, MAX_PLOT_POINTS / window)
            if initial_values is not None:
                resampler.initialize(initial_values[channel])
    return banks


def _read_segment(segment: VibrationSegment, sample_rate: float, channels: tuple[str, ...],
                  unit: str) -> tuple[np.ndarray, np.ndarray]:
    # 一次解压，避免 np.load 在 XZ 流上 seek 导致重复解压。
    with lzma.open(segment.path, "rb") as stream:
        payload = stream.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        data = np.asarray(archive["data"], dtype=np.float64)
        actual_channels = tuple(str(value) for value in archive["channels"].reshape(-1))
        rate = float(archive["sample_rate_hz"].reshape(-1)[0])
        start = int(archive["sample_start_index"].reshape(-1)[0])
        signal_type = str(archive["signal_type"].reshape(-1)[0])
        actual_unit = str(archive["unit"].reshape(-1)[0])
    if signal_type != "acceleration" or actual_channels != channels or actual_unit != unit:
        raise ValueError(f"振动分片的通道、类型或单位不一致：{segment.path.name}")
    if not math.isclose(rate, sample_rate) or start != segment.sample_start_index:
        raise ValueError(f"振动分片采样频率或起始索引不一致：{segment.path.name}")
    if time_s.shape != (segment.sample_count,) or data.shape != (3, segment.sample_count):
        raise ValueError(f"振动分片样本数或通道数不一致：{segment.path.name}")
    if not np.isfinite(data).all() or not np.isfinite(time_s).all():
        raise ValueError(f"振动分片存在非有限数据：{segment.path.name}")
    expected_time = segment.time_start_s + np.arange(segment.sample_count) / sample_rate
    if not np.allclose(time_s, expected_time, rtol=0, atol=0.01 / sample_rate):
        raise ValueError(f"振动分片 time_s 不连续或与采样频率不符：{segment.path.name}")
    return time_s, data


def load_vibration_recording(folder: str | Path,
                             progress: Callable[[int, str], None] | None = None
                             ) -> VibrationRecording:
    """后台完整预扫，验证原始时间轴，并为各显示窗口确定固定纵轴。"""
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
    groups = [group for group in manifest["configuration"]["groups"]
              if group["signal_type"] == "acceleration"]
    if len(groups) != 1:
        raise ValueError("采集记录需要一个三通道振动组")
    group = groups[0]
    channels = tuple(str(channel["physical_name"]) for channel in group["channels"])
    sample_rate = float(group["settings"]["sample_rate_hz"])
    if len(channels) != 3 or len(set(channels)) != 3:
        raise ValueError("采集记录需要三个不同的振动通道")
    banks = _resamplers(sample_rate)
    with (folder / "segment_records.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["signal_type"] == "acceleration"]
    if not rows:
        raise ValueError("采集记录中没有振动分片")
    rows.sort(key=lambda row: int(row["sample_start_index"]))
    unit = rows[0]["unit"]
    files = {path.name: path for path in folder.rglob("*.npz.xz")}
    segments = []
    for row in rows:
        file_name = row["data_file"]
        if file_name not in files:
            raise ValueError(f"振动分片缺失：{file_name}")
        segment = VibrationSegment(files[file_name], int(row["sample_start_index"]),
                                   int(row["sample_count"]), float(row["time_start_s"]))
        if segment.sample_count <= 0:
            raise ValueError(f"振动分片没有有效样本：{file_name}")
        if not math.isclose(float(row["sample_rate_hz"]), sample_rate) or row["unit"] != unit:
            raise ValueError(f"振动分片记录的频率或单位不一致：{file_name}")
        if segments:
            previous = segments[-1]
            if segment.sample_start_index != previous.sample_start_index + previous.sample_count:
                raise ValueError(f"振动分片索引存在间隙或重叠：{file_name}")
            expected_start = previous.time_start_s + previous.sample_count / sample_rate
            if not math.isclose(segment.time_start_s, expected_start, rel_tol=0, abs_tol=0.01 / sample_rate):
                raise ValueError(f"振动分片时间轴存在间隙或重叠：{file_name}")
        segments.append(segment)
    extrema = {window: np.array([[np.inf, -np.inf]] * 3) for window in WINDOWS}
    window_bounds = {window: [[] for _ in range(3)] for window in WINDOWS}
    for index, segment in enumerate(segments):
        _, data = _read_segment(segment, sample_rate, channels, unit)
        if index == 0:
            initial_values = tuple(np.mean(data[:, :round(sample_rate)], axis=1))
            banks = _resamplers(sample_rate, initial_values)
        for window, bank in banks.items():
            for channel, resampler in enumerate(bank):
                display = resampler.process(data[channel])
                if display.size:
                    extrema[window][channel, 0] = min(extrema[window][channel, 0], float(display.min()))
                    extrema[window][channel, 1] = max(extrema[window][channel, 1], float(display.max()))
                    length = round(window * resampler.output_sample_rate_hz)
                    for start in range(0, display.size, length):
                        part = display[start:start + length]
                        if part.size >= length * .9:
                            window_bounds[window][channel].append((float(part.min()), float(part.max())))
        if progress:
            progress(round((index + 1) / len(segments) * 100), f"检查振动分片 {index + 1}/{len(segments)}")
    ranges = {}
    peak_ranges = {}
    for window, bounds in extrema.items():
        if not np.isfinite(bounds).all():
            raise ValueError("振动记录太短，无法生成显示波形")
        ranges[window], peak_ranges[window] = [], []
        for channel, (lower, upper) in enumerate(bounds):
            center = (lower + upper) / 2
            half_span = max(upper - lower, 1e-6) / (2 * 0.75)
            peak_ranges[window].append((float(center - half_span), float(center + half_span)))
            typical = np.array(window_bounds[window][channel])
            if typical.size:
                center = float(np.median(typical.mean(axis=1)))
                half_span = max(float(np.median(typical[:, 1] - typical[:, 0])), 1e-6) / (2 * .70)
            ranges[window].append((float(center - half_span), float(center + half_span)))
    total = sum(segment.sample_count for segment in segments)
    duration = segments[-1].time_start_s + segments[-1].sample_count / sample_rate - segments[0].time_start_s
    return VibrationRecording(folder, folder.name, duration, sample_rate, unit, total, channels,
                              tuple(segments), ranges,
                              {window: bank[0].output_sample_rate_hz for window, bank in banks.items()},
                              {window: bank[0].passband_hz for window, bank in banks.items()}, initial_values, peak_ranges)


class VibrationReplay:
    """后台按原频率处理，只向界面交付真实时间已经到达的样本。"""

    def __init__(self, recording: VibrationRecording):
        self.recording = recording
        self.samples_consumed = 0
        self.error: Exception | None = None
        self.chunk_samples = max(1, round(recording.sample_rate_hz * 0.02))
        self._queue: Queue[ReplayChunk] = Queue(maxsize=50)
        self._pending: ReplayChunk | None = None
        self._stop = Event()
        self._done = Event()
        self._thread: Thread | None = None

    @property
    def finished(self) -> bool:
        return self.samples_consumed == self.recording.total_samples and self._done.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._produce, name="vibration-replay", daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        try:
            recording = self.recording
            banks = _resamplers(recording.sample_rate_hz, recording.initial_values)
            end_sample = 0
            for segment in recording.segments:
                if self._stop.is_set():
                    break
                _, data = _read_segment(segment, recording.sample_rate_hz, recording.channels, recording.unit)
                for start in range(0, segment.sample_count, self.chunk_samples):
                    if self._stop.is_set():
                        return
                    raw = data[:, start:start + self.chunk_samples]
                    values = {window: np.stack([resampler.process(raw[channel])
                                               for channel, resampler in enumerate(bank)])
                              for window, bank in banks.items()}
                    end_sample += raw.shape[1]
                    chunk = ReplayChunk(raw.shape[1], end_sample, values)
                    while not self._stop.is_set():
                        try:
                            self._queue.put(chunk, timeout=0.05)
                            break
                        except Full:
                            continue
        except Exception as error:
            self.error = error
        finally:
            self._done.set()

    def poll(self, elapsed_s: float) -> list[ReplayChunk]:
        if self.error is not None:
            raise self.error
        due_sample = (self.recording.total_samples if elapsed_s >= self.recording.duration_s
                      else math.floor(max(0, elapsed_s) * self.recording.sample_rate_hz))
        chunks = []
        while True:
            if self._pending is None:
                try:
                    self._pending = self._queue.get_nowait()
                except Empty:
                    break
            if self._pending.end_sample > due_sample:
                break
            chunks.append(self._pending)
            self.samples_consumed = self._pending.end_sample
            self._pending = None
        return chunks

    def close(self) -> None:
        self._stop.set()
