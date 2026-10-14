"""显示用抗混叠降采样和定长环形波形缓存，不改变原始测量数据。

分级低通与环形缓存复用 docs/code/src/nidata_collector/ui 中
plot_resampling.py 和 qt_app.py 的实时绘图方案。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal


MAX_PLOT_POINTS = 4096


@dataclass(frozen=True)
class AntiAliasResamplePlan:
    input_sample_rate_hz: float
    stage_factors: tuple[int, ...]

    @property
    def output_sample_rate_hz(self) -> float:
        return self.input_sample_rate_hz / math.prod(self.stage_factors)

    @property
    def passband_hz(self) -> float:
        fraction = 0.8 if self.stage_factors else 1.0
        return self.output_sample_rate_hz * 0.5 * fraction


def build_resample_plan(input_rate: float, max_output_rate: float) -> AntiAliasResamplePlan:
    if not math.isfinite(input_rate) or input_rate <= 0:
        raise ValueError("采样频率必须为有限正数")
    if not math.isfinite(max_output_rate) or max_output_rate <= 0:
        raise ValueError("显示频率必须为有限正数")
    required_factor = max(1, math.ceil(input_rate / max_output_rate))
    stage_count = max(1, math.ceil(math.log(required_factor, 16)))
    factors = []
    remaining_factor = required_factor
    if required_factor > 1:
        for remaining_stages in range(stage_count, 0, -1):
            factor = max(2, min(16, math.ceil(remaining_factor ** (1 / remaining_stages))))
            factors.append(factor)
            remaining_factor = math.ceil(remaining_factor / factor)
    return AntiAliasResamplePlan(float(input_rate), tuple(factors))


class _StreamingDecimationStage:
    def __init__(self, factor: int):
        self.factor = factor
        order, frequency = signal.ellipord(0.8 / factor, 1 / factor, 0.1, 80)
        self.sos = signal.ellip(order, 0.1, 80, frequency, btype="lowpass", output="sos")
        self.zi: np.ndarray | None = None
        self.phase = 0

    def process(self, values: np.ndarray) -> np.ndarray:
        if not values.size:
            return values
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * values[0]
        filtered, self.zi = signal.sosfilt(self.sos, values, zi=self.zi)
        start = (self.factor - 1 - self.phase) % self.factor
        self.phase = (self.phase + values.size) % self.factor
        return filtered[start::self.factor]


class StreamingAntiAliasResampler:
    """有状态低通后降采样，跨输入块保持滤波状态和抽取相位。"""

    def __init__(self):
        self.plan: AntiAliasResamplePlan | None = None
        self._stages: list[_StreamingDecimationStage] = []

    @property
    def output_sample_rate_hz(self) -> float:
        return self.plan.output_sample_rate_hz if self.plan else 0.0

    @property
    def passband_hz(self) -> float:
        return self.plan.passband_hz if self.plan else 0.0

    def configure(self, input_rate: float, max_output_rate: float) -> bool:
        plan = build_resample_plan(input_rate, max_output_rate)
        changed = self.plan != plan
        self.plan = plan
        if changed:
            self._stages = [_StreamingDecimationStage(factor) for factor in plan.stage_factors]
        return changed

    def initialize(self, mean_value: float) -> None:
        """用首秒实测均值初始化，避免首个随机采样值引入长瞬态。"""
        for stage in self._stages:
            stage.zi = signal.sosfilt_zi(stage.sos) * mean_value
            stage.phase = 0

    def process(self, values: np.ndarray) -> np.ndarray:
        if self.plan is None:
            raise RuntimeError("请先设置显示采样频率")
        output = np.asarray(values, dtype=np.float64)
        if output.ndim != 1:
            raise ValueError("振动通道必须是一维数组")
        for stage in self._stages:
            output = stage.process(output)
        return output


class PlotBuffer:
    """仅增量写入最新显示样本，绘图取最多两段数组视图。"""

    def __init__(self, input_rate: float, window_s: float, max_points: int = MAX_PLOT_POINTS):
        if window_s <= 0 or max_points < 2:
            raise ValueError("时间窗口必须为正数，绘图点数至少为 2")
        plan = build_resample_plan(input_rate, max_points / window_s)
        self.output_sample_rate_hz = plan.output_sample_rate_hz
        self.passband_hz = plan.passband_hz
        self.window_s = float(window_s)
        self.capacity = min(max_points, math.ceil(window_s * self.output_sample_rate_hz))
        self.x = np.arange(-self.capacity + 1, 1, dtype=np.float64) / self.output_sample_rate_hz
        self.y = np.empty(self.capacity, dtype=np.float64)
        self.start = 0
        self.count = 0

    def append_display(self, values: np.ndarray) -> None:
        samples = np.asarray(values, dtype=np.float64)
        if samples.ndim != 1:
            raise ValueError("振动通道必须是一维数组")
        size = samples.size
        if not size:
            return
        if size >= self.capacity:
            self.y[:] = samples[-self.capacity:]
            self.start = 0
            self.count = self.capacity
            return
        end = (self.start + self.count) % self.capacity
        overflow = max(0, self.count + size - self.capacity)
        self.start = (self.start + overflow) % self.capacity
        self.count = min(self.capacity, self.count + size)
        first_count = min(size, self.capacity - end)
        self.y[end:end + first_count] = samples[:first_count]
        self.y[:size - first_count] = samples[first_count:]

    def display_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if not self.count:
            return []
        if self.count < self.capacity:
            return [(self.x[-self.count:], self.y[:self.count])]
        if not self.start:
            return [(self.x, self.y)]
        first_count = self.capacity - self.start
        return [(self.x[:first_count], self.y[self.start:]),
                (self.x[first_count:], self.y[:self.start])]
