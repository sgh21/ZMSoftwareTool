"""加工过程实时监测：通道布局和实测振动的连续回放。"""

from time import perf_counter

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QSize, Qt, QThreadPool, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLayout, QMessageBox, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from app.resources import DISPLAY, PROJECT_ROOT, UiScale, fit_dialog, make_button, make_note
from app.tasks import ServiceTask
from core.algorithms.realtime_vibration import PlotBuffer
from core.services.vibration_replay import WINDOWS, VibrationReplay, load_vibration_recording


CHANNELS = (
    ("acc1", "前轴承 Y · ACC1"),
    ("acc2", "前轴承 X · ACC2"),
    ("acc3", "台面 Y · ACC3"),
    ("ntc1", "轴承温度 · NTC1"),
    ("rtd1", "环境温度 · RTD1"),
    ("speed", "主轴转速"),
)
VIBRATION_TITLES = {"acc2": "前轴X", "acc1": "前轴Y", "acc3": "台面"}


class RealtimeTimeAxis(QWidget):
    """波形框外的公共时间轴，按实际绘图区位置对齐。"""

    def __init__(self, parent):
        super().__init__(parent)
        self.window_s = 5
        self.reference_plot = None
        self.reference_widget = None
        self.ui_scale = 1.0
        self.setFixedHeight(32)
        self.hide()

    def set_window(self, window_s):
        self.window_s = window_s
        self.update()

    def paintEvent(self, event):
        if self.reference_widget is None:
            return
        plot = self.reference_plot
        if plot is not None and plot.isVisible():
            bounds = plot.getViewBox().sceneBoundingRect()
            left = self.mapFromGlobal(plot.viewport().mapToGlobal(plot.mapFromScene(bounds.topLeft()))).x()
            right = self.mapFromGlobal(plot.viewport().mapToGlobal(plot.mapFromScene(bounds.topRight()))).x()
        else:
            # 尚未接入数据的温度、转速框也为纵轴预留同样的宽度。
            widget = self.reference_widget
            bounds = widget.rect().adjusted(round(56 * self.ui_scale) + 1, 0, -1, 0)
            left = self.mapFromGlobal(widget.mapToGlobal(bounds.topLeft())).x()
            right = self.mapFromGlobal(widget.mapToGlobal(bounds.topRight())).x()
        scale = self.ui_scale
        painter = QPainter(self)
        painter.setPen(QColor(DISPLAY["colors"]["border"]))
        line_y = round(8 * scale)
        # 单位横排在轴的右端，留一小段空隙避免与轴线重叠。
        painter.drawLine(left, line_y, right - round(32 * scale), line_y)
        metrics = self.fontMetrics()
        text_y = line_y + round(6 * scale) + metrics.ascent()
        for index in range(6):
            value = self.window_s * (index - 5) / 5
            x = round(left + (value + self.window_s) / self.window_s * (right - left))
            painter.drawLine(x, line_y, x, line_y + round(4 * scale))
            painter.setPen(QColor(DISPLAY["colors"]["inactive"]))
            text = f"{value:g}"
            painter.drawText(x - metrics.horizontalAdvance(text) // 2, text_y, text)
            painter.setPen(QColor(DISPLAY["colors"]["border"]))
        painter.setPen(QColor(DISPLAY["colors"]["inactive"]))
        painter.drawText(right - metrics.horizontalAdvance("t/s") - round(5 * scale),
                         line_y + metrics.ascent() // 2, "t/s")


class RealtimeVibrationPlot(QWidget):
    """复用曲线与环形数组，只绘制最近窗口内的有限点。"""

    def __init__(self, color, parent):
        super().__init__(parent)
        self.buffers = {}
        self.buffer = None
        self.clipped = False
        self.plot = pg.PlotWidget(background="w")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.plot.showGrid(x=True, y=True, alpha=.18)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.enableAutoRange(x=False, y=False)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setTextPen(DISPLAY["colors"]["inactive"])
            self.plot.getAxis(axis).setPen(DISPLAY["colors"]["border"])
        bottom_axis = self.plot.getAxis("bottom")
        bottom_axis.setStyle(showValues=False)
        bottom_axis.setHeight(0)
        self.curves = [self.plot.plot(pen=pg.mkPen(color, width=1), antialias=False, connect="all")
                       for _ in range(2)]
        for curve in self.curves:
            curve.setClipToView(True)
            curve.setSkipFiniteCheck(True)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 0, 0)
        self.clipping_note = QLabel("峰值超出显示范围")
        self.clipping_note.setProperty("realtimeClipping", True)
        layout.addWidget(self.clipping_note, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.clipping_note.hide()

    def configure(self, recording, channel, window_s):
        self.ranges = {window: recording.ranges[window][channel] for window in WINDOWS}
        self.buffers = {window: PlotBuffer(recording.sample_rate_hz, window) for window in WINDOWS}
        self.plot.setLabel("left", units=recording.unit, color=DISPLAY["colors"]["inactive"])
        self.set_window(window_s)

    def set_window(self, window_s):
        if self.buffers:
            self.buffer = self.buffers[window_s]
            self.plot.setXRange(-window_s, 0, padding=0)
            self.plot.setYRange(*self.ranges[window_s], padding=0)
            self.refresh()

    def append(self, values, channel):
        for window_s, samples in values.items():
            self.buffers[window_s].append_display(samples[channel])

    def refresh(self):
        segments = self.buffer.display_segments() if self.buffer is not None else []
        for index, curve in enumerate(self.curves):
            if index < len(segments):
                curve.setData(*segments[index])
            else:
                curve.clear()
        limits = self.plot.viewRange()[1]
        clipped = any(values.size and (values.min() < limits[0] or values.max() > limits[1]) for _, values in segments)
        if clipped != self.clipped:
            self.clipped = clipped
            self.clipping_note.setVisible(clipped)


class RealtimeMonitorDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("实时监测")
        self.setObjectName("RealtimeMonitorDialog")
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.ui_scale = parent.window().ui_scale
        self.two_columns = None
        self.recording = None
        self.replay = None
        self.replay_state = "empty"
        self.load_task = None
        self.elapsed_before_pause = 0.0
        self.play_started = 0.0
        self.last_frame_at = 0.0
        self.frame_stats = {}
        self.replay_timer = QTimer(self)
        self.replay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.replay_timer.setInterval(33)
        self.replay_timer.timeout.connect(self._update_replay)
        body = QVBoxLayout(self)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(10)
        heading = QHBoxLayout()
        title = QLabel("加工过程实时监测")
        title.setProperty("monitorTitle", True)
        heading.addWidget(title)
        self.simulate_button = make_button("模拟采集", self._toggle_replay, True)
        heading.addWidget(self.simulate_button)
        self.stop_button = make_button("停止", self._stop_replay)
        self.stop_button.setEnabled(False)
        heading.addWidget(self.stop_button)
        heading.addStretch()
        self.replay_status = make_note("数据状态：尚未接入")
        heading.addWidget(self.replay_status)
        body.addLayout(heading)

        selection = QFrame()
        selection.setObjectName("RealtimeChannelSelection")
        controls = QVBoxLayout(selection)
        controls.setContentsMargins(16, 8, 16, 8)
        controls.setSpacing(6)
        display_row = QHBoxLayout()
        label = QLabel("显示通道")
        label.setProperty("robotSectionTitle", True)
        display_row.addWidget(label)
        display_row.addWidget(make_note("显示窗口"))
        self.window_select = QComboBox()
        self.window_select.setProperty("robotInput", True)
        for window in WINDOWS:
            self.window_select.addItem(f"最近 {window:g} 秒", window)
        self.window_select.setCurrentIndex(self.window_select.findData(5))
        self.window_select.currentIndexChanged.connect(self._set_plot_window)
        display_row.addWidget(self.window_select)
        display_row.addStretch()
        self.waveform_check = QCheckBox("振动波形")
        self.spectrum_check = QCheckBox("振动频谱")
        for check in (self.waveform_check, self.spectrum_check):
            check.setChecked(True)
            display_row.addWidget(check)
        controls.addLayout(display_row)
        self.channel_grid = QGridLayout()
        self.channel_grid.setHorizontalSpacing(24)
        self.channel_grid.setVerticalSpacing(4)
        self.channel_checks = {}
        for key, text in CHANNELS:
            check = QCheckBox(text)
            check.setChecked(True)
            self.channel_checks[key] = check
        controls.addLayout(self.channel_grid)
        body.addWidget(selection)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.plot_scroll.setObjectName("RealtimePlotScroll")
        self.plot_canvas = QWidget()
        self.plot_canvas.setObjectName("RealtimePlotCanvas")
        self.plot_grid = QGridLayout(self.plot_canvas)
        self.plot_grid.setContentsMargins(0, 0, 0, 0)
        self.plot_grid.setSpacing(8)
        self.plot_grid.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.plot_scroll.setWidget(self.plot_canvas)
        body.addWidget(self.plot_scroll, 1)

        self.plot_frames = {}
        self.wave_plots = {}
        self.plot_messages = {}
        self.plot_contents = {}
        for index, (key, text) in enumerate(CHANNELS[:3]):
            for kind, name in (("waveform", "波形"), ("spectrum", "频谱")):
                self.plot_frames[f"{key}_{kind}"] = self._placeholder(VIBRATION_TITLES[key])
                self.plot_frames[f"{key}_{kind}"].setToolTip(f"{text} · {name}")
            frame = self.plot_frames[f"{key}_waveform"]
            self.wave_plots[key] = RealtimeVibrationPlot(
                (DISPLAY["colors"]["action"], DISPLAY["colors"]["axis_y"], DISPLAY["colors"]["axis_z"])[index], frame)
            self.plot_contents[frame].layout().addWidget(self.wave_plots[key], 0, 0)
            self.wave_plots[key].hide()
        self.plot_frames["temperature"] = self._placeholder("温度趋势")
        self.plot_frames["speed"] = self._placeholder("主轴转速")
        self.time_axis = RealtimeTimeAxis(self.plot_canvas)
        self.trend_time_axis = RealtimeTimeAxis(self.plot_canvas)
        self.plot_headers = {}
        for kind, text in (("waveform", "波形"), ("spectrum", "频谱")):
            label = QLabel(text, self.plot_canvas)
            label.setProperty("realtimeHeading", True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedHeight(24)
            label.hide()
            self.plot_headers[kind] = label
        legend = QWidget()
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(12)
        self.temperature_legends = {}
        for (key, text), color in zip(CHANNELS[3:5], ("action", "axis_z")):
            label = QLabel(f'<span style="color:{DISPLAY["colors"][color]}">━━</span> {text}')
            label.setProperty("realtimeLegend", True)
            legend_layout.addWidget(label)
            self.temperature_legends[key] = label
        self.plot_contents[self.plot_frames["temperature"]].layout().addWidget(
            legend, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.empty_state = make_note("请选择要显示的通道或振动图表类型")
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setParent(self.plot_canvas)
        self.empty_state.hide()

        # 保存设计尺寸，后续跟随主窗口缩放时始终从原值计算。
        self.ui_metrics = UiScale(self)
        fit_dialog(self, 1400, 900)
        self.ui_metrics.apply(self.ui_scale)
        self.setMinimumSize(QSize(round(640 * self.ui_scale), round(420 * self.ui_scale)))
        available = self.screen().availableGeometry()
        self.resize(self.size().boundedTo(QSize(round(available.width() * .9), round(available.height() * .9))))
        self.move(available.center() - self.rect().center())
        for check in (*self.channel_checks.values(), self.waveform_check, self.spectrum_check):
            check.toggled.connect(self._rebuild_layout)
        self.plot_scroll.viewport().installEventFilter(self)
        self._rebuild_layout()

    def _placeholder(self, title):
        frame = QFrame(self.plot_canvas)
        frame.setProperty("realtimePlot", True)
        frame.hide()
        frame.setMinimumHeight(120)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        label = QLabel("\n".join(title))
        label.setObjectName("PlotTitle")
        label.setProperty("realtimeVerticalTitle", True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedWidth(26)
        label.setToolTip(title)
        label.setAccessibleName(title)
        layout.addWidget(label)
        content = QWidget()
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content, 1)
        message = make_note("数据显示区域\n尚未接入数据")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(message, 0, 0)
        self.plot_messages[frame] = message
        self.plot_contents[frame] = content
        return frame

    def _toggle_replay(self):
        if self.replay_state == "playing":
            self.elapsed_before_pause = self.replay_elapsed_s()
            self.replay_timer.stop()
            self.replay_state = "paused"
            self.simulate_button.setText("继续模拟")
            self.replay_status.setText("模拟采集已暂停")
        elif self.replay_state == "paused":
            self.play_started = perf_counter()
            self.last_frame_at = self.play_started
            self.replay_state = "playing"
            self.simulate_button.setText("暂停模拟")
            self.replay_timer.start()
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择完整的振动采集目录", str(PROJECT_ROOT / "data/spindle_error"))
            if folder:
                self.load_recording(folder)

    def load_recording(self, folder):
        self._stop_replay()
        self.replay_state = "loading"
        self.simulate_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        task = ServiceTask(lambda progress: load_vibration_recording(folder, progress))
        self.load_task = task
        task.signals.progress.connect(lambda percent, message: self.replay_status.setText(f"准备数据 {percent}% · {message}"))
        task.signals.completed.connect(self._recording_ready)
        task.signals.failed.connect(self._replay_failed)
        QThreadPool.globalInstance().start(task)

    def _recording_ready(self, recording):
        self.load_task = None
        self.recording = recording
        window_s = self.window_select.currentData()
        for channel, (key, _) in enumerate(CHANNELS[:3]):
            frame = self.plot_frames[f"{key}_waveform"]
            self.plot_messages[frame].hide()
            self.wave_plots[key].configure(recording, channel, window_s)
            self.wave_plots[key].show()
        self._set_plot_window()
        self.replay = VibrationReplay(recording)
        self.replay.start()
        self.elapsed_before_pause = 0.0
        self.play_started = self.last_frame_at = perf_counter()
        self.replay_state = "playing"
        self.simulate_button.setEnabled(True)
        self.simulate_button.setText("暂停模拟")
        self.stop_button.setEnabled(True)
        self.replay_timer.start()
        if not self.isVisible():
            self._toggle_replay()

    def _set_plot_window(self):
        window_s = self.window_select.currentData()
        self.time_axis.set_window(window_s)
        self.trend_time_axis.set_window(window_s)
        for key, text in CHANNELS[:3]:
            plot = self.wave_plots[key]
            plot.set_window(window_s)
            if self.recording is not None:
                title = self.plot_frames[f"{key}_waveform"].findChild(QLabel, "PlotTitle")
                title.setToolTip(f"{text} · 最近 {window_s} 秒\n"
                                 f"原始 {self.recording.sample_rate_hz:g} Hz；显示 {self.recording.display_rates[window_s]:g} Hz；"
                                 f"显示带宽 ≤ {self.recording.passbands[window_s]:g} Hz。纵轴按实测常规幅值固定；"
                                 "超出范围的峰值明确标记，数据和原始采样率保留。")

    def replay_elapsed_s(self):
        return self.elapsed_before_pause + (perf_counter() - self.play_started if self.replay_state == "playing" else 0.0)

    def _update_replay(self):
        started = perf_counter()
        elapsed = min(self.replay_elapsed_s(), self.recording.duration_s)
        try:
            chunks = self.replay.poll(elapsed)
        except Exception as error:
            self._replay_failed(str(error))
            return
        for chunk in chunks:
            for channel, (key, _) in enumerate(CHANNELS[:3]):
                self.wave_plots[key].append(chunk.values, channel)
        if chunks:
            for plot in self.wave_plots.values():
                if plot.isVisible():
                    plot.refresh()
        source_s = self.replay.samples_consumed / self.recording.sample_rate_hz
        self.frame_stats = {"interval_ms": (started - self.last_frame_at) * 1000,
                            "update_ms": (perf_counter() - started) * 1000,
                            "source_s": source_s, "lag_ms": (elapsed - source_s) * 1000}
        self.last_frame_at = started
        self.replay_status.setText(f"1× 模拟 · {int(source_s) // 60:02d}:{int(source_s) % 60:02d}"
                                  f" / {int(self.recording.duration_s) // 60:02d}:{int(self.recording.duration_s) % 60:02d}")
        if self.replay.finished:
            self.elapsed_before_pause = self.recording.duration_s
            self.replay_state = "finished"
            self.replay_timer.stop()
            self.simulate_button.setText("模拟采集")
            self.replay_status.setText(f"完整回放结束 · {self.recording.duration_s / 60:.2f} min")

    def _stop_replay(self):
        self.replay_timer.stop()
        if self.replay is not None:
            self.replay.close()
        self.replay = None
        self.replay_state = "empty"
        self.simulate_button.setText("模拟采集")
        self.stop_button.setEnabled(False)
        self.replay_status.setText("模拟采集已停止")

    def shutdown(self):
        self._stop_replay()

    def _replay_failed(self, message):
        self.load_task = None
        self._stop_replay()
        self.replay_state = "error"
        self.simulate_button.setEnabled(True)
        self.replay_status.setText("数据准备或回放失败")
        QMessageBox.warning(self, "模拟采集失败", message)

    def closeEvent(self, event):
        if self.replay_state == "playing":
            self._toggle_replay()
        super().closeEvent(event)

    def reject(self):
        if self.replay_state == "playing":
            self._toggle_replay()
        super().reject()

    def _rebuild_layout(self):
        self.two_columns = self.plot_scroll.viewport().width() >= round(800 * self.ui_scale)
        selection_columns = 3 if self.two_columns else 2
        while self.channel_grid.count():
            self.channel_grid.takeAt(0)
        ordered_keys = (*VIBRATION_TITLES, *(key for key, _ in CHANNELS[3:]))
        for index, key in enumerate(ordered_keys):
            check = self.channel_checks[key]
            self.channel_grid.addWidget(check, index // selection_columns, index % selection_columns)
        for column in range(3):
            self.channel_grid.setColumnStretch(column, int(column < selection_columns))

        while self.plot_grid.count():
            self.plot_grid.takeAt(0).widget().hide()
        for row in range(self.plot_grid.rowCount()):
            self.plot_grid.setRowStretch(row, 0)
        columns = 2 if self.two_columns else 1
        self.plot_grid.setColumnStretch(0, 1)
        self.plot_grid.setColumnStretch(1, int(self.two_columns))
        rows = []
        selected = [key for key in VIBRATION_TITLES if self.channel_checks[key].isChecked()]
        kinds = [kind for kind, check in (("waveform", self.waveform_check), ("spectrum", self.spectrum_check))
                 if check.isChecked()]
        if selected and "waveform" in kinds:
            self.time_axis.reference_plot = self.wave_plots[selected[-1]].plot
            self.time_axis.reference_widget = self.plot_contents[self.plot_frames[f"{selected[-1]}_waveform"]]
        if selected and kinds:
            if self.two_columns and len(kinds) == 2:
                rows.append([self.plot_headers[kind] for kind in kinds])
                rows.extend([self.plot_frames[f"{key}_{kind}"] for kind in kinds] for key in selected)
                rows.append([self.time_axis])
            else:
                for kind in kinds:
                    rows.append([self.plot_headers[kind]])
                    rows.extend([self.plot_frames[f"{key}_{kind}"]] for key in selected)
                    if kind == "waveform":
                        rows.append([self.time_axis])
        for key, label in self.temperature_legends.items():
            label.setVisible(self.channel_checks[key].isChecked())
        trend_keys = []
        if any(self.channel_checks[key].isChecked() for key, _ in CHANNELS[3:5]):
            rows.append([self.plot_frames["temperature"]])
            trend_keys.append("temperature")
        if self.channel_checks["speed"].isChecked():
            rows.append([self.plot_frames["speed"]])
            trend_keys.append("speed")
        if trend_keys:
            self.trend_time_axis.reference_widget = self.plot_contents[self.plot_frames[trend_keys[-1]]]
            rows.append([self.trend_time_axis])
        graph_frames = set(self.plot_frames.values())
        for row, plots in enumerate(rows or [[self.empty_state]]):
            for column, plot in enumerate(plots):
                self.plot_grid.addWidget(plot, row, column, 1, columns if len(plots) == 1 else 1)
                plot.show()
            self.plot_grid.setRowStretch(row, int(plots[0] in graph_frames or plots[0] is self.empty_state))
        for wave in self.wave_plots.values():
            wave.plot.getAxis("left").setWidth(round(56 * self.ui_scale))
            if wave.isVisible() and wave.buffer is not None:
                wave.refresh()
        for axis in (self.time_axis, self.trend_time_axis):
            axis.ui_scale = self.ui_scale
            axis.update()

    def sync_scale(self, scale):
        if scale != self.ui_scale:
            self.ui_scale = scale
            self.ui_metrics.apply(scale)
            self.setMinimumSize(round(640 * scale), round(420 * scale))
            self._rebuild_layout()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            if (watched.width() >= round(800 * self.ui_scale)) != self.two_columns:
                self._rebuild_layout()
        return super().eventFilter(watched, event)
