"""主轴回转精度布局预览；曲线为绘图样例，不执行采集、信号处理或训练。"""

from math import exp, pi, sin

from PyQt6.QtCore import QDateTime, QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.resources import DISPLAY, fit_dialog


class SpindleRotationPage(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("SpindleRotationPage")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        canvas = QWidget()
        canvas.setObjectName("SpindleCanvas")
        self.setWidget(canvas)
        columns = QHBoxLayout(canvas)
        columns.setContentsMargins(0, 0, 8, 0)
        columns.setSpacing(14)
        columns.addWidget(self._results_view(), 13)
        columns.addWidget(self._settings_view(), 7)

    @staticmethod
    def _note(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("robotNote", True)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _button(text: str, callback, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("robotAction", True)
        button.setProperty("primary", primary)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _planned_button(self, text: str, message: str, primary=True) -> QPushButton:
        button = self._button(text, lambda: self._log(message, "WARN"), primary)
        button.setToolTip(message)
        return button

    def _log(self, message: str, level="INFO") -> None:
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.process_log.appendPlainText(f"{timestamp} [{level}] {message}")

    def _results_view(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("SpindleResults")
        body = QVBoxLayout(panel)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel("主轴回转精度监控")
        title.setProperty("monitorTitle", True)
        heading.addWidget(title)
        heading.addStretch()
        badge = QLabel("示意数据")
        badge.setToolTip("图中曲线仅用于布局预览，非实测数据。")
        badge.setObjectName("SpindleDemoBadge")
        heading.addWidget(badge)
        body.addLayout(heading)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.result_values = {}
        for key, title, hint in (
            ("analysis", "解析评价", ""),
            ("network", "网络评分", ""),
            ("temperature", "温度", "°C"),
            ("current", "电流", "A"),
        ):
            metric = QFrame()
            metric.setProperty("axisMetric", True)
            content = QVBoxLayout(metric)
            content.setContentsMargins(8, 10, 8, 10)
            content.setSpacing(4)
            label = QLabel(title)
            label.setProperty("axisTitle", True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value = QLabel("—")
            value.setProperty("spindleValue", True)
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_values[key] = value
            note = self._note(hint)
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            for widget in (label, value, note):
                content.addWidget(widget)
            metrics.addWidget(metric, 1)
        body.addLayout(metrics)
        self.conclusion = QLabel("尚未评估")
        self.conclusion.setObjectName("SpindleConclusion")
        self.conclusion.setWordWrap(True)
        status = QHBoxLayout()
        status.addWidget(self.conclusion, 1)
        for name, series in (("━ 正常", "normal"), ("━ 本次", "current")):
            entry = QLabel(name)
            entry.setProperty("spindleSeries", series)
            status.addWidget(entry)
        body.addLayout(status)

        charts = QGridLayout()
        charts.setSpacing(14)
        self.signal_select = QToolButton()
        self.signal_select.setObjectName("SpindleSignalSelect")
        self.signal_select.setToolTip("选择信号类型与通道")
        self.signal_select.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.signal_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.signal_menu = QMenu(self.signal_select)
        self.signal_menu.setProperty("spindleMenu", True)
        self.signal_actions = QActionGroup(self)
        for title, kind in (
            ("振动", "vibration"),
            ("温度", "temperature"),
            ("电流", "current"),
        ):
            submenu = self.signal_menu.addMenu(title)
            submenu.setProperty("spindleMenu", True)
            for index, channel in enumerate(DISPLAY["spindle_preview_channels"][kind]):
                action = submenu.addAction(channel)
                action.setData((kind, title, channel, index))
                action.setCheckable(True)
                self.signal_actions.addAction(action)
        self.signal_select.setMenu(self.signal_menu)
        self.signal_actions.triggered.connect(self._change_signal)
        self.plot_titles = {}
        signal_area, self.signal_plot = self._plot_area(
            "采集信号", "vibration", self.signal_select
        )
        spectrum_area, self.spectrum_plot = self._plot_area("振动频谱", "spectrum")
        energy_area, self.energy_plot = self._plot_area("频带能量对比", "energy")
        network_area, self.network_plot = self._plot_area(
            "网络辅助评估", "distribution"
        )
        for row, column, area in (
            (0, 0, signal_area),
            (0, 1, spectrum_area),
            (1, 0, energy_area),
            (1, 1, network_area),
        ):
            charts.addWidget(area, row, column)
        charts.setColumnStretch(0, 1)
        charts.setColumnStretch(1, 1)
        charts.setRowStretch(0, 1)
        charts.setRowStretch(1, 1)
        body.addLayout(charts, 1)
        self._change_signal(self.signal_actions.actions()[0])
        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(self._button("检测历史", self._show_history, True))
        footer.addWidget(self._button("预警记录", self._show_alerts, True))
        body.addLayout(footer)
        return panel

    def _plot_area(
        self, title: str, kind: str, selector=None
    ) -> tuple[QWidget, QWidget]:
        area = QWidget()
        body = QVBoxLayout(area)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        heading = QHBoxLayout()
        label = QLabel(title)
        label.setProperty("robotSectionTitle", True)
        label.setMinimumHeight(36)
        self.plot_titles[kind] = label
        heading.addWidget(label)
        heading.addStretch()
        if selector is not None:
            heading.addWidget(selector)
        body.addLayout(heading)
        plot = SpindlePreviewPlot(kind)
        body.addWidget(plot, 1)
        return area, plot

    def _change_signal(self, action: QAction) -> None:
        kind, title, channel, index = action.data()
        action.setChecked(True)
        self.signal_select.setText(f"{title} · {channel}")
        self.signal_select.setAccessibleName(f"采集信号：{title}，{channel} 通道")
        self.signal_plot.kind = kind
        self.signal_plot.channel_index = index
        self.signal_plot.setAccessibleName(f"{title} {channel} 通道示意图")
        self.signal_plot.update()
        if kind == "vibration":
            for plot, name in (
                (self.spectrum_plot, "振动频谱"),
                (self.energy_plot, "频带能量"),
            ):
                self.plot_titles[plot.kind].setText(f"{name} · {channel}")
                plot.channel_index = index
                plot.setAccessibleName(f"{name} {channel} 通道示意图")
                plot.update()

    def _settings_view(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("SpindleSettings")
        panel.setMinimumWidth(332)
        body = QVBoxLayout(panel)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(14)
        title = QLabel("检测设置")
        title.setProperty("monitorSettingsTitle", True)
        body.addWidget(title)

        acquisition = self._section(body, "采集方案")
        acquisition.addWidget(self._note("转速：—    时长：—\n设备未连接"))
        acquisition.addWidget(self._button("采集设置", self._show_acquisition))

        training = self._section(body, "正常样本与网络")
        training.addWidget(self._note("正常样本：0 批    模型：未训练"))
        self.normal_review = QCheckBox("人工确认正常")
        self.normal_review.setToolTip(
            "仅将人工校核正常、工况匹配的本次数据加入训练集。"
        )
        self.normal_review.setObjectName("SpindleNormalReview")
        training.addWidget(self.normal_review)
        actions = QHBoxLayout()
        self.add_sample_button = self._planned_button(
            "加入训练集",
            "训练集接口尚未接入；本次正常标记仅作界面预览，未写入任何样本。",
        )
        self.add_sample_button.setEnabled(False)
        self.normal_review.toggled.connect(self.add_sample_button.setEnabled)
        actions.addWidget(self.add_sample_button)
        actions.addWidget(
            self._planned_button(
                "训练网络", "训练功能尚未接入，未启动网络训练或更新模型。"
            )
        )
        training.addLayout(actions)
        training.addWidget(self._button("样本与模型", self._show_samples))

        thresholds = self._section(body, "判定阈值")
        thresholds.addWidget(self._note("网络 / 综合：未设置"))
        thresholds.addWidget(self._button("阈值设置", self._show_thresholds))

        label = QLabel("运行日志")
        label.setProperty("robotSectionTitle", True)
        body.addWidget(label)
        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("SpindleProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setPlaceholderText("暂无日志")
        self.process_log.setMinimumHeight(100)
        self.process_log.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        body.addWidget(self.process_log, 1)
        actions = QFrame()
        actions.setObjectName("SpindleActions")
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(
            self._planned_button(
                "采集数据", "采集接口尚未接入，未开始采集；主轴转动由原设备软件控制。"
            ),
            1,
        )
        row.addWidget(
            self._planned_button(
                "导入数据", "导入接口尚未接入，未读取振动、温度或电流数据。"
            ),
            1,
        )
        row.addWidget(self._button("清空日志", self.process_log.clear, True), 1)
        row.addWidget(
            self._planned_button(
                "开始评估",
                "信号分析与网络推理均未接入，未生成评分或预警；演示曲线不参与评估。",
            ),
            1,
        )
        body.addWidget(actions)
        return panel

    @staticmethod
    def _section(parent: QVBoxLayout, title: str) -> QVBoxLayout:
        frame = QFrame()
        frame.setProperty("settingsGroup", True)
        body = QVBoxLayout(frame)
        body.setContentsMargins(0, 0, 0, 12)
        body.setSpacing(8)
        label = QLabel(title)
        label.setProperty("robotSectionTitle", True)
        body.addWidget(label)
        parent.addWidget(frame)
        return body

    def _show_form(self, title: str, fields: list[tuple[str, str]], hint: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        body = QVBoxLayout(dialog)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(14)
        body.addWidget(self._note("设置预览 · 暂不保存"))
        form = QFormLayout()
        form.setSpacing(12)
        for name, placeholder in fields:
            field = QLineEdit()
            field.setProperty("robotInput", True)
            field.setAccessibleName(name)
            field.setPlaceholderText(placeholder)
            form.addRow(name, field)
        body.addLayout(form)
        body.addWidget(self._note(hint))
        body.addStretch()
        body.addWidget(
            self._button("关闭", dialog.reject), 0, Qt.AlignmentFlag.AlignRight
        )
        fit_dialog(dialog, 560, 360)
        dialog.exec()

    def _show_acquisition(self) -> None:
        self._show_form(
            "主轴采集设置",
            [
                ("指定转速 / r/min", "未设置"),
                ("采集时长 / s", "未设置"),
                ("振动采样率 / Hz", "未设置"),
            ],
            "按规定工况同步采集振动、温度和电流。此处不控制主轴运动。",
        )

    def _show_thresholds(self) -> None:
        self._show_form(
            "主轴判定阈值",
            [
                ("网络评分阈值", "未设置"),
                ("综合评价阈值", "未设置"),
            ],
            "阈值用于网络评分与综合评价。",
        )

    def _show_records(self, title: str, columns: list[str], hint: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        body = QVBoxLayout(dialog)
        body.setContentsMargins(18, 18, 18, 18)
        body.addWidget(self._note(hint))
        table = QTableWidget(0, len(columns))
        table.setProperty("robotTable", True)
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        body.addWidget(table, 1)
        body.addWidget(self._note("暂无记录"))
        body.addWidget(
            self._button("关闭", dialog.reject), 0, Qt.AlignmentFlag.AlignRight
        )
        fit_dialog(dialog, 820, 420)
        dialog.exec()

    def _show_samples(self) -> None:
        self._show_records(
            "正常样本与网络模型",
            ["采集批次", "指定转速", "采集时间", "人工校核", "入集状态"],
            "正常样本：0 批；网络尚未训练，模型版本：—。日常评估不会自动加入训练集。",
        )

    def _show_history(self) -> None:
        self._show_records(
            "主轴检测历史",
            ["检测时间", "指定转速", "解析评价", "网络评分", "模型版本"],
            "追溯每次检测的工况、解析结果及辅助评分。",
        )

    def _show_alerts(self) -> None:
        self._show_records(
            "主轴预警记录",
            ["预警时间", "异常指标", "解析依据", "网络评分", "处理状态"],
            "后续关联频谱、频带能量与人工复核结论；当前尚未生成预警。",
        )


class SpindlePreviewPlot(QWidget):
    """原生 Qt 占位图；各曲线独立绘制，不是 FFT、能量计算或网络输出。"""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.channel_index = 0
        self.setMinimumHeight(160)
        self.setAccessibleName(f"主轴 {kind} 示意图，非实测数据")
        samples = [index / 240 for index in range(241)]
        # 以下数值仅用于展示线形、柱形与分布叠加的排版。
        self.demo_series = {
            "vibration": (
                [
                    0.45 * sin(2 * pi * 25 * x)
                    + 0.18 * sin(2 * pi * 67 * x)
                    + 0.08 * sin(2 * pi * 11 * x)
                    for x in samples
                ],
            ),
            "temperature": (
                [41 + 3 * (1 - exp(-3 * x)) + 0.12 * sin(18 * x) for x in samples],
            ),
            "current": (
                [2.4 + 0.1 * sin(24 * x) + 0.04 * sin(83 * x) for x in samples],
            ),
            "spectrum": tuple(
                [
                    0.025
                    + 0.012 * sin(110 * x) ** 2
                    + first * exp(-(((x - 0.12) / 0.015) ** 2))
                    + second * exp(-(((x - 0.24) / 0.022) ** 2))
                    + 0.12 * exp(-(((x - 0.48) / 0.026) ** 2))
                    for x in samples
                ]
                for first, second in ((0.52, 0.2), (0.85, 0.44))
            ),
            "energy": ([0.22, 0.36, 0.18, 0.13], [0.31, 0.65, 0.27, 0.2]),
            "distribution": tuple(
                [exp(-0.5 * ((x - mean) / spread) ** 2) for x in samples]
                for mean, spread in ((0.26, 0.075), (0.48, 0.11))
            ),
        }

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font())
        colors = DISPLAY["colors"]
        painter.fillRect(self.rect(), QColor(colors["background"]))
        fm = painter.fontMetrics()
        height = fm.height()
        low, high, xmax, ylabel, xlabel = {
            "vibration": (-1, 1, 10, "归一化幅值", "时间 / s"),
            "temperature": (40, 45, 10, "温度 / °C", "时间 / s"),
            "current": (2, 3, 10, "电流 / A", "时间 / s"),
            "spectrum": (0, 1, 1000, "归一化幅值", "频率 / Hz"),
            "energy": (0, 1, 4, "相对能量", "频带"),
            "distribution": (0, 1.1, 1, "相对密度", "归一化重建误差"),
        }[self.kind]
        margin = fm.horizontalAdvance("−1.0") + 8
        right_margin = max(12, fm.horizontalAdvance(f"{xmax:g}") / 2 + 4)
        plot = QRectF(
            margin,
            height * 1.8,
            self.width() - margin - right_margin,
            self.height() - height * 4.2,
        )
        painter.setPen(QColor(colors["inactive"]))
        painter.drawText(
            QRectF(8, 3, self.width() - 16, height), Qt.AlignmentFlag.AlignLeft, ylabel
        )
        y_steps = 4 if plot.height() >= height * 5 else 2
        for step in range(y_steps + 1):
            y = plot.bottom() - plot.height() * step / y_steps
            painter.setPen(QPen(QColor(colors["border"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(colors["inactive"]))
            value = low + (high - low) * step / y_steps
            painter.drawText(
                QRectF(0, y - height / 2, margin - 6, height),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value:.2f}".rstrip("0").rstrip("."),
            )
        painter.setPen(QPen(QColor(colors["placeholder"]), 1))
        painter.drawLine(plot.topLeft(), plot.bottomLeft())
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        series = self.demo_series[self.kind]
        for series_index, values in enumerate(series):
            # 给占位通道不同的幅值，仅供观察菜单切换，不是通道换算。
            baseline = {"temperature": 40, "current": 2}.get(self.kind, 0)
            values = [
                baseline + (value - baseline) * 0.82**self.channel_index
                for value in values
            ]
            color = QColor(
                colors["axis_y"]
                if len(series) == 2 and series_index == 0
                else colors["action"]
            )
            if self.kind == "energy":
                width = plot.width() / 13
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                for index, value in enumerate(values):
                    x = (
                        plot.left()
                        + (index + 0.5) * plot.width() / 4
                        + (series_index - 1) * width
                    )
                    bar_height = plot.height() * value
                    painter.drawRect(
                        QRectF(x, plot.bottom() - bar_height, width * 0.85, bar_height)
                    )
            else:
                path = QPainterPath()
                for index, value in enumerate(values):
                    point = QPointF(
                        plot.left() + index * plot.width() / (len(values) - 1),
                        plot.bottom() - (value - low) / (high - low) * plot.height(),
                    )
                    if index == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
                if self.kind == "distribution":
                    area = QPainterPath(path)
                    area.lineTo(plot.bottomRight())
                    area.lineTo(plot.bottomLeft())
                    area.closeSubpath()
                    fill = QColor(color)
                    fill.setAlpha(30)
                    painter.fillPath(area, fill)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, max(1.2, height / 10)))
                painter.drawPath(path)
        painter.setPen(QColor(colors["inactive"]))
        x_steps = 4 if plot.width() >= fm.horizontalAdvance("1000") * 6 else 2
        labels = (
            [f"B{index + 1}" for index in range(4)]
            if self.kind == "energy"
            else [f"{xmax * index / x_steps:g}" for index in range(x_steps + 1)]
        )
        for index, label in enumerate(labels):
            fraction = (index + 0.5) / 4 if self.kind == "energy" else index / x_steps
            x = plot.left() + plot.width() * fraction
            label_left = min(max(0, x - margin / 2), self.width() - margin)
            painter.drawText(
                QRectF(label_left, plot.bottom() + 2, margin, height),
                Qt.AlignmentFlag.AlignHCenter,
                label,
            )
        painter.drawText(
            QRectF(0, self.height() - height - 2, self.width(), height),
            Qt.AlignmentFlag.AlignHCenter,
            xlabel,
        )
