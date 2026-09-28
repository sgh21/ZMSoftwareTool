"""机器人定位监控：按用户约定的区域 1—8 组织结果、设置和日志。"""

import json
from math import isfinite
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, QDateTime, QPointF, QRectF, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.resources import DISPLAY, fit_dialog
from core.services.position_monitoring_service import (
    METRIC_LABELS, PositionMonitoringService, assess_metric, metric_values, read_document,
)


PREDICTION_COLUMNS = ["点位", "X / mm", "Y / mm", "Z / mm", "空间指标 / mm", "判定"]
RESULT_COLUMNS = ["测点 / 方向", "X / mm", "Y / mm", "Z / mm", "空间指标 / mm", "到达次数"]
METRIC_HINTS = {
    "absolute_change": "各轴误差大小及空间 AP 的变化；正值为增大，负值为减小。",
    "repeatability_change": "各轴 3σ 及空间 RP 的变化；正值为散布增大，负值为减小。",
    "repeatability": "各轴为 3σ 补充统计，空间指标为 RP；按同点、同方向重复到达计算。",
}


class TaskSignals(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class ServiceTask(QRunnable):
    """在工作线程调用服务，所有界面更新经信号回到主线程。"""

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = TaskSignals()

    def run(self):
        try:
            self.signals.completed.emit(self.operation(self.signals.progress.emit))
        except Exception as error:
            self.signals.failed.emit(str(error))


class RobotPositionPage(QWidget):
    def __init__(self, service=None) -> None:
        super().__init__()
        self.service = service or PositionMonitoringService()
        self.processing_points = self.service.settings.get("processing_points", [])
        self.result = None
        self.history_batches = None
        self.task = None
        self.mutation_buttons = []
        self.setObjectName("RobotPositionPage")
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("RobotPageScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        canvas = QWidget()
        canvas.setObjectName("RobotPageCanvas")
        self.scroll_area.setWidget(canvas)
        layout = QVBoxLayout(canvas)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        self.columns = QHBoxLayout()
        self.columns.setSpacing(14)
        self.columns.addWidget(self._results_view(), 13)
        self.columns.addWidget(self._settings_sidebar(), 7)
        layout.addLayout(self.columns, 1)
        page_layout.addWidget(self.scroll_area, 1)
        self._refresh_settings()
        self._fill_table(self.point_table, self._prediction_rows())
        self._render_result()
        self.append_log("页面已就绪，等待手眼参数、基准及观测数据。")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        stacked = self.width() < 1000
        direction = QBoxLayout.Direction.TopToBottom if stacked else QBoxLayout.Direction.LeftToRight
        if self.columns.direction() != direction:
            self.columns.setDirection(direction)
            self.columns.setStretch(0, 0 if stacked else 13)
            self.columns.setStretch(1, 0 if stacked else 7)

    def _mutation_button(self, text, callback, primary=False):
        button = self._button(text, callback, primary)
        self.mutation_buttons.append(button)
        return button

    def append_log(self, message: str, level: str = "INFO") -> None:
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.process_log.appendPlainText(f"{timestamp} [{level}] {message}")

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

    def _planned_button(
        self, text: str, description: str, primary=False
    ) -> QPushButton:
        message = f"{text}：{description}".replace("\n", " ")
        button = self._button(
            text,
            lambda: self.append_log(message, "WARN"),
            primary,
        )
        button.setToolTip(description)
        return button

    def _results_view(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("RobotResults")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        title = QLabel("机器人定位精度")
        title.setObjectName("RobotPageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        heading = QHBoxLayout()
        heading.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)
        heading.addStretch()
        self.result_metric = QComboBox()
        self.result_metric.setObjectName("RobotMetricSelect")
        self.result_metric.setProperty("robotInput", True)
        self.result_metric.setAccessibleName("评价指标")
        self.result_metric.setToolTip("切换整个结果区的评价指标")
        self.result_metric.setMinimumWidth(240)
        self.result_metric.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for key, label in METRIC_LABELS.items():
            self.result_metric.addItem(label, key)
            self.result_metric.setItemData(
                self.result_metric.count() - 1, METRIC_HINTS[key], Qt.ItemDataRole.ToolTipRole,
            )
        self.result_metric.setCurrentIndex(2)
        heading.addWidget(self.result_metric, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(heading)
        summary = QFrame()
        summary.setObjectName("Region1")
        body = QVBoxLayout(summary)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        timestamps = QHBoxLayout()
        self.evaluation_time_label = self._note("评价更新时间：—")
        self.baseline_time_label = self._note("基准建立时间：—")
        timestamps.addWidget(self.evaluation_time_label, 1)
        timestamps.addWidget(self.baseline_time_label, 1)
        body.addLayout(timestamps)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.axis_values = {}
        self.axis_titles = {}
        self.axis_cards = {}
        for axis, title in (
            ("X", "X 方向"),
            ("Y", "Y 方向"),
            ("Z", "Z 方向"),
            ("distance", "空间指标"),
        ):
            metric = QFrame()
            metric.setProperty("axisMetric", True)
            metric.setProperty("axis", axis)
            metric.setProperty("overLimit", False)
            self.axis_cards[axis] = metric
            content = QVBoxLayout(metric)
            content.setContentsMargins(10, 12, 10, 12)
            content.setSpacing(4)
            name = QLabel(title)
            name.setProperty("axisTitle", True)
            self.axis_titles[axis] = name
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content.addWidget(name)
            value_row = QHBoxLayout()
            value_row.setSpacing(6)
            value = MetricValueLabel("—")
            value.setProperty("axisValue", True)
            self.axis_values[axis] = value
            value_row.addWidget(value, 1)
            value_row.addWidget(QLabel("mm"), 0, Qt.AlignmentFlag.AlignBottom)
            content.addLayout(value_row)
            metrics.addWidget(metric, 1)
        body.addLayout(metrics)
        layout.addWidget(summary)
        layout.addWidget(self._trend_view(), 1)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("RobotDetailTabs")
        self.detail_tabs.setProperty("region", 3)
        self.detail_tabs.setAccessibleName("定位监控结果详情")
        self.detail_tabs.addTab(self._measured_results(), "逐点结果")
        self.detail_tabs.addTab(self._point_results(), "加工点位预测")
        self.detail_tabs.addTab(self._observation_view(), "观测图像")
        self.detail_tabs.addTab(self._alarm_view(), "报警与维护")
        layout.addWidget(self.detail_tabs, 1)
        self.result_metric.currentIndexChanged.connect(self._render_result)
        return panel

    def _metric_mode(self):
        return self.result_metric.currentData()

    def _metric_name(self):
        return METRIC_LABELS[self._metric_mode()]

    def _measured_results(self):
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 10, 0, 0)
        self.measured_table = self._table(RESULT_COLUMNS)
        self.measured_table.setMinimumHeight(114)
        self.measured_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        body.addWidget(self.measured_table, 1)
        footer = QHBoxLayout()
        self.result_hint = self._note("等待评估")
        footer.addWidget(self.result_hint, 1)
        footer.addWidget(self._button("历史记录", self._show_history, True))
        footer.addWidget(self._button("展开明细", self._show_result_details, True))
        body.addLayout(footer)
        return page

    def _trend_view(self) -> QWidget:
        area = QWidget()
        area.setObjectName("Region2")
        body = QVBoxLayout(area)
        body.setContentsMargins(0, 2, 0, 0)
        body.setSpacing(8)
        self.trend_title = QLabel("定位精度趋势")
        self.trend_title.setProperty("robotSectionTitle", True)
        body.addWidget(self.trend_title)
        self.trend_metric = QComboBox()
        self.trend_metric.setProperty("robotInput", True)
        self.trend_metric.setAccessibleName("曲线显示方式")
        self.trend_metric.addItem("XYZ 三轴", "xyz")
        self.trend_metric.addItem("空间指标", "distance")
        legend = QHBoxLayout()
        self.trend_axis_title = self._note("基座 XYZ / mm")
        self.trend_axis_title.setWordWrap(False)
        legend.addWidget(self.trend_axis_title, 0, Qt.AlignmentFlag.AlignVCenter)
        legend.addStretch()
        self.trend_legend = {}
        for axis, name in (("X", "X"), ("Y", "Y"), ("Z", "Z"), ("distance", "空间")):
            label = QLabel(f"━ {name}")
            label.setProperty("axis", axis)
            label.setVisible(axis != "distance")
            self.trend_legend[axis] = label
            legend.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        legend.addWidget(self.trend_metric, 0, Qt.AlignmentFlag.AlignVCenter)
        body.addLayout(legend)
        self.trend_chart = PositionTrendChart()
        body.addWidget(self.trend_chart, 1)
        self.trend_metric.currentIndexChanged.connect(self._change_trend_metric)
        caption = self._note("评估时间 / 天")
        caption.setToolTip("相对于基准建立时间的经过天数，1 天 = 24 小时")
        caption.setAlignment(Qt.AlignmentFlag.AlignRight)
        body.addWidget(caption)
        return area

    def _change_trend_metric(self) -> None:
        distance = self.trend_metric.currentData() == "distance"
        for axis, label in self.trend_legend.items():
            label.setVisible((axis == "distance") == distance)
        self.trend_chart.mode = self.trend_metric.currentData()
        self._refresh_history()
        self.trend_chart.update()
        self.append_log(f"趋势指标已切换为{self.trend_metric.currentText()}。")

    def _point_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)
        self.prediction_title = self._note("")
        layout.addWidget(self.prediction_title)
        self.point_table = self._table(PREDICTION_COLUMNS)
        self.point_table.setMinimumHeight(114)
        self.point_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        layout.addWidget(self.point_table, 1)
        footer = QHBoxLayout()
        message = self._note("预测模型尚未启用")
        message.setWordWrap(False)
        footer.addWidget(message)
        footer.addStretch()
        footer.addWidget(self._button("历史记录", self._show_history, True))
        footer.addWidget(self._button("展开明细", self._show_points, True))
        layout.addLayout(footer)
        return page

    def _observation_view(self) -> QWidget:
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 10, 0, 0)
        body.setSpacing(8)
        row = QHBoxLayout()
        self.observation_source = QComboBox()
        self.observation_source.setProperty("robotInput", True)
        self.observation_source.setAccessibleName("观测图像来源")
        self.observation_source.addItems(["本次观测", "基准观测"])
        row.addWidget(self.observation_source)
        self.observation_sample = QComboBox()
        self.observation_sample.setProperty("robotInput", True)
        self.observation_sample.setAccessibleName("观测样本")
        row.addWidget(self.observation_sample, 1)
        body.addLayout(row)
        image_area = QFrame()
        image_area.setObjectName("ObservationImage")
        image_area.setMinimumHeight(116)
        image_layout = QVBoxLayout(image_area)
        self.observation_image = ObservationImage()
        image_layout.addWidget(self.observation_image, 1)
        self.image_message = QLabel("尚无本次标志点图像")
        self.image_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.image_message)
        body.addWidget(image_area, 1)
        self.observation_source.currentIndexChanged.connect(self._refresh_observations)
        self.observation_sample.currentIndexChanged.connect(self._show_observation)
        return page

    def _alarm_view(self) -> QWidget:
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 14, 0, 0)
        self.alarm_status = QLabel("报警状态：尚未评估")
        body.addWidget(self.alarm_status)
        self.alarm_message = self._note("按当前所选指标显示判定及超限项。")
        body.addWidget(self.alarm_message)
        body.addStretch()
        actions = QHBoxLayout()
        actions.addWidget(self._button("报警记录", self._show_alarms, True))
        actions.addWidget(self._button("维护记录", self._show_maintenance, True))
        actions.addStretch()
        body.addLayout(actions)
        return page

    def _settings_sidebar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("RobotSettings")
        panel.setMinimumWidth(332)
        body = QVBoxLayout(panel)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(14)
        title = QLabel("检测设置")
        title.setObjectName("RobotSettingsTitle")
        heading = QHBoxLayout()
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self._mutation_button("调试", self._show_debug))
        body.addLayout(heading)

        self.status_lights = {}
        hand_eye = self._settings_section(body, 4, "机器人手眼参数", "parameter")
        actions = QHBoxLayout()
        actions.addWidget(
            self._mutation_button("加载参数", self._load_parameters)
        )
        actions.addWidget(
            self._mutation_button("参数设置", self._show_parameters)
        )
        hand_eye.addLayout(actions)

        baseline = self._settings_section(body, 5, "测量基准", "baseline")
        actions = QHBoxLayout()
        actions.addWidget(
            self._mutation_button("建立基准", self._create_baseline)
        )
        actions.addWidget(
            self._mutation_button("选择基准", self._select_baseline)
        )
        baseline.addLayout(actions)

        conditions = self._settings_section(body, 7, "判定与点位设置", "conditions")
        actions = QHBoxLayout()
        actions.addWidget(self._mutation_button("阈值设置", self._show_settings))
        actions.addWidget(self._mutation_button("点位管理", self._show_point_editor))
        conditions.addLayout(actions)

        log_area = QFrame()
        log_area.setObjectName("Region8")
        log_layout = QVBoxLayout(log_area)
        log_layout.setContentsMargins(0, 0, 0, 0)
        heading = QHBoxLayout()
        title = QLabel("运行日志")
        title.setProperty("robotSectionTitle", True)
        heading.addWidget(title)
        heading.addStretch()
        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("RobotProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setMinimumHeight(110)
        self.process_log.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.process_log.setPlaceholderText("操作过程、提示与错误将在这里显示")
        log_layout.addLayout(heading)
        log_layout.addWidget(self.process_log, 1)
        body.addWidget(log_area, 1)
        body.addWidget(self._action_bar())
        return panel

    def _action_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Region6")
        bar.setAccessibleName("定位监控操作栏")
        actions = QHBoxLayout(bar)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        actions.addWidget(
            self._planned_button(
                "采集靶标",
                "相机接口尚未接入，未开始采集。后续按与基准相同的程序、固定位姿拍摄靶标。",
                True,
            ),
            1,
        )
        actions.addWidget(
            self._mutation_button("导入观测", self._import_observations, True),
            1,
        )
        actions.addWidget(self._button("清空日志", self.process_log.clear, True), 1)
        actions.addWidget(
            self._mutation_button("评估精度", self._evaluate, True),
            1,
        )
        return bar

    def _settings_section(
        self, parent: QVBoxLayout, number: int, title: str, status_key: str
    ) -> QVBoxLayout:
        group = QFrame()
        group.setObjectName(f"Region{number}")
        group.setProperty("settingsGroup", True)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setProperty("robotSectionTitle", True)
        heading = QHBoxLayout()
        heading.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        heading.addStretch()
        light = QLabel()
        light.setProperty("statusLight", True)
        light.setFixedSize(12, 12)
        light.setAccessibleName(f"{title}状态")
        self.status_lights[status_key] = light
        heading.addWidget(light, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(heading)
        parent.addWidget(group)
        return layout

    @staticmethod
    def _table(columns: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setProperty("robotTable", True)
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return table

    @staticmethod
    def _fill_table(table: QTableWidget, rows) -> None:
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)

    def _prediction_rows(self) -> list[list[str]]:
        return [
            [point["id"], "—", "—", "—", "—", "待预测"]
            for point in self.processing_points
        ]

    def _show_point_editor(self) -> None:
        dialog = PointEditor(self)
        fit_dialog(dialog, 820, 460)
        dialog.exec()

    def _show_records(self, title: str, columns: list[str], hint: str, rows=()) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(self._note(hint))
        table = self._table(columns)
        self._fill_table(table, rows)
        layout.addWidget(table, 1)
        layout.addWidget(
            self._note(f"共 {len(rows)} 条记录" if rows else "暂无记录")
        )
        layout.addWidget(
            self._button("关闭", dialog.reject), 0, Qt.AlignmentFlag.AlignRight
        )
        fit_dialog(dialog, 820, 420)
        dialog.exec()

    def _show_points(self) -> None:
        self._show_records(
            f"{self._metric_name()} · 加工点位预测",
            PREDICTION_COLUMNS,
            "根据当前评估结果预测各加工点位的定位精度；不表示已在这些点位完成实测。",
            self._prediction_rows(),
        )

    def _show_history(self) -> None:
        records = self.service.list_history()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self._metric_name()} · 检测历史")
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note("双击记录查看已保存结果；趋势只比较相同基准和参数版本。"))
        table = self._table(["测量批次", "评估时间", "基准版本", "手眼版本", "结论"])
        self._fill_table(table, [[*[str(row.get(key) or "—") for key in
                                  ("batch_id", "created_at", "baseline_id", "parameter_version")],
                                 assess_metric(row, self._metric_mode())["status"]]
                                for row in records])
        layout.addWidget(table)

        def select(row, _column):
            self.result = records[row]
            self.history_batches = {"current": None, "baseline": None}
            for source, field in (("current", "current_batch_path"), ("baseline", "baseline_path")):
                path = self.result.get(field)
                if path:
                    try:
                        document = read_document(path)
                        self.history_batches[source] = document["batch"] if source == "baseline" else document
                    except (OSError, ValueError, KeyError) as error:
                        self.append_log(f"历史图像快照未能载入：{error}", "WARN")
            self._render_result()
            self._refresh_observations()
            self.append_log(f"查看历史结果：{self.result['id']}。")
            dialog.accept()

        table.cellDoubleClicked.connect(select)
        layout.addWidget(self._button("关闭", dialog.reject))
        fit_dialog(dialog, 960, 460)
        dialog.exec()

    def _show_alarms(self) -> None:
        rows = []
        for result in self.service.list_history():
            for alarm in assess_metric(result, self._metric_mode())["alarms"]:
                rows.append([result["created_at"], result["batch_id"],
                             f"{alarm['point_id']} / {alarm['direction_id']} / {alarm['axis']}",
                             f"{alarm['value']:.4f} / {alarm['threshold']:.4f}", "待复测确认"])
        self._show_records(
            f"{self._metric_name()} · 报警记录",
            ["报警时间", "测量批次", "超限方向", "指标值 / 阈值", "处理状态"],
            "显示所选指标的历史超限项，使用各次评估保存的阈值。",
            rows,
        )

    def _show_maintenance(self) -> None:
        self._show_records(
            "维护记录",
            ["维护时间", "关联报警", "维护内容", "处理人员", "复测批次"],
            "维护记录尚未接入；请按设备维护流程处理，复测后核对精度指标。",
        )

    def _show_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("定位精度阈值")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addWidget(self._note("三项指标分别设置上限；逐测点、接近方向判定，留空不判定该项。"))
        tabs = QTabWidget()
        fields = {}
        for mode, label in METRIC_LABELS.items():
            tab = QWidget()
            form = QFormLayout(tab)
            form.setSpacing(12)
            fields[mode] = {}
            thresholds = self.service.settings.get("metric_thresholds", {}).get(mode, {})
            for key, name in (("X", "X 方向阈值 / mm"), ("Y", "Y 方向阈值 / mm"),
                              ("Z", "Z 方向阈值 / mm"), ("distance", "空间指标阈值 / mm")):
                field = QLineEdit()
                field.setProperty("robotInput", True)
                field.setObjectName(f"threshold_{mode}_{key}")
                field.setPlaceholderText("未设置")
                field.setText("" if thresholds.get(key) is None else str(thresholds[key]))
                field.setAccessibleName(f"{label} · {name}")
                fields[mode][key] = field
                form.addRow(name, field)
            form.addRow(self._note(METRIC_HINTS[mode]))
            tabs.addTab(tab, label)
        tabs.setCurrentIndex(self.result_metric.currentIndex())
        layout.addWidget(tabs)
        layout.addWidget(self._note("退化量直接与上限比较，负值表示改善；历史结果保留评估时的阈值。"))
        error_label = self._note("")
        layout.addWidget(error_label)
        layout.addStretch()

        def save():
            try:
                values = {mode: {key: float(field.text()) if field.text().strip() else None
                                 for key, field in mode_fields.items()}
                          for mode, mode_fields in fields.items()}
                if any(value is not None and (not isfinite(value) or value < 0)
                       for mode_values in values.values() for value in mode_values.values()):
                    raise ValueError("阈值必须是有限非负数。")
                self.service.save_settings({"metric_thresholds": values})
            except (ValueError, OSError) as error:
                error_label.setText(str(error))
                return
            self._refresh_settings()
            self._render_result()
            self.append_log("阈值已保存；新阈值将在下次评估时用于判定。")
            dialog.accept()

        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("取消", dialog.reject))
        actions.addWidget(self._button("保存", save, True))
        layout.addLayout(actions)
        fit_dialog(dialog, 760, 470)
        dialog.exec()

    @staticmethod
    def _number(value):
        return "—" if value is None else f"{value:.4f}"

    @staticmethod
    def _display_time(value):
        parsed = QDateTime.fromString(str(value), Qt.DateFormat.ISODateWithMs)
        return parsed.toLocalTime().toString("yyyy-MM-dd HH:mm:ss") if parsed.isValid() else "—"

    def _baseline_created_at(self, result=None):
        baseline = self.service.baseline
        if result is None:
            return baseline.get("created_at") if baseline else None
        if not result.get("baseline_id"):
            return None
        if result.get("baseline_created_at"):
            return result["baseline_created_at"]
        if baseline and baseline["id"] == result["baseline_id"]:
            return baseline.get("created_at")
        # 旧历史没有时间快照，只读取该历史自己的基准，不借用其他基准。
        if result.get("baseline_path"):
            try:
                baseline = read_document(result["baseline_path"])
            except (OSError, ValueError):
                return None
            if baseline.get("id") == result["baseline_id"]:
                return baseline.get("created_at")
        return None

    def _run_task(self, title, operation, completed):
        if self.task is not None:
            self.append_log("当前任务仍在执行，请等待完成。", "WARN")
            return
        self.append_log(f"{title}开始。")
        for button in self.mutation_buttons:
            button.setEnabled(False)
        self._task_completed_callback = completed
        self.task = ServiceTask(operation)
        self.task.signals.completed.connect(self._task_completed)
        self.task.signals.failed.connect(self._task_failed)
        self.task.signals.progress.connect(self._task_progress)
        QThreadPool.globalInstance().start(self.task)

    def _task_progress(self, percent, message):
        self.append_log(f"{percent}% · {message}")

    def _task_completed(self, result):
        callback = self._task_completed_callback
        self._finish_task()
        callback(result)

    def _task_failed(self, message):
        self._finish_task()
        self.append_log(message, "ERROR")

    def _finish_task(self):
        self.task = None
        self._task_completed_callback = None
        for button in self.mutation_buttons:
            button.setEnabled(True)

    def _refresh_settings(self):
        parameters = self.service.parameters
        loaded = parameters.get("hand_eye") is not None
        parameter_state = "ready" if loaded else (
            "partial" if parameters.get("camera_matrix") is not None else "missing"
        )
        self._set_status_light("parameter", parameter_state,
            f"参数状态：{'已加载' if loaded else '未加载'} · 相机 → TCP\n"
            f"参数版本：{parameters.get('version') or '—'}"
        )
        baseline = self.service.baseline
        if baseline:
            matching = loaded and baseline.get("parameters", {}).get("version") == parameters.get("version")
            self._set_status_light("baseline", "ready" if matching else "partial",
                f"基准：{baseline.get('label') or baseline['id']}\n"
                f"编号：{baseline['id']}\n建立时间：{self._display_time(baseline.get('created_at'))}\n"
                + ("与当前参数匹配" if matching else "与当前参数不匹配，请重新建立或选择基准")
            )
        else:
            self._set_status_light("baseline", "missing", "尚未建立测量基准")
        metric_thresholds = self.service.settings.get("metric_thresholds", {})
        counts = {mode: sum(metric_thresholds.get(mode, {}).get(axis) is not None
                            for axis in self.axis_cards) for mode in METRIC_LABELS}
        point_count = len(self.processing_points)
        state = "missing"
        if all(count == 4 for count in counts.values()) and point_count:
            state = "ready"
        elif any(counts.values()) or point_count:
            state = "partial"
        self._set_status_light("conditions", state,
            "\n".join(f"{label}：已设置 {counts[mode]}/4 项阈值"
                      for mode, label in METRIC_LABELS.items())
            + f"\n加工点位：{point_count} 个"
        )

    def _set_status_light(self, key, state, details):
        light = self.status_lights[key]
        light.setProperty("state", state)
        status = {"missing": "未配置", "partial": "配置不全或不匹配", "ready": "已就绪"}[state]
        light.setToolTip(f"{status}\n{details}")
        light.setAccessibleDescription(f"{status}；{details}")
        light.style().unpolish(light)
        light.style().polish(light)
        light.update()

    def _invalidate_result(self):
        self.result = None
        self.history_batches = None
        self._render_result()
        self._refresh_settings()
        self._refresh_observations()

    def _load_parameters(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载视觉与手眼参数", "", "参数文件 (*.json *.yaml *.yml)")
        if not path:
            return
        try:
            self.service.load_parameters(path)
        except (ValueError, TypeError, OSError, KeyError) as error:
            self.append_log(f"参数加载失败：{error}", "ERROR")
            return
        self._invalidate_result()
        self.append_log(f"参数已加载：{Path(path).name}；请重新导入观测并建立或选择匹配基准。")

    def _show_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("视觉与手眼参数")
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note(
            "平移和棋盘格尺寸单位为 mm；hand_eye 为相机到被评估 TCP 的 4×4 变换。\n"
            "camera_matrix 为 3×3 内参，dist_coeffs 为畸变，board_grid 为内角点列数、行数。"
        ))
        editor = QPlainTextEdit()
        editor.setAccessibleName("视觉与手眼参数 JSON")
        editor.setPlainText(json.dumps(self.service.parameters, ensure_ascii=False, indent=2))
        layout.addWidget(editor, 1)
        error_label = self._note("")
        layout.addWidget(error_label)

        def save():
            try:
                self.service.save_parameters(json.loads(editor.toPlainText()))
            except (ValueError, TypeError, OSError, KeyError) as error:
                error_label.setText(str(error))
                return
            self._invalidate_result()
            self.append_log("参数已保存为新版本；原参数和历史基准仍保留。")
            dialog.accept()

        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("关闭", dialog.reject))
        actions.addWidget(self._button("保存新版本", save, True))
        layout.addLayout(actions)
        fit_dialog(dialog, 780, 610)
        dialog.exec()

    def _import_observations(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入观测清单", "", "观测清单 (*.json *.yaml *.yml)")
        if not path:
            return
        self._invalidate_result()
        self._run_task("导入观测", lambda progress: self.service.load_observations(path, progress),
                       self._observations_loaded)

    def _observations_loaded(self, batch):
        self._refresh_observations()
        self._render_result()
        self.append_log(f"已导入 {len(batch['samples'])} 个观测样本。")
        for warning in batch.get("warnings", []):
            self.append_log(warning, "WARN")

    def _create_baseline(self):
        self._run_task("建立基准", lambda _progress: self.service.create_baseline(),
                       self._baseline_created)

    def _baseline_created(self, baseline):
        self._invalidate_result()
        self.append_log(f"基准已建立：{baseline['id']}。请导入复测观测。")

    def _select_baseline(self):
        entries = self.service.list_baselines()
        if not entries:
            self.append_log("尚无保存的基准，请先导入初始观测并建立基准。", "WARN")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("选择测量基准")
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note("选择后恢复基准关联参数，请重新导入复测观测。"))
        choice = QComboBox()
        choice.setProperty("robotInput", True)
        for entry in entries:
            choice.addItem(f"{entry['created_at']} · {entry.get('label') or entry['id']}", entry["path"])
        layout.addWidget(choice)
        error_label = self._note("")
        layout.addWidget(error_label)

        def select():
            try:
                baseline = self.service.select_baseline(choice.currentData())
            except (ValueError, OSError, KeyError) as error:
                error_label.setText(str(error))
                return
            self._invalidate_result()
            self.append_log(f"基准已选择：{baseline['id']}；请重新导入复测观测。")
            dialog.accept()

        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("取消", dialog.reject))
        actions.addWidget(self._button("选择", select, True))
        layout.addLayout(actions)
        fit_dialog(dialog, 700, 230)
        dialog.exec()

    def _evaluate(self):
        allow_current_only = self._metric_mode() == "repeatability"
        self._run_task(
            "定位评估",
            lambda progress: self.service.evaluate(progress, allow_current_only=allow_current_only),
            self._evaluation_completed,
        )

    def _evaluation_completed(self, result):
        self.result = result
        self.history_batches = None
        self._render_result()
        self._refresh_observations()
        self.append_log(f"评估完成：{result['id']} · {result['status']}，结果已保存。")
        for warning in result.get("warnings", []):
            self.append_log(warning, "WARN")

    def _metric_values(self, result, group=None):
        return metric_values(result, self._metric_mode(), group)

    def _metric_rows(self):
        rows = []
        for group in self.result["groups"]:
            count = str(group["current"]["count"])
            if self._metric_mode() != "repeatability":
                initial_count = (group.get("baseline") or {}).get("count", "—")
                count = f"{initial_count} → {count}"
            rows.append([f"{group['point_id']} / {group['direction_id']}",
                         *[self._number(value) for value in self._metric_values(self.result, group)], count])
        return rows

    def _unavailable_reasons(self):
        mode = self._metric_mode()
        if mode != "repeatability" and not self.result.get("baseline_id"):
            return ["需要匹配的基准与本次观测比较结果"]
        reasons = []
        values = self._metric_values(self.result)
        if mode.startswith("repeatability"):
            periods = ("current",) if mode == "repeatability" else ("baseline", "current")
            if any((group.get(period) or {}).get("count", 0) < 2
                   for group in self.result["groups"] for period in periods):
                reasons.append("同点同方向重复到达样本不足")
            if any(group["current"].get("mean_base") is None for group in self.result["groups"]):
                reasons.append("缺参考朝向 Qᵢ，基座三轴不可用")
        elif any(value is None for value in values):
            reasons.append("缺初始绝对误差向量或参考朝向 Qᵢ")
        return reasons

    def _render_result(self):
        mode = self._metric_mode()
        name = self._metric_name()
        hint = METRIC_HINTS[mode]
        self.prediction_title.setText(f"{name} · 加工点位预测")
        assessment = assess_metric(self.result, mode) if self.result is not None else None
        thresholds = (assessment["thresholds"] if assessment else
                      self.service.settings.get("metric_thresholds", {}).get(mode, {}))
        alarm_axes = {alarm["axis"] for alarm in assessment["alarms"]} if assessment else set()
        for key, card in self.axis_cards.items():
            card.setProperty("overLimit", key in alarm_axes)
            threshold = thresholds.get(key)
            threshold_hint = "阈值未设置" if threshold is None else f"阈值：{self._number(threshold)} mm"
            source = "本次评估保存的阈值" if assessment else "下一次评估使用的阈值"
            card.setToolTip(f"{source} · {threshold_hint}" + ("\n存在逐点超限，请查看报警与维护" if key in alarm_axes else ""))
            self.axis_titles[key].setToolTip(hint)
            for label in card.findChildren(QLabel):
                label.style().unpolish(label)
                label.style().polish(label)
                label.update()
        self.measured_table.horizontalHeaderItem(4).setToolTip(hint)
        context = self.result or self.service.current_batch or {}
        baseline = self.service.baseline or {}
        baseline_time = self._baseline_created_at(self.result)
        self.evaluation_time_label.setText(
            f"评价更新时间：{self._display_time(self.result.get('created_at') if self.result else None)}"
        )
        self.baseline_time_label.setText(f"基准建立时间：{self._display_time(baseline_time)}")
        self.evaluation_time_label.setToolTip(
            f"观测批次：{context.get('batch_label') or context.get('label') or context.get('batch_id') or '—'}\n"
            f"参数版本：{context.get('parameter_version') or '—'}"
        )
        baseline_id = self.result.get("baseline_id") if self.result is not None else baseline.get("id")
        self.baseline_time_label.setToolTip(f"基准编号：{baseline_id or '—'}")
        if self.result is None:
            for label in self.axis_values.values():
                label.setText("—")
            pending = "等待本次观测并评估" if mode == "repeatability" else "需要基准与本次观测并评估"
            self.result_hint.setText(pending)
            self.result_hint.setToolTip(hint)
            self.alarm_status.setText(f"{name} · 尚未评估")
            self.alarm_message.setText("尚无该指标的判定结果。")
            self._fill_table(self.measured_table, [])
        else:
            result = self.result
            for key, value in zip(("X", "Y", "Z", "distance"), self._metric_values(result)):
                self.axis_values[key].setText(self._number(value))
            self._fill_table(self.measured_table, self._metric_rows())
            unavailable = self._unavailable_reasons()
            self.result_hint.setText("；".join(unavailable) if unavailable else "逐点等权汇总 · 单位 mm")
            warnings = result.get("warnings", [])
            self.result_hint.setToolTip(hint + "\n" + "\n".join(warnings))
            alarms = assessment["alarms"]
            alarm_count = f" · {len(alarms)} 项超限" if alarms else ""
            self.alarm_status.setText(f"{name} · {assessment['status']}{alarm_count}")
            self.alarm_message.setText(
                "；".join(f"{alarm['point_id']}/{alarm['direction_id']} {alarm['axis']}："
                         f"{alarm['value']:.4f} > {alarm['threshold']:.4f} mm" for alarm in alarms[:8])
                + "\n请核对测量条件与安装稳定性，再按设备维护流程检查并复测。"
                if alarms else "；".join(unavailable) or assessment["status"]
            )
        self._refresh_history()

    def _refresh_history(self):
        baseline_id = self.result.get("baseline_id") if self.result else (
            self.service.baseline.get("id") if self.service.baseline else None
        )
        history = self.service.list_history()
        parameter_version = self.result.get("parameter_version") if self.result else self.service.parameters.get("version")
        context = self.result or self.service.current_batch or {}
        origin = QDateTime.fromString(str(self._baseline_created_at(self.result)), Qt.DateFormat.ISODateWithMs)
        self.trend_chart.empty_message = (
            "暂无历史评估记录" if origin.isValid() else "缺少基准建立时间，无法显示相对时间趋势"
        )
        groups = context.get("groups", context.get("samples", []))
        point_directions = {(group["point_id"], group["direction_id"]) for group in groups}
        records = []
        for result in history:
            if (result.get("parameter_version") != parameter_version
                    or result.get("baseline_id") != baseline_id):
                continue
            if any(context.get(field) and result.get(field) != context[field]
                   for field in ("program_id", "target_id")):
                continue
            record_points = {(group["point_id"], group["direction_id"]) for group in result["groups"]}
            if point_directions and record_points != point_directions:
                continue
            timestamp = QDateTime.fromString(str(result.get("created_at")), Qt.DateFormat.ISODateWithMs)
            if not origin.isValid() or not timestamp.isValid():
                continue
            values = self._metric_values(result)
            records.append({"days": origin.msecsTo(timestamp) / 86400000,
                            **dict(zip(("X", "Y", "Z", "distance"), values))})
        self.trend_chart.set_history(records)
        axis = "空间" if self.trend_metric.currentData() == "distance" else "基座 XYZ"
        self.trend_axis_title.setText(f"{axis} / mm")

    def _refresh_observations(self):
        if self.history_batches is not None:
            source = "current" if self.observation_source.currentIndex() == 0 else "baseline"
            batch = self.history_batches[source]
        elif self.observation_source.currentIndex() == 0:
            batch = self.service.current_batch
        else:
            batch = self.service.baseline["batch"] if self.service.baseline else None
        self.observation_sample.blockSignals(True)
        self.observation_sample.clear()
        for sample in (batch or {}).get("samples", []):
            self.observation_sample.addItem(
                f"{sample['point_id']} / {sample['direction_id']} / {sample.get('sample_id', '')}", sample
            )
        self.observation_sample.blockSignals(False)
        self._show_observation()
        if self.history_batches is not None and batch is None:
            self.image_message.setText("此历史结果未保存图像快照或快照不可用")

    def _show_observation(self):
        sample = self.observation_sample.currentData()
        path = sample.get("image_path") if sample else None
        self.observation_image.set_image(path)
        self.image_message.setText(Path(path).name if path else "该观测没有附图" if sample else "尚无观测图像")
        if path and self.observation_image.pixmap.isNull():
            self.image_message.setText(f"图像无法读取：{Path(path).name}")
        self.image_message.setToolTip(path or "")

    def _show_result_details(self):
        if self.result is None:
            self.append_log("尚无评估结果。", "WARN")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self._metric_name()} · 评估明细")
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note(METRIC_HINTS[self._metric_mode()]))
        table = self._table(RESULT_COLUMNS)
        self._fill_table(table, self._metric_rows())
        layout.addWidget(table, 1)
        notes = QPlainTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(self._result_details_text())
        layout.addWidget(notes)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self._button("导出结果", self._export_result, True))
        actions.addWidget(self._button("关闭", dialog.reject))
        layout.addLayout(actions)
        fit_dialog(dialog, 980, 580)
        dialog.exec()

    def _result_details_text(self):
        lines = [METRIC_HINTS[self._metric_mode()],
                 "单位 mm。以下诊断用 Δp 表达在各测点的固定初始参考末端系，不能混称基座 XYZ。"]
        for group in self.result["groups"]:
            if group.get("drift_local") is None:
                continue
            components = ", ".join(self._number(value) for value in group["drift_local"])
            lines.append(f"{group['point_id']}/{group['direction_id']} 初始参考末端系 Δp (mm)：[{components}]")
        for point in self.result.get("points", []):
            if point.get("baseline_vap") is not None or point.get("current_vap") is not None:
                lines.append(
                    f"{point['point_id']} 接近方向均值最大间距："
                    f"基准 {self._number(point.get('baseline_vap'))} → 本次 {self._number(point.get('current_vap'))}；"
                    f"变化 {self._number(point.get('vap_change'))}（非同方向 RP）。"
                )
        lines.extend(self.result.get("warnings", []))
        return "\n".join(lines)

    def _export_result(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出定位结果", f"{self.result['id']}.json", "JSON (*.json)")
        if path:
            try:
                self.service.export_result(path, self.result)
            except OSError as error:
                self.append_log(f"导出失败：{error}", "ERROR")
                return
            self.append_log(f"评估结果已导出：{path}")

    def _show_debug(self):
        from app.dialogs.robot_position_debug_dialog import RobotPositionDebugDialog

        dialog = RobotPositionDebugDialog(self)
        fit_dialog(dialog, 920, 620)
        dialog.exec()


class MetricValueLabel(QLabel):
    """数值保持完整；窄卡片按可用宽度缩字，不撑宽整页。"""

    def __init__(self, text):
        super().__init__(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setFont(self.font())
        painter.setPen(self.palette().windowText().color())
        bounds = self.contentsRect()
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        scale = min(1.0, bounds.width() / max(1, text_width))
        if scale <= 0:
            return
        painter.translate(bounds.topLeft())
        painter.scale(scale, scale)
        painter.drawText(QRectF(0, 0, bounds.width() / scale, bounds.height() / scale),
                         Qt.AlignmentFlag.AlignCenter, self.text())


class ObservationImage(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)

    def set_image(self, path):
        self.pixmap = QPixmap(str(path)) if path else QPixmap()
        self.update()

    def paintEvent(self, event):
        if self.pixmap.isNull():
            return
        painter = QPainter(self)
        scaled = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap((self.width() - scaled.width()) // 2,
                           (self.height() - scaled.height()) // 2, scaled)


class PointEditor(QDialog):
    """加工点位与观测点分开保存，预测模型尚未启用。"""

    def __init__(self, page: RobotPositionPage) -> None:
        super().__init__(page)
        self.page = page
        self.setWindowTitle("加工点位管理")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(
            page._note("双击修改编号和理论坐标，可先留空坐标。应用后保存配置。")
        )
        actions = QHBoxLayout()
        for title, callback in (
            ("新增点位", self._add_point),
            ("删除点位", self._delete_points),
            ("上移", lambda: self._move_point(-1)),
            ("下移", lambda: self._move_point(1)),
            ("顺序编号", self._renumber),
        ):
            actions.addWidget(page._button(title, callback))
        actions.addStretch()
        layout.addLayout(actions)
        self.table = page._table(["编号", "理论 X / mm", "理论 Y / mm", "理论 Z / mm"])
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        page._fill_table(
            self.table,
            [
                [point["id"]]
                + [
                    "" if point[axis] is None else f"{point[axis]:g}"
                    for axis in ("x", "y", "z")
                ]
                for point in page.processing_points
            ],
        )
        layout.addWidget(self.table, 1)
        self.error_label = page._note("")
        self.error_label.setProperty("validationError", True)
        layout.addWidget(self.error_label)
        footer = QHBoxLayout()
        footer.addWidget(
            page._note("加工点位用于预测，与拍摄靶标的观测点位分别管理。"), 1
        )
        footer.addWidget(page._button("取消", self.reject))
        footer.addWidget(page._button("应用", self._apply, True))
        layout.addLayout(footer)

    def _add_point(self) -> None:
        identifiers = {
            self.table.item(row, 0).text() for row in range(self.table.rowCount())
        }
        number = 1
        while f"P{number:03}" in identifiers:
            number += 1
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((f"P{number:03}", "", "", "")):
            self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.selectRow(row)
        self.page.append_log(f"点位草稿已新增 P{number:03}，等待应用。")

    def _delete_points(self) -> None:
        rows = sorted(
            {item.row() for item in self.table.selectionModel().selectedRows()},
            reverse=True,
        )
        if not rows:
            self.page.append_log("删除点位：请先选择点位。", "WARN")
            return
        for row in rows:
            self.table.removeRow(row)
        self.page.append_log(f"点位草稿已删除 {len(rows)} 个点位，等待应用。")

    def _move_point(self, offset: int) -> None:
        row = self.table.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        for column in range(self.table.columnCount()):
            item = self.table.takeItem(row, column)
            other = self.table.takeItem(target, column)
            self.table.setItem(row, column, other)
            self.table.setItem(target, column, item)
        self.table.selectRow(target)
        self.page.append_log("点位草稿的顺序已调整，等待应用。")

    def _renumber(self) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setText(f"P{row + 1:03}")
        self.page.append_log("点位草稿已按当前行顺序重新编号，等待应用。")

    def _apply(self) -> None:
        points = []
        identifiers = set()
        for row in range(self.table.rowCount()):
            identifier = self.table.item(row, 0).text().strip()
            if not identifier or identifier in identifiers:
                self._report_error(f"第 {row + 1} 行编号为空或重复，请修改编号。")
                return
            identifiers.add(identifier)
            point = {"id": identifier}
            for column, axis in enumerate(("x", "y", "z"), 1):
                text = self.table.item(row, column).text().strip()
                try:
                    value = float(text) if text else None
                except ValueError:
                    self._report_error(
                        f"点位 {identifier} 的 {axis.upper()} 坐标不是有效数字。"
                    )
                    return
                if value is not None and not isfinite(value):
                    self._report_error(
                        f"点位 {identifier} 的 {axis.upper()} 坐标必须是有限数字。"
                    )
                    return
                point[axis] = value
            points.append(point)
        try:
            self.page.service.save_settings({**self.page.service.settings, "processing_points": points})
        except (ValueError, OSError) as error:
            self._report_error(str(error))
            return
        self.page.processing_points = points
        self.page._refresh_settings()
        self.page._fill_table(self.page.point_table, self.page._prediction_rows())
        self.page.append_log(
            f"加工点位已保存：{len(points)} 个；预测模型尚未启用。"
        )
        self.accept()

    def _report_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.page.append_log(message, "ERROR")


class PositionTrendChart(QWidget):
    """只绘制传入的历史结果，不计算视觉误差、距离或精度指标。"""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(150)
        self.mode = "xyz"
        self.history = []
        self.empty_message = "暂无历史评估记录"
        self.setAccessibleName("多次评估的定位精度趋势图")

    def set_history(self, records: list[dict]) -> None:
        """days 为相对基准的天数；X、Y、Z、distance 未计算时为 None。"""
        self.history = sorted(records, key=lambda record: record["days"])
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font())
        colors = DISPLAY["colors"]
        painter.fillRect(self.rect(), QColor(colors["background"]))
        metrics = painter.fontMetrics()
        line_height = metrics.height()
        margin = metrics.horizontalAdvance("-0.000") + line_height / 2
        plot = QRectF(
            margin,
            line_height / 2,
            self.width() - margin - line_height,
            self.height() - line_height * 2,
        )
        painter.setPen(QPen(QColor(colors["border"]), 1, Qt.PenStyle.DashLine))
        for step in range(5):
            y = plot.top() + plot.height() * step / 4
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(QColor(colors["placeholder"]), 1))
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        axes = ("distance",) if self.mode == "distance" else ("X", "Y", "Z")
        values = [
            record[axis]
            for record in self.history
            for axis in axes
            if record.get(axis) is not None
        ]
        if not values:
            painter.setPen(QColor(colors["inactive"]))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self.empty_message)
            return
        low, high = min(0, min(values)), max(0, max(values))
        if low == high:
            high = low + 1
        painter.setPen(QColor(colors["inactive"]))
        for step in range(5):
            value = high - (high - low) * step / 4
            y = plot.top() + plot.height() * step / 4
            painter.drawText(
                QRectF(0, y - line_height / 2, margin - 6, line_height),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value:.3f}",
            )
        first_day = min(0, self.history[0]["days"])
        last_day = max(0, self.history[-1]["days"])
        if first_day == last_day:
            last_day = first_day + 1
        for step in range(5):
            day = first_day + (last_day - first_day) * step / 4
            x = plot.left() + plot.width() * step / 4
            tick_width = margin
            painter.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 3))
            painter.drawText(
                QRectF(x - tick_width / 2, plot.bottom() + 3, tick_width, line_height),
                Qt.AlignmentFlag.AlignCenter,
                f"{day:.3g}",
            )
        color_keys = {
            "X": "action",
            "Y": "axis_y",
            "Z": "axis_z",
            "distance": "distance",
        }
        for axis in axes:
            painter.setPen(
                QPen(QColor(colors[color_keys[axis]]), max(1, line_height / 9))
            )
            path = QPainterPath()
            connected = False
            for record in self.history:
                value = record.get(axis)
                if value is None:
                    connected = False
                    continue
                x = plot.left() + plot.width() * (record["days"] - first_day) / (last_day - first_day)
                point = QPointF(
                    x, plot.bottom() - plot.height() * (value - low) / (high - low)
                )
                if connected:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                connected = True
                painter.drawEllipse(point, line_height / 8, line_height / 8)
            painter.drawPath(path)
