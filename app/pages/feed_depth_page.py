"""进给轴窝深导入、分组评估与独立模拟演示。"""

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QDateTime, QEvent, QPointF, QRectF, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QHeaderView, QLabel, QLayout, QLineEdit,
    QPlainTextEdit, QProgressBar, QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget,
)

from app.resources import DISPLAY, fit_dialog, make_button, make_note
from app.tasks import ServiceTask
from core.runtime_paths import data_root
from core.services.feed_depth_service import (
    evaluate_feed_depth, load_feed_depth_data, load_feed_depth_settings, save_feed_depth_settings,
    validate_feed_depth_settings,
)
from core.services.feed_depth_simulation import generate_feed_depth_simulation


def _source_label(record):
    return "模拟数据" if record.get("is_simulated") else "导入数据"


def _number(value, format_spec=".3f"):
    return format(value, format_spec) if value is not None else "—"


class FeedDepthPage(QScrollArea):
    def __init__(self, simulation_root=None, settings_path=None):
        super().__init__()
        self.simulation_root = Path(simulation_root) if simulation_root is not None else data_root() / "data/feed_depth_simulation"
        self.settings_path = Path(settings_path) if settings_path is not None else data_root() / "storage/feed_depth/settings.json"
        settings = load_feed_depth_settings(self.settings_path)
        self.simulation = None
        self.measurements = None
        self.evaluation = None
        self.selected_batch_index = -1
        self._alarm_key = None
        self.task = None
        self.pending_settings = None
        self.selected_file = ""
        self.uniform_depth = settings["uniform_depth"]
        self.depth_mode = settings["depth_mode"]
        self.error_lower_mm = settings["error_lower_mm"]
        self.error_upper_mm = settings["error_upper_mm"]
        self.setObjectName("FeedDepthPage")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        canvas = QWidget()
        canvas.setObjectName("FeedDepthCanvas")
        self.setWidget(canvas)
        self.columns = QHBoxLayout(canvas)
        self.columns.setContentsMargins(0, 0, 8, 0)
        self.columns.setSpacing(14)
        self.columns.addWidget(self._results_view(), 13)
        self.columns.addWidget(self._settings_view(), 7)
        self._update_settings_tooltip()
        self._update_actions()

    def _results_view(self):
        panel = QFrame()
        panel.setObjectName("FeedDepthResults")
        body = QVBoxLayout(panel)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel("进给轴窝深")
        title.setProperty("monitorTitle", True)
        heading.addWidget(title)
        heading.addStretch()
        self.source_badge = make_note("暂无数据")
        heading.addWidget(self.source_badge)
        body.addLayout(heading)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.result_values = {}
        for key, title, unit in (
            ("count", "有效孔数", "当前分组 / 个"),
            ("theory", "理论窝深", "当前分组 / mm"),
            ("mean", "平均窝深", "实测 / mm"),
            ("variance", "窝深方差", "总体方差 / mm²"),
        ):
            card = QFrame()
            card.setProperty("axisMetric", True)
            content = QVBoxLayout(card)
            content.setContentsMargins(8, 10, 8, 10)
            content.setSpacing(4)
            label = QLabel(title)
            label.setProperty("axisTitle", True)
            value = QLabel("—")
            value.setProperty("spindleValue", True)
            self.result_values[key] = value
            for widget in (label, value, make_note(unit)):
                widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                content.addWidget(widget)
            metrics.addWidget(card, 1)
        body.addLayout(metrics)

        plots = QVBoxLayout()
        plots.setSpacing(14)
        self.plot_frames = {}
        self.trend_charts = {}
        for key, title, y_label, x_label, hint, details in (
            ("bias", "平均偏差趋势", "平均偏差 / mm", "测量日期",
             "等待可比历史批次\n正值偏深，负值偏浅",
             "每批平均偏差 = mean(实测 − 理论)。保留正负，离零越远表示整体偏移越大；靠近零可能是改善。"),
            ("spread", "窝深标准差趋势", "标准差 / mm", "测量日期",
             "等待可比历史批次\n数值越大，孔间波动越大",
             "每批标准差 = √总体方差，与窝深同为 mm。至少两个有效孔才进入此图，单孔批次标记为样本不足。"),
        ):
            area = QVBoxLayout()
            area.setSpacing(6)
            label = QLabel(title)
            label.setProperty("robotSectionTitle", True)
            label.setToolTip(details)
            area.addWidget(label)
            frame = QFrame()
            frame.setProperty("feedDepthPlot", True)
            frame.setMinimumHeight(175)
            frame.setAccessibleName(title)
            frame.setToolTip(details)
            self.plot_frames[key] = frame
            content = QVBoxLayout(frame)
            content.setContentsMargins(10, 6, 10, 6)
            content.setSpacing(2)
            content.addWidget(make_note(y_label))
            chart = FeedDepthTrendChart("bias_mm" if key == "bias" else "stddev_mm", hint)
            chart.setAccessibleName(title)
            chart.batch_selected.connect(self._show_batch)
            self.trend_charts[key] = chart
            content.addWidget(chart, 1)
            axis_title = make_note(x_label)
            axis_title.setAlignment(Qt.AlignmentFlag.AlignRight)
            content.addWidget(axis_title)
            area.addWidget(frame, 1)
            plots.addLayout(area, 1)
        body.addLayout(plots, 1)

        footer = QHBoxLayout()
        self.history_note = make_note("暂无数据")
        footer.addWidget(self.history_note, 1)
        footer.addWidget(make_button("历史记录", self._show_history, True))
        footer.addWidget(make_button("孔位明细", self._show_hole_details, True))
        body.addLayout(footer)
        return panel

    def _show_hole_details(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("孔位窝深明细")
        body = QVBoxLayout(dialog)
        batch = self.evaluation["history"][self.selected_batch_index] if self.evaluation is not None else None
        rows = [row for row in self.evaluation["rows"]
                if (row["batch_id"], row.get("row_id", ""), row["theoretical_depth_mm"]) ==
                (batch["batch_id"], batch.get("row_id", ""), batch["theoretical_depth_mm"])] if batch else []
        body.addWidget(make_note(
            (f"{_source_label(batch)} · {batch['batch_id']} · {len(rows)} 孔"
             + ("，非真实测量。" if batch.get("is_simulated") else "。")) if batch else
            "尚未导入数据。偏差 = 实测 − 理论；缺测不补零。"
        ))
        table = QTableWidget(len(rows), 6)
        table.setProperty("robotTable", True)
        table.setAccessibleName("孔位窝深明细")
        table.setHorizontalHeaderLabels([
            "孔排", "孔号", "理论 / mm", "实测 / mm", "偏差 / mm", "数据状态",
        ])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for index, row in enumerate(rows):
            state = " · 缺测" if row["actual_depth_mm"] is None else " · 超限" if row["alarm"] else ""
            values = (row["row_id"], row["hole_id"], _number(row["theoretical_depth_mm"]),
                      _number(row["actual_depth_mm"]), _number(row["bias_mm"], "+.3f"),
                      _source_label(row) + state)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row["alarm"] and column in (3, 4):
                    item.setForeground(QColor(DISPLAY["colors"]["error"]))
                table.setItem(index, column, item)
        body.addWidget(table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        body.addWidget(buttons)
        fit_dialog(dialog, 960, 460)
        dialog.exec()

    def _show_history(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("窝深历史记录")
        body = QVBoxLayout(dialog)
        records = self.evaluation["history"] if self.evaluation is not None else []
        table = QTableWidget(len(records), 6)
        table.setProperty("robotTable", True)
        table.setHorizontalHeaderLabels(["测量日期", "孔数", "平均 / mm", "偏差 / mm", "标准差 / mm", "来源"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for index, record in enumerate(records):
            values = (record["measured_at"][:10] or "未提供", str(record["count"]), _number(record["mean_depth_mm"]),
                      _number(record["bias_mm"], "+.4f"), _number(record["stddev_mm"], ".4f"),
                      f"{_source_label(record)} · {record['batch_id']} / {record.get('row_id') or '整批'} / {record['theoretical_depth_mm']:g} mm")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if record["alarm"] and column in (2, 3):
                    item.setForeground(QColor(DISPLAY["colors"]["error"]))
                table.setItem(index, column, item)
        if records:
            table.selectRow(self.selected_batch_index)
            table.scrollToItem(table.item(self.selected_batch_index, 0))
        body.addWidget(table, 1)

        def select():
            if table.currentRow() >= 0:
                self._show_batch(table.currentRow())
                dialog.accept()

        table.cellDoubleClicked.connect(lambda _row, _column: select())
        actions = QHBoxLayout()
        actions.addStretch()
        choose = make_button("查看批次", select, True)
        choose.setEnabled(bool(records))
        actions.addWidget(choose)
        actions.addWidget(make_button("关闭", dialog.reject))
        body.addLayout(actions)
        fit_dialog(dialog, 960, 500)
        dialog.exec()

    @staticmethod
    def _section(parent, title):
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

    def _settings_view(self):
        panel = QFrame()
        panel.setObjectName("FeedDepthSettings")
        panel.setMinimumWidth(332)
        body = QVBoxLayout(panel)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(14)
        title = QLabel("检测设置")
        title.setProperty("monitorSettingsTitle", True)
        title.setMinimumHeight(36)
        heading = QHBoxLayout()
        heading.addWidget(title)
        heading.addStretch()
        self.debug_button = make_button("调试", self._show_debug)
        heading.addWidget(self.debug_button)
        body.addLayout(heading)

        source = self._section(body, "窝深数据")
        row = QHBoxLayout()
        self.choose_button = make_button("选择文件", self._choose_file)
        row.addWidget(self.choose_button)
        row.addWidget(make_button("数据格式", self._show_data_format))
        source.addLayout(row)

        theory = self._section(body, "理论窝深与报警")
        self.theory_button = make_button("理论窝深与报警设置", self._show_theory_settings)
        theory.addWidget(self.theory_button)

        statistics = self._section(body, "统计口径")
        statistics.addWidget(make_button("统计说明", self._show_statistics_settings))

        log_area = QFrame()
        log_layout = QVBoxLayout(log_area)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        label = QLabel("运行日志")
        label.setProperty("robotSectionTitle", True)
        log_layout.addWidget(label)
        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("FeedDepthProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setMinimumHeight(110)
        self.process_log.setMaximumBlockCount(500)
        self.process_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.process_log.setPlaceholderText("文件选择与设置变更将在这里显示")
        log_layout.addWidget(self.process_log, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAccessibleName("窝深任务进度")
        log_layout.addWidget(self.progress)
        self.task_progress_note = make_note("尚未开始处理")
        self.task_progress_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        log_layout.addWidget(self.task_progress_note)
        body.addWidget(log_area, 1)

        actions = QFrame()
        actions.setObjectName("FeedDepthActions")
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.import_button = make_button("导入数据", self._import_data, True)
        self.calculate_button = make_button("评估精度", self._evaluate, True)
        for button in (self.import_button, make_button("清空日志", self.process_log.clear, True),
                       self.calculate_button):
            row.addWidget(button, 1)
        body.addWidget(actions)
        return panel

    def _update_actions(self):
        busy = self.task is not None
        for control in (self.choose_button, self.debug_button, self.theory_button, self.import_button):
            control.setEnabled(not busy)
        self.import_button.setToolTip("正在处理任务，请等待完成后再导入。" if busy else
                                      "导入 CSV 或 Excel 窝深数据；未选择文件时会打开文件选择窗口。")
        data = self.simulation or self.measurements
        self.calculate_button.setEnabled(not busy and data is not None)
        self.calculate_button.setToolTip(
            "正在处理任务，请等待完成后再评估。" if busy else
            "请先点击“导入数据”载入窝深数据，再评估精度。" if data is None else
            "按当前理论窝深和报警设置，重新评估已载入的数据。"
        )

    def _import_data(self):
        if not self.selected_file or self.selected_file == (self.measurements or {}).get("source_path"):
            if not self._choose_file():
                return
        path, settings = self.selected_file, self._current_settings()
        previous = self.measurements

        def read(progress):
            data = load_feed_depth_data(path, settings, progress)
            if previous:
                batch_id = data["history"][0]["batch_id"]
                for key in ("rows", "history"):
                    data[key] = [row for row in previous[key] if row["batch_id"] != batch_id] + data[key]
            return data, evaluate_feed_depth(data, settings)

        self._append_log(f"开始导入窝深数据：{Path(path).name}")
        self._start_evaluation_task(read, self._import_ready)

    def _import_ready(self, result):
        data, evaluation = result
        self.measurements = data
        self.simulation = None
        self._show_evaluation(evaluation, len(evaluation["history"]) - 1)
        self._append_log(f"导入并评估完成：{Path(data['source_path']).name}；共 {len(evaluation['history'])} 个批次分组。")
        self._finish_evaluation_task()

    def _evaluate(self):
        data, settings = self.simulation or self.measurements, self._current_settings()
        self._start_evaluation_task(lambda _progress: evaluate_feed_depth(data, settings), self._evaluation_ready)

    def _start_evaluation_task(self, operation, completed):
        self.progress.setValue(0)
        self.task_progress_note.setText("正在读取并评估窝深数据…")
        self.task = ServiceTask(operation)
        self._update_actions()
        self.task.signals.progress.connect(self._simulation_progress)
        self.task.signals.completed.connect(completed)
        self.task.signals.failed.connect(self._evaluation_failed)
        QThreadPool.globalInstance().start(self.task)

    def _evaluation_ready(self, evaluation):
        self._show_evaluation(evaluation, min(self.selected_batch_index, len(evaluation["history"]) - 1))
        self._append_log("窝深精度评估完成。")
        self._finish_evaluation_task()

    def _finish_evaluation_task(self):
        self.progress.setValue(100)
        self.task = None
        self._update_actions()

    def _evaluation_failed(self, message):
        self._append_log(f"处理失败：{message}", "ERROR")
        self.task_progress_note.setText(f"处理失败：{message}；原已导入数据保留。")
        self.progress.setValue(0)
        self.task = None
        self._update_actions()

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择窝深数据", self.selected_file,
            "窝深数据 (*.csv *.xlsx);;CSV 文件 (*.csv);;Excel 文件 (*.xlsx)",
        )
        if not path:
            return False
        self.selected_file = path
        self.choose_button.setToolTip(path)
        self.task_progress_note.setText("已选择文件，待导入")
        self._append_log(f"已选择文件：{Path(path).name}；尚未读取数据。")
        return True

    def _show_debug(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("进给轴窝深调试")
        body = QVBoxLayout(dialog)
        body.addWidget(make_note("90 天 × 30 孔 · 理论窝深 1.500 mm"))
        body.addWidget(make_note("生成基准为 1.500 mm；显示和报警使用当前理论设置。仅用于调试。"))
        form = QFormLayout()
        output = QLineEdit(self.simulation["output_dir"] if self.simulation else str(self.simulation_root))
        output.setProperty("robotInput", True)
        output.setReadOnly(True)
        form.addRow("输出目录", output)
        body.addLayout(form)

        def toggle():
            dialog.accept()
            self._toggle_simulation()

        actions = QHBoxLayout()
        label = "退出调试模式" if self.simulation is not None else "生成90天模拟数据"
        actions.addWidget(make_button(label, toggle, True))
        actions.addWidget(make_button("关闭", dialog.reject))
        body.addLayout(actions)
        fit_dialog(dialog, 680)
        dialog.exec()

    def _toggle_simulation(self):
        if self.simulation is not None:
            self._clear_simulation()
            return
        output = self.simulation_root / QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
        self._append_log("开始生成90天窝深模拟数据：每天30孔，理论窝深1.500 mm。")
        self.progress.setValue(0)
        self.task_progress_note.setText("正在生成模拟孔位与历史统计…")
        settings = self._current_settings()

        def generate(progress):
            result = generate_feed_depth_simulation(output, progress)
            result["evaluation"] = evaluate_feed_depth(result, settings)
            return result

        self.task = ServiceTask(generate)
        self._update_actions()
        self.task.signals.progress.connect(self._simulation_progress)
        self.task.signals.completed.connect(self._simulation_ready)
        self.task.signals.failed.connect(self._simulation_failed)
        QThreadPool.globalInstance().start(self.task)

    def _simulation_progress(self, value, message):
        self.progress.setValue(value)
        self.task_progress_note.setText(message)

    def _simulation_ready(self, result):
        self.simulation = result
        history = result["evaluation"]["history"]
        self._show_evaluation(result["evaluation"], len(history) - 1)
        self.source_badge.setText(f"模拟数据 · {len(history)} 天")
        self._append_log(f"模拟数据生成并显示完成：{len(history)} 批、{len(result['rows'])} 孔；CSV：{result['output_dir']}")
        self.progress.setValue(100)
        self.task = None
        self._update_actions()

    def _simulation_failed(self, message):
        self._append_log(f"模拟生成失败：{message}")
        self.task_progress_note.setText(f"模拟生成失败：{message}")
        self.progress.setValue(0)
        self.task = None
        self._update_actions()

    def _show_batch(self, index):
        if self.evaluation is None:
            return
        self.selected_batch_index = index
        record = self.evaluation["history"][index]
        self.source_badge.setText(_source_label(record))
        self.history_note.setText(f"{record['measured_at'][:10] or '测量时间未提供'} · {record.get('row_id') or '整批'} · {record['count']} 孔")
        for key, text in (
            ("count", str(record["count"])), ("theory", f"{record['theoretical_depth_mm']:.3f}"),
            ("mean", _number(record["mean_depth_mm"])), ("variance", _number(record["variance_mm2"], ".6f")),
        ):
            self.result_values[key].setText(text)
        for chart in self.trend_charts.values():
            chart.selected_index = index
            chart.update()
        self._show_alarm(record)

    def _show_evaluation(self, evaluation, index):
        self.evaluation = evaluation
        for chart in self.trend_charts.values():
            chart.set_history(evaluation["history"])
        self._show_batch(index)

    def _set_mean_alarm(self, alarm):
        value = self.result_values["mean"]
        value.setProperty("overLimit", alarm)
        value.style().unpolish(value)
        value.style().polish(value)
        value.update()

    def _show_alarm(self, record):
        self._set_mean_alarm(record["alarm"])
        if record["count"] == 0:
            self.result_values["mean"].setToolTip("当前分组没有有效实测窝深，缺测不补零。")
            self.task_progress_note.setText("当前分组全部缺测，请检查数据或选择其他分组。")
            self._alarm_key = None
            return
        if self.error_lower_mm is None:
            self.result_values["mean"].setToolTip("尚未设置误差报警范围。")
            self.task_progress_note.setText(f"{_source_label(record)}已就绪")
            self._alarm_key = None
            return
        detail = (f"平均偏差 {record['bias_mm']:+.4f} mm；允许范围 "
                  f"[{self.error_lower_mm:+g}, {self.error_upper_mm:+g}] mm；"
                  f"{record['alarm_count']} 孔超限。")
        self.result_values["mean"].setToolTip(detail)
        self.task_progress_note.setText(
            f"{'平均窝深超限 · ' if record['alarm'] else ''}{record['alarm_count']} 孔超限"
            if record["alarm"] or record["alarm_count"] else "当前批次未超限"
        )
        alarm_key = (record["batch_id"], record["theoretical_depth_mm"],
                     self.error_lower_mm, self.error_upper_mm) if record["alarm"] or record["alarm_count"] else None
        if alarm_key is not None and alarm_key != self._alarm_key:
            self._append_log(f"{_source_label(record)} {record['measured_at'][:10]}：{detail}", "WARNING")
        self._alarm_key = alarm_key

    def _clear_simulation(self):
        self.simulation = None
        if self.measurements is not None:
            self._append_log("已退出调试模式，正在恢复已导入数据；模拟文件保留。")
            self._evaluate()
            return
        self.evaluation = None
        self.selected_batch_index = -1
        self._alarm_key = None
        for value in self.result_values.values():
            value.setText("—")
        self._set_mean_alarm(False)
        self.result_values["mean"].setToolTip("")
        for chart in self.trend_charts.values():
            chart.set_history([])
        self.source_badge.setText("暂无数据")
        self.history_note.setText("暂无数据")
        self.choose_button.setToolTip(self.selected_file)
        self.theory_button.setEnabled(True)
        self._update_settings_tooltip()
        self.progress.setValue(0)
        self.task_progress_note.setText("已退出调试模式")
        self._append_log("已退出调试模式，导出的模拟文件保留。")
        self._update_actions()

    def _show_data_format(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("窝深数据格式")
        body = QVBoxLayout(dialog)
        body.addWidget(make_note("约定格式：CSV 使用 UTF-8 编码（可带 BOM）；Excel 读取首个工作表。"
                                 "首行为以下字段名，每个文件一批，一行一个孔。"))
        fields = (
            ("孔号", "hole_id", "必填", "按文本读取，保留前导零"),
            ("实测窝深", "actual_depth_mm", "必填", "数值，单位 mm；缺测不补零"),
            ("理论窝深", "theoretical_depth_mm", "逐孔模式必填", "数值，单位 mm；统一模式使用设置值"),
            ("孔排", "row_id", "选填", "未提供时整批作为一组"),
            ("批次", "batch_id", "选填", "整列一致；未提供时使用文件名"),
            ("测量时间", "measured_at", "历史比较时提供", "当批统计可缺；不用导入时间补填"),
            ("工况组", "condition_id", "历史比较时提供", "同批一致，标识可比加工与测量条件"),
        )
        table = QTableWidget(len(fields), 4)
        table.setProperty("robotTable", True)
        table.setHorizontalHeaderLabels(["内容", "字段名", "要求", "说明"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().hide()
        for row, values in enumerate(fields):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        body.addWidget(table, 1)
        body.addWidget(make_note("同批次内“孔排 + 孔号”应唯一；不同理论窝深分别统计。"
                                 "后续流程：读取 → 校验与预览 → 分组统计 → 保存结果。"))
        body.addWidget(make_note("本轮仅约定输入格式，尚未读取所选文件。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        body.addWidget(buttons)
        fit_dialog(dialog, 980, 440)
        dialog.exec()

    def _show_theory_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("理论窝深与报警设置")
        body = QVBoxLayout(dialog)
        form = QFormLayout()
        mode = QComboBox()
        mode.setProperty("robotInput", True)
        mode.addItem("逐孔读取文件", "per_hole")
        mode.addItem("整批统一填写", "uniform")
        mode.setCurrentIndex(mode.findData(self.depth_mode))
        depth = QLineEdit("" if self.uniform_depth is None else str(self.uniform_depth))
        depth.setObjectName("feedDepthTheory")
        depth.setProperty("robotInput", True)
        depth.setPlaceholderText("请输入大于 0 的理论窝深")
        depth.setEnabled(mode.currentData() == "uniform")
        mode.currentIndexChanged.connect(lambda: depth.setEnabled(mode.currentData() == "uniform"))
        form.addRow("理论值来源", mode)
        form.addRow("统一窝深 / mm", depth)
        lower = QLineEdit("" if self.error_lower_mm is None else str(self.error_lower_mm))
        lower.setObjectName("feedDepthErrorLower")
        upper = QLineEdit("" if self.error_upper_mm is None else str(self.error_upper_mm))
        upper.setObjectName("feedDepthErrorUpper")
        for field in (lower, upper):
            field.setProperty("robotInput", True)
            field.setPlaceholderText("未设置")
        form.addRow("允许误差下限 / mm", lower)
        form.addRow("允许误差上限 / mm", upper)
        body.addLayout(form)
        body.addWidget(make_note("偏差 = 实测 − 理论；超出允许范围标红，等于边界不报警。上下限同时留空则关闭报警。"))
        error = make_note("")
        error.setProperty("validationError", True)
        body.addWidget(error)

        def apply():
            try:
                settings = validate_feed_depth_settings({
                    "depth_mode": mode.currentData(),
                    "uniform_depth": depth.text().strip() if mode.currentData() == "uniform" else self.uniform_depth,
                    "error_lower_mm": lower.text().strip() or None,
                    "error_upper_mm": upper.text().strip() or None,
                })
            except ValueError as exc:
                error.setText(str(exc))
                return
            dialog.accept()
            self._apply_settings(settings)

        self._settings_buttons(body, dialog, apply)
        fit_dialog(dialog, 590)
        dialog.exec()

    def _current_settings(self):
        return {"depth_mode": self.depth_mode, "uniform_depth": self.uniform_depth,
                "error_lower_mm": self.error_lower_mm, "error_upper_mm": self.error_upper_mm}

    def _update_settings_tooltip(self):
        theory = (f"整批统一理论窝深：{self.uniform_depth:g} mm" if self.depth_mode == "uniform"
                  else "逐孔读取文件中的理论窝深，单位 mm。")
        limits = ("未设置误差报警范围" if self.error_lower_mm is None else
                  f"允许误差：[{self.error_lower_mm:+g}, {self.error_upper_mm:+g}] mm（含边界）")
        self.theory_button.setToolTip(f"{theory}\n{limits}")
        if self.evaluation is None:
            self.result_values["theory"].setText(f"{self.uniform_depth:.3f}" if self.depth_mode == "uniform" else "—")

    def _apply_settings(self, settings):
        data = self.simulation or self.measurements

        def update(_progress):
            evaluation = evaluate_feed_depth(data, settings) if data is not None else None
            save_feed_depth_settings(self.settings_path, settings)
            return evaluation

        self.progress.setValue(0)
        self.task_progress_note.setText("正在应用理论窝深与报警设置…")
        self.pending_settings = settings
        self.task = ServiceTask(update)
        self._update_actions()
        self.task.signals.completed.connect(self._settings_ready)
        self.task.signals.failed.connect(self._settings_failed)
        QThreadPool.globalInstance().start(self.task)

    def _settings_ready(self, evaluation):
        settings = self.pending_settings
        self.pending_settings = None
        self.depth_mode = settings["depth_mode"]
        self.uniform_depth = settings["uniform_depth"]
        self.error_lower_mm = settings["error_lower_mm"]
        self.error_upper_mm = settings["error_upper_mm"]
        self._update_settings_tooltip()
        description = self.theory_button.toolTip().replace("\n", "；")
        self._append_log(f"已保存理论窝深与报警设置：{description}")
        if evaluation is not None:
            self._show_evaluation(evaluation, min(self.selected_batch_index, len(evaluation["history"]) - 1))
        else:
            self.task_progress_note.setText("设置已保存，等待数据")
        self.progress.setValue(100)
        self.task = None
        self._update_actions()

    def _settings_failed(self, message):
        self.pending_settings = None
        self._append_log(f"设置未应用：{message}", "ERROR")
        self.task_progress_note.setText(f"设置未应用：{message}")
        self.progress.setValue(0)
        self.task = None
        self._update_actions()

    def _show_statistics_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("窝深统计说明")
        body = QVBoxLayout(dialog)
        body.addWidget(make_note("按孔排和理论窝深分组，针对有效实测值计算平均值和总体方差（除以 N）。"
                                 "没有有效孔时留空；仅一个孔时方差为 0，不能据此判断重复性。"))
        body.addWidget(make_note("缺测不补零，不自动剔除偏大或偏小的有效值。支持 CSV、Excel 与调试模拟数据，来源分别标注。"))
        body.addWidget(make_note("历史分别看平均偏差（实测 − 理论）和标准差（方差开平方）。"
                                 "只比较同工况、同孔排、同理论窝深的批次；单孔批次不进入标准差趋势。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        body.addWidget(buttons)
        fit_dialog(dialog, 590)
        dialog.exec()

    @staticmethod
    def _settings_buttons(body, dialog, apply):
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用设置")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        body.addWidget(buttons)

    def _append_log(self, message, level="INFO"):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.process_log.appendPlainText(f"{timestamp} [{level}] {message}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_layout_direction()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange and hasattr(self, "columns"):
            self._update_layout_direction()

    def _update_layout_direction(self):
        stacked = self.width() < 1000 * getattr(self.window(), "ui_scale", 1.0)
        self.columns.setDirection(QHBoxLayout.Direction.TopToBottom if stacked else QHBoxLayout.Direction.LeftToRight)
        self.columns.setStretch(0, 0 if stacked else 13)
        self.columns.setStretch(1, 0 if stacked else 7)


class FeedDepthTrendChart(QWidget):
    """沿用现有页面的 Qt 曲线画法，只显示服务返回的批次统计。"""

    batch_selected = pyqtSignal(int)

    def __init__(self, metric, empty_message):
        super().__init__()
        self.metric = metric
        self.empty_message = empty_message
        self.history = []
        self.times = []
        self.selected_index = -1
        self.points = []
        self.plot_rect = QRectF()
        self.setMouseTracking(True)
        self.setMinimumHeight(95)

    def set_history(self, records):
        self.history = records
        self.times = [datetime.fromisoformat(row["measured_at"]).timestamp() if row.get("measured_at") else None
                      for row in records]
        self.points = []
        self.selected_index = -1
        QToolTip.hideText()
        self.update()

    def _value(self, row):
        if not row.get("measured_at") or not row.get("condition_id"):
            return None
        if self.selected_index >= 0:
            selected = self.history[self.selected_index]
            if any(row.get(key) != selected.get(key) for key in
                   ("row_id", "theoretical_depth_mm", "condition_id", "is_simulated")):
                return None
        return row[self.metric] if self.metric != "stddev_mm" or row["count"] >= 2 else None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font())
        colors = DISPLAY["colors"]
        font = painter.fontMetrics()
        height = font.height()
        left = font.horizontalAdvance("−0.000") + 8
        right = max(height, font.horizontalAdvance("00-00") / 2 + 4)
        rect = QRectF(left, height / 2, max(1, self.width() - left - right),
                      max(1, self.height() - height * 2.2))
        self.plot_rect = rect
        self.points = []
        for step in range(5):
            y = rect.top() + rect.height() * step / 4
            painter.setPen(QPen(QColor(colors["border"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(QColor(colors["placeholder"]), 1))
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        values = [self._value(row) for row in self.history if self._value(row) is not None]
        if not values:
            painter.setPen(QColor(colors["inactive"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.empty_message)
            return

        low, high = min(0, min(values)), max(0, max(values))
        padding = (high - low or 0.001) * 0.08
        low = low - padding if self.metric == "bias_mm" else 0
        high += padding
        times = [stamp for row, stamp in zip(self.history, self.times) if self._value(row) is not None]
        first, last = min(times), max(times)
        duration = last - first or 86400

        def point(timestamp, value):
            return QPointF(rect.left() + (timestamp - first) / duration * rect.width(),
                           rect.bottom() - (value - low) / (high - low) * rect.height())

        painter.setPen(QColor(colors["inactive"]))
        for step in range(5):
            value = high - (high - low) * step / 4
            y = rect.top() + rect.height() * step / 4
            painter.drawText(QRectF(0, y - height / 2, left - 6, height),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{value:.3f}")
        first_record = next(row for row in self.history if self._value(row) is not None)
        timezone = datetime.fromisoformat(first_record["measured_at"]).tzinfo
        for step in range(5):
            timestamp = first + (last - first) * step / 4
            x = point(timestamp, 0).x()
            label = datetime.fromtimestamp(timestamp, timezone).strftime("%m-%d")
            label_width = font.horizontalAdvance(label) + 8
            painter.drawText(QRectF(x - label_width / 2, rect.bottom() + 3, label_width, height),
                             Qt.AlignmentFlag.AlignCenter, label)

        if self.metric == "bias_mm":
            zero = point(first, 0).y()
            painter.setPen(QPen(QColor(colors["placeholder"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), zero), QPointF(rect.right(), zero))
            painter.drawText(QRectF(rect.right() - 6 * height, zero - height - 2, 6 * height, height),
                             Qt.AlignmentFlag.AlignRight, "零偏差")

        color = QColor(colors["action"] if self.metric == "bias_mm" else colors["axis_y"])
        path = QPainterPath()
        started = False
        for index in sorted(range(len(self.history)), key=lambda index: self.times[index] or 0):
            row, timestamp = self.history[index], self.times[index]
            value = self._value(row)
            if value is None:
                started = False
                continue
            target = point(timestamp, value)
            self.points.append((index, target))
            if started:
                path.lineTo(target)
            else:
                path.moveTo(target)
                started = True
        painter.setPen(QPen(color, 1.6))
        painter.drawPath(path)
        painter.setBrush(color)
        for index, target in self.points:
            radius = 3.5 if index == self.selected_index else 1.6
            painter.drawEllipse(target, radius, radius)

    def _nearest_index(self, position):
        if not self.plot_rect.contains(position) or not self.points:
            return None
        return min(self.points, key=lambda entry: abs(entry[1].x() - position.x()))[0]

    def mouseMoveEvent(self, event):
        index = self._nearest_index(event.position())
        if index is None:
            QToolTip.hideText()
            return
        row = self.history[index]
        name = "平均偏差" if self.metric == "bias_mm" else "标准差"
        QToolTip.showText(event.globalPosition().toPoint(),
                         f"{_source_label(row)} · {row['measured_at'][:10]}\n{row['batch_id']}\n"
                         f"{name}：{row[self.metric]:.4f} mm · {row['count']} 孔\n点击查看此批次", self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._nearest_index(event.position())
            if index is not None:
                self.batch_selected.emit(index)
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)
