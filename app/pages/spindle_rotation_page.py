"""主轴空启动建模、日常检测、人工标签和模型版本回看。"""

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor, isfinite
import numpy as np

from PyQt6.QtCore import QObject, QDateTime, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QStyle, QStyledItemDelegate, QStyleOptionViewItem,
    QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from app.resources import DISPLAY, fit_dialog
from core.services.spindle_monitoring_service import SpindleMonitoringService, assess_score


class TaskSignals(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class SpindleTask:
    def __init__(self, operation):
        self.operation = operation
        self.signals = TaskSignals()

    def run(self):
        try:
            self.signals.completed.emit(self.operation(self.signals.progress.emit))
        except Exception as error:
            self.signals.failed.emit(str(error))


LABELS = {"unconfirmed": "未判定", "healthy": "正常", "abnormal": "异常"}
CHANNELS = ["前轴承 Y · ACC1", "前轴承 X · ACC2", "平台 Y · ACC3", "温度 · NTC1 / RTD1", "主轴电流"]


def _local_time(value):
    if len(value) == 10:
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class CurrentModelDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if not index.data(Qt.ItemDataRole.UserRole + 1):
            super().paint(painter, option, index)
            return
        text_option = QStyleOptionViewItem(option)
        self.initStyleOption(text_option, index)
        text_option.text = option.fontMetrics.elidedText(text_option.text, Qt.TextElideMode.ElideRight,
                                                        option.rect.width() - 2 * option.fontMetrics.height())
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, text_option, painter, option.widget)
        painter.save()
        painter.setFont(option.font)
        painter.setPen(QColor(DISPLAY["colors"]["text"]))
        painter.drawText(option.rect.adjusted(0, 0, -4, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "◀")
        painter.restore()


class SpindleRotationPage(QScrollArea):
    label_saved = pyqtSignal()

    def __init__(self, service=None):
        super().__init__()
        self.service = service or SpindleMonitoringService()
        self.result = None
        self.uses_current_thresholds = False
        self.task = None
        self.task_completed = None
        self.task_status_key = "acquisition"
        self.review_run_ids = None
        self.energy_task = None
        self.order_energy_by_run = {}
        # 本机 CUDA 在 Qt QRunnable 任务切换后会卡住；复用常驻 Python 工作线程。
        self.worker_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spindle")
        self.destroyed.connect(lambda _=None, pool=self.worker_pool: pool.shutdown(wait=False))
        self.actions = []
        self.vibration_channel = 0
        self.setObjectName("SpindleRotationPage")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        canvas = QWidget()
        canvas.setObjectName("SpindleCanvas")
        self.setWidget(canvas)
        self._create_detail_controls()
        self.columns = QHBoxLayout(canvas)
        self.columns.setContentsMargins(0, 0, 8, 0)
        self.columns.setSpacing(14)
        self.columns.addWidget(self._results_view(), 13)
        self.columns.addWidget(self._settings_view(), 7)
        self._refresh()

    @staticmethod
    def _note(text):
        label = QLabel(text)
        label.setProperty("robotNote", True)
        label.setWordWrap(True)
        return label

    def _button(self, text, callback, primary=False, track=True):
        button = QPushButton(text)
        button.setProperty("robotAction", True)
        button.setProperty("primary", primary)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        if track:
            self.actions.append(button)
        return button

    @staticmethod
    def _combo():
        combo = QComboBox()
        combo.setProperty("robotInput", True)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    def _create_detail_controls(self):
        # 这些控件长期保留选中状态，仅在详情弹窗内显示。
        self.detail_holder = QWidget(self)
        self.detail_holder.hide()
        self.run_select = self._combo()
        self.run_select.setToolTip("按真实采集日期选择每次检测")
        self.model_select = self._combo()
        self.model_select.setItemDelegate(CurrentModelDelegate(self.model_select))
        self.model_select.setToolTip("按模型版本回看，旧评价保持不变")
        self.run_select.currentIndexChanged.connect(self._selection_changed)
        self.model_select.currentIndexChanged.connect(self._selection_changed)
        self.signal_select = self._combo()
        self.signal_select.addItems(CHANNELS)
        self.signal_select.currentIndexChanged.connect(self._update_plots)
        self.label_select = self._combo()
        for key, label in LABELS.items():
            self.label_select.addItem(label, key)
        self.label_select.currentIndexChanged.connect(self._label_changed)
        self.training_check = QCheckBox("纳入下次训练")
        self.label_note = QLineEdit()
        self.label_note.setProperty("robotInput", True)
        self.label_note.setPlaceholderText("判定备注（可选）")
        self.label_button = self._button("保存人工判定", self._save_label, True)
        self.result_detail = self._note("")
        self.initial_status = self._note("")
        self.conclusion = self._note("尚未评估")
        for widget in (self.run_select, self.model_select, self.signal_select,
                       self.label_select, self.training_check, self.label_note,
                       self.label_button, self.result_detail, self.initial_status, self.conclusion):
            widget.setParent(self.detail_holder)

    def _results_view(self):
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
        self.source_badge = QLabel("暂无数据")
        self.source_badge.setObjectName("SpindleDemoBadge")
        heading.addWidget(self.source_badge)
        body.addLayout(heading)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.result_values = {}
        for key, title, hint in (
            ("analysis", "解析评价", "正常均值 / 倍"),
            ("network", "网络评分", "P95 / 倍"),
            ("temperature", "温度", "NTC1 / °C"),
            ("current", "电流", "驱动器读数 / A"),
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
        self.result_values["analysis"].setToolTip("本次前轴承 X/Y 合成速度 RMS / 模型正常样本的平均 RMS；正常平均水平为 1 倍。")
        self.result_values["network"].setToolTip("相对健康校准参考的重建误差倍率 P95；反映信号差异，不能直接换算为 μm。")
        self.result_values["temperature"].setToolTip("轴承温度有效采样点的中位数；缺失点不参与统计。")
        self.result_values["current"].setToolTip("稳定采集段内驱动器有效读数的有符号中位数，不是交流电流有效值；不参与网络评分。")
        body.addLayout(metrics)

        self.signal_button = QToolButton()
        self.signal_button.setObjectName("SpindleSignalSelect")
        self.signal_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.signal_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.signal_menu = QMenu(self.signal_button)
        self.signal_menu.setProperty("spindleMenu", True)
        self.signal_actions = QActionGroup(self)
        for title, indices in (("振动", range(3)), ("温度", [3]), ("电流", [4])):
            menu = self.signal_menu.addMenu(title)
            menu.setProperty("spindleMenu", True)
            for index in indices:
                action = menu.addAction(CHANNELS[index])
                action.setData(index)
                action.setCheckable(True)
                self.signal_actions.addAction(action)
        self.signal_button.setMenu(self.signal_menu)
        self.signal_actions.triggered.connect(lambda action: self.signal_select.setCurrentIndex(action.data()))
        grid = QGridLayout()
        grid.setSpacing(14)
        self.plots, self.plot_titles = {}, {}
        for index, (key, title) in enumerate((
            ("waveform", "采集信号"), ("spectrum", "振动频谱"),
            ("energy", "倍频能量占比"), ("distribution", "网络辅助评估"),
        )):
            area = QWidget()
            layout = QVBoxLayout(area)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            row = QHBoxLayout()
            label = QLabel(title)
            label.setProperty("robotSectionTitle", True)
            label.setMinimumHeight(36)
            self.plot_titles[key] = label
            row.addWidget(label)
            row.addStretch()
            if key == "waveform":
                row.addWidget(self.signal_button)
            layout.addLayout(row)
            plot = SpindlePlot(fill=key == "distribution")
            self.plots[key] = plot
            layout.addWidget(plot, 1)
            grid.addWidget(area, index // 2, index % 2)
        for index in range(2):
            grid.setColumnStretch(index, 1)
            grid.setRowStretch(index, 1)
        body.addLayout(grid, 1)
        footer = QHBoxLayout()
        self.current_run_button = QToolButton()
        self.current_run_button.setAutoRaise(True)
        self.current_run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.current_run_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.current_run_button.clicked.connect(self._show_samples)
        footer.addWidget(self.current_run_button, 1)
        footer.addWidget(self._button("检测历史", self._show_history, True))
        footer.addWidget(self._button("预警记录", self._show_alerts, True))
        body.addLayout(footer)
        return panel

    def _section(self, parent, title, status_key):
        frame = QFrame()
        frame.setProperty("settingsGroup", True)
        body = QVBoxLayout(frame)
        body.setContentsMargins(0, 0, 0, 12)
        body.setSpacing(8)
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
        body.addLayout(heading)
        parent.addWidget(frame)
        return body

    def _settings_view(self):
        panel = QFrame()
        panel.setObjectName("SpindleSettings")
        panel.setMinimumWidth(332)
        body = QVBoxLayout(panel)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(14)
        title = QLabel("检测设置")
        title.setProperty("monitorSettingsTitle", True)
        title.setMinimumHeight(36)
        body.addWidget(title)
        self.status_lights = {}
        acquisition = self._section(body, "采集方案", "acquisition")
        acquisition.addWidget(self._button("采集详情", self._show_acquisition))
        training = self._section(body, "正常样本与网络", "model")
        row = QHBoxLayout()
        self.review_button = self._button("人工判定", self._show_review, True)
        self.train_button = self._button("训练网络", self._train, True)
        row.addWidget(self.review_button)
        row.addWidget(self.train_button)
        row.addWidget(self._button("样本与模型", self._show_samples))
        training.addLayout(row)
        thresholds = self._section(body, "判定阈值", "thresholds")
        thresholds.addWidget(self._button("阈值设置", self._show_thresholds))
        log_area = QFrame()
        log_area.setObjectName("SpindleLogArea")
        log_layout = QVBoxLayout(log_area)
        log_layout.setContentsMargins(0, 0, 0, 0)
        heading = QHBoxLayout()
        label = QLabel("运行日志")
        label.setProperty("robotSectionTitle", True)
        heading.addWidget(label)
        heading.addStretch()
        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("SpindleProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setPlaceholderText("操作过程、提示与错误将在这里显示")
        self.process_log.setMinimumHeight(110)
        self.process_log.setMaximumBlockCount(500)
        self.process_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        log_layout.addLayout(heading)
        log_layout.addWidget(self.process_log, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAccessibleName("当前任务进度")
        log_layout.addWidget(self.progress)
        self.task_progress_note = self._note("尚未开始处理")
        log_layout.addWidget(self.task_progress_note)
        body.addWidget(log_area, 1)
        actions = QFrame()
        actions.setObjectName("SpindleActions")
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.initial_button = self._button("批量导入", lambda: self._choose_packages("initial"), True)
        self.initial_button.setToolTip("选择一个 ZIP，递归导入其中各级文件夹和 ZIP 内的初始正常样本。")
        self.daily_button = self._button("日常导入", lambda: self._choose_packages("daily"), True)
        self.evaluate_button = self._button("开始评估", self._evaluate, True)
        for button in (self.initial_button, self.daily_button,
                       self._button("清空日志", self.process_log.clear, True), self.evaluate_button):
            row.addWidget(button, 1)
        body.addWidget(actions)
        return panel

    def _show_acquisition(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("采集方案详情")
        body = QVBoxLayout(dialog)
        form = QFormLayout()
        config = self.service.config
        form.addRow("指定工况", QLabel(f"{config['target_speed_rpm']:g} r/min · 空转"))
        form.addRow("数据来源", QLabel("每日 ZIP，可含多次独立采集"))
        form.addRow("速度采样率", QLabel(f"{config['preprocessing']['sample_rate_hz']:g} Hz"))
        form.addRow("速度单位", QLabel(config["preprocessing"]["unit"]))
        body.addLayout(form)
        body.addWidget(self._note("主轴动作由原设备软件控制；此页导入已采集的数据包。工况与处理参数随数据和评价保留。"))
        body.addWidget(self._button("关闭", dialog.reject, track=False), 0, Qt.AlignmentFlag.AlignRight)
        fit_dialog(dialog, 570)
        dialog.exec()

    def _show_samples(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("正常样本与网络模型")
        body = QVBoxLayout(dialog)
        form = QFormLayout()
        form.addRow("采集批次", self.run_select)
        form.addRow("评价模型", self.model_select)
        body.addLayout(form)
        borrowed = (self.run_select, self.model_select)
        for widget in borrowed:
            widget.show()
        table = QTableWidget(0, 5)
        table.setProperty("robotTable", True)
        table.setHorizontalHeaderLabels(["采集时间", "采集名称", "来源", "人工判定", "训练候选"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()

        def populate():
            table.setRowCount(0)
            for run in self.service.list_runs():
                row = table.rowCount()
                table.insertRow(row)
                values = [_local_time(run["captured_at"]), run["run_name"],
                          "初始建模" if run["purpose"] == "initial" else "日常检测",
                          LABELS[run["manual_label"]], "是" if run["training_eligible"] else "否"]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    table.setItem(row, column, item)
                table.item(row, 0).setData(Qt.ItemDataRole.UserRole, run["run_id"])

        def select(row, column):
            self.run_select.setCurrentIndex(self.run_select.findData(table.item(row, 0).data(Qt.ItemDataRole.UserRole)))

        def sync_row():
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item.data(Qt.ItemDataRole.UserRole) == self.run_select.currentData():
                    table.setCurrentCell(row, 0)
                    table.scrollToItem(item)
                    break

        def review():
            if self.run_select.currentData():
                self._show_review()
                populate()
                sync_row()

        table.cellClicked.connect(select)
        table.currentCellChanged.connect(lambda row, column, *_: select(row, column) if row >= 0 else None)
        self.run_select.currentIndexChanged.connect(sync_row)
        populate()
        sync_row()
        body.addWidget(table, 1)
        row = QHBoxLayout()
        row.addWidget(self._button("人工判定", review, track=False))
        retry_requested = False
        def retry():
            nonlocal retry_requested
            retry_requested = True
            dialog.accept()
        retry_button = self._button("补算缺失评价", retry, track=False)
        model = self.service.current_model
        evaluated = {entry["run_id"] for entry in self.service.state["results"]
                     if model and entry["model_version"] == model["version"]}
        retry_button.setEnabled(bool(model and (model.get("reanalysis_status") != "complete"
                                                or set(self.service.runs) - evaluated)))
        retry_button.setToolTip("沿用当前模型补齐缺失评价，无需重新训练。")
        row.addWidget(retry_button)
        row.addStretch()
        row.addWidget(self._button("关闭", dialog.reject, track=False))
        body.addLayout(row)
        fit_dialog(dialog, 980, min(580, 190 + 32 * table.rowCount()))
        try:
            dialog.exec()
        finally:
            self.run_select.currentIndexChanged.disconnect(sync_row)
            for widget in borrowed:
                widget.setParent(self.detail_holder)
                widget.hide()
            dialog.deleteLater()
        if retry_requested:
            self.model_select.setCurrentIndex(0)
            self._run_task("补算缺失评价", self.service.reanalyze_history, status_key="model")

    def _show_review(self):
        displayed_result = self.result
        uses_current_thresholds = self.uses_current_thresholds
        dialog = QDialog(self)
        dialog.setWindowTitle("本次人工判定")
        body = QVBoxLayout(dialog)
        form = QFormLayout()
        package_select = self._combo()
        package_select.setObjectName("spindle_review_package")
        package_select.setPlaceholderText("暂无可判定的样本")
        package_select.setToolTip("选择尚未参与建模的 ZIP / 采集。")
        form.addRow("", package_select)
        original_button_text = self.label_button.text()
        trained = self.service.trained_run_ids()
        seen = set()
        for run in sorted(self.service.list_runs(),
                          key=lambda item: (item["imported_at"], item["captured_at"], item["run_id"]), reverse=True):
            if run["run_id"] in seen:
                continue
            group = self.service.import_group(run["run_id"])
            seen.update(group)
            if trained.intersection(group):
                continue
            source = run["source_filename"].split("!/")[0]
            name = source
            if run["purpose"] != "initial" and source.rsplit(".", 1)[0] != run["run_name"]:
                name += f" · {run['run_name']}"
            package_select.addItem(name, group)
            index = package_select.count() - 1
            scope = f"整批 {len(group)} 条样本" if run["purpose"] == "initial" else "本次采集（1 条）"
            package_select.setItemData(index, f"{source}\n{scope}\n导入：{_local_time(run['imported_at'])}", Qt.ItemDataRole.ToolTipRole)

        def select_package():
            self.review_run_ids = package_select.currentData() or []
            if not self.review_run_ids:
                self.label_select.setCurrentIndex(self.label_select.findData("unconfirmed"))
                self.training_check.setChecked(False)
                self.label_note.clear()
                self._label_changed()
                return
            run = self.service.runs[self.review_run_ids[0]]
            is_batch = run["purpose"] == "initial"
            dialog.setWindowTitle("整批人工判定" if is_batch else "本次人工判定")
            count = len(self.review_run_ids)
            self.label_select.blockSignals(True)
            self.label_select.setCurrentIndex(self.label_select.findData(run["manual_label"]))
            self.label_select.blockSignals(False)
            self.training_check.setChecked(run["training_eligible"])
            self.label_note.setText(run["label_note"])
            self.label_button.setText(f"保存整批判定（{count} 条）" if is_batch else original_button_text)
            self._label_changed()

        package_select.currentIndexChanged.connect(select_package)
        select_package()
        form.addRow("判定", self.label_select)
        form.addRow("训练用途", self.training_check)
        form.addRow("备注", self.label_note)
        body.addLayout(form)
        body.addWidget(self._note("只有判定正常且勾选入训的数据才能训练。批量导入的数据统一按整批判定；未判定和异常不入训。"))
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.label_button)
        row.addWidget(self._button("关闭", dialog.reject, track=False))
        body.addLayout(row)
        borrowed = (self.label_select, self.training_check, self.label_note, self.label_button)
        for widget in borrowed:
            widget.show()
        fit_dialog(dialog, 720)
        self.label_saved.connect(dialog.accept)
        try:
            dialog.exec()
        finally:
            self.review_run_ids = None
            self.label_button.setText(original_button_text)
            for widget in borrowed:
                widget.setParent(self.detail_holder)
                widget.hide()
            self.label_saved.disconnect(dialog.accept)
            self._selection_changed()
            self._show_result(displayed_result, current_thresholds=uses_current_thresholds)
            dialog.deleteLater()

    def _log(self, message, level="INFO"):
        self.process_log.appendPlainText(f"{QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')} [{level}] {message}")

    def _model_name(self, version):
        model = next((item for item in self.service.models if item["version"] == version), None)
        if model is None:
            return version or "未建模"
        architecture = model.get("architecture", "tcn_1s")
        architecture = {"tcn_1s": "TCN-1s"}.get(architecture, architecture)
        epochs = model.get("trained_epochs", model["training_config"]["epochs"])
        return f"{architecture} · {epochs}轮 · {_local_time(model['created_at'])}"

    def _refresh(self, selected=None):
        run_id = selected or self.run_select.currentData()
        model_version = self.model_select.currentData()
        self.run_select.blockSignals(True)
        self.run_select.clear()
        for run in self.service.list_runs():
            label = run["run_name"].split("rpm_")[-1]
            text = f"{_local_time(run['captured_at'])} · {label}"
            self.run_select.addItem(text, run["run_id"])
            self.run_select.setItemData(self.run_select.count() - 1, run["run_name"], Qt.ItemDataRole.ToolTipRole)
        index = self.run_select.findData(run_id)
        self.run_select.setCurrentIndex(index if index >= 0 else self.run_select.count() - 1)
        self.run_select.blockSignals(False)
        self.model_select.blockSignals(True)
        self.model_select.clear()
        model = self.service.current_model
        current_name = self._model_name(model["version"]) if model else "未建模"
        self.model_select.addItem(f"最新评价 → {current_name}", None)
        self.model_select.addItem("未建模", "unmodeled")
        for item in reversed(self.service.models):
            name = self._model_name(item["version"])
            self.model_select.addItem(name, item["version"])
            self.model_select.setItemData(self.model_select.count() - 1,
                                         bool(model and item["version"] == model["version"]), Qt.ItemDataRole.UserRole + 1)
            self.model_select.setItemData(self.model_select.count() - 1, item["version"], Qt.ItemDataRole.ToolTipRole)
        self.model_select.setCurrentIndex(max(0, self.model_select.findData(model_version)))
        self.model_select.blockSignals(False)
        self.initial_status.setText("批量数据整批判定为正常后才能训练" if not model else
                                   f"当前模型 → {current_name}\n新增数据请日常导入；仅正常数据可入训")
        self.train_button.setText("重新训练" if model else "训练网络")
        self._selection_changed()
        self._set_busy(self.task is not None)

    def _set_status_light(self, key, state, details):
        light = self.status_lights[key]
        if light.property("state") != state:
            light.setProperty("state", state)
            light.style().unpolish(light)
            light.style().polish(light)
            light.update()
        status = {"missing": "未就绪", "partial": "待处理", "ready": "已就绪"}[state]
        light.setToolTip(f"{status}\n{details}")
        light.setAccessibleDescription(f"{status}；{details}")

    def _refresh_status_lights(self):
        count = len(self.service.runs)
        self._set_status_light("acquisition", "ready" if count else "missing",
            f"指定工况：{self.service.config['target_speed_rpm']:g} r/min · 空转\n"
            f"数据入口：每日 ZIP\n已导入：{count} 次独立采集")
        model = self.service.current_model
        candidates = len(self.service.training_candidates())
        state = "partial" if count else "missing"
        details = f"正常候选：{candidates} 次采集\n当前模型：{self._model_name(model['version']) if model else '尚未训练'}"
        if model:
            details += f"\n内部编号：{model['version']}"
            state = "ready" if model.get("reanalysis_status") == "complete" else "partial"
            if state == "partial":
                details += "\n历史评价未完成，请在样本与模型中补算"
        elif count:
            details += "\n至少 3 次正常候选采集后可训练网络"
        self._set_status_light("model", state, details)
        thresholds = self.service.settings["thresholds"]
        configured = sum(value is not None for value in thresholds.values())
        state = "partial" if configured else "missing"
        if configured == 2 and model and not self.service.thresholds_review_required:
            state = "ready"
        details = "\n".join(f"{label}：{'未设置' if thresholds[key] is None else f'{thresholds[key]:g} 倍'}"
                            for key, label in (("warning", "预警"), ("fault", "故障")))
        if self.service.thresholds_review_required:
            details += "\n新模型阈值待复核"
        elif configured and model is None:
            details += "\n建模后需复核阈值"
        self._set_status_light("thresholds", state, details)

    def _selection_changed(self):
        run_id = self.run_select.currentData()
        run = self.service.runs.get(run_id)
        self.label_select.blockSignals(True)
        self.label_select.setCurrentIndex(self.label_select.findData(run["manual_label"] if run else "unconfirmed"))
        self.label_select.blockSignals(False)
        self.training_check.setChecked(bool(run and run["training_eligible"]))
        self.label_note.setText(run["label_note"] if run else "")
        self._label_changed()
        version = self.model_select.currentData()
        if run and version == "unmodeled":
            result = next((item for item in reversed(self.service.history(run_id))
                           if item["model_version"] is None), None)
        else:
            result = self.service.latest_result(run_id, version) if run else None
        self._show_result(result, current_thresholds=version is None)

    def _label_changed(self):
        run_ids = self.review_run_ids if self.review_run_ids is not None else [self.run_select.currentData()]
        run = self.service.runs.get(run_ids[0]) if run_ids else None
        editable = bool(run and self.task is None and not self.service.trained_run_ids().intersection(run_ids))
        for widget in (self.label_select, self.label_note, self.label_button):
            widget.setEnabled(editable)
        eligible = self.label_select.currentData() == "healthy"
        self.training_check.setEnabled(eligible and editable)
        if not eligible:
            self.training_check.setChecked(False)
        elif run and run["purpose"] == "initial" and run["manual_label"] != "healthy":
            self.training_check.setChecked(True)

    def _show_result(self, result, current_thresholds=False):
        self.result = result
        self.uses_current_thresholds = current_thresholds
        analysis_score, analysis_reference = self.service.analysis_metrics(result)
        values = {"analysis": analysis_score,
                  "network": result.get("score") if result else None,
                  "temperature": result.get("temperature_c", [None])[0] if result else None,
                  "current": result.get("current_a") if result else None}
        for key, value in values.items():
            self.result_values[key].setText("—" if value is None else f"{value:.4g}")
            self.result_values[key].setStyleSheet("")
        analysis_note = "本次径向速度 RMS / 模型正常采集的平均 RMS，正常平均水平为 1 倍。"
        if result and result.get("xy_rms_mm_s") is not None:
            analysis_note += f"\n本次径向 RMS：{result['xy_rms_mm_s']:.5g} mm/s。"
        analysis_note += (f"\n正常参考均值：{analysis_reference:.5g} mm/s。"
                          if analysis_reference is not None else "\n尚无正常参考，请先确认正常样本并建立模型。")
        self.result_values["analysis"].setToolTip(analysis_note)
        if result:
            self.source_badge.setText("历史回放" if result["source_type"] == "historical_replay" else "检测数据")
            assessment = result["assessment"]
            thresholds = result["thresholds"]
            if current_thresholds:
                thresholds = self.service.settings["thresholds"]
                assessment = assess_score(result["score"], thresholds)
                if result["score"] is not None and self.service.thresholds_review_required:
                    assessment = {"status": "review_required", "message": "新模型阈值待复核"}
            self.conclusion.setText(assessment["message"])
            color = "#ba3030" if assessment["status"] == "fault" else "#956300" if assessment["status"] == "warning" else "#263c52"
            self.conclusion.setStyleSheet(f"color:{color};")
            if assessment["status"] in ("warning", "fault"):
                for key in ("analysis", "network"):
                    self.result_values[key].setStyleSheet(f"color:{DISPLAY['colors']['error']};")
            role = {"training": "训练样本回评", "calibration": "校准样本回评", "independent": "未参与此模型建模"}[result["role"]]
            run = self.service.runs[result["run_id"]]
            condition = run.get("condition", {})
            remounted = "刀具重装" if condition.get("tool_remounted") else "未标记重装"
            pressure = condition.get("seal_pressure_mpa")
            pressure_text = "密封气压未记录" if pressure is None else f"密封气压 {pressure:g} MPa"
            operation = "空转" if run["operation"] == "idle" else run["operation"]
            warning = "未设置" if thresholds["warning"] is None else f"{thresholds['warning']:g} 倍"
            fault = "未设置" if thresholds["fault"] is None else f"{thresholds['fault']:g} 倍"
            threshold_title = "当前阈值" if current_thresholds else "当时阈值"
            self.result_detail.setText(
                f"{run['speed_rpm']:g} rpm · {operation} · {remounted} · {pressure_text}\n"
                f"采集：{_local_time(result['captured_at'])} · {role}\n"
                f"评估：{_local_time(result['evaluated_at'])} · 模型：{self._model_name(result['model_version'])}\n"
                f"{result['window_count']} 个 1 秒窗口 · {threshold_title}：预警 {warning} / 故障 {fault}"
            )
        else:
            self.conclusion.setStyleSheet("")
            self.source_badge.setText("暂无数据" if not self.service.runs else "暂无评价")
            self.conclusion.setText("尚未评估" if not self.service.current_model else "所选数据暂无对应评价")
            self.result_detail.clear()
        run = self.service.runs.get(self.run_select.currentData())
        if run:
            short_name = run["run_name"].split("rpm_", 1)[-1]
            self.current_run_button.setText(f"{_local_time(run['captured_at'])} · {short_name}")
            self.current_run_button.setToolTip(self.result_detail.text() or self.run_select.currentText())
        else:
            self.current_run_button.setText("选择采集记录")
            self.current_run_button.setToolTip("在样本与模型中选择采集批次与模型版本")
        self._update_plots()

    def _update_plots(self):
        channel = self.signal_select.currentIndex()
        short_names = ["振动 · 前 Y", "振动 · 前 X", "振动 · 台 Y", "温度 · NTC1 / RTD1", "电流"]
        self.signal_button.setText(short_names[channel])
        self.signal_button.setToolTip(CHANNELS[channel])
        self.signal_actions.actions()[channel].setChecked(True)
        if channel < 3:
            self.vibration_channel = channel
        axis = ("前 Y", "前 X", "台 Y")[self.vibration_channel]
        self.plot_titles["spectrum"].setText(f"振动频谱 · {axis}")
        self.plot_titles["energy"].setText(f"倍频能量占比 · {axis}")
        if not self.result:
            for plot in self.plots.values():
                plot.set_data([], "", "")
            return
        r = self.result
        channel = self.signal_select.currentIndex()
        if channel < 3:
            self.vibration_channel = channel
            series = [(r["waveform"]["time_s"], r["waveform"]["values"][channel], "本次")]
            if r["reconstruction"]:
                series.append((r["reconstruction"]["time_s"], r["reconstruction"]["reconstructed"][channel], "重建"))
            self.plots["waveform"].set_data(series, "时间 / s", "mm/s")
        elif channel == 3:
            series = [(r["temperature"]["time_s"], values, name)
                      for values, name in zip(r["temperature"]["values"], r["temperature"]["channels"])]
            self.plots["waveform"].set_data(series, "采集时间 / s", "°C")
        else:
            self.plots["waveform"].set_data([(r["telemetry"]["time_s"], r["telemetry"]["current_a"], "本次")], "采集时间 / s", "A")
        index = self.vibration_channel
        self.plots["spectrum"].set_data([(r["spectrum"]["frequency_hz"], r["spectrum"]["amplitude_mm_s"][index], "本次")], "频率 / Hz", CHANNELS[index] + " / mm/s")
        energy = r["band_energy"]
        if energy.get("basis") != "shaft_order":
            energy = self.order_energy_by_run.get(r["run_id"])
            if r["run_id"] not in self.order_energy_by_run and self.energy_task is None:
                self._load_order_energy(r["run_id"])
        if energy:
            labels = energy["labels"]
            values = energy["values"][index]
            total = sum(values)
            percentages = [value / total * 100 for value in values] if total else [None] * len(values)
            self.plots["energy"].set_data([(list(range(len(labels))), percentages, "本次")],
                                          "倍频 / ×", "能量占比 / %", bars=labels)
            source = "有效转速中位数" if energy["speed_source"] == "actual" else "目标转速（遥测缺失）"
            self.plots["energy"].setToolTip(
                f"1× = {energy['rotation_hz']:.3g} Hz，来自{source}。\n"
                "各倍频带宽 ±0.1×（1×：0.9–1.1×，2×：1.9–2.1×），截取于 10–900 Hz。\n"
                "各柱为该通道分量能量 / 10–900 Hz 总能量 × 100%，合计 100%。\n"
                "异步分量为所有倍频带之外的能量。" +
                ("\n该通道总能量为零，无法计算占比。" if not total else ""))
        else:
            self.plots["energy"].set_data([], "", "正在分析倍频能量" if self.energy_task else "倍频能量不可用")
            self.plots["energy"].setToolTip("")
        distributions = [r["window_scores"], r["reference_scores"]]
        if distributions[0] and distributions[1]:
            edges = np.histogram_bin_edges(distributions[0] + distributions[1], bins=35)
            centers = ((edges[:-1] + edges[1:]) / 2).tolist()
            series = [(centers, (np.histogram(values, edges)[0] / len(values)).tolist(), name)
                      for values, name in zip(distributions, ("本次", "健康参考"))]
            self.plots["distribution"].set_data(series, "重建误差倍率", "窗口比例")
        else:
            self.plots["distribution"].set_data([], "", "尚未建立模型")

    def _load_order_energy(self, run_id):
        task = SpindleTask(lambda progress: (run_id, self.service.order_band_energy(run_id)))
        self.energy_task = task
        self.energy_run_id = run_id
        task.signals.completed.connect(self._order_energy_done)
        task.signals.failed.connect(self._order_energy_failed)
        self.worker_pool.submit(task.run)

    def _order_energy_done(self, value):
        run_id, energy = value
        self.order_energy_by_run[run_id] = energy
        self.energy_task = None
        self._update_plots()

    def _order_energy_failed(self, message):
        self.order_energy_by_run[self.energy_run_id] = None
        self.energy_task = None
        self._log(f"倍频能量分析失败：{message}", "ERROR")
        self._update_plots()

    def _set_busy(self, busy):
        for button in self.actions:
            button.setEnabled(not busy)
        has_run = self.run_select.currentData() is not None
        self.initial_button.setEnabled(not busy and self.service.current_model is None)
        self.train_button.setEnabled(not busy and len(self.service.training_candidates()) >= 3)
        self.evaluate_button.setEnabled(not busy and has_run)
        self.label_button.setEnabled(not busy and has_run)
        self.review_button.setEnabled(not busy and has_run)
        self.current_run_button.setEnabled(not busy)
        self.signal_button.setEnabled(not busy and has_run)
        for widget in (self.run_select, self.model_select, self.label_select, self.label_note):
            widget.setEnabled(not busy and has_run)
        self._label_changed()
        self._refresh_status_lights()
        if busy:
            self._set_status_light(self.task_status_key, "partial", self.task_progress_note.text())

    def _run_task(self, title, operation, completed=None, *, status_key="acquisition"):
        if self.task is not None:
            return
        self._log(title)
        task = SpindleTask(operation)
        self.task = task
        self.task_completed = completed
        self.task_status_key = status_key
        self.progress.setValue(0)
        self.task_progress_note.setText(f"{title}：准备处理")
        self._set_busy(True)
        task.signals.progress.connect(self._task_progress)
        task.signals.completed.connect(self._task_done)
        task.signals.failed.connect(self._task_failed)
        self.worker_pool.submit(task.run)

    def _task_progress(self, value, message):
        self.progress.setValue(min(value, 99))
        self.progress.setToolTip(message)
        self.task_progress_note.setText(message)
        self._set_status_light(self.task_status_key, "partial", message)
        self._log(message)

    def _task_done(self, value):
        completed = self.task_completed
        self.task_completed = None
        self.task = None
        self._refresh()
        self._log("操作完成")
        if completed:
            completed(value)
        self.progress.setValue(100)
        self.task_progress_note.setText(value.get("summary", "处理完成") if isinstance(value, dict) else "处理完成")

    def _task_failed(self, message):
        self.task_completed = None
        self.task = None
        self._refresh()
        self._log(message, "ERROR")
        self.progress.setValue(0)
        self.task_progress_note.setText(f"处理失败：{message}")
        QMessageBox.warning(self, "操作未完成", message)

    def _choose_packages(self, purpose):
        title = "批量导入建模数据（递归检索 ZIP）" if purpose == "initial" else "导入一天检测数据"
        path, _ = QFileDialog.getOpenFileName(self, title, "", "检测数据包 (*.zip)")
        if path:
            label = "unconfirmed"
            if purpose == "initial":
                text, accepted = QInputDialog.getItem(
                    self, "整批数据判定", "判定所选 ZIP 中整批数据；只有正常数据可入训。\n重复采集保留原有判定。",
                    ["正常", "异常", "未判定"], 0, False)
                if not accepted:
                    return
                label = next(key for key, value in LABELS.items() if value == text)
            self.import_packages([path], purpose, batch_label=label)

    def import_packages(self, paths, purpose="daily", *, batch_label="unconfirmed"):
        def completed(value):
            runs = value["runs"] if purpose == "initial" else value
            if runs:
                self.model_select.setCurrentIndex(0)
                self._refresh(runs[-1]["run_id"])
        self._run_task("导入建模数据" if purpose == "initial" else "导入并分析日常检测",
                       lambda progress: (self.service.import_batch(paths, progress, label=batch_label) if purpose == "initial"
                                         else self.service.import_packages(paths, purpose, progress)), completed)

    def _evaluate(self):
        run_id = self.run_select.currentData()
        if run_id:
            self.model_select.setCurrentIndex(0)
            self._run_task("重新评估", lambda progress: self.service.evaluate(run_id, progress))

    def _save_label(self):
        run_id = self.run_select.currentData()
        if run_id:
            run_ids = self.review_run_ids if self.review_run_ids is not None else [run_id]
            if not run_ids:
                return
            try:
                self.service.set_labels(run_ids, self.label_select.currentData(), self.label_note.text(), self.training_check.isChecked())
            except (OSError, ValueError) as error:
                self._log(str(error), "ERROR")
                QMessageBox.warning(self.label_button.window(), "人工判定未保存", str(error))
                return
            self._log(f"已保存 {len(run_ids)} 条样本的人工判定：" + self.label_select.currentText())
            self._refresh()
            self.label_saved.emit()

    def _train(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("从头训练网络")
        form = QFormLayout(dialog)
        epochs = QSpinBox()
        epochs.setRange(1, 1000)
        epochs.setValue(self.service.config["training"]["epochs"])
        form.addRow("训练轮数", epochs)
        form.addRow(self._note(f"使用 {len(self.service.training_candidates())} 次可建模采集；训练与校准按采集隔离。\n每次随机初始化，完成后自动重算历史。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        fit_dialog(dialog, 500)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 后台任务只接收普通数值，不持有或访问弹窗内的 Qt 控件。
            training_config = {"epochs": epochs.value()}
            self.model_select.setCurrentIndex(0)
            self._run_task("从头训练并重算历史", lambda progress: self.service.train(progress, training_config),
                           status_key="model")
        dialog.deleteLater()

    def _show_thresholds(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("人工判定阈值")
        form = QFormLayout(dialog)
        fields = {}
        for key, title in (("warning", "预警阈值 / 倍"), ("fault", "故障检修阈值 / 倍")):
            value = self.service.settings["thresholds"][key]
            field = QLineEdit("" if value is None else str(value))
            field.setObjectName("spindle_" + key + "_threshold")
            field.setProperty("robotInput", True)
            field.setPlaceholderText("留空表示未设置")
            form.addRow(title, field)
            fields[key] = field
        form.addRow(self._note("评分为各1秒窗口误差倍率的P95。倍率参照模型的健康校准数据；故障阈值须高于预警阈值。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        def save():
            try:
                values = {key: float(field.text()) if field.text().strip() else None for key, field in fields.items()}
                self.service.set_thresholds(**values)
            except (ValueError, OSError) as error:
                QMessageBox.warning(dialog, "阈值未保存", str(error))
                return
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        fit_dialog(dialog, 520)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._refresh()
            self._evaluate()

    def _show_alerts(self):
        self._show_history(alerts_only=True)

    def _show_history(self, alerts_only=False):
        dialog = QDialog(self)
        dialog.setWindowTitle("主轴预警记录" if alerts_only else "检测历史 · 保留各模型版本与当时阈值")
        layout = QVBoxLayout(dialog)
        table = QTableWidget()
        table.setProperty("robotTable", True)
        columns = ["采集时间", "采集名称", "模型", "评分 / 倍", "判定", "数据用途"]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        entries = [entry for entry in reversed(self.service.state["results"])
                   if not alerts_only or entry["assessment"]["status"] in ("warning", "fault")]
        table.setRowCount(len(entries))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, result in enumerate(entries):
            values = [_local_time(result["captured_at"]), result["run_name"], self._model_name(result["model_version"]),
                      "—" if result["score"] is None else f"{result['score']:.4g}", result["assessment"]["message"],
                      {"training": "训练回评", "calibration": "校准回评", "independent": "独立于此模型"}[result["role"]]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, column, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        def show(row, column):
            entry = entries[row]
            from core.services.position_persistence import read_json
            self.run_select.setCurrentIndex(self.run_select.findData(entry["run_id"]))
            self.model_select.setCurrentIndex(max(0, self.model_select.findData(entry["model_version"] or "unmodeled")))
            self._show_result(read_json(self.service.root / entry["result_file"]))
            dialog.accept()
        table.cellActivated.connect(show)
        layout.addWidget(table)
        layout.addWidget(self._note("双击或回车查看当时结果。人工标签保持独立；重算不覆盖旧评价。"))
        layout.addWidget(self._button("关闭", dialog.reject, track=False), 0, Qt.AlignmentFlag.AlignRight)
        fit_dialog(dialog, 1100, min(560, 150 + 32 * table.rowCount()))
        dialog.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.columns.setDirection(QHBoxLayout.Direction.TopToBottom if self.width() < 1000 else QHBoxLayout.Direction.LeftToRight)


class SpindlePlot(QWidget):
    """沿用原生 Qt 图表风格，只绘制真实数组，缺失值断线。"""
    def __init__(self, fill=False):
        super().__init__()
        self.series = []
        self.xlabel = self.ylabel = ""
        self.bars = None
        self.fill = fill
        self.setMinimumHeight(160)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, series, xlabel, ylabel, bars=None):
        self.series, self.xlabel, self.ylabel, self.bars = series, xlabel, ylabel, bars
        self.update()

    def _y_limits(self, ymin, ymax):
        if self.ylabel == "°C":
            span = max(5, (ymax - ymin) * 1.16)
            center = (ymin + ymax) / 2
            return floor(center - span / 2), ceil(center + span / 2)
        if self.ylabel == "窗口比例":
            return 0, 1
        if self.ylabel == "能量占比 / %":
            return 0, 100
        padding = (ymax - ymin or max(abs(ymax), 1) * .1) * .08
        zero_based = bool(self.bars) or self.xlabel == "频率 / Hz"
        return (0 if ymin >= 0 and zero_based else ymin - padding), ymax + padding

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font())
        colors = DISPLAY["colors"]
        painter.fillRect(self.rect(), QColor(colors["background"]))
        fm = painter.fontMetrics()
        height = fm.height()
        painter.setPen(QColor(colors["inactive"]))
        painter.drawText(QRectF(8, 3, self.width() - 16, height), Qt.AlignmentFlag.AlignLeft, self.ylabel)
        series_colors = [QColor(colors["action"]), QColor(colors["axis_y"])]
        if len(self.series) > 1:
            widths = [fm.horizontalAdvance(name) + 28 for _, _, name in self.series]
            x = self.width() - sum(widths) - 8
            for number, ((_, _, name), width) in enumerate(zip(self.series, widths)):
                painter.setPen(QPen(series_colors[number % 2], 2))
                painter.drawLine(QPointF(x, 3 + height / 2), QPointF(x + 12, 3 + height / 2))
                painter.setPen(QColor(colors["inactive"]))
                painter.drawText(QRectF(x + 16, 3, width - 16, height), Qt.AlignmentFlag.AlignLeft, name)
                x += width
        points = [(float(x), float(y)) for xs, ys, _ in self.series for x, y in zip(xs, ys)
                  if x is not None and y is not None and isfinite(x) and isfinite(y)]
        if not points:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "该信号无有效数据" if self.series else "暂无数据")
            return
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin, ymax = min(y for _, y in points), max(y for _, y in points)
        if self.bars:
            xmin, xmax, ymin = -0.5, len(self.bars) - 0.5, 0
        dx = xmax - xmin or 1
        ymin, ymax = self._y_limits(ymin, ymax)
        y_steps = 4 if self.height() - height * 4.2 >= height * 5 else 2
        tick_labels = [f"{ymin + index / y_steps * (ymax - ymin):.3g}" for index in range(y_steps + 1)]
        if self.ylabel == "能量占比 / %":
            tick_labels = [f"{label}%" for label in tick_labels]
        left = max(fm.horizontalAdvance(label) for label in tick_labels) + 8
        right = max(12, fm.horizontalAdvance(f"{xmax:.4g}") / 2 + 4)
        bar_labels = [label.replace(" Hz", "") for label in self.bars] if self.bars else []
        stagger_labels = bool(bar_labels and max(fm.horizontalAdvance(label) + 8 for label in bar_labels)
                              > (self.width() - left - right) / len(bar_labels))
        rect = QRectF(left, height * 1.8, max(1, self.width() - left - right),
                      max(1, self.height() - height * (5.2 if stagger_labels else 4.2)))

        def point(x, y):
            return QPointF(rect.left() + (x - xmin) / dx * rect.width(),
                           rect.bottom() - (y - ymin) / (ymax - ymin) * rect.height())

        for index, label in enumerate(tick_labels):
            y = rect.bottom() - index / y_steps * rect.height()
            painter.setPen(QPen(QColor(colors["border"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor(colors["inactive"]))
            painter.drawText(QRectF(0, y - height / 2, left - 6, height),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
        painter.setPen(QPen(QColor(colors["placeholder"]), 1))
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.save()
        painter.setClipRect(rect.adjusted(-1, -1, 1, 1))
        for number, (xs, ys, _) in enumerate(self.series):
            color = series_colors[number % 2]
            painter.setPen(QPen(color, max(1.2, height / 10)))
            path = QPainterPath()
            started = False
            first_point = last_point = None
            for x, y in zip(xs, ys):
                if y is None or x is None or not isfinite(x) or not isfinite(y):
                    started = False
                    continue
                target = point(x, y)
                first_point = target if first_point is None else first_point
                last_point = target
                if self.bars:
                    bottom = point(x, 0)
                    width = rect.width() / len(self.bars) * .55
                    painter.fillRect(QRectF(target.x() - width / 2, target.y(), width,
                                            bottom.y() - target.y()), color)
                elif started:
                    path.lineTo(target)
                else:
                    path.moveTo(target)
                    started = True
                if len(xs) == 1 and not self.bars:
                    painter.drawEllipse(target, 2, 2)
            if self.fill and first_point is not None:
                area = QPainterPath(path)
                area.lineTo(QPointF(last_point.x(), rect.bottom()))
                area.lineTo(QPointF(first_point.x(), rect.bottom()))
                area.closeSubpath()
                fill = QColor(color)
                fill.setAlpha(30)
                painter.fillPath(area, fill)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        painter.restore()
        painter.setPen(QColor(colors["inactive"]))
        if self.bars:
            labels = bar_labels
            positions = [point(index, 0).x() for index in range(len(labels))]
        else:
            x_steps = 4 if rect.width() >= fm.horizontalAdvance(f"{xmax:.4g}") * 6 else 2
            values = [xmin + index / x_steps * dx for index in range(x_steps + 1)]
            labels = [f"{value:.4g}" for value in values]
            positions = [point(value, ymin).x() for value in values]
        for index, (label, x) in enumerate(zip(labels, positions)):
            width = fm.horizontalAdvance(label) + 8
            label_left = max(0, min(self.width() - width, x - width / 2))
            offset = height * (index % 2) if stagger_labels else 0
            painter.drawText(QRectF(label_left, rect.bottom() + 2 + offset, width, height),
                             Qt.AlignmentFlag.AlignHCenter, label)
        painter.drawText(QRectF(0, self.height() - height - 2, self.width(), height),
                         Qt.AlignmentFlag.AlignHCenter, self.xlabel)
