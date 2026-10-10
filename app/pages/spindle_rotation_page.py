"""主轴空启动建模、日常检测、人工标签和模型版本回看。"""

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor, isfinite
from zipfile import BadZipFile
import numpy as np

from PyQt6.QtCore import QDateTime, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLayout, QLineEdit, QMenu, QMessageBox,
    QPlainTextEdit, QProgressBar, QScrollArea, QSizePolicy, QSpinBox,
    QStyle, QStyledItemDelegate, QStyleOptionComboBox, QStyleOptionViewItem, QStylePainter,
    QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from app.resources import DISPLAY, fit_dialog, load_icon, make_button, make_note, set_status_light
from app.tasks import ServiceTask
from core.services.spindle_monitoring_service import SpindleMonitoringService, assess_compatibility, assess_score


LABELS = {"unconfirmed": "未判定", "healthy": "正常", "abnormal": "异常"}
CHANNELS = ["前轴承 Y · ACC1", "前轴承 X · ACC2", "平台 Y · ACC3", "温度 · NTC1 / RTD1", "主轴电流"]


def _local_time(value):
    if len(value) == 10:
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class CurrentModelDelegate(QStyledItemDelegate):
    """在模型文字右侧显示标准 SVG，列表与收起后的选择框共用绘制。"""

    def __init__(self, parent):
        super().__init__(parent)
        self.markers = [load_icon("common/current-model.svg", color, 16)
                        for color in (DISPLAY["colors"]["text"], "#ffffff")]

    def paint(self, painter, option, index):
        if not index.data(Qt.ItemDataRole.UserRole + 1):
            super().paint(painter, option, index)
            return
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        style = styled.widget.style()
        text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, styled, styled.widget)
        size = round(styled.fontMetrics.height() * 0.8)
        gap = round(size * 0.75)
        styled.text = styled.fontMetrics.elidedText(styled.text, Qt.TextElideMode.ElideRight,
                                                   text_rect.width() - size - 2 * gap)
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, styled, painter, styled.widget)
        marker_rect = QRect(text_rect.left() + styled.fontMetrics.horizontalAdvance(styled.text) + gap,
                            text_rect.center().y() - size // 2, size, size)
        selected = bool(styled.state & QStyle.StateFlag.State_Selected
                        and styled.state & QStyle.StateFlag.State_Active)
        self.markers[selected].paint(painter, marker_rect)


class ModelComboBox(QComboBox):
    """Qt 6.7 复用同一委托绘制收起标签；Qt 6.9 起沿用原生接口。"""

    def __init__(self):
        super().__init__()
        if hasattr(QComboBox, "setLabelDrawingMode"):
            self.setLabelDrawingMode(QComboBox.LabelDrawingMode.UseDelegate)

    def paintEvent(self, event):
        if hasattr(QComboBox, "setLabelDrawingMode"):
            super().paintEvent(event)
            return
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        if self.currentIndex() < 0:
            return
        label = QStyleOptionViewItem()
        label.initFrom(self)
        label.widget = self
        label.rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, option, QStyle.SubControl.SC_ComboBoxEditField, self)
        label.displayAlignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        index = self.model().index(self.currentIndex(), self.modelColumn(), self.rootModelIndex())
        self.itemDelegate().paint(painter, label, index)


class SpindleRotationPage(QScrollArea):
    label_saved = pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, service=None):
        super().__init__()
        self.service = service or SpindleMonitoringService()
        self.result = None
        self.uses_current_thresholds = False
        self.task = None
        self.task_completed = None
        self.task_status_key = "acquisition"
        self.auto_evaluation_attempts = set()
        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.timeout.connect(self._ensure_selection_result)
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

    def _button(self, text, callback, primary=False, track=True):
        button = make_button(text, callback, primary)
        if track:
            self.actions.append(button)
        return button

    @staticmethod
    def _combo(combo_class=QComboBox):
        combo = combo_class()
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
        self.model_select = self._combo(ModelComboBox)
        self.model_select.setItemDelegate(CurrentModelDelegate(self.model_select))
        self.model_select.setToolTip("右侧星标表示当前模型；按模型版本回看，旧评价保持不变")
        self.run_select.currentIndexChanged.connect(self._selection_changed)
        self.run_select.activated.connect(self._selection_changed)
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
        self.label_note.setPlaceholderText("填写判定原因或补充说明（选填）")
        self.label_button = self._button("保存人工判定", self._save_label, True)
        for widget in (self.run_select, self.model_select, self.signal_select,
                       self.label_select, self.training_check, self.label_note,
                       self.label_button):
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
        self.source_badge.setWordWrap(True)
        heading.addWidget(self.source_badge)
        body.addLayout(heading)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.result_values = {}
        self.result_notes = {}
        for key, title, hint in (
            ("analysis", "振动正常相容度", "径向 RMS · 尚无评价"),
            ("network", "网络正常相容度", "重建误差 P95 · 尚无评价"),
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
            note = make_note(hint)
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_notes[key] = note
            for widget in (label, value, note):
                content.addWidget(widget)
            metrics.addWidget(metric, 1)
        self.result_values["analysis"].setToolTip("径向速度 RMS 的双侧正常相容度；正常接受范围内为 1，过大或过小均降低。")
        self.result_values["network"].setToolTip("采集级重建误差 P95 的上尾正常相容度；仅误差增大降低。不是设备正常的后验概率。")
        self.result_values["temperature"].setToolTip("轴承温度有效采样点的中位数；缺失点不参与统计。")
        self.result_values["current"].setToolTip("稳定采集段内驱动器有效读数的有符号中位数，不是交流电流有效值；不参与网络评分。")
        body.addLayout(metrics)
        context = QHBoxLayout()
        self.result_context = make_note("")
        context.addWidget(self.result_context, 1)
        self.latest_button = self._button("返回最新评价", self._show_latest, True)
        self.latest_button.setVisible(False)
        context.addWidget(self.latest_button)
        body.addLayout(context)

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
        self.workflow_hint = make_note("")
        training.addWidget(self.workflow_hint)
        thresholds = self._section(body, "判定阈值", "thresholds")
        thresholds.addWidget(self._button("阈值设置", self._show_thresholds))
        log_area = QFrame()
        log_area.setObjectName("SpindleLogArea")
        log_layout = QVBoxLayout(log_area)
        log_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
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
        self.task_progress_note = make_note("尚未开始处理")
        self.task_progress_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        log_layout.addWidget(self.task_progress_note)
        body.addWidget(log_area, 1)
        actions = QFrame()
        actions.setObjectName("SpindleActions")
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.initial_button = self._button("批量导入", lambda: self._choose_packages("initial"), True)
        self.initial_button.setToolTip("选择一个 ZIP 导入整批采集，并确认正常、异常或未判定；正常样本可参与训练。")
        self.daily_button = self._button("日常导入", lambda: self._choose_packages("daily"), True)
        self.daily_button.setToolTip("导入后自动分析；已有模型时自动评估并保存结果，无需再点击重新评估。")
        self.evaluate_button = self._button("评估精度", self._evaluate, True)
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
        body.addWidget(make_note("主轴动作由原设备软件控制；此页导入已采集的数据包。工况与处理参数随数据和评价保留。"))
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
            table.blockSignals(True)
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
            table.blockSignals(False)

        def select(row, column):
            index = self.run_select.findData(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            if index == self.run_select.currentIndex():
                self._selection_changed()
            else:
                self.run_select.setCurrentIndex(index)

        def sync_row():
            run_id = self.run_select.currentData()
            editable = bool(self.task is None and run_id
                            and not self.service.trained_run_ids().intersection(self.service.import_group(run_id)))
            review_button.setEnabled(editable)
            review_button.setToolTip("判定当前选中的采集；批量数据按整批保存。" if editable
                                     else "已参与模型训练或校准的样本不能改判。" if run_id else "请先导入采集数据。")
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item.data(Qt.ItemDataRole.UserRole) == self.run_select.currentData():
                    table.setCurrentCell(row, 0)
                    table.scrollToItem(item)
                    break

        def review():
            if self.task is None and self.run_select.currentData():
                self._show_review(run_id=self.run_select.currentData())
                populate()
                sync_row()

        review_button = self._button("人工判定", review, track=False)
        table.cellClicked.connect(select)
        table.currentCellChanged.connect(lambda row, column, *_: select(row, column) if row >= 0 else None)
        self.run_select.currentIndexChanged.connect(sync_row)
        populate()
        sync_row()
        body.addWidget(table, 1)
        row = QHBoxLayout()
        row.addWidget(review_button)
        retry_requested = False
        def retry():
            nonlocal retry_requested
            if self.task is not None:
                return
            retry_requested = True
            dialog.accept()
        retry_button = self._button("补算缺失评价", retry, track=False)

        def update_actions(busy):
            sync_row()
            model = self.service.current_model
            evaluated = {entry["run_id"] for entry in self.service.state["results"]
                         if model and entry["model_version"] == model["version"]
                         and entry.get("score_kind") == "normal_compatibility_v1"}
            retry_button.setEnabled(bool(not busy and model and (model.get("reanalysis_status") != "complete"
                                                                  or set(self.service.runs) - evaluated)))

        self.busy_changed.connect(update_actions)
        update_actions(self.task is not None)
        retry_button.setToolTip("沿用当前网络，补算缺失或旧倍率评价并重建正常参考，无需重训。")
        row.addWidget(retry_button)
        row.addStretch()
        row.addWidget(self._button("关闭", dialog.reject, track=False))
        body.addLayout(row)
        fit_dialog(dialog, 980, min(580, 190 + 32 * table.rowCount()))
        try:
            dialog.exec()
        finally:
            self.run_select.currentIndexChanged.disconnect(sync_row)
            self.busy_changed.disconnect(update_actions)
            for widget in borrowed:
                widget.setParent(self.detail_holder)
                widget.hide()
            dialog.deleteLater()
        if retry_requested:
            self.model_select.setCurrentIndex(0)
            self._run_task("补算缺失评价", self.service.reanalyze_history, status_key="model")

    def _show_review(self, *, run_id=None):
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
        form.addRow("选择样本", package_select)
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
        index = 0 if run_id is None else next(
            (index for index in range(package_select.count()) if run_id in package_select.itemData(index)), -1)
        package_select.setCurrentIndex(index)

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
        form.addRow("判定结果", self.label_select)
        form.addRow("参与训练", self.training_check)
        form.addRow("判定备注", self.label_note)
        body.addLayout(form)
        body.addWidget(make_note("仅判定为正常且勾选“纳入下次训练”的样本可参与训练；批量导入按整批判定。"))
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
            if not uses_current_thresholds:
                self._show_result(displayed_result)
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
        self.model_select.addItem(f"最新评价 · {current_name}", None)
        self.model_select.addItem("未建模", "unmodeled")
        for item in reversed(self.service.models):
            name = self._model_name(item["version"])
            self.model_select.addItem(name, item["version"])
            self.model_select.setItemData(self.model_select.count() - 1,
                                         bool(model and item["version"] == model["version"]), Qt.ItemDataRole.UserRole + 1)
            self.model_select.setItemData(self.model_select.count() - 1, item["version"], Qt.ItemDataRole.ToolTipRole)
        self.model_select.setCurrentIndex(max(0, self.model_select.findData(model_version)))
        self.model_select.blockSignals(False)
        self.train_button.setText("重新训练" if model else "训练网络")
        self._selection_changed()
        self._set_busy(self.task is not None)

    def _set_status_light(self, key, state, details):
        status = {"missing": "未就绪", "partial": "待处理", "ready": "已就绪"}[state]
        set_status_light(self.status_lights[key], state, details, status)

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
            details += "\n至少 5 次正常候选采集（2 次训练、至少 3 次独立校准）"
        self._set_status_light("model", state, details)
        metrics = self.service.metric_settings
        configured = sum(metric[key] is not None for metric in metrics.values() for key in ("warning", "fault"))
        state = "partial" if configured else "missing"
        if configured == 4 and model and not self.service.thresholds_review_required:
            state = "ready"
        lines = []
        for feature, name in (("vibration", "振动"), ("network", "网络")):
            metric = metrics[feature]
            limits = " / ".join(f"{label} {'未设置' if metric[key] is None else f'≤ {metric[key]:g}'}"
                                for key, label in (("warning", "预警"), ("fault", "故障")))
            lines.append(f"{name}：α = {metric['alpha']:g}；{limits}")
        details = "\n".join(lines) + "\n两项相容度各用各自设置，任一触发即报警。"
        if self.service.thresholds_review_required:
            details += "\n新模型阈值待复核"
        elif configured and model is None:
            details += "\n建模后需复核阈值"
        self._set_status_light("thresholds", state, details)
        if not count:
            hint = "先批量导入正常样本，再训练网络；日常数据用“日常导入”。"
        elif not model:
            hint = (f"已有 {candidates} 次正常候选，可点击“训练网络”。" if candidates >= 5 else
                    f"正常候选 {candidates}/5 次；请导入样本或在“人工判定”中纳入训练。")
        elif model.get("reanalysis_status") != "complete":
            hint = "历史评价未完成，请在“样本与模型”中补算。"
        elif self.service.thresholds_review_required:
            hint = "新模型已建立，请在“阈值设置”中复核并保存。"
        elif configured < 4:
            hint = "模型已就绪；请补齐判定阈值，日常导入后会自动评估。"
        else:
            hint = "模型与阈值已就绪；日常导入后自动评估并保存。"
        self.workflow_hint.setText(hint)
        self.train_button.setToolTip(f"使用 {candidates} 次正常候选从头训练；至少 2 次训练、3 次独立校准。3 次仅为计算门槛。")

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
        self.selection_timer.start(0)

    def _ensure_selection_result(self):
        """合并选择信号；只为最新视图补算缺失或旧倍率评价。"""
        run_id = self.run_select.currentData()
        if (not run_id or self.model_select.currentData() is not None
                or not self.uses_current_thresholds or self.task is not None):
            return
        if self.result and self.result.get("score_kind") == "normal_compatibility_v1":
            return
        model = self.service.current_model
        key = (run_id, model["version"] if model else None)
        if key in self.auto_evaluation_attempts:
            return
        self.auto_evaluation_attempts.add(key)
        self.source_badge.setText("正在更新当前采集评价")
        self._run_task("自动更新当前采集评价", lambda progress: self.service.ensure_latest_result(run_id, progress))

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
        if not current_thresholds:
            self.selection_timer.stop()
        result = self.service.display_result(result, current_settings=current_thresholds)
        metrics = self.service.result_metric_settings(result) if result else self.service.metric_settings
        self.result = result
        self.uses_current_thresholds = current_thresholds
        self.latest_button.setVisible(bool(result) and not current_thresholds)
        self.result_context.setVisible(bool(result))
        if result:
            mode = "当前设置" if current_thresholds else "历史快照 · 当时设置"
            role = {"training": "训练样本回评，不能作为独立验证", "calibration": "校准样本回评，不能作为独立验证",
                    "independent": "未参与此模型建模"}[result["role"]]
            self.result_context.setText(f"{mode} · {self._model_name(result['model_version'])} · {role}")
        values = {"analysis": result.get("analysis_score") if result else None,
                  "network": result.get("score") if result else None,
                  "temperature": result.get("temperature_c", [None])[0] if result else None,
                  "current": result.get("current_a") if result else None}
        for key, value in values.items():
            self.result_values[key].setText("—" if value is None or not isfinite(value) else f"{value:.4g}")
            self.result_values[key].setStyleSheet("")
        for key, feature, raw_key, p_key, description in (
            ("analysis", "vibration", "xy_rms_mm_s", "vibration_p_value", "本次径向 RMS（R，mm/s）；双侧检验，过大或过小均扣分。"),
            ("network", "network", "reconstruction_error_p95", "network_p_value", "本次全部窗口重建误差的 P95（E）；上尾检验，仅误差增大扣分。"),
        ):
            notes = [description, "C = min(1, p / α)，正常接受范围内为 1；不是设备正常的后验概率。",
                     "两项针对不同特征，数值不要求相近；低于 1 均表示偏离各自的正常接受范围。"]
            score = values[key]
            state = "尚无评价" if not result else "参考无效 / 未校准"
            if score is not None and isfinite(score):
                state = "正常接受范围" if score == 1 else "偏离正常参考"
            self.result_notes[key].setText(f"{'径向 RMS' if key == 'analysis' else '重建误差 P95'}\n{state}")
            if result:
                raw_value = result.get(raw_key)
                p_value = result.get(p_key)
                notes.append(f"{'R' if feature == 'vibration' else 'E'}：{'无效 / 未计算' if raw_value is None else f'{raw_value:.6g}'}")
                notes.append(f"p：{'无效 / 未计算' if p_value is None else f'{p_value:.6g}'}；α：{metrics[feature]['alpha']:g}")
                calibration = result.get("compatibility", {}).get(feature, {})
                if calibration.get("message"):
                    notes.append(calibration["message"])
                if calibration.get("M") is not None:
                    notes.append(f"独立正常校准：{calibration['M']} 次")
                if calibration.get("mu") is not None and calibration.get("s") is not None:
                    notes.append(f"对数参考 μ = {calibration['mu']:.6g}，s = {calibration['s']:.6g}")
                if calibration.get("u") is not None:
                    notes.append(f"预测偏离 u = {calibration['u']:.6g}")
                if calibration.get("normality_p_value") is not None:
                    notes.append(f"对数高斯适用性检验 p = {calibration['normality_p_value']:.4g}；不能证明独立性或高斯分布。")
                if result.get("score_kind") != "normal_compatibility_v1" and result.get("model_version"):
                    notes.append("旧倍率记录：请重新评估以计算正常相容度。")
            else:
                notes.append("尚无正常参考，请先确认正常样本并建立模型。")
            self.result_values[key].setToolTip("\n".join(notes))
        result_detail = ""
        self.source_badge.setStyleSheet("")
        self.source_badge.setToolTip("")
        if result:
            source = "历史回放" if result["source_type"] == "historical_replay" else "检测数据"
            assessment = (assess_compatibility(result, metrics, result.get("thresholds_review_required", False))
                          if result.get("score_kind") == "normal_compatibility_v1" else result["assessment"])
            status_text = {"unavailable": "正常相容度无效", "legacy": "旧倍率记录 · 待重新评估",
                           "fault": "故障 / 建议检修", "warning": "预警 / 建议复测",
                           "deviation": "偏离参考 · 未达报警阈值"}.get(
                assessment["status"], assessment["message"])
            self.source_badge.setText(f"{source} · {status_text}")
            if any(item.get("calibration_level") == "preliminary" for item in result.get("compatibility", {}).values()):
                self.source_badge.setText(self.source_badge.text() + " · 初步校准")
            if assessment["status"] in ("warning", "fault"):
                self.source_badge.setStyleSheet(f"color:{DISPLAY['colors']['error']};")
            elif assessment["status"] == "deviation" or (
                    assessment["status"] == "unconfigured" and any(
                        value is not None and value < 1 for value in (result.get("score"), result.get("analysis_score")))):
                self.source_badge.setStyleSheet(f"color:{DISPLAY['colors']['warning']};")
            if (assessment["status"] not in ("review_required", "legacy")
                    and not result.get("thresholds_review_required")):
                for key, score_key, feature in (("analysis", "analysis_score", "vibration"), ("network", "score", "network")):
                    if assess_score(result.get(score_key), metrics[feature])["status"] in ("warning", "fault"):
                        self.result_values[key].setStyleSheet(f"color:{DISPLAY['colors']['error']};")
                    elif result.get(score_key) is not None and result[score_key] < 1:
                        self.result_values[key].setStyleSheet(f"color:{DISPLAY['colors']['warning']};")
            role = {"training": "训练样本回评", "calibration": "校准样本回评", "independent": "未参与此模型建模"}[result["role"]]
            run = self.service.runs[result["run_id"]]
            condition = run.get("condition", {})
            remounted = "刀具重装" if condition.get("tool_remounted") else "未标记重装"
            pressure = condition.get("seal_pressure_mpa")
            pressure_text = "密封气压未记录" if pressure is None else f"密封气压 {pressure:g} MPa"
            operation = "空转" if run["operation"] == "idle" else run["operation"]
            legacy = result.get("score_kind") != "normal_compatibility_v1" and result.get("model_version")
            unit = " 倍（旧）" if legacy else ""
            threshold_title = "当前阈值" if current_thresholds and not legacy else "当时阈值"
            result_detail = (
                f"{run['speed_rpm']:g} rpm · {operation} · {remounted} · {pressure_text}\n"
                f"采集：{_local_time(result['captured_at'])} · {role}\n"
                f"评估：{_local_time(result['evaluated_at'])} · 模型：{self._model_name(result['model_version'])}\n"
                f"{result['window_count']} 个 1 秒窗口"
            )
            for feature, name in (("vibration", "振动"), ("network", "网络")):
                metric = metrics[feature]
                warning = "未设置" if metric["warning"] is None else f"{metric['warning']:g}{unit}"
                fault = "未设置" if metric["fault"] is None else f"{metric['fault']:g}{unit}"
                result_detail += f"\n{threshold_title}（{name}）：预警 {warning} / 故障 {fault}；α = {metric['alpha']:g}"
            if not legacy:
                result_detail += "\n两项相容度任一 ≤ 阈值即触发；故障优先。"
            elif result.get("legacy_assessment"):
                result_detail += f"\n旧倍率当时判定：{result['legacy_assessment']['message']}"
            self.source_badge.setToolTip(f"{assessment['message']}\n{result_detail}")
        else:
            self.source_badge.setText("暂无数据" if not self.service.runs else "暂无评价")
        run = self.service.runs.get(self.run_select.currentData())
        if run:
            short_name = run["run_name"].split("rpm_", 1)[-1]
            mode = "当前设置" if current_thresholds else "历史设置"
            self.current_run_button.setText(f"{_local_time(run['captured_at'])} · {short_name} · {mode}")
            self.current_run_button.setToolTip(result_detail or self.run_select.currentText())
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
        if channel < 3:
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
        reference = r.get("reference_features", {}).get("network", [])
        current = r.get("reconstruction_error_p95")
        if (reference and current is not None and isfinite(current)
                and r.get("compatibility", {}).get("network", {}).get("status") == "valid"):
            edges = np.histogram_bin_edges(reference + [current], bins=20)
            centers = ((edges[:-1] + edges[1:]) / 2).tolist()
            series = [(centers, (np.histogram(reference, edges)[0] / len(reference)).tolist(), "正常采集"),
                      ([current, current], [0, 1], "本次 P95")]
            self.plots["distribution"].set_data(series, "采集级重建误差 P95", "采集比例")
            self.plots["distribution"].setToolTip("每次独立校准采集仅贡献一个 P95；竖线为本次采集 P95。未将窗口误差混入正常参考。")
        else:
            self.plots["distribution"].set_data([], "", "正常参考不可用")
            self.plots["distribution"].setToolTip("请建立与当前工况、采集长度和预处理匹配的独立正常参考。")

    def _load_order_energy(self, run_id):
        task = ServiceTask(lambda progress: (run_id, self.service.order_band_energy(run_id)))
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
        self.train_button.setEnabled(not busy and len(self.service.training_candidates()) >= 5)
        self.evaluate_button.setEnabled(not busy and has_run)
        if busy:
            hint = "正在处理任务，请等待完成后再评估。"
        elif not has_run:
            hint = "请先通过“批量导入”或“日常导入”载入采集数据，再评估精度。"
        elif self.service.current_model:
            hint = "使用当前模型和阈值重新评估所选采集，并追加评价记录；日常导入已自动评估。"
        else:
            hint = "分析所选采集的真实信号；建立正常参考后才显示相容度。"
        self.evaluate_button.setToolTip(hint)
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
        self.busy_changed.emit(busy)

    def _run_task(self, title, operation, completed=None, *, status_key="acquisition"):
        if self.task is not None:
            return
        self._log(title)
        task = ServiceTask(operation)
        self.task = task
        self.task_completed = completed
        self.task_status_key = status_key
        self.progress.setValue(0)
        self.task_progress_note.setText(f"{title}：准备处理")
        self.progress.setToolTip(self.task_progress_note.text())
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

    def _task_done(self, value):
        historical_result = self.result if not self.uses_current_thresholds else None
        completed = self.task_completed
        self.task_completed = None
        self.task = None
        self._refresh()
        if historical_result is not None:
            self._show_result(historical_result)
        if completed:
            completed(value)
        summary = value.get("summary", "处理完成") if isinstance(value, dict) else "处理完成"
        if isinstance(value, dict):
            for package in value.get("packages", []):
                if package["status"] == "failed":
                    self._log(package["message"], "ERROR")
        self._log(summary)
        self.progress.setValue(100)
        self.task_progress_note.setText(summary)
        self.progress.setToolTip(self.task_progress_note.text())

    def _task_failed(self, message):
        historical_result = self.result if not self.uses_current_thresholds else None
        self.task_completed = None
        self.task = None
        self._refresh()
        if historical_result is not None:
            self._show_result(historical_result)
        self._log(message, "ERROR")
        self.progress.setValue(0)
        self.task_progress_note.setText(f"处理失败：{message}")
        self.progress.setToolTip(self.task_progress_note.text())
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
        def operation(progress):
            try:
                return (self.service.import_batch(paths, progress, label=batch_label) if purpose == "initial"
                        else self.service.import_packages(paths, purpose, progress))
            except BadZipFile as error:
                raise ValueError("数据包不是有效的 ZIP，或文件已损坏。请重新选择或导出数据包。") from error

        def completed(value):
            runs = value["runs"] if purpose == "initial" else value
            if runs:
                self.model_select.setCurrentIndex(0)
                self._refresh(runs[-1]["run_id"])
        self._run_task("导入建模数据" if purpose == "initial" else "导入并分析日常检测",
                       operation, completed)

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
        form.addRow(make_note(f"使用 {len(self.service.training_candidates())} 次可建模采集；训练与校准按采集隔离。\n每次随机初始化，完成后自动重算历史。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始训练")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
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
        for feature, name in (("vibration", "振动正常相容度"), ("network", "网络正常相容度")):
            heading = QLabel(name)
            heading.setProperty("robotSectionTitle", True)
            form.addRow(heading)
            fields[feature] = {}
            for key, title in (("alpha", "评分灵敏度 α"), ("warning", "预警相容度 ≤"), ("fault", "故障相容度 ≤")):
                value = self.service.metric_settings[feature][key]
                field = QLineEdit("" if value is None else str(value))
                field.setObjectName(f"spindle_{feature}_{key}" + ("" if key == "alpha" else "_threshold"))
                field.setProperty("robotInput", True)
                if key != "alpha":
                    field.setPlaceholderText("留空表示未设置")
                form.addRow(title, field)
                fields[feature][key] = field
        form.addRow(make_note("两项各用各自设置，任一相容度 ≤ 对应阈值即触发，故障优先。每项 0 ≤ 故障 < 预警 < 1；阈值可留空。\n"
                             "α 越大，越早降低评分；默认 0.05 对应模型成立时单项 95% 的正常接受范围。"
                             "α 须在 0～1 之间。调报警阈值不改变评分；满分不保证设备正常。"))
        preview = make_note("")
        preview.setObjectName("spindle_threshold_preview")
        form.addRow(preview)
        current = self.service.latest_result(self.run_select.currentData())

        def read_values():
            return {feature: {key: float(field.text()) if field.text().strip() or key == "alpha" else None
                              for key, field in entries.items()} for feature, entries in fields.items()}

        def update_preview():
            try:
                values = read_values()
                shown = self.service.display_result(current, metric_settings=values)
            except ValueError:
                preview.setText("当前采集预览：请填写有效的 α 和报警阈值。")
                return
            if not shown or shown.get("score_kind") != "normal_compatibility_v1":
                preview.setText("当前采集尚无相容度评价；保存后自动补算。")
                return
            scores = ["—" if shown[key] is None else f"{shown[key]:.4g}" for key in ("analysis_score", "score")]
            coverage = "；".join(f"{name} {(1 - values[feature]['alpha']) * 100:g}%"
                                 for feature, name in (("vibration", "振动"), ("network", "网络")))
            preview.setText(f"当前模型最新评价（保存后）：振动 {scores[0]}，网络 {scores[1]}\n"
                            f"{shown['assessment']['message']}\n模型成立时单项正常接受覆盖：{coverage}")

        for entries in fields.values():
            for field in entries.values():
                field.textChanged.connect(update_preview)
        update_preview()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存阈值")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        def save():
            try:
                values = read_values()
            except ValueError:
                QMessageBox.warning(dialog, "阈值未保存", "请输入有效数字；报警阈值可留空，两项 α 必须填写。")
                return
            try:
                self.service.set_metric_settings(values)
            except (ValueError, OSError) as error:
                QMessageBox.warning(dialog, "阈值未保存", str(error))
                return
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        if not self.uses_current_thresholds and self.result:
            form.addRow(make_note("正在查看历史快照。保存后返回当前模型的最新评价；历史分数及当时设置保留。"))
        form.addRow(buttons)
        fit_dialog(dialog, 520)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.model_select.blockSignals(True)
            self.model_select.setCurrentIndex(0)
            self.model_select.blockSignals(False)
            self._refresh()

    def _show_latest(self):
        self.model_select.blockSignals(True)
        self.model_select.setCurrentIndex(0)
        self.model_select.blockSignals(False)
        self._selection_changed()

    def _show_alerts(self):
        self._show_history(alerts_only=True)

    def _show_history(self, alerts_only=False):
        dialog = QDialog(self)
        dialog.setWindowTitle("主轴预警记录" if alerts_only else "检测历史 · 保留各模型版本与当时阈值")
        layout = QVBoxLayout(dialog)
        table = QTableWidget()
        table.setProperty("robotTable", True)
        columns = ["采集时间", "采集名称", "模型", "振动相容度", "网络相容度", "判定", "数据用途"]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        entries = [entry for entry in reversed(self.service.state["results"])
                   if not alerts_only or entry["assessment"]["status"] in ("warning", "fault")]
        table.setRowCount(len(entries))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, result in enumerate(entries):
            legacy = result.get("score_kind") != "normal_compatibility_v1" and result.get("model_version")
            scores = ["—" if result.get(key) is None else f"{result[key]:.4g}"
                      for key in ("analysis_score", "score")]
            if legacy:
                scores = ["旧倍率记录", "—" if result.get("score") is None else f"旧：{result['score']:.4g} 倍"]
            values = [_local_time(result["captured_at"]), result["run_name"], self._model_name(result["model_version"]),
                      *scores, result["assessment"]["message"],
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
        layout.addWidget(make_note("双击或回车查看当时结果。人工标签保持独立；重算不覆盖旧评价。"))
        layout.addWidget(self._button("关闭", dialog.reject, track=False), 0, Qt.AlignmentFlag.AlignRight)
        fit_dialog(dialog, 1100, min(560, 150 + 32 * table.rowCount()))
        dialog.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_layout_direction()

    def _update_layout_direction(self):
        stacked = self.width() < 1000 * getattr(self.window(), "ui_scale", 1.0)
        self.columns.setDirection(QHBoxLayout.Direction.TopToBottom if stacked else QHBoxLayout.Direction.LeftToRight)
        self.columns.setStretch(0, 0 if stacked else 13)
        self.columns.setStretch(1, 0 if stacked else 7)


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
        if self.ylabel == "采集比例":
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
