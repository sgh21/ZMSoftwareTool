"""仿真图像核验窗口；核验服务使用独立输出目录，不修改主页面数据。"""

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThreadPool, Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QProgressBar, QTabWidget, QVBoxLayout,
)

from app.dialogs import read_batch_records
from app.resources import fit_dialog, make_button, make_note
from app.tasks import ServiceTask
from core.services.position_monitoring_service import METRIC_LABELS
from core.services.position_simulation_debug import run_simulation_check


class RobotPositionSimulationDialog(QDialog):
    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.task = None
        self.dataset = None
        self.runs = {}
        self.report = None
        self.setWindowTitle("机器人定位仿真核验")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(make_note(
            "从仿真图像重新测量，再与末端真值对照。仿真结果用于检查算法，不能代替实机精度试验。"
        ))
        form = QFormLayout()
        source_row = QHBoxLayout()
        self.data_root = QLineEdit(str(page.service.root / "debug"))
        self.data_root.setProperty("robotInput", True)
        self.data_root.setAccessibleName("仿真数据目录")
        self.browse_button = make_button("选择目录", self._browse)
        source_row.addWidget(self.data_root, 1)
        source_row.addWidget(self.browse_button)
        form.addRow("仿真数据目录", source_row)
        batch_row = QHBoxLayout()
        self.baseline_batch = QComboBox()
        self.current_batch = QComboBox()
        for title, combo in (("基准批次", self.baseline_batch), ("复测批次", self.current_batch)):
            combo.setProperty("robotInput", True)
            combo.setAccessibleName(title)
            batch_row.addWidget(QLabel(title))
            batch_row.addWidget(combo, 1)
            combo.currentIndexChanged.connect(self._check_selection)
        form.addRow("对照批次", batch_row)
        layout.addLayout(form)
        self.source_status = QLabel("请选择包含 parameters.json 的仿真数据目录。")
        self.source_status.setWordWrap(True)
        self.source_status.setProperty("robotNote", True)
        layout.addWidget(self.source_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.progress_note = QLabel("尚未开始核验")
        self.progress_note.setWordWrap(True)
        layout.addWidget(self.progress_note)
        self.summary = QLabel("末端测量误差：—")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.results = QTabWidget()
        self.metric_table = page._table([
            "指标", "对照项", "X / mm", "Y / mm", "Z / mm", "空间指标 / mm",
        ])
        self.metric_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.error_table = page._table([
            "统计项", "X / mm", "Y / mm", "Z / mm", "位置误差 / mm", "姿态误差 / °",
        ])
        self.results.addTab(self.metric_table, "三项指标对照")
        self.results.addTab(self.error_table, "末端测量误差")
        layout.addWidget(self.results, 1)
        self.report_path = QLabel("报告位置：—")
        self.report_path.setWordWrap(True)
        self.report_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.report_path.setProperty("robotNote", True)
        layout.addWidget(self.report_path)
        actions = QHBoxLayout()
        self.calibration_button = make_button("相机与手眼标定", self._open_calibration)
        self.run_button = make_button("开始仿真核验", self._run, True)
        self.run_button.setEnabled(False)
        self.close_button = make_button("关闭", self.reject)
        actions.addWidget(self.calibration_button)
        actions.addStretch()
        actions.addWidget(self.run_button)
        actions.addWidget(self.close_button)
        layout.addLayout(actions)
        self.data_root.textChanged.connect(self._source_changed)
        self.data_root.editingFinished.connect(self._load_dataset)

    def _browse(self):
        selected = QFileDialog.getExistingDirectory(self, "选择仿真数据目录", self.data_root.text())
        if selected:
            self.data_root.setText(selected)
            self._load_dataset()

    def _source_changed(self):
        self.dataset = None
        self.runs = {}
        self.baseline_batch.clear()
        self.current_batch.clear()
        self.run_button.setEnabled(False)
        self._reset_result()
        self.source_status.setText("目录已更改，离开输入框后读取批次。")

    def _reset_result(self):
        self.report = None
        self.metric_table.setRowCount(0)
        self.error_table.setRowCount(0)
        self.summary.setText("末端测量误差：—")
        self.report_path.setText("报告位置：—")
        self.progress.setValue(0)
        self.progress_note.setText("尚未开始核验")

    def _load_dataset(self):
        if self.task is not None:
            return
        dataset = Path(self.data_root.text().strip())
        try:
            parsed = read_batch_records(dataset)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._source_changed()
            self.source_status.setText(f"无法读取仿真数据：{error}")
            return
        previous = (self.baseline_batch.currentData(), self.current_batch.currentData())
        self.dataset = dataset.resolve()
        self.runs = parsed
        for combo, preferred, fallback in (
            (self.baseline_batch, previous[0] or "B001", 0),
            (self.current_batch, previous[1] or "B002", min(1, len(parsed) - 1)),
        ):
            combo.blockSignals(True)
            combo.clear()
            for batch_id in parsed:
                combo.addItem(batch_id, batch_id)
            index = combo.findData(preferred)
            combo.setCurrentIndex(index if index >= 0 else fallback)
            combo.blockSignals(False)
        self._check_selection()

    def _check_selection(self):
        if not self.runs:
            return
        baseline, current = self.baseline_batch.currentData(), self.current_batch.currentData()
        if baseline is None or current is None:
            return
        if self.report is not None and (self.report["baseline_batch_id"], self.report["current_batch_id"]) != (baseline, current):
            self._reset_result()
        missing = [str(self.runs[batch]) for batch in (baseline, current)
                   if not self.runs[batch].is_file()]
        if missing:
            message = "缺少批次记录：" + "；".join(missing)
        elif baseline == current:
            message = "请选择两个不同批次。"
        else:
            message = f"已读取 {len(self.runs)} 个批次，将对照 {baseline} → {current}。"
        self.source_status.setText(message)
        self.run_button.setEnabled(not missing and baseline != current and self.task is None)

    def _set_running(self, running):
        for widget in (self.data_root, self.browse_button, self.baseline_batch,
                       self.current_batch, self.calibration_button, self.close_button):
            widget.setEnabled(not running)
        self.run_button.setEnabled(False)
        if not running:
            self._check_selection()

    def _run(self):
        if self.task is not None or self.dataset is None:
            return
        self._check_selection()
        if not self.run_button.isEnabled():
            return
        dataset = self.dataset
        baseline = self.baseline_batch.currentData()
        current = self.current_batch.currentData()
        output = (self.page.service.root / "data" / "processed" / "robot_position_simulation"
                  / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self._reset_result()
        self.summary.setText("正在测量所选批次图像…")
        self.report_path.setText("报告位置：核验完成后显示")
        self.progress.setValue(0)
        self.progress_note.setText("准备开始仿真核验")
        self.task = ServiceTask(lambda progress: run_simulation_check(
            dataset, output, baseline, current, progress,
        ))
        self.task.signals.progress.connect(self._progress)
        self.task.signals.completed.connect(self._completed)
        self.task.signals.failed.connect(self._failed)
        self._set_running(True)
        QThreadPool.globalInstance().start(self.task)

    def _progress(self, percent, message):
        self.progress.setValue(percent)
        self.progress_note.setText(message)

    @staticmethod
    def _number(value):
        return "—" if value is None else f"{value:.6f}"

    def _completed(self, result):
        self.task = None
        self._set_running(False)
        self.report = result["report"]
        metric_rows = []
        for metric in self.report["metric_comparison"]:
            for key, label in (("measured", "图像测量"), ("truth", "仿真真值"),
                               ("difference", "测量 − 真值")):
                metric_rows.append([METRIC_LABELS[metric["metric"]], label, *[self._number(value) for value in metric[key]]])
        self.page._fill_table(self.metric_table, metric_rows)
        summary = self.report["measurement_summary"]
        error_rows = []
        for prefix, label in (("mean", "平均绝对值"), ("p95", "95% 分位数"), ("max", "最大值")):
            values = [*summary[f"{prefix}_abs_xyz_mm"], summary[f"{prefix}_distance_mm"],
                      summary[f"{prefix}_rotation_deg"]]
            error_rows.append([label, *[self._number(value) for value in values]])
        self.page._fill_table(self.error_table, error_rows)
        self.summary.setText(
            f"已核验 {summary['sample_count']} 张图像；末端位置误差均值 "
            f"{self._number(summary['mean_distance_mm'])} mm，"
            f"最大值 {self._number(summary['max_distance_mm'])} mm。"
        )
        self.report_path.setText(f"报告位置：{result['report_path']}")
        self.progress.setValue(100)
        self.progress_note.setText("仿真核验完成，结果及测量记录已保存在独立目录。")

    def _failed(self, message):
        self.task = None
        self._set_running(False)
        self.progress_note.setText(f"核验失败：{message}")
        self.summary.setText("核验未完成，未生成完整对照结果。")
        self.report_path.setText("报告位置：未生成完整报告")

    def _open_calibration(self):
        from app.dialogs.robot_camera_calibration_dialog import RobotCameraCalibrationDialog

        dialog = RobotCameraCalibrationDialog(self.page)
        if self.dataset is not None:
            dialog.data_root.setText(str(self.dataset))
            dialog._load_dataset()
        fit_dialog(dialog, 920, 720)
        dialog.exec()

    def reject(self):
        if self.task is not None:
            self.progress_note.setText("核验正在运行，请等待完成后关闭。")
            return
        super().reject()

    def closeEvent(self, event):
        if self.task is not None:
            event.ignore()
        else:
            super().closeEvent(event)
