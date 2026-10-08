"""独立相机与手眼标定调试窗口；只导出文件，不更改主页面参数。"""

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThreadPool, Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QVBoxLayout,
)

from app.dialogs import read_batch_records
from app.resources import make_button, make_note
from app.tasks import ServiceTask
from debug.diagnostics.camera_calibration import calibrate_dataset


class RobotCameraCalibrationDialog(QDialog):
    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.task = None
        self.dataset = None
        self.runs = {}
        self.report = None
        self.setWindowTitle("相机与手眼标定")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(make_note(
            "使用所选批次图像标定相机内参，并结合机器人记录位姿标定手眼。"
            "生成的参数请在主页面手动加载。"
        ))
        form = QFormLayout()
        source_row = QHBoxLayout()
        self.data_root = QLineEdit(str(page.service.root / "data" / "robot_error"))
        self.data_root.setProperty("robotInput", True)
        self.data_root.setAccessibleName("标定数据目录")
        self.browse_button = make_button("选择目录", self._browse)
        source_row.addWidget(self.data_root, 1)
        source_row.addWidget(self.browse_button)
        form.addRow("标定数据目录", source_row)
        self.batch = self._combo(form, "标定批次", [])
        self.intrinsic_mode = self._combo(form, "相机内参来源", [
            ("从图像标定", "calibrate"), ("读取记录参考值", "reference"),
        ])
        self.distortion_mode = self._combo(form, "畸变处理", [
            ("标定 5 项畸变", "estimate"), ("固定零畸变（理想针孔）", "zero"),
        ])
        self.hand_eye_mode = self._combo(form, "手眼参数来源", [
            ("从图像与机器人位姿标定", "calibrate"), ("读取记录参考值", "reference"),
        ])
        self.method = self._combo(form, "手眼标定方法", [("PARK", "PARK"), ("TSAI", "TSAI")])
        layout.addLayout(form)
        self.source_status = QLabel("请选择包含 parameters.json 的数据目录。")
        self.source_status.setWordWrap(True)
        self.source_status.setProperty("robotNote", True)
        layout.addWidget(self.source_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.summary = QLabel("尚未开始标定")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("标定进度、重投影误差和手眼一致性残差")
        layout.addWidget(self.output, 1)
        self.report_path = QLabel("输出文件：—")
        self.report_path.setWordWrap(True)
        self.report_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.report_path.setProperty("robotNote", True)
        layout.addWidget(self.report_path)
        actions = QHBoxLayout()
        actions.addStretch()
        self.run_button = make_button("开始标定 / 导出", self._run, True)
        self.run_button.setEnabled(False)
        self.close_button = make_button("关闭", self.reject)
        actions.addWidget(self.run_button)
        actions.addWidget(self.close_button)
        layout.addLayout(actions)
        self.data_root.textChanged.connect(self._source_changed)
        self.data_root.editingFinished.connect(self._load_dataset)
        for combo in (self.batch, self.intrinsic_mode, self.distortion_mode,
                      self.hand_eye_mode, self.method):
            combo.currentIndexChanged.connect(self._selection_changed)

    def _combo(self, form, label, options):
        combo = QComboBox()
        combo.setProperty("robotInput", True)
        combo.setAccessibleName(label)
        for text, value in options:
            combo.addItem(text, value)
        form.addRow(label, combo)
        return combo

    def _browse(self):
        selected = QFileDialog.getExistingDirectory(self, "选择标定数据目录", self.data_root.text())
        if selected:
            self.data_root.setText(selected)
            self._load_dataset()

    def _reset_result(self):
        self.report = None
        self.progress.setValue(0)
        self.summary.setText("尚未开始标定")
        self.output.clear()
        self.report_path.setText("输出文件：—")

    def _source_changed(self):
        self.dataset = None
        self.runs = {}
        self.batch.clear()
        self._reset_result()
        self.run_button.setEnabled(False)
        self.source_status.setText("目录已更改，离开输入框后读取批次。")

    def _load_dataset(self):
        if self.task is not None:
            return
        dataset = Path(self.data_root.text().strip())
        try:
            parsed = read_batch_records(dataset)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._source_changed()
            self.source_status.setText(f"无法读取标定数据：{error}")
            return
        selected = self.batch.currentData() or "B000"
        self.dataset = dataset.resolve()
        self.runs = parsed
        self.batch.blockSignals(True)
        self.batch.clear()
        for batch in parsed:
            self.batch.addItem(batch, batch)
        index = self.batch.findData(selected)
        self.batch.setCurrentIndex(index if index >= 0 else 0)
        self.batch.blockSignals(False)
        self._check_selection()

    def _selection_changed(self):
        self._reset_result()
        self._check_selection()

    def _check_selection(self):
        running = self.task is not None
        self.method.setEnabled(not running and self.hand_eye_mode.currentData() == "calibrate")
        self.distortion_mode.setEnabled(not running and self.intrinsic_mode.currentData() == "calibrate")
        batch = self.batch.currentData()
        valid = batch in self.runs and self.runs[batch].is_file()
        if batch in self.runs:
            if valid:
                self.source_status.setText(f"已读取 {len(self.runs)} 个批次；本次使用 {batch}。")
            else:
                self.source_status.setText(f"缺少批次记录：{self.runs[batch]}")
        self.run_button.setEnabled(valid and not running)

    def _set_running(self, running):
        for widget in (self.data_root, self.browse_button, self.batch, self.intrinsic_mode,
                       self.hand_eye_mode, self.close_button):
            widget.setEnabled(not running)
        self._check_selection()

    def _run(self):
        if self.task is not None or self.dataset is None:
            return
        self._check_selection()
        if not self.run_button.isEnabled():
            return
        dataset = self.dataset
        options = {
            "batch_id": self.batch.currentData(),
            "intrinsic_mode": self.intrinsic_mode.currentData(),
            "distortion_mode": self.distortion_mode.currentData(),
            "hand_eye_mode": self.hand_eye_mode.currentData(),
            "method": self.method.currentData(),
        }
        output = (self.page.service.root / "data" / "processed" / "robot_camera_calibration"
                  / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self._reset_result()
        self.summary.setText("正在处理标定图像…")
        self.task = ServiceTask(lambda progress: calibrate_dataset(dataset, output, progress=progress, **options))
        self.task.signals.progress.connect(self._progress)
        self.task.signals.completed.connect(self._completed)
        self.task.signals.failed.connect(self._failed)
        self._set_running(True)
        QThreadPool.globalInstance().start(self.task)

    def _progress(self, percent, message):
        self.progress.setValue(percent)
        self.summary.setText(f"{percent}% · {message}")

    @staticmethod
    def _number(value):
        return "—" if value is None else f"{value:.6f}"

    def _completed(self, result):
        self.task = None
        self._set_running(False)
        self.report = result["report"]
        camera = self.report["camera_calibration"]
        hand_eye = self.report["hand_eye_calibration"]
        self.progress.setValue(100)
        self.summary.setText(
            f"{self.report['batch_id']} · {self.report['sample_count']} 张图像处理完成；"
            f"相机重投影 RMS {self._number(camera['rms_px'])} px。"
        )
        matrix = camera["camera_matrix"]
        self.output.appendPlainText(
            f"\n相机内参来源：{self.intrinsic_mode.currentText()}\n"
            f"fx={matrix[0][0]:.6f}，fy={matrix[1][1]:.6f}，"
            f"cx={matrix[0][2]:.6f}，cy={matrix[1][2]:.6f} px\n"
            f"畸变参数：{camera['dist_coeffs']}\n"
            f"手眼来源：{self.hand_eye_mode.currentText()}\n"
            f"固定靶标一致性残差：平移 RMS {self._number(hand_eye['translation_rms_mm'])} mm，"
            f"旋转 RMS {self._number(hand_eye['rotation_rms_deg'])}°。"
        )
        for warning in self.report.get("warnings", []):
            self.output.appendPlainText(warning)
        self.output.appendPlainText("参数已导出。请在主页面加载系统参数文件；一致性残差不等于机器人定位精度。")
        self.report_path.setText(
            f"系统参数：{result['parameters_path']}\n"
            f"手眼参数：{result['hand_eye_path']}\n报告：{result['report_path']}"
        )

    def _failed(self, message):
        self.task = None
        self._set_running(False)
        self.summary.setText(f"标定失败：{message}")
        self.output.appendPlainText(f"标定未完成：{message}")
        self.report_path.setText("输出文件：未生成完整结果")

    def reject(self):
        if self.task is not None:
            self.output.appendPlainText("标定正在运行，请等待完成后关闭。")
            return
        super().reject()

    def closeEvent(self, event):
        if self.task is not None:
            event.ignore()
        else:
            super().closeEvent(event)
