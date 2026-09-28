"""机器人定位调试入口；原始数据适配与正常监控页面分开。"""

from datetime import datetime
import json

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QPlainTextEdit, QVBoxLayout,
)

from app.pages.robot_position_page import ServiceTask
from experiments.robot_position_debug import make_debug_manifests


class RobotPositionDebugDialog(QDialog):
    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.service = page.service
        self.task = None
        self.calibration = None
        self.buttons = []
        self.paths = {}
        self.setWindowTitle("机器人定位调试")
        layout = QVBoxLayout(self)
        layout.addWidget(page._note(
            "原始 robot_error 数据只读。旧数据未记录重复到达和接近方向，不能计算 RP。\n"
            "基准/复测仅使用 calib_00 演示；同指令位姿及安装稳定性待确认，不代表实测精度退化。"
        ))
        form = QFormLayout()
        self.data_root = self._path_field(form, "旧数据目录", directory=True)
        self.data_root.setText(str(self.service.root / "data" / "robot_error"))
        self.parameter_path = self._path_field(form, "视觉参数")
        self.handeye_path = self._path_field(form, "手眼标定清单")
        self.baseline_path = self._path_field(form, "初始基准清单")
        self.current_path = self._path_field(form, "复测清单")
        layout.addLayout(form)

        prepare = QHBoxLayout()
        prepare.addWidget(self._button("生成旧数据输入", self._generate))
        prepare.addWidget(self._button("载入视觉参数", self._load_parameters))
        prepare.addStretch()
        layout.addLayout(prepare)

        calibration_row = QHBoxLayout()
        self.method = QComboBox()
        self.method.setProperty("robotInput", True)
        self.method.addItems(["PARK", "TSAI"])
        calibration_row.addWidget(self.method)
        calibration_row.addWidget(self._button("载入并标定手眼", self._calibrate))
        self.apply_button = self._button("应用标定手眼", self._apply_handeye)
        self.apply_button.setEnabled(False)
        calibration_row.addWidget(self.apply_button)
        layout.addLayout(calibration_row)

        monitoring = QHBoxLayout()
        monitoring.addWidget(self._button("建立演示基准", self._build_baseline))
        monitoring.addWidget(self._button("运行演示复测", self._evaluate))
        monitoring.addStretch()
        layout.addLayout(monitoring)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("操作进度、数据限制、PnP / 手眼质量和评估输出")
        layout.addWidget(self.output, 1)
        layout.addWidget(page._button("关闭", self.reject))

    def _button(self, label, callback):
        button = self.page._button(label, callback, True)
        self.buttons.append(button)
        return button

    def _path_field(self, form, label, directory=False):
        field = QLineEdit()
        field.setProperty("robotInput", True)
        field.setAccessibleName(label)
        row = QHBoxLayout()
        row.addWidget(field, 1)

        def browse():
            if directory:
                selected = QFileDialog.getExistingDirectory(self, label, field.text())
            else:
                selected, _ = QFileDialog.getOpenFileName(self, label, field.text(), "JSON/YAML (*.json *.yaml *.yml)")
            if selected:
                field.setText(selected)

        row.addWidget(self._button("选择", browse))
        form.addRow(label, row)
        return field

    def _start(self, title, operation, callback):
        self.output.appendPlainText(f"\n{title}开始。")
        for button in self.buttons:
            button.setEnabled(False)
        self.method.setEnabled(False)
        self._callback = callback
        self.task = ServiceTask(operation)
        self.task.signals.progress.connect(self._progress)
        self.task.signals.completed.connect(self._completed)
        self.task.signals.failed.connect(self._failed)
        QThreadPool.globalInstance().start(self.task)

    def _progress(self, percent, message):
        self.output.appendPlainText(f"{percent}% · {message}")

    def _finish(self):
        self.task = None
        for button in self.buttons:
            button.setEnabled(True)
        self.apply_button.setEnabled(self.calibration is not None)
        self.method.setEnabled(True)

    def _completed(self, result):
        callback = self._callback
        self._finish()
        callback(result)

    def _failed(self, message):
        self._finish()
        self.output.appendPlainText(f"操作失败：{message}")
        self.page.append_log(f"调试任务失败：{message}", "ERROR")
        self.page._invalidate_result()

    def _generate(self):
        source = self.data_root.text().strip()
        output = (self.service.root / "data" / "processed" / "robot_position_debug"
                  / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self._start("生成调试输入", lambda _progress: make_debug_manifests(source, output), self._generated)

    def _generated(self, paths):
        self.paths = paths
        for key, field in (("parameters", self.parameter_path), ("handeye_manifest", self.handeye_path),
                           ("baseline_manifest", self.baseline_path), ("current_manifest", self.current_path)):
            field.setText(paths[key])
        self.output.appendPlainText(json.dumps(paths, ensure_ascii=False, indent=2))
        self.output.appendPlainText("已生成输入清单。请先载入视觉参数，再进行手眼标定。未修改原始图像。")

    def _load_parameters(self):
        try:
            self.service.load_parameters(self.parameter_path.text().strip())
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.output.appendPlainText(str(error))
            return
        self.page._invalidate_result()
        self.output.appendPlainText("视觉参数已加载；棋盘与相机内参沿用旧代码，需按实际设备确认。")
        self.calibration = None
        self.apply_button.setEnabled(False)

    def _calibrate(self):
        path = self.handeye_path.text().strip()
        method = self.method.currentText()
        self.calibration = None
        self.apply_button.setEnabled(False)
        self.page._invalidate_result()

        def calculate(progress):
            self.service.load_observations(path, progress)
            return self.service.calibrate_hand_eye(method=method, progress=progress)

        self._start("手眼标定", calculate, self._calibrated)

    def _calibrated(self, result):
        self.calibration = result
        self.apply_button.setEnabled(True)
        self.output.appendPlainText(json.dumps(result, ensure_ascii=False, indent=2))
        self.output.appendPlainText("以上为固定靶标一致性残差，不是绝对定位误差。检查残差后可应用手眼参数。")
        self.page._refresh_observations()

    def _apply_handeye(self):
        try:
            self.service.save_parameters({
                "hand_eye": self.calibration["hand_eye"],
                "calibration_source": self.calibration["path"],
            })
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.output.appendPlainText(str(error))
            return
        self.page._invalidate_result()
        self.output.appendPlainText("手眼已保存为新版本。下一步载入初始观测并建立基准。")
        self.page.append_log("已应用调试标定手眼，旧参数保留。")

    def _build_baseline(self):
        path = self.baseline_path.text().strip()

        def calculate(progress):
            self.service.load_observations(path, progress)
            return self.service.create_baseline("旧数据调试基准 · 采样关系待确认")

        self._start("建立演示基准", calculate, self._baseline_created)

    def _baseline_created(self, baseline):
        self.page._baseline_created(baseline)
        self.output.appendPlainText(f"基准已保存：{baseline['path']}")
        for warning in baseline["batch"].get("warnings", []):
            self.output.appendPlainText(warning)

    def _evaluate(self):
        path = self.current_path.text().strip()

        def calculate(progress):
            self.service.load_observations(path, progress)
            return self.service.evaluate(progress)

        self._start("演示复测", calculate, self._evaluated)

    def _evaluated(self, result):
        self.page._evaluation_completed(result)
        self.page._refresh_observations()
        self.output.appendPlainText(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        self.output.appendPlainText("\n".join(result.get("warnings", [])))
        self.output.appendPlainText("已显示并保存调试结果；单次到达 RP 留空，缺初始绝对误差 AP 留空。")

    def reject(self):
        if self.task is not None:
            self.output.appendPlainText("任务正在运行，请等待完成后关闭。")
            return
        super().reject()

    def closeEvent(self, event):
        if self.task is not None:
            event.ignore()
        else:
            super().closeEvent(event)
