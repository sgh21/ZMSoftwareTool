from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.model_degradation_monitoring import (
    DegradationEvaluationResult,
    ModelDegradationMonitoringService,
    MonitoringProgress,
)


class ModelDegradationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(
        self,
        project_root: Path,
        reference_path: str,
        current_path: str,
        model_path: Path,
    ) -> None:
        super().__init__()
        self._project_root = project_root
        self._reference_path = reference_path
        self._current_path = current_path
        self._model_path = model_path

    @Slot()
    def run(self) -> None:
        try:
            service = ModelDegradationMonitoringService(self._project_root)
            result = service.evaluate(
                self._reference_path,
                self._current_path,
                calibration_result_path=self._model_path,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class ModelDegradationDialog(QDialog):
    """Evaluate model drift from fixed-board PnP observations."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        model_path_getter: Callable[[], Path],
        on_model_updated: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("model_degradation_dialog")
        self.setWindowTitle("模型评估")
        self.resize(920, 640)
        self.setMinimumSize(780, 520)

        self.project_root = Path(project_root).resolve()
        self._model_path_getter = model_path_getter
        self._on_model_updated = on_model_updated
        self._service = ModelDegradationMonitoringService(self.project_root)
        self._last_result: DegradationEvaluationResult | None = None
        self._evaluation_thread: QThread | None = None
        self._evaluation_worker: ModelDegradationWorker | None = None
        self._finished_evaluation_threads: list[QThread] = []
        self._finished_evaluation_workers: list[ModelDegradationWorker] = []
        self._evaluation_progress: QProgressDialog | None = None

        self._build_ui()
        self._apply_style()
        self.refresh_model_path()

    def refresh_model_path(self) -> None:
        self.model_path_edit.setText(str(self._active_model_path()))

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        title = QLabel("模型长期退化评估")
        title.setObjectName("model_degradation_title")
        layout.addWidget(title)

        guide = QLabel("加载初始参考 PnP 观测和当前观测，评估末端相对漂移。")
        guide.setObjectName("model_degradation_guide")
        guide.setWordWrap(True)
        layout.addWidget(guide)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel("当前模型"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setObjectName("model_degradation_model_path_edit")
        self.model_path_edit.setReadOnly(True)
        form.addWidget(self.model_path_edit, 0, 1, 1, 3)

        form.addWidget(QLabel("参考观测"), 1, 0)
        self.reference_path_edit = QLineEdit()
        self.reference_path_edit.setObjectName("model_degradation_reference_path_edit")
        form.addWidget(self.reference_path_edit, 1, 1)
        ref_file_button = QPushButton("结果文件")
        ref_file_button.setObjectName("model_degradation_reference_file_button")
        ref_file_button.clicked.connect(lambda: self._choose_observation_file(self.reference_path_edit))
        form.addWidget(ref_file_button, 1, 2)
        ref_dir_button = QPushButton("图片目录")
        ref_dir_button.setObjectName("model_degradation_reference_dir_button")
        ref_dir_button.clicked.connect(lambda: self._choose_observation_dir(self.reference_path_edit))
        form.addWidget(ref_dir_button, 1, 3)

        form.addWidget(QLabel("当前观测"), 2, 0)
        self.current_path_edit = QLineEdit()
        self.current_path_edit.setObjectName("model_degradation_current_path_edit")
        form.addWidget(self.current_path_edit, 2, 1)
        cur_file_button = QPushButton("结果文件")
        cur_file_button.setObjectName("model_degradation_current_file_button")
        cur_file_button.clicked.connect(lambda: self._choose_observation_file(self.current_path_edit))
        form.addWidget(cur_file_button, 2, 2)
        cur_dir_button = QPushButton("图片目录")
        cur_dir_button.setObjectName("model_degradation_current_dir_button")
        cur_dir_button.clicked.connect(lambda: self._choose_observation_dir(self.current_path_edit))
        form.addWidget(cur_dir_button, 2, 3)
        layout.addLayout(form)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setObjectName("model_degradation_log_edit")
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("评估日志")
        layout.addWidget(self.log_edit, stretch=1)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.evaluate_button = QPushButton("执行评估")
        self.evaluate_button.setObjectName("model_degradation_evaluate_button")
        self.evaluate_button.clicked.connect(self.evaluate)
        action_row.addWidget(self.evaluate_button)

        self.apply_button = QPushButton("更新当前模型")
        self.apply_button.setObjectName("model_degradation_apply_button")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_update)
        action_row.addWidget(self.apply_button)

        close_button = QPushButton("关闭")
        close_button.setObjectName("model_degradation_close_button")
        close_button.clicked.connect(self.close)
        action_row.addWidget(close_button)
        layout.addLayout(action_row)

    def _active_model_path(self) -> Path:
        path = self._model_path_getter()
        if path.exists():
            return path.resolve()
        return (self.project_root / "config" / "calibration_result.yaml").resolve()

    def _choose_observation_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 PnP 观测结果",
            str(self.project_root / "data"),
            "PnP observations (*.npz *.yaml *.yml *.json)",
        )
        if path:
            target.setText(path)

    def _choose_observation_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择 PnP 图片目录",
            str(self.project_root / "data"),
        )
        if path:
            target.setText(path)

    def evaluate(self) -> None:
        if self._evaluation_thread is not None:
            return
        self._finished_evaluation_threads.clear()
        self._finished_evaluation_workers.clear()
        self.refresh_model_path()
        reference = self.reference_path_edit.text().strip()
        current = self.current_path_edit.text().strip()
        if not reference or not current:
            QMessageBox.warning(self, "模型评估", "请先选择参考观测和当前观测。")
            return
        self._last_result = None
        self.apply_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)
        self.log_edit.setPlainText("模型评估已启动...")

        self._evaluation_progress = QProgressDialog(
            "正在准备模型评估...",
            "",
            0,
            0,
            self,
        )
        self._evaluation_progress.setObjectName("model_degradation_progress_dialog")
        self._evaluation_progress.setWindowTitle("模型评估进度")
        self._evaluation_progress.setCancelButton(None)
        self._evaluation_progress.setMinimumDuration(0)
        self._evaluation_progress.setModal(True)
        self._evaluation_progress.show()

        thread = QThread(self)
        worker = ModelDegradationWorker(
            self.project_root,
            reference,
            current,
            self._active_model_path(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_evaluation_progress)
        worker.finished.connect(self._on_evaluation_finished)
        worker.failed.connect(self._on_evaluation_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_evaluation_thread_finished)
        self._evaluation_thread = thread
        self._evaluation_worker = worker
        thread.start()

    @Slot(object)
    def _on_evaluation_progress(self, progress: MonitoringProgress) -> None:
        if self._evaluation_progress is not None:
            total = max(0, int(progress.total))
            current = max(0, min(int(progress.current), total))
            if self._evaluation_progress.maximum() != total:
                self._evaluation_progress.setMaximum(total)
            self._evaluation_progress.setValue(current)
            self._evaluation_progress.setLabelText(progress.message)
        self.log_edit.appendPlainText(progress.message)

    @Slot(object)
    def _on_evaluation_finished(self, result: DegradationEvaluationResult) -> None:
        if self._evaluation_progress is not None:
            self._evaluation_progress.close()
            self._evaluation_progress = None
        self._last_result = result
        self.apply_button.setEnabled(True)
        self.evaluate_button.setEnabled(True)
        self.log_edit.setPlainText(result.format_log())

    @Slot(str)
    def _on_evaluation_failed(self, message: str) -> None:
        if self._evaluation_progress is not None:
            self._evaluation_progress.close()
            self._evaluation_progress = None
        self._last_result = None
        self.apply_button.setEnabled(False)
        self.evaluate_button.setEnabled(True)
        self.log_edit.setPlainText(f"评估失败：{message}")
        QMessageBox.warning(self, "模型评估失败", message)

    @Slot()
    def _on_evaluation_thread_finished(self) -> None:
        if self._evaluation_thread is not None:
            self._finished_evaluation_threads.append(self._evaluation_thread)
        if self._evaluation_worker is not None:
            self._finished_evaluation_workers.append(self._evaluation_worker)
        self._evaluation_thread = None
        self._evaluation_worker = None

    def apply_update(self) -> None:
        if self._last_result is None:
            return
        model_path = self._active_model_path()
        answer = QMessageBox.question(
            self,
            "更新当前模型",
            "是否将本次退化评估结果写回当前模型置信度和定位不确定性？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            backup_path = self._service.apply_recommended_update(model_path, self._last_result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "模型更新失败", str(exc))
            return
        if self._on_model_updated is not None:
            self._on_model_updated()
        self.log_edit.appendPlainText("")
        self.log_edit.appendPlainText(f"已更新模型：{model_path}")
        if backup_path is not None:
            self.log_edit.appendPlainText(f"已备份旧模型：{backup_path}")
        else:
            self.log_edit.appendPlainText("已将置信度写入当前辨识模型的历史记录")
        self.apply_button.setEnabled(False)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: #eef4fb;
                color: #172033;
                font-family: "Microsoft YaHei", "Segoe UI", Arial;
                font-size: 13px;
            }
            QLabel#model_degradation_title {
                font-size: 18px;
                font-weight: 700;
                color: #172033;
            }
            QLabel#model_degradation_guide {
                color: #516075;
            }
            QLineEdit,
            QPlainTextEdit {
                border: 1px solid #d4dfed;
                border-radius: 5px;
                background: #ffffff;
                padding: 4px 6px;
                color: #28364d;
            }
            QPushButton {
                min-height: 30px;
                border: 1px solid #cdd9eb;
                border-radius: 6px;
                background: #ffffff;
                padding: 5px 12px;
                color: #1d3557;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #4f8df7;
                background: #f2f7ff;
            }
            QPushButton#model_degradation_evaluate_button,
            QPushButton#model_degradation_apply_button {
                color: #ffffff;
                background: #0f62d9;
                border-color: #0953c7;
            }
            QPushButton#model_degradation_apply_button:disabled {
                color: #64748b;
                background: #e2e8f0;
                border-color: #cbd5e1;
            }
            """
        )
