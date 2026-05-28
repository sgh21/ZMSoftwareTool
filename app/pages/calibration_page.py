"""Calibration page for robot kinematic calibration and accuracy reporting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.calibration_persistence import (
    record_identification_history,
    save_identification_result,
)
from core.calibration_service import CalibrationResult, CalibrationService, IdentificationOptions

DEFAULT_JOINT_ANGLES = (0.0, -58.0, 82.0, -112.0, -90.0, 0.0)


class IdentificationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service: CalibrationService,
        data: dict[str, np.ndarray],
        paths: list[Path],
    ) -> None:
        super().__init__()
        self._service = service
        self._data = data
        self._paths = list(paths)

    @Slot()
    def run(self) -> None:
        try:
            result = self._service.run_identification(
                self._data["joints"],
                self._data["measured_positions"],
                payloads=self._data.get("payloads"),
                directions=self._data.get("directions"),
                dataset_paths=self._paths,
                options=IdentificationOptions(),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class CalibrationPage(QWidget):
    """Robot kinematic calibration and accuracy analysis page."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        calibration_service: CalibrationService | None = None,
        open_url: Callable[[QUrl], bool] | None = None,
    ) -> None:
        super().__init__()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._open_url = open_url or QDesktopServices.openUrl
        self._service = calibration_service or CalibrationService(project_root=self.project_root)
        self._calib_result: CalibrationResult | None = None
        self._calib_data: dict[str, np.ndarray] | None = None
        self._calib_paths: list[Path] = []
        self._nominal_position: np.ndarray | None = None
        self._joint_angles: list[float] = list(DEFAULT_JOINT_ANGLES)
        self._identification_thread: QThread | None = None
        self._identification_worker: IdentificationWorker | None = None
        self._identification_progress: QProgressDialog | None = None
        self.setObjectName("calibration_page")
        self._build_ui()
        self._apply_style()
        self._update_nominal_position_display()

    def set_joint_angles(self, angles: list[float]) -> None:
        self._joint_angles = list(angles)[:6]
        self._update_nominal_position_display()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QWidget()
        body.setObjectName("calib_body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 14, 16, 8)
        body_layout.setSpacing(12)
        body_layout.addWidget(self._build_data_card())
        body_layout.addWidget(self._build_result_card())
        body_layout.addStretch(1)

        root.addWidget(body, stretch=1)
        root.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 0, 18, 0)
        layout.setSpacing(18)

        title = QLabel("机器人参数辨识与精度分析")
        title.setObjectName("app_title")
        layout.addWidget(title)

        back_btn = QPushButton("← 返回初始化")
        back_btn.setObjectName("back_button")
        back_btn.clicked.connect(self._on_back)
        layout.addWidget(back_btn)

        layout.addStretch(1)

        self._page_status = QLabel("● 就绪")
        self._page_status.setObjectName("page_status_label")
        layout.addWidget(self._page_status)
        return header

    def _build_nominal_card(self) -> QWidget:
        card = self._card_frame("nominal_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.addWidget(self._section_title("📐 名义末端位置（基于名义 MD-H 参数）"))
        title_row.addStretch(1)
        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("refresh_nominal_button")
        refresh_btn.clicked.connect(self._update_nominal_position_display)
        title_row.addWidget(refresh_btn)
        layout.addLayout(title_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        for col, (label, key) in enumerate(
            [("X (m)", "x"), ("Y (m)", "y"), ("Z (m)", "z")]
        ):
            grid.addWidget(QLabel(label), 0, col)
            value_label = QLabel("--")
            value_label.setObjectName(f"nominal_{key}_label")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(value_label, 1, col)
        layout.addLayout(grid)

        joint_info = QHBoxLayout()
        joint_info.addWidget(QLabel("当前关节角:"))
        self._joint_display = QLabel("--")
        self._joint_display.setObjectName("joint_angle_display")
        joint_info.addWidget(self._joint_display)
        joint_info.addStretch(1)
        layout.addLayout(joint_info)

        note = QLabel("此位置为集控系统基于名义运动学参数计算的理论末端位置。")
        note.setObjectName("nominal_note")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    def _build_data_card(self) -> QWidget:
        card = self._card_frame("data_card")
        card.setObjectName("calibration_toolbar")
        card.setMaximumHeight(150)
        layout = QGridLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        layout.setColumnStretch(2, 1)

        layout.addWidget(self._section_title("\u53c2\u6570\u8fa8\u8bc6\u5de5\u5177\u680f"), 0, 0, 1, 3)

        self._load_data_btn = QPushButton("\u52a0\u8f7d\u8fa8\u8bc6\u6570\u636e (.pkl\uff0c\u53ef\u591a\u9009)")
        self._load_data_btn.setObjectName("load_calib_data_button")
        self._load_data_btn.clicked.connect(self._choose_calib_data)
        layout.addWidget(self._load_data_btn, 1, 0)

        self._run_calib_btn = QPushButton("\u6267\u884c S1 \u53c2\u6570\u8fa8\u8bc6")
        self._run_calib_btn.setObjectName("run_calibration_button")
        self._run_calib_btn.setEnabled(False)
        self._run_calib_btn.clicked.connect(self._run_calibration)
        layout.addWidget(self._run_calib_btn, 1, 1)

        self._data_info = QLabel("\u5c1a\u672a\u52a0\u8f7d\u8fa8\u8bc6\u6570\u636e")
        self._data_info.setObjectName("data_info_label")
        self._data_info.setWordWrap(True)
        layout.addWidget(self._data_info, 1, 2)

        self._joint_display = QLabel("\u5f53\u524d\u5173\u8282\u89d2\uff1a--")
        self._joint_display.setObjectName("joint_angle_display")
        layout.addWidget(self._joint_display, 2, 0, 1, 3)
        return card

    def _build_result_card(self) -> QWidget:
        card = self._card_frame("result_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        layout.addWidget(self._section_title("📋 辨识结果与精度报告"))

        # Metrics grid
        metrics = QFrame()
        metrics.setObjectName("metrics_frame")
        mg = QGridLayout(metrics)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setHorizontalSpacing(14)
        mg.setVerticalSpacing(8)

        labels = [
            ("定位误差 RMS:", "rmse_label", "-- mm"),
            ("最大定位误差:", "max_error_label", "-- mm"),
            ("辨识样本数:", "sample_count_label", "--"),
            ("置信度:", "confidence_label", "100%"),
            ("优化迭代:", "nfev_label", "--"),
            ("辨识状态:", "calib_status_label", "未执行"),
        ]
        for row, (title, obj_name, default) in enumerate(labels):
            mg.addWidget(QLabel(title), row, 0)
            value = QLabel(default)
            value.setObjectName(obj_name)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            mg.addWidget(value, row, 1)
        layout.addWidget(metrics)

        # Error parameter summary
        self._param_summary = QLabel("辨识参数将在执行后显示")
        self._param_summary.setObjectName("param_summary_label")
        self._param_summary.setWordWrap(True)
        layout.addWidget(self._param_summary)

        # Action buttons
        action_row = QHBoxLayout()
        self._save_btn = QPushButton("💾 保存辨识结果 (YAML)")
        self._save_btn.setObjectName("save_calib_button")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_calibration)
        action_row.addWidget(self._save_btn)

        self._report_btn = QPushButton("📄 生成精度报告 (HTML)")
        self._report_btn.setObjectName("generate_report_button")
        self._report_btn.setEnabled(False)
        self._report_btn.clicked.connect(self._generate_report)
        action_row.addWidget(self._report_btn)
        layout.addLayout(action_row)

        return card

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(22)
        layout.addWidget(QLabel("机器人：UR10 | 坐标系：基坐标系"))
        layout.addStretch(1)
        self._footer_status = QLabel("就绪")
        self._footer_status.setObjectName("footer_status_label")
        layout.addWidget(self._footer_status)
        return footer

    # ── Actions ──────────────────────────────────────────────────────

    def _update_nominal_position_display(self) -> None:
        try:
            pos = self._service.compute_nominal_position(self._joint_angles)
            self._nominal_position = pos
            for axis, key in enumerate(("x", "y", "z")):
                label = self.findChild(QLabel, f"nominal_{key}_label")
                if label:
                    label.setText(f"{pos[axis]:.6f}")
            angles_str = ", ".join(f"{a:.1f}°" for a in self._joint_angles)
            self._joint_display.setText(f"当前关节角：{angles_str}")
        except Exception as exc:
            self._set_status(f"名义位置计算失败: {exc}")

    def _choose_calib_data(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "加载辨识数据",
            str(self.project_root / "data"),
            "Calibration data (*.pkl *.pickle);;All files (*)",
        )
        if paths:
            self._load_calib_data([Path(path) for path in paths])

    def _load_calib_data(self, paths: Path | list[Path]) -> None:
        try:
            path_list = [paths] if isinstance(paths, Path) else list(paths)
            data = self._service.load_identification_data(path_list)
            joints = np.asarray(data["joints"])
            positions = np.asarray(data["measured_positions"])
            self._calib_data = data
            self._calib_paths = path_list
            names = f"{len(path_list)} 个文件: " + ", ".join(path.name for path in path_list[:3])
            if len(path_list) > 3:
                names += f" ... 共 {len(path_list)} 个文件"
            self._data_info.setText(
                f"已加载: {names}\n"
                f"样本数: {len(joints)} | 关节1范围: [{joints[:, 0].min():.3f}, {joints[:, 0].max():.3f}] rad | "
                f"位置范围: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]m"
            )
            self._run_calib_btn.setEnabled(True)
            self._set_status(f"辨识数据已加载: {len(path_list)} 个文件，{len(joints)} 个样本")
        except Exception as exc:
            self._data_info.setText(f"加载失败: {exc}")
            self._set_status(f"数据加载失败: {exc}")

    def _run_calibration(self) -> None:
        if self._calib_data is None:
            self._set_status("请先加载辨识数据")
            return

        self._set_status("正在执行 S1 参数辨识...")
        self._run_calib_btn.setEnabled(False)
        self._identification_progress = QProgressDialog("S1 参数辨识正在运行，请稍候...", "", 0, 0, self)
        self._identification_progress.setObjectName("identification_progress_dialog")
        self._identification_progress.setWindowTitle("参数辨识进度")
        self._identification_progress.setCancelButton(None)
        self._identification_progress.setMinimumDuration(0)
        self._identification_progress.setModal(True)
        self._identification_progress.show()

        thread = QThread(self)
        worker = IdentificationWorker(self._service, self._calib_data, self._calib_paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_identification_finished)
        worker.failed.connect(self._on_identification_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_identification_thread_finished)
        self._identification_thread = thread
        self._identification_worker = worker
        thread.start()

    @Slot(object)
    def _on_identification_finished(self, result: CalibrationResult) -> None:
        if self._identification_progress is not None:
            self._identification_progress.close()
            self._identification_progress = None
        self._calib_result = result
        self._display_result(result)
        if result.success:
            saved = self._persist_identification_result(result)
            if saved is not None:
                self._ask_to_apply_identification_result(result, saved)
        self._run_calib_btn.setEnabled(True)

    @Slot(str)
    def _on_identification_failed(self, message: str) -> None:
        if self._identification_progress is not None:
            self._identification_progress.close()
            self._identification_progress = None
        self._set_status(f"辨识失败: {message}")
        self._run_calib_btn.setEnabled(True)

    @Slot()
    def _on_identification_thread_finished(self) -> None:
        self._identification_thread = None
        self._identification_worker = None

    def _display_result(self, result: CalibrationResult) -> None:
        self.findChild(QLabel, "rmse_label").setText(f"{result.position_error_rmse_mm:.4f} mm")
        self.findChild(QLabel, "max_error_label").setText(f"{result.position_error_max_mm:.4f} mm")
        self.findChild(QLabel, "sample_count_label").setText(str(result.joint_count))
        self.findChild(QLabel, "confidence_label").setText(f"{result.confidence:.0f}%")
        self.findChild(QLabel, "nfev_label").setText(str(result.nfev))

        if result.success:
            self.findChild(QLabel, "calib_status_label").setText("✔ S1 辨识成功")
            self._page_status.setText("● 辨识完成")
        else:
            self.findChild(QLabel, "calib_status_label").setText(f"⚠ {result.message}")

        # Summarize top error parameters
        if result.parameter_values:
            significant = [
                (name, val)
                for name, val in result.parameter_values.items()
                if abs(val) > 1e-10
            ]
            significant.sort(key=lambda x: abs(x[1]), reverse=True)
            top_n = 8
            if significant:
                lines = [
                    "主要辨识参数:",
                    f"  S1 λ: {result.selected_lambda:.3g}",
                    f"  拟合残差 RMSE: {result.rmse_mm:.4f} mm",
                    "  定位误差定义: 预测模型位置 - 名义模型位置",
                ]
                for name, val in significant[:top_n]:
                    param = next(
                        (p for p in result.error_parameters if p.name == name), None
                    )
                    unit = param.unit if param else ""
                    lines.append(f"  {name}: {val:.6f} {unit}")
                if len(significant) > top_n:
                    lines.append(f"  ... 共 {len(significant)} 个非零参数")
                self._param_summary.setText("\n".join(lines))
            else:
                self._param_summary.setText("所有误差参数接近零，模型与测量数据高度一致。")

        self._save_btn.setEnabled(True)
        self._report_btn.setEnabled(True)
        self._set_status(
            f"S1 辨识完成: 定位误差RMSE={result.position_error_rmse_mm:.4f}mm, 拟合RMSE={result.rmse_mm:.4f}mm"
        )

    def _persist_identification_result(self, result: CalibrationResult) -> Path | None:
        try:
            yaml_path = self.project_root / "config" / "calibration_result.yaml"
            saved = save_identification_result(
                yaml_path,
                result.parameter_values,
                fit_rmse_mm=result.rmse_mm,
                fit_max_error_mm=result.max_error_mm,
                position_error_rmse_mm=result.position_error_rmse_mm,
                position_error_max_mm=result.position_error_max_mm,
                sample_count=result.joint_count,
                confidence=result.confidence,
                method=result.method,
                selected_lambda=result.selected_lambda,
                dataset_paths=result.dataset_paths,
                cv_scores=result.cv_scores,
                subspace_summary=result.subspace_summary,
                extra_metadata=result.metadata,
            )
            record_identification_history(
                self.project_root / "storage" / "records" / "identification_history.sqlite",
                result_yaml_path=saved,
                method=result.method,
                success=result.success,
                message=result.message,
                sample_count=result.joint_count,
                fit_rmse_mm=result.rmse_mm,
                fit_max_error_mm=result.max_error_mm,
                position_error_rmse_mm=result.position_error_rmse_mm,
                position_error_max_mm=result.position_error_max_mm,
                selected_lambda=result.selected_lambda,
                confidence=result.confidence,
                dataset_paths=result.dataset_paths,
            )
            self._set_status(f"S1 辨识结果已保存: {saved.name}，历史已入库")
            return saved
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"辨识完成，但持久化失败: {exc}")
            return None

    def _ask_to_apply_identification_result(
        self,
        result: CalibrationResult,
        saved_path: Path,
    ) -> None:
        if not self._should_suggest_model_update(result):
            self._set_status("辨识结果已保存；当前指标未触发模型更新建议")
            return

        reply = QMessageBox.question(
            self,
            "应用辨识参数",
            (
                "辨识结果已持久化，当前指标建议更新精度模型。\n"
                f"是否将新参数文件加载到主界面参数文件栏？\n\n{saved_path}"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status("辨识结果已保存，未加载为当前参数模型")
            return

        from app.main_window import MainWindow

        widget = self.parent()
        while widget is not None:
            if isinstance(widget, MainWindow):
                widget.load_identification_parameter_file(saved_path)
                self._set_status(f"已加载新参数文件: {saved_path.name}")
                return
            widget = widget.parent()
        self._set_status("辨识结果已保存，但未找到主窗口加载参数文件")

    def _should_suggest_model_update(self, result: CalibrationResult) -> bool:
        threshold_path = self.project_root / "config" / "thresholds.yaml"
        rms_limit = 0.5
        max_limit = 1.0
        try:
            import yaml

            data = yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
            accuracy = data.get("accuracy", {}) if isinstance(data, dict) else {}
            rms_limit = float(accuracy.get("position_rms_limit_mm", rms_limit))
            max_limit = float(accuracy.get("max_error_limit_mm", max_limit))
        except Exception:
            pass
        improved_fit = result.rmse_mm < result.nominal_to_measured_rmse_mm
        exceeds_limit = (
            result.position_error_rmse_mm > rms_limit
            or result.position_error_max_mm > max_limit
        )
        return bool(improved_fit or exceeds_limit)

    def _save_calibration(self) -> None:
        if self._calib_result is None:
            return
        default_dir = self.project_root / "data" / "calibration"
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存辨识结果",
            str(default_dir / "calibration_result.yaml"),
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            result = self._calib_result
            saved = save_identification_result(
                path,
                result.parameter_values,
                fit_rmse_mm=result.rmse_mm,
                fit_max_error_mm=result.max_error_mm,
                position_error_rmse_mm=result.position_error_rmse_mm,
                position_error_max_mm=result.position_error_max_mm,
                sample_count=result.joint_count,
                confidence=result.confidence,
                method=result.method,
                selected_lambda=result.selected_lambda,
                dataset_paths=result.dataset_paths,
                cv_scores=result.cv_scores,
                subspace_summary=result.subspace_summary,
                extra_metadata=result.metadata,
            )
            self._set_status(f"辨识结果已保存: {saved.name}")
        except Exception as exc:
            self._set_status(f"保存失败: {exc}")

    def _generate_report(self) -> None:
        if self._calib_result is None:
            return
        default_dir = self.project_root / "data" / "reports"
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存精度报告",
            str(default_dir / "calibration_report.html"),
            "HTML files (*.html)",
        )
        if not path:
            return
        try:
            html = self._build_report_html()
            Path(path).write_text(html, encoding="utf-8")
            self._set_status(f"精度报告已生成: {Path(path).name}")
            self._open_url(QUrl.fromLocalFile(str(Path(path).resolve())))
        except Exception as exc:
            self._set_status(f"报告生成失败: {exc}")

    def _build_report_html(self) -> str:
        result = self._calib_result
        if result is None:
            return "<html><body>No calibration result</body></html>"

        nominal = result.nominal_positions
        measured = result.measured_positions
        calibrated = result.calibrated_positions

        rows_html = ""
        for i in range(min(len(measured), 50)):
            pos_err = np.linalg.norm(calibrated[i] - nominal[i]) * 1000.0
            fit_err = np.linalg.norm(calibrated[i] - measured[i]) * 1000.0
            rows_html += (
                f"<tr><td>{i + 1}</td>"
                f"<td>{nominal[i, 0]:.4f}, {nominal[i, 1]:.4f}, {nominal[i, 2]:.4f}</td>"
                f"<td>{calibrated[i, 0]:.4f}, {calibrated[i, 1]:.4f}, {calibrated[i, 2]:.4f}</td>"
                f"<td>{measured[i, 0]:.4f}, {measured[i, 1]:.4f}, {measured[i, 2]:.4f}</td>"
                f"<td>{pos_err:.4f}</td><td>{fit_err:.4f}</td></tr>"
            )

        param_rows = ""
        significant = sorted(
            [(n, v) for n, v in result.parameter_values.items() if abs(v) > 1e-10],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for name, val in significant[:20]:
            param_rows += f"<tr><td>{name}</td><td>{val:.8f}</td></tr>"

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>机器人参数辨识精度报告</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
h1, h2, h3 {{ color: #102a43; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; }}
th {{ background: #f0f4f8; }}
.ok {{ color: #0b6b3a; font-weight: 700; }}
.warn {{ color: #9a6700; font-weight: 700; }}
.card {{ background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px 18px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>机器人参数辨识精度报告</h1>

<div class="card">
<h2>概要</h2>
<table>
<tr><th>项目</th><th>结果</th></tr>
<tr><td>辨识状态</td><td class="{'ok' if result.success else 'warn'}">{'成功' if result.success else '未收敛'}</td></tr>
<tr><td>辨识方法</td><td>{result.method}</td></tr>
<tr><td>辨识样本数</td><td>{result.joint_count}</td></tr>
<tr><td>S1 选择 λ</td><td>{result.selected_lambda:.6g}</td></tr>
<tr><td>定位误差 RMSE（预测-名义）</td><td>{result.position_error_rmse_mm:.4f} mm</td></tr>
<tr><td>最大定位误差（预测-名义）</td><td>{result.position_error_max_mm:.4f} mm</td></tr>
<tr><td>拟合残差 RMSE（预测-测量）</td><td>{result.rmse_mm:.4f} mm</td></tr>
<tr><td>最大拟合残差（预测-测量）</td><td>{result.max_error_mm:.4f} mm</td></tr>
<tr><td>优化迭代次数</td><td>{result.nfev}</td></tr>
<tr><td>初始置信度</td><td>{result.confidence:.0f}%</td></tr>
</table>
</div>

<div class="card">
<h2>辨识误差参数（前20项）</h2>
<table>
<tr><th>参数名称</th><th>辨识值</th></tr>
{param_rows if param_rows else '<tr><td colspan="2">所有参数接近零</td></tr>'}
</table>
</div>

<div class="card">
<h2>逐点误差对比（前50点）</h2>
<table>
<tr><th>#</th><th>名义位置 (m)</th><th>预测位置 (m)</th><th>测量位置 (m)</th><th>定位误差 (mm)</th><th>拟合残差 (mm)</th></tr>
{rows_html}
</table>
</div>

<div class="card">
<h2>方法说明</h2>
<ul>
<li>运动学模型: 改进 D-H (Modified Denavit-Hartenberg) 六关节串联机器人</li>
<li>辨识参数: 33 几何参数 (24 MD-H + 6 基座 + 3 工具平移)</li>
<li>优化方法: S1 子空间可辨识性加权 + 交叉验证正则化 + Levenberg-Marquardt</li>
<li>拟合目标: min ||p(q; theta) - y_measured||^2 + 正则项</li>
<li>定位误差定义: p_identified(q) - p_nominal(q)</li>
</ul>
</div>

</body>
</html>"""

    def _on_back(self) -> None:
        from app.main_window import MainWindow

        widget = self.parent()
        while widget is not None:
            if isinstance(widget, MainWindow):
                widget.show_initialization_page()
                return
            widget = widget.parent()

    # ── Helpers ──────────────────────────────────────────────────────

    def _card_frame(self, name: str) -> QFrame:
        card = QFrame()
        card.setObjectName(name)
        card.setFrameShape(QFrame.Shape.NoFrame)
        return card

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section_title")
        return label

    def _set_status(self, message: str) -> None:
        self._footer_status.setText(message)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget#calibration_page {
                background: #eef4fb;
                color: #172033;
                font-family: "Microsoft YaHei", "Segoe UI", Arial;
                font-size: 13px;
            }
            QWidget#header {
                min-height: 56px;
                max-height: 56px;
                background: #f7faff;
                border-bottom: 1px solid #d9e4f2;
            }
            QWidget#calib_body {
                background: #eef4fb;
            }
            QWidget#footer {
                min-height: 40px;
                max-height: 40px;
                background: #f7faff;
                border-top: 1px solid #d9e4f2;
                color: #516075;
            }
            QLabel#app_title {
                font-size: 18px;
                font-weight: 700;
                color: #172033;
            }
            QLabel#section_title {
                font-size: 16px;
                font-weight: 700;
                color: #162033;
            }
            QLabel#page_status_label {
                color: #22b573;
                font-weight: 600;
            }
            QFrame#nominal_card, QFrame#data_card, QFrame#result_card {
                background: #f8fbff;
                border: 1px solid #d5e0ef;
                border-radius: 8px;
            }
            QFrame#metrics_frame {
                background: #f0f4f8;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                padding: 8px;
            }
            QLabel#nominal_x_label, QLabel#nominal_y_label, QLabel#nominal_z_label {
                font-size: 22px;
                font-weight: 700;
                color: #0f62d9;
                background: #eef5ff;
                border: 1px solid #bdd3f5;
                border-radius: 6px;
                padding: 8px;
            }
            QLabel#nominal_note, QLabel#data_info_label, QLabel#param_summary_label {
                color: #49566c;
                background: #f8fafc;
                border: 1px solid #e0e8f4;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton {
                min-height: 32px;
                border: 1px solid #cdd9eb;
                border-radius: 6px;
                background: #ffffff;
                padding: 5px 14px;
                color: #1d3557;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #4f8df7;
                background: #f2f7ff;
            }
            QPushButton#load_calib_data_button, QPushButton#run_calibration_button {
                min-height: 40px;
                color: #0f62d9;
                border: 1px solid #3982ff;
                background: #ffffff;
            }
            QPushButton#save_calib_button, QPushButton#generate_report_button {
                min-height: 36px;
            }
            QPushButton#back_button {
                color: #44546a;
            }
            QLabel#rmse_label, QLabel#max_error_label {
                font-weight: 700;
            }
        """)
